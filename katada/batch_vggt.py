"""
Batched VGGT inference for long sequences (Colab T4 / Hammer pipeline).

Processes every uploaded frame in OOM-safe chunks with overlap-frame rigid
alignment — same strategy as Katada App `vggt_point_cloud.aggregate_overlapping_batches`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch
import torch.nn.functional as F

from vggt.utils.geometry import unproject_depth_map_to_point_map


from katada.batch_plan import chunk_with_overlap


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama similarity: target ≈ scale * (source @ R.T) + t."""
    if len(source) < 3:
        return 1.0, np.eye(3), np.zeros(3)

    src = source.astype(np.float64)
    dst = target.astype(np.float64)
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float((src_c**2).sum() / n)
    if var_src < 1e-12:
        return 1.0, np.eye(3), (mu_dst - mu_src).astype(np.float64)

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    trans = mu_dst - scale * (mu_src @ rot.T)
    return scale, rot.astype(np.float64), trans.astype(np.float64)


def estimate_rigid(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation + translation only (no scale drift between batches)."""
    _, rot, trans = estimate_sim3(source, target)
    return rot, trans


def apply_rigid_volume(volume: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    shape = volume.shape
    flat = volume.reshape(-1, 3)
    return (flat @ rot.T + trans).reshape(shape)


def apply_rigid_to_extrinsics(extrinsic: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Transform world-to-camera extrinsics after world-frame rigid alignment."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    transform_inv = np.linalg.inv(transform)
    out = np.zeros_like(extrinsic)
    for i in range(extrinsic.shape[0]):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = extrinsic[i]
        aligned = matrix @ transform_inv
        out[i] = aligned[:3, :]
    return out


def _correspondences_from_overlap_frames(
    ref_points: np.ndarray,
    ref_conf: np.ndarray,
    src_points: np.ndarray,
    src_conf: np.ndarray,
    conf_percentile: float,
    min_points: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    ref_thresh = float(np.percentile(ref_conf.reshape(-1), conf_percentile))
    src_thresh = float(np.percentile(src_conf.reshape(-1), conf_percentile))
    thresh = max(ref_thresh, src_thresh)
    mask = (ref_conf.reshape(-1) >= thresh) & (src_conf.reshape(-1) >= thresh)
    ref_pts = ref_points.reshape(-1, 3)[mask]
    src_pts = src_points.reshape(-1, 3)[mask]
    n_corr = len(ref_pts)
    if n_corr < min_points:
        return ref_pts, src_pts, n_corr

    if n_corr > 12_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_corr, size=12_000, replace=False)
        ref_pts = ref_pts[idx]
        src_pts = src_pts[idx]

    return ref_pts, src_pts, n_corr


def _rigid_from_overlap_frames(
    ref_points: np.ndarray,
    ref_conf: np.ndarray,
    src_points: np.ndarray,
    src_conf: np.ndarray,
    conf_percentile: float,
    min_points: int = 64,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    ref_pts, src_pts, n_corr = _correspondences_from_overlap_frames(
        ref_points,
        ref_conf,
        src_points,
        src_conf,
        conf_percentile,
        min_points,
    )
    if n_corr < min_points:
        return np.eye(3), np.zeros(3), n_corr, 1.0

    scale, rot, trans = estimate_sim3(src_pts, ref_pts)
    return rot, trans, n_corr, scale


@dataclass
class VggtBatchSlice:
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_conf: np.ndarray
    images: torch.Tensor
    original_coords: np.ndarray
    image_names: list[str]


@dataclass
class MergedVggtSequence:
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_conf: np.ndarray
    images: torch.Tensor
    original_coords: np.ndarray
    image_names: list[str]
    batch_count: int
    alignments: list[dict]


def run_vggt_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    resolution: int = 518,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single forward pass — images shape (S, 3, H, W) on device."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    assert images.ndim == 4 and images.shape[1] == 3
    infer = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            batch = infer[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


def _log_loop_closure_drift(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    *,
    conf_percentile: float,
    min_points: int = 128,
) -> dict | None:
    frame_count = extrinsic.shape[0]
    if frame_count < 60:
        return None

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    ref_pts, src_pts, n_corr = _correspondences_from_overlap_frames(
        points_3d[0],
        depth_conf[0],
        points_3d[-1],
        depth_conf[-1],
        conf_percentile,
        min_points,
    )
    if n_corr < min_points:
        return {
            "loop_closure": True,
            "correspondences": n_corr,
            "scale_ratio": 1.0,
            "translation_norm": 0.0,
            "warn": "weak loop-closure correspondences",
        }

    scale, _, trans = estimate_sim3(src_pts, ref_pts)
    meta = {
        "loop_closure": True,
        "correspondences": n_corr,
        "scale_ratio": float(scale),
        "translation_norm": float(np.linalg.norm(trans)),
        "translation": trans.astype(np.float64),
    }
    if abs(scale - 1.0) > 0.05:
        meta["warn"] = f"loop scale drift {scale:.3f} (>5%)"
    return meta


def _apply_loop_closure_translation(
    extrinsic: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Distribute a loop-closure translation correction across the sequence."""
    frame_count = extrinsic.shape[0]
    if frame_count < 2:
        return extrinsic

    out = extrinsic.copy()
    for frame_idx in range(1, frame_count):
        alpha = frame_idx / (frame_count - 1)
        out[frame_idx : frame_idx + 1] = apply_rigid_to_extrinsics(
            out[frame_idx : frame_idx + 1],
            np.eye(3),
            alpha * translation,
        )
    return out


def merge_vggt_batches(
    batches: list[VggtBatchSlice],
    overlap: int,
    *,
    conf_percentile: float = 25.0,
    loop_closure: bool = False,
) -> MergedVggtSequence:
    """Align overlapping VGGT batches into one global sequence (all frames kept)."""
    if not batches:
        raise RuntimeError("No VGGT batches to merge")

    overlap = max(0, overlap) if len(batches) > 1 else 0
    alignments: list[dict] = []

    ext_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    image_parts: list[torch.Tensor] = []
    coord_parts: list[np.ndarray] = []
    name_parts: list[str] = []

    ref_points: np.ndarray | None = None
    ref_conf: np.ndarray | None = None

    for batch_idx, batch in enumerate(batches):
        extrinsic = batch.extrinsic.copy()
        intrinsic = batch.intrinsic.copy()
        depth_map = batch.depth_map.copy()
        depth_conf = batch.depth_conf.copy()
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        frame_start = 0
        scale_ratio = 1.0
        if batch_idx == 0:
            rot, trans, n_corr = np.eye(3), np.zeros(3), 0
        else:
            if ref_points is None or ref_conf is None:
                raise RuntimeError("Missing overlap reference from previous batch")
            rot, trans, n_corr, scale_ratio = _rigid_from_overlap_frames(
                ref_points,
                ref_conf,
                points_3d[0],
                depth_conf[0],
                conf_percentile,
            )
            extrinsic = apply_rigid_to_extrinsics(extrinsic, rot, trans)
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            frame_start = overlap

        entry = {
            "batch": batch_idx + 1,
            "frames": batch.image_names,
            "correspondences": n_corr,
            "frame_start": frame_start,
            "translation_norm": float(np.linalg.norm(trans)),
            "scale_ratio": float(scale_ratio),
        }
        if batch_idx > 0 and n_corr < 64:
            entry["warn"] = "weak overlap alignment"
        if batch_idx > 0 and abs(scale_ratio - 1.0) > 0.05:
            entry["warn"] = f"batch scale drift {scale_ratio:.3f} (>5%)"
        alignments.append(entry)

        ext_parts.append(extrinsic[frame_start:])
        int_parts.append(intrinsic[frame_start:])
        depth_parts.append(depth_map[frame_start:])
        conf_parts.append(depth_conf[frame_start:])
        image_parts.append(batch.images[frame_start:].cpu())
        coord_parts.append(batch.original_coords[frame_start:])
        name_parts.extend(batch.image_names[frame_start:])

        ref_points = points_3d[-1]
        ref_conf = depth_conf[-1]

    extrinsic = np.concatenate(ext_parts, axis=0)
    intrinsic = np.concatenate(int_parts, axis=0)
    depth_map = np.concatenate(depth_parts, axis=0)
    depth_conf = np.concatenate(conf_parts, axis=0)
    images = torch.cat(image_parts, dim=0)

    loop_meta = _log_loop_closure_drift(
        extrinsic,
        intrinsic,
        depth_map,
        depth_conf,
        conf_percentile=conf_percentile,
    )
    if loop_meta is not None:
        alignments.append({key: loop_meta[key] for key in loop_meta if key != "translation"})
        if loop_closure and loop_meta.get("correspondences", 0) >= 128 and "warn" not in loop_meta:
            extrinsic = _apply_loop_closure_translation(extrinsic, loop_meta["translation"])
            print(">> Loop closure: applied distributed translation correction", flush=True)

    return MergedVggtSequence(
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_conf=depth_conf,
        images=images,
        original_coords=np.concatenate(coord_parts, axis=0),
        image_names=name_parts,
        batch_count=len(batches),
        alignments=alignments,
    )


def run_batched_vggt_sequence(
    model: torch.nn.Module,
    image_path_list: list[str],
    *,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    batch_overlap: int,
    load_resolution: int = 1024,
    inference_resolution: int = 518,
    conf_percentile: float = 25.0,
    loop_closure: bool = False,
) -> MergedVggtSequence:
    """Run VGGT on every frame via overlapping GPU batches."""
    from vggt.utils.load_fn import load_and_preprocess_images_square

    chunks = chunk_with_overlap(image_path_list, batch_size, batch_overlap)
    print(
        f">> Batched VGGT: {len(image_path_list)} frames → {len(chunks)} batch(es) "
        f"(size≤{batch_size}, overlap={batch_overlap})",
        flush=True,
    )

    batch_slices: list[VggtBatchSlice] = []
    for batch_idx, paths in enumerate(chunks, start=1):
        names = ", ".join(os.path.basename(p) for p in paths[:3])
        if len(paths) > 3:
            names += f", … (+{len(paths) - 3})"
        print(f">> VGGT batch {batch_idx}/{len(chunks)}: {len(paths)} frames [{names}]", flush=True)

        images, original_coords = load_and_preprocess_images_square(paths, load_resolution)
        images = images.to(device)
        extrinsic, intrinsic, depth_map, depth_conf = run_vggt_forward(
            model, images, dtype, inference_resolution
        )
        batch_slices.append(
            VggtBatchSlice(
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                depth_map=depth_map,
                depth_conf=depth_conf,
                images=images,
                original_coords=original_coords.cpu().numpy(),
                image_names=[os.path.basename(p) for p in paths],
            )
        )
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged = merge_vggt_batches(
        batch_slices,
        batch_overlap,
        conf_percentile=conf_percentile,
        loop_closure=loop_closure,
    )
    print(
        f">> Batched VGGT merged: {len(merged.image_names)} frames from {merged.batch_count} batches",
        flush=True,
    )
    for entry in merged.alignments:
        label = "loop" if entry.get("loop_closure") else f"batch {entry.get('batch')}"
        scale_ratio = entry.get("scale_ratio")
        scale_note = f" scale={scale_ratio:.3f}" if scale_ratio is not None else ""
        corr = entry.get("correspondences", 0)
        trans = entry.get("translation_norm", 0.0)
        print(
            f">>   {label}: corr={corr} trans={trans:.4f}{scale_note}",
            flush=True,
        )
        if entry.get("warn"):
            print(f">>   {label}: WARN {entry['warn']}", flush=True)
    return merged
