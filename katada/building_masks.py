"""
Building-focused masks for Hammer-style facade / orbit captures.

Produces nerfstudio-compatible mask PNGs (0 = train, 255 = exclude) and
optional depth-confidence multipliers for VGGT COLMAP export.
"""

from __future__ import annotations

import copy
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

SKYSEG_REPO = "JianyuanWang/skyseg"
SKYSEG_FILENAME = "skyseg.onnx"
SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"

_MASK_WORKER_SESSION = None


def _cache_dir() -> Path:
    root = Path(os.getenv("KATADA_VGGT_CACHE_DIR", "/content/katada_vggt_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_skyseg_model() -> Path:
    model_path = _cache_dir() / SKYSEG_FILENAME
    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return model_path

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=SKYSEG_REPO,
            filename=SKYSEG_FILENAME,
            local_dir=str(_cache_dir()),
        )
        return Path(downloaded)
    except Exception:
        urllib.request.urlretrieve(SKYSEG_URL, model_path)
    return model_path


def _run_skyseg(onnx_session, input_size: tuple[int, int], image: np.ndarray) -> np.ndarray:
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=input_size)
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = np.array(onnx_session.run([output_name], {input_name: x})).squeeze()
    min_value = float(np.min(onnx_result))
    max_value = float(np.max(onnx_result))
    if max_value - min_value < 1e-8:
        return np.zeros_like(onnx_result, dtype=np.uint8)
    normalized = (onnx_result - min_value) / (max_value - min_value)
    return (normalized * 255.0).astype("uint8")


def sky_exclude_mask(image_bgr: np.ndarray, onnx_session) -> np.ndarray:
    """Return uint8 mask: 255 = sky (exclude), 0 = keep."""
    result_map = _run_skyseg(onnx_session, (320, 320), image_bgr)
    result_map = cv2.resize(result_map, (image_bgr.shape[1], image_bgr.shape[0]))
    return np.where(result_map >= 32, 255, 0).astype(np.uint8)


def color_clutter_exclude_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Heuristic exclude mask for clouds and pale sky missed by skyseg."""
    rgb = image_bgr.astype(np.int16)
    red, green, blue = rgb[..., 2], rgb[..., 1], rgb[..., 0]
    white = (red > 235) & (green > 235) & (blue > 235)
    blue_sky = (blue > 150) & (blue > red + 18) & (blue > green + 8)
    pale = (blue > 120) & (green > 110) & (red > 100) & (blue >= green)
    return np.where(white | blue_sky | pale, 255, 0).astype(np.uint8)


def facade_focus_exclude_mask(
    height: int,
    width: int,
    *,
    ground_frac: float = 0.12,
    center_keep_frac: float = 0.90,
) -> np.ndarray:
    """
    Exclude ground strip + outer margins — building usually centered in facade/orbit frames.
    """
    exclude = np.zeros((height, width), dtype=np.uint8)
    ground_rows = int(height * ground_frac)
    if ground_rows > 0:
        exclude[-ground_rows:, :] = 255

    keep_w = int(width * center_keep_frac)
    if keep_w < width:
        pad = (width - keep_w) // 2
        exclude[:, :pad] = 255
        exclude[:, pad + keep_w :] = 255

    return exclude


def combine_exclude_masks(*masks: np.ndarray) -> np.ndarray:
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask)
    return combined


def build_exclude_mask_for_image(
    image_path: Path | str,
    onnx_session,
    *,
    mask_sky: bool = True,
    building_focus: bool = True,
    color_filter: bool = True,
) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image for mask: {path}")

    height, width = image.shape[:2]
    parts: list[np.ndarray] = []

    if mask_sky:
        parts.append(sky_exclude_mask(image, onnx_session))
    if color_filter:
        parts.append(color_clutter_exclude_mask(image))
    if building_focus:
        parts.append(facade_focus_exclude_mask(height, width))

    if not parts:
        return np.zeros((height, width), dtype=np.uint8)
    return combine_exclude_masks(*parts)


def _init_mask_worker() -> None:
    global _MASK_WORKER_SESSION
    import onnxruntime

    _MASK_WORKER_SESSION = onnxruntime.InferenceSession(str(ensure_skyseg_model()))


def _mask_worker_task(
    image_path_str: str,
    mask_out_str: str,
    cache_path_str: str | None,
    mask_sky: bool,
    building_focus: bool,
    color_filter: bool,
) -> tuple[str, float]:
    path = Path(image_path_str)
    cache_path = Path(cache_path_str) if cache_path_str else None
    if cache_path and cache_path.is_file():
        exclude = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
    else:
        exclude = build_exclude_mask_for_image(
            path,
            _MASK_WORKER_SESSION,
            mask_sky=mask_sky,
            building_focus=building_focus,
            color_filter=color_filter,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), exclude)

    out_path = Path(mask_out_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), exclude)
    return path.name, float(np.mean(exclude > 127))


def _resolve_exclude_mask_path(
    image_path: Path | str,
    *,
    masks_dir: Path | None,
    mask_cache_dir: Path | None,
) -> Path | None:
    path = Path(image_path)
    if masks_dir:
        candidate = masks_dir / path.name
        if candidate.is_file():
            return candidate
    if mask_cache_dir:
        candidate = mask_cache_dir / f"{path.stem}_exclude.png"
        if candidate.is_file():
            return candidate
    return None


def apply_exclude_masks_from_disk(
    depth_conf: np.ndarray,
    image_paths: list[Path | str],
    *,
    masks_dir: Path | None = None,
    mask_cache_dir: Path | None = None,
) -> tuple[np.ndarray, int]:
    """Apply precomputed exclude masks to depth confidence. Returns (depth_conf, reused_count)."""
    if depth_conf.ndim != 3:
        raise ValueError(f"Expected depth_conf (S,H,W), got {depth_conf.shape}")

    out = depth_conf.copy()
    frames, height, width = out.shape
    reused = 0

    for frame_idx, image_path in enumerate(image_paths[:frames]):
        mask_path = _resolve_exclude_mask_path(
            image_path,
            masks_dir=masks_dir,
            mask_cache_dir=mask_cache_dir,
        )
        if mask_path is None:
            continue

        exclude = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if exclude is None:
            continue
        if exclude.shape[0] != height or exclude.shape[1] != width:
            exclude = cv2.resize(exclude, (width, height), interpolation=cv.INTER_NEAREST)
        keep = exclude < 128
        out[frame_idx] *= keep.astype(np.float32)
        reused += 1

    return out, reused


def write_training_masks(
    image_paths: list[Path | str],
    mask_dir: Path,
    *,
    mask_sky: bool = True,
    building_focus: bool = True,
    color_filter: bool = True,
    cache_dir: Path | None = None,
    workers: int = 4,
) -> dict[str, int | float]:
    """
    Write nerfstudio mask PNGs (0=train, 255=exclude) beside images in mask_dir.
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in image_paths]
    worker_count = max(1, min(int(workers), len(paths)))
    written = 0
    total_exclude_pct = 0.0

    if worker_count <= 1 or len(paths) < 8:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(ensure_skyseg_model()))
        for path in paths:
            cache_path = cache_dir / f"{path.stem}_exclude.png" if cache_dir else None
            if cache_path and cache_path.is_file():
                exclude = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            else:
                exclude = build_exclude_mask_for_image(
                    path,
                    session,
                    mask_sky=mask_sky,
                    building_focus=building_focus,
                    color_filter=color_filter,
                )
                if cache_path:
                    cv2.imwrite(str(cache_path), exclude)

            out_path = mask_dir / path.name
            cv2.imwrite(str(out_path), exclude)
            written += 1
            total_exclude_pct += float(np.mean(exclude > 127))
    else:
        tasks = []
        for path in paths:
            cache_path = cache_dir / f"{path.stem}_exclude.png" if cache_dir else None
            tasks.append((
                str(path),
                str(mask_dir / path.name),
                str(cache_path) if cache_path else None,
                mask_sky,
                building_focus,
                color_filter,
            ))
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_mask_worker) as pool:
            futures = [pool.submit(_mask_worker_task, *task) for task in tasks]
            for future in as_completed(futures):
                _name, exclude_frac = future.result()
                written += 1
                total_exclude_pct += exclude_frac

    avg_exclude = total_exclude_pct / max(written, 1)
    return {"mask_count": written, "avg_exclude_fraction": round(avg_exclude, 4)}


def apply_exclude_masks_to_depth_conf(
    depth_conf: np.ndarray,
    image_paths: list[Path | str],
    *,
    mask_sky: bool = True,
    building_focus: bool = True,
    color_filter: bool = True,
    masks_dir: Path | None = None,
    mask_cache_dir: Path | None = None,
) -> np.ndarray:
    """Zero VGGT depth confidence on excluded pixels (sky, ground, clutter)."""
    if masks_dir or mask_cache_dir:
        out, reused = apply_exclude_masks_from_disk(
            depth_conf,
            image_paths,
            masks_dir=masks_dir,
            mask_cache_dir=mask_cache_dir,
        )
        if reused >= max(1, min(len(image_paths), depth_conf.shape[0]) // 2):
            return out

    import onnxruntime

    if depth_conf.ndim != 3:
        raise ValueError(f"Expected depth_conf (S,H,W), got {depth_conf.shape}")

    session = onnxruntime.InferenceSession(str(ensure_skyseg_model()))
    out = depth_conf.copy()
    frames, height, width = out.shape

    for frame_idx, image_path in enumerate(image_paths[:frames]):
        exclude = build_exclude_mask_for_image(
            image_path,
            session,
            mask_sky=mask_sky,
            building_focus=building_focus,
            color_filter=color_filter,
        )
        if exclude.shape[0] != height or exclude.shape[1] != width:
            exclude = cv2.resize(exclude, (width, height), interpolation=cv.INTER_NEAREST)
        keep = exclude < 128
        out[frame_idx] *= keep.astype(np.float32)

    return out
