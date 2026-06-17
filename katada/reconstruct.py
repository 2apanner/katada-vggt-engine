"""Full VGGT + splatfacto reconstruction (container / cloud GPU)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from katada.connection import ConnectionSettings
from katada.pipeline import copy_scene_images, engine_root, run_demo_colmap
from katada.version import ENGINE_VERSION


def _run(cmd: list[str], *, label: str, env: dict | None = None, check: bool = True) -> int:
    print(f"\n>> [{label}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env={**os.environ, **(env or {})})
    if check and proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")
    return proc.returncode or 0


def apply_checkpoint_env(settings: ConnectionSettings) -> dict[str, str]:
    env = {"VGGT_CHECKPOINT_URL": settings.vggt_checkpoint_url}
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    return env


def run_ns_process_colmap(scene_dir: Path, processed_dir: Path) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    _run(
        [
            "ns-process-data",
            "colmap",
            "--data",
            str(scene_dir),
            "--output-dir",
            str(processed_dir),
        ],
        label="NS-COLMAP-IMPORT",
    )


def run_splatfacto_train_and_export(
    processed_dir: Path,
    work_dir: Path,
    *,
    train_iters: int,
    output_stem: str = "shot1_vggt",
) -> Path:
    train_dir = work_dir / "nerfstudio_outputs"
    output_dir = work_dir / "outputs"
    export_dir = work_dir / "nerfstudio_export"
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_dir.exists():
        shutil.rmtree(train_dir)

    print(f"\n=== STEP 3b: Train splatfacto ({train_iters} iters) ===", flush=True)
    _run(
        [
            "ns-train",
            "splatfacto",
            "--data",
            str(processed_dir),
            "--vis",
            "tensorboard",
            "--viewer.quit-on-train-completion",
            "--max-num-iterations",
            str(train_iters),
            "--output-dir",
            str(train_dir),
            "--logging.steps-per-log",
            "50",
        ],
        label="TRAIN",
    )

    configs = sorted(train_dir.glob("**/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        raise RuntimeError("Nerfstudio finished but config.yml not found")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    print("\n=== STEP 3c: Export gaussian splat ===", flush=True)
    _run(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(configs[0]),
            "--output-dir",
            str(export_dir),
        ],
        label="EXPORT",
    )

    for pattern in ("*.splat", "*.ply"):
        matches = sorted(export_dir.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)
        if matches:
            dest = output_dir / f"{output_stem}{matches[0].suffix}"
            shutil.copy2(matches[0], dest)
            print(f">> Output: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
            return dest

    raise RuntimeError("Export produced no .splat or .ply file")


def run_full_reconstruction(
    input_dir: Path,
    work_dir: Path,
    settings: ConnectionSettings,
) -> Path:
    """STEP 3: VGGT poses + splatfacto train + export."""
    print(f"\n=== STEP 3: VGGT + 3DGS (engine v{ENGINE_VERSION}) ===", flush=True)
    if not (engine_root() / "demo_colmap.py").is_file():
        raise FileNotFoundError(f"Engine missing at {engine_root()}")

    scene_dir = work_dir / "facade_scene"
    processed_dir = work_dir / "nerfstudio_processed"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    tier = settings.quality_tier
    use_ba = tier in ("balanced", "full")
    frame_count = copy_scene_images(input_dir, scene_dir, max_frames=None)
    print(f"\n=== STEP 3a: VGGT poses ({tier}, {frame_count} frames) ===", flush=True)

    ckpt_env = apply_checkpoint_env(settings)
    os.environ.update(ckpt_env)
    run_demo_colmap(scene_dir, use_ba=use_ba)
    run_ns_process_colmap(scene_dir, processed_dir)

    return run_splatfacto_train_and_export(
        processed_dir,
        work_dir,
        train_iters=settings.train_iters,
        output_stem="shot1_vggt",
    )
