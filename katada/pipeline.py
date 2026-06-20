#!/usr/bin/env python3
"""
Katada GPU pipeline entry — used inside the sealed runtime container / Colab bundle.

Stage 3a: demo_colmap.py (VGGT poses → COLMAP sparse)
Stage 3b+: delegated to colab_pilot_runner (nerfstudio splatfacto) outside this module.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from katada.version import ENGINE_VERSION

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def engine_root() -> Path:
    return Path(__file__).resolve().parent.parent


def copy_scene_images(input_dir: Path, scene_dir: Path, max_frames: int | None) -> int:
    images_dir = scene_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images) < 3:
        images = sorted(input_dir.rglob("frame_*.jpg"))
    if len(images) < 3:
        raise RuntimeError(f"Need >= 3 images in {input_dir}, found {len(images)}")

    if max_frames is not None and len(images) > max_frames:
        step = max(1, len(images) // max_frames)
        images = images[::step][:max_frames]

    for idx, src in enumerate(images):
        shutil.copy2(src, images_dir / f"frame_{idx:04d}{src.suffix.lower()}")
    return len(images)


def run_demo_colmap(
    scene_dir: Path,
    *,
    use_ba: bool,
    batch_size: int | None = None,
    batch_overlap: int | None = None,
) -> int:
    root = engine_root()
    demo = root / "demo_colmap.py"
    if not demo.is_file():
        raise FileNotFoundError(f"demo_colmap.py missing in {root}")

    cmd = [sys.executable, str(demo), "--scene_dir", str(scene_dir)]
    if batch_size is not None and batch_size > 0:
        cmd.extend(["--batch-size", str(batch_size)])
    if batch_overlap is not None and batch_overlap >= 0:
        cmd.extend(["--batch-overlap", str(batch_overlap)])
    if use_ba:
        cmd.append("--use_ba")

    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYTHONUNBUFFERED": "1",
    }
    print(f">> [{ENGINE_VERSION}] demo_colmap: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    if proc.returncode != 0 and use_ba:
        cmd_no_ba = [c for c in cmd if c != "--use_ba"]
        print(">> demo_colmap BA failed — retry without --use_ba", flush=True)
        proc = subprocess.run(cmd_no_ba, cwd=str(root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"demo_colmap failed (exit {proc.returncode})")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Katada VGGT pose estimation (container entry)")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--quality-tier", default="balanced")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--use-ba", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    scene_dir = Path(args.scene_dir).resolve()
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    tier = (args.quality_tier or "balanced").strip().lower()
    use_ba = args.use_ba or tier in ("balanced", "full")
    frame_count = copy_scene_images(input_dir, scene_dir, args.max_frames)
    print(f">> Katada engine {ENGINE_VERSION} | {frame_count} frames | BA={'on' if use_ba else 'off'}", flush=True)
    run_demo_colmap(scene_dir, use_ba=use_ba)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
