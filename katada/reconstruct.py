"""Full VGGT + splatfacto reconstruction (container / cloud GPU)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from katada.connection import ConnectionSettings
from katada.batch_plan import vggt_batch_plan_from_vram
from katada.pipeline import copy_scene_images, engine_root, run_demo_colmap
from katada.version import ENGINE_VERSION


def _patch_mediapy_for_numpy2() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("mediapy")
    if spec is None or not spec.origin:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "mediapy>=1.2.2"],
            check=True,
        )
        spec = importlib.util.find_spec("mediapy")
    if spec is None or not spec.origin:
        raise RuntimeError("mediapy not installed")

    init_path = Path(spec.origin).resolve()
    text = init_path.read_text(encoding="utf-8")
    needle = "class _VideoArray(npt.NDArray[Any]):"
    replacement = "class _VideoArray(np.ndarray):"
    if needle in text:
        init_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
        print(">> Patched mediapy _VideoArray for numpy 2", flush=True)


def _patch_nerfstudio_torch_load() -> None:
    import importlib.util

    spec = importlib.util.find_spec("nerfstudio.utils.eval_utils")
    if spec is None or not spec.origin:
        print(">> WARN: nerfstudio eval_utils not found", flush=True)
        return

    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    replacements = (
        ('torch.load(load_path, map_location="cpu")', 'torch.load(load_path, map_location="cpu", weights_only=False)'),
        ("torch.load(load_path, map_location='cpu')", "torch.load(load_path, map_location='cpu', weights_only=False)"),
    )
    for old, new in replacements:
        if old in text and new not in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            print(">> Patched nerfstudio eval_utils torch.load (weights_only=False)", flush=True)
            return


def _ensure_ns_export_deps() -> None:
    print(">> Preparing ns-export deps…", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mediapy>=1.2.2"],
        check=True,
    )
    _patch_mediapy_for_numpy2()
    _patch_nerfstudio_torch_load()


def _ns_export_env() -> dict[str, str]:
    return {}


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


def run_ns_process_colmap(scene_dir: Path, processed_dir: Path, *, num_downscales: int = 2) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    _run(
        [
            "ns-process-data",
            "images",
            "--data",
            str(scene_dir / "images"),
            "--output-dir",
            str(processed_dir),
            "--skip-colmap",
            "--colmap-model-path",
            str(scene_dir / "sparse"),
            "--num-downscales",
            str(num_downscales),
        ],
        label="NS-COLMAP-IMPORT",
    )
    masks_src = scene_dir / "masks"
    if masks_src.is_dir():
        masks_dst = processed_dir / "masks"
        if masks_dst.exists():
            shutil.rmtree(masks_dst)
        shutil.copytree(masks_src, masks_dst)


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
            "True",
            "--max-num-iterations",
            str(train_iters),
            "--output-dir",
            str(train_dir),
            "--logging.steps-per-log",
            "100",
            "--pipeline.model.camera-optimizer.mode",
            "off",
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
    _ensure_ns_export_deps()
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
        env=_ns_export_env(),
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
    frame_count = copy_scene_images(input_dir, scene_dir, max_frames=None)
    batch_plan = vggt_batch_plan_from_vram(float(os.environ.get("KATADA_VRAM_GB", "15")), frame_count)
    use_ba = tier in ("balanced", "full") and batch_plan.get("vggt_single_pass", False)
    print(f"\n=== STEP 3a: VGGT poses ({tier}, {frame_count} frames) ===", flush=True)

    ckpt_env = apply_checkpoint_env(settings)
    os.environ.update(ckpt_env)
    run_demo_colmap(
        scene_dir,
        use_ba=use_ba,
        batch_size=batch_plan["vggt_batch_size"],
        batch_overlap=batch_plan["vggt_batch_overlap"],
    )
    run_ns_process_colmap(scene_dir, processed_dir)

    return run_splatfacto_train_and_export(
        processed_dir,
        work_dir,
        train_iters=settings.train_iters,
        output_stem="shot1_vggt",
    )
