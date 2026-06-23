"""VRAM-scaled VGGT batch sizing (no torch — safe to import anywhere)."""

from __future__ import annotations

MIN_BATCH_OVERLAP = 2


def chunk_with_overlap(items: list, batch_size: int, overlap: int) -> list[list]:
    """Split into overlapping batches (shared frames for rigid alignment)."""
    if batch_size <= 0 or len(items) <= batch_size:
        return [items]
    overlap = max(MIN_BATCH_OVERLAP, min(overlap, batch_size - 1))
    if overlap == 0:
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    chunks: list[list] = []
    start = 0
    while start < len(items):
        end = min(start + batch_size, len(items))
        chunks.append(items[start:end])
        if end >= len(items):
            break
        start = end - overlap
    return chunks


def count_overlapping_batches(frame_count: int, batch_size: int, overlap: int) -> int:
    if frame_count <= 0:
        return 0
    if frame_count <= batch_size:
        return 1
    overlap = max(MIN_BATCH_OVERLAP, min(overlap, batch_size - 1))
    step = max(1, batch_size - overlap)
    return 1 + max(0, (frame_count - batch_size + step - 1) // step)


def _single_pass_ceiling(vram_gb: float) -> int:
    """Optimistic max frames for one VGGT aggregator pass (OOM retry backs off at runtime)."""
    vram = max(float(vram_gb), 4.0)
    if vram >= 80:
        return 512
    if vram >= 40:
        return 320
    if vram >= 24:
        return 96
    if vram >= 16:
        return 48
    if vram >= 12:
        return 24
    return 8


def _max_frames_per_vggt_pass(vram_gb: float) -> tuple[int, int, str, str]:
    """Batched fallback: max frames per pass, overlap, tier id, quality class."""
    vram = max(float(vram_gb), 4.0)
    if vram >= 80:
        return 96, 6, "a100_80gb", "ultra"
    if vram >= 40:
        return 64, 5, "a100_40gb", "final"
    if vram >= 24:
        return 32, 4, "l4_a10_24gb", "final"
    if vram >= 16:
        return 16, MIN_BATCH_OVERLAP, "gpu_16gb", "preview"
    if vram >= 12:
        return 10, MIN_BATCH_OVERLAP, "t4_15gb", "preview"
    return 4, MIN_BATCH_OVERLAP, "gpu_small", "preview"


def vggt_batch_plan_from_vram(vram_gb: float, frame_count: int | None = None) -> dict:
    """
    Pick batch size + overlap from detected GPU VRAM and optional frame count.

    Strategy:
    - 80+ GB: single-pass up to 512 frames; ultra quality class.
    - 40+ GB: single-pass up to 320 frames.
    - Batched fallback uses larger passes + overlap >= 2 for seam alignment.
    """
    max_pass, overlap, tier, quality_class = _max_frames_per_vggt_pass(vram_gb)
    single_ceiling = _single_pass_ceiling(vram_gb)
    frames = int(frame_count) if frame_count and frame_count > 0 else 0

    if frames > 0 and frames <= single_ceiling and vram_gb >= 40:
        return {
            "vggt_batch_size": frames,
            "vggt_batch_overlap": 0,
            "vggt_batch_count": 1,
            "vggt_single_pass": True,
            "vggt_single_pass_attempt": True,
            "vggt_gpu_tier": tier,
            "vggt_quality_class": quality_class,
            "vggt_max_pass": single_ceiling,
        }

    if frames > 0 and frames <= max_pass:
        return {
            "vggt_batch_size": frames,
            "vggt_batch_overlap": 0,
            "vggt_batch_count": 1,
            "vggt_single_pass": True,
            "vggt_single_pass_attempt": False,
            "vggt_gpu_tier": tier,
            "vggt_quality_class": quality_class,
            "vggt_max_pass": max_pass,
        }

    batch_size = max(max_pass, MIN_BATCH_OVERLAP + 1)
    overlap = max(MIN_BATCH_OVERLAP, min(overlap, batch_size - 1))
    batch_count = count_overlapping_batches(frames, batch_size, overlap) if frames else 0

    if frames > batch_size and batch_count > 10 and vram_gb >= 24:
        target_passes = 5 if vram_gb >= 80 else 7 if vram_gb >= 40 else 10
        for try_size in range(batch_size + 4, min(frames, max_pass * 6) + 1, 4):
            try_overlap = max(MIN_BATCH_OVERLAP, min(overlap, try_size - 1))
            try_count = count_overlapping_batches(frames, try_size, try_overlap)
            if try_count <= target_passes:
                batch_size = try_size
                overlap = try_overlap
                batch_count = try_count
                break

    return {
        "vggt_batch_size": batch_size,
        "vggt_batch_overlap": overlap,
        "vggt_batch_count": batch_count,
        "vggt_single_pass": batch_count <= 1,
        "vggt_single_pass_attempt": False,
        "vggt_gpu_tier": tier,
        "vggt_quality_class": quality_class,
        "vggt_max_pass": max_pass,
    }
