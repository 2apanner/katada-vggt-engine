#!/usr/bin/env python3
"""
Post-filter Nerfstudio 3DGS PLY — keep building cluster, drop sky floaters and low-opacity splats.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def _parse_ply_header(header_lines: list[str]) -> tuple[int, list[tuple[str, str]], bool]:
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    binary = False
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            properties.append((parts[1], parts[2]))
        elif "binary_little_endian" in line:
            binary = True
    return vertex_count, properties, binary


def _dtype_for_property(prop_type: str) -> np.dtype:
    mapping = {
        "float": np.float32,
        "double": np.float64,
        "uchar": np.uint8,
        "int": np.int32,
    }
    return mapping.get(prop_type, np.float32)


def read_gaussian_ply(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count, properties, binary = _parse_ply_header(header_lines)
        if not binary:
            raise ValueError("Only binary PLY supported")

        names = [name for _, name in properties]
        dtype = np.dtype([(name, _dtype_for_property(ptype)) for ptype, name in properties])
        data = np.frombuffer(handle.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
        return names, data


def write_gaussian_ply(path: Path, names: list[str], data: np.ndarray) -> None:
    dtype = data.dtype
    with path.open("wb") as handle:
        handle.write(b"ply\n")
        handle.write(b"format binary_little_endian 1.0\n")
        handle.write(b"comment Filtered by katada.filter_splat_building\n")
        handle.write(f"element vertex {len(data)}\n".encode("ascii"))
        for name in names:
            field_dtype = dtype.fields[name][0]
            if field_dtype == np.float32:
                ptype = "float"
            elif field_dtype == np.uint8:
                ptype = "uchar"
            else:
                ptype = "float"
            handle.write(f"property {ptype} {name}\n".encode("ascii"))
        handle.write(b"end_header\n")
        handle.write(data.tobytes())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def building_keep_indices(
    data: np.ndarray,
    *,
    min_opacity: float = 0.04,
    percentile_low: float = 6.0,
    percentile_high: float = 94.0,
    margin: float = 0.06,
    radial_percentile: float = 98.0,
) -> np.ndarray:
    names = data.dtype.names or ()
    if not {"x", "y", "z", "opacity"}.issubset(names):
        raise ValueError("PLY missing x,y,z,opacity fields")

    points = np.column_stack([data["x"], data["y"], data["z"]])
    opacity = _sigmoid(data["opacity"].astype(np.float64))
    core = opacity >= min_opacity
    if not np.any(core):
        core = np.ones(len(points), dtype=bool)

    pts = points[core]
    total = len(points)
    if len(pts) < 500:
        return core

    lo = np.percentile(pts, percentile_low, axis=0)
    hi = np.percentile(pts, percentile_high, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - span * margin
    hi = hi + span * margin
    in_box = np.all((points >= lo) & (points <= hi), axis=1)

    centroid = np.median(pts, axis=0)
    dist = np.linalg.norm(points - centroid, axis=1)
    radial_cap = float(np.percentile(dist[core], radial_percentile))
    radial = dist <= radial_cap

    keep = core & in_box & radial

    if "f_dc_0" in names and "f_dc_1" in names and "f_dc_2" in names:
        rgb = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]])
        rgb = np.clip(0.5 + 0.282095 * rgb, 0.0, 1.0)
        red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        sky = (blue > 0.55) & (blue > red + 0.08) & (blue > green + 0.04)
        cloud = (red > 0.92) & (green > 0.92) & (blue > 0.92)
        keep &= ~(sky | cloud)

    kept_count = int(np.sum(keep))
    if kept_count < max(1000, int(total * 0.05)):
        return core
    return keep


def filter_building_ply(
    input_path: Path,
    output_path: Path,
    *,
    min_opacity: float = 0.04,
) -> dict[str, float | int]:
    names, data = read_gaussian_ply(input_path)
    total = len(data)
    keep = building_keep_indices(data, min_opacity=min_opacity)
    filtered = data[keep]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(output_path, names, filtered)
    return {
        "input_count": total,
        "kept_count": int(len(filtered)),
        "kept_ratio": round(len(filtered) / max(total, 1), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter 3DGS PLY to building-focused cluster")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-opacity", type=float, default=0.04)
    args = parser.parse_args()

    stats = filter_building_ply(args.input.resolve(), args.output.resolve(), min_opacity=args.min_opacity)
    print(
        f"kept {stats['kept_count']:,} / {stats['input_count']:,} "
        f"({stats['kept_ratio']*100:.1f}%) → {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
