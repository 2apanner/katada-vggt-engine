#!/usr/bin/env python3
"""
Katada container entry — full S3 pipeline.

  connection.json → download images.zip → VGGT + splatfacto → upload .splat

Usage (Docker / cloud GPU / Colab bundle):
  python3 -m katada.run_pilot --connection-file /run/connection.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from katada.connection import resolve_connection
from katada.reconstruct import run_full_reconstruction
from katada.s3_io import download_images_zip, make_s3_client, upload_pilot_outputs
from katada.version import ENGINE_VERSION


def default_work_dir() -> Path:
    for candidate in (Path("/workspace"), Path("/content"), Path("/tmp/katada_run")):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/katada_run")


def main() -> int:
    parser = argparse.ArgumentParser(description="Katada S3 pilot pipeline (container)")
    parser.add_argument(
        "--connection-file",
        default=os.getenv("KATADA_CONNECTION_FILE"),
        help="JSON with AWS keys, bucket, storage_prefix (default: KATADA_CONNECTION_FILE or /run/connection.json)",
    )
    parser.add_argument("--prefix", default=None, help="Override storage_prefix")
    parser.add_argument("--bucket", default=None, help="Override S3 bucket")
    parser.add_argument("--work-dir", default=None, help="Scratch dir (default: /workspace)")
    args = parser.parse_args()

    conn_path = Path(args.connection_file).resolve() if args.connection_file else None
    settings = resolve_connection(
        connection_file=conn_path,
        prefix=args.prefix,
        bucket=args.bucket,
    )

    work_dir = Path(args.work_dir).resolve() if args.work_dir else default_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path = work_dir / "images.zip"
    images_dir = work_dir / "raw_images"
    log_path = work_dir / "outputs" / "katada_pilot.log"

    print(f">> Katada run_pilot v{ENGINE_VERSION}", flush=True)
    print(f">> Project: {settings.storage_prefix} | bucket: {settings.bucket}", flush=True)
    print(f">> Work dir: {work_dir}", flush=True)

    client = make_s3_client(settings)
    frame_count = download_images_zip(client, settings, zip_path=zip_path, images_dir=images_dir)
    output_file = run_full_reconstruction(images_dir, work_dir, settings)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"engine_version={ENGINE_VERSION}\n"
        f"frames={frame_count}\n"
        f"output={output_file.name}\n"
        f"bytes={output_file.stat().st_size}\n",
        encoding="utf-8",
    )

    keys = upload_pilot_outputs(
        client,
        settings,
        output_file=output_file,
        log_file=log_path,
    )
    print(f"\n=== PILOT COMPLETE ===", flush=True)
    print(f"Output: s3://{settings.bucket}/{keys[0]}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
