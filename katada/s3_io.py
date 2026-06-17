"""S3 download/upload for Katada pilot pipeline (inside container)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config

from katada.connection import ConnectionSettings


def project_root(storage_prefix: str) -> str:
    return f"projects/{storage_prefix}"


def connection_key(storage_prefix: str) -> str:
    return f"{project_root(storage_prefix)}/pilot/connection.json"


def raw_images_zip_key(storage_prefix: str) -> str:
    return f"{project_root(storage_prefix)}/raw/images.zip"


def pilot_output_key(storage_prefix: str, filename: str) -> str:
    return f"{project_root(storage_prefix)}/processed/pilot/{filename}"


def make_s3_client(settings: ConnectionSettings):
    client_kwargs: dict = {
        "region_name": settings.aws_region,
        "config": Config(signature_version="s3v4"),
    }
    if settings.s3_endpoint_url:
        client_kwargs["endpoint_url"] = settings.s3_endpoint_url
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        **client_kwargs,
    )


def download_connection_from_s3(client, settings: ConnectionSettings) -> dict:
    import json

    key = connection_key(settings.storage_prefix)
    obj = client.get_object(Bucket=settings.bucket, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    print(f">> Loaded connection from s3://{settings.bucket}/{key}", flush=True)
    return data


def download_images_zip(
    client,
    settings: ConnectionSettings,
    *,
    zip_path: Path,
    images_dir: Path,
) -> int:
    key = raw_images_zip_key(settings.storage_prefix)
    print(f"\n=== STEP 2: Download images from S3 ===", flush=True)
    print(f">> s3://{settings.bucket}/{key} → {zip_path}", flush=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(settings.bucket, key, str(zip_path))

    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(images_dir)

    frames = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    if len(frames) < 3:
        nested = list(images_dir.rglob("frame_*.jpg"))
        if len(nested) >= 3:
            for img in nested:
                shutil.copy2(img, images_dir / img.name)
            frames = list(images_dir.glob("frame_*.jpg"))

    if len(frames) < 3:
        raise RuntimeError(f"Need >= 3 images in zip, found {len(frames)}")
    print(f">> Extracted {len(frames)} frames → {images_dir}", flush=True)
    return len(frames)


def upload_pilot_outputs(
    client,
    settings: ConnectionSettings,
    *,
    output_file: Path,
    log_file: Path | None = None,
    extra_files: list[Path] | None = None,
) -> list[str]:
    print("\n=== STEP 4: Upload outputs to S3 ===", flush=True)
    uploaded: list[str] = []
    uploads: list[tuple[Path, str]] = [
        (output_file, pilot_output_key(settings.storage_prefix, output_file.name)),
    ]
    if log_file and log_file.is_file():
        uploads.append((log_file, pilot_output_key(settings.storage_prefix, log_file.name)))
    for path in extra_files or []:
        if path.is_file():
            uploads.append((path, pilot_output_key(settings.storage_prefix, path.name)))

    for local_path, s3_key in uploads:
        print(f">> Upload {local_path.name} → s3://{settings.bucket}/{s3_key}", flush=True)
        client.upload_file(str(local_path), settings.bucket, s3_key)
        uploaded.append(s3_key)
    print(">> Upload complete", flush=True)
    return uploaded
