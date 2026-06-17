"""Load Katada pilot connection.json (S3 credentials + project config)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/buckets/apanner/VGGT-1B-Commercial-bucket"
    "/resolve/vggt_1B_commercial.pt"
)

TRAIN_ITERS_BY_TIER = {
    "fast": 5500,
    "balanced": 7500,
    "full": 10000,
}


@dataclass(frozen=True)
class ConnectionSettings:
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    bucket: str
    storage_prefix: str
    s3_endpoint_url: str | None
    model: str
    quality_tier: str
    train_iters: int
    vggt_checkpoint_url: str
    hf_token: str | None
    frame_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConnectionSettings:
        tier = (data.get("quality_tier") or "balanced").strip().lower()
        if tier not in TRAIN_ITERS_BY_TIER:
            tier = "balanced"
        train_iters = int(data.get("train_iters") or TRAIN_ITERS_BY_TIER[tier])
        return cls(
            aws_access_key_id=str(data.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID", "")),
            aws_secret_access_key=str(
                data.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
            ),
            aws_region=str(data.get("aws_region") or os.getenv("AWS_REGION", "ap-southeast-1")),
            bucket=str(data.get("bucket") or os.getenv("S3_BUCKET", "")),
            storage_prefix=str(data.get("storage_prefix") or os.getenv("KATADA_STORAGE_PREFIX", "")),
            s3_endpoint_url=data.get("s3_endpoint_url") or os.getenv("S3_ENDPOINT_URL") or None,
            model=str(data.get("model") or "nerfstudio"),
            quality_tier=tier,
            train_iters=train_iters,
            vggt_checkpoint_url=str(
                data.get("vggt_checkpoint_url")
                or os.getenv("VGGT_CHECKPOINT_URL")
                or DEFAULT_CHECKPOINT_URL
            ),
            hf_token=data.get("hf_token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            frame_count=int(data.get("frame_count") or 0),
        )

    def validate(self) -> None:
        missing: list[str] = []
        if not self.aws_access_key_id:
            missing.append("aws_access_key_id")
        if not self.aws_secret_access_key:
            missing.append("aws_secret_access_key")
        if not self.bucket:
            missing.append("bucket")
        if not self.storage_prefix:
            missing.append("storage_prefix")
        if missing:
            raise ValueError(f"connection missing required fields: {', '.join(missing)}")


def load_connection_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f">> Loaded connection: {path}", flush=True)
    return data


def resolve_connection(
    *,
    connection_file: Path | None = None,
    prefix: str | None = None,
    bucket: str | None = None,
    fetch_from_s3: bool = True,
) -> ConnectionSettings:
    """Resolve pilot connection for Docker / cloud GPU.

    Priority:
    1. --connection-file / KATADA_CONNECTION_FILE / /run/connection.json
    2. AWS_* + KATADA_STORAGE_PREFIX env vars
    3. Fetch projects/{prefix}/pilot/connection.json from S3 (if creds present)
    """
    from katada.s3_io import download_connection_from_s3, make_s3_client

    payload: dict[str, Any] = {}

    candidates: list[Path] = []
    if connection_file:
        candidates.append(connection_file)
    env_conn = os.getenv("KATADA_CONNECTION_FILE", "").strip()
    if env_conn:
        candidates.append(Path(env_conn))
    candidates.append(Path("/run/connection.json"))

    for path in candidates:
        if path.is_file():
            payload.update(load_connection_file(path))
            break

    if prefix:
        payload["storage_prefix"] = prefix
    if bucket:
        payload["bucket"] = bucket

    # Env-only bootstrap (typical cloud GPU — no mounted file)
    if not payload.get("storage_prefix") and os.getenv("KATADA_STORAGE_PREFIX"):
        payload["storage_prefix"] = os.getenv("KATADA_STORAGE_PREFIX", "")

    partial = ConnectionSettings.from_dict(payload)
    has_creds = bool(partial.aws_access_key_id and partial.aws_secret_access_key)
    has_target = bool(partial.bucket and partial.storage_prefix)

    if fetch_from_s3 and has_creds and has_target and not payload.get("quality_tier"):
        try:
            client = make_s3_client(partial)
            remote = download_connection_from_s3(client, partial)
            payload.update(remote)
            print(">> Merged connection from S3 pilot/connection.json", flush=True)
        except Exception as exc:
            print(f">> S3 connection.json not fetched ({exc}) — using env/file only", flush=True)

    if prefix:
        payload["storage_prefix"] = prefix
    if bucket:
        payload["bucket"] = bucket

    settings = ConnectionSettings.from_dict(payload)
    settings.validate()
    return settings
