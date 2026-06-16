"""Commercial VGGT checkpoint loading for Katada (env-driven, no hardcoded HF paths)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch


DEFAULT_COMMERCIAL_URL = (
    "https://huggingface.co/buckets/apanner/VGGT-1B-Commercial-bucket"
    "/resolve/vggt_1B_commercial.pt"
)
FALLBACK_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def resolve_checkpoint_url() -> str:
    return (
        os.environ.get("VGGT_CHECKPOINT_URL")
        or os.environ.get("KATADA_VGGT_CHECKPOINT_URL")
        or DEFAULT_COMMERCIAL_URL
    ).strip()


def resolve_checkpoint_path() -> Path | None:
    raw = os.environ.get("VGGT_CHECKPOINT_PATH") or os.environ.get("KATADA_VGGT_CHECKPOINT_PATH")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_file() else None


def load_vggt_weights(model, *, log: Callable[[str], None] | None = None) -> None:
    """Load VGGT weights from local path or URL (commercial bucket default)."""
    out = log or (lambda msg: print(msg, flush=True))
    local = resolve_checkpoint_path()
    if local is not None:
        out(f">> VGGT checkpoint (local): {local}")
        state = torch.load(str(local), map_location="cpu")
        model.load_state_dict(state)
        return

    url = resolve_checkpoint_url()
    out(f">> VGGT checkpoint (url): {url}")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and "huggingface.co" in url:
        try:
            from huggingface_hub import hf_hub_download
            import re

            match = re.match(
                r"https://huggingface\.co/(?:buckets/)?([^/]+)/([^/]+)/resolve/(.+)",
                url,
            )
            if match:
                repo_id, _, filename = match.groups()
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=hf_token,
                )
                state = torch.load(path, map_location="cpu")
                model.load_state_dict(state)
                return
        except Exception as exc:
            out(f">> HF hub download failed ({exc}) — trying torch.hub")

    try:
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    except Exception as exc:
        out(f">> Primary checkpoint failed ({exc}) — fallback {FALLBACK_URL}")
        model.load_state_dict(torch.hub.load_state_dict_from_url(FALLBACK_URL))
