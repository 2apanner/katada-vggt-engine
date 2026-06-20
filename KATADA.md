# Katada VGGT Engine

Fork of [facebookresearch/vggt](https://github.com/facebookresearch/vggt) — Katada-owned reconstruction engine.

**Local path:** `production/dev/engines/katada_vggt_engine`  
**GitHub:** https://github.com/2apanner/katada-vggt-engine

## Katada additions

| Path | Purpose |
|------|---------|
| `katada/checkpoint.py` | Commercial VGGT weights via `VGGT_CHECKPOINT_URL` |
| `katada/run_pilot.py` | **Full S3 pipeline** (download → reconstruct → upload) |
| `katada/s3_io.py` | S3 keys + download/upload |
| `katada/connection.py` | Parse `connection.json` |
| `katada/version.py` | Engine version for manifests |
| `container/` | Docker image + sealed `.tar.gz` bundle for Colab/cloud GPU |
| `Dockerfile` | CUDA runtime for Vast / RunPod |

## Mac workflow (edit only — do not run GPU here)

```bash
cd production/dev/pilot
./colab/scripts/setup_engine.sh
# edit files
./colab/scripts/push_engine.sh
```

## Ship to Colab / cloud

**Colab** clones the engine from GitHub at runtime (no S3 bundle upload):

```bash
cd production/dev/pilot
./colab/scripts/push_engine.sh   # after editing engine
```

**Cloud GPU (Docker)** — image has engine baked in:

```bash
cd production/dev/engines/katada_vggt_engine
./container/build_image.sh
./container/run_cloud_gpu.sh
```

Credentials are **never** baked into the bundle — injected at runtime via `connection.json`.

See `container/README.md` for Docker usage.
