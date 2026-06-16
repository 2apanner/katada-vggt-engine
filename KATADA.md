# Katada VGGT Engine

Fork of [facebookresearch/vggt](https://github.com/facebookresearch/vggt) — Katada-owned reconstruction engine.

**Local path:** `production/engines/katada_vggt_engine`  
**GitHub:** https://github.com/2apanner/katada-vggt-engine

## Katada additions

| Path | Purpose |
|------|---------|
| `katada/checkpoint.py` | Commercial VGGT weights via `VGGT_CHECKPOINT_URL` |
| `katada/pipeline.py` | Container entry (`poses` stage) |
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

```bash
cd production/engines/katada_vggt_engine
./container/build_bundle.sh
cd ../../dev/pilot
python3 colab/scripts/upload_runtime_bundle.py --prefix pilot-sg-drone-360
```

Credentials are **never** baked into the bundle — injected at runtime via `connection.json`.

See `container/README.md` for Docker usage.
