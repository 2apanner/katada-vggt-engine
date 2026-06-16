# Sealed runtime for Colab / cloud GPU

## Two distribution modes

| Mode | Artifact | Use |
|------|----------|-----|
| **Bundle** | `katada_vggt_runtime.tar.gz` on S3 | Colab: download, extract to `/content/katada_vggt_engine` |
| **Docker** | `katada/vggt-runtime` image | Vast, RunPod, AWS Batch |

## Build bundle (Mac — no GPU run)

```bash
./container/build_bundle.sh
# → dist/katada_vggt_runtime_0.3.0-container_*.tar.gz
```

Upload from pilot:

```bash
python3 colab/scripts/upload_runtime_bundle.py
```

## Build Docker image (machine with Docker + optional GPU)

```bash
./container/build_image.sh
docker run --gpus all katada/vggt-runtime:latest version
docker run --gpus all -v /data/scene:/data katada/vggt-runtime:latest poses \
  --input-dir /data/frames --scene-dir /data/scene --use-ba
```

## Security

- No AWS keys or HF tokens baked into the image/bundle
- Credentials injected at runtime via `connection.json` (Colab) or env vars (cloud)
- `manifest.json` records `git_sha` + `version` for audit
