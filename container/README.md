# Sealed runtime + Docker for cloud GPU

## Docker image = lightweight shell (engine from GitHub at runtime)

The image **does not** bake in engine source or pip deps. On first start it:

1. `git clone` [katada-vggt-engine](https://github.com/2apanner/katada-vggt-engine) from GitHub
2. `pip install` requirements (cached on volume after first run)
3. Resolves S3 connection (file, env, or fetch `pilot/connection.json`)
4. Downloads `projects/{prefix}/raw/images.zip`
5. Runs VGGT + splatfacto
6. Uploads `projects/{prefix}/processed/pilot/*.splat`

**Mac upload** only sends footage + `connection.json` — no 128 MB bundle.

## Build

```bash
./container/build_image.sh
# → katada/vggt-runtime:latest  (~500 MB base CUDA image, not 2+ GB with engine)
```

## Run on cloud GPU

### Option A — env vars only (Vast / RunPod)

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export S3_BUCKET=katadas3
export KATADA_STORAGE_PREFIX=pilot-sg-drone-360
export AWS_REGION=ap-southeast-1
# optional: HF_TOKEN, S3_ENDPOINT_URL, VGGT_CHECKPOINT_URL

./container/run_cloud_gpu.sh
```

Or directly:

```bash
docker run --gpus all --rm \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e S3_BUCKET -e KATADA_STORAGE_PREFIX -e AWS_REGION \
  -v katada-work:/workspace \
  katada/vggt-runtime:latest
```

Default `CMD` is `run` — no extra args needed.

### Option B — mount connection.json

```bash
docker run --gpus all --rm \
  -v $(pwd)/connection.json:/run/connection.json:ro \
  -v katada-work:/workspace \
  katada/vggt-runtime:latest
```

Pull `connection.json` from S3 first:

```bash
cd production/dev/pilot
python3 colab/scripts/pull_connection.py --prefix pilot-sg-drone-360 -o /tmp/connection.json
```

## Env vars

| Variable | Required | Purpose |
|----------|----------|---------|
| `AWS_ACCESS_KEY_ID` | yes | S3 read/write |
| `AWS_SECRET_ACCESS_KEY` | yes | S3 read/write |
| `S3_BUCKET` | yes | e.g. `katadas3` |
| `KATADA_STORAGE_PREFIX` | yes | e.g. `pilot-sg-drone-360` |
| `AWS_REGION` | no | default `ap-southeast-1` |
| `S3_ENDPOINT_URL` | no | S3-compatible endpoint |
| `HF_TOKEN` | no | private HF checkpoint |
| `KATADA_CONNECTION_FILE` | no | override connection path |

## Colab bundle (tar.gz)

Same `run_pilot.py` inside bundle — use Colab `connection.json`:

```python
!python3 /content/katada_vggt_engine/katada/run_pilot.py \
  --connection-file /content/katada_connection.json
```
