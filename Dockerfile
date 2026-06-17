# Katada VGGT GPU runtime — CUDA 12.1 + Python 3.10
# Build:  ./container/build_image.sh
# Run:    see container/README.md

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    KATADA_ENGINE_ROOT=/opt/katada/katada_vggt_engine \
    KATADA_CONNECTION_FILE=/run/connection.json \
    KATADA_WORK_DIR=/workspace \
    PYTHONPATH=/opt/katada/katada_vggt_engine

RUN apt-get update -qq && apt-get install -y -qq \
    python3.10 python3-pip python3.10-venv git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN mkdir -p /workspace /run

COPY . /opt/katada/katada_vggt_engine

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        -r /opt/katada/katada_vggt_engine/requirements.txt \
        -r /opt/katada/katada_vggt_engine/requirements_demo.txt \
        huggingface_hub boto3 pillow \
    && python3 -m pip install --no-cache-dir nerfstudio open3d gsplat opencv-python-headless

COPY container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chmod +x /opt/katada/katada_vggt_engine/container/run_cloud_gpu.sh

VOLUME ["/workspace"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
