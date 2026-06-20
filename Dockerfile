# Katada VGGT GPU runtime — lightweight image (engine cloned from GitHub at start).
# Build:  ./container/build_image.sh
# Run:    ./container/run_cloud_gpu.sh

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    KATADA_ENGINE_ROOT=/opt/katada/katada_vggt_engine \
    KATADA_ENGINE_REPO=https://github.com/2apanner/katada-vggt-engine.git \
    KATADA_ENGINE_REF=main \
    KATADA_CONNECTION_FILE=/run/connection.json \
    KATADA_WORK_DIR=/workspace

RUN apt-get update -qq && apt-get install -y -qq \
    python3.10 python3-pip python3.10-venv git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN mkdir -p /workspace /run /opt/katada

COPY container/bootstrap.sh /bootstrap.sh
COPY container/entrypoint.sh /entrypoint.sh
COPY container/run_cloud_gpu.sh /opt/katada/run_cloud_gpu.sh

RUN chmod +x /bootstrap.sh /entrypoint.sh /opt/katada/run_cloud_gpu.sh

VOLUME ["/workspace"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
