# ONNM inference container, for Cloudflare Containers (instance type standard-1:
# 1/2 vCPU, 4 GiB memory, 8 GB disk).
#
# BUILD CONTEXT IS THE REPOSITORY ROOT, not this directory:
#
#     python scripts/stage_inference_model.py     # puts the serving model in inference/model/
#     docker build -f inference/Dockerfile -t onnm-inference .
#
# The staging step is not optional. It reads reports/PRODUCTION, proves the
# checkpoint's sha256 matches the version marked `serving` in
# model_versions.json, and copies best.pt plus calibration.json into
# inference/model/. Skipping it either fails the COPY below or -- worse -- bakes
# whatever was staged last, which is how a site ends up quietly serving old
# weights while every document says otherwise.

FROM python:3.12-slim

# Bump when the Worker-to-container startup contract changes. Cloudflare only
# rolls a running container when its image changes; this revision ensures the
# corrected `envVars` bridge in worker/container.js reaches a fresh process.
LABEL org.opencontainers.image.revision="inference-env-bridge-v1"

# libgomp1 is torch's OpenMP runtime and is genuinely required; python:*-slim
# does not ship it and the failure is an obscure load-time error rather than a
# missing-package message. libglib2.0-0 is opencv-python-headless's one
# remaining system dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first, and in their own layer: they are ~1 GB and change far less
# often than the application code, so a code edit must not re-download torch.
COPY inference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# The whole src/ tree, because the service imports both the `onnm` package and
# the top-level `community` module (reused for the shared-image encoder, so the
# bytes written to D1 stay byte-identical to what Streamlit wrote).
COPY src/ /app/src/
COPY inference/service.py inference/main.py /app/inference/

# Weights and calibration, staged by scripts/stage_inference_model.py. They must
# land in the SAME directory: Calibration.for_checkpoint looks for
# calibration.json beside the checkpoint, and without it the model runs at
# temperature 1.0 and a naive 0.50 threshold, silently changing every number the
# app reports.
COPY inference/model/ /opt/onnm/

ENV PYTHONPATH=/app/src:/app/inference \
    ONNM_CHECKPOINT=/opt/onnm/best.pt \
    PORT=8080

# The instance has half a vCPU. Letting OpenMP and MKL each spin up a thread per
# detected core makes them fight over a fraction of one, and a DenseNet-121
# forward pass gets slower rather than faster. Pinning to a single thread is a
# latency optimisation here, not a limitation.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 8080

# One worker on purpose. The model is a single in-process object holding ~1 GB
# resident; a second worker would duplicate it and exceed the 4 GiB instance.
# Concurrency is handled by the Durable Object queueing in front of the
# container, not by forking here.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
