# TRAIN_GPU reproducibility image — the euromonitor training lane.
#
# Build (repo root):
#   docker build -t broadway-train-gpu -f project/experiments/euromonitor/TRAIN_GPU/Dockerfile .
#
# Run (WORKDIR = the TRAIN_GPU lane root; artifacts/ is the volume point):
#   docker run --rm -v $(pwd)/project/experiments/euromonitor/TRAIN_GPU/artifacts:/app/artifacts \
#     broadway-train-gpu python TRAIN/01_data_prep.py
#   docker run --gpus all --rm \
#     -v $(pwd)/project/experiments/euromonitor/TRAIN_GPU/artifacts:/app/artifacts \
#     -v $(pwd)/project/experiments/euromonitor/TRAIN_GPU/models:/app/artifacts/models \
#     broadway-train-gpu python TRAIN/05_train.py --model artifacts/models/all-MiniLM-L6-v2
#
# CPU vs GPU: the image ships CUDA-enabled torch (2.13+cu130). For CPU-only
# smokes swap the base tag for pytorch/pytorch:2.13.0-cpython3.12-cuda13.0-runtime
# → the cu wheels also run on CPU (deberta note: deberta-v3 is ~2000x slower
# on CPU; deberta scoring/training belongs on the GPU lane).

# ── stage 1: dependency lock (uv resolves once, cache-mountable) ───────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps
WORKDIR /repo
COPY pyproject.toml uv.lock* ./
# the euromonitor lane needs the nlp extra (torch, sentence-transformers,
# datasets, accelerate) + dev tools (ruff) + mlflow + optuna (in base deps)
RUN uv export --frozen --no-dev --extra nlp --format requirements-txt \
      -o /tmp/requirements.txt 2>/dev/null \
    || uv export --no-dev --extra nlp --format requirements-txt -o /tmp/requirements.txt
# drop the project itself (file:/// broadway line) — the image installs the
# repo by COPY, not as a wheel; keep only third-party pins
RUN grep -vE "^(broadway|file://|#|-e )" /tmp/requirements.txt > /tmp/locked.txt \
    && grep -c . /tmp/locked.txt

# ── stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# tini: proper signal handling (SIGTERM kills training runs cleanly);
# git: reproducible provenance stamps; curl: healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /tmp/locked.txt /tmp/locked.txt
RUN pip install --no-cache-dir -r /tmp/locked.txt \
    && rm /tmp/locked.txt \
    && pip install --no-cache-dir ruff==0.16.3

# verified pins from the working lane (2026-09-07): torch 2.13.0+cu130,
# transformers 5.16.1, sentence-transformers 6.0.1, scikit-learn 1.7.2,
# pandas 2.3.3, numpy 2.5.2, mlflow 3.15.1, optuna 4.4.0, sentencepiece 0.2.2

# non-root
RUN useradd -m -u 1000 trainer \
    && mkdir -p /home/trainer/.cache \
    && chown -R trainer:trainer /home/trainer
# docker HOME = the TRAIN_GPU lane root itself: 00_config.yaml, 06_run_all.py,
# data_pipe.py, STEPS.md, TRAIN/, EDA/, lib/, artifacts/ all at $PWD — the
# same layout as running in the repo. Only the lane is COPYed (+ src/ for
# the broadway packages lib imports), never the whole repo.
WORKDIR /app
COPY --chown=trainer:trainer pyproject.toml uv.lock /app/
COPY --chown=trainer:trainer src /app/src
COPY --chown=trainer:trainer       project/experiments/euromonitor/TRAIN_GPU/ /app/
RUN chown trainer:trainer /app
# /app must be trainer-owned: the lane mkdirs artifacts/ + logs/ at runtime,
# and ruff writes its cache under $PWD — both need write access

USER trainer
# mlflow local backend + artifacts default to artifacts/mlruns inside the lane
ENV MLFLOW_TRACKING_URI="" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
