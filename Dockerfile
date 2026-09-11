# EuromonitoR reproducibility image — the euromonitor training lane.
#
# Build (from THIS standalone folder — the lane root):
#   docker build -t euromonitor-train-gpu .
#
# Run (WORKDIR = the lane root; artifacts/ is the volume point):
#   docker run --rm -v $(pwd)/artifacts:/app/artifacts \
#     euromonitor-train-gpu python src/training/data_prep.py
#   docker run --gpus all --rm \
#     -v $(pwd)/artifacts:/app/artifacts \
#     euromonitor-train-gpu python src/training/train.py --model artifacts/models/all-MiniLM-L6-v2
#
# CPU vs GPU: install the +cu torch wheel for CUDA hosts (swap the
# requirements pin to torch==2.14.0+cu130 or use --extra-index-url
# https://download.pytorch.org/whl/cu130); the plain wheel runs CPU-only.
# deberta note: deberta-v3 is ~2000x slower on CPU; deberta scoring and
# training belong on the GPU lane.

# ── stage 1: dependency install (single layer, cache-friendly) ─────────────
FROM python:3.12-slim-bookworm AS deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# tini: proper signal handling (SIGTERM kills training runs cleanly);
# git: reproducible provenance stamps; curl: healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini git curl \
    && rm -rf /var/src/core/apt/lists/*

COPY --from=deps /usr/local/src/core/python3.12/site-packages /usr/local/src/core/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# non-root
RUN useradd -m -u 1000 trainer \
    && mkdir -p /home/trainer/.cache \
    && chown -R trainer:trainer /home/trainer
# docker HOME = the lane root itself: config/, src/, run_all.py, STEPS.md,
# artifacts/ — all at $PWD, the same layout as running in the repo.
WORKDIR /app

# COPY-source guard: every src below must exist at the repo root — docker
# only fails on a missing COPY after the apt layers have baked, so check
# before building (from the repo root; EDA/ left this list 2026-09-10 when
# the dir was deleted):
#   for s in requirements.txt STEPS.md README.md config src run_all.py \
#            colab_backend.py; do
#     [ -e "$s" ] || echo "MISSING COPY source: $s"; done
COPY --chown=trainer:trainer requirements.txt STEPS.md README.md /app/
COPY --chown=trainer:trainer config /app/config
COPY --chown=trainer:trainer src /app/src
COPY --chown=trainer:trainer run_all.py colab_backend.py /app/
RUN chown trainer:trainer /app
# /app must be trainer-owned: the lane mkdirs artifacts/ + logs/ at runtime,
# and ruff writes its cache under $PWD — both need write access

USER trainer
# mlflow local backend + artifacts default to artifacts/mlruns inside the lane
ENV MLFLOW_TRACKING_URI="" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/home/trainer/.cache \
    TOKENIZERS_PARALLELISM=false

# lint + contract gate: the image must boot the config SSOT and pass every
# oracle before it is usable (byte-determinism of the lane depends on it)
RUN PYTHONPATH=/app/src python -c "import core.common; import pipeline; import training.masking; \
    import training.folds; import core.schemas; print('config SSOT + schemas import OK')" \
    && ruff check src/ run_all.py colab_backend.py

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/training/selftest.py"]
