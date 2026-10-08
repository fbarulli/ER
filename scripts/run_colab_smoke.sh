#!/usr/bin/env bash
set -euo pipefail

# Config-owned smoke parameters: colab.smoke_epochs, sweep.smoke_sample,
# sweep.train_fracs[0]. CPU is the safe default; override through COLAB_GPU.
# Anything other than CPU also needs the acknowledgement flag that gates
# non-CPU provisioning, so it cannot be requested by accident.
# Sanctioned smoke path (owner ruling: single S suite = smoke_200):
# --what smoke is gate-held for legacy sampled preparation; the tracks
# lane with the frozen S suite is the working command (docs/colab-lane.md).
args=(--what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu "${COLAB_GPU:-CPU}")
if [ "${COLAB_GPU:-CPU}" != "CPU" ]; then
  args+=(--allow-gpu)
fi
PYTHONPATH=src exec .venv/bin/python -u colab_backend.py "${args[@]}"
