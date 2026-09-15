#!/usr/bin/env bash
set -euo pipefail

# Config-owned smoke parameters: colab.smoke_epochs, sweep.smoke_sample,
# sweep.train_fracs[0]. CPU is the safe default; override through COLAB_GPU.
# Anything other than CPU also needs the acknowledgement flag that gates
# non-CPU provisioning, so it cannot be requested by accident.
args=(--what smoke --gpu "${COLAB_GPU:-CPU}")
if [ "${COLAB_GPU:-CPU}" != "CPU" ]; then
  args+=(--allow-gpu)
fi
exec python -u colab_backend.py "${args[@]}"
