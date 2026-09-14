#!/usr/bin/env bash
set -euo pipefail

# Config-owned smoke parameters: colab.smoke_epochs, sweep.smoke_sample,
# sweep.train_fracs[0]. CPU is the safe default; override through COLAB_GPU.
exec python -u colab_backend.py --what smoke --gpu "${COLAB_GPU:-CPU}"
