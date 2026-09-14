#!/usr/bin/env bash
set -euo pipefail

# Full-data concurrent training. Worker count, epochs, timeout, DVC, and paths
# are resolved by config/training.yaml through colab_backend.py. CPU is the
# default; a GPU must be requested explicitly with COLAB_GPU and --allow-gpu.
exec python -u colab_backend.py --what train --gpu "${COLAB_GPU:-CPU}"
