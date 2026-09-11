#!/usr/bin/env bash
set -euo pipefail

# Full-data concurrent training. Worker count, epochs, timeout, DVC, and paths
# are resolved by config/training.yaml through colab_backend.py.
exec python -u colab_backend.py --what train --gpu "${COLAB_GPU:-A100}"
