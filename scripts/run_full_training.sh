#!/usr/bin/env bash
set -euo pipefail

# Full baseline embedding export and concurrent text/GNN/hybrid training,
# followed by DVC collection, shutdown, local reports, and final publication.
# Runtime defaults to colab.gpu in config/training.yaml (T4).
args=(--what tracks --allow-gpu)
if [ -n "${COLAB_GPU:-}" ]; then
  args+=(--gpu "$COLAB_GPU")
fi
exec python -u colab_backend.py "${args[@]}" "$@"
