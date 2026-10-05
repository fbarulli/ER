#!/usr/bin/env bash
# Run the CSV-to-inputs bundle lifecycle on a Colab VM CPU.
#
# Usage:
#   scripts/run_colab_bundle_cpu.sh [raw-export.csv]
#
# Uses the standalone lane (src/cli/colab_bundle.py): uploads the raw
# export, re-pins the VM audit config to the uploaded cohort with a plain
# line filter (no regex), runs prepare_all end to end on the VM, and
# downloads one delivery archive to training_results/colab_bundle_<id>/.
# The VM is never torn down by this lane.
set -euo pipefail
cd "$(dirname "$0")/.."

CSV="${1:-dataset.csv}"
if [[ ! -f "$CSV" ]]; then
  echo "raw export not found: $CSV" >&2
  exit 1
fi
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv "$CSV"
