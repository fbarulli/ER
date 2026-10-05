#!/usr/bin/env bash
# Run the full CSV-to-inputs bundle lifecycle on a Colab VM CPU.
#
# Usage:
#   scripts/run_colab_bundle.sh [raw-export.csv]
#
# - Uploads the raw export (default: ER/dataset.csv) to the VM checkout.
# - The VM already holds a fresh git checkout (standard session bootstrap);
#   the lane runs `training.prepare_all` there end to end (CPU only).
# - Downloads one delivery archive: results/training_prep/<run>/ (manifest,
#   handoff.json, timing_offenders.log, per-stage logs) + the regenerated
#   data artifacts (canonical/gate/deduped/labeled/final_validation CSVs,
#   track_setup/, prepared bundles).
# - The VM's committed config audit pins enforce WHICH export this is:
#   uploading a cohort CSV against full-cohort pins fails loudly at dedupe.
set -euo pipefail
cd "$(dirname "$0")/.."

CSV="${1:-dataset.csv}"
if [[ ! -f "$CSV" ]]; then
  echo "raw export not found: $CSV" >&2
  exit 1
fi
echo "[colab-bundle] raw export: $CSV ($(du -h "$CSV" | cut -f1))"
PYTHONPATH=src .venv/bin/python -m cli.colab --what bundle
