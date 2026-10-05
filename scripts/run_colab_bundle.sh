#!/usr/bin/env bash
# Run the full CSV-to-inputs bundle lifecycle on a Colab VM CPU.
#
# Usage:
#   scripts/run_colab_bundle.sh [raw-export.csv] [extra cli.colab flags...]
#
# - The first positional argument is the raw export to upload (default:
#   ER/dataset.csv); it is forwarded to the lane as --dataset-csv. dataset.csv
#   IS git-tracked (commit 1084010 "track the five CSVs a clone needs, ignore
#   the rest"), so the upload is a freshness override that replaces the VM
#   checkout's committed bytes with an uncommitted export — not a gitignore
#   workaround.
# - The VM already holds a fresh git checkout (standard session bootstrap);
#   the lane runs `training.prepare_all` there end to end (CPU by default).
# - Downloads one delivery archive: results/training_prep/<run>/ (manifest,
#   handoff.json, timing_offenders.log, per-stage logs) + the regenerated
#   data artifacts (canonical/gate/deduped/labeled/final_validation CSVs,
#   track_setup/, prepared bundles).
# - Enforcement of WHICH export this is comes from the committed config
#   audit pins (audit.source_export_expected_rows / _sha256 in
#   config/training.yaml, drift threshold 0.0), asserted the first time the
#   prep loads the raw export: a cohort CSV against full-cohort pins fails
#   loudly at dedupe. rand_matching.gate_census_pin is NOT the enforcement
#   point — the gate_census stage re-records it from the regenerated
#   gate_results.csv before labeled_pairs compares against it.
# - CPU is the safe default and no flag is needed for it; COLAB_GPU overrides
#   it exactly like run_colab_smoke.sh, and anything other than CPU also
#   needs the --allow-gpu acknowledgement flag, so a non-CPU request cannot
#   happen by accident. Extra arguments pass through to cli.colab.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
  CSV="$1"
  shift
else
  CSV="dataset.csv"
fi
if [[ ! -f "$CSV" ]]; then
  echo "raw export not found: $CSV" >&2
  exit 1
fi
echo "[colab-bundle] raw export: $CSV ($(du -h "$CSV" | cut -f1))"
args=(--what bundle --dataset-csv "$CSV" --gpu "${COLAB_GPU:-CPU}")
if [ "${COLAB_GPU:-CPU}" != "CPU" ]; then
  args+=(--allow-gpu)
fi
PYTHONPATH=src .venv/bin/python -m cli.colab "${args[@]}" "$@"
