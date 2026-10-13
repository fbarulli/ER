#!/usr/bin/env bash
# Run the full CSV-to-inputs bundle lifecycle on a Colab VM CPU.
#
# Usage:
#   scripts/run_colab_bundle.sh [committed-export.csv] [extra cli.colab_bundle flags...]
#
# The committed-export CPU bundle lane (`cli.colab_bundle` -> ColabCPULane):
# the VM SPARSE-checks out the declared paths plus the resolved base-model
# directory, `data/prepared/smoke_200` and the committed export, so the raw
# export RIDES the checkout and the remote launcher remaps it onto
# `dataset.csv` — this lane uploads no export and never full-clones. The
# provisioning order (check_colab_cli -> ensure_session -> sparse
# prepare_remote_layout -> minimal deps) is owned by the lane class
# (`ColabCPULaneProvision`), never re-spelled here.
#
# - The first positional argument is the committed export (default:
#   ER/dataset.csv); it must be a config `bundle_prep.export_csvs` entry
#   (`dataset.csv` / `dataset_3k.csv`) that is committed and pushed, because
#   the sparse checkout is the only source of its bytes.
# - Downloads one delivery archive to the declared bundle delivery root
#   (paths.training_results_dir -> TRAINING_RESULTS/colab_bundle_bundle_<run_id>/
#   bundle_delivery.tar.zst, via ColabCPULane().delivery_root). The archive
#   CONTAINS the VM's preparation run tree (results/training_prep/<run>/) --
#   manifest.json, handoff.json, timing_offenders.log, per-stage logs -- plus
#   the regenerated data artifacts (canonical/gate/deduped/labeled/
#   final_validation CSVs, track_setup/, prepared bundles).
# - The export's cohort is detected structurally (a byte-size compare against
#   the staged cohort CSVs, core.common.mounted_cohort) and tagged
#   (ER_COHORT_TAG); there is no content-hash pin and no drift gate.
# - CPU only: the committed-export lane never allocates an accelerator.
#   Extra arguments pass through to cli.colab_bundle (--resume-from,
#   --resume-run-id, --resume-state, --preflight-only).
set -euo pipefail
cd "$(dirname "$0")/.."

# The ONE file this run writes is the bundle lane's DECLARED transcript: the
# launcher prints the lane's declarations (ColabSpec.lane_env_exports ->
# session, transcript, suite config) via --print-lane-env and this evaluates
# them, so no name is spelled in shell; an already-exported value wins.
# The lane is its own opener: cli.colab.start_live_log truncates the declared
# transcript exactly once per run and then holds it O_APPEND. This wrapper's
# tee is a second writer on that SAME file, so it must only append (tee -a)
# from t=0: a plain tee opens O_TRUNC at exec time, and its stale write offset
# lands after the lane's truncation — the kernel zero-fills the gap (the
# 60-NUL hole that made the transcript unreadable in editors).
eval "$(PYTHONPATH=src .venv/bin/python colab_backend.py --print-lane-env --gpu CPU)"
RUN_LOG="$(PYTHONPATH=src .venv/bin/python -m cli.log_capture --lane cpu_lane/log \
  --name "$EUROMONITOR_LANE_LOG")"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[colab-bundle $(TZ='Europe/Paris' date '+%Y-%m-%dT%H:%M:%S %Z')] run log: $RUN_LOG"

if [[ $# -gt 0 ]]; then
  CSV="$1"
  shift
else
  CSV="dataset.csv"
fi
if [[ ! -f "$CSV" ]]; then
  echo "[colab-bundle $(TZ='Europe/Paris' date '+%Y-%m-%dT%H:%M:%S %Z')] committed export not found: $CSV" >&2
  exit 1
fi
echo "[colab-bundle $(TZ='Europe/Paris' date '+%Y-%m-%dT%H:%M:%S %Z')] committed export: $CSV ($(du -h "$CSV" | cut -f1))"
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv "$CSV" "$@"
