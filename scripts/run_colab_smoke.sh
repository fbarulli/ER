#!/usr/bin/env bash
set -euo pipefail

# Config-owned smoke parameters: colab.smoke_epochs, sweep.smoke_sample,
# sweep.train_fracs[0]. CPU is the safe default; override through COLAB_GPU.
# Anything other than CPU also needs the acknowledgement flag that gates
# non-CPU provisioning, so it cannot be requested by accident.
# Sanctioned smoke path (owner ruling: single S suite = smoke_200):
# --what smoke is gate-held for legacy sampled preparation; the tracks
# lane with the frozen S suite is the working command (docs/colab-lane.md).
LANE_GPU="${COLAB_GPU:-CPU}"
# Nothing below is spelled here: the launcher prints the selected lane's
# declarations (ColabSpec.lanes) and the ONE data-bundle suite config both lanes
# train from (ColabSpec.data_bundle). An exported override still wins, because
# the eval runs last.
eval "$(PYTHONPATH=src .venv/bin/python colab_backend.py --print-lane-env --gpu "$LANE_GPU")"
args=(--what tracks --tracks-config "$EUROMONITOR_LANE_SUITE_CONFIG" --gpu "$LANE_GPU")
if [ "$LANE_GPU" != "CPU" ]; then
  args+=(--allow-gpu)
fi
# Distinct session + transcript per lane, so a CPU and a GPU smoke can run
# concurrently without sharing the launcher lock or truncating one file.
PYTHONPATH=src exec .venv/bin/python -u colab_backend.py "${args[@]}"
