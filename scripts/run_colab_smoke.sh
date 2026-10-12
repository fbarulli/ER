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
# declarations (ColabSpec.lane_env_exports -> session, transcript, and the ONE
# data-bundle suite config both lanes train from) and this evaluates them. An
# already-exported value wins: the launcher reads the current environment first,
# so an operator override is never clobbered. The wrapper then announces the
# declared transcript; the lane itself is its opener — cli.colab.start_live_log
# truncates it ONCE at t=0 and holds it open in APPEND mode (never "w"), so the
# detached self-watch child can share the ONE file through its inherited
# O_APPEND fd. The wrapper must not tee this same file: a second opener would
# fight that truncation.
eval "$(PYTHONPATH=src .venv/bin/python colab_backend.py --print-lane-env --gpu "$LANE_GPU")"
RUN_LOG="$(PYTHONPATH=src .venv/bin/python -m cli.log_capture --lane colab \
  --name "$EUROMONITOR_LANE_LOG")"
echo "[colab-smoke $(TZ='Europe/Paris' date '+%Y-%m-%dT%H:%M:%S %Z')] run log: $RUN_LOG"
args=(--what tracks --tracks-config "$EUROMONITOR_LANE_SUITE_CONFIG" --gpu "$LANE_GPU")
if [ "$LANE_GPU" != "CPU" ]; then
  args+=(--allow-gpu)
fi
# The local suite packaging publishes its staged DisposableSetup by swapping
# the tree into the canonical setup root, rewriting tracked fixtures IN PLACE
# under that dir. Keep the checkout pristine: the fixture guard snapshots
# EVERY tracked file under that dir — the scope is derived from the same SSOT
# the swap targets (model_tracks.config.setup_root of the lane's suite config),
# never a hand-listed file set — and restores on exit.
. scripts/smoke_fixture_guard.sh
smoke_guard_install "$EUROMONITOR_LANE_SUITE_CONFIG"
# Distinct session + transcript per lane, so a CPU and a GPU smoke can run
# concurrently without sharing the launcher lock or truncating one file.
PYTHONPATH=src .venv/bin/python -u colab_backend.py "${args[@]}"
