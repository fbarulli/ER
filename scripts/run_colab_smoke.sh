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
# so an operator override is never clobbered.
eval "$(PYTHONPATH=src .venv/bin/python colab_backend.py --print-lane-env --gpu "$LANE_GPU")"
args=(--what tracks --tracks-config "$EUROMONITOR_LANE_SUITE_CONFIG" --gpu "$LANE_GPU")
if [ "$LANE_GPU" != "CPU" ]; then
  args+=(--allow-gpu)
fi
# The local suite packaging rewrites these committed fixtures IN PLACE (the
# suite-matrix device/cohort serialization is written into the setup dir). Keep
# the checkout pristine: snapshot them before the lane runs and put them back on
# exit. No content is inspected or compared (data is never checked).
SMOKE_FIXTURES=(
  "data/prepared/smoke_200/ablation_templates/gnn_only/request.json"
  "data/prepared/smoke_200/ablation_templates/text/request.json"
  "data/prepared/smoke_200/text_export_request.json"
)
_smoke_backup="$(mktemp -d)"
_fixture_key() { printf '%s' "$1" | tr '/' '_'; }
for _fixture in "${SMOKE_FIXTURES[@]}"; do
  if [ -f "$_fixture" ]; then cp -p "$_fixture" "$_smoke_backup/$(_fixture_key "$_fixture")"; fi
done
_restore_smoke_fixtures() {
  for _fixture in "${SMOKE_FIXTURES[@]}"; do
    if [ -f "$_smoke_backup/$(_fixture_key "$_fixture")" ]; then
      cp -p "$_smoke_backup/$(_fixture_key "$_fixture")" "$_fixture"
    fi
  done
  # The ablation staging dirs the packaging writes beside the fixed templates.
  rm -rf data/prepared/smoke_200/ablation_templates/*_selected
  rm -rf "$_smoke_backup"
}
trap _restore_smoke_fixtures EXIT
# Distinct session + transcript per lane, so a CPU and a GPU smoke can run
# concurrently without sharing the launcher lock or truncating one file.
PYTHONPATH=src .venv/bin/python -u colab_backend.py "${args[@]}"
