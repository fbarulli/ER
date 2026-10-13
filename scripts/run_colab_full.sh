#!/usr/bin/env bash
# Full E2E in one command: the THREE text-loss suites (mnrl, contrastive,
# triplet) serially on ONE Colab GPU T4 session, then release.
#
# Usage:
#   bash scripts/run_colab_full.sh            # launch the suites
#   bash scripts/run_colab_full.sh --dry-run  # print the resolved commands only
#
# Transcript: logs/colab/lane.log (the full/default session) —
#   tail -f logs/colab/lane.log
# The launcher owns the transcript: each launch is one run and truncates it
# exactly once (cli.colab.start_live_log, colab.py:974), so the file always
# holds the CURRENT suite; this wrapper never tees it (a second opener fights
# the truncation — see README-lanes.md).
#
# Inputs reuse: --prepared-input-package makes the lane COPY the sealed
# all_tracks_inputs package instead of re-packaging (colab.py:2464-2468), so
# all three suites ride one bundle and no local suite packaging runs (the
# smoke fixture guard is packaging-path only and is not needed here).
# Each launch releases the VM on exit; the next launch
# re-attaches the same session.
#
# Nothing is hardcoded: session/transcript/gpu read from config/training.yaml
# (colab block), the sealed package resolves through the canonical
# default_prepared_input_package()/resolve_prepared_input_package() pair, and
# the suites derive from the config/model_tracks*.yaml naming contract (the
# canonical MNRL file first, then the promoted loss variants).
#
# Credentials: AGENTS.md §7 documented temporary measure — ../.env is sourced.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; . /home/opc/ONE/.env; set +a

dry=""
case "${1:-}" in
  "") ;;
  --dry-run) dry=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

cfg_env="$(PYTHONPATH=src .venv/bin/python - <<'PY'
import shlex
from pathlib import Path
import yaml
from cli.colab import default_prepared_input_package, resolve_prepared_input_package
from cli.log_capture import lane_log

colab = yaml.safe_load(Path('config/training.yaml').read_text(encoding='utf-8'))['colab']
package = default_prepared_input_package()
if package is None:
    raise SystemExit('no sealed all_tracks_inputs package found: run the CPU bundle '
                     'lane (scripts/run_colab_bundle.sh) first; the full E2E never '
                     're-packages inputs')
print('COLAB_GPU=' + shlex.quote(str(colab['gpu'])))
print('COLAB_SESSION=' + shlex.quote(str(colab['session'])))
print('RUN_LOG=' + shlex.quote(str(lane_log(colab['log_dir'], colab['default_log_name']))))
print('PKG=' + shlex.quote(str(resolve_prepared_input_package(package))))
PY
)"
# shellcheck disable=SC2163  # values are shlex.quote'd by the emitter above
eval "$cfg_env"

# Suites derive from the promotion naming contract: the canonical MNRL
# manifest (text_loss null -> the trainer's SSOT default) plus every promoted
# loss variant. The canonical file leads so the baseline trains first.
suites=(config/model_tracks.yaml)
for suite in config/model_tracks__*.yaml; do suites+=("$suite"); done
if [ "${#suites[@]}" -ne 3 ]; then
  echo "expected 3 text-loss suites, derived ${#suites[@]}: ${suites[*]}" >&2
  exit 1
fi

echo "[colab-full] gpu=$COLAB_GPU session=$COLAB_SESSION package=$PKG transcript=$RUN_LOG"

for suite in "${suites[@]}"; do
  if [ ! -f "$suite" ]; then
    echo "SUITE $suite FAILED — missing manifest: promote it under config/ first" >&2
    exit 1
  fi
  name="$(basename "$suite" .yaml)"
  echo "=== [colab-full] SUITE $name ==="
  cmd=(".venv/bin/er-colab" --what tracks --tracks-config "$suite"
       --prepared-input-package "$PKG" --gpu "$COLAB_GPU")
  if [ "$COLAB_GPU" != "CPU" ]; then
    cmd+=(--allow-gpu)
  fi
  if [ -n "$dry" ]; then
    printf '[dry-run] %s\n' "${cmd[*]}"
  else
    PYTHONPATH=src "${cmd[@]}" || {
      echo "SUITE $name FAILED — see $RUN_LOG (tail below); chain aborted, VM released by the run" >&2
      [ -f "$RUN_LOG" ] && tail -n 20 "$RUN_LOG"
      exit 1
    }
  fi
done

echo "ALL 3 SUITES COMPLETE"
