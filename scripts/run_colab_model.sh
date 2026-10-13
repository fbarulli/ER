#!/usr/bin/env bash
# Choose which Colab GPU model(s) to run — ONE launch per model, each on its
# OWN fresh Colab session, serially. Use this when the Colab lane misbehaves
# and you want per-model control instead of the run_colab_full.sh chain.
#
# What each launch already does (nothing re-spelled here): it trains the
# chosen suite on the sealed canonical bundle, runs the suite's declared
# post-training attribute ablation, downloads the result archive with ALL
# track artifacts (results/model_tracks/<run_id>/...), and releases the VM at
# the end. Transcript per session: logs/colab/lane_<session>-<model>.log
# (the SSOT session-derived name).
#
# Usage:
#   bash scripts/run_colab_model.sh --list                 # model names to choose from
#   bash scripts/run_colab_model.sh --dry-run --all        # print the resolved commands
#   bash scripts/run_colab_model.sh contrastive            # one model, fresh session
#   bash scripts/run_colab_model.sh model_tracks:text      # one track of one model
#   bash scripts/run_colab_model.sh --gpu A100 contrastive:gnn_only triplet:text
#
# Tracks are the suite taxonomy: text, gnn_only, cascade (cascade only with
# both trained lanes — the suite config validator refuses it alone). With no
# :tracks the whole declared cohort runs, exactly like the chain launch.
#
# Model names are derived from the suite catalogue (scripts/colab_env.sh):
# the canonical manifest is 'model_tracks', promoted loss variants are their
# stems. A serial failure aborts the list loud — fix or rerun the rest.
#
# Credentials: AGENTS.md §7 documented temporary measure — ../.env is sourced.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; . /home/opc/ONE/.env; set +a
. scripts/colab_env.sh

dry="" gpu="" want_all="" want_list="" models=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry=1 ;;
    --gpu) gpu="${2:?--gpu needs a value (e.g. T4|A100|CPU)}"; shift ;;
    --all) want_all=1 ;;
    --list) want_list=1 ;;
    -*) echo "unknown flag: $1 (see header usage)" >&2; exit 2 ;;
    *) models+=("$1") ;;
  esac
  shift
done

labels=()
while IFS= read -r suite; do labels+=("$(suite_label "$suite")"); done < <(resolve_colab_suites)
if [ -n "$want_list" ]; then
  printf '%s\n' "${labels[@]}"
  exit 0
fi

cfg_env="$(resolve_colab_env)" || exit 1
# shellcheck disable=SC2163  # values are shlex.quote'd by the emitter above
eval "$cfg_env"
[ -n "$gpu" ] && COLAB_GPU="$gpu"

declare -A label_to_file=()
while IFS= read -r suite; do
  label="$(suite_label "$suite")"
  label_to_file["$label"]="$suite"
done < <(resolve_colab_suites)
[ -n "$want_all" ] && models=("${labels[@]}")
if [ "${#models[@]}" -eq 0 ]; then
  echo "usage: $0 [--dry-run] [--gpu G] <model>... | --all | --list" >&2
  echo "models: ${labels[*]}" >&2
  exit 2
fi

cleanup_release() { .venv/bin/er-colab --what stop >/dev/null 2>&1 || true; }
trap cleanup_release EXIT

for model in "${models[@]}"; do
  spec="${model%%:*}"; tracks="${model#*:}"; [ "$tracks" = "$spec" ] && tracks=""
  suite="${label_to_file[$spec]:-}"
  if [ -z "$suite" ]; then
    echo "unknown model '$spec'; choose from: ${labels[*]} (optionally <model>:<track,...>)" >&2
    exit 2
  fi
  # One fresh session per model (and per selection): the launcher derives its
  # transcript from the session name; never reuse a churned runtime's binding.
  export EUROMONITOR_COLAB_SESSION="${COLAB_SESSION}-${spec}"
  [ -n "$tracks" ] && export EUROMONITOR_COLAB_SESSION+="-${tracks//,/-}"
  unset EUROMONITOR_LANE_APPEND
  transcript="$(PYTHONPATH=src .venv/bin/python -c 'import cli.colab as c; print(c.lane_transcript_path())')"
  cmd=(".venv/bin/er-colab" --what tracks --tracks-config "$suite"
       --prepared-input-package "$PKG" --gpu "$COLAB_GPU")
  [ -n "$tracks" ] && cmd+=(--tracks "${tracks//,/ }")
  [ "$COLAB_GPU" != "CPU" ] && cmd+=(--allow-gpu)
  if [ -n "$dry" ]; then
    printf '[dry-run] session=%s transcript=%s\n[dry-run] %s\n' \
      "$EUROMONITOR_COLAB_SESSION" "$transcript" "${cmd[*]}"
    continue
  fi
  echo "[colab-model] START model=$model session=$EUROMONITOR_COLAB_SESSION transcript=$transcript"
  # One bounded retry on a fresh runtime (the chain rule, 762fa9b): Colab host
  # churn kills SOME GPU runtimes 60-90s after handshake; a dead-early VM is
  # transient infra, not a model defect. Second attempt keeps the fail-loud rule.
  if ! PYTHONPATH=src "${cmd[@]}" > /dev/null; then
    echo "[colab-model] model=$model attempt 1 failed — retrying once on a fresh runtime"
    export EUROMONITOR_COLAB_SESSION="${COLAB_SESSION}-${model}-r2"
    cmd=(".venv/bin/er-colab" --what tracks --tracks-config "$suite"
         --prepared-input-package "$PKG" --gpu "$COLAB_GPU")
    [ -n "$tracks" ] && cmd+=(--tracks "${tracks//,/ }")
    [ "$COLAB_GPU" != "CPU" ] && cmd+=(--allow-gpu)
    if ! PYTHONPATH=src "${cmd[@]}" > /dev/null; then
      echo "[colab-model] model=$model FAILED twice — driver aborted, remaining models not started" >&2
      exit 1
    fi
  fi
  echo "[colab-model] DONE model=$model — results + artifacts + ablation under results/model_tracks/"
done
[ -n "$dry" ] || echo "[colab-model] ALL CHOSEN MODELS COMPLETE: ${models[*]}"
