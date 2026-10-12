#!/usr/bin/env bash
# Smoke fixture guard (sourced by scripts/run_colab_smoke.sh; not executed).
#
# The local suite packaging stages a DisposableSetup copy and PUBLISHES it by
# swapping the tree into the canonical setup root (model_tracks.config.setup_root
# of the lane's suite config), rewriting tracked fixtures IN PLACE under that
# directory (src/model_tracks/package.py). The swap is DIR-SCOPED, so the guard
# snapshots every tracked file under the derived setup dir — never a hand-listed
# file set (the post-9b342be class: the writers grew, the list did not, and the
# checkout kept the disposable-slot paths the deleted copy had serialized).
# No content is inspected or compared (data is never checked); restore is
# byte-for-byte by construction (cp -p of the snapshot).

smoke_guard_derive() {
  # $1: python source to run with the repo src on the path (wrapper convention).
  PYTHONPATH=src .venv/bin/python -c "$1"
}

smoke_guard_setup_dir() {
  # The ONE setup root the publish swap targets, read from the suite config.
  SMOKE_GUARD_SUITE_CONFIG="$1" smoke_guard_derive '
import os
from pathlib import Path
from model_tracks.config import load_config, setup_root
print(setup_root(load_config(Path(os.environ["SMOKE_GUARD_SUITE_CONFIG"]))))'
}

smoke_guard_slot() {
  # The snapshot lives under the ONE configured scratch root (core.workspace),
  # never /tmp; pid-scoped so concurrent lanes own distinct slots.
  smoke_guard_derive '
import os
from core.workspace import Workspace
print(Workspace.from_config().slot(f"smoke-guard-{os.getpid()}"))'
}

_smoke_guard_key() { printf '%s' "$1" | tr '/' '_'; }

smoke_guard_snapshot() {
  # $1 setup dir, $2 backup slot. The list is DERIVED from git: every tracked
  # file under the setup dir — the same scope the swap rewrites.
  local _fixture
  SMOKE_GUARD_SETUP_DIR="$1"
  SMOKE_GUARD_BACKUP="$2"
  mapfile -t SMOKE_GUARD_FIXTURES < <(git ls-files -- "$SMOKE_GUARD_SETUP_DIR")
  mkdir -p "$SMOKE_GUARD_BACKUP"
  for _fixture in "${SMOKE_GUARD_FIXTURES[@]}"; do
    if [ -f "$_fixture" ]; then
      cp -p "$_fixture" "$SMOKE_GUARD_BACKUP/$(_smoke_guard_key "$_fixture")"
    fi
  done
}

smoke_guard_restore() {
  local _fixture
  for _fixture in "${SMOKE_GUARD_FIXTURES[@]}"; do
    if [ -f "$SMOKE_GUARD_BACKUP/$(_smoke_guard_key "$_fixture")" ]; then
      cp -p "$SMOKE_GUARD_BACKUP/$(_smoke_guard_key "$_fixture")" "$_fixture"
    fi
  done
  # The ablation staging dirs the packaging writes beside the fixed templates.
  rm -rf "${SMOKE_GUARD_SETUP_DIR:?}"/ablation_templates/*_selected
  rm -rf "${SMOKE_GUARD_BACKUP:?}"
}

smoke_guard_install() {
  # The wrapper's ONE call: derive the setup root from the lane's suite config,
  # snapshot every tracked fixture under it, and put it all back on exit.
  local _setup_dir _backup
  _setup_dir="$(smoke_guard_setup_dir "$1")"
  _backup="$(smoke_guard_slot)"
  smoke_guard_snapshot "$_setup_dir" "$_backup"
  trap smoke_guard_restore EXIT
}
