#!/usr/bin/env bash
# scripts/er.sh — THE entry point for every ER command.
#
# Runs an ER command under uv with the canonical environment loaded, so no
# caller has to hand-source a venv or export keys:
#   * PYTHONPATH=<this checkout>/src   (the branch's code — a worktree keeps its own)
#   * --env-file=<the repo .env>       (the ONE secrets file, one level above the
#                                       canonical checkout)
#
# Neither path is hardcoded. The canonical checkout is whatever `git` calls the
# common dir — the SAME answer from the main checkout and from any linked
# worktree — and the .env sits beside it, exactly where core.env_file.EnvFile
# looks. Override for an unusual layout with EUROMONITOR_ENV_FILE.
#
# Usage (args after the script go straight to `uv run`):
#   scripts/er.sh python -m cli.laya_lane --kind kaggle --decision finetune --execute
#   scripts/er.sh python -m pytest tests/test_laya_smoke.py -q
set -euo pipefail

# The CHECKOUT whose code runs is the caller's (so a worktree keeps its branch's
# code), not the script's own location.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! checkout="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null)"; then
  checkout="$script_dir"
fi
# `--git-common-dir` is shared by the main checkout and every linked worktree, so
# its parent is the canonical checkout root from anywhere — no literal, no env var.
common_dir="$(git -C "$checkout" rev-parse --path-format=absolute --git-common-dir)"
canonical_root="$(dirname "$common_dir")"

env_file="${EUROMONITOR_ENV_FILE:-$canonical_root/../.env}"
if [[ ! -f "$env_file" ]]; then
  echo "er.sh: env file not found: $env_file (set EUROMONITOR_ENV_FILE)" >&2
  exit 2
fi

export PYTHONPATH="$checkout/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync --env-file "$env_file" "$@"
