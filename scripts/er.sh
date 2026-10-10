#!/usr/bin/env bash
# scripts/er.sh — THE entry point for every ER command.
#
# Runs an ER command under uv with the canonical environment loaded, so no
# caller has to hand-source a venv or export keys:
#   * PYTHONPATH=<this checkout>/src   (the branch's code — a worktree keeps its own)
#   * --env-file=<the repo .env>       (the ONE secrets file, one level above the
#                                       canonical checkout)
#
# The env-file location is NOT spelled here: it is asked of the ONE owner,
# core.project_root.ProjectRoot (which resolves the canonical checkout through
# git's common dir — the same answer from the main checkout and every linked
# worktree), alongside core.env_file.EnvFile which loads it in-process. Override
# for an unusual layout with EUROMONITOR_ENV_FILE.
#
# Usage (args after the script go straight to `uv run`):
#   scripts/er.sh python -m cli.laya_lane --kind kaggle --decision finetune --execute
#   scripts/er.sh python -m pytest tests/test_env_file.py -q
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The CHECKOUT whose code runs is the caller's (so a worktree keeps its branch's
# code), not the script's own location.
if ! checkout="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null)"; then
  checkout="$script_dir"
fi
export PYTHONPATH="$checkout/src${PYTHONPATH:+:$PYTHONPATH}"

env_file="${EUROMONITOR_ENV_FILE:-$(uv run --no-sync python -c \
  'import sys; from core.project_root import ProjectRoot; print(ProjectRoot.canonical(sys.argv[1]).parent / ".env")' \
  "$checkout")}"
if [[ ! -f "$env_file" ]]; then
  echo "er.sh: env file not found: $env_file (set EUROMONITOR_ENV_FILE)" >&2
  exit 2
fi

exec uv run --no-sync --env-file "$env_file" "$@"
