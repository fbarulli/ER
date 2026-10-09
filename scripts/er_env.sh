#!/usr/bin/env bash
# er_env.sh — the ONE rule for locating and attaching the ER virtualenv.
#
# ER keeps a single uv-managed venv in the MAIN worktree (<main>/.venv). Every
# linked worktree (`git worktree add`) and every branch checkout must resolve to
# that same venv so a run behaves identically wherever it is launched. This
# script is the SSOT for that rule. `~/.bashrc` walks up for the same marker, so
# interactive shells agree with it without shelling out to git on every prompt.
#
# A linked worktree on another branch may not carry this script (it is committed
# on main). The optional `worktree_root` argument lets the MAIN worktree's copy
# act on any registered worktree, so `.githooks/post-checkout` can provision a
# fresh worktree that has no copy of its own.
#
# Usage:
#   scripts/er_env.sh venv     [worktree_root]  # print shared venv dir (exit 1 if absent)
#   scripts/er_env.sh link     [worktree_root]  # ensure that worktree's .venv points at it
#   scripts/er_env.sh activate [worktree_root]  # print exports to `eval` for it
#
# `worktree_root` defaults to the current directory. Exit codes: 0 ok; non-zero
# = not an ER checkout, or the shared venv is missing (a real error — never
# silently succeed).
set -euo pipefail

readonly VENV_PROMPT_DIR_MARKER='euromonitor-reconciliation'

die() {
  printf 'er_env: %s\n' "$*" >&2
  exit 1
}

# Run git as if started in $1 (empty => current directory). This is what lets
# the main worktree's copy of this script operate on an arbitrary worktree.
git_at() {
  local root="${1:-}"
  if [[ -n "$root" ]]; then
    git -C "$root" "${@:2}"
  else
    git "${@:2}"
  fi
}

# Main worktree root for the repo containing $1 (empty => CWD); derived from the
# shared git dir so every worktree of the same repo agrees on one answer.
main_worktree_root() {
  local common
  common="$(git_at "${1:-}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" ||
    die "not inside a git checkout"
  dirname "$common"
}

worktree_root() {
  git_at "${1:-}" rev-parse --show-toplevel 2>/dev/null || die "not inside a git checkout"
}

shared_venv_dir() {
  printf '%s/.venv' "$(main_worktree_root "${1:-}")"
}

cmd_venv() {
  local shared
  shared="$(shared_venv_dir "${1:-}")"
  [[ -x "$shared/bin/python" ]] || exit 1
  printf '%s\n' "$shared"
}

cmd_link() {
  local root main_root shared target
  root="$(worktree_root "${1:-}")"
  main_root="$(main_worktree_root "${1:-}")"
  shared="$main_root/.venv"
  [[ -x "$shared/bin/python" ]] || die "shared venv missing: $shared/bin/python"
  target="$root/.venv"
  if [[ "$root" == "$main_root" ]]; then
    : # the main worktree owns $shared as a real directory
  elif [[ -L "$target" ]]; then
    [[ "$(readlink -f "$target")" == "$(readlink -f "$shared")" ]] || ln -sfn "$shared" "$target"
  elif [[ -e "$target" ]]; then
    die "$target exists and is not a symlink; refusing to clobber a real venv"
  else
    ln -s "$shared" "$target"
  fi
  # .venv-link was a redundant second name for the same env; drop it.
  rm -f "$root/.venv-link"
  printf 'er_env: %s -> %s\n' "$target" "$shared"
}

cmd_activate() {
  local shared
  shared="$(shared_venv_dir "${1:-}")"
  [[ -x "$shared/bin/python" ]] || die "shared venv missing: $shared/bin/python"
  printf 'export VIRTUAL_ENV=%q\n' "$shared"
  printf 'case ":$PATH:" in *":%s/bin:"*) ;; *) export PATH=%q/bin:"$PATH" ;; esac\n' "$shared" "$shared"
  printf 'export ER_VENV_PROMPT_MARKER=%q\n' "$VENV_PROMPT_DIR_MARKER"
}

case "${1:-}" in
  venv)     cmd_venv "${2:-}" ;;
  link)     cmd_link "${2:-}" ;;
  activate) cmd_activate "${2:-}" ;;
  *) die "usage: er_env.sh {venv|link|activate} [worktree_root]" ;;
esac
