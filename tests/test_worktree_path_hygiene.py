"""Guard: tests must import THIS worktree's modules, not the shared primary tree.

The venv is shared across worktrees (its editable ``.pth`` points at the primary
checkout). ``tests/conftest.py`` pins sys.path to this worktree and fails loud;
these tests make that guarantee visible and regression-proof.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cli
import core
import training

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_SRC = (REPO_ROOT / "src").resolve()


def _assert_under_repo(module):
    resolved = Path(module.__file__).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"{module.__name__} resolved outside the worktree: {resolved}")


def test_core_training_cli_resolve_to_this_worktree():
    for module in (core, training, cli):
        _assert_under_repo(module)


def test_laya_hpo_modules_resolve_to_this_worktree():
    from cli import laya_hpo, laya_lane
    from training import (
        hpo_control_plane,
        hpo_observability,
        hpo_registry,
        laya_hpo_options,
        laya_hpo_runtime,
    )
    for module in (laya_hpo, laya_lane, hpo_control_plane, hpo_observability,
                   hpo_registry, laya_hpo_options, laya_hpo_runtime):
        _assert_under_repo(module)


def test_worktree_src_precedes_any_other_checkout_src():
    """The worktree src must win the import race against the shared .pth.

    Some tests legitimately add paths, so the invariant is ORDER, not absence:
    no other checkout's ``src`` may sit ahead of this worktree's ``src``.
    """
    resolved = [Path(entry).resolve() for entry in sys.path if entry]
    assert WORKTREE_SRC in resolved, "worktree src is not on sys.path"
    worktree_index = resolved.index(WORKTREE_SRC)
    ahead = [
        str(path) for index, path in enumerate(resolved)
        if index < worktree_index and path.name == "src" and path != WORKTREE_SRC
    ]
    assert ahead == [], f"a foreign src precedes the worktree src: {ahead}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
