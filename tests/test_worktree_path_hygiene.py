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


def test_no_foreign_src_is_on_sys_path():
    foreign = [
        entry for entry in sys.path
        if entry.endswith("/src")
        and Path(entry).resolve() != WORKTREE_SRC
    ]
    assert foreign == [], f"foreign src on sys.path: {foreign}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
