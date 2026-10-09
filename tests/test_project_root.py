"""Canonical root: a run launched from a nested worktree logs at the canonical root.

A training checkout under ``<canonical>/.worktrees/<name>`` must not move
``logs/laya/lane.log``: with ``EUROMONITOR_TRAIN_ROOT`` exported the root is
canonical regardless of which checkout launched the run; unset, the nearest
marker-bearing checkout (the sane default) is used.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.project_root import ROOT_ENV_VAR, TRAIN_ROOT_ENV_VAR, find_project_root


def _mark_root(path: Path) -> Path:
    (path / "config").mkdir(parents=True)
    (path / "pyproject.toml").write_text("", encoding="utf-8")
    return path


def test_nested_worktree_run_writes_to_the_canonical_log_root(tmp_path, monkeypatch):
    canonical = _mark_root(tmp_path / "canonical").resolve()
    worktree = _mark_root(canonical / ".worktrees" / "laya-logroot")
    source = worktree / "src" / "core" / "common.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")

    monkeypatch.delenv(TRAIN_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    assert find_project_root(source) == worktree.resolve()

    monkeypatch.setenv(TRAIN_ROOT_ENV_VAR, str(canonical))
    assert find_project_root(source) == canonical
    assert (canonical / "logs" / "laya" / "lane.log").parent == canonical / "logs" / "laya"

    monkeypatch.setenv(TRAIN_ROOT_ENV_VAR, str(tmp_path / "not-a-root"))
    with pytest.raises(RuntimeError):
        find_project_root(source)
