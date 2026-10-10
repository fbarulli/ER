"""ProjectRoot: the marker walk and the canonical-checkout step.

A training checkout under ``<canonical>/.worktrees/<name>`` must not move the
shared root. ``find`` returns the launching checkout (its own config/code, so a
diverged worktree is never paired with another branch's config), while
``canonical`` steps out to the checkout the worktrees share — what the ``.env``
and the shared log roof hang off. An explicit ``EUROMONITOR_PROJECT_ROOT``
override wins for ``find`` and fails loud when it names a non-root.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.project_root import ProjectRoot


def _mark_root(path: Path) -> Path:
    (path / "config").mkdir(parents=True)
    (path / "config" / "paths.yaml").write_text("{}\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_canonical_steps_out_of_a_linked_worktree(tmp_path, monkeypatch):
    canonical = _mark_root(tmp_path / "canonical")
    _git(canonical, "init", "-q")
    _git(canonical, "config", "user.email", "t@example.com")
    _git(canonical, "config", "user.name", "t")
    _git(canonical, "add", "-A")
    _git(canonical, "commit", "-qm", "init")
    worktree = tmp_path / "linked"
    _git(canonical, "worktree", "add", "-q", str(worktree))
    source = worktree / "src" / "core" / "common.py"

    monkeypatch.delenv(ProjectRoot.ENV_VAR, raising=False)
    # the launching checkout owns the code/config...
    assert ProjectRoot.find(source) == worktree.resolve()
    # ...but the shared root (and the .env beside it) is the canonical checkout.
    assert ProjectRoot.canonical(source) == canonical.resolve()

    monkeypatch.setenv(ProjectRoot.ENV_VAR, str(canonical))
    assert ProjectRoot.find(source) == canonical.resolve()

    monkeypatch.setenv(ProjectRoot.ENV_VAR, str(tmp_path / "not-a-root"))
    with pytest.raises(RuntimeError):
        ProjectRoot.find(source)
