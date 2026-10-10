"""EnvFile: the ONE reader for the repo ``.env`` (offline, hermetic).

``search_paths`` defaulted to ``<root>/.env`` + ``<root>.parent/.env``, so a
linked worktree (``<canonical>/.worktrees/<name>``) looked at ``.worktrees/.env``
and never found the repo's secrets — every credential had to be exported by
hand. The location is now resolved through git's common dir (the same answer
from the main checkout and every worktree), so finding and loading the file is
reproducible and no caller spells a path.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core.env_file import EnvFile


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def test_env_file_resolves_and_loads_from_a_linked_worktree(
        tmp_path, monkeypatch):
    """A worktree resolves the canonical ``.env`` and ``apply`` loads it."""
    canonical = tmp_path / "canonical"
    (canonical / "config").mkdir(parents=True)
    (canonical / "config" / "paths.yaml").write_text("{}\n", encoding="utf-8")
    (canonical / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _git(canonical, "init", "-q")
    _git(canonical, "config", "user.email", "t@example.com")
    _git(canonical, "config", "user.name", "t")
    _git(canonical, "add", "-A")
    _git(canonical, "commit", "-qm", "init")
    # the linked worktree is a full checkout too: it carries the root markers.
    worktree = tmp_path / "linked"
    _git(canonical, "worktree", "add", "-q", str(worktree))
    # the ONE env file lives beside the CANONICAL checkout (its parent).
    (tmp_path / ".env").write_text("ER_TEST_SECRET=from_canonical\n",
                                   encoding="utf-8")
    monkeypatch.delenv("ER_TEST_SECRET", raising=False)

    assert EnvFile.path(canonical) == tmp_path / ".env"
    assert EnvFile.path(worktree) == tmp_path / ".env"
    assert EnvFile.apply(worktree) == tmp_path / ".env"
    assert os.environ["ER_TEST_SECRET"] == "from_canonical"
