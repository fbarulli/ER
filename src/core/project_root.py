"""Project-root discovery — the leaf SSOT, imported by every config reader.

Why a leaf: three modules used to re-implement the same marker walk
(``config/`` + ``pyproject.toml``), and they had drifted — the root-override
env var ``EUROMONITOR_PROJECT_ROOT`` was honored by core.common but NOT by
core.critical_attributes' vocabulary reader, so an env-rooted test/CI run
could load two different vocabularies. One function, imported everywhere,
ends that.

Import safety: this module imports nothing from ``core`` (only ``os`` and
``pathlib``), so it can be loaded by common (common → schemas → … →
attribute_conflicts → critical_attributes) AND by critical_attributes
itself, during the very same import cycle, without ordering hazards.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT_ENV_VAR = "EUROMONITOR_PROJECT_ROOT"
_ROOT_MARKERS = ("config", "pyproject.toml")


def _has_root_markers(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in _ROOT_MARKERS)


def find_project_root(source_file: Path) -> Path:
    """Locate the project from stable markers, never a magic parent offset.

    The ``EUROMONITOR_PROJECT_ROOT`` env var, when set, must name a directory
    that carries the root markers (config/ + pyproject.toml); anything else
    fails loudly instead of silently falling back to the file-derived root.
    With no override the walk starts from ``source_file``'s parents.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if _has_root_markers(root):
            return root
        raise RuntimeError(
            f"{ROOT_ENV_VAR} must contain config/ and pyproject.toml: "
            f"{root}"
        )
    for candidate in Path(source_file).resolve().parents:
        if _has_root_markers(candidate):
            return candidate
    raise RuntimeError(f"Could not locate project root from {source_file}")


@lru_cache(maxsize=None)
def canonical_root(project_root: Path) -> Path:
    """The repository's ONE canonical root for ``project_root``.

    A linked git worktree (``git worktree add``) is itself a project root, but
    state that must be shared across worktrees — the logs roof — belongs in the
    MAIN worktree, not a per-checkout copy. A linked worktree's ``.git`` is a
    file pointing at ``<main>/.git/worktrees/<name>``, so the main root is three
    parents up from that git dir; a plain checkout's ``.git`` is a directory and
    is already canonical. Reading the pointer directly (no git subprocess) keeps
    the resolver usable under a mocked/subprocess-less runtime. Falls back to
    ``project_root`` when it is not inside a checkout (a bare VM checkout, a
    test's tmp root).
    """
    root = Path(project_root).resolve()
    git_entry = root / ".git"
    if git_entry.is_dir():
        return root
    if not git_entry.is_file():
        return root
    text = git_entry.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return root
    git_dir = Path(text.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    # Linked worktree: <main>/.git/worktrees/<name> -> <main>.
    if git_dir.parent.name == "worktrees" and git_dir.parent.parent.name == ".git":
        return git_dir.parent.parent.parent
    return root
