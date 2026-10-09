"""Project-root discovery — the leaf SSOT, imported by every config reader.

Why a leaf: modules used to re-implement the same marker walk
(``config/`` + ``pyproject.toml``), and they had drifted — the root-override
env var ``EUROMONITOR_PROJECT_ROOT`` was honored by core.common but NOT by
core.critical_attributes' vocabulary reader, so an env-rooted test/CI run
could load two different vocabularies. One function, imported everywhere,
ends that.

Two overrides, most specific first. Both are validated against the root
markers and fail loudly when set to a non-root:

  EUROMONITOR_TRAIN_ROOT    the CANONICAL, checkout-independent data/log root.
                            A run launched from ANY checkout (including a
                            linked worktree) resolves here, so
                            ``logs/laya/lane.log`` and every shared artifact
                            land at the canonical root instead of the scratch
                            checkout that happened to launch the run.
  EUROMONITOR_PROJECT_ROOT  the legacy checkout/code root (tests, CI).

With neither set the root is the nearest marker-bearing ancestor of
``source_file`` — the checkout's own root, so a worktree test run still reads
that worktree. Training worktrees must live under
``<canonical>/.worktrees/<name>`` (never ``/tmp``); with TRAIN_ROOT exported
their location then never changes where logs or artifacts land.

Import safety: this module imports nothing from ``core`` (only ``os`` and
``pathlib``), so it can be loaded by common (common → schemas → … →
attribute_conflicts → critical_attributes) AND by critical_attributes
itself, during the very same import cycle, without ordering hazards.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV_VAR = "EUROMONITOR_PROJECT_ROOT"
TRAIN_ROOT_ENV_VAR = "EUROMONITOR_TRAIN_ROOT"
_ROOT_MARKERS = ("config", "pyproject.toml")


def _has_root_markers(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in _ROOT_MARKERS)


def _root_from_env(var: str) -> Path | None:
    """The validated root named by ``var``, or None when unset.

    A set-but-invalid override fails loudly; it never falls through to the
    file-derived default (a stale path must not silently run elsewhere).
    """
    override = os.environ.get(var)
    if not override:
        return None
    root = Path(override).expanduser().resolve()
    if _has_root_markers(root):
        return root
    raise RuntimeError(f"{var} must contain config/ and pyproject.toml: {root}")


def find_project_root(source_file: Path) -> Path:
    """Locate the project from stable markers, never a magic parent offset.

    ``EUROMONITOR_TRAIN_ROOT`` (canonical) wins when set, then
    ``EUROMONITOR_PROJECT_ROOT``; otherwise the walk starts from
    ``source_file``'s parents. Any set override without the root markers
    crashes here rather than silently falling back.
    """
    for var in (TRAIN_ROOT_ENV_VAR, ROOT_ENV_VAR):
        root = _root_from_env(var)
        if root is not None:
            return root
    for candidate in Path(source_file).resolve().parents:
        if _has_root_markers(candidate):
            return candidate
    raise RuntimeError(f"Could not locate project root from {source_file}")
