"""Project-root discovery — the leaf SSOT, imported by every config reader.

Why a leaf: three modules used to re-implement the same marker walk
(``config/`` + ``pyproject.toml``), and they had drifted — the root-override
env var ``EUROMONITOR_PROJECT_ROOT`` was honored by core.common but NOT by
core.critical_attributes' vocabulary reader, so an env-rooted test/CI run
could load two different vocabularies. One class, imported everywhere, ends
that.

Import safety: this module imports nothing from ``core`` (only ``os``,
``subprocess`` and ``pathlib``), so it can be loaded by common (common →
schemas → … → attribute_conflicts → critical_attributes) AND by
critical_attributes itself, during the very same import cycle, without
ordering hazards.
"""

from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

_log = logging.getLogger(__name__)


class ProjectRoot:
    """Locate the project root, and the canonical checkout behind a worktree.

    ``find`` is the one marker walk every config reader shares, so the
    vocabulary reader and ``core.common`` can never disagree. ``canonical``
    adds the single step a linked worktree needs: the marker-derived root is the
    worktree, but the shared data/log root and the ``.env`` beside it belong to
    the canonical checkout, which git names identically from everywhere.
    """

    ENV_VAR = "EUROMONITOR_PROJECT_ROOT"
    _MARKERS = ("config", "pyproject.toml")

    @classmethod
    def _has_markers(cls, candidate: Path) -> bool:
        return all((candidate / marker).exists() for marker in cls._MARKERS)

    @classmethod
    def find(cls, source_file: Path) -> Path:
        """Locate the project from stable markers, never a magic parent offset.

        The ``EUROMONITOR_PROJECT_ROOT`` env var, when set, must name a directory
        that carries the root markers (config/ + pyproject.toml); anything else
        fails loudly instead of silently falling back to the file-derived root.
        With no override the walk starts from ``source_file``'s parents.
        """
        override = os.environ.get(cls.ENV_VAR)
        if override:
            root = Path(override).expanduser().resolve()
            if cls._has_markers(root):
                return root
            raise RuntimeError(
                f"{cls.ENV_VAR} must contain config/ and pyproject.toml: {root}")
        resolved = Path(source_file).resolve()
        for candidate in (resolved, *resolved.parents):
            if cls._has_markers(candidate):
                return candidate
        raise RuntimeError(f"Could not locate project root from {source_file}")

    @classmethod
    @lru_cache(maxsize=None)
    def canonical(cls, source_file: Path) -> Path:
        """The CANONICAL checkout root, shared by every linked worktree.

        A training worktree lives under ``<canonical>/.worktrees/<name>``, so its
        marker-derived root is the worktree itself — but the shared data/log root
        and the ``.env`` beside it belong to the canonical checkout. ``git
        rev-parse --git-common-dir`` names that shared ``.git`` (the same answer
        from the main checkout and any worktree), so this is the ONE home for the
        worktree step: ``EnvFile`` and the credential store call it, the launcher
        asks it — none re-derives it. A non-git tree falls back to the
        marker-derived root. Cached per process — the root cannot move mid-run.
        """
        try:
            marker = cls.find(source_file)
        except RuntimeError:
            # Not a marker-bearing checkout (an ad-hoc/test root): there is no
            # canonical checkout to step out of, so answer with the path itself.
            marker = Path(source_file).resolve()
        try:
            result = subprocess.run(
                ["git", "-C", str(marker), "rev-parse", "--path-format=absolute",
                 "--git-common-dir"],
                capture_output=True, text=True, check=True)
        except Exception:
            # git absent, the root is not a checkout, or a caller has disabled
            # subprocess (tests): there is no shared checkout to step out of, so
            # the marker root stands — recorded, never a silent swallow.
            _log.debug("canonical root: git common dir unavailable at %s", marker,
                       exc_info=True)
            return marker
        common = result.stdout.strip()
        return Path(common).parent if common else marker


# Back-compat entry point for the ~50 existing callers (``scripts/``,
# ``dashboard/``, ``jev/``) that import ``find_project_root``. The behaviour
# lives in :meth:`ProjectRoot.find`; this is a name binding, not a second
# implementation. New code should call ``ProjectRoot.find`` directly.
find_project_root = ProjectRoot.find
