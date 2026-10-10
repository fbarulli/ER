"""Project-root discovery — the leaf SSOT, imported by every config reader.

Why a leaf: modules used to re-implement the same marker walk
(``config/`` + ``pyproject.toml``), and they had drifted — the root-override
env var ``EUROMONITOR_PROJECT_ROOT`` was honored by core.common but NOT by
core.critical_attributes' vocabulary reader, so an env-rooted test/CI run
could load two different vocabularies. One class, imported everywhere, ends
that.

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
``<canonical>/.worktrees/<name>`` (never ``/tmp``); :meth:`ProjectRoot.canonical`
resolves the shared root without needing ``EUROMONITOR_TRAIN_ROOT`` exported.

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

    ``find`` is the one marker walk every config reader shares (one class, so
    the vocabulary reader and ``core.common`` cannot disagree). ``canonical``
    adds the single step a linked worktree needs: the marker-derived root is the
    worktree, but the shared data/log root and the ``.env`` beside it belong to
    the canonical checkout, which git names identically from everywhere.
    """

    ROOT_ENV_VAR = "EUROMONITOR_PROJECT_ROOT"
    TRAIN_ROOT_ENV_VAR = "EUROMONITOR_TRAIN_ROOT"
    _MARKERS = ("config", "pyproject.toml")

    @classmethod
    def _has_markers(cls, candidate: Path) -> bool:
        return all((candidate / marker).exists() for marker in cls._MARKERS)

    @classmethod
    def _from_env(cls, var: str) -> Path | None:
        """The validated root named by ``var``, or None when unset.

        A set-but-invalid override fails loudly; it never falls through to the
        file-derived default (a stale path must not silently run elsewhere).
        """
        override = os.environ.get(var)
        if not override:
            return None
        root = Path(override).expanduser().resolve()
        if cls._has_markers(root):
            return root
        raise RuntimeError(
            f"{var} must contain config/ and pyproject.toml: {root}")

    @classmethod
    def find(cls, source_file: Path) -> Path:
        """Locate the project from stable markers, never a magic parent offset.

        ``EUROMONITOR_TRAIN_ROOT`` (canonical) wins when set, then
        ``EUROMONITOR_PROJECT_ROOT``; otherwise the walk starts from
        ``source_file`` and its parents. Any set override without the root
        markers crashes here rather than silently falling back.
        """
        for var in (cls.TRAIN_ROOT_ENV_VAR, cls.ROOT_ENV_VAR):
            root = cls._from_env(var)
            if root is not None:
                return root
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
        worktree step: ``EnvFile``, ``CredentialStore`` and the launcher call it,
        none re-derives it. An explicit root override or a non-git tree falls
        back to the marker-derived root. Cached per process — the root cannot
        move mid-run.
        """
        if os.environ.get(cls.TRAIN_ROOT_ENV_VAR) or os.environ.get(cls.ROOT_ENV_VAR):
            return cls.find(source_file)
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
