"""The repository ``.env``: the ONE reader for locally stored secrets.

``training/dvc_store`` states the contract for the DVC credential — "``.env``
owns ``DVC_API_KEY``" and "``.env`` stays authoritative: applying it only at init
would silently ignore a rotated token" — while the Colab/Kaggle runtimes read the
same file to inject a key into a REMOTE process. Both directions therefore read
one file through one reader: no caller re-spells the search order, the parsing or
the quoting rules, and a value never has to be exported by hand for a lane to
find it.

The file lives beside the CANONICAL checkout (one level above it), not beside
whichever linked worktree launched the run, so its location is resolved through
git's common dir — the same answer from the main checkout and every worktree —
never spelled as a path. ``path``/``apply`` are the ONE way a caller locates or
loads it; the uv launcher (``scripts/er.sh``) feeds the same path to
``uv run --env-file``.

Values are read, never printed: a caller decides whether a secret may travel.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path


class EnvFile:
    """Read ``KEY=VALUE`` entries from the repo's ``.env`` (or one level above)."""

    @staticmethod
    @lru_cache(maxsize=None)
    def _canonical_root(root: Path) -> Path:
        """The canonical checkout root shared by every linked worktree.

        ``git rev-parse --git-common-dir`` names the shared ``.git`` (identical
        from the main checkout and every worktree), so its parent is the
        canonical checkout from anywhere. A non-git tree falls back to ``root``.
        Cached per process: the checkout root cannot change under a run.
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--path-format=absolute",
                 "--git-common-dir"],
                capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return root
        common = result.stdout.strip()
        return Path(common).parent if common else root

    @classmethod
    def search_paths(cls, root: Path | None = None) -> tuple[Path, ...]:
        """The accepted homes, in precedence order (checkout-local first).

        ``root`` defaults to the project root; a caller whose lookup must
        survive a re-pointed working tree (staging knobs, alternate checkouts)
        passes the root it means. The canonical ``.env`` is the SSOT for a
        worktree that has no local override.
        """
        if root is None:
            from core.common import TRAIN_ROOT

            resolved = Path(TRAIN_ROOT)
        else:
            resolved = Path(root)
        canonical = cls._canonical_root(resolved)
        candidates = [resolved / ".env", canonical.parent / ".env"]
        if canonical != resolved:
            candidates.append(resolved.parent / ".env")
        return tuple(dict.fromkeys(candidates))

    @classmethod
    def path(cls, root: Path | None = None) -> Path | None:
        """The first EXISTING env file (what a loader should use), else None."""
        for candidate in cls.search_paths(root):
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _entries(cls, path: Path) -> Iterator[tuple[str, str]]:
        """The parseable ``KEY=VALUE`` pairs of one env file (unquoted)."""
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            key = key.strip()
            if not separator or not key or key.startswith("#"):
                continue
            value = raw.strip().strip('"').strip("'")
            if value:
                yield key, value

    @classmethod
    def value(cls, name: str, root: Path | None = None) -> str | None:
        """The entry for ``name``: the file wins, then the ambient environment."""
        for candidate in cls.search_paths(root):
            if not candidate.is_file():
                continue
            for key, value in cls._entries(candidate):
                if key == name:
                    return value
        return os.environ.get(name) or None

    @classmethod
    def apply(cls, root: Path | None = None) -> Path | None:
        """Load the env file into ``os.environ``; a caller's setting always wins.

        ONE place turns the file into process environment, so subprocesses (the
        kaggle CLI, wandb) inherit the keys instead of every caller exporting
        them by hand. ``setdefault`` never clobbers an explicit per-run override
        (the file and the loader agree on the ambient environment). Values are
        never printed. Returns the file used, or None when none exists.
        """
        path = cls.path(root)
        if path is None:
            return None
        for key, value in cls._entries(path):
            os.environ.setdefault(key, value)
        return path
