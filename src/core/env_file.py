"""The repository ``.env``: the ONE reader for locally stored secrets.

``training/dvc_store`` states the contract for the DVC credential — "``.env``
owns ``DVC_API_KEY``" and "``.env`` stays authoritative: applying it only at init
would silently ignore a rotated token" — while the Colab/Kaggle runtimes read the
same file to inject a key into a REMOTE process. Both directions therefore read
one file through one reader: no caller re-spells the search order, the parsing or
the quoting rules, and a value never has to be exported by hand for a lane to
find it.

Values are read, never printed: a caller decides whether a secret may travel.
"""
from __future__ import annotations

import os
from pathlib import Path


class EnvFile:
    """Read ``KEY=VALUE`` entries from the repo's ``.env`` or its parent's."""

    @staticmethod
    def search_paths(root: Path | None = None) -> tuple[Path, Path]:
        """The two accepted homes, in precedence order.

        ``root`` defaults to the project root; a caller whose lookup must
        survive a re-pointed working tree (staging knobs, alternate checkouts)
        passes the root it means.
        """
        from core.common import TRAIN_ROOT

        resolved = Path(TRAIN_ROOT) if root is None else Path(root)
        return (resolved / ".env", resolved.parent / ".env")

    @classmethod
    def value(cls, name: str, root: Path | None = None) -> str | None:
        """The entry for ``name``: the file wins, then the ambient environment."""
        for path in cls.search_paths(root):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, raw = line.partition("=")
                if separator and key.strip() == name:
                    value = raw.strip().strip('"').strip("'")
                    if value:
                        return value
        return os.environ.get(name) or None
