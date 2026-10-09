"""workspace.py — the ONE root disposable worktrees and lane scratch live under.

Owner mandate 2026-10-09: a worktree or lane scratch directory is created under
the configured, project-local root (``config/paths.yaml`` ``paths.worktrees_dir``,
default ``.worktrees``), NEVER under ``/tmp`` — and it is disposable: each is
promoted into a proper code/artifact change or deleted, never left behind.

The root is declared exactly once (the config SSOT, validated by
``DataPathsSpec`` against a ``/tmp`` root at load) and resolved here, so no
caller hand-types a path. A caller asks for a named slot, and a slot name is a
bare name, so a slot can never escape the root. Construction is a factory
(``from_config``); the resolved root is injectable for a caller that already
holds it.
"""
from __future__ import annotations

from pathlib import Path


class Workspace:
    """The configured root for disposable worktrees and lane scratch."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def from_config(cls) -> Workspace:
        """Build from the config SSOT (``paths.worktrees_dir``)."""
        from core.common import TRAIN_ROOT, data_cfg

        return cls((TRAIN_ROOT / data_cfg().paths.worktrees_dir).resolve())

    def slot(self, name: str) -> Path:
        """Resolve one named worktree/scratch slot under the configured root.

        ``name`` is a bare directory name, never a path: a caller that passes a
        separator or ``..`` is refused rather than re-anchored, so a slot can
        never be a second way to spell a location outside ``root``.
        """
        if not name or name in {".", ".."} or name != Path(name).name:
            raise ValueError(f"workspace slot must be a bare name: {name!r}")
        return (self.root / name).resolve()
