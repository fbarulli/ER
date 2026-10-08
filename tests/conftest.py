"""Test bootstrap: force THIS worktree's ``src`` onto the import path.

The shared virtualenv is an editable install of the PRIMARY checkout
(``/home/opc/ONE/ER/.venv`` whose ``.pth`` adds ``/home/opc/ONE/ER/src``).
Running pytest from a git worktree would therefore import ``core`` /
``training`` from the primary tree and silently validate the wrong code.

This conftest prepends the worktree root AND the worktree ``src``, then
asserts the tree-discriminating packages resolve under this worktree's ``src``
— so the silent wrong-tree run can never happen again.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# Worktree root first (legacy layout), then worktree src at the very front so it
# shadows the editable .pth's primary-tree src.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))


def _assert_local_imports() -> None:
    import core

    resolved = Path(core.__file__).resolve()
    if not resolved.is_relative_to(_SRC):
        raise RuntimeError(
            "REFUSING TO RUN: tests imported 'core' from "
            f"{resolved}, not this worktree's {_SRC}. The shared venv's "
            "editable .pth points at the primary checkout; export "
            "PYTHONPATH=src (or rely on this conftest) so tests exercise the "
            "worktree."
        )


_assert_local_imports()
