"""Test bootstrap: force imports to resolve to THIS checkout's ``src``.

The shared ``.venv`` may carry an editable ``.pth`` pointing at a SIBLING
checkout (e.g. the primary tree), so a worktree run would otherwise silently
test the wrong code. We prepend this checkout's ``src`` (and repo root), DROP
any other checkout's ``src`` so it can never shadow us, then ASSERT that
``core`` / ``cli`` / ``training`` resolved here — a wrong-tree run aborts
collection instead of passing against stale code.

The pin is re-applied before EVERY test (autouse fixture): a test that inserts a
temporary ``.../src`` into ``sys.path`` cannot leak it into a later test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = (_ROOT / "src").resolve()


def _pin_import_paths() -> None:
    """Put this worktree's src first and drop any foreign ``.../src``."""
    for entry in (str(_ROOT), str(_SRC)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_SRC))
    for entry in list(sys.path):
        if entry.endswith("/src"):
            try:
                resolved = Path(entry).resolve()
            except OSError:  # pragma: no cover - non-existent path
                continue
            if resolved != _SRC and resolved not in _ROOT.parents:
                sys.path.remove(entry)


def _assert_worktree_modules() -> None:
    expected = str(_SRC)
    import cli
    import core
    import training

    wrong = {}
    for name, module in (("core", core), ("cli", cli), ("training", training)):
        resolved = str(Path(module.__file__).resolve())
        if not resolved.startswith(expected + "/"):
            wrong[name] = resolved
    if wrong:
        raise RuntimeError(
            "tests resolved modules outside this checkout's src "
            f"({expected}): {wrong}; the shared venv editable install is "
            "shadowing the worktree — prepend PYTHONPATH=src or fix the .pth")


_pin_import_paths()
_assert_worktree_modules()


@pytest.fixture(autouse=True)
def _pin_import_paths_per_test():
    """Re-pin before every test so a leaked foreign src cannot shadow us."""
    _pin_import_paths()
    yield
