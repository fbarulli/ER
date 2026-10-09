"""Test bootstrap: force imports to resolve to THIS checkout's ``src``.

The shared ``.venv`` may carry an editable ``.pth`` pointing at a SIBLING
checkout (e.g. the primary tree), so a worktree run would otherwise silently
test the wrong code. We prepend this checkout's ``src`` (and repo root) and
then ASSERT that ``core`` / ``cli`` / ``training`` resolved here — a wrong-tree
run aborts collection instead of passing against stale code.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# ``src`` FIRST so it beats the shared editable install on every import.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _forbid_detached_watcher(monkeypatch):
    """Fail a test that spawns a REAL detached watcher (the 429-storm leak).

    The watcher spawn is a process boundary; a test that reaches it without a
    fake leaks a live ``--what autowatch`` process which polls Kaggle, self-
    inflicts 429s, and outlives the suite. A test that exercises the spawn seam
    must monkeypatch ``KernelWatcher.spawn`` (or the lane's ``_spawn_autowatch``
    seam) itself, which overrides this guard.
    """
    from cli import kaggle_watcher

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "test attempted to spawn a real detached watcher; patch "
            "KernelWatcher.spawn or the lane _spawn_autowatch seam")

    monkeypatch.setattr(kaggle_watcher.KernelWatcher, "spawn", _forbidden)


def _assert_worktree_modules():
    expected = str(_SRC.resolve())
    import core  # noqa: F401  (imported for the path assertion below)
    import cli  # noqa: F401
    import training  # noqa: F401

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


_assert_worktree_modules()
