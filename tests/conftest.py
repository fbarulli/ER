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

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = (_ROOT / "src").resolve()

#: Flags that mark a Popen argv as an ER detached lane watcher. Every watcher
#: re-invokes its lane entry point with one of these; managed local workers
#: (``model_tracks.parallel``) never carry them, so they pass through untouched.
_WATCHER_ENTRY_FLAGS = frozenset({"autowatch", "self-watch", "keep-alive"})


def _spawns_watcher(command: object) -> bool:
    """True when a Popen argv re-invokes an ER detached watcher entry point."""
    if isinstance(command, bytes):
        parts: list = command.decode(errors="replace").split()
    elif isinstance(command, str):
        parts = command.split()
    else:
        try:
            parts = list(command)
        except TypeError:
            return False
    return any(str(part) in _WATCHER_ENTRY_FLAGS for part in parts)


@pytest.fixture(scope="session", autouse=True)
def _forbid_detached_watcher():
    """Fail any test that spawns a REAL detached lane watcher (the 429-storm leak).

    The watcher spawn is a process boundary; a test that reaches it without a
    fake leaks a live ``--what autowatch`` / ``--what self-watch`` / ``keep-alive``
    process which polls the platform, self-inflicts 429s, and outlives the suite.
    The guard wraps the one construction point every spawn crosses
    (``subprocess.Popen.__init__``) plus ``os.setsid``, so it catches kaggle,
    colab self-watch, and keep-alive seams alike. A test that exercises a spawn
    seam patches that seam (or ``subprocess.Popen``) itself, which overrides this
    guard; managed local workers carry no watcher flag and pass through.
    """
    import subprocess

    real_init = subprocess.Popen.__init__
    real_setsid = os.setsid

    def _guarded_init(self, command, *args, **kwargs):
        if _spawns_watcher(command):
            raise AssertionError(
                "test attempted to spawn a real detached lane watcher "
                f"({command!r}); patch the spawn seam (KernelWatcher.spawn / "
                "spawn_self_watch / _spawn_keep_alive) or subprocess.Popen")
        return real_init(self, command, *args, **kwargs)

    def _forbidden_setsid():
        raise AssertionError(
            "test attempted os.setsid(); detached children are forbidden in tests")

    subprocess.Popen.__init__ = _guarded_init
    os.setsid = _forbidden_setsid
    try:
        yield
    finally:
        subprocess.Popen.__init__ = real_init
        os.setsid = real_setsid


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
