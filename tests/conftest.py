"""Test bootstrap: force imports to resolve to THIS checkout's ``src``.

The shared ``.venv`` may carry an editable ``.pth`` pointing at a SIBLING
checkout (e.g. the primary tree), so a worktree run would otherwise silently
test the wrong code. We prepend this checkout's ``src`` (and repo root) and
then ASSERT that ``core`` / ``cli`` / ``training`` resolved here — a wrong-tree
run aborts collection instead of passing against stale code.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# ``src`` FIRST so it beats the shared editable install on every import.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))


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
