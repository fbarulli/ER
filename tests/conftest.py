"""Test bootstrap: pin imports to THIS worktree, never the shared editable tree.

Worktrees share one venv whose editable ``.pth`` adds the PRIMARY checkout's
``src`` (``/home/opc/ONE/ER/src``). If pytest resolved ``core`` / ``training`` /
``cli`` there, the worktree's new modules would be invisible and a green run
would be meaningless (a "the files aren't even loaded" class of false pass).

So: prepend the worktree ``src`` + repo root, DROP any foreign ``.../src`` entry
so it can never shadow us, then ASSERT the import root is this worktree and
fail loud if it is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = (REPO_ROOT / "src").resolve()

# 1) This worktree wins.
for entry in (str(REPO_ROOT), str(SRC)):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC))

# 2) Remove any OTHER checkout's src (the shared editable install) so it can
#    never shadow the worktree even if something re-inserts it later.
for entry in list(sys.path):
    if entry.endswith("/src"):
        try:
            resolved = Path(entry).resolve()
        except OSError:  # pragma: no cover - non-existent path
            continue
        if resolved != SRC and resolved not in REPO_ROOT.parents:
            sys.path.remove(entry)

# 3) Fail loud if the import root is not the worktree.
import core

_resolved_core = Path(core.__file__).resolve()
if not _resolved_core.is_relative_to(REPO_ROOT):
    raise RuntimeError(
        "test bootstrap resolved `core` outside this worktree: "
        f"{_resolved_core} (expected under {REPO_ROOT}); refusing to run tests "
        "against the primary/shared tree")
