"""tests/test_colab_hub.py — ONE home for the running ``cli.colab`` identity.

Defect this pins (consolidation audit 2026-10-08, finding 1). Nine bodies, three
names and two semantics resolved "the module that is currently running colab":

    * ``_hub`` ×6 — ``colab_transport``, ``colab_bundle_prewarm``,
      ``colab_launch``, ``colab_validation_upload``, ``colab_runtime`` (a
      different body: ``... or sys.modules["cli.colab"]``) and
      ``colab_result_sync`` (the same body, now consolidated too);
    * ``_colab_hub`` in ``colab_lane_contracts`` and ``_colab`` in
      ``colab_self_watch`` / ``colab_retention`` — the BARE
      ``sys.modules["__colab_runtime_self__"]``, which raises ``KeyError``
      instead of importing the launcher;
    * ``_timed_colab`` ×5 — the same lazy step-timing shim, re-spelled per
      phase.

One concept, nine declarations: a phase that resolved the wrong copy would read
a stale SESSION/state dir, and the fakes that patch ``cli.colab`` would stop
steering it. ``cli/colab_hub.py`` is now the single home (``hub()`` +
``timed_colab``), semantically get-or-import: the registered running identity
always wins, and only a not-yet-imported launcher triggers the import that
registers it.

These tests pin the home, not the classes: identity/semantics of ``hub()``, the
lazy timing shim resolving the LIVE decorator, and the structural rule that no
phase re-spells the lookup.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cli.colab as colab
from cli import colab_hub, colab_launch

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "src" / "cli"

# The launcher itself registers the identity and the home resolves it. Every
# phase, including ``colab_result_sync.py`` (whose inline ``_hub`` was the last
# duplicate), must resolve through the home.
REGISTRATION_HOME = "colab.py"
RESOLUTION_HOME = "colab_hub.py"


def _names_running_identity(node: ast.AST) -> bool:
    """Whether an AST node is ``sys.modules["__colab_runtime_self__"]`` / .get(...)."""
    key = '"__colab_runtime_self__"'
    if isinstance(node, ast.Subscript):
        value, slc = node.value, node.slice
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        value, slc = node.func.value, (node.args[0] if node.args else None)
    else:
        return False
    target = value
    if isinstance(target, ast.Attribute) and target.attr == "modules":
        target = target.value
    if not (isinstance(target, ast.Name) and target.id == "sys"):
        return False
    return isinstance(slc, ast.Constant) and ast.unparse(slc) == key


def _resolver_spellings(path: Path) -> list[str]:
    """The resolver definitions/lookups a phase spells for itself."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [
        f"def {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"_hub", "_colab", "_colab_hub"}
    ]
    found.extend(
        f"line {node.lineno}: sys.modules[...__colab_runtime_self__]"
        for node in ast.walk(tree)
        if _names_running_identity(node)
    )
    return found


def test_hub_is_the_running_launcher_identity(monkeypatch):
    """The home returns the registered running identity, never a second copy."""
    assert colab_hub.hub() is colab
    assert sys.modules["__colab_runtime_self__"] is colab
    sentinel = object()
    monkeypatch.setitem(sys.modules, "__colab_runtime_self__", sentinel)
    assert colab_hub.hub() is sentinel


def test_hub_imports_and_registers_the_launcher_when_none_is_running():
    """A phase imported on its own gets the real launcher, registered by import.

    Run out of process: importing ``cli.colab_runtime`` first must leave
    ``__colab_runtime_self__`` unset, and ``hub()`` must then import
    ``cli.colab`` (which registers itself) rather than fail with KeyError.
    """
    script = (
        "import sys\n"
        "assert '__colab_runtime_self__' not in sys.modules\n"
        "from cli.colab_hub import hub\n"
        "surface = hub()\n"
        "assert surface is sys.modules['cli.colab'], surface\n"
        "assert sys.modules['__colab_runtime_self__'] is surface\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import cli.colab_runtime\n" + script],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_timed_colab_resolves_the_live_decorator_at_call_time(monkeypatch):
    """The shim asks the running launcher for ``_timed_colab`` on every call.

    That is what lets a phase define its steps while ``cli.colab`` still owns
    the timing surface: the decorator is never captured at decoration time.
    """
    seen: list[tuple[str, str]] = []

    def fake_timed(kind):
        def decorate(function):
            def wrapped(*args, **kwargs):
                seen.append((kind, function.__name__))
                return function(*args, **kwargs)
            return wrapped
        return decorate

    monkeypatch.setattr(colab, "_timed_colab", fake_timed)

    @colab_hub.timed_colab("step")
    def phase_step(value):
        return value + 1

    assert phase_step(1) == 2
    assert phase_step(2) == 3
    assert phase_step.__name__ == "phase_step"
    assert seen == [("step", "phase_step"), ("step", "phase_step")]
    monkeypatch.undo()
    assert colab._timed_colab.__module__ == "cli.colab"


def test_a_phase_reads_the_live_surface_through_the_shared_hub(monkeypatch):
    """Patching ``cli.colab`` still steers a split phase (no captured copy)."""
    monkeypatch.setattr(colab, "SESSION", "hub-home-vm")
    monkeypatch.setattr(colab, "_COLAB_CLI_STATE_DIR", Path("/tmp/hub-home-state"))
    assert colab_launch._colab_launch_lock_path() == Path(
        "/tmp/hub-home-state/launcher-hub-home-vm.lock")


def test_no_phase_module_re_spells_the_running_identity():
    """The structural half: the nine bodies must not come back."""
    offenders = {}
    for path in sorted(CLI.glob("colab*.py")):
        if path.name in {REGISTRATION_HOME, RESOLUTION_HOME}:
            continue
        spellings = _resolver_spellings(path)
        if spellings:
            offenders[path.name] = spellings
    assert offenders == {}, (
        "colab phases must resolve the running identity through cli.colab_hub; "
        f"local resolvers found: {offenders}")


def test_the_phases_import_the_one_home():
    """Every phase that reads the launcher surface imports the shared home."""
    importers = {
        "colab_runtime.py", "colab_transport.py", "colab_launch.py",
        "colab_bundle_prewarm.py", "colab_validation_upload.py",
        "colab_self_watch.py", "colab_retention.py", "colab_lane_contracts.py",
        "colab_result_sync.py",
    }
    missing = sorted(
        name for name in importers
        if not re.search(r"^from cli\.colab_hub import ", (CLI / name).read_text(encoding="utf-8"),
                         flags=re.MULTILINE))
    assert missing == []


def test_the_home_is_a_leaf_module():
    """``cli.colab_hub`` must stay importable in any order: no launcher import."""
    tree = ast.parse((CLI / RESOLUTION_HOME).read_text(encoding="utf-8"))
    module_level = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = {ast.unparse(node) for node in module_level}
    assert imported == {
        "from __future__ import annotations",
        "import functools",
        "import sys",
        "from typing import Any",
    }


@pytest.mark.parametrize("name", ["_hub", "_colab", "_colab_hub"])
def test_the_legacy_resolver_names_are_gone(name):
    """No phase re-declares the old local resolvers under any of the three names."""
    for path in sorted(CLI.glob("colab*.py")):
        if path.name in {REGISTRATION_HOME, RESOLUTION_HOME}:
            continue
        assert f"def {name}(" not in path.read_text(encoding="utf-8"), path
