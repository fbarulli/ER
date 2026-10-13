"""Pin the lane live-log open-mode contract: one ".log" per run, opened once
with a truncating creator and appended to by every other writer.

The contract has two halves, both pinned here:

Behaviour (``cli.colab.start_live_log``, the canonical lane opener):

1. a second run's transcript begins with the capture header and carries none of
   the previous run's content (truncate-once per run, not mid-run);
2. an independent ``open(path, "a")`` co-writer (the detached self-watch child
   that simulates it) appends without clobbering the tee or the header;
3. interleaved append-only writers never punch NUL holes into the transcript
   (the fda0830 tee-hole regression class: O_APPEND writers can only append).

Source contract (every shell wrapper): the second writer on a lane transcript —
the wrapper's ``tee`` — must append (``tee -a``), never a bare ``tee`` (O_TRUNC
with a stale positional offset) and never a truncating ``>`` onto a ``.log``.
``scripts/run_colab_bundle.sh`` runs ``tee -a`` beside the Python opener; this
scan fails loud if that regresses anywhere under ``src/`` or ``scripts/``.

The lane log root is re-pointed with the canonical mechanism used by
test_log_capture.py (monkeypatched module ``TRAIN_ROOT``), never a hand-typed
path; the transcript path itself comes from the public ``lane_transcript_path()``.
"""
from __future__ import annotations

import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from cli import colab

HEADER_MARK = "[log] capturing Colab output"

REPO_ROOT = Path(__file__).resolve().parents[1]

# A "tee" invocation that opens its target O_TRUNC (no -a); paired with a log
# operand it is the fda0830 bug class.
_TEE = re.compile(r"\btee\b(?P<args>[^|;&]*)")
# A truncating redirection: ">" not followed by ">"/"&" and not part of "N>".
_REDIRECT = re.compile(r"""(?<![>0-9&])>(?!>|&)\s*(?P<target>"[^"]*"|'[^']*'|\S+)""")
# A log destination: a ".log" suffix or a "$...LOG..." shell variable.
_LOG_TARGET = re.compile(r"(?i)(\.log\b|\$[a-z_]*log[a-z_]*)")


def _shell_code_lines() -> Iterator[tuple[Path, int, str, str]]:
    """Every code (comment-stripped) line of a shell script under src/scripts."""
    scripts = [*sorted((REPO_ROOT / "src").rglob("*.sh")),
               *sorted((REPO_ROOT / "scripts").rglob("*.sh"))]
    assert scripts, "shell-contract scan found no scripts to check"
    for path in scripts:
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = raw.split("#", 1)[0]
            if code.strip():
                yield path, number, raw, code


def assert_no_truncating_log_writer() -> None:
    """Fail loud on a bare ``tee``/``>`` aimed at a log beside an appender.

    ONE sanctioned exception (owner ruling 2026-10-13, one log per lane run):
    a chain script may truncate its transcript exactly once when it exports
    ``EUROMONITOR_LANE_APPEND`` — the proof its launcher appends instead of
    wiping (pair pinned by test_chain_truncation_and_launcher_append_are_one_writer_pair).
    """
    offenders = []
    paired: set[str] = set()
    for path, _number, _raw, code in _shell_code_lines():
        if "export EUROMONITOR_LANE_APPEND" in Path(path).read_text(encoding="utf-8"):
            paired.add(str(path))
    for path, number, raw, code in _shell_code_lines():
        for match in _TEE.finditer(code):
            args = match.group("args")
            if re.search(r"(?<!\S)-a\b", args):
                continue
            if _LOG_TARGET.search(args):
                offenders.append(f"{path}:{number}: bare tee (needs -a): {raw.strip()}")
        for match in _REDIRECT.finditer(code):
            target = match.group("target").strip("\"'")
            if str(path) in paired or path in paired:
                continue
            if _LOG_TARGET.search(target):
                offenders.append(f"{path}:{number}: truncating '> {target}': {raw.strip()}")
    assert not offenders, (
        "shell log writers must append (tee -a / >>); a second truncating opener "
        "on a live log punches NUL holes:\n" + "\n".join(offenders)
    )


def test_shell_log_writers_are_append_only_contract():
    # Positive control: the canonical wrapper's append tee is present, so a
    # scan that silently stopped matching could not pass.
    bundle = (REPO_ROOT / "scripts" / "run_colab_bundle.sh").read_text(encoding="utf-8")
    assert re.search(r"tee\s+-a\b", bundle), "run_colab_bundle.sh lost its tee -a opener"
    assert_no_truncating_log_writer()


@pytest.fixture
def lane_root(tmp_path, monkeypatch) -> Iterator[types.ModuleType]:
    """Re-point the lane roof to tmp_path and guarantee stream restoration."""
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)
    try:
        yield colab
    finally:
        colab.close_live_log()


def _run(lane_root: types.ModuleType, line: str) -> Path:
    """One lane run: open the transcript, write one tee line, close it."""
    lane_root.start_live_log()
    path = lane_root.lane_transcript_path()
    print(line, flush=True)
    lane_root.close_live_log()
    return path


def test_second_run_truncates_once_and_begins_with_header(lane_root):
    path = _run(lane_root, "RUN-1-UNIQUE-LINE")
    assert "RUN-1-UNIQUE-LINE" in path.read_text(encoding="utf-8")

    path = _run(lane_root, "RUN-2-UNIQUE-LINE")
    second = path.read_text(encoding="utf-8")
    lines = second.splitlines()
    assert lines[0].startswith("[") and HEADER_MARK in lines[0], \
        "the run's transcript begins with the capture header"
    assert "RUN-2-UNIQUE-LINE" in second
    assert "RUN-1-UNIQUE-LINE" not in second, "exactly one truncation per run"


def test_stdout_stderr_tee_and_append_child_coexist(lane_root):
    lane_root.start_live_log()
    path = lane_root.lane_transcript_path()
    print("TEE-STDOUT-LINE", flush=True)
    print("TEE-STDERR-LINE", file=sys.stderr, flush=True)
    with path.open("a", encoding="utf-8") as child:
        child.write("CHILD-APPEND-LINE\n")
    lane_root.close_live_log()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert HEADER_MARK in lines[0], "no mid-run truncation clobbered the header"
    positions = {mark: i for i, line in enumerate(lines)
                 for mark in ("TEE-STDOUT-LINE", "TEE-STDERR-LINE", "CHILD-APPEND-LINE")
                 if mark in line}
    assert set(positions) == {"TEE-STDOUT-LINE", "TEE-STDERR-LINE", "CHILD-APPEND-LINE"}
    assert all(i > 0 for i in positions.values()), "nothing landed before the header"


def test_interleaved_append_writers_leave_no_nul_holes(lane_root):
    lane_root.start_live_log()
    path = lane_root.lane_transcript_path()
    for i in range(5):
        print(f"TEE-{i}", flush=True)
        with path.open("a", encoding="utf-8") as child:
            child.write(f"CHILD-{i}\n")
    lane_root.close_live_log()

    data = path.read_bytes()
    assert b"\x00" not in data, "append-only co-writers cannot punch NUL holes"
    text = data.decode("utf-8")
    assert text.count("TEE-") == 5 and text.count("CHILD-") == 5


def test_chain_truncation_and_launcher_append_are_one_writer_pair(monkeypatch, tmp_path):
    """One log per run: the chain owns the single truncation; the launcher,
    when EUROMONITOR_LANE_APPEND is set, joins it instead of wiping it."""
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)
    monkeypatch.setenv("EUROMONITOR_LANE_APPEND", "1")
    try:
        path = colab.lane_transcript_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("chain banner\\n", encoding="utf-8")
        colab.start_live_log()
        print("launcher line", flush=True)
        colab.close_live_log()
        body = path.read_text(encoding="utf-8")
    finally:
        monkeypatch.delenv("EUROMONITOR_LANE_APPEND", raising=False)
    assert body.startswith("chain banner")
    assert "launcher line" in body
