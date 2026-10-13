"""Pin the lane live-log opener's "w" contract as observable transcript behavior.

``cli.colab.start_live_log`` must truncate the lane transcript exactly once per
run (one ``write_text("")`` at open) and then hold the handle in append mode,
with sys/stdout/stderr tee'd onto that same handle, so that:

1. a second run's transcript begins with the capture header and carries none of
   the previous run's content (truncate-once per run, not mid-run);
2. an independent ``open(path, "a")`` co-writer (the detached self-watch child
   that simulates it) appends without clobbering the tee or the header;
3. interleaved append-only writers never punch NUL holes into the transcript
   (the fda0830 tee-hole regression class: O_APPEND writers can only append).

The lane log root is re-pointed with the canonical mechanism used by
test_log_capture.py (monkeypatched module ``TRAIN_ROOT``), never a hand-typed
path; the transcript path itself comes from the public ``lane_transcript_path()``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterator

import pytest

import cli.colab as colab

HEADER_MARK = "[log] capturing Colab output"


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
