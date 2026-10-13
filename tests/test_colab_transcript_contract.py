"""The lane transcript contract: the markers operators and scrapers grep.

The Colab transcript is a grep surface, not just a log file. These markers are
pinned by what operators read (README-lanes.md) and by what scripts parse:

  [timing]          the canonical timing emitter (core.timing, reached through
                    RunLogger.section and the @timed decorators)
  [tqdm]            a CR-separated progress frame, re-emitted as ONE tagged
                    line by the capture seam (cli.log_capture)
  [done]            the lane completed and the VM was released
  [failed]          the lane did not complete (the VM is still released)
  [worker-process]  the remote training worker's own process lines

A logging consolidation that drops or re-formats any of them silently breaks
every scraper built on them, so these tests drive the REAL emitters and the
REAL capture seam rather than asserting on source text. tqdm itself is
deliberately out of scope (owner constraint): only its re-emission into the
transcript is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from cli import colab
from cli.log_capture import LAST_BAR_TAG
from core.run_log import RunLogger

#: Every transport/provisioning surface ``main`` touches. Nothing here may be
#: left real: a lane test that dials Colab allocates a VM.
_OFFLINE = ("check_colab_cli", "ensure_session", "prepare_remote_layout",
            "install_deps", "verify_remote_models", "log_gpu_profile",
            "spawn_self_watch", "acquire_colab_launch_lock",
            "release_colab_launch_lock", "verify_training_inputs")


def _lane_main(monkeypatch, *, run_train) -> str:
    """Run one real ``cli.colab.main`` launch (offline) and return its transcript."""
    monkeypatch.setattr(sys, "argv", ["colab.py", "--what", "train", "--gpu", "CPU"])
    for name in _OFFLINE:
        monkeypatch.setattr(colab, name, mock.Mock())
    monkeypatch.setattr(colab, "run_train", run_train)
    monkeypatch.setattr(colab, "stop", mock.Mock(return_value=True))
    saved_gpu = colab.GPU
    try:
        colab.main()
    finally:
        colab.GPU = saved_gpu
    return colab.lane_transcript_path().read_text(encoding="utf-8")


def test_a_completed_lane_transcript_carries_timing_tqdm_and_done(monkeypatch, tmp_path):
    """One real launch: its transcript keeps every pinned marker, once each."""
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)

    def run_train(*_args, **_kwargs):
        logger = RunLogger("probe.transcript")
        with logger.section("transcript_probe"):
            logger.info("PROBE-INFO")
            logger.warning("PROBE-WARN")
            logger.error("PROBE-ERROR")
            # The real tqdm frame, as core.progress emits it: tqdm writes the
            # whole strip INCLUDING the CR in one write, which is what the
            # capture seam turns into one tagged line.
            for _ in logger.progress(range(4), desc="probe_bar", unit="row"):
                pass
        return ("run", 0)

    transcript = _lane_main(monkeypatch, run_train=run_train)

    assert "[timing]" in transcript and "transcript_probe" in transcript, transcript
    # The last CR frame is re-emitted as ONE tagged, grep-able line.
    assert LAST_BAR_TAG in transcript and "probe_bar" in transcript, transcript
    assert transcript.count("[done]") == 1, transcript
    assert "[failed]" not in transcript, transcript
    # ONE mechanism, ONE line: every canonical level and the timing section
    # reach the transcript exactly once, never once per emission surface.
    assert transcript.count("transcript_probe state=completed") == 1, transcript
    assert transcript.count("PROBE-INFO") == 1, transcript
    assert transcript.count("PROBE-WARN") == 1, transcript
    assert transcript.count("PROBE-ERROR") == 1, transcript
    assert transcript.count("[warning] PROBE-WARN") == 1, transcript
    assert transcript.count("[error] PROBE-ERROR") == 1, transcript


def test_a_failed_lane_says_failed_and_never_done(monkeypatch, tmp_path):
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)

    def run_train(*_args, **_kwargs):
        raise RuntimeError("lane failed on purpose")

    with pytest.raises(RuntimeError, match="lane failed on purpose"):
        _lane_main(monkeypatch, run_train=run_train)

    transcript = colab.lane_transcript_path().read_text(encoding="utf-8")
    assert "[failed]" in transcript and "[done]" not in transcript, transcript


def test_the_tracks_poll_tee_re_emits_cr_frames_as_tagged_lines(monkeypatch):
    """The detached all_tracks poll prints the raw chunk so the ONE system tee
    (start_live_log) expands CR tqdm frames into grep-able lines and appends
    the tagged [tqdm] strip — every frame emitted exactly once."""
    import io
    import json

    import cli.colab_transport as transport

    transcript = io.StringIO()
    monkeypatch.setattr(sys, "stdout", colab._Tee(io.StringIO(), transcript))
    payloads = [
        {"pid": 1, "log": "/remote/all_tracks.log", "status": "/remote/all_tracks.log.status"},
        {"offset": 42,
         "chunk": "[track/text]   2%|\u258f | 164/8150 [01:00<43:13,  3.08it/s]\r"
                  "[track/text]   3%|\u258e | 165/8150 [01:01<42:28,  3.13it/s]\r",
         "done": True, "returncode": "0"},
    ]
    calls = iter(payloads)
    monkeypatch.setattr(colab, "run_colab_exec_capture",
                        lambda session, script, timeout=None, **kw: json.dumps(next(calls)))

    transport.run_detached_stage("all_tracks", ["python3", "-c", "train()"], timeout=60)

    text = transcript.getvalue()
    assert "[track/text]   2%" in text and "[track/text]   3%" in text, text
    assert text.count(LAST_BAR_TAG) == 1, text
    assert f"{LAST_BAR_TAG} [track/text]   3%" in text, text
    assert "[all_tracks] [track/" not in text, text


def test_the_worker_process_markers_stay_in_the_worker_launcher_script():
    """[worker-process] is emitted by the remote worker's generated script."""
    source = Path(colab.__file__).read_text(encoding="utf-8")
    for marker in ("[worker-process] starting pid=",
                   "[worker-process] exited rc=",
                   "[worker-process] resource snapshot before training",
                   "[worker-process] resource snapshot after training"):
        assert marker in source, marker