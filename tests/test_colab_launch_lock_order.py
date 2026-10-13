"""One launcher per session, from the first byte.

Public behaviors pinned (the crash class of 2026-10-12: two same-session
launches opened the transcript twice and duplicated the package work before
either met the lock):

1. a second same-session launcher is refused BEFORE the truncating transcript
   open and before any suite packaging work happens (the launch lock moved
   ahead of both);
2. a stale lock left by a dead holder is reclaimed — the recorded pid is
   inert metadata (the kernel released the flock when that process exited) —
   and a LIVE holder is refused by the same call.

The transport is fully mocked (no VM); the state dir is monkeypatched to a
tmp path so the live colab_cli_state is never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest import mock

import pytest

from cli import colab, colab_launch


def test_second_launcher_refuses_before_transcript_or_package_work(tmp_path):
    transcript_opened = mock.Mock()
    packaged = mock.Mock()
    argv = ["colab.py", "--what", "tracks", "--tracks-config", "config/model_tracks.yaml",
            "--gpu", "T4", "--allow-gpu"]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(colab, "acquire_colab_launch_lock",
                           side_effect=RuntimeError("session lock held")), \
         mock.patch.object(colab, "start_live_log", transcript_opened), \
         mock.patch("model_tracks.package.package", packaged), \
         pytest.raises(RuntimeError, match="session lock held"):
        colab.main()
    transcript_opened.assert_not_called()
    packaged.assert_not_called()


def test_stale_lock_is_reclaimed_and_a_live_holder_is_refused(tmp_path):
    short = subprocess.Popen([sys.executable, "-c", "pass"])
    dead = short.pid
    short.wait()
    state = tmp_path / "colab_cli_state"
    state.mkdir()
    lock_path = state / colab._COLAB.launcher_lock_name(colab.SESSION, colab.GPU)
    lock_path.write_text(json.dumps({"pid": dead, "session": colab.SESSION}),
                         encoding="utf-8")
    assert not colab_launch._colab_launch_lock_is_held(lock_path)
    with mock.patch.object(colab, "_COLAB_CLI_STATE_DIR", state):
        handle = colab.acquire_colab_launch_lock()
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        with pytest.raises(RuntimeError, match="already owns"):
            colab.acquire_colab_launch_lock()
        colab.release_colab_launch_lock(handle)
    assert not colab_launch._colab_launch_lock_is_held(lock_path)
