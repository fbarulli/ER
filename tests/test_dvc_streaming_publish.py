"""Streaming DVC publication: push promptly, never block training, batch bursts.

The publisher is driven against a scripted ``dvc`` on PATH, so the tests observe
exactly what the real binary would be asked to do — how many pushes, with which
targets, and how long after a checkpoint became available.  The scripted binary
can also be made slow or made to fail, which is how the "a broken remote cannot
stall training" property is proved rather than asserted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from core import common
from training import dvc_store

FAKE_DVC = """#!/usr/bin/env python3
import json, os, pathlib, sys, time
argv = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_DVC_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": argv, "at": time.monotonic()}) + "\\n")
if argv[:1] == ["init"]:
    (pathlib.Path.cwd() / ".dvc").mkdir(exist_ok=True)
if argv[:1] == ["add"]:
    for target in argv[1:]:
        path = pathlib.Path(target)
        path.with_name(path.name + ".dvc").write_text("outs: []\\n")
if argv[:1] == ["push"]:
    delay = float(os.environ.get("FAKE_DVC_PUSH_SECONDS", "0"))
    if delay:
        time.sleep(delay)
    if os.environ.get("FAKE_DVC_PUSH_FAILS"):
        sys.stderr.write("simulated remote failure\\n")
        sys.exit(1)
sys.exit(0)
"""


class _Workspace:
    """A throwaway worker directory plus a scripted dvc on PATH."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "worker"
        self.source.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "dvc-calls.jsonl"
        dvc = self.bin / "dvc"
        dvc.write_text(FAKE_DVC)
        dvc.chmod(0o755)
        self.env = mock.patch.dict(
            os.environ,
            {
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "FAKE_DVC_LOG": str(self.log),
                "DVC_API_KEY": "test-token",
            },
        )

    def __enter__(self):
        self.env.start()
        return self

    def __exit__(self, *_exc):
        dvc_store.stop_checkpoint_streamer(self.source)
        self.env.stop()
        self._tmp.cleanup()
        return False

    def calls(self) -> list[dict]:
        import json

        if not self.log.is_file():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]

    def pushes(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls() if c["argv"][:1] == ["push"]]

    def stage(self, name: str) -> Path:
        checkpoint = self.source / "_checkpoints" / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "trainer_state.json").write_text("{}")
        return checkpoint


def _debounce(seconds: float):
    """Pin the configured quiet window for one test.

    Patches the accessor the publisher reads, so the test drives the same code
    path production does rather than a test-only constructor argument.
    """
    import types

    colab = types.SimpleNamespace(
        dvc_remote_url="https://dagshub.com/owner/repo.dvc",
        dvc_events_file="dvc_events.jsonl",
        dvc_jobs=4,
        dvc_push_retries=1,
        dvc_push_backoff_seconds=1,
        dvc_publish_debounce_seconds=seconds,
    )
    return mock.patch.object(
        dvc_store.common, "training_cfg",
        return_value=types.SimpleNamespace(colab=colab),
    )


class StreamingPublishTests(unittest.TestCase):
    def test_a_burst_of_ready_files_leaves_as_one_push(self):
        with _Workspace() as workspace, _debounce(0.4):
            streamer = dvc_store.checkpoint_streamer(workspace.source)
            for name in ("checkpoint-10", "checkpoint-20", "checkpoint-30"):
                dvc_store.stage_checkpoint(workspace.source, workspace.stage(name))
            deadline = time.monotonic() + 20
            while len(streamer.verified()) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(len(streamer.verified()), 3)
            pushes = workspace.pushes()
            self.assertEqual(len(pushes), 1, f"expected one push, got {pushes}")
            targets = [part for part in pushes[0][1:] if part.endswith(".dvc")]
            self.assertEqual(len(targets), 3, targets)

    def test_a_lone_ready_file_is_published_promptly(self):
        with _Workspace() as workspace, _debounce(0.4):
            streamer = dvc_store.checkpoint_streamer(workspace.source)
            signalled = time.monotonic()
            dvc_store.stage_checkpoint(workspace.source, workspace.stage("checkpoint-10"))
            deadline = time.monotonic() + 20
            while not streamer.verified() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(streamer.verified()), 1)
            first_push = min(c["at"] for c in workspace.calls() if c["argv"][:1] == ["push"])
            delay = first_push - signalled
            self.assertEqual(len(workspace.pushes()), 1)
            # Prompt means within the window plus the round trip, not at train end.
            self.assertLess(delay, 5.0, f"publish delay was {delay:.2f}s")

    def test_the_quiet_window_comes_from_config(self):
        for window in (0.3, 1.2):
            with _Workspace() as workspace, _debounce(window):
                streamer = dvc_store.checkpoint_streamer(workspace.source)
                signalled = time.monotonic()
                dvc_store.stage_checkpoint(
                    workspace.source, workspace.stage("checkpoint-10")
                )
                deadline = time.monotonic() + 20
                while not streamer.verified() and time.monotonic() < deadline:
                    time.sleep(0.02)
                pushed = min(
                    c["at"] for c in workspace.calls() if c["argv"][:1] == ["push"]
                )
                self.assertGreaterEqual(pushed - signalled, window * 0.8, window)
                self.assertLess(pushed - signalled, window + 4.0, window)

    def test_a_failing_remote_warns_and_leaves_the_target_for_the_final_batch(self):
        with _Workspace() as workspace, _debounce(0.4):
            os.environ["FAKE_DVC_PUSH_FAILS"] = "1"
            try:
                streamer = dvc_store.checkpoint_streamer(workspace.source)
                dvc_store.stage_checkpoint(
                    workspace.source, workspace.stage("checkpoint-10")
                )
                deadline = time.monotonic() + 20
                while not streamer.failures() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(len(streamer.failures()), 1)
                self.assertEqual(streamer.verified(), set())
            finally:
                del os.environ["FAKE_DVC_PUSH_FAILS"]
            # The final batch still pushes it, so durability is not lost.
            checkpoint = workspace.stage("checkpoint-10")
            dvc_store.publish_checkpoints(
                workspace.source, [(checkpoint, "checkpoint-10--x", checkpoint)],
                already_staged=True,
            )
            self.assertGreaterEqual(len(workspace.pushes()), 2)

    def test_the_final_batch_does_not_re_push_what_was_already_streamed(self):
        with _Workspace() as workspace, _debounce(0.4):
            streamer = dvc_store.checkpoint_streamer(workspace.source)
            checkpoint = workspace.stage("checkpoint-10")
            dvc_store.stage_checkpoint(workspace.source, checkpoint)
            deadline = time.monotonic() + 20
            while not streamer.verified() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(len(workspace.pushes()), 1)
            dvc_store.publish_checkpoints(
                workspace.source, [(checkpoint, "checkpoint-10--x", checkpoint)],
                already_staged=True,
            )
            self.assertEqual(
                len(workspace.pushes()), 1,
                "the final batch must not re-push a streamed target",
            )


class TrainingIsNeverBlockedTests(unittest.TestCase):
    def test_signalling_costs_the_trainer_almost_nothing_while_a_push_is_in_flight(self):
        """The training thread's only interaction with the publisher is an append."""
        with _Workspace() as workspace, _debounce(0.3):
            os.environ["FAKE_DVC_PUSH_SECONDS"] = "3"
            try:
                streamer = dvc_store.checkpoint_streamer(workspace.source)
                # Put a push in flight and keep it there.
                streamer.signal("_checkpoints/slow/checkpoint-1.dvc")
                time.sleep(0.6)
                self.assertTrue(streamer._thread.is_alive())
                # Now behave like a training loop that keeps stepping.
                steps = 0
                worst = 0.0
                loop_end = time.monotonic() + 2.0
                while time.monotonic() < loop_end:
                    began = time.perf_counter()
                    streamer.signal(f"_checkpoints/live/checkpoint-{steps}.dvc")
                    worst = max(worst, time.perf_counter() - began)
                    steps += 1
                    time.sleep(0.01)
                self.assertGreater(steps, 50, "the training loop did not keep stepping")
                self.assertLess(
                    worst, 0.05,
                    f"signalling blocked the caller for {worst * 1000:.1f}ms",
                )
            finally:
                del os.environ["FAKE_DVC_PUSH_SECONDS"]

    def test_training_path_does_not_take_the_locks_the_publisher_holds(self):
        """Who acquires .dvc-push.lock, and on which thread.

        The trainer's own thread only snapshots and submits; the lock is taken
        by the staging executor thread and by the publisher.  This pins that,
        because a lock the training thread wanted would make a slow remote a
        stalled run.
        """
        source_file = Path(dvc_store.__file__).read_text(encoding="utf-8")
        training_callback = (
            Path(common.TRAIN_ROOT) / "src" / "training" / "training.py"
        ).read_text(encoding="utf-8")
        start = training_callback.index("def on_save")
        # training.py has more than one callback class; take the on_save whose
        # own on_train_end follows it.
        end = training_callback.index("def on_train_end", start)
        on_save = training_callback[start:end]
        self.assertNotIn(
            ".dvc-push.lock", on_save,
            "the training thread must not take the lock the publisher holds",
        )
        self.assertIn(
            "self._stage_executor.submit(stage_checkpoint", on_save,
            "staging must be submitted to the executor, not run inline",
        )
        # And the streaming publisher must take that lock only on its own thread.
        loop = source_file[source_file.index("def _loop(self)"):source_file.index("def _push(self")]
        self.assertNotIn("dvc-push.lock", loop)


class StreamerLifecycleTests(unittest.TestCase):
    def test_the_streamer_stops_with_the_final_batch(self):
        with _Workspace() as workspace, _debounce(0.3):
            with _debounce(0.3):
                streamer = dvc_store.checkpoint_streamer(workspace.source)
                dvc_store.stop_checkpoint_streamer(workspace.source)
                self.assertFalse(streamer._thread.is_alive())

    def test_status_reports_what_happened(self):
        with _Workspace() as workspace, _debounce(0.3):
            streamer = dvc_store.checkpoint_streamer(workspace.source)
            dvc_store.stage_checkpoint(workspace.source, workspace.stage("checkpoint-10"))
            deadline = time.monotonic() + 20
            while not streamer.verified() and time.monotonic() < deadline:
                time.sleep(0.05)
            status = streamer.snapshot()
            self.assertEqual(status["verified"], 1)
            self.assertEqual(status["pushes"], 1)
            self.assertEqual(status["failed"], 0)


if __name__ == "__main__":
    unittest.main()
