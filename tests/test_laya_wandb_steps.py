"""The staged kernels' W&B step + teardown contract (the 8146f87 regression class).

Pins the ONE shared wandb-helpers block (``cli.laya_kernel_text_wandb``): every
sink rides wandb's monotonic auto-increment step (never an explicit ``step=``),
``wandb_init`` declares ``epoch`` the step metric of the per-epoch namespaces,
and ``wandb_finish`` re-drives ``wandb sync`` so an interrupted upload is
recovered instead of silently lost.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from cli.laya_kernel_text_wandb import WANDB_HELPERS_SOURCE


class FakeWandbRun:
    """wandb's step semantics: auto-increment; an explicit step that regresses
    is DROPPED ("Tried to log to step N that is less than the current step")."""

    def __init__(self):
        self.step = 0
        self.rows: list[tuple[int, dict]] = []
        self.dropped: list[dict] = []
        self.finished = False
        self.dir = "/tmp/kernel/wandb/run-20260101_000000-abc123/files"

    def log(self, payload, step=None):
        if step is not None:
            if step < self.step:
                self.dropped.append(dict(payload))
                return
            self.step = int(step)
        self.rows.append((self.step, dict(payload)))
        self.step += 1

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeWandbRun()
        self.defined: list[tuple] = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))


class FakeSubprocess:
    def __init__(self, returncode=0):
        self.calls: list[list] = []
        self.returncode = returncode

    def run(self, command, **kwargs):
        self.calls.append(list(command))

        class Completed:
            pass

        completed = Completed()
        completed.returncode = self.returncode
        completed.stdout = "synced"
        completed.stderr = "" if self.returncode == 0 else "upload failed"
        return completed


def _exec_helpers(fake_wandb, fake_subprocess, monkeypatch):
    """Exec the shared kernel-text helpers in a kernel-like namespace."""
    lines: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    namespace = {
        "os": __import__("os"),
        "sys": sys,
        "time": time,
        "traceback": traceback,
        "Path": Path,
        "subprocess": fake_subprocess,
        "WANDB_API_KEY": "test-key",
        "WANDB_PROJECT": "e-r",
        "RUN_TAG": "laya_test",
        "WANDB_INIT_CONFIG": {"recipe": True},
        "log": lines.append,
    }
    exec(compile(WANDB_HELPERS_SOURCE, "<wandb-helpers>", "exec"), namespace)
    return namespace, lines


def test_per_epoch_rows_survive_the_interleaved_sinks_on_monotonic_steps(
        monkeypatch):
    """The regression: once the auto-increment sinks (log/line, events, timing)
    advanced the global step, the epoch rows' explicit 0-based step regressed
    and wandb dropped EVERY per-epoch metric row. The shared helpers must keep
    the step monotonic across the mixed sinks, land one row per epoch with the
    epoch as a value, and declare `epoch` the per-epoch step metric."""
    fake_wandb = FakeWandb()
    namespace, _lines = _exec_helpers(fake_wandb, FakeSubprocess(), monkeypatch)
    run = fake_wandb.run

    namespace["wandb_init"]()
    for index in range(25):  # boot lines: two log/line flushes advance the step
        namespace["LOG_SINK"].add("boot line %d" % index)
    namespace["wandb_log_event"]("kernel_boot", run_tag="laya_test")
    for epoch in range(3):
        namespace["wandb_log_epoch"](
            epoch, 0.5 - 0.1 * epoch,
            {"epoch": epoch + 1, "train/lr": 1e-5, "epoch_time_s": 12.0,
             "dev/accuracy": 0.4 + 0.2 * epoch, "dev/loss": 0.3,
             "select/best_dev_accuracy": 0.4 + 0.2 * epoch,
             "select/bad_epochs": 0, "skipped_none": None})
        namespace["LOG_SINK"].add("interleaved noise")
        namespace["LOG_SINK"].flush()
        namespace["wandb_log_event"]("timing", epoch=epoch + 1)
    namespace["wandb_log_control_summary"](
        {"stopped": True, "stopped_epoch": 3, "best_dev_accuracy": 0.8,
         "bad_epochs": 0})
    namespace["wandb_finish"]()

    assert run.dropped == []
    steps = [step for step, _payload in run.rows]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    epoch_rows = [payload for _step, payload in run.rows
                  if "train/mean_loss" in payload]
    assert [row["epoch"] for row in epoch_rows] == [1, 2, 3]
    assert "skipped_none" not in epoch_rows[0]
    epoch_declarations = [name for name, kwargs in fake_wandb.defined
                          if kwargs.get("step_metric") == "epoch"]
    for pattern in ("train/*", "dev/*", "select/*", "early_stop/*",
                    "epoch_time_s"):
        assert pattern in epoch_declarations


def test_wandb_finish_redrives_the_upload_with_wandb_sync(monkeypatch):
    """The run logged "Fatal error while uploading data ... Use wandb sync":
    teardown must re-drive the upload for the run dir (fail-soft, recorded)."""
    fake_subprocess = FakeSubprocess()
    namespace, _lines = _exec_helpers(FakeWandb(), fake_subprocess, monkeypatch)
    namespace["wandb_init"]()
    namespace["wandb_finish"]()
    assert fake_subprocess.calls == [
        [sys.executable, "-m", "wandb", "sync",
         "/tmp/kernel/wandb/run-20260101_000000-abc123"]]

    # A failing sync is recorded with its reason, never raised.
    failing = FakeSubprocess(returncode=1)
    namespace, lines = _exec_helpers(FakeWandb(), failing, monkeypatch)
    namespace["wandb_init"]()
    namespace["wandb_finish"]()
    assert any("sync failed" in line for line in lines)
