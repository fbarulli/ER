"""The staged kernels' W&B step + teardown contract (the 8146f87 regression class).

Pins the ONE shared wandb-helpers block (``cli.laya_kernel_text_wandb``): every
sink rides wandb's monotonic auto-increment step (never an explicit ``step=``),
``wandb_init`` declares ``epoch`` the step metric of the per-epoch namespaces,
``wandb_init`` honors the staged ``tracking.wandb.mode`` (online-without-key
raises, offline needs no key, disabled is a loud skip), and ``wandb_finish``
re-drives ``wandb sync`` so an interrupted upload is recovered instead of
silently lost.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pytest

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


def _exec_helpers(fake_wandb, fake_subprocess, monkeypatch, *,
                  api_key="test-key", mode="online"):
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
        "WANDB_API_KEY": api_key,
        "WANDB_PROJECT": "e-r",
        "WANDB_MODE": mode,
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


def test_training_batch_metrics_survive_system_rows_with_their_own_axis(monkeypatch):
    fake = FakeWandb()
    namespace, _ = _exec_helpers(fake, FakeSubprocess(), monkeypatch)
    namespace["wandb_init"]()
    sink = namespace["WandbTrainingMetrics"]
    sink.publish(1, 0.75, 2e-5)
    fake.run.log({"gpu/util_pct": 100})
    sink.publish(100, 0.5, 1e-5)
    namespace["wandb_log_epoch"](0, 0.6)
    rows = [payload for _, payload in fake.run.rows if "train_step" in payload]
    assert rows == [
        {"train_step": 1, "train/batch_loss": 0.75, "train/batch_lr": 2e-5},
        {"train_step": 100, "train/batch_loss": 0.5, "train/batch_lr": 1e-5},
    ]
    assert fake.run.dropped == []
    for metric in ("train/batch_loss", "train/batch_lr"):
        assert (metric, {"step_metric": "train_step"}) in fake.defined


def test_online_mode_without_a_key_raises_instead_of_silently_disabling(
        monkeypatch):
    """The silent-disable regression: staged tracking.wandb.mode=online with no
    baked key must fail loud (the WandbCtx contract), never return None."""
    fake = FakeWandb()
    namespace, _lines = _exec_helpers(fake, FakeSubprocess(), monkeypatch,
                                      api_key="", mode="online")
    with pytest.raises(RuntimeError) as raised:
        namespace["wandb_init"]()
    message = str(raised.value)
    assert "WANDB_API_KEY absent" in message
    assert "tracking.wandb.mode=online" in message
    assert not hasattr(fake, "init_kwargs")  # never reached wandb.init


def test_offline_mode_inits_without_a_key(monkeypatch):
    """mode=offline needs no credential: the run starts local-offline."""
    fake = FakeWandb()
    namespace, _lines = _exec_helpers(fake, FakeSubprocess(), monkeypatch,
                                      api_key="", mode="offline")
    assert namespace["wandb_init"]() is fake.run
    assert fake.init_kwargs["mode"] == "offline"


def test_disabled_mode_skips_loudly_naming_the_mode_source(monkeypatch):
    """mode=disabled is a recorded no-op, never a silent one."""
    fake = FakeWandb()
    namespace, lines = _exec_helpers(fake, FakeSubprocess(), monkeypatch,
                                     mode="disabled")
    assert namespace["wandb_init"]() is None
    assert not hasattr(fake, "init_kwargs")
    assert any("disabled" in line and "tracking.wandb.mode" in line
               for line in lines)
