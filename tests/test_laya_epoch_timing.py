"""Epoch telemetry includes evaluation and the epoch-end checkpoint work."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_laya_ddp import _Cfg, _load_ddp_namespace, _make_fake_laya, _TinyNet

from cli import laya_lane


class EpochClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def checkpoint(self, epoch: int, loss: float) -> None:
        self.seconds += 17.0


class TestEpochTiming:
    def test_epoch_log_includes_checkpoint_callback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        fake = _make_fake_laya([])
        monkeypatch.setitem(sys.modules, "laya", fake)
        monkeypatch.setitem(sys.modules, "laya.train", fake.train)
        namespace = _load_ddp_namespace()
        namespace["FINETUNE_CONTROL"] = {
            **laya_lane.finetune_control(), "resume": False, "save_each_epoch": True,
        }
        namespace["FINETUNE_CHECKPOINT_DIR"] = str(tmp_path / "checkpoint")
        namespace["FINETUNE_OUTPUT_DIR"] = str(tmp_path)
        namespace["FINETUNE_RUN_TAG"] = "epoch-timing"
        namespace["FINETUNE_DEV_ROWS"] = [{"state": "x"}]
        logged = []

        def log_epoch(epoch: int, mean: float, extra: dict[str, object]) -> None:
            logged.append(extra)

        namespace["wandb_log_epoch"] = log_epoch
        clock = EpochClock()
        monkeypatch.setattr(time, "time", clock.now)
        config = _Cfg()
        config.epochs = 1
        namespace["_perf_train_model"](
            _TinyNet(), SimpleNamespace(pad_token_id=0),
            [{"i": index, "label": index % 3} for index in range(4)], config,
            torch.device("cpu"), 8, 4, on_epoch_end=clock.checkpoint)
        assert logged[0]["epoch_time_s"] == 17.0
