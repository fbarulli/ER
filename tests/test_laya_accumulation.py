"""Public training parity and communication budget across accumulation windows."""
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from test_laya_ddp import _Cfg, _load_ddp_namespace, _make_fake_laya, _TinyNet

from cli import laya_lane


class TestAccumulation:
    @staticmethod
    def worker(rank: int, destination: str, grad_accum: int, adv_eps: float) -> None:
        import torch.distributed as dist
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
            allreduce_hook,
        )

        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method="file://" + destination,
                                rank=rank, world_size=2)
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        namespace = _load_ddp_namespace()
        owner = namespace["DeterministicDdp"]
        accumulate = owner.accumulate
        wrap = owner.wrap
        counts = []

        def counted_wrap(*args: Any) -> Any:
            model = wrap(*args)

            def reduce(state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
                counts[-1] += 1
                return allreduce_hook(state, bucket)

            model.register_comm_hook(None, reduce)
            return model

        owner.wrap = staticmethod(counted_wrap)
        fake = _make_fake_laya([])
        sys.modules.update({"laya": fake, "laya.train": fake.train})
        namespace["FINETUNE_CONTROL"] = {
            **laya_lane.finetune_control(), "save_each_epoch": False,
            "resume": False, "eval_dev": False, "adv_eps": adv_eps,
            "compile_model": False,
        }
        config = _Cfg()
        config.epochs, config.micro_batch, config.grad_accum = 1, 1, grad_accum
        items = [{"i": i, "label": i % 3} for i in range(8)]
        states = []
        try:
            for optimized in (False, True):
                owner.accumulate = staticmethod(accumulate if optimized else
                    lambda *args: nullcontext())
                counts.append(0)
                torch.manual_seed(config.seed)
                model = _TinyNet()
                namespace["_perf_train_model"](
                    model, types.SimpleNamespace(pad_token_id=0), items,
                    config, torch.device("cpu"), 8, 4)
                states.append({key: value.clone() for key, value in model.state_dict().items()})
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key])
            baseline = 8 if adv_eps else 4
            assert counts == [baseline, baseline if adv_eps else (4 + grad_accum - 1) // grad_accum]
        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("grad_accum", [1, 3])
    @pytest.mark.parametrize("adv_eps", [0.0, 0.01])
    def test_accumulated_training_preserves_updates_and_reduces_collectives(
        self, tmp_path: Path, grad_accum: int, adv_eps: float,
    ) -> None:
        torch.multiprocessing.spawn(self.worker, args=(str(tmp_path / "rendezvous"), grad_accum, adv_eps),
                                    nprocs=2, join=True)
