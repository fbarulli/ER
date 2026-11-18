"""Train through the public injected loop; count real forward weight casts."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_laya_ddp import _Cfg, _load_ddp_namespace, _make_fake_laya
from torch.utils.checkpoint import checkpoint

from cli import laya_lane
from core.laya_eval_fast import EvalRecordsHook, EvalRecordsPolicy
from core.laya_train_fast import AutocastCacheFactory
from core.laya_train_parts import Forwarder


class CacheTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(0.1))
        self.head = torch.nn.Linear(6, 3)
        self.head_checkpointing = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.head_checkpointing:
            return checkpoint(self.head, checkpoint(self.encoder, inputs,
                use_reentrant=False), use_reentrant=False)
        return self.head(self.encoder(inputs))


class CacheTestFactory:
    @staticmethod
    def decision_model(checkpointing: bool) -> torch.nn.Module:
        from laya.common import DecisionModel
        from transformers import ModernBertConfig, ModernBertModel

        config = ModernBertConfig(vocab_size=128, hidden_size=64,
                                 pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                 intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, max_position_embeddings=32,
                                 local_attention=8)
        config._attn_implementation = "sdpa"
        model = DecisionModel(ModernBertModel(config), head_layers=1)
        model.head_checkpointing = checkpointing
        if checkpointing:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    @staticmethod
    def build(monkeypatch: pytest.MonkeyPatch) -> CacheTestModel:
        fake = _make_fake_laya([])
        original_collate = fake.train.collate_items

        def collate(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            batch = original_collate(*args, **kwargs)
            batch["marker_mask"] = torch.ones((len(batch["target"]), 3), dtype=torch.bool)
            return batch

        def forward(model: Any, batch: Any, device: Any, amp: bool,
                    freeze_encoder: bool) -> torch.Tensor:
            with torch.profiler.record_function("forward_weights"), torch.autocast(
                device.type, dtype=torch.float16, enabled=amp,
            ):
                return model(batch["x"]).float()

        fake.train.collate_items = collate
        fake.train._forward = forward
        monkeypatch.setitem(sys.modules, "laya", fake)
        monkeypatch.setitem(sys.modules, "laya.train", fake.train)
        torch.manual_seed(_Cfg.seed)
        return CacheTestModel()

    @staticmethod
    def forward_casts(profile: Any, model: CacheTestModel) -> int:
        shapes = {tuple(parameter.shape) for parameter in model.parameters()}
        count = 0
        for event in profile.events():
            if event.name != "aten::_to_copy" or tuple(event.input_shapes[0]) not in shapes:
                continue
            parent = event.cpu_parent
            while parent is not None and parent.name != "forward_weights":
                parent = parent.cpu_parent
            count += parent is not None
        return count


class TestAutocastCache:
    @pytest.mark.parametrize("checkpointing", [False, True])
    def test_laya_modernbert_forward_and_gradients_match_with_variable_lengths(
        self, checkpointing: bool,
    ) -> None:
        from laya import train

        original_cache = torch.is_autocast_cache_enabled()
        results = []
        try:
            for enabled in (False, True):
                torch.set_autocast_cache_enabled(enabled)
                torch.manual_seed(_Cfg.seed)
                model = CacheTestFactory.decision_model(checkpointing).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                losses = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with AutocastCacheFactory.build(torch, torch.device("cpu"), True):
                        for length in (8, 12):
                            batch = {
                                "input_ids": torch.arange(2 * length).reshape(2, length),
                                "attention_mask": torch.ones(2, length, dtype=torch.long),
                                "marker_pos": torch.tensor([[1, 3], [2, 4]]),
                                "marker_mask": torch.ones(2, 2, dtype=torch.bool),
                                "qtype": torch.tensor([0, 1]),
                            }
                            logits = Forwarder.run(torch, train, model, batch,
                                torch.device("cpu"), True, False, "fp16")
                            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
                            loss.backward()
                            losses.append(loss.detach())
                    optimizer.step()
                results.append((model.state_dict(), torch.stack(losses)))
        finally:
            torch.set_autocast_cache_enabled(original_cache)
            torch.clear_autocast_cache()
        torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
        for name in results[0][0]:
            torch.testing.assert_close(results[0][0][name], results[1][0][name], rtol=0, atol=0)

    def test_eval_records_reuse_casts_and_release_between_calls(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_cache = torch.is_autocast_cache_enabled()
        results = []
        try:
            for enabled in (False, True):
                torch.set_autocast_cache_enabled(enabled)
                model = CacheTestFactory.build(monkeypatch)
                train = sys.modules["laya.train"]

                @torch.no_grad()
                def records(model: Any, tok: Any, items: Any, device: Any,
                            *args: Any, _train: Any = train, **kwargs: Any) -> list[torch.Tensor]:
                    model.eval()
                    return [_train._forward(model, {"x": torch.ones(1, 4)},
                                           device, True, False) for _ in items]

                train.calibration_records = records
                EvalRecordsHook.install(train, EvalRecordsPolicy(eval_amp=True))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                             record_shapes=True) as profile:
                    first = train.calibration_records(model, None, range(4), torch.device("cpu"))
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(0.25)
                    second = train.calibration_records(model, None, range(4), torch.device("cpu"))
                results.append((first + second, CacheTestFactory.forward_casts(profile, model)))
        finally:
            torch.set_autocast_cache_enabled(original_cache)
            torch.clear_autocast_cache()
        for reference, cached in zip(results[0][0], results[1][0], strict=True):
            torch.testing.assert_close(reference, cached, rtol=0, atol=0)
        assert (results[0][1], results[1][1]) == (32, 8)

    @pytest.mark.parametrize(("checkpointing", "r_drop", "adv_eps", "compiled"), [
        (False, False, 0.0, False), (True, False, 0.0, False),
        (False, True, 0.0, False), (False, False, 0.01, False),
        (False, False, 0.0, True),
    ])
    def test_weight_cast_reuse_preserves_updates_and_refreshes_after_mutation(
        self, monkeypatch: pytest.MonkeyPatch, checkpointing: bool,
        r_drop: bool, adv_eps: float, compiled: bool,
    ) -> None:
        original_cache = torch.is_autocast_cache_enabled()
        results = []
        try:
            for enabled in (False, True):
                torch.set_autocast_cache_enabled(enabled)
                namespace = _load_ddp_namespace()
                namespace["FINETUNE_CONTROL"] = {
                    **laya_lane.finetune_control(), "compile_model": compiled,
                    "eval_dev": False, "resume": False, "save_each_epoch": False,
                    "r_drop": r_drop, "adv_eps": adv_eps,
                }
                config = _Cfg()
                config.amp, config.micro_batch, config.grad_accum = True, 1, 3
                config.gradient_checkpointing = checkpointing
                model = CacheTestFactory.build(monkeypatch)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                             record_shapes=True) as profile:
                    history = namespace["_perf_train_model"](
                        model, SimpleNamespace(pad_token_id=0),
                        [{"i": index, "label": index % 3} for index in range(5)],
                        config, torch.device("cpu"), 8, 4)
                results.append((model.state_dict(), history,
                                CacheTestFactory.forward_casts(profile, model)))
        finally:
            torch.set_autocast_cache_enabled(original_cache)
            torch.clear_autocast_cache()
        reference, cached = results
        assert reference[1] == cached[1]
        for name in reference[0]:
            torch.testing.assert_close(reference[0][name], cached[0][name], rtol=0, atol=0)
        # Adversarial updates invalidate both original and perturbed weights.
        if compiled:
            return
        if adv_eps:
            assert cached[2] == reference[2]
        else:
            assert cached[2] < reference[2]
            if not r_drop:
                assert (reference[2], cached[2]) == (40, 16)
