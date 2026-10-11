"""Layer decay keeps optimizer semantics while batching equal-rate parameters."""
from __future__ import annotations

import copy

import torch

from core.laya_train_parts import TrainingOptimizer


class TestOptimizerGroups:
    def test_layer_decay_groups_preserve_options_and_adamw_updates(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.layer = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])
        model.head = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(model)
        groups = [{"params": list(model.parameters()), "lr": 0.01,
                   "weight_decay": 0.2, "betas": (0.8, 0.9)}]
        actual = TrainingOptimizer.apply_layer_decay(model, groups, 0.5)
        assert len(actual) == 2
        assert all(group["weight_decay"] == 0.2 for group in actual)
        expected = [{"params": [param], "lr": 0.005 if name.startswith("layer.0.") else 0.01,
                     "weight_decay": 0.2, "betas": (0.8, 0.9)}
                    for name, param in reference.named_parameters()]
        optimizers = [torch.optim.AdamW(actual), torch.optim.AdamW(expected)]
        for _ in range(2):
            for left, right in zip(model.parameters(), reference.parameters(), strict=True):
                left.grad = torch.ones_like(left)
                right.grad = left.grad.clone()
            for optimizer in optimizers:
                optimizer.step()
        for left, right in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
