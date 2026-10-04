"""Explicit execution policies and reusable detached GPU diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from core.execution_policy import OptimizerBackend, ResolvedOptimizerBackend
from pydantic import BaseModel, ConfigDict


class OptimizerExecution(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    backend: OptimizerBackend

    def resolved_backend(self, device: str | torch.device) -> ResolvedOptimizerBackend:
        if self.backend == 'cuda_fused':
            return 'fused' if torch.device(device).type == 'cuda' else 'auto'
        return self.backend

    def kwargs(self, device: str | torch.device) -> dict[str, bool]:
        backend = self.resolved_backend(device)
        if backend == 'fused' and torch.device(device).type != 'cuda':
            raise ValueError('fused AdamW requires a configured CUDA device')
        return {} if backend == 'auto' else {backend: True}

    def validate_restored(self, optimizer: torch.optim.Optimizer) -> None:
        expected = self.kwargs(optimizer.param_groups[0]['params'][0].device)
        for group in optimizer.param_groups:
            for name in ('fused', 'foreach'):
                if group.get(name) != expected.get(name):
                    raise ValueError(f'restored AdamW {name} differs from configured {self.backend} backend')


@dataclass(frozen=True)
class GradientStatistics:
    names: tuple[str, ...]
    parameters: tuple[torch.Tensor, ...]
    norms: tuple[torch.Tensor, ...]
    total_norm: torch.Tensor

    @classmethod
    def collect(cls, named_parameters: Iterable[tuple[str, torch.Tensor]]) -> GradientStatistics:
        retained = tuple((name, p) for name, p in named_parameters if p.grad is not None)
        if not retained:
            raise RuntimeError('training objective produced no parameter gradients')
        norms = tuple(torch.linalg.vector_norm(p.grad, 2) for _, p in retained)
        # This is PyTorch get_total_norm's norm-of-norms arithmetic. Graph
        # parameters share a device/dtype, as their model contract requires.
        total = torch.linalg.vector_norm(torch.stack(norms), 2)
        return cls(tuple(name for name, _ in retained), tuple(p for _, p in retained), norms, total)

    def clip(self, maximum: float) -> None:
        # Use the public PyTorch implementation, including its clamp/epsilon,
        # while reusing norms already needed for pre-clipping telemetry.
        torch.nn.utils.clip_grads_with_norm_(self.parameters, maximum, self.total_norm)
