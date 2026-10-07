"""Explicit execution policies and reusable detached GPU diagnostics.

The ``auto`` optimizer backend is upgraded by two opt-in switches (both
disabled by ``ER_PERF_LEGACY=1``):

  * ``accel.optimizer_fused``   -- on CUDA, prefer fused AdamW when the
    installed torch actually exposes it (``auto`` -> ``fused``).
  * ``accel.optimizer_foreach`` -- otherwise, on CPU or fused-less CUDA, use
    the ``foreach`` AdamW path when available (``auto`` -> ``foreach``).

Explicit backends (``foreach``/``fused``/``cuda_fused``) are unchanged and the
frozen ``validate_restored`` contract still compares the recorded flags
against the same resolution used at construction.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Iterable

import torch

from core.execution_policy import OptimizerBackend, ResolvedOptimizerBackend
from core.perf_switches import perf_enabled
from pydantic import BaseModel, ConfigDict


@functools.lru_cache(maxsize=1)
def _adamw_supports(name: str) -> bool:
    """Probe whether this torch's AdamW accepts ``name`` (never assume)."""
    try:
        return name in inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError):
        return False


def _device_type(device: str | torch.device) -> str:
    try:
        return torch.device(device).type
    except (RuntimeError, TypeError, ValueError):
        return 'cpu'


class OptimizerExecution(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    backend: OptimizerBackend

    def resolved_backend(self, device: str | torch.device) -> ResolvedOptimizerBackend:
        device_type = _device_type(device)
        if self.backend == 'cuda_fused':
            if device_type == 'cuda' and torch.cuda.is_available() and _adamw_supports('fused'):
                return 'fused'
            return 'auto'
        if self.backend != 'auto':
            return self.backend
        # ``auto``: opt-in accelerators, only when the installed torch supports
        # the flag and the device is actually usable. Explicit backends never
        # change, and an unavailable device stays ``auto`` (construction fails
        # later, exactly as before).
        if device_type == 'cuda' and torch.cuda.is_available():
            if perf_enabled('accel.optimizer_fused') and _adamw_supports('fused'):
                return 'fused'
            if perf_enabled('accel.optimizer_foreach') and _adamw_supports('foreach'):
                return 'foreach'
            return 'auto'
        if device_type == 'cpu' and perf_enabled('accel.optimizer_foreach') and _adamw_supports('foreach'):
            return 'foreach'
        return 'auto'

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
