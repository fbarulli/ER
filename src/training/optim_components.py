"""Class-based optimizer / regularization / schedule components (SSOT-driven).

Every component is a small, GPU-free-testable class. Config keys live in
``config/training.yaml`` (validated by ``core.schemas``); each component has a
live consumer in the text and/or GNN lane. Defaults are OFF so the shipped
baseline is byte-identical until an experiment flips a knob.

Precedence rules (documented so two dials cannot silently fight):

* ``optimizer.lr_scaling`` vs explicit ``training.lr`` — ``lr`` is the
  reference peak LR *at* ``optimizer.base_batch``. With ``lr_scaling: none``
  the explicit ``lr`` is used verbatim. Otherwise the peak is multiplied by
  the effective-batch factor; at ``effective_batch == base_batch`` the factor
  is exactly 1.0, so the two agree.
* ``optimizer.no_decay_bias_norm`` vs ``training.weight_decay`` — no-decay
  params get ``weight_decay=0`` in their own param group; ``weight_decay``
  still applies to every other param. Turning no-decay off restores a single
  weight decay.
* ``training.batch_size_ramp`` vs the frozen batch plan — the MICRO batch size
  is owned by the frozen sampler, so the ramp adjusts gradient accumulation
  only (effective batch), never the frozen batch composition.
* ``data.dynamic_padding`` vs a fixed ``max_seq_length`` — the prepared-token
  path pads to the longest row in the batch regardless; the knob only rounds
  that width up to ``pad_to_multiple``. Disabled = exact longest row (today).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

_OPT_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


class ComponentConfigError(ValueError):
    """Raised when a component knob is internally inconsistent."""


# ─────────────────────────────────────────────────────────────────────────────
# 1. no-decay param groups (bias / LayerNorm / norm)
# ─────────────────────────────────────────────────────────────────────────────


class NoDecayParamGroups:
    """Split optimizer param groups into weight-decay and no-decay subsets.

    The canonical rule (AdamW paper / HuggingFace convention): 1-D params
    (biases, LayerNorm/norm scales) are excluded from weight decay. Matching is
    name-based AND shape-based so a name that says "norm" is excluded even when
    wrapped, and a bias is excluded even if its name is unusual.
    """

    def __init__(self, weight_decay: float, enabled: bool = True):
        if weight_decay < 0.0:
            raise ComponentConfigError("weight_decay must be >= 0")
        self.weight_decay = float(weight_decay)
        self.enabled = bool(enabled)

    @staticmethod
    def is_no_decay(name: str, param: torch.Tensor) -> bool:
        if param.ndim <= 1:
            return True
        lowered = name.lower()
        return lowered.endswith("bias") or "norm" in lowered

    def apply(
        self,
        groups: Sequence[Mapping[str, Any]],
        name_by_id: Mapping[int, str],
    ) -> list[dict]:
        """Return groups tagged with ``weight_decay`` in the AdamW contract."""
        if not self.enabled:
            return [
                {**dict(group), "weight_decay": self.weight_decay}
                for group in groups
            ]
        out: list[dict] = []
        for group in groups:
            decay, no_decay = [], []
            for param in group["params"]:
                target = (
                    no_decay
                    if self.is_no_decay(name_by_id.get(id(param), ""), param)
                    else decay
                )
                target.append(param)
            for params, weight_decay in ((decay, self.weight_decay), (no_decay, 0.0)):
                if not params:
                    continue
                rebuilt = {k: v for k, v in group.items() if k != "params"}
                rebuilt["weight_decay"] = weight_decay
                rebuilt["params"] = params
                out.append(rebuilt)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. optimizer state precision (bf16 optimizer states / master weights)
# ─────────────────────────────────────────────────────────────────────────────


class OptimizerStatePrecision:
    """Cast Adam-family optimizer state to a lower precision to save memory.

    Parameters (and the master weights) stay in their own dtype; only the
    moment estimates (``exp_avg``/``exp_avg_sq``/``max_exp_avg_sq``) are cast.
    Called after every optimizer step so states rest in ``state_dtype`` between
    steps and are upcast on demand inside the next step (AMP-compatible: the
    GradScaler still tracks the fp32 loss/gradients).
    """

    def __init__(self, state_dtype: str):
        if state_dtype not in _OPT_DTYPES:
            raise ComponentConfigError(
                f"optimizer.state_dtype must be one of {sorted(_OPT_DTYPES)}, "
                f"got {state_dtype!r}"
            )
        self.state_dtype = _OPT_DTYPES[state_dtype]

    @property
    def enabled(self) -> bool:
        return self.state_dtype != torch.float32

    def cast_(self, optimizer) -> int:
        """Cast moment buffers in place; returns the number of tensors cast."""
        if not self.enabled:
            return 0
        cast = 0
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if key == "step" or not torch.is_tensor(value):
                    continue
                if value.is_floating_point() and value.dtype != self.state_dtype:
                    state[key] = value.to(self.state_dtype)
                    cast += 1
        return cast


# ─────────────────────────────────────────────────────────────────────────────
# 3. LR scaling from effective batch
# ─────────────────────────────────────────────────────────────────────────────


class LrScalingRule:
    """Scale the peak LR by the effective batch ratio.

    ``none`` returns the reference LR verbatim (explicit LR wins). ``linear``
    and ``sqrt`` multiply it by ``effective / base_batch`` (or its square
    root). Effective batch = micro_batch * grad_accum * world_size.
    """

    def __init__(self, scaling: str = "none", base_batch: int | None = None):
        if scaling not in ("none", "linear", "sqrt"):
            raise ComponentConfigError(
                f"optimizer.lr_scaling must be none|linear|sqrt, got {scaling!r}"
            )
        if scaling != "none" and (base_batch is None or base_batch < 1):
            raise ComponentConfigError(
                "optimizer.base_batch is required (>0) when lr_scaling != none"
            )
        self.scaling = scaling
        self.base_batch = None if base_batch is None else int(base_batch)

    @staticmethod
    def effective_batch(micro_batch: int, grad_accum: int, world_size: int) -> int:
        if min(micro_batch, grad_accum, world_size) < 1:
            raise ComponentConfigError("batch/accum/world_size must be >= 1")
        return int(micro_batch) * int(grad_accum) * int(world_size)

    def factor(self, effective_batch: int) -> float:
        if self.scaling == "none":
            return 1.0
        ratio = float(effective_batch) / float(self.base_batch)
        return ratio if self.scaling == "linear" else math.sqrt(ratio)

    def scale(self, reference_lr: float, effective_batch: int) -> float:
        return float(reference_lr) * self.factor(effective_batch)

    def peak_lr(
        self,
        reference_lr: float,
        *,
        micro_batch: int,
        grad_accum: int,
        world_size: int,
    ) -> float:
        return self.scale(
            reference_lr,
            self.effective_batch(micro_batch, grad_accum, world_size),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. R-Drop (bidirectional KL consistency)
# ─────────────────────────────────────────────────────────────────────────────


class RDropRegularizer:
    """R-Drop penalty: ``alpha`` * mean bidirectional KL between two logits."""

    def __init__(self, alpha: float = 0.0):
        if alpha < 0.0:
            raise ComponentConfigError("regularization.r_drop.alpha must be >= 0")
        self.alpha = float(alpha)

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    @staticmethod
    def _binary_kl(p_logits, q_logits):
        # KL(p || q) for Bernoulli with BCEWithLogits: E_p[log p - log q].
        p = torch.sigmoid(p_logits)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            q_logits, p
        ) - torch.nn.functional.binary_cross_entropy_with_logits(p_logits, p)

    def penalty(self, logits_a, logits_b):
        """Symmetric KL penalty; zero tensor when disabled."""
        if not self.enabled:
            return logits_a.sum() * 0.0
        symmetric = 0.5 * (
            self._binary_kl(logits_a, logits_b)
            + self._binary_kl(logits_b, logits_a)
        )
        return self.alpha * symmetric


# ─────────────────────────────────────────────────────────────────────────────
# 5. drop path / stochastic depth
# ─────────────────────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch (B, ...) input."""

    def __init__(self, rate: float = 0.0):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ComponentConfigError("drop_path rate must be in [0, 1)")
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"rate={self.rate}"


def drop_path_rate(
    rate: float, epoch: int, epochs: int, *, schedule: str = "constant"
) -> float:
    """Per-epoch drop-path rate. ``linear`` ramps 0 -> rate; ``constant`` flat."""
    if schedule not in ("linear", "constant"):
        raise ComponentConfigError(f"drop_path.schedule must be linear|constant, got {schedule!r}")
    if epochs < 1:
        raise ComponentConfigError("epochs must be >= 1")
    if schedule == "constant":
        return float(rate)
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, epochs - 1)))
    return float(rate) * progress


class _DropPathAfter(nn.Module):
    """Apply an existing branch module, then stochastic depth on the branch.

    Inserted in place of a residual branch's ``.dropout`` so the branch output
    is scaled BEFORE it is added to the shortcut — the textbook stochastic
    depth placement, while preserving the original dropout.
    """

    def __init__(self, base: nn.Module, rate: float = 0.0):
        super().__init__()
        self.base = base
        self.drop_path = DropPath(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.base(x))


def apply_drop_path(branch_modules: Sequence[nn.Module], rate_fn: Callable[[int], float]) -> Callable[[int], None]:
    """Inject drop-path into residual branch modules exposing a ``.dropout``.

    For each module (e.g. a HF ``BertSelfOutput``/``BertOutput``) the existing
    ``.dropout`` is replaced by :class:`_DropPathAfter`, so stochastic depth is
    applied to the branch output before the residual add. Returns a
    ``set_epoch(epoch)`` handle so the caller advances the linear schedule.
    """
    injected: list[DropPath] = []
    for module in branch_modules:
        base = getattr(module, "dropout", None)
        if not isinstance(base, nn.Module):
            raise ComponentConfigError(
                "drop_path requires branch modules exposing a nn.Module .dropout"
            )
        wrapper = _DropPathAfter(base, 0.0)
        module.dropout = wrapper
        injected.append(wrapper.drop_path)

    def set_epoch(epoch: int) -> None:
        new_rate = float(rate_fn(epoch))
        for drop in injected:
            drop.rate = new_rate

    return set_epoch


# ─────────────────────────────────────────────────────────────────────────────
# 6. dynamic padding
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_pad_width(lengths: Sequence[int], *, pad_to_multiple: int = 1) -> int:
    """Longest length in the batch, rounded up to ``pad_to_multiple``."""
    if not lengths:
        raise ComponentConfigError("cannot pad an empty batch")
    if pad_to_multiple < 1:
        raise ComponentConfigError("pad_to_multiple must be >= 1")
    width = max(int(length) for length in lengths)
    if pad_to_multiple == 1:
        return width
    return int(math.ceil(width / pad_to_multiple) * pad_to_multiple)


# ─────────────────────────────────────────────────────────────────────────────
# 7. batch-size (effective) ramp
# ─────────────────────────────────────────────────────────────────────────────


class BatchSizeRamp:
    """Ramp the EFFECTIVE batch via gradient accumulation (frozen batch safe).

    The micro batch is owned by the frozen sampler, so this only scales
    accumulation: epoch 1 starts at ``start_frac`` of the base accumulation and
    reaches the full value by ``ramp_epochs``.
    """

    def __init__(self, enabled: bool, start_frac: float, ramp_epochs: int):
        if not 0.0 < start_frac <= 1.0:
            raise ComponentConfigError("batch_size_ramp.start_frac must be in (0, 1]")
        if ramp_epochs < 1:
            raise ComponentConfigError("batch_size_ramp.ramp_epochs must be >= 1")
        self.enabled = bool(enabled)
        self.start_frac = float(start_frac)
        self.ramp_epochs = int(ramp_epochs)

    def factor(self, epoch: int) -> float:
        if not self.enabled:
            return 1.0
        if self.ramp_epochs == 1:
            return 1.0
        progress = min(1.0, max(0.0, (epoch - 1) / (self.ramp_epochs - 1)))
        return self.start_frac + (1.0 - self.start_frac) * progress

    def grad_accum(self, epoch: int, base_grad_accum: int) -> int:
        return max(1, int(round(int(base_grad_accum) * self.factor(epoch))))


# ─────────────────────────────────────────────────────────────────────────────
# 8/9. loss-weight schedules (uniformity weight, contrastive margin, aux)
# ─────────────────────────────────────────────────────────────────────────────


class LossWeightSchedule:
    """Epoch-wise multiplier for loss-term weights (uniformity/margin/aux).

    All scheduled terms ramp from 0 to their configured value across the first
    ``warmup_epochs`` (``schedule``: linear or cosine), then hold. Disabled ->
    a constant multiplier of 1.0, so every weight is the one-shot config value.
    """

    def __init__(
        self,
        enabled: bool,
        warmup_epochs: int,
        schedule: str,
        *,
        uniformity: bool = False,
        margin: bool = False,
        auxiliary: bool = False,
    ):
        if warmup_epochs < 1:
            raise ComponentConfigError("loss_schedules.warmup_epochs must be >= 1")
        if schedule not in ("linear", "cosine", "constant"):
            raise ComponentConfigError(
                f"loss_schedules.schedule must be linear|cosine|constant, got {schedule!r}"
            )
        self.enabled = bool(enabled)
        self.warmup_epochs = int(warmup_epochs)
        self.schedule = schedule
        self.terms = {
            "uniformity": bool(uniformity),
            "margin": bool(margin),
            "auxiliary": bool(auxiliary),
        }

    def factor(self, epoch: int) -> float:
        if not self.enabled:
            return 1.0
        if self.schedule == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (epoch - 1) / max(1, self.warmup_epochs - 1)))
        if self.schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        return progress

    def scheduled(self, term: str, base_value: float, epoch: int) -> float:
        if term not in self.terms:
            raise ComponentConfigError(f"unknown loss term {term!r}")
        if not self.terms[term]:
            return float(base_value)
        return float(base_value) * self.factor(epoch)
