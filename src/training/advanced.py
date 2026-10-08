"""Config-gated training enhancements — pure, testable logic.

Every public helper here is a small, side-effect-free function (or a tiny
stateful tracker) so it can be unit-tested without a GPU and so the training
loops only wire it behind an ``advanced.*`` config gate. The module must stay
importable without torch: torch is imported lazily inside the few functions
that need it.

Leakage discipline: the calibration helpers (``fit_temperature`` and the
reliability metrics) are fit-only utilities — callers fit them on the
dev/calibration carve and report on the test quarter. Nothing here reads a
test split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

# The one declared scheduler menu. ``plateau`` is special (ReduceLROnPlateau
# needs the dev metric), the rest are step-based. Validated by the config
# schema and by ``validate_lr_scheduler`` at wire time.
LR_SCHEDULERS: tuple[str, ...] = (
    "linear",
    "cosine",
    "one_cycle",
    "plateau",
    "constant",
)


class AdvancedConfigError(ValueError):
    """Raised when an advanced.* knob is internally inconsistent."""


def validate_lr_scheduler(name: str) -> str:
    """Return ``name`` when it is a declared scheduler, else raise."""
    if name not in LR_SCHEDULERS:
        raise AdvancedConfigError(
            f"unknown lr_scheduler {name!r}; expected one of {LR_SCHEDULERS}"
        )
    return name


# ─────────────────────────────────────────────────────────────────────────────
# EMA — exponential moving average of a model's parameters
# ─────────────────────────────────────────────────────────────────────────────


def ema_update(
    shadow: np.ndarray, current: np.ndarray, decay: float
) -> np.ndarray:
    """One EMA step: ``decay * shadow + (1 - decay) * current``."""
    if not 0.0 <= decay <= 1.0:
        raise AdvancedConfigError(f"EMA decay must be in [0, 1], got {decay}")
    return decay * shadow + (1.0 - decay) * current


def ema_decay_for_step(
    step: int, decay: float, *, warmup_updates: int = 0
) -> float:
    """Warmup-adjusted decay (``min(decay, (1+n)/(10+n))``) while warming up.

    With ``warmup_updates=0`` the configured decay is returned unchanged, so
    the tracked average is a pure EMA.
    """
    if warmup_updates <= 0:
        return float(decay)
    return min(float(decay), (1.0 + step) / (10.0 + step))


@dataclass
class EmaTracker:
    """Numpy EMA over a flat ``name -> array`` parameter mapping.

    The tracker is device-agnostic and float-only: non-floating entries
    (integer buffers such as ``num_batches_tracked``) are carried through
    unchanged. ``step`` drives the warmup decay, so a resumed run can reload
    the step count from the checkpoint manifest for reproducibility.
    """

    decay: float
    warmup_updates: int = 0
    step: int = 0
    _shadow: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise AdvancedConfigError(
                f"EMA decay must be in [0, 1], got {self.decay}"
            )

    def update(self, params: Mapping[str, Any]) -> None:
        """Fold ``params`` (arrays/tensors/numbers) into the shadow copy."""
        decay = ema_decay_for_step(
            self.step, self.decay, warmup_updates=self.warmup_updates
        )
        for name, value in params.items():
            array = _as_numpy_array(value)
            if array.dtype.kind != "f":
                self._shadow.setdefault(name, array)
                continue
            if name not in self._shadow:
                self._shadow[name] = array.astype(np.float64, copy=True)
            else:
                self._shadow[name] = ema_update(
                    self._shadow[name], array.astype(np.float64), decay
                )
        self.step += 1

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint-friendly payload (arrays + resume counters)."""
        return {
            "decay": float(self.decay),
            "warmup_updates": int(self.warmup_updates),
            "step": int(self.step),
            "shadow": {name: array for name, array in self._shadow.items()},
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "EmaTracker":
        tracker = cls(
            decay=float(payload["decay"]),
            warmup_updates=int(payload.get("warmup_updates", 0)),
            step=int(payload.get("step", 0)),
        )
        tracker._shadow = {
            str(name): _as_numpy_array(array)
            for name, array in dict(payload.get("shadow", {})).items()
        }
        return tracker

    def shadow(self) -> dict[str, np.ndarray]:
        return {name: array.copy() for name, array in self._shadow.items()}


def _as_numpy_array(value: Any) -> np.ndarray:
    """Convert a numpy array / torch tensor / scalar to an ndarray."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — temperature scaling + ECE/Brier + reliability diagram
# ─────────────────────────────────────────────────────────────────────────────


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled probabilities (binary logits, sigmoid link)."""
    if temperature <= 0.0:
        raise AdvancedConfigError(
            f"temperature must be > 0, got {temperature}"
        )
    return _sigmoid(np.asarray(logits, dtype=np.float64) / float(temperature))


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = np.clip(apply_temperature(logits, temperature), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def fit_temperature(
    logits: Sequence[float],
    labels: Sequence[int],
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 50.0,
) -> float:
    """Fit ONE scalar temperature on DEV logits/labels (never test).

    Minimizes binary NLL over ``T`` with a bounded scalar search. Returns the
    best temperature; T == 1.0 means the model was already calibrated.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits.size != labels.size:
        raise AdvancedConfigError("logits and labels must have equal length")
    if logits.size == 0:
        raise AdvancedConfigError("cannot fit temperature on an empty split")
    if not (min_temperature > 0.0 and max_temperature > min_temperature):
        raise AdvancedConfigError("temperature bounds must satisfy 0 < lo < hi")

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda t: _nll(logits, labels, float(t)),
            bounds=(min_temperature, max_temperature),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(np.clip(result.x, min_temperature, max_temperature))
    except Exception:
        # Deterministic fallback: log-spaced grid + local refinement, no scipy.
        grid = np.geomspace(min_temperature, max_temperature, 200)
        best = min(grid, key=lambda t: _nll(logits, labels, float(t)))
        return float(best)


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], *, n_bins: int = 15
) -> float:
    """Expected calibration error (equal-width probability bins)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size != labels.size:
        raise AdvancedConfigError("probs and labels must have equal length")
    if probs.size == 0:
        return float("nan")
    if n_bins < 1:
        raise AdvancedConfigError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / probs.size) * abs(confidence - accuracy)
    return float(ece)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between probabilities and binary labels."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size != labels.size:
        raise AdvancedConfigError("probs and labels must have equal length")
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram(
    probs: Sequence[float],
    labels: Sequence[int],
    *,
    n_bins: int = 15,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Reliability-diagram artifact payload (JSON-serializable).

    ``temperature`` is recorded for provenance; it is NOT applied here — the
    caller decides whether to fit/report calibrated or raw probabilities.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size != labels.size:
        raise AdvancedConfigError("probs and labels must have equal length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_payload: list[dict[str, Any]] = []
    if probs.size:
        assignment = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            mask = assignment == b
            count = int(mask.sum())
            bins_payload.append(
                {
                    "bin": b,
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": count,
                    "confidence": float(probs[mask].mean()) if count else 0.0,
                    "accuracy": float(labels[mask].mean()) if count else 0.0,
                }
            )
    return {
        "n_bins": int(n_bins),
        "n": int(probs.size),
        "temperature": None if temperature is None else float(temperature),
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
        "brier": brier_score(probs, labels),
        "bins": bins_payload,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Focal / class-weighted BCE (graph scorer loss)
# ─────────────────────────────────────────────────────────────────────────────


def sigmoid_focal_bce_with_logits(
    logits,
    targets,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: float | None = None,
    reduction: str = "mean",
):
    """Focal BCE with optional class weighting; torch tensors in/out.

    ``focal = (1 - p_t) ** gamma * BCE``. With ``gamma=0`` and no alpha/pos
    weight this is exactly ``binary_cross_entropy_with_logits``.
    """
    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    pw = None if pos_weight is None else torch.as_tensor(
        pos_weight, dtype=logits.dtype, device=logits.device
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw
    )
    if gamma and gamma > 0.0:
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        bce = (1.0 - p_t).pow(gamma) * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        bce = alpha_t * bce
    if reduction == "mean":
        return bce.mean()
    if reduction == "sum":
        return bce.sum()
    if reduction == "none":
        return bce
    raise AdvancedConfigError(f"unknown reduction {reduction!r}")


# ─────────────────────────────────────────────────────────────────────────────
# LR-scheduler factory (enum-validated)
# ─────────────────────────────────────────────────────────────────────────────


def lr_lambda(
    name: str,
    *,
    num_training_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Return a step->multiplier callable for a step-based scheduler.

    ``plateau`` has no step lambda (it needs the dev metric); callers detect it
    with :func:`is_plateau_scheduler`. ``constant`` returns 1.0 after warmup.
    """
    validate_lr_scheduler(name)
    if name == "plateau":
        raise AdvancedConfigError(
            "plateau is metric-driven; use is_plateau_scheduler() and "
            "ReduceLROnPlateau instead"
        )
    if num_training_steps <= 0:
        raise AdvancedConfigError("num_training_steps must be > 0")
    warmup_steps = max(0, min(int(warmup_steps), int(num_training_steps)))
    min_lr_ratio = float(min_lr_ratio)
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise AdvancedConfigError("min_lr_ratio must be in [0, 1]")

    def _warmup(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    if name == "constant":
        return lambda step: _warmup(step)

    if name == "linear":
        def _linear(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step)
            progress = min(
                1.0, (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        return _linear

    if name == "cosine":
        def _cosine(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step)
            progress = min(
                1.0, (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return _cosine

    # one_cycle: linear warmup to peak, then cosine anneal to the floor.
    def _one_cycle(step: int) -> float:
        if warmup_steps <= 0:
            warmup_steps_ = max(1, num_training_steps // 10)
        else:
            warmup_steps_ = warmup_steps
        if step < warmup_steps_:
            return max(min_lr_ratio, (step + 1) / warmup_steps_)
        progress = min(
            1.0, (step - warmup_steps_) / max(1, num_training_steps - warmup_steps_)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return _one_cycle


def is_plateau_scheduler(name: str) -> bool:
    validate_lr_scheduler(name)
    return name == "plateau"


def build_lr_scheduler(optimizer, *, name: str, num_training_steps: int,
                       warmup_steps: int = 0, min_lr_ratio: float = 0.0):
    """Build a torch scheduler from the validated enum (or None for plateau).

    Plateau must be constructed by the caller with the dev metric, so this
    returns ``None`` for it after validating the name.
    """
    import torch

    validate_lr_scheduler(name)
    if name == "plateau":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda(
            name,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SWA / top-k checkpoint averaging
# ─────────────────────────────────────────────────────────────────────────────


def average_state_dicts(
    states: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Weighted element-wise average of flat parameter mappings.

    Floating tensors/arrays are averaged; non-floating entries (e.g.
    ``num_batches_tracked`` ints) are taken from the first state. Returns
    numpy arrays when any input is numpy, otherwise the first state's tensor
    type via each tensor's own arithmetic (torch safe).
    """
    if not states:
        raise AdvancedConfigError("cannot average an empty checkpoint list")
    if weights is not None:
        if len(weights) != len(states):
            raise AdvancedConfigError("weights must match the number of states")
        total = float(sum(weights))
        if total <= 0.0:
            raise AdvancedConfigError("average weights must sum to > 0")
        norm = [float(w) / total for w in weights]
    else:
        norm = [1.0 / len(states)] * len(states)

    keys = set(states[0])
    for state in states[1:]:
        missing = keys - set(state)
        if missing:
            raise AdvancedConfigError(
                f"checkpoints disagree on parameters: missing {sorted(missing)}"
            )

    averaged: dict[str, Any] = {}
    for key in states[0]:
        first = states[0][key]
        if hasattr(first, "detach"):  # torch tensor
            import torch

            acc = torch.zeros_like(first, dtype=torch.float64)
            for state, weight in zip(states, norm, strict=True):
                acc = acc + state[key].to(torch.float64) * weight
            averaged[key] = acc.to(first.dtype)
        else:
            first_arr = _as_numpy_array(first)
            if first_arr.dtype.kind != "f":
                averaged[key] = first
                continue
            acc = np.zeros_like(first_arr, dtype=np.float64)
            for state, weight in zip(states, norm, strict=True):
                acc += _as_numpy_array(state[key]).astype(np.float64) * weight
            averaged[key] = acc.astype(first_arr.dtype)
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum on pair difficulty
# ─────────────────────────────────────────────────────────────────────────────

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "unknown": 1}


def difficulty_rank(value: Any) -> float:
    """Map a difficulty label/score to an ascending comparable rank."""
    if isinstance(value, str):
        return float(_DIFFICULTY_ORDER.get(value, 1))
    return float(value)


def curriculum_plan(
    difficulty: Sequence[Any],
    epochs: int,
    *,
    schedule: str = "easy_to_hard",
    warmup_fraction: float = 0.0,
) -> list[list[int]]:
    """Deterministic epoch -> available-item index plan (train side only).

    ``schedule`` is ``easy_to_hard`` or ``hard_to_easy``. The pool is sorted
    by difficulty once; each epoch exposes a growing prefix of that order,
    starting at ``warmup_fraction`` of the data and reaching 100% on the last
    epoch. Ties keep input order (stable), so the plan is reproducible.
    """
    if schedule not in ("easy_to_hard", "hard_to_easy"):
        raise AdvancedConfigError(
            f"curriculum schedule must be easy_to_hard|hard_to_easy, got {schedule!r}"
        )
    if epochs < 1:
        raise AdvancedConfigError("curriculum epochs must be >= 1")
    if not 0.0 <= warmup_fraction < 1.0:
        raise AdvancedConfigError("curriculum warmup_fraction must be in [0, 1)")
    n = len(difficulty)
    if n == 0:
        return [[] for _ in range(epochs)]
    order = sorted(range(n), key=lambda i: difficulty_rank(difficulty[i]))
    if schedule == "hard_to_easy":
        order = order[::-1]
    start = max(1, int(math.ceil(warmup_fraction * n)))
    plan: list[list[int]] = []
    for epoch in range(1, epochs + 1):
        if epochs == 1:
            count = n
        else:
            progress = (epoch - 1) / (epochs - 1)
            count = int(round(start + (n - start) * progress))
        count = min(n, max(start, count))
        plan.append(sorted(order[:count]))
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial FGM (embeddings)
# ─────────────────────────────────────────────────────────────────────────────


def adversarial_perturbation(x, grad, epsilon: float, *, norm: str = "l2",
                             emb_name: str = "embeddings"):
    """FGM perturbation: ``epsilon * grad / ||grad||`` (l2) or sign (linf).

    Returns the adversarial embedding ``x + delta`` detached from the graph.
    Pure torch helper; used train-only to harden the encoder against small
    embedding-space shifts.
    """
    import torch

    if epsilon < 0.0:
        raise AdvancedConfigError("FGM epsilon must be >= 0")
    g = grad if grad is not None else torch.zeros_like(x)
    if norm == "l2":
        value = g / (g.norm(p=2) + 1e-12)
    elif norm == "linf":
        value = g.sign()
    else:
        raise AdvancedConfigError(f"FGM norm must be l2|linf, got {norm!r}")
    return (x.detach() + epsilon * value).detach()


# ─────────────────────────────────────────────────────────────────────────────
# GPU-utilization telemetry parsing (nvidia-smi / NVML)
# ─────────────────────────────────────────────────────────────────────────────

_GPU_QUERY_FIELDS: tuple[tuple[str, str], ...] = (
    ("utilization.gpu", "gpu_util_pct"),
    ("power.draw", "power_w"),
    ("clocks.sm", "sm_clock_mhz"),
    ("clocks.mem", "mem_clock_mhz"),
    ("temperature.gpu", "temperature_c"),
    ("memory.used", "memory_used_mb"),
)


def parse_gpu_query(raw: str) -> dict[str, float]:
    """Parse one ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``
    row into a float mapping.

    Unknown/NA cells are dropped rather than guessed. The field order is the
    canonical query order above; a shorter row is accepted (trailing fields
    omitted).
    """
    cells = [cell.strip() for cell in str(raw).strip().split(",")]
    if not cells or cells == [""]:
        return {}
    telemetry: dict[str, float] = {}
    for (_, key), cell in zip(_GPU_QUERY_FIELDS, cells):
        try:
            telemetry[key] = float(cell)
        except ValueError:
            continue
    return telemetry


def gpu_query() -> str:
    """The canonical query string for :func:`parse_gpu_query` (display only)."""
    return ",".join(field for field, _ in _GPU_QUERY_FIELDS)


def collect_nvml_telemetry() -> dict[str, float]:
    """Best-effort NVML utilization/power/clocks for device 0.

    Returns ``{}`` when NVML/pynvml is unavailable — telemetry must never
    crash a run. Parsing lives in :func:`parse_gpu_query` so the pure logic is
    testable without a GPU.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
            return {
                "gpu_util_pct": float(util.gpu),
                "power_w": float(power),
                "sm_clock_mhz": float(sm),
                "mem_clock_mhz": float(mem),
                "temperature_c": float(temp),
                "memory_used_mb": float(used),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Gradient accumulation + embedding ensembling + distillation
# ─────────────────────────────────────────────────────────────────────────────


def accumulates_now(global_step: int, accumulation_steps: int) -> bool:
    """True when this micro-step closes an accumulation window (step on it).

    ``global_step`` is 0-indexed; with accumulation=1 every step closes.
    """
    if accumulation_steps < 1:
        raise AdvancedConfigError("gradient_accumulation_steps must be >= 1")
    return (global_step + 1) % accumulation_steps == 0


def scale_accumulated_loss(loss, accumulation_steps: int):
    """Divide a micro-batch loss so accumulated gradients match the mean."""
    if accumulation_steps < 1:
        raise AdvancedConfigError("gradient_accumulation_steps must be >= 1")
    return loss / accumulation_steps


def average_embeddings(
    embeddings: Iterable[np.ndarray], *, normalize: bool = True
) -> np.ndarray:
    """Average dense embeddings across folds/seeds, then optionally L2-norm."""
    arrays = [np.asarray(e, dtype=np.float64) for e in embeddings]
    if not arrays:
        raise AdvancedConfigError("cannot average zero embedding sets")
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise AdvancedConfigError(f"embedding shapes disagree: {sorted(shapes)}")
    mean = np.mean(np.stack(arrays, axis=0), axis=0)
    if normalize:
        norms = np.linalg.norm(mean, axis=1, keepdims=True)
        mean = mean / np.clip(norms, 1e-12, None)
    return mean


def distillation_loss(
    student_logits,
    teacher_logits,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    hard_labels=None,
):
    """KD loss: ``alpha * soft(student||teacher) + (1-alpha) * hard_BCE``.

    When ``hard_labels`` is None the hard term is dropped and the soft term is
    returned. Teacher logits are detached (no gradient path into the teacher).
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0.0:
        raise AdvancedConfigError("distillation temperature must be > 0")
    if not 0.0 <= alpha <= 1.0:
        raise AdvancedConfigError("distillation alpha must be in [0, 1]")
    teacher = torch.as_tensor(teacher_logits).detach().to(student_logits.device)
    t = float(temperature)
    soft = F.binary_cross_entropy_with_logits(
        student_logits / t, torch.sigmoid(teacher / t)
    ) * (t * t)
    if hard_labels is None:
        return soft
    hard = F.binary_cross_entropy_with_logits(
        student_logits,
        torch.as_tensor(hard_labels, dtype=student_logits.dtype,
                        device=student_logits.device),
    )
    return alpha * soft + (1.0 - alpha) * hard
