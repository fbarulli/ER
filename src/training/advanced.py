"""Config-gated training enhancements — pure, testable logic.

Every public helper here is a small, side-effect-free function (or a tiny
stateful tracker) so it can be unit-tested without a GPU and so the training
loops only wire it behind an ``advanced.*`` config gate. The module must stay
importable without torch: torch is imported lazily inside the few functions
that need it.

The ``advanced.*`` config block is deliberately limited to dials that HAVE a
live consumer:

  * ``advanced.calibration``        -> text fold metrics + reliability artifact
  * ``advanced.accel``              -> text TF32 / torch.compile
  * ``advanced.telemetry``          -> text NVML telemetry
  * ``advanced.gradient_accumulation_steps`` -> text STArgs
  * ``advanced.graph.{ema,calibration,focal,swa,arch,telemetry}`` -> GNN lane

Leakage discipline: the calibration helpers fit on the dev/calibration carve
ONLY. Nothing here reads a test split.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from core.gpu_execution import gpu_query_string, parse_gpu_query
from core.schemas import LR_SCHEDULERS, hf_scheduler_type

__all__ = [
    "AdvancedConfigError",
    "LR_SCHEDULERS",
    "hf_scheduler_type",
    "validate_lr_scheduler",
    "ema_update",
    "ema_decay_for_step",
    "EmaTracker",
    "apply_temperature",
    "fit_temperature",
    "expected_calibration_error",
    "brier_score",
    "reliability_diagram",
    "calibration_report",
    "sigmoid_focal_bce_with_logits",
    "average_state_dicts",
    "parse_gpu_query",
    "collect_nvml_telemetry",
]


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

    Device-agnostic and float-only: non-floating entries (integer buffers such
    as ``num_batches_tracked``) are carried through unchanged. ``step`` drives
    the warmup decay, so a resumed run reloads the step count from the
    checkpoint manifest for reproducibility.
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
        raise AdvancedConfigError(f"temperature must be > 0, got {temperature}")
    return _sigmoid(np.asarray(logits, dtype=np.float64) / float(temperature))


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = np.clip(apply_temperature(logits, temperature), 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    )


def fit_temperature(
    logits: Sequence[float],
    labels: Sequence[int],
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 50.0,
) -> float:
    """Fit ONE scalar temperature on DEV logits/labels (never test).

    Minimizes binary NLL over ``T``. ``T == 1.0`` means the model was already
    calibrated.
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
        # Deterministic fallback: log-spaced grid, no scipy.
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
        ece += (count / probs.size) * abs(
            float(probs[mask].mean()) - float(labels[mask].mean())
        )
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
    """Reliability-diagram artifact payload (JSON-serializable)."""
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


def calibration_report(
    dev_scores: Sequence[float],
    dev_labels: Sequence[int],
    test_scores: Sequence[float],
    test_labels: Sequence[int],
    *,
    n_bins: int = 15,
    min_temperature: float = 0.05,
    max_temperature: float = 50.0,
    fit: bool = True,
) -> dict[str, Any]:
    """Fit a temperature on DEV and report ECE/Brier on DEV and TEST.

    Leakage contract: the temperature is fitted on the dev/calibration carve
    ONLY. Test scores are only scored with the dev-fitted temperature.
    """
    dev_scores = np.asarray(dev_scores, dtype=np.float64).reshape(-1)
    dev_labels = np.asarray(dev_labels, dtype=np.float64).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    test_labels = np.asarray(test_labels, dtype=np.float64).reshape(-1)
    temperature = 1.0
    if fit and dev_scores.size:
        temperature = fit_temperature(
            dev_scores,
            dev_labels,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
        )
    dev_probs = (
        apply_temperature(dev_scores, temperature) if dev_scores.size else dev_scores
    )
    test_probs = (
        apply_temperature(test_scores, temperature) if test_scores.size else test_scores
    )
    return {
        "temperature": float(temperature),
        "dev_ece": expected_calibration_error(dev_probs, dev_labels, n_bins=n_bins)
        if dev_scores.size
        else float("nan"),
        "dev_brier": brier_score(dev_probs, dev_labels)
        if dev_scores.size
        else float("nan"),
        "test_ece": expected_calibration_error(test_probs, test_labels, n_bins=n_bins)
        if test_scores.size
        else float("nan"),
        "test_brier": brier_score(test_probs, test_labels)
        if test_scores.size
        else float("nan"),
        "test_raw_ece": expected_calibration_error(test_scores, test_labels, n_bins=n_bins)
        if test_scores.size
        else float("nan"),
        "reliability": reliability_diagram(
            test_probs, test_labels, n_bins=n_bins, temperature=temperature
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Focal / class-weighted BCE (GNN scorer loss)
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
# SWA / top-k checkpoint averaging (GNN lane)
# ─────────────────────────────────────────────────────────────────────────────


def average_state_dicts(
    states: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Weighted element-wise average of flat parameter mappings.

    Floating tensors/arrays are averaged; non-floating entries (e.g.
    ``num_batches_tracked`` ints) are taken from the first state.
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
# GPU-utilization telemetry (nvidia-smi; shared query in core.gpu_execution)
# ─────────────────────────────────────────────────────────────────────────────


def collect_nvml_telemetry() -> dict[str, float]:
    """Best-effort one-shot GPU utilization/power/clocks for device 0.

    Shells out to ``nvidia-smi`` with the shared query in
    ``core.gpu_execution`` and parses with the shared parser. Returns ``{}``
    when nvidia-smi is unavailable — telemetry must never crash a run.
    """
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {}
    try:
        completed = subprocess.run(
            [
                binary,
                f"--query-gpu={gpu_query_string()}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return {}
    row = completed.stdout.strip().splitlines()
    return parse_gpu_query(row[0]) if row else {}
