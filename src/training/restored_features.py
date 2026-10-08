"""Owner-requested features KEPT and wired (restored after the 2026-10-08 audit).

These seven features were briefly deleted as "dead dials"; the owner wants them
kept and WORKING. Every class here has a live consumer in the text or GNN lane
(see ``config/training.yaml`` dated notes). GPU-free public-API tests live in
``tests/test_restored_features.py``.

Modules:
  * FGM adversarial perturbation  -> GNN train step (train-only)
  * pair-difficulty curriculum    -> text FrozenBatchSampler batch order
  * distillation (teacher/student) -> GNN scorer vs text-embedding teacher
  * checkpoint SWA                -> text post-train averaging + re-eval
  * embedding ensemble            -> text publish-time fold/seed averaging
  * cross-encoder rerank tuning   -> src/training/rerank.py stage 2
  * weight EMA                    -> advanced.EmaTracker + text _WeightEmaCallback
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "unknown": 1}


class RestoredConfigError(ValueError):
    """Raised when a restored-feature knob is internally inconsistent."""


# ─────────────────────────────────────────────────────────────────────────────
# FGM / adversarial (train-only embedding perturbation)
# ─────────────────────────────────────────────────────────────────────────────


class FGMAttack:
    """Fast Gradient Method: ``x + epsilon * g/||g||`` (l2) or ``sign(g)``."""

    def __init__(self, epsilon: float = 0.0, norm: str = "l2"):
        if epsilon < 0.0:
            raise RestoredConfigError("adversarial.epsilon must be >= 0")
        if norm not in ("l2", "linf"):
            raise RestoredConfigError(f"adversarial.norm must be l2|linf, got {norm!r}")
        self.epsilon = float(epsilon)
        self.norm = norm

    @property
    def enabled(self) -> bool:
        return self.epsilon > 0.0

    def perturb(self, x, grad):
        """Return the adversarial tensor (detached from the graph)."""
        import torch

        if not self.enabled:
            return x
        g = torch.zeros_like(x) if grad is None else grad
        if self.norm == "l2":
            step = g / (g.norm(p=2) + 1e-12)
        else:
            step = g.sign()
        return (x.detach() + self.epsilon * step).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum on pair/batch difficulty
# ─────────────────────────────────────────────────────────────────────────────


def difficulty_rank(value: Any) -> float:
    """Map a difficulty label/score to an ascending comparable rank."""
    if isinstance(value, str):
        return float(_DIFFICULTY_ORDER.get(value, 1))
    return float(value)


class CurriculumPlanner:
    """Epoch-wise easy->hard (or hard->easy) BATCH presentation order.

    Difficulty is computed per frozen batch; the plan reorders whole batches
    (composition is never touched, so the frozen batch contract holds) and
    grows the exposed set from ``warmup_fraction`` to 100% across epochs.
    """

    def __init__(self, schedule: str = "easy_to_hard", warmup_fraction: float = 0.0):
        if schedule not in ("easy_to_hard", "hard_to_easy"):
            raise RestoredConfigError(
                f"curriculum.schedule must be easy_to_hard|hard_to_easy, got {schedule!r}"
            )
        if not 0.0 <= warmup_fraction < 1.0:
            raise RestoredConfigError("curriculum.warmup_fraction must be in [0, 1)")
        self.schedule = schedule
        self.warmup_fraction = float(warmup_fraction)

    def order_batches(
        self, batch_difficulty: Sequence[Any], epochs: int
    ) -> list[list[int]]:
        if epochs < 1:
            raise RestoredConfigError("curriculum epochs must be >= 1")
        n = len(batch_difficulty)
        if n == 0:
            return [[] for _ in range(epochs)]
        order = sorted(range(n), key=lambda i: difficulty_rank(batch_difficulty[i]))
        if self.schedule == "hard_to_easy":
            order = order[::-1]
        start = max(1, int(math.ceil(self.warmup_fraction * n)))
        plan: list[list[int]] = []
        for epoch in range(1, epochs + 1):
            if epochs == 1:
                count = n
            else:
                progress = (epoch - 1) / (epochs - 1)
                count = int(round(start + (n - start) * progress))
            plan.append(list(order[: min(n, max(start, count))]))
        return plan


# ─────────────────────────────────────────────────────────────────────────────
# Distillation (teacher -> student)
# ─────────────────────────────────────────────────────────────────────────────


class DistillationRegularizer:
    """KD term: ``alpha * soft(student||teacher) + (1-alpha) * hard_BCE``.

    Teacher logits are detached (no gradient path into the teacher). The
    teacher is supplied by the caller and must be trained on the train split
    only.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.0,
                 teacher_model: str | None = None):
        if temperature <= 0.0:
            raise RestoredConfigError("distillation.temperature must be > 0")
        if not 0.0 <= alpha <= 1.0:
            raise RestoredConfigError("distillation.alpha must be in [0, 1]")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_model = teacher_model

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    def loss(self, student_logits, teacher_logits, hard_labels=None):
        import torch
        import torch.nn.functional as F

        teacher = torch.as_tensor(teacher_logits).detach().to(student_logits.device)
        t = self.temperature
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
        return self.alpha * soft + (1.0 - self.alpha) * hard


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint SWA (text post-train) + embedding ensemble (publish)
# ─────────────────────────────────────────────────────────────────────────────


def average_embeddings(
    embeddings: Iterable[np.ndarray], *, normalize: bool = True
) -> np.ndarray:
    """Average dense embeddings across folds/seeds, then optionally L2-norm."""
    arrays = [np.asarray(e, dtype=np.float64) for e in embeddings]
    if not arrays:
        raise RestoredConfigError("cannot average zero embedding sets")
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise RestoredConfigError(f"embedding shapes disagree: {sorted(shapes)}")
    mean = np.mean(np.stack(arrays, axis=0), axis=0)
    if normalize:
        norms = np.linalg.norm(mean, axis=1, keepdims=True)
        mean = mean / np.clip(norms, 1e-12, None)
    return mean


class EmbeddingEnsemble:
    """Accumulates fold/seed embedding matrices and produces the ensemble."""

    def __init__(self, normalize: bool = True):
        self.normalize = bool(normalize)
        self._parts: list[np.ndarray] = []

    def add(self, embeddings: np.ndarray) -> None:
        self._parts.append(np.asarray(embeddings, dtype=np.float64))

    def __len__(self) -> int:
        return len(self._parts)

    def combine(self) -> np.ndarray:
        return average_embeddings(self._parts, normalize=self.normalize)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder rerank tuning
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RerankTuner:
    """Tuning knobs + a thin fit wrapper for the stage-2 cross-encoder."""

    learning_rate: float = 2.0e-5
    epochs: int = 3
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise RestoredConfigError("rerank.learning_rate must be > 0")
        if self.epochs < 1:
            raise RestoredConfigError("rerank.epochs must be >= 1")
        if self.batch_size < 1:
            raise RestoredConfigError("rerank.batch_size must be >= 1")

    def fit(self, cross_encoder, pairs: Sequence[tuple[str, str]], labels: Sequence[int]):
        """Fine-tune the cross-encoder on labelled pairs (train split only)."""
        if len(pairs) != len(labels):
            raise RestoredConfigError("rerank tuning pairs/labels length mismatch")
        if not pairs:
            raise RestoredConfigError("rerank tuning needs at least one pair")
        cross_encoder.fit(
            train_dataloader=[
                ([list(pair), float(label)] for pair, label in zip(pairs, labels))
            ],
            epochs=self.epochs,
            warmup_steps=max(1, len(pairs) // self.batch_size // 10),
            optimizer_params={"lr": self.learning_rate},
            show_progress_bar=False,
        )
        return {"pairs": len(pairs), "epochs": self.epochs,
                "learning_rate": self.learning_rate}


# Back-compat pure functions (used by tests and, where noted, the wiring).
def adversarial_perturbation(x, grad, epsilon: float, *, norm: str = "l2"):
    return FGMAttack(epsilon, norm).perturb(x, grad)


def curriculum_plan(difficulty: Sequence[Any], epochs: int, *,
                    schedule: str = "easy_to_hard", warmup_fraction: float = 0.0):
    return CurriculumPlanner(schedule, warmup_fraction).order_batches(difficulty, epochs)


def distillation_loss(student_logits, teacher_logits, *, temperature: float = 2.0,
                      alpha: float = 0.5, hard_labels=None):
    return DistillationRegularizer(temperature, alpha).loss(
        student_logits, teacher_logits, hard_labels
    )
