"""Public-API, GPU-free tests for the restored owner-requested features.

Seven features were briefly deleted (commit 374e712); they are kept and wired.
These tests pin each component's contract so the wiring call sites are safe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training import restored_features as rf


# ── FGM / adversarial ───────────────────────────────────────────────────────

def test_fgm_disabled_is_identity():
    attack = rf.FGMAttack(0.0, "l2")
    assert attack.enabled is False
    x = torch.zeros(2, 3)
    assert torch.equal(attack.perturb(x, torch.ones_like(x)), x)


def test_fgm_l2_global_norm_is_epsilon():
    attack = rf.FGMAttack(0.5, "l2")
    x = torch.zeros(2, 4)
    out = attack.perturb(x, torch.ones(2, 4))
    assert out.norm(p=2) == pytest.approx(0.5, abs=1e-5)


def test_fgm_linf_uses_sign():
    attack = rf.FGMAttack(0.1, "linf")
    grad = torch.tensor([[-2.0, 3.0, 0.0]])
    out = attack.perturb(torch.zeros(1, 3), grad)
    assert torch.allclose(out, torch.tensor([[-0.1, 0.1, 0.0]]))


def test_fgm_rejects_bad_norm_and_negative_epsilon():
    with pytest.raises(rf.RestoredConfigError):
        rf.FGMAttack(0.1, "bogus")
    with pytest.raises(rf.RestoredConfigError):
        rf.FGMAttack(-0.1, "l2")


# ── curriculum ──────────────────────────────────────────────────────────────

def test_curriculum_easy_to_hard_order_is_deterministic():
    planner = rf.CurriculumPlanner("easy_to_hard", warmup_fraction=0.4)
    plan = planner.order_batches(["hard", "easy", "medium", "hard", "easy"], epochs=4)
    # full difficulty order (easy -> hard): indices 1,4 (easy), 2 (medium), 0,3 (hard)
    assert plan[-1] == [1, 4, 2, 0, 3]
    assert set(plan[0]) == {1, 4}
    assert plan == planner.order_batches(
        ["hard", "easy", "medium", "hard", "easy"], epochs=4
    )


def test_curriculum_hard_to_easy_and_rejections():
    planner = rf.CurriculumPlanner("hard_to_easy", warmup_fraction=0.0)
    plan = planner.order_batches([1, 2, 3], epochs=1)
    assert plan[0] == [2, 1, 0]
    with pytest.raises(rf.RestoredConfigError):
        rf.CurriculumPlanner("zigzag", 0.0)
    with pytest.raises(rf.RestoredConfigError):
        rf.CurriculumPlanner("easy_to_hard", 1.0)


def test_difficulty_rank_and_backcompat_plan():
    assert rf.difficulty_rank("easy") < rf.difficulty_rank("medium")
    assert rf.difficulty_rank("hard") > rf.difficulty_rank("medium")
    assert rf.curriculum_plan([3, 1, 2], 2) == rf.curriculum_plan([3, 1, 2], 2)


# ── distillation ────────────────────────────────────────────────────────────

def test_distillation_soft_only_and_hard_blend():
    reg = rf.DistillationRegularizer(temperature=2.0, alpha=1.0)
    student = torch.tensor([0.0, 1.0])
    teacher = torch.tensor([2.0, -2.0])
    soft = reg.loss(student, teacher)
    assert soft.item() > 0.0
    blended = rf.DistillationRegularizer(2.0, 0.5).loss(
        student, teacher, torch.tensor([1.0, 0.0])
    )
    assert blended.item() > 0.0
    assert torch.equal(teacher, torch.tensor([2.0, -2.0]))  # teacher untouched


def test_distillation_disabled_and_rejections():
    reg = rf.DistillationRegularizer(2.0, 0.0)
    assert reg.enabled is False
    with pytest.raises(rf.RestoredConfigError):
        rf.DistillationRegularizer(0.0, 0.5)
    with pytest.raises(rf.RestoredConfigError):
        rf.DistillationRegularizer(2.0, 1.5)


# ── embedding ensemble ──────────────────────────────────────────────────────

def test_embedding_ensemble_combines_and_normalizes():
    ensemble = rf.EmbeddingEnsemble(normalize=True)
    assert len(ensemble) == 0
    ensemble.add(np.array([[3.0, 0.0]]))
    ensemble.add(np.array([[0.0, 4.0]]))
    assert len(ensemble) == 2
    out = ensemble.combine()
    assert out.shape == (1, 2)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    raw = rf.average_embeddings(
        [np.array([[3.0, 0.0]]), np.array([[0.0, 4.0]])], normalize=False
    )
    assert np.allclose(raw, [[1.5, 2.0]])


def test_embedding_ensemble_shape_guard():
    with pytest.raises(rf.RestoredConfigError):
        rf.EmbeddingEnsemble().combine()  # empty
    with pytest.raises(rf.RestoredConfigError):
        rf.average_embeddings([np.zeros((2, 3)), np.zeros((2, 4))])


# ── rerank tuner ────────────────────────────────────────────────────────────

class _FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def fit(self, **kwargs):
        self.calls.append(kwargs)


def test_rerank_tuner_validates_and_fits():
    with pytest.raises(rf.RestoredConfigError):
        rf.RerankTuner(learning_rate=0.0)
    with pytest.raises(rf.RestoredConfigError):
        rf.RerankTuner(epochs=0)
    tuner = rf.RerankTuner(learning_rate=1e-5, epochs=2, batch_size=4)
    ce = _FakeCrossEncoder()
    report = tuner.fit(ce, [("a", "b"), ("c", "d")], [1, 0])
    assert report["pairs"] == 2
    assert ce.calls and ce.calls[0]["epochs"] == 2
    assert ce.calls[0]["optimizer_params"]["lr"] == 1e-5
    with pytest.raises(rf.RestoredConfigError):
        tuner.fit(ce, [("a", "b")], [1, 0])
    with pytest.raises(rf.RestoredConfigError):
        tuner.fit(ce, [], [])


# ── config wiring (all restored knobs default OFF, with dated keep-notes) ────

def test_restored_knobs_default_off_and_shared_homes():
    from core.common import training_cfg

    advanced = training_cfg().advanced
    assert advanced.ema.enabled is False
    assert advanced.adversarial.enabled is False
    assert advanced.curriculum.enabled is False
    assert advanced.distillation.enabled is False
    assert advanced.swa.enabled is False
    assert advanced.rerank.enabled is False
    assert advanced.embedding_ensemble.enabled is False
    # ONE home each: ema/swa live only at the top level, not under graph.
    assert "ema" not in advanced.graph.model_fields
    assert "swa" not in advanced.graph.model_fields
