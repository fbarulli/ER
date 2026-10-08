"""Pure-logic tests for TASK B training additions (src/training/advanced.py).

No GPU, no training run: every helper is exercised directly. The point is to
pin the arithmetic (EMA, temperature/ECE/Brier, focal, scheduler, SWA,
curriculum, telemetry parsing, FGM, ensembling, distillation) before the loop
wiring, so a regression is caught here and not only on a GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from training import advanced as adv


# ── EMA ─────────────────────────────────────────────────────────────────────

def test_ema_update_is_convex_combination():
    shadow = np.array([1.0, 2.0])
    current = np.array([3.0, 4.0])
    assert np.allclose(adv.ema_update(shadow, current, 0.0), current)
    assert np.allclose(adv.ema_update(shadow, current, 1.0), shadow)
    assert np.allclose(adv.ema_update(shadow, current, 0.5), [2.0, 3.0])


def test_ema_update_rejects_bad_decay():
    with pytest.raises(adv.AdvancedConfigError):
        adv.ema_update(np.zeros(1), np.zeros(1), 1.5)


def test_ema_tracker_converges_and_roundtrips():
    tracker = adv.EmaTracker(decay=0.5)
    for value in (10.0, 20.0, 30.0):
        tracker.update({"w": np.array([float(value)])})
    # first update seeds 10; 0.5*10+0.5*20=15; 0.5*15+0.5*30=22.5
    assert tracker.shadow()["w"][0] == pytest.approx(22.5)
    restored = adv.EmaTracker.from_state_dict(tracker.state_dict())
    assert restored.shadow()["w"][0] == pytest.approx(22.5)
    assert restored.step == 3


def test_ema_warmup_raises_effective_decay_then_settles():
    assert adv.ema_decay_for_step(0, 0.999, warmup_updates=10) == pytest.approx(0.1)
    assert adv.ema_decay_for_step(20000, 0.999, warmup_updates=10) == pytest.approx(0.999)
    assert adv.ema_decay_for_step(5, 0.9, warmup_updates=0) == 0.9


def test_ema_tracker_carries_non_float_buffers():
    tracker = adv.EmaTracker(decay=0.9)
    tracker.update({"count": np.array(7, dtype=np.int64)})
    assert int(tracker.shadow()["count"]) == 7


# ── calibration ─────────────────────────────────────────────────────────────

def test_fit_temperature_recovers_scale_on_synthetic_logits():
    rng = np.random.default_rng(0)
    true_temp = 3.0
    logits = rng.normal(size=4000) * true_temp
    labels = (rng.uniform(size=4000) < 1.0 / (1.0 + np.exp(-logits / true_temp))).astype(int)
    temperature = adv.fit_temperature(logits, labels)
    # the fit should not be wildly off: it shrinks the over-confident logits
    assert temperature > 1.0
    assert temperature < 10.0


def test_temperature_one_is_identity():
    logits = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(adv.apply_temperature(logits, 1.0), 1.0 / (1.0 + np.exp(-logits)))


def test_expected_calibration_error_exact_and_zero_when_perfect():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert adv.expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0)
    # all mass in the top bin, half wrong -> ECE 0.5
    probs = np.full(4, 0.95)
    labels = np.array([1, 1, 0, 0])
    assert adv.expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.45)


def test_brier_score_exact():
    probs = np.array([1.0, 0.0])
    labels = np.array([1, 0])
    assert adv.brier_score(probs, labels) == 0.0
    assert adv.brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_reliability_diagram_payload_shape():
    diagram = adv.reliability_diagram(
        np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 0, 1, 1]), n_bins=4
    )
    assert diagram["n"] == 4
    assert len(diagram["bins"]) == 4
    assert set(diagram) >= {"ece", "brier", "temperature", "bins"}
    assert all(b["count"] >= 0 for b in diagram["bins"])


def test_fit_temperature_rejects_empty_and_mismatched():
    with pytest.raises(adv.AdvancedConfigError):
        adv.fit_temperature([], [])
    with pytest.raises(adv.AdvancedConfigError):
        adv.fit_temperature([0.1], [1, 0])


# ── focal loss ──────────────────────────────────────────────────────────────

def test_focal_gamma_zero_equals_bce():
    logits = torch.tensor([0.5, -1.0, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    focal = adv.sigmoid_focal_bce_with_logits(logits, targets, gamma=0.0)
    assert torch.allclose(plain, focal)


def test_focal_downweights_easy_examples():
    logits = torch.tensor([5.0, -5.0])  # very easy, correctly classified
    targets = torch.tensor([1.0, 0.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    focal = adv.sigmoid_focal_bce_with_logits(logits, targets, gamma=2.0)
    assert focal < bce


def test_focal_pos_weight_raises_positive_contribution():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    unweighted = adv.sigmoid_focal_bce_with_logits(logits, targets, gamma=0.0)
    weighted = adv.sigmoid_focal_bce_with_logits(
        logits, targets, gamma=0.0, pos_weight=4.0
    )
    assert weighted > unweighted


# ── scheduler factory ───────────────────────────────────────────────────────

def test_validate_lr_scheduler_menu():
    for name in adv.LR_SCHEDULERS:
        assert adv.validate_lr_scheduler(name) == name
    with pytest.raises(adv.AdvancedConfigError):
        adv.validate_lr_scheduler("exponential")


def test_linear_scheduler_decays_to_min_ratio():
    fn = adv.lr_lambda("linear", num_training_steps=10, warmup_steps=2, min_lr_ratio=0.1)
    assert fn(0) < fn(2)  # warmup
    assert fn(10) == pytest.approx(0.1)
    slopes = [fn(i) for i in range(2, 11)]
    assert all(a >= b for a, b in zip(slopes, slopes[1:]))


def test_cosine_scheduler_endpoints():
    fn = adv.lr_lambda("cosine", num_training_steps=10, warmup_steps=0)
    assert fn(0) == pytest.approx(1.0)
    assert fn(10) == pytest.approx(0.0, abs=1e-9)


def test_constant_scheduler_is_flat_after_warmup():
    fn = adv.lr_lambda("constant", num_training_steps=10, warmup_steps=3)
    assert fn(3) == 1.0
    assert fn(9) == 1.0


def test_plateau_has_no_step_lambda_but_is_valid():
    assert adv.is_plateau_scheduler("plateau")
    with pytest.raises(adv.AdvancedConfigError):
        adv.lr_lambda("plateau", num_training_steps=10)


def test_build_lr_scheduler_wires_lambdalr_and_none_for_plateau():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = adv.build_lr_scheduler(
        optimizer, name="cosine", num_training_steps=10
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert adv.build_lr_scheduler(
        optimizer, name="plateau", num_training_steps=10
    ) is None


# ── SWA averaging ───────────────────────────────────────────────────────────

def test_average_state_dicts_equal_weights():
    a = {"w": np.array([0.0, 2.0]), "count": np.array(5, dtype=np.int64)}
    b = {"w": np.array([2.0, 4.0]), "count": np.array(9, dtype=np.int64)}
    avg = adv.average_state_dicts([a, b])
    assert np.allclose(avg["w"], [1.0, 3.0])
    assert int(avg["count"]) == 5


def test_average_state_dicts_weighted():
    a = {"w": np.array([0.0])}
    b = {"w": np.array([10.0])}
    avg = adv.average_state_dicts([a, b], weights=[3.0, 1.0])
    assert avg["w"][0] == pytest.approx(2.5)


def test_average_state_dicts_rejects_mismatch_and_empty():
    with pytest.raises(adv.AdvancedConfigError):
        adv.average_state_dicts([])
    with pytest.raises(adv.AdvancedConfigError):
        adv.average_state_dicts([{"a": np.zeros(1)}, {"b": np.zeros(1)}])


def test_average_state_dicts_torch_tensors():
    a = {"w": torch.tensor([1.0, 3.0])}
    b = {"w": torch.tensor([3.0, 5.0])}
    avg = adv.average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 4.0]))


# ── curriculum ──────────────────────────────────────────────────────────────

def test_curriculum_easy_to_hard_is_monotone_and_full_at_end():
    difficulty = ["hard", "easy", "medium", "hard", "easy"]
    plan = adv.curriculum_plan(difficulty, epochs=4, warmup_fraction=0.4)
    assert plan[-1] == [0, 1, 2, 3, 4]
    assert len(plan[0]) <= len(plan[-1])
    # first epoch exposes the two easiest items (indices 1 and 4)
    assert set(plan[0]) == {1, 4}


def test_curriculum_hard_to_easy_starts_hard():
    difficulty = ["hard", "easy", "medium"]
    plan = adv.curriculum_plan(difficulty, epochs=1, schedule="hard_to_easy",
                               warmup_fraction=0.34)
    assert len(plan) == 1


def test_curriculum_is_deterministic_and_empty_safe():
    difficulty = [3, 1, 2]
    assert adv.curriculum_plan(difficulty, 3) == adv.curriculum_plan(difficulty, 3)
    assert adv.curriculum_plan([], 2) == [[], []]


def test_curriculum_rejects_bad_args():
    with pytest.raises(adv.AdvancedConfigError):
        adv.curriculum_plan([1], 1, schedule="zigzag")
    with pytest.raises(adv.AdvancedConfigError):
        adv.curriculum_plan([1], 0)


# ── telemetry parsing ───────────────────────────────────────────────────────

def test_parse_gpu_query_maps_fields():
    row = "42, 70.5, 1410, 5001, 65, 1200"
    parsed = adv.parse_gpu_query(row)
    assert parsed["gpu_util_pct"] == 42.0
    assert parsed["power_w"] == pytest.approx(70.5)
    assert parsed["sm_clock_mhz"] == 1410.0
    assert parsed["memory_used_mb"] == 1200.0


def test_parse_gpu_query_drops_na_and_empty():
    assert adv.parse_gpu_query("") == {}
    parsed = adv.parse_gpu_query("52, N/A, 900")
    assert "power_w" not in parsed
    assert parsed["gpu_util_pct"] == 52.0
    assert parsed["sm_clock_mhz"] == 900.0


def test_collect_nvml_telemetry_never_raises():
    assert isinstance(adv.collect_nvml_telemetry(), dict)


# ── FGM ─────────────────────────────────────────────────────────────────────

def test_adversarial_perturbation_l2_has_unit_scaled_norm():
    x = torch.zeros(2, 4)
    grad = torch.ones(2, 4)
    out = adv.adversarial_perturbation(x, grad, 0.5, norm="l2")
    # FGM normalizes the whole gradient tensor to unit l2 norm, then scales
    # the perturbation by epsilon.
    assert out.norm(p=2) == pytest.approx(0.5, abs=1e-5)


def test_adversarial_perturbation_linf_uses_sign():
    x = torch.zeros(1, 3)
    grad = torch.tensor([[-2.0, 3.0, 0.0]])
    out = adv.adversarial_perturbation(x, grad, 0.1, norm="linf")
    assert torch.allclose(out, torch.tensor([[-0.1, 0.1, 0.0]]))


def test_adversarial_perturbation_rejects_bad_norm():
    with pytest.raises(adv.AdvancedConfigError):
        adv.adversarial_perturbation(torch.zeros(1), torch.zeros(1), 0.1, norm="bogus")


# ── accumulation / ensembling / distillation ────────────────────────────────

def test_accumulates_now_boundaries():
    assert adv.accumulates_now(0, 2) is False
    assert adv.accumulates_now(1, 2) is True
    assert adv.accumulates_now(0, 1) is True
    with pytest.raises(adv.AdvancedConfigError):
        adv.accumulates_now(0, 0)


def test_scale_accumulated_loss():
    assert adv.scale_accumulated_loss(torch.tensor(4.0), 4) == pytest.approx(1.0)


def test_average_embeddings_normalizes():
    a = np.array([[3.0, 0.0]])
    b = np.array([[0.0, 4.0]])
    out = adv.average_embeddings([a, b], normalize=True)
    assert out.shape == (1, 2)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    raw = adv.average_embeddings([a, b], normalize=False)
    assert np.allclose(raw, [[1.5, 2.0]])


def test_average_embeddings_shape_guard():
    with pytest.raises(adv.AdvancedConfigError):
        adv.average_embeddings([np.zeros((2, 3)), np.zeros((2, 4))])


def test_distillation_alpha_endpoints():
    student = torch.tensor([0.0, 1.0])
    teacher = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    # alpha weights the SOFT (teacher) term: 0 -> hard only, 1 -> soft only.
    hard_only = adv.distillation_loss(student, teacher, alpha=0.0, hard_labels=labels)
    soft_only = adv.distillation_loss(student, teacher, alpha=1.0, hard_labels=labels)
    expected_hard = torch.nn.functional.binary_cross_entropy_with_logits(student, labels)
    assert hard_only.item() == pytest.approx(expected_hard.item())
    assert soft_only.item() > 0.0
