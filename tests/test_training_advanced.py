"""Pure-logic tests for the surviving TASK B helpers (src/training/advanced.py).

No GPU, no training run: every helper is exercised directly. The pruned API
contains ONLY helpers with live consumers (EMA, calibration, focal, SWA
averaging, scheduler menu/translation, GPU telemetry parsing).
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
    assert temperature > 1.0
    assert temperature < 10.0


def test_temperature_one_is_identity():
    logits = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(adv.apply_temperature(logits, 1.0), 1.0 / (1.0 + np.exp(-logits)))


def test_expected_calibration_error_exact_and_zero_when_perfect():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert adv.expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0)
    probs = np.full(4, 0.95)
    labels = np.array([1, 1, 0, 0])
    assert adv.expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.45)


def test_brier_score_exact():
    assert adv.brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert adv.brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_reliability_diagram_payload_shape():
    diagram = adv.reliability_diagram(
        np.array([0.1, 0.4, 0.6, 0.9]), np.array([0, 0, 1, 1]), n_bins=4
    )
    assert diagram["n"] == 4
    assert len(diagram["bins"]) == 4
    assert set(diagram) >= {"ece", "brier", "temperature", "bins"}


def test_fit_temperature_rejects_empty_and_mismatched():
    with pytest.raises(adv.AdvancedConfigError):
        adv.fit_temperature([], [])
    with pytest.raises(adv.AdvancedConfigError):
        adv.fit_temperature([0.1], [1, 0])


def test_calibration_report_fits_on_dev_only():
    rng = np.random.default_rng(1)
    dev = rng.normal(size=500)
    dev_y = (dev > 0).astype(int)
    test = rng.normal(size=500)
    test_y = (test > 0).astype(int)
    report = adv.calibration_report(dev, dev_y, test, test_y, n_bins=5)
    assert set(report) >= {"temperature", "dev_ece", "test_ece", "test_brier", "reliability"}
    assert report["reliability"]["temperature"] == report["temperature"]
    report2 = adv.calibration_report(dev, dev_y, test * 10.0, test_y, n_bins=5)
    assert report2["temperature"] == pytest.approx(report["temperature"])


def test_calibration_report_handles_empty_dev():
    report = adv.calibration_report([], [], [0.2, 0.8], [0, 1])
    assert report["temperature"] == 1.0
    assert math.isnan(report["dev_ece"])


# ── focal loss ──────────────────────────────────────────────────────────────

def test_focal_gamma_zero_equals_bce():
    logits = torch.tensor([0.5, -1.0, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    focal = adv.sigmoid_focal_bce_with_logits(logits, targets, gamma=0.0)
    assert torch.allclose(plain, focal)


def test_focal_downweights_easy_examples():
    logits = torch.tensor([5.0, -5.0])
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


# ── scheduler menu (one registry, HF translation) ───────────────────────────

def test_scheduler_menu_is_hf_translatable():
    for name in adv.LR_SCHEDULERS:
        assert adv.validate_lr_scheduler(name) == name
        assert adv.hf_scheduler_type(name) in {
            "linear", "cosine", "constant", "reduce_lr_on_plateau",
        }
    assert adv.hf_scheduler_type("plateau") == "reduce_lr_on_plateau"
    assert "one_cycle" not in adv.LR_SCHEDULERS


def test_unknown_scheduler_rejected_by_both_apis():
    with pytest.raises(adv.AdvancedConfigError):
        adv.validate_lr_scheduler("exponential")
    with pytest.raises(ValueError):
        adv.hf_scheduler_type("warp-speed")


# ── SWA averaging ───────────────────────────────────────────────────────────

def test_average_state_dicts_equal_weights():
    a = {"w": np.array([0.0, 2.0]), "count": np.array(5, dtype=np.int64)}
    b = {"w": np.array([2.0, 4.0]), "count": np.array(9, dtype=np.int64)}
    avg = adv.average_state_dicts([a, b])
    assert np.allclose(avg["w"], [1.0, 3.0])
    assert int(avg["count"]) == 5


def test_average_state_dicts_weighted():
    avg = adv.average_state_dicts(
        [{"w": np.array([0.0])}, {"w": np.array([10.0])}], weights=[3.0, 1.0]
    )
    assert avg["w"][0] == pytest.approx(2.5)


def test_average_state_dicts_rejects_mismatch_and_empty():
    with pytest.raises(adv.AdvancedConfigError):
        adv.average_state_dicts([])
    with pytest.raises(adv.AdvancedConfigError):
        adv.average_state_dicts([{"a": np.zeros(1)}, {"b": np.zeros(1)}])


def test_average_state_dicts_torch_tensors():
    avg = adv.average_state_dicts(
        [{"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([3.0, 5.0])}]
    )
    assert torch.allclose(avg["w"], torch.tensor([2.0, 4.0]))


# ── GPU telemetry parsing (shared core query) ───────────────────────────────

def test_parse_gpu_query_maps_shared_fields_by_position():
    row = "2026/01/01 00:00:00, 0, 42, 30, 1200, 16000, 70.5, 65, 1410, 5001"
    parsed = adv.parse_gpu_query(row)
    assert parsed["gpu_index"] == 0.0
    assert parsed["gpu_util_pct"] == 42.0
    assert parsed["gpu_memory_used_mb"] == 1200.0
    assert parsed["power_w"] == pytest.approx(70.5)
    assert parsed["sm_clock_mhz"] == 1410.0
    assert parsed["mem_clock_mhz"] == 5001.0
    assert "timestamp" not in parsed  # non-numeric column skipped


def test_parse_gpu_query_drops_na_and_empty():
    assert adv.parse_gpu_query("") == {}
    parsed = adv.parse_gpu_query("2026/01/01, 0, 52, N/A, N/A, N/A, N/A, 60, 900, 800")
    assert parsed["gpu_util_pct"] == 52.0
    assert parsed["sm_clock_mhz"] == 900.0
    assert "power_w" not in parsed


def test_collect_telemetry_never_raises():
    assert isinstance(adv.collect_nvml_telemetry(), dict)
