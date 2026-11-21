"""TASK B config wiring: dials map to real consumers, are enabled for the
GPU run, and the schema validates nonsense.

These pin the SSOT contract — the shipped config enables the improved advanced
GNN for the GPU run, no unread dial may ship, and the schema keys reject
nonsense at load.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.common import training_cfg
from core.schemas import AdvancedSpec, HpoSpaceSpec, LR_SCHEDULERS, TrainingSpec


def test_advanced_features_are_enabled_for_the_gpu_run():
    advanced = training_cfg().advanced
    assert advanced.calibration.enabled is True
    assert advanced.accel.tf32 is True
    assert advanced.accel.compile is True
    assert advanced.telemetry.nvml is True
    assert advanced.ema.enabled is True  # ONE ema home (text + GNN)
    assert advanced.swa.enabled is True  # ONE swa home (text + GNN)
    assert advanced.adversarial.enabled is True
    assert advanced.curriculum.enabled is True
    assert advanced.distillation.enabled is True
    assert advanced.rerank.enabled is True
    assert advanced.embedding_ensemble.enabled is True
    assert advanced.graph.calibration.enabled is True
    assert advanced.graph.focal.enabled is True
    assert advanced.graph.arch.two_hop is True


def test_all_advanced_fields_have_a_live_consumer():
    """Every advanced.* field is wired (owner-requested features kept 2026-10-09)."""
    fields = set(AdvancedSpec.model_fields)
    assert fields == {
        "ema", "swa", "adversarial", "curriculum", "distillation", "rerank",
        "embedding_ensemble", "calibration", "accel", "telemetry",
        "gradient_accumulation_steps", "graph",
    }


def test_gradient_accumulation_defaults_to_one():
    assert training_cfg().advanced.gradient_accumulation_steps == 1


def test_hpo_scheduler_is_not_swept_by_default():
    assert training_cfg().hpo.tpe_space.lr_scheduler is None


def test_advanced_spec_is_additive_with_defaults():
    spec = AdvancedSpec()
    assert spec.gradient_accumulation_steps == 1
    assert spec.calibration.max_temperature > spec.calibration.min_temperature


def test_calibration_temperature_bounds_are_validated():
    with pytest.raises(ValidationError):
        AdvancedSpec.model_validate(
            {"calibration": {"min_temperature": 10.0, "max_temperature": 5.0}}
        )


def _space(**overrides):
    base = {
        "epochs": (1, 2), "lr": (1e-5, 1e-4), "warmup_ratio": (0.0, 0.1),
        "weight_decay": (0.0, 0.1), "negative_mask_frac": (0.0, 0.5),
        "uniformity_weight": (0.0, 0.05), "lr_scheduler": None,
    }
    base.update(overrides)
    return HpoSpaceSpec.model_validate(base)


def test_hpo_scheduler_choices_are_validated():
    assert _space(lr_scheduler=["linear", "plateau"]).lr_scheduler == ["linear", "plateau"]
    with pytest.raises(ValidationError):
        _space(lr_scheduler=["warp-speed"])
    with pytest.raises(ValidationError):
        _space(lr_scheduler=[])


def test_lr_scheduler_enum_is_the_single_hf_menu():
    # ONE menu: LrScheduler Literal -> LR_SCHEDULERS, and no one_cycle (HF's
    # Trainer cannot schedule it through lr_scheduler_type).
    assert set(LR_SCHEDULERS) == {"linear", "cosine", "constant", "plateau"}
    field = TrainingSpec.model_fields["lr_scheduler"]
    assert set(field.annotation.__args__) == set(LR_SCHEDULERS)


def test_hpo_space_dict_excludes_the_scheduler_key():
    import training.training as training_module

    assert "lr_scheduler" not in training_module.HPO_SPACE
    assert training_module.HPO_SCHEDULERS is None
