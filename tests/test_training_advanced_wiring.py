"""TASK B config wiring: gates default OFF and new schema keys validate.

These pin the SSOT contract — the shipped config must NOT silently turn any
TASK B feature on (TASK A is the only explicit turn-on), and the new schema
keys must reject nonsense at load.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.common import training_cfg
from core.schemas import AdvancedSpec, HpoSpaceSpec, TrainingSpec


def test_all_advanced_features_default_off():
    advanced = training_cfg().advanced
    assert advanced.ema.enabled is False
    assert advanced.calibration.enabled is False
    assert advanced.adversarial.enabled is False
    assert advanced.swa.enabled is False
    assert advanced.curriculum.enabled is False
    assert advanced.accel.tf32 is False
    assert advanced.accel.compile is False
    assert advanced.telemetry.nvml is False
    assert advanced.focal.enabled is False
    assert advanced.rerank.enabled is False
    assert advanced.distillation.enabled is False
    assert advanced.embedding_ensemble.enabled is False
    assert advanced.graph.focal.enabled is False
    assert advanced.graph.arch.two_hop is False


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


def test_hpo_scheduler_choices_are_validated():
    ok = HpoSpaceSpec.model_validate(
        {"lr_scheduler": ["linear", "cosine"], "epochs": (1, 2), "lr": (1e-5, 1e-4),
         "warmup_ratio": (0.0, 0.1), "weight_decay": (0.0, 0.1),
         "negative_mask_frac": (0.0, 0.5), "uniformity_weight": (0.0, 0.05)}
    )
    assert ok.lr_scheduler == ["linear", "cosine"]
    with pytest.raises(ValidationError):
        HpoSpaceSpec.model_validate(
            {"lr_scheduler": ["warp-speed"], "epochs": (1, 2), "lr": (1e-5, 1e-4),
             "warmup_ratio": (0.0, 0.1), "weight_decay": (0.0, 0.1),
             "negative_mask_frac": (0.0, 0.5), "uniformity_weight": (0.0, 0.05)}
        )


def test_lr_scheduler_enum_rejects_unknown_value():
    field = TrainingSpec.model_fields["lr_scheduler"]
    literal_values = set(getattr(field.annotation, "__args__", ()))
    assert literal_values == {"linear", "cosine", "one_cycle", "plateau", "constant"}


def test_hpo_space_dict_excludes_the_scheduler_key():
    # HPO_SPACE must stay a pure lo/hi mapping; the categorical scheduler is
    # carried separately (HPO_SCHEDULERS) so the TPE layout is unchanged.
    import training.training as training_module

    assert "lr_scheduler" not in training_module.HPO_SPACE
    assert training_module.HPO_SCHEDULERS is None
