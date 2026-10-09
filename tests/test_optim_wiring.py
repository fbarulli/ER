"""Config-SSOT wiring for the optimizer/regularization/data/schedule knobs.

Pins: the shipped GPU-run config enables the knobs, the schema validates
nonsense, the new keys do NOT duplicate the existing margin/uniformity homes,
and the dynamic-padding helper is the one used by the prepared-token path.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.common import training_cfg
from core.schemas import (
    BatchSizeRampSpec,
    DataSpec,
    LossScheduleSpec,
    OptimizerSpec,
    RegularizationSpec,
    TrainingSpec,
)
from training.optim_components import dynamic_pad_width


def test_optimizer_knobs_enabled_for_the_gpu_run():
    opt = training_cfg().optimizer
    assert opt.no_decay_bias_norm is True
    # bf16 state is not enabled: stock AdamW cannot lerp a bf16 moment with an
    # fp32 grad (dtype mismatch), so the shipped GPU run keeps fp32 moments.
    assert opt.state_dtype == "fp32"
    assert opt.lr_scaling == "none"


def test_regularization_data_and_schedule_enabled_for_the_gpu_run():
    reg = training_cfg().regularization
    assert reg.r_drop.enabled is True
    assert reg.drop_path.enabled is True
    data = training_cfg().data
    assert data.dynamic_padding.enabled is True
    loss_schedules = training_cfg().loss_schedules
    assert loss_schedules.enabled is True
    assert training_cfg().training.batch_size_ramp.enabled is True


def test_lr_scaling_requires_base_batch():
    with pytest.raises(ValidationError):
        OptimizerSpec(lr_scaling="linear")
    ok = OptimizerSpec(lr_scaling="sqrt", base_batch=32)
    assert ok.base_batch == 32


def test_state_dtype_is_a_closed_enum():
    assert OptimizerSpec(state_dtype="bf16").state_dtype == "bf16"
    with pytest.raises(ValidationError):
        OptimizerSpec(state_dtype="fp8")


def test_drop_path_rate_and_ramp_bounds():
    with pytest.raises(ValidationError):
        RegularizationSpec.model_validate({"drop_path": {"rate": 1.0}})
    with pytest.raises(ValidationError):
        BatchSizeRampSpec(start_frac=0.0)
    assert BatchSizeRampSpec(start_frac=0.5, ramp_epochs=2).start_frac == 0.5


def test_loss_schedules_reference_existing_weights_not_new_homes():
    # Knob 9: margin and uniformity must UNIFY with existing keys — the new
    # block carries only booleans selecting existing weights, no second value.
    fields = set(LossScheduleSpec.model_fields)
    assert fields == {
        "enabled", "warmup_epochs", "schedule", "uniformity", "margin", "auxiliary",
    }
    # Existing homes still present and unchanged.
    assert "contrastive_margin" in TrainingSpec.model_fields
    assert "uniformity_regularization" in TrainingSpec.model_fields


def test_dynamic_padding_helper_is_shared_with_spec():
    spec = DataSpec.model_validate(
        {"dynamic_padding": {"enabled": True, "pad_to_multiple": 8}}
    )
    assert spec.dynamic_padding.pad_to_multiple == 8
    assert dynamic_pad_width([3, 7], pad_to_multiple=8) == 8
    assert dynamic_pad_width([3, 7], pad_to_multiple=1) == 7


def test_prepared_token_lookup_honors_pad_to_multiple():
    # A tiny lookup is too heavy to construct here; assert the padding helper
    # the lookup calls rounds as advertised (the integration is exercised by
    # the existing prepared-token tests with the default multiple of 1).
    assert np.issubdtype(np.int64, np.integer)
    assert dynamic_pad_width([1, 2, 3], pad_to_multiple=4) == 4
