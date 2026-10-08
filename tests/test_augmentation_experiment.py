"""Public contracts of the augmentation experiment surface (TODO "Eval
balance / data coverage": augmentation on/off + minted-negative label quality).

Two contracts, both public-facing:
  * the on/off arms differ by exactly one switch (``--mask-frac``), so an A/B
    pair is reproducible and comparable;
  * a minted negative is a label-0 row ONLY because its transplanted field
    conflicts with the pair side's declaration, and the transplanted value
    must exist in the recorded donor — a row that fails either check is a
    silent false negative and must be refused.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.augmentation_experiment import (
    AUGMENTATION_ARMS,
    arm_arguments,
    assert_label_quality,
    experiment_plan,
    minted_negative_label_quality,
)


def test_off_arm_zeroes_the_single_and_only_switch() -> None:
    assert arm_arguments("off", configured_frac=0.8)["mask_frac"] == 0.0
    assert arm_arguments("on", configured_frac=0.8)["mask_frac"] == 0.8
    plan = experiment_plan(0.8, seed=42)
    assert set(plan["arms"]) == set(AUGMENTATION_ARMS)
    assert plan["arms"]["on"]["seed"] == plan["arms"]["off"]["seed"] == 42
    assert plan["arms"]["on"]["mask_frac"] != plan["arms"]["off"]["mask_frac"]


def test_unknown_arm_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown augmentation arm"):
        arm_arguments("maybe", configured_frac=0.8)


# ── label quality ───────────────────────────────────────────────────────────
#  0 anchor, 1 pair side (agrees with the anchor), 2 donor, 3 minted copy.
_PAYLOAD = [
    "cola water volume_ml_500 sugar",
    "cola aqua volume_ml_500 sugar",
    "lime soda volume_ml_1500 sugar",
    "cola water volume_ml_1500 sugar",
]
_ROW_BC = np.asarray(["g1", "g2", "g3", "g1"], dtype=object)


def _row(**overrides) -> dict:
    row = {
        "population": "hard_negative",
        "generation_variant": "minted",
        "anchor_payload_idx": 0,
        "pair_payload_idx": 1,
        "copy_payload_idx": 3,
        "gtin": "g1",
        "fields_hit": ["volume"],
        "fields_after": {"volume": ["volume_ml_1500"]},
        "donor_anchor_payload_idx": 2,
    }
    row.update(overrides)
    return row


def test_a_genuine_minted_negative_passes() -> None:
    report = minted_negative_label_quality([_row()], _PAYLOAD, _ROW_BC)
    assert report["checked"] == 1
    assert report["quality"] == 1.0
    assert report["violations"] == []
    assert report["per_field"] == {"volume": 1}


def test_a_copy_that_does_not_conflict_is_refused() -> None:
    """Identical declarations labeled 0 are a silent false negative."""
    payload = list(_PAYLOAD)
    payload[3] = "cola water volume_ml_500 sugar"  # no transplant survived
    report = minted_negative_label_quality([_row()], payload, _ROW_BC)
    assert report["quality"] == 0.0
    assert "conflict" in report["violations"][0]["reason"]
    with pytest.raises(ValueError, match="label quality failed"):
        assert_label_quality(report)


def test_a_value_absent_from_the_donor_is_refused() -> None:
    """Nothing is invented: the transplant must exist in the recorded donor."""
    payload = list(_PAYLOAD)
    payload[3] = "cola water volume_ml_900 sugar"
    row = _row(fields_after={"volume": ["volume_ml_900"]})
    report = minted_negative_label_quality([row], payload, _ROW_BC)
    assert report["quality"] == 0.0
    assert "donor" in report["violations"][0]["reason"]


def test_masking_that_erases_the_transplant_is_refused() -> None:
    """A masked variant may drop prose, never the field that makes the row 0."""
    payload = list(_PAYLOAD)
    payload[3] = "cola water volume_ml_500 sugar"  # masking erased the field
    payload[1] = "cola aqua volume_ml_1500 sugar"  # pair side now agrees with donor
    row = _row(generation_variant="minted_masked")
    report = minted_negative_label_quality([row], payload, _ROW_BC)
    assert report["quality"] == 0.0
    assert report["violations"]


def test_the_shipped_bundle_minted_negatives_are_genuine() -> None:
    """Measured on the prepared fixture (skips when it is absent)."""
    from pathlib import Path

    from core.common import TRAIN_ROOT
    from training.prepared_bundle import load_prepared_bundle

    bundle = Path(TRAIN_ROOT) / "data/track_setup/text_prepared.pkl.gz"
    if not bundle.exists():
        pytest.skip(f"{bundle} not present in this checkout")
    _, data = load_prepared_bundle(bundle)
    report = minted_negative_label_quality(
        data["hard_negative_mask_audit"], data["payload"], data["row_bc"]
    )
    assert report["checked"] > 0
    assert report["quality"] == 1.0
    assert report["violations"] == []
