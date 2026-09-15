from __future__ import annotations

import pandas as pd
import pytest

from training.rand_matching import candidate_gate_fields, confidence_penalty_mask

SETTINGS = {
    "enabled": True,
    "critical_attributes": ["volume", "pack", "flavor"],
    "minimum_joint_missing": 2,
    "penalty_per_joint_missing": 0.01,
    "max_penalty": 0.03,
    "preserve_exact_gtin": True,
}


def _info(*, volume=(), pack=(), flavor="") -> dict[str, object]:
    return {"volume": set(volume), "pack": set(pack), "flavor": flavor}


def test_mask_penalizes_only_shared_missing_evidence() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(volume={500}),
        _info(volume={500}),
        exact_gtin=False,
        config=SETTINGS,
    )
    assert penalty == pytest.approx(0.02)
    assert reason == "jointly_missing:pack,flavor"


def test_mask_preserves_exact_gtin_even_when_all_attributes_are_missing() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(), _info(), exact_gtin=True, config=SETTINGS
    )
    assert penalty == 0.0
    assert reason == "exact_gtin_preserved"


def test_mask_does_not_penalize_when_joint_missing_minimum_is_not_met() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(volume={500}, pack={12}),
        _info(volume={500}, pack={12}),
        exact_gtin=False,
        config=SETTINGS,
    )
    assert penalty == 0.0
    assert reason == "sufficient_attribute_evidence"


def test_mask_is_capped_and_never_negative() -> None:
    settings = {**SETTINGS, "penalty_per_joint_missing": 0.2, "max_penalty": 0.04}
    penalty, _ = confidence_penalty_mask(
        _info(), _info(), exact_gtin=False, config=settings
    )
    assert penalty == pytest.approx(0.04)


def test_candidate_score_uses_mask_but_keeps_raw_score() -> None:
    row = pd.Series({"SKU_ID": "sku-1", "barcode": "", "title": "water"})
    result = candidate_gate_fields(
        row,
        _info(),
        "candidate-1",
        {},
        0.83,
        sku_id="sku-1",
        source_row_index="0",
    )
    assert result["raw_score"] == pytest.approx(0.83)
    assert result["confidence_penalty"] == pytest.approx(0.03)
    assert result["score"] == pytest.approx(0.80)
