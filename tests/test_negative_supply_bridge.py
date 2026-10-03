"""Lane bridge (scope A): negative swap, minted leaves, cap, parity, leakage.

The gate path is the default and must stay byte-for-byte unchanged; the lane
is opt-in via config and its output is checked here without the live
canonical/gate artifacts.
"""

import numpy as np
import pandas as pd
import pytest

from core.schemas import NegativeSupplyModeSpec, TrainingData, TrainingStats
from training.negative_supply import (
    POPULATION_BASE_NEGATIVE,
    POPULATION_MINTED_PARTNER,
    POPULATION_REAL_PARTNER,
    assemble_lane_bundle,
    gtin_group_split,
    token_move,
)

PAIR_COLUMNS = [
    "anchor_row", "partner_row", "label", "population", "is_real",
    "anchor_gtin", "partner_gtin", "anchor_text", "partner_text", "score",
    "diff_dimension", "edit_field", "edit_from", "edit_to",
    "shadow_gate_decision", "shadow_gate_reason",
]


def _stats() -> dict:
    values = {name: 0 for name in TrainingStats.model_fields}
    values["negative_supply_mode"] = "gate"
    return TrainingStats(**values).model_dump()


def _base() -> dict:
    # 2 sku rows + 2 canonical rows; gtins 100 and 200.
    return {
        "payload": [
            "sku0 [FIELD_FLAVOR] flavor_apple",
            "sku1 [FIELD_FLAVOR] flavor_lime",
            "canon0 flavor_apple",
            "canon1 flavor_lime",
        ],
        "structured_features": [[0.0], [0.0], [0.0], [0.0]],
        "row_bc": np.array(["100", "200", "100", "200"], dtype=object),
        "pos": np.array([[0, 2], [1, 3]], dtype=int),
        "neg": np.array([[0, 3]], dtype=int),
        "targeted_attribute_neg": np.empty((0, 2), dtype=int),
        "cross_brand_neg": np.empty((0, 2), dtype=int),
        "gtin_to_row": {"100": 0, "200": 1},
        "stats": _stats(),
    }


def _pair(**overrides) -> dict:
    row = {column: "" for column in PAIR_COLUMNS}
    row.update({"anchor_row": 0, "partner_row": -1, "label": 0, "is_real": True})
    row.update(overrides)
    return row


def test_real_partner_maps_through_gtin_to_row_and_canonical():
    pairs = pd.DataFrame([
        _pair(anchor_gtin="100", partner_gtin="200", population=POPULATION_REAL_PARTNER,
              diff_dimension="flavor"),
    ])
    out = assemble_lane_bundle(_base(), pairs, n_sku=2)
    bundle = TrainingData(**out)
    # anchor -> gtin_to_row(100)=0 ; target -> canon idx of 200 = 3.
    assert bundle.neg.tolist() == [[0, 3]]
    assert bundle.neg_source == [POPULATION_REAL_PARTNER]
    assert bundle.neg_minted.size == 0


def test_base_negative_maps_like_the_gate_path():
    pairs = pd.DataFrame([
        _pair(anchor_gtin="100", partner_gtin="200", population=POPULATION_BASE_NEGATIVE),
    ])
    out = assemble_lane_bundle(_base(), pairs, n_sku=2)
    assert TrainingData(**out).neg.tolist() == [[0, 3]]


def test_minted_partner_is_a_leaf_replayed_on_the_trainer_text():
    pairs = pd.DataFrame([
        _pair(population=POPULATION_MINTED_PARTNER, is_real=False,
              anchor_gtin="100", edit_field="flavor",
              edit_from="apple", edit_to="lime", partner_text="ignored"),
    ])
    out = assemble_lane_bundle(_base(), pairs, n_sku=2)
    bundle = TrainingData(**out)
    # anchor text is untouched
    assert bundle.payload[0] == "sku0 [FIELD_FLAVOR] flavor_apple"
    # exactly one appended leaf, token moved, tagged, and NOT in eval `neg`
    assert bundle.payload[-1] == "sku0 [FIELD_FLAVOR] flavor_lime"
    assert str(bundle.row_bc[-1]).startswith("minted:0:")
    assert bundle.payload_source[-1] == "minted"
    assert bundle.neg_minted.tolist() == [[0, 4]]  # appended after the 4 base rows
    assert bundle.neg.tolist() == []  # minted never enters eval
    assert bundle.stats.negative_supply_mode == "lane"
    assert bundle.stats.n_lane_minted == 1


def test_minted_move_is_whitelisted_and_changes_the_value():
    moved = token_move("x [FIELD_FLAVOR] flavor_apple", "flavor",
                       {"apple": "lime"})
    assert moved == ("x [FIELD_FLAVOR] flavor_lime", "apple", "lime")
    with pytest.raises(ValueError):
        token_move("x", "pack", {})


def test_mint_cap_drops_excess_minted_rows_deterministically():
    base = _base()
    real = [_pair(anchor_gtin="100", partner_gtin="200",
                  population=POPULATION_REAL_PARTNER, diff_dimension="flavor")]
    minted = [
        _pair(anchor_row=0, population=POPULATION_MINTED_PARTNER, is_real=False,
              edit_field="flavor", edit_from="apple", edit_to="lime")
        for _ in range(4)
    ]
    out = assemble_lane_bundle(base, pd.DataFrame(real + minted), n_sku=2, mint_cap=0.5)
    bundle = TrainingData(**out)
    # share m/(r+m) <= 0.5 with r=1 -> m <= 1.
    assert bundle.stats.n_lane_minted == 1
    assert bundle.stats.n_lane_minted_dropped_cap == 3


def test_leakage_real_pairs_share_a_fold_and_minted_are_unscored():
    frame = pd.DataFrame([
        _pair(anchor_gtin="100", partner_gtin="200", population=POPULATION_REAL_PARTNER),
        _pair(anchor_gtin="200", partner_gtin="300", population=POPULATION_REAL_PARTNER),
        _pair(anchor_gtin="100", partner_gtin="", population=POPULATION_MINTED_PARTNER,
              is_real=False),
    ])
    folds = gtin_group_split(frame, k=4, seed=1337)
    # 100-200-300 are one union component -> one fold; minted is -1 (never scored).
    assert folds.iloc[0] == folds.iloc[1]
    assert folds.iloc[2] == -1


def test_default_mode_is_gate_and_lane_requires_a_run_tag():
    assert NegativeSupplyModeSpec().mode == "gate"
    with pytest.raises(ValueError, match="requires pairs_run_tag"):
        NegativeSupplyModeSpec(mode="lane")
    assert NegativeSupplyModeSpec(mode="lane", pairs_run_tag="t").mode == "lane"


def test_default_path_lane_fields_stay_empty():
    # Parity guard: the gate path never populates the lane provenance fields.
    bundle = TrainingData(**_base())
    assert bundle.payload_source == []
    assert bundle.neg_source == []
    assert bundle.neg_minted.size == 0


def test_default_mode_dispatches_straight_to_build_training_data(monkeypatch):
    # Parity: with the default mode 'gate', load_base_data is a pass-through to
    # build_training_data — the lane branch is never taken.
    import pipeline
    from training.base_data import load_base_data

    sentinel = {"payload": ["untouched"], "stats": {}}
    monkeypatch.setattr(
        pipeline, "build_training_data",
        lambda df, payload_variant="full": sentinel,
    )
    assert load_base_data("df-sentinel", payload_variant="full") is sentinel
