"""tests/test_prediction_export.py — the eval emits one row per scored pair.

The traceability gap this pins: ``training.evaluate_models`` computed a
per-pair prediction for every scored pair and discarded it (only the aggregate
summary survived). ``core.prediction_export.PredictionExport`` now owns that
artifact, keyed by the SSOT ``pair_id`` and joined to the validation slice
columns. These tests assert the acceptance contract on synthetic frames:
one row per scored pair, with pair_id + label + score + threshold + predicted
and the ``v1_*``/``v2_*`` slices present, joined through normalization (a
UPC-12 in one frame and its GTIN-13 sibling in the other are the same pair).
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.pair_identity import PairIdentity
from core.prediction_export import PredictionExport

UPC12 = "036000291452"
EAN13 = "0" + UPC12
OTHER = "4006381333931"
THIRD = "5901234123457"

SLICE_COLUMNS = ("v1_volume", "v2_volume")


def _scored() -> pd.DataFrame:
    """Two scored pairs, each spelling one endpoint differently from the
    validation frame (a UPC-12 sibling, and a swapped direction).

    A non-contiguous index mirrors the real lane (the TEST half is a boolean
    mask over the merged frame), so index handling is exercised here too.
    """
    frame = pd.DataFrame(
        {
            "gtin1": [EAN13, OTHER],
            "gtin2": [OTHER, THIRD],
            "true_label": [1, 0],
            "fold": [3, 3],
            "sim_model": [0.9, 0.2],
        },
        index=[5, 9],
    )
    return frame


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gtin1": [UPC12, THIRD],
            "gtin2": [OTHER, OTHER],
            "v1_volume": ["[1, 2]", "[3]"],
            "v2_volume": ["[1, 2]", "[4]"],
        }
    )


def _export() -> pd.DataFrame:
    return PredictionExport.frame(
        _scored(),
        PredictionExport.slice_frame(_validation()),
        model="minilm_l6",
        eval_half="test",
        score_column="sim_model",
        threshold=0.5,
    )


def test_one_row_per_scored_pair_with_the_required_and_slice_columns():
    frame = _export()
    assert len(frame) == 2
    assert set(PredictionExport.REQUIRED_COLUMNS) <= set(frame.columns)
    assert set(SLICE_COLUMNS) <= set(frame.columns)


def test_pair_id_is_the_ssot_key_and_joins_across_endpoint_spellings():
    frame = _export()
    assert list(frame["pair_id"]) == [
        PairIdentity.of(EAN13, OTHER),
        PairIdentity.of(OTHER, THIRD),
    ]
    # row 1 spells the UPC-12 while the validation frame spells its GTIN-13
    # sibling; row 2 is spelled in the opposite direction in the two frames:
    # both slices must still join (normalization + ordering in the key)
    assert list(frame["v1_volume"]) == ["[1, 2]", "[3]"]
    assert frame["pair_id"].is_unique


def test_label_score_threshold_and_predicted_are_carried():
    frame = _export()
    assert list(frame["label"]) == [1, 0]
    assert list(frame["score"]) == [0.9, 0.2]
    assert list(frame["predicted"]) == [1, 0]
    assert list(frame["threshold"]) == [0.5, 0.5]
    assert list(frame["gtin1_norm"]) == [PairIdentity.endpoint_key(EAN13),
                                         PairIdentity.endpoint_key(OTHER)]
    # ONE ``_norm`` meaning across artifacts: the 14-digit width every
    # validation/prediction frame shares (13-digit sources are left-padded).
    assert all(len(value) == 14 for value in frame["gtin1_norm"])


def test_filename_parallels_the_trained_lane_pair_dump():
    assert (
        PredictionExport.filename("minilm_l6", "test", 3)
        == "eval_minilm_l6_test_fold3_pairs.csv"
    )


def test_a_scored_pair_with_no_validation_slice_fails_loud():
    scored = _scored()
    scored.loc[5, "gtin2"] = "9" * 13  # a pair the validation frame does not hold
    with pytest.raises(ValueError, match="no validation slice row"):
        PredictionExport.frame(
            scored,
            PredictionExport.slice_frame(_validation()),
            model="m",
            eval_half="test",
            score_column="sim_model",
            threshold=0.5,
        )


def test_a_duplicate_pair_id_fails_loud():
    scored = pd.concat([_scored(), _scored()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate pair_id"):
        PredictionExport.frame(
            scored,
            PredictionExport.slice_frame(_validation()),
            model="m",
            eval_half="test",
            score_column="sim_model",
            threshold=0.5,
        )
