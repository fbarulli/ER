"""tests/test_pair_identity.py — ONE pair id, direction-independent.

The traceability defect this pins: the raw->canonical->pair->split half of the
pipeline is traceable, but the pair->prediction->metric half is not, because
each artifact spelled its pair key by hand and direction-dependently
(``f"{gtin1}|{gtin2}"`` in several places). ``(a, b)`` and ``(b, a)`` are the
same pair everywhere else in the pipeline, so the hand-spelled key silently
split one pair into two ids and no join could reassemble a single sample's
error.

``core.pair_identity.PairIdentity`` is the one place the key is computed.
These tests pin the two properties every consumer depends on — swap invariance
and uniqueness — and pin that the three data-producing paths stamp it.
"""
from __future__ import annotations

import pandas as pd

from core.pair_identity import PairIdentity
from core.schemas import (
    GATE_RESULTS_COLUMNS,
    LABELED_PAIRS_COLUMNS,
    check_gate_results_frame,
    check_labeled_pairs_frame,
)

# A UPC-12 and its zero-prefixed GTIN-13 sibling are the SAME code (core.gtin).
UPC12 = "036000291452"
EAN13 = "0" + UPC12
OTHER = "4006381333931"


def test_pair_id_is_invariant_under_endpoint_swap():
    assert PairIdentity.of(EAN13, OTHER) == PairIdentity.of(OTHER, EAN13)


def test_pair_id_folds_a_upc12_to_its_gtin13_sibling():
    assert PairIdentity.of(UPC12, OTHER) == PairIdentity.of(EAN13, OTHER)


def test_pair_id_is_unique_per_gate_pair():
    pairs = [("1" * 13, "2" * 13), ("2" * 13, "3" * 13), ("1" * 13, "3" * 13)]
    ids = {PairIdentity.of(a, b) for a, b in pairs}
    assert len(ids) == len(pairs)


def test_pair_id_keeps_distinct_malformed_endpoints_apart():
    # Non-conforming endpoints fall back to their raw spelling: two different
    # junk endpoints must never collapse to one shared key.
    assert PairIdentity.of("not-a-gtin", "also-not") != PairIdentity.of(
        "not-a-gtin", "third-not"
    )


def test_column_matches_scalar_of_and_is_swap_invariant():
    frame = pd.DataFrame(
        {"gtin1": [EAN13, OTHER], "gtin2": [OTHER, UPC12]}
    )
    ids = PairIdentity.column(frame["gtin1"], frame["gtin2"])
    assert list(ids) == [
        PairIdentity.of(EAN13, OTHER),
        PairIdentity.of(OTHER, UPC12),
    ]
    swapped = PairIdentity.column(frame["gtin2"], frame["gtin1"])
    assert list(ids) == list(swapped)


def test_gate_contract_requires_a_stamped_pair_id():
    row = {
        "gtin1": EAN13,
        "gtin2": OTHER,
        "canon1": "a",
        "canon2": "b",
        "gate_decision": "proceed",
        "gate_reason": "volume overlap",
        "similarity": 0.81,
        "pair_id": "",
    }
    frame = pd.DataFrame([row], columns=list(GATE_RESULTS_COLUMNS))
    try:
        check_gate_results_frame(frame)
    except ValueError as error:
        assert "pair_id" in str(error)
    else:
        raise AssertionError("gate contract accepted an empty pair_id")


def test_labeled_contract_requires_a_stamped_pair_id():
    row = {
        "gtin1": EAN13,
        "gtin2": OTHER,
        "true_label": 1,
        "pair_id": PairIdentity.of(EAN13, OTHER),
    }
    assert check_labeled_pairs_frame(
        pd.DataFrame([row], columns=list(LABELED_PAIRS_COLUMNS))
    ).shape == (1, 4)


def test_validation_assembler_stamps_the_pair_id():
    from training.build_final_validation import FoldResolver, ValidationRowAssembler

    resolver = FoldResolver({EAN13: 2, OTHER: 3})
    assembler = ValidationRowAssembler(resolver, {}, {}, n_folds=4)
    row, reason = assembler.assemble_with_reason(EAN13, OTHER, 1)
    assert reason == "scored_row"
    assert row["pair_id"] == PairIdentity.of(EAN13, OTHER)
    # an unresolved endpoint emits no row, so no unstamped row can leak out
    missing, missing_reason = assembler.assemble_with_reason(EAN13, "9" * 13, 1)
    assert missing is None
    assert missing_reason == "unresolved_endpoint"
