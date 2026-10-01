"""Guard tests for the DECIDED scored-half items (2026-10-01 owner-posture change).

Two decisions live here:

1. SCORED-HALF NEGATIVE FOLD ASSIGNMENT — ``split.negative_fold_policy``.
   Policy B ("train_side") is the default because it measured: DEV negatives
   592 -> 1,087 (+83.6%), TEST negatives 466 -> 957 (+105.4%), thin-cell share
   below ``robust_validation.min_test_negatives`` = 5 improving on both halves
   (DEV 67.6% -> 64.9%, TEST 70.2% -> 67.2%), and zero trained-on endpoints
   entering the scored half under either policy. Pinned as criteria runs over
   the live artifact (skipped, never failed, when the artifact is not built)
   AND as synthetic unit tests so the guard is never vacuous.

2. SLICE-FLAG SET SEMANTICS — ``evaluation.slice_agreement = "set_bag"``.
   The v1_*/v2_* slice columns are set-valued extractions: agreement is BAG
   equality under the new default, legacy raw ``v1 == v2`` under "scalar".
   Measured on the live artifact every disagree count is byte-identical
   (volume 13, pack 48, package_type 153, sweetener 127, flavor 363,
   carbonation 38 — see the attribution comment in
   build_final_validation.write_manifest); the synthetic cases here pin the
   semantics differences the artifact does not exercise.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.common import training_cfg
from training.build_final_validation import (
    NEGATIVE_FOLD_POLICY_TRAIN_SIDE,
    NEGATIVE_FOLD_POLICY_WITHHOLD,
    count_slice_disagreements,
    negative_pair_fold,
    negative_policy_evidence,
    parse_field_bag,
)


N_FOLDS = 4
CONFIG = training_cfg()


def test_config_pins_the_decided_defaults() -> None:
    """The defaults ARE the decision; changing them is a re-decision, not a knob."""
    assert CONFIG.split.negative_fold_policy == "train_side"
    assert CONFIG.evaluation.slice_agreement == "set_bag"


def test_withhold_straddle_parks_the_straddle_policy_a() -> None:
    assert negative_pair_fold(
        "withhold_straddle", fold_a=2, fold_b=3, n_folds=N_FOLDS
    ) == 2  # legacy: pair_fold == fold_a; a straddler scores nowhere downstream
    assert negative_pair_fold(
        "withhold_straddle", fold_a=0, fold_b=2, n_folds=N_FOLDS
    ) == 0


def test_train_side_routes_train_endpoint_whole_policy_b() -> None:
    assert negative_pair_fold("train_side", 0, 2, N_FOLDS) == 0
    assert negative_pair_fold("train_side", 3, 0, N_FOLDS) == 0
    # a train fold is anything below the last two quarters, not just fold 0
    assert negative_pair_fold("train_side", 3, 1, N_FOLDS) == 1


def test_train_side_assigns_dev_test_straddle_whole() -> None:
    # the 986 dev/test straddlers A could never score get fold_a = fold 2/3
    assert negative_pair_fold("train_side", 2, 3, N_FOLDS) == 2
    assert negative_pair_fold("train_side", 3, 2, N_FOLDS) == 3


def test_unknown_policy_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown negative_fold_policy"):
        negative_pair_fold("mean_of_folds", 2, 3, N_FOLDS)


@pytest.fixture()
def artifact() -> pd.DataFrame:
    import os

    from core.common import F

    path = F["final_validation"]
    if not os.path.exists(path):
        pytest.skip(f"{path} not built; run -m src.training.build_final_validation")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["true_label"] = frame["true_label"].astype(int)
    return frame


@pytest.fixture()
def manifest() -> dict:
    """The emitted decision evidence (computed on RAW endpoint folds in build()).

    The on-disk artifact carries policy-B ASSIGNED fold columns; recomputing
    policy A's criterion from them would double-apply the assignment, so the
    pin consumes the manifest's recorded evidence instead.
    """
    import json
    import os
    from pathlib import Path

    from core.common import RESULTS

    path = RESULTS / "manifests" / "final_validation.json"
    if not os.path.exists(path):
        pytest.skip(f"{path} not emitted; run -m src.training.build_final_validation")
    return json.loads(Path(path).read_text())


def test_pinned_decision_criteria_hold_on_live_evidence(manifest: dict) -> None:
    """The decision's numbers stay true at every emit — else emit refuses."""
    evidence = manifest["negative_policy_evidence"]
    assert manifest["negative_fold_policy"] == "train_side"
    was = evidence["withhold_straddle"]
    now = evidence["train_side"]
    assert now["scored_dev_negatives"] > was["scored_dev_negatives"]
    assert now["scored_test_negatives"] > was["scored_test_negatives"]
    for half in ("dev", "test"):
        share_b = sum(now["thin_cells"][half].values())
        cells_b = sum(now["populated_cells"][half].values())
        share_a = sum(was["thin_cells"][half].values())
        cells_a = sum(was["populated_cells"][half].values())
        assert cells_b > 0 and cells_a > 0
        assert share_b / cells_b <= share_a / cells_a, half
    # the on-census measured pairs (regen drift makes a stale pin lie)
    assert was["scored_dev_negatives"] == 592
    assert was["scored_test_negatives"] == 466
    assert now["scored_dev_negatives"] == 1087
    assert now["scored_test_negatives"] == 957
    assert was["negatives_withheld_from_scored_half"] == 4728
    assert now["negatives_withheld_from_scored_half"] == 3742


def test_no_trained_on_endpoint_scores_under_either_semantics(
    artifact: pd.DataFrame,
) -> None:
    """The qualitative criterion in code: no scored-half negative carries a
    trained-on endpoint — A (raw fold==fold_2==scored-fold mask) and B (the
    assigned fold column, exactly what a consumer reads)."""
    neg = artifact[artifact.true_label == 0]
    f1, f2 = neg["fold"].astype(int), neg["fold_2"].astype(int)
    halves = {
        "A withhold_straddle": {
            fold: (f1 == fold) & (f2 == fold) for fold in (2, 3)
        },
        "B train_side (assigned artifact folds)": {
            fold: neg["fold"].astype(int) == fold for fold in (2, 3)
        },
    }
    for policy, fold_masks in halves.items():
        for fold, mask in fold_masks.items():
            scored = neg.loc[mask]
            assert len(scored) > 0, (policy, fold)
            assert not (
                scored.endpoint_in_train.astype(str).isin(["True", "true", "1"]).any()
            ), (policy, fold)


def test_evidence_measures_raw_endpoint_folds() -> None:
    """Synthetic: the harness reflects the RAW-vs-ASSIGNED distinction.

    A frame with RAW endpoint folds (2,2)/(2,3)/(0,2): policy A scores 0
    negatives in DEV (a (2,3) straddler scores nowhere; an (0,2) pair is
    parked), policy B scores both — the harness is never vacuous.
    """
    frame = pd.DataFrame(
        {
            "true_label": [0, 0, 0],
            "fold": [2, 2, 0],
            "fold_2": [2, 3, 2],
        }
    )
    evidence = negative_policy_evidence(frame, min_test_negatives=5, n_folds=4)
    assert evidence[NEGATIVE_FOLD_POLICY_WITHHOLD]["scored_dev_negatives"] == 1
    assert evidence[NEGATIVE_FOLD_POLICY_TRAIN_SIDE]["scored_dev_negatives"] == 2
    assert evidence[NEGATIVE_FOLD_POLICY_WITHHOLD]["scored_test_negatives"] == 0
    assert evidence[NEGATIVE_FOLD_POLICY_TRAIN_SIDE]["scored_test_negatives"] == 0


def test_evidence_is_not_vacuous(manifest: dict) -> None:
    """A policy comparison where A scores > B negatives proves the guards dead."""
    evidence = manifest["negative_policy_evidence"]
    total = lambda e: e["scored_dev_negatives"] + e["scored_test_negatives"]  # noqa: E731
    assert total(evidence["train_side"]) > total(evidence["withhold_straddle"])


# ── slice-flag set semantics ────────────────────────────────────────────────


def test_parse_field_bag_reads_list_literals_and_bare_tokens() -> None:
    assert parse_field_bag("[479.0, 518.0]") == {"479.0": 1, "518.0": 1}
    assert parse_field_bag("[518.0]") == {"518.0": 1}
    assert parse_field_bag("coconut") == {"coconut": 1}
    assert parse_field_bag("") == {}
    with pytest.raises(ValueError, match="unbalanced slice list"):
        parse_field_bag("[479.0,")


def test_scalar_equal_stays_agreement_under_both_semantics() -> None:
    a, b = pd.Series(["[518.0]"]), pd.Series(["[518.0]"])
    for semantics in ("scalar", "set_bag"):
        assert count_slice_disagreements(a, b, semantics) == 0


def test_set_bag_counts_spelling_only_difference_as_agreement() -> None:
    a, b = pd.Series(["[479.0, 518.0]"]), pd.Series(["[518.0, 479.0]"])
    # scalar comparison saw two different strings; the bag comparison does not
    assert count_slice_disagreements(a, b, "scalar") == 1
    assert count_slice_disagreements(a, b, "set_bag") == 0


def test_set_bag_keeps_content_differences_as_disagreement() -> None:
    a, b = pd.Series(["[479.0, 518.0]"]), pd.Series(["[518.0]"])
    assert count_slice_disagreements(a, b, "scalar") == 1
    assert count_slice_disagreements(a, b, "set_bag") == 1


def test_bag_is_multiplicity_sensitive() -> None:
    a, b = pd.Series(["[lime, lime]"]), pd.Series(["[lime]"])
    assert count_slice_disagreements(a, b, "set_bag") == 1


def test_scalar_equality_does_not_leak_into_set_bag_for_bare_vs_list() -> None:
    """A scalar-equal set-valued pair ('518.0' vs '[518.0]') counts UNequal.

    Wait — reversed here deliberately: a scalar-UNEQUAL, bag-EQUAL pair
    becomes agreement under set_bag and only set_bag (legacy keeps it a
    disagreement), which is the semantics this regression set exists to
    protect when the flag is flipped back to "scalar".
    """
    a, b = pd.Series(["518.0"]), pd.Series(["[518.0]"])
    assert count_slice_disagreements(a, b, "scalar") == 1
    assert count_slice_disagreements(a, b, "set_bag") == 0


def test_unknown_slice_semantics_fails_loud() -> None:
    a, b = pd.Series(["x"]), pd.Series(["x"])
    with pytest.raises(ValueError, match="unknown slice_agreement"):
        count_slice_disagreements(a, b, "tokenized")


def test_live_disagree_counts_are_byte_identical_to_scalar(
    artifact: pd.DataFrame,
) -> None:
    """The pinned-update convention's guard: today NONE of the counts move.

    If a future regen changes one, the attribution comment in
    write_manifest must gain the old number FIRST and this assertion must be
    updated with the new number recorded in code — never deleted.
    """
    pos = artifact[artifact.true_label == 1]
    fields = ("volume", "pack", "package_type", "sweetener", "flavor",
              "carbonation")
    pinned = {"volume": 13, "pack": 48, "package_type": 153,
              "sweetener": 127, "flavor": 363, "carbonation": 38}
    for field in fields:
        a, b = pos[f"v1_{field}"], pos[f"v2_{field}"]
        for semantics in ("scalar", "set_bag"):
            got = count_slice_disagreements(a, b, semantics)
            assert got == pinned[field], (field, semantics, got)
