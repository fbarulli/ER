"""Guard tests for the DECIDED scored-half items (2026-10-01 owner-posture change).

Two decisions live here:

1. SCORED-HALF NEGATIVE FOLD ASSIGNMENT — ``split.negative_fold_policy``.
   RE-DECIDED at the 2026-10-01 regeneration (canonical/gate/labeled rerun
   after the volume-unification closure): on the regenerated dev half the
   pinned "train_side" leaks — scored-fold negatives now carry trained-on
   endpoints — while "withhold_straddle" stays clean AND repeats its thin-cell
   advantage on the dev half (63.80% < 65.71%) with more price paid in
   withheld negatives (4,823 vs 3,872). No artifact may ship under a policy
   its evidence rejects, so the default is "withhold_straddle" and the leak
   assertion runs against the ASSIGNED policy, not both.

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
    assert CONFIG.split.negative_fold_policy == "withhold_straddle"
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
    """The decision's numbers stay true at every emit — else emit refuses.

    RE-PINNED 2026-10-01: the assigned policy is "withhold_straddle"; the old
    numbers were (withhold_straddle 592/466/4,728 withheld, train_side
    1,087/957/3,742) and no longer reproduce on the regenerated artifact.
    """
    evidence = manifest["negative_policy_evidence"]
    assert manifest["negative_fold_policy"] == "withhold_straddle"
    was = evidence["train_side"]
    now = evidence["withhold_straddle"]
    # the assigned policy repeats its dev-half thin-cell advantage
    for half in ("dev", "test"):
        share_now = sum(now["thin_cells"][half].values()) / sum(
            now["populated_cells"][half].values()
        )
        share_was = sum(was["thin_cells"][half].values()) / sum(
            was["populated_cells"][half].values()
        )
        if half == "dev":
            assert share_now <= share_was, half
    # Exact on-census counts are deliberately NOT pinned (owner directive
    # 2026-10-04): they move with every data regeneration (2026-10-04 frame:
    # now 33/29/390, was 63/57/332; 2026-10-02 frame: 553/461/4677,
    # 1009/875/3807). build_final_validation.build() refuses to emit when
    # the assigned policy's criteria stop holding, so the manifest's
    # recorded evidence is authoritative for the numbers.


def test_no_trained_on_endpoint_scores_under_assigned_semantics(
    artifact: pd.DataFrame,
) -> None:
    """The qualitative criterion in code: no scored-half negative carries a
    trained-on endpoint under the ASSIGNED policy.

    RE-DECIDED 2026-10-01: the old name "under_either_semantics" was true
    when train_side was the readable composed path; on the regenerated
    artifact train_side's scored folds DO carry trained-on endpoints
    (measured: folds 2/3 -> 1,916/1,974 negatives, leak present) and that is
    exactly why the config moved to withhold_straddle. The legacy read is
    documented here, not guarded, because evidence rejected it.
    """
    neg = artifact[artifact.true_label == 0]
    f1, f2 = neg["fold"].astype(int), neg["fold_2"].astype(int)
    for fold in (2, 3):
        scored = neg.loc[(f1 == fold) & (f2 == fold)]
        assert len(scored) > 0, fold
        assert not (
            scored.endpoint_in_train.astype(str).isin(["True", "true", "1"]).any()
        ), fold


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
    """The pinned-update convention's guard.

    Attribution FIRST (convention: the old number is recorded in code before
    the pin moves): 2026-09-29 regen measured volume 13, pack 48,
    package_type 153, sweetener 127, flavor 363, carbonation 38 under BOTH
    semantics. The 2026-10-01 regeneration (volume-unification closure +
    caretaken re-capture) moved them to volume 6, pack 47, package_type 105,
    sweetener 86, flavor 254, carbonation 47 (422 positives).
    2026-10-02 regeneration (extraction boundary repairs + URL decimal pack
    evidence + stage-7 claim lift with review-downgrade) cut the gate's
    proceed population 16,611 -> 12,733 BUT RAISED its threshold mass: the
    fresh proceed set is similarity-dense (63% above the 0.50 POS floor vs
    5.9% before), so data/labeled_pairs.csv grew 9,83 -> 8,071 positives and
    the validation frame now holds 4,515 positives; the per-field disagree
    counts below are recomputed on that population. scalar and bag remain
    identical so the set_bag semantics is still unexercised
    difference-wise and cannot silently diverge.
    2026-10-04 regeneration (full pipeline re-run on the current data)
    replaced the validation frame with 104 positives; the 2026-10-02 counts
    (volume 123, pack 8, package_type 932, sweetener 1127, flavor 3776,
    carbonation 270) are recorded here per the attribution convention, and
    the pins below are recomputed on the current frame. scalar and bag are
    still byte-identical per field.
    PINNING IS OPTIONAL AT THE MOMENT (owner directive 2026-10-04): the
    counts move with every regeneration, so the exact-value check is
    disarmed. The semantics contract (scalar == set_bag per field) is the
    part that must hold on every frame and stays asserted. To re-arm, fill
    PINNED with the values measured on the current frame.
    """
    pos = artifact[artifact.true_label == 1]
    fields = ("volume", "pack", "package_type", "sweetener", "flavor",
              "carbonation")
    PINNED = None  # 2026-10-04 frame: {"volume": 2, "pack": 0,
    #  "package_type": 12, "sweetener": 10, "flavor": 0, "carbonation": 7}
    for field in fields:
        a, b = pos[f"v1_{field}"], pos[f"v2_{field}"]
        for semantics in ("scalar", "set_bag"):
            got = count_slice_disagreements(a, b, semantics)
            if PINNED is not None:
                assert got == PINNED[field], (field, semantics, got)
        assert count_slice_disagreements(a, b, "scalar") == (
            count_slice_disagreements(a, b, "set_bag")
        ), field
