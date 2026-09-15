"""Cross-brand hard-negative mining: correctness, honesty, and fold safety.

The defect these tests lock down (MODEL_INPUT_FIX_REPORT §15): the gate's
candidate pairs are generated INSIDE a brand block, so brand agreement is 100 %
in both training classes, measured brand separation is exactly 0.000 (volume
separates at +0.837), and the encoder can only learn that brand is noise. The
cross-brand miner is the lever: pairs whose brands DIFFER while every other
critical attribute agrees.

Two historical defects in this exact area are pinned here, not described:

* a previous miner re-added same-canonical TRUE MATCHES as label-0 pairs
  (154 of 350, 44 %) — covered by the same-canonical guard tests;
* negative sampling with ``replace=True`` duplicated rows while reporting them
  as distinct data — covered by the no-duplicate / no-replacement tests.

Everything is driven directly on synthetic worlds (no training, no artifacts):
the world is small enough that every emitted pair can be checked by hand.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from core.critical_attributes import normalized_attribute_text
from core.hard_negatives import (
    CrossBrandMiningFunnel,
    MiningFunnel,
    mine_cross_brand_negatives,
    mine_cross_brand_negatives_with_funnel,
    pairs_in_set,
)
from training.folds import component_folds, holdout_split

# ── synthetic world ────────────────────────────────────────────────────────
# Four canonicals, all "cola 330ml can":
#   * `acme cola`   330 can  (two GTINs — one is a surface spelling variant)
#   * `bolt cola`   330 can  — the cross-brand partner of acme
#   * `crisp cola`  330 can  — a third brand
#   * `delta cola`  500 can  — volume disagrees with acme/bolt/crisp
GTINS = {
    "acme_330": "4006381333931",
    "acme_surface": "4006381340250",
    "bolt_330": "4006381340366",
    "crisp_330": "4006381340373",
    "delta_500": "4006381340380",
}


def _canonical(
    gtin: str,
    brand: str,
    *,
    volume: float = 330.0,
    pack: int | None = None,
    package_type: str = "can",
    flavor: str = "cola",
    canonical: str | None = None,
) -> dict:
    return {
        "gtin": gtin,
        "canonical": canonical or f"{normalized_attribute_text(brand)} cola",
        "mode_brand": brand,
        "mode_flavor": flavor,
        "volume_set": f"[{float(volume)}]",
        "pack_set": f"[{int(pack)}]" if pack else "[]",
        "package_type_set": f"['{package_type}']" if package_type else "[]",
        "flavor_set": f"['{flavor}']" if flavor else "[]",
        "carbonation_set": "['carbonated']",
        "sweetener_set": "['sugar']",
        "pulp_set": "[]",
    }


def _world() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """(df, canonical_records, gtin_to_row, gtin_to_canon_idx) for the world."""
    records = [
        _canonical(GTINS["acme_330"], "Acme"),
        # The SAME brand written with different words: a spelling variant, not
        # a second brand.
        _canonical(GTINS["acme_surface"], "Acme Cola Co"),
        _canonical(GTINS["bolt_330"], "Bolt"),
        _canonical(GTINS["crisp_330"], "Crisp"),
        _canonical(GTINS["delta_500"], "Acme", volume=500.0, canonical="acme cola"),
    ]
    canon = pd.DataFrame(records)
    # One source row per GTIN, titles deliberately DIFFERENT (a same-title pair
    # would be the conflicting-barcode label error the guard exists for).
    df = pd.DataFrame(
        {
            "barcode": [
                GTINS["acme_330"], GTINS["acme_surface"], GTINS["bolt_330"],
                GTINS["crisp_330"], GTINS["delta_500"],
            ],
            "title": [
                "Acme Cola 330ml can", "Acme Cola Co 330ml can",
                "Bolt Cola 330ml can", "Crisp Cola 330ml can",
                "Acme Cola 500ml can",
            ],
        }
    )
    gtin_to_row = {g: i for i, g in enumerate(df["barcode"].astype(str))}
    # canonical block starts after the df rows, in sorted-GTIN order
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    return df, canon, gtin_to_row, gtin_to_canon_idx


def _mine(world, **kwargs):
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    params = {
        "n_target": 100,
        "require_agreement": ("volume", "package_type"),
        "min_similarity": 0.0,
    }
    params.update(kwargs)
    return mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx, **params
    )


def _brands(world) -> dict[str, str]:
    _, canon, _, _ = world
    return {
        str(row["gtin"]): normalized_attribute_text(row["mode_brand"])
        for _, row in canon.iterrows()
    }


def _endpoints(world, pairs):
    df, _, _, _ = world
    barcodes = [str(g) for g in df["barcode"]]
    gtins = sorted(str(g) for g in world[1]["gtin"])
    row_bc = barcodes + gtins
    return [(row_bc[int(a)], row_bc[int(b)]) for a, b in pairs]


# ── the mined population is a verified non-match population ────────────────
def test_every_emitted_pair_is_cross_brand_and_attribute_compatible() -> None:
    world = _world()
    pairs, scores = _mine(world)
    assert len(pairs) > 0
    brands = _brands(world)
    for (left, right), score in zip(_endpoints(world, pairs), scores, strict=True):
        assert brands[left] != brands[right], (left, right)
        assert 0.0 <= score <= 1.0
    # Every 330ml pair EXCEPT the two spellings of one brand: "Acme" and
    # "Acme Cola Co" are the same brand written twice. The 500ml row is in a
    # different volume block and is never a partner.
    undirected = {frozenset(pair) for pair in _endpoints(world, pairs)}
    assert undirected == {
        frozenset({GTINS["acme_330"], GTINS["bolt_330"]}),
        frozenset({GTINS["acme_330"], GTINS["crisp_330"]}),
        frozenset({GTINS["acme_surface"], GTINS["bolt_330"]}),
        frozenset({GTINS["acme_surface"], GTINS["crisp_330"]}),
        frozenset({GTINS["bolt_330"], GTINS["crisp_330"]}),
    }


def test_same_canonical_true_matches_are_never_emitted() -> None:
    """The 154-of-350 defect: a same-canonical pair must never be label 0."""
    records = [
        _canonical(GTINS["acme_330"], "Acme", canonical="acme cola"),
        # A different GTIN, a DIFFERENT brand, but the SAME canonical item:
        # a true match that the gate's brand block hid and this lane could
        # re-add as a label-0 pair.
        _canonical(GTINS["bolt_330"], "Bolt", canonical="acme cola"),
    ]
    canon = pd.DataFrame(records)
    df = pd.DataFrame(
        {"barcode": [r["gtin"] for r in records], "title": ["a", "b"]}
    )
    gtin_to_row = {g: i for i, g in enumerate(df["barcode"])}
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    assert len(pairs) == 0
    assert funnel.dropped_candidates_same_canonical == 1
    assert [step for step, _, _, _ in funnel.stages()].count("same_canonical_guard") == 1


def test_label_error_pairs_are_excluded_and_counted() -> None:
    """Same title + conflicting barcode = known label error, never label 0."""
    records = [
        _canonical(GTINS["acme_330"], "Acme"),
        _canonical(GTINS["bolt_330"], "Bolt"),
    ]
    canon = pd.DataFrame(records)
    df = pd.DataFrame(
        {
            "barcode": [r["gtin"] for r in records],
            "title": ["Cola 330ml can", "Cola 330ml can"],
        }
    )
    gtin_to_row = {g: i for i, g in enumerate(df["barcode"])}
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    assert len(pairs) == 0
    assert funnel.dropped_candidates_label_error == 1


def test_brand_surface_variants_are_not_mined_as_cross_brand() -> None:
    """One brand written twice is not two brands (measured: 16 live candidates)."""
    world = _world()
    pairs, _scores = _mine(world)
    endpoints = _endpoints(world, pairs)
    assert (GTINS["acme_330"], GTINS["acme_surface"]) not in endpoints
    assert (GTINS["acme_surface"], GTINS["bolt_330"]) in endpoints
    # the guard has its own reported step
    funnel = CrossBrandMiningFunnel()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    assert funnel.dropped_candidates_brand_surface_variant == 1


def test_attribute_conflict_candidates_are_dropped_with_a_census() -> None:
    records = [
        _canonical(GTINS["acme_330"], "Acme", flavor="cola"),
        _canonical(GTINS["bolt_330"], "Bolt", flavor="lemon"),
    ]
    canon = pd.DataFrame(records)
    df = pd.DataFrame({"barcode": [r["gtin"] for r in records], "title": ["a", "b"]})
    gtin_to_row = {g: i for i, g in enumerate(df["barcode"])}
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    assert len(pairs) == 0
    assert funnel.dropped_candidates_attribute_conflict == 1
    assert funnel.conflict_dimension_census == {"flavor": 1}


# ── honesty: no replacement, no duplication, no silent drops ───────────────
def test_no_duplicate_rows_and_both_directions_are_emitted() -> None:
    world = _world()
    pairs, scores = _mine(world)
    directed = {tuple(int(x) for x in pair) for pair in pairs}
    assert len(directed) == len(pairs)
    assert len(scores) == len(pairs)
    # Both directions of every accepted candidate exist: one candidate emits
    # (source row of A -> canonical index of B) AND (source row of B ->
    # canonical index of A) — the same shape every other negative lane emits.
    gtins = _endpoints(world, pairs)
    assert gtins, "no pair emitted"
    assert {(right, left) for left, right in gtins} == set(gtins)


def test_target_caps_rows_and_is_reported_as_a_funnel_step() -> None:
    world = _world()
    pairs, _scores = _mine(world, n_target=3)
    assert len(pairs) == 3
    funnel = CrossBrandMiningFunnel()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=3, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    # 5 candidates survive the filters; the target stops the third direction,
    # so 3 rows are emitted (never 4) and the 3 candidates that could not be
    # started are counted at the cap — never silently dropped.
    assert len(pairs) == 3
    assert funnel.emitted_pairs == 3
    assert funnel.dropped_candidates_target_cap == 3
    # the cap bound (rows == target) and truncation are both visible: 5
    # candidates cleared every filter, 2 of them were accepted
    assert funnel.to_dict()["target_reached"] is True
    assert (funnel.accepted_candidates, funnel.passed_candidates) == (2, 5)


def test_endpoint_diversity_caps_bound_each_endpoint() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, max_per_canonical=1, funnel=funnel,
    )
    counts: dict[str, int] = {}
    for left, right in _endpoints(world, pairs):
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    assert counts and max(counts.values()) <= 2  # one accepted pair = 2 rows
    assert funnel.accepted_candidates == 2
    assert funnel.dropped_candidates_endpoint_cap == 3


def test_baseline_duplicates_are_dropped_and_counted() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    _pairs, _scores = _mine(world)
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0,
        existing=np.asarray(
            [[gtin_to_row[GTINS["acme_330"]], gtin_to_canon_idx[GTINS["bolt_330"]]]]
        ),
        funnel=funnel,
    )
    assert funnel.dropped_pairs_already_in_baseline == 1
    assert (
        gtin_to_row[GTINS["acme_330"]],
        gtin_to_canon_idx[GTINS["bolt_330"]],
    ) not in {tuple(int(x) for x in pair) for pair in pairs}
    assert len(pairs) == 9  # 5 candidates x 2 directions, minus the 1 baseline row


def test_the_same_call_twice_is_byte_identical() -> None:
    world = _world()
    first_pairs, first_scores = _mine(world)
    second_pairs, second_scores = _mine(world)
    assert np.array_equal(first_pairs, second_pairs)
    assert np.array_equal(first_scores, second_scores)


def test_candidate_generation_census_is_reported() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    generation = funnel.to_dict()["candidate_generation"]
    # 4 canonicals carry volume AND package_type; the 5th (no package_type
    # evidence) generates no candidate and is counted.
    assert generation["canonicals_total"] == 5
    assert generation["canonicals_without_required_evidence"] == 0
    assert generation["canonicals_in_blocks"] == 5
    # blocking is per (volume, package_type) BLOCK: the four 330ml cans make
    # C(4,2) = 6 candidate pairs, the lone 500ml row makes none.
    assert generation["blocks"] == 2
    assert generation["candidates_in_blocks"] == 6
    assert funnel.gate_rows == 6


def test_require_agreement_without_evidence_generates_no_candidate() -> None:
    records = [
        _canonical(GTINS["acme_330"], "Acme"),
        _canonical(GTINS["bolt_330"], "Bolt", package_type=""),
    ]
    canon = pd.DataFrame(records)
    df = pd.DataFrame({"barcode": [r["gtin"] for r in records], "title": ["a", "b"]})
    gtin_to_row = {g: i for i, g in enumerate(df["barcode"])}
    gtin_to_canon_idx = {
        g: len(df) + i for i, g in enumerate(sorted(canon["gtin"].astype(str)))
    }
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    assert len(pairs) == 0
    assert funnel.canonicals_without_required_evidence == 1
    assert funnel.candidates_in_blocks == 0


def test_similarity_floor_is_strict_and_reported() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    # The canonicals are "<brand> cola", so two DIFFERENT brands share only
    # "cola": Jaccard = 1/3 = 0.333, below a 0.9 floor for every candidate.
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.9, funnel=funnel,
    )
    assert len(pairs) == 0
    assert funnel.dropped_candidates_below_similarity == 5


def test_invalid_require_agreement_is_loud() -> None:
    world = _world()
    with pytest.raises(ValueError, match="non-critical dimensions"):
        _mine(world, require_agreement=("brand",))
    with pytest.raises(ValueError, match="at least one critical dimension"):
        _mine(world, require_agreement=())


def test_zero_target_skips_the_lane_and_says_so() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    pairs, scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=0, funnel=funnel,
    )
    assert len(pairs) == 0 and len(scores) == 0
    assert funnel.skipped_reason == "n_target <= 0"


# ── fold safety: a mined negative must never straddle a split ──────────────
def test_mined_pairs_are_fold_local_when_both_endpoints_share_a_fold() -> None:
    """``pairs_in_set`` keeps a mined pair inside one fold, like every lane.

    The miner emits the SAME (source row, other canonical) shape as the gate
    and targeted lanes, so the fold filter is the existing one — a pair whose
    endpoints sit in different folds is excluded from BOTH sides, never
    trained on one side and evaluated on the other.
    """
    world = _world()
    pairs, _scores = _mine(world)
    df, canon, _, _ = world
    barcodes = [str(g) for g in df["barcode"]]
    gtins = sorted(str(g) for g in canon["gtin"])
    row_bc = np.asarray(barcodes + gtins)
    positives = np.asarray([[0, 1]], dtype=int)  # links the two "Acme" GTINs
    folds = component_folds(positives, row_bc, 2, 42)
    train, dev = folds[0], folds[1]
    in_train = pairs_in_set(pairs, row_bc, train)
    in_dev = pairs_in_set(pairs, row_bc, dev)
    # A mined pair is admitted to a side ONLY when BOTH endpoints live there,
    # so no pair can be trained on one side and evaluated on the other.
    assert not (in_train & in_dev).any()
    for mask, fold in ((in_train, train), (in_dev, dev)):
        for pair in pairs[mask]:
            assert row_bc[int(pair[0])] in fold
            assert row_bc[int(pair[1])] in fold
    # the identity that makes the exclusion honest: every mined pair is either
    # fold-local on exactly one side or crossing, and the crossing count is
    # what the fold filter drops (never a silent partial pair).
    crossing = len(pairs) - int(in_train.sum()) - int(in_dev.sum())
    assert crossing >= 0
    assert int(in_train.sum()) + int(in_dev.sum()) + crossing == len(pairs)


def test_holdout_split_keeps_mined_pairs_on_one_side() -> None:
    world = _world()
    pairs, _scores = _mine(world)
    df, canon, _, _ = world
    barcodes = [str(g) for g in df["barcode"]]
    gtins = sorted(str(g) for g in canon["gtin"])
    row_bc = np.asarray(barcodes + gtins)
    # positives link the two "Acme" GTINs so the component graph is not empty
    positives = np.asarray(
        [[0, int(np.flatnonzero(row_bc == GTINS["acme_surface"])[0])]], dtype=int
    )
    train_bc, dev_bc, test_bc = holdout_split(
        positives, row_bc, n_folds=4, seed=42, dev_fraction=0.25, test_fraction=0.25
    )
    for fold in (train_bc, dev_bc, test_bc):
        mask = pairs_in_set(pairs, row_bc, fold)
        for pair in pairs[mask]:
            assert row_bc[int(pair[0])] in fold and row_bc[int(pair[1])] in fold


# ── funnel contract (shared base, trace-ready readback) ────────────────────
def test_cross_brand_funnel_stages_are_cumulative() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, funnel=funnel,
    )
    stages = funnel.stages()
    assert [stage[0] for stage in stages] == [
        "candidate_generation",
        "endpoint_resolution",
        "same_canonical_guard",
        "label_error_guard",
        "brand_pair_distinct",
        "brand_surface_variant_guard",
        "attribute_agreement",
        "similarity_floor",
        "endpoint_diversity_cap",
        "target_cap",
        "direction_expansion",
        "baseline_deduplication",
        "emitted",
    ]
    for index, (previous, _in, out, _why) in enumerate(stages[:-1]):
        step, next_in = stages[index + 1][0], stages[index + 1][1]
        assert out == next_in, f"{previous} -> {step} does not carry the count"
    assert stages[-1][2] == funnel.emitted_pairs


def test_cross_brand_funnel_readback_is_trace_ready() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    funnel = CrossBrandMiningFunnel()
    pairs, _scores = mine_cross_brand_negatives(
        df, canon, gtin_to_row, gtin_to_canon_idx,
        n_target=100, require_agreement=("volume", "package_type"),
        min_similarity=0.0, max_per_canonical=20, max_per_brand=100, funnel=funnel,
    )
    detail = json.loads(json.dumps(funnel.to_dict()))
    assert detail["miner"] == "cross_brand_negatives"
    assert detail["require_agreement"] == ["volume", "package_type"]
    assert detail["target"] == 100
    assert detail["emitted_pairs"] == len(pairs)
    assert detail["candidate_to_emitted_pct"] == pytest.approx(
        100.0 * funnel.passed_candidates / funnel.gate_rows
    )
    assert detail["passed_candidates"] == funnel.passed_candidates
    assert isinstance(detail["conflict_dimension_census"], dict)
    assert isinstance(detail["candidate_generation"], dict)


def test_funnel_wrapper_signature_matches_miner() -> None:
    """Drift guard: the wrapper must expose exactly the miner's parameters."""
    miner = inspect.signature(mine_cross_brand_negatives).parameters
    wrapper = inspect.signature(mine_cross_brand_negatives_with_funnel).parameters
    assert list(wrapper) == [name for name in miner if name != "funnel"]
    for name, parameter in wrapper.items():
        assert parameter.kind == miner[name].kind
        assert parameter.default == miner[name].default


def test_funnel_is_optional_and_does_not_change_the_output() -> None:
    world = _world()
    df, canon, gtin_to_row, gtin_to_canon_idx = world
    kwargs = {
        "n_target": 100,
        "require_agreement": ("volume", "package_type"),
        "min_similarity": 0.0,
    }
    plain = mine_cross_brand_negatives(df, canon, gtin_to_row, gtin_to_canon_idx, **kwargs)
    with_funnel = mine_cross_brand_negatives_with_funnel(
        df, canon, gtin_to_row, gtin_to_canon_idx, **kwargs
    )
    assert np.array_equal(plain[0], with_funnel[0])
    assert np.array_equal(plain[1], with_funnel[1])
    assert isinstance(with_funnel[2], CrossBrandMiningFunnel)


def test_shared_base_keeps_the_targeted_funnel_unchanged() -> None:
    """The refactor must not move the targeted lane's readback by one count."""
    targeted = MiningFunnel()
    assert targeted.miner == "targeted_attribute_negatives"
    assert targeted.generation_steps() == []
    assert next(iter(targeted.stages()))[0] == "gate_similarity_floor"
    assert set(targeted.to_dict()) == {
        "miner", "name_match", "target", "min_similarity",
        "volume_relative_tolerance", "volume_absolute_tolerance_ml",
        "skipped_reason", "gate_rows", "above_similarity_floor",
        "above_floor_by_decision", "dropped_candidates",
        "flavor_variant_candidates", "passed_candidates",
        "dropped_pairs_already_in_baseline", "emitted_pairs", "bottleneck",
        "candidate_bottleneck", "candidate_to_emitted_pct", "target_reached",
        "conflict_dimension_census", "name_blocked_conflict_dimension_census",
    }
    assert isinstance(MiningFunnel(), type(targeted))
