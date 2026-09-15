from __future__ import annotations

import pandas as pd
import pytest

from core.attribute_conflicts import critical_attribute_evaluation
from core.common import rand_matching_cfg
from training.rand_matching import (
    _annotate_candidates,
    _assignments_with_trace,
    candidate_gate_fields,
    targeted_veto_gate,
)


SETTINGS = {
    "enabled": True,
    # The measured optimum: every critical dimension except sweetener, whose
    # veto costs 74 true matches to remove 6 false merges.
    "veto_dimensions": ["volume", "pack", "package_type", "flavor", "carbonation", "pulp"],
    "pack_mismatch_veto": True,
    "volume_mismatch_veto": True,
    "package_type_mismatch_veto": True,
    "brand_mismatch_veto": True,
    "missing_pack_or_volume_route": "human_review",
    "volume_relative_tolerance": 0.05,
    "volume_absolute_tolerance_ml": 5.0,
    "preserve_exact_gtin": True,
}


def _info(
    *,
    pack=(),
    volume=(),
    package_type=(),
    flavor=(),
    carbonation=(),
    sweetener=(),
    pulp=(),
) -> dict[str, object]:
    """Build the record shape the gate actually consumes.

    Two readers meet on this mapping: ``critical_attribute_evaluation`` reads
    the SET keys (``flavor_set``), while the candidate diagnostics read the
    scalar ``flavor`` display key. The helper used to declare only a scalar
    ``flavor`` string, so the gate silently reported the flavor dimension as
    missing in every case. It now mirrors the real contract for both readers.
    Optional categorical dimensions default to empty (unknown), so a case only
    asserts on the evidence it declares.
    """
    flavor_values = set(flavor)
    return {
        "pack": set(pack),
        "volume": set(volume),
        "package_type": set(package_type),
        "flavor": " ".join(sorted(flavor_values)),
        "flavor_set": flavor_values,
        "carbonation": set(carbonation),
        "carbonation_set": set(carbonation),
        "sweetener": set(sweetener),
        "sweetener_set": set(sweetener),
        "pulp": set(pulp),
        "pulp_set": set(pulp),
    }


@pytest.mark.parametrize(
    ("left", "right", "left_brand", "right_brand", "reason"),
    [
        (
            _info(pack={12}, volume={750}),
            _info(pack={6}, volume={750}),
            "Acme",
            "Acme",
            "pack_mismatch",
        ),
        (
            _info(pack={6}, volume={750}),
            _info(pack={6}, volume={1000}),
            "Acme",
            "Acme",
            "volume_mismatch",
        ),
        (
            _info(pack={6}, volume={750}),
            _info(pack={6}, volume={750}),
            "Acme",
            "Other",
            "brand_mismatch",
        ),
        (
            _info(pack={6}, volume={355}, package_type={"bottle"}),
            _info(pack={6}, volume={355}, package_type={"can"}),
            "Acme",
            "Acme",
            "package_type_mismatch",
        ),
    ],
)
def test_known_conflicts_are_hard_vetoes(left, right, left_brand, right_brand, reason):
    gate = targeted_veto_gate(
        left,
        right,
        sku_brand=left_brand,
        candidate_brand=right_brand,
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_route"] == "reject"
    assert reason in gate["targeted_gate_reason"]


def test_volume_tolerance_is_canonical_and_configurable():
    # Every critical dimension is declared on both sides so the ONLY thing this
    # case exercises is the configurable volume tolerance (780 vs 750 is 3.8%,
    # inside the 5% relative tolerance).
    common = dict(
        pack={6},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    gate = targeted_veto_gate(
        _info(volume={750}, **common),
        _info(volume={780}, **common),
        sku_brand="Acme",
        candidate_brand="acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_volume_conflict"] == 0
    assert gate["targeted_gate_route"] == "auto_merge"
    assert gate["targeted_volume_ml_a"] == "[750]"
    assert gate["targeted_volume_ml_b"] == "[780]"


def test_missing_pack_or_volume_is_human_review_not_auto_merge():
    # The candidate declares every dimension EXCEPT pack, so pack_b is the only
    # missing evidence and the missingness that routes to review is precisely
    # the one this test names.
    gate = targeted_veto_gate(
        _info(
            pack={12},
            volume={750},
            package_type={"bottle"},
            flavor={"cola"},
            carbonation={"carbonated"},
            sweetener={"no_sugar"},
            pulp={"no_pulp"},
        ),
        _info(
            volume={750},
            package_type={"bottle"},
            flavor={"cola"},
            carbonation={"carbonated"},
            sweetener={"no_sugar"},
            pulp={"no_pulp"},
        ),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "defer"
    assert gate["targeted_gate_route"] == "human_review"
    # The deferral reason names the SCOPED evidence that actually deferred, so
    # the audit trail can never claim a categorical dimension deferred a pair
    # (the setting is ``missing_pack_or_volume_route``).
    assert gate["targeted_gate_reason"] == "missing_pack_or_volume:pack_b"
    assert gate["targeted_missing_attributes"] == "pack_b"
    assert gate["targeted_missing_attribute_count"] == 1
    # Unknown evidence is NOT a conflict, and must not be reported as one.
    assert gate["targeted_critical_conflicts"] == ""
    # The audit boolean mirrors the same evaluation: partial evidence cannot
    # pass a full-evidence predicate.
    assert gate["targeted_pack_gate_pass"] == 0


def test_absent_non_decisive_dimensions_do_not_defer_an_otherwise_clean_pair():
    """Absence of a veto-only dimension is reported, not treated as deferral.

    Regression (commit 346f401 widened ``missing`` from pack/volume to all
    seven critical dimensions): on the live 1,592-pair gate-positive population
    that left 1 pair (0.06%) auto-mergeable, because ``pulp_set`` is populated
    on only ~2% of canonical records, so ``auto_merge`` was effectively dead.
    ``pipeline.three_way_gate`` labels such a pair ``proceed`` with reason
    "Known critical attributes compatible" -- unknown is neither conflict nor
    agreement there -- so deferring on it here contradicts the very label the
    candidate population is drawn from.
    """
    pair = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    # Both endpoints lose pulp/sweetener/carbonation: veto-only dimensions with
    # no evidence on either side, and no conflict anywhere.
    left = {**_info(**pair), "pulp": set(), "pulp_set": set(),
            "sweetener": set(), "sweetener_set": set()}
    right = {**_info(**pair), "pulp": set(), "pulp_set": set(),
             "sweetener": set(), "sweetener_set": set(),
             "carbonation": set(), "carbonation_set": set()}
    gate = targeted_veto_gate(
        left,
        right,
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_route"] == "auto_merge"
    assert gate["targeted_gate_decision"] == "allow"
    # The audit census still reports EVERY missing dimension, in dimension
    # order and with the flavor_set key honoured.
    assert gate["targeted_missing_attributes"] == (
        "carbonation_b,sweetener_a,sweetener_b,pulp_a,pulp_b"
    )
    assert gate["targeted_missing_attribute_count"] == 5
    assert gate["targeted_critical_conflicts"] == ""
    assert gate["targeted_pack_gate_pass"] == 0


def test_non_decisive_dimensions_still_reject_when_they_actually_conflict():
    """The scoped deferral must not weaken a real veto.

    Only the DEFERRAL scope narrowed. Every critical dimension still hard
    rejects on explicit conflict, including while another dimension's evidence
    is absent.
    """
    base = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    # sweetener is deliberately absent: the measurement put its veto at 6 false
    # merges removed against 74 TRUE MATCHES lost, so it is excluded by config
    # (see the veto_dimensions rationale). It is still audited -- asserted
    # separately below -- it simply no longer hard-blocks.
    for dimension, value, expected in (
        ("package_type", {"can"}, "package_type_mismatch"),
        ("flavor", {"orange"}, "flavor_mismatch"),
        ("carbonation", {"still"}, "carbonation_mismatch"),
        ("pulp", {"with_pulp"}, "pulp_mismatch"),
        ("pack", {12}, "pack_mismatch"),
        ("volume", {1000}, "volume_mismatch"),
    ):
        conflicted = {**base, dimension: value}
        # ``pulp`` is additionally absent on the candidate in every case, so a
        # blanket missing-deferral would have masked the veto.
        conflicted["pulp"] = conflicted.get("pulp", set())
        candidate = _info(**{**conflicted, "pulp": set() if dimension != "pulp" else value})
        gate = targeted_veto_gate(
            _info(**base),
            candidate,
            sku_brand="Acme",
            candidate_brand="Acme",
            exact_gtin=False,
            config=SETTINGS,
        )
        assert gate["targeted_gate_route"] == "reject", dimension
        assert expected in gate["targeted_gate_reason"], dimension


def test_jointly_absent_decisive_evidence_still_defers():
    """A pair with no decisive pack/volume evidence at all still defers.

    This is the case the ``missing_pack_or_volume_route`` dial exists for: the
    decision cannot be made, so it must not be made automatically.
    """
    gate = targeted_veto_gate(
        _info(),
        _info(),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "defer"
    assert gate["targeted_gate_route"] == "human_review"
    assert gate["targeted_gate_reason"] == (
        "missing_pack_or_volume:volume_a,volume_b,pack_a,pack_b"
    )
    assert gate["targeted_missing_attribute_count"] == 14


def test_deferral_scope_is_pack_and_volume():
    """Pin the scope itself so a future widening cannot silently return."""
    from training.rand_matching import DEFERRAL_DIMENSIONS

    assert DEFERRAL_DIMENSIONS == ("pack", "volume")
    # Every deferral dimension must be a real critical dimension, and the
    # reported census must cover strictly more than the deferral scope.
    from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS

    assert set(DEFERRAL_DIMENSIONS) < set(CRITICAL_ATTRIBUTE_DIMENSIONS)


# The five veto-only dimensions and every record key each one is read through.
# ``flavor`` is the asymmetric case: ``critical_attribute_evaluation`` falls back
# to the scalar display key, while the gate's census reads ``flavor_set``.
_VETO_ONLY_DIMENSION_KEYS = {
    "package_type": ("package_type",),
    "flavor": ("flavor", "flavor_set"),
    "carbonation": ("carbonation", "carbonation_set"),
    "sweetener": ("sweetener", "sweetener_set"),
    "pulp": ("pulp", "pulp_set"),
}


def _wipe(info: dict, dimension: str) -> dict:
    for key in _VETO_ONLY_DIMENSION_KEYS[dimension]:
        info[key] = set()
    return info


_VETO_ONLY_DIMENSIONS = tuple(_VETO_ONLY_DIMENSION_KEYS)


@pytest.mark.parametrize("absent", range(1 << len(_VETO_ONLY_DIMENSIONS)))
def test_no_absent_veto_only_dimension_can_ever_defer_or_enter_the_deferral_reason(absent):
    """Property pin over EVERY combination of absent veto-only dimensions.

    A blanket deferral on absence is wrong for exactly these five dimensions, so
    this asserts the property rather than one hand-picked case: wip any subset of
    ``package_type``/``flavor``/``carbonation``/``sweetener``/``pulp`` on either
    endpoint while pack and volume stay explicit and agreeing, and the pair must
    still auto-merge -- with a deferral reason that can never name a dimension
    outside ``DEFERRAL_DIMENSIONS``.

    On a revert of ``DEFERRAL_DIMENSIONS`` to all seven this fails for all 31
    non-empty subsets, which is what makes it a pin rather than a restatement of
    ``test_absent_non_decisive_dimensions_do_not_defer_an_otherwise_clean_pair``
    (one case, two dimensions). It also pins the AUDIT census independently: the
    census must still report every wiped dimension on every wiped side, so the
    two halves of the change cannot drift apart.
    """
    from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS
    from training.rand_matching import DEFERRAL_DIMENSIONS

    full = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    left, right = _info(**full), _info(**full)
    wiped: set[str] = set()
    for index, dimension in enumerate(_VETO_ONLY_DIMENSIONS):
        if absent & (1 << index):
            wiped.add(dimension)
            _wipe(left, dimension)
            _wipe(right, dimension)

    gate = targeted_veto_gate(
        left,
        right,
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )

    # (a) ABSENCE of a veto-only dimension is never deferral.
    assert gate["targeted_gate_decision"] == "allow", wiped
    assert gate["targeted_gate_route"] == "auto_merge", wiped
    assert gate["targeted_gate_reason"] == "attributes_compatible", wiped
    assert gate["targeted_critical_conflicts"] == "", wiped

    # (b) The AUDIT census still reports every absent dimension, both sides.
    census = [
        token for token in str(gate["targeted_missing_attributes"]).split(",") if token
    ]
    assert {token.rsplit("_", 1)[0] for token in census} == wiped, wiped
    assert sorted(census) == sorted(
        f"{dimension}_{side}" for dimension in wiped for side in ("a", "b")
    ), wiped
    assert gate["targeted_missing_attribute_count"] == 2 * len(wiped), wiped
    # (c) The deferral scope is unchanged by any of this.
    assert set(DEFERRAL_DIMENSIONS) == {"pack", "volume"}
    assert set(DEFERRAL_DIMENSIONS) < set(CRITICAL_ATTRIBUTE_DIMENSIONS)


def test_audit_census_names_all_seven_dimensions_while_the_deferral_scope_names_two():
    """The census contract and the routing contract are separate and both load-bearing.

    With no evidence at all on either side, ``targeted_missing_attributes`` /
    ``..._count`` must still describe the FULL critical-attribute contract (all
    seven dimensions x two endpoints = 14), because the diagnostics consume that
    census as the audit trail. The deferral REASON, by contrast, may only name
    the decisive subset — the tokens the ``missing_pack_or_volume_route`` dial
    actually governs. A revert of ``DEFERRAL_DIMENSIONS`` to all seven changes
    the reason to 14 tokens and leaves the census at 14, so asserting both
    fields together is what distinguishes a scoped deferral from a blanket one.
    """
    from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS
    from training.rand_matching import DEFERRAL_DIMENSIONS

    gate = targeted_veto_gate(
        _info(),
        _info(),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_route"] == "human_review"
    assert gate["targeted_gate_decision"] == "defer"

    census = str(gate["targeted_missing_attributes"]).split(",")
    assert len(census) == 14
    assert len(set(census)) == 14
    assert {token.rsplit("_", 1)[0] for token in census} == set(
        CRITICAL_ATTRIBUTE_DIMENSIONS
    )
    assert gate["targeted_missing_attribute_count"] == len(census) == 14

    reason_tokens = str(gate["targeted_gate_reason"]).split(":", 1)[1].split(",")
    assert {token.rsplit("_", 1)[0] for token in reason_tokens} == set(
        DEFERRAL_DIMENSIONS
    )
    assert len(reason_tokens) == 4 < len(census)


def test_live_gate_positive_population_is_not_starved_of_auto_merge():
    """The regression, pinned on the LIVE frozen artifacts.

    Every pair the training gate labelled ``proceed`` is by construction
    compatible on every dimension with explicit evidence on both sides
    (``pipeline.three_way_gate`` hard-rejects on volume/pack overlap failure),
    so the calibration gate must not take their automatic merge away. Before
    the fix this population routed 1,591/1,592 (99.94%) to human review,
    because ``pulp_set`` is populated on ~2% of canonical records.
    """
    from core.attribute_conflicts import canonical_attribute_info
    from core.common import F, canonical_records_frame

    gate_path = F["gate_results"]
    if not gate_path.exists():
        pytest.skip("frozen gate_results.csv is unavailable")
    gates = pd.read_csv(gate_path, dtype=str, keep_default_na=False)
    records = canonical_records_frame()
    by_gtin = {str(row["gtin"]): row.to_dict() for _, row in records.iterrows()}
    proceed = gates[gates["gate_decision"] == "proceed"]
    assert len(proceed) > 0, "frozen gate artifact has no proceed population"

    info_cache: dict[str, dict] = {}

    def info(gtin: str) -> dict:
        if gtin not in info_cache:
            info_cache[gtin] = canonical_attribute_info(by_gtin[gtin])
        return info_cache[gtin]

    routes = {"auto_merge": 0, "human_review": 0, "reject": 0}
    resolved = 0
    missing_census: dict[str, int] = {}
    for _, row in proceed.iterrows():
        left_gtin, right_gtin = str(row["gtin1"]), str(row["gtin2"])
        if left_gtin not in by_gtin or right_gtin not in by_gtin:
            continue
        resolved += 1
        gate = targeted_veto_gate(
            info(left_gtin),
            info(right_gtin),
            sku_brand=by_gtin[left_gtin].get("mode_brand"),
            candidate_brand=by_gtin[right_gtin].get("mode_brand"),
            exact_gtin=left_gtin == right_gtin,
        )
        routes[str(gate["targeted_gate_route"])] += 1
        for token in str(gate["targeted_missing_attributes"]).split(","):
            if token:
                missing_census[token] = missing_census.get(token, 0) + 1

    assert resolved == len(proceed)
    # No conflict can exist in this population: an explicit conflict is exactly
    # what the training gate refuses to label ``proceed``.
    assert routes["reject"] == 0
    assert routes["human_review"] == 0
    assert routes["auto_merge"] == resolved
    # The audit census is untouched by the routing scope: these absence counts
    # are what the diagnostics consume, and they must keep being reported.
    assert missing_census["pulp_a"] > 0
    assert missing_census["pulp_b"] > 0
    assert missing_census["package_type_a"] > 0


def test_all_dimensions_known_and_agreeing_is_the_cleanest_auto_merge_path():
    """Complete, agreeing evidence passes the strict audit predicate.

    Note the predicate is an AUDIT field (``targeted_pack_gate_pass``), not the
    routing rule: ``targeted_gate_route`` may also auto-merge a pair whose
    veto-only dimensions are absent, as
    ``test_absent_non_decisive_dimensions_do_not_defer_an_otherwise_clean_pair``
    pins.
    """
    full = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    gate = targeted_veto_gate(
        _info(**full),
        _info(**full),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_missing_attributes"] == ""
    assert gate["targeted_critical_conflicts"] == ""
    assert gate["targeted_pack_gate_pass"] == 1
    assert gate["targeted_gate_route"] == "auto_merge"


def test_extended_critical_dimensions_conflict_like_pack_and_volume():
    """carbonation/sweetener/pulp conflicts veto exactly like pack/volume."""
    base = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    for dimension, value, expected in (
        ("carbonation", {"still"}, "carbonation_mismatch"),
        ("pulp", {"with_pulp"}, "pulp_mismatch"),
        ("flavor", {"orange"}, "flavor_mismatch"),
    ):
        gate = targeted_veto_gate(
            _info(**base),
            _info(**{**base, dimension: value}),
            sku_brand="Acme",
            candidate_brand="Acme",
            exact_gtin=False,
            config=SETTINGS,
        )
        assert gate["targeted_gate_route"] == "reject", dimension
        assert expected in gate["targeted_gate_reason"], dimension
        assert expected.split("_")[0] in gate["targeted_critical_conflicts"], dimension


def test_a_sweetener_clash_is_audited_but_does_not_veto():
    """The measured exclusion: reported, never hidden, and selectable back.

    Measured on the 6,898 labelled holdout pairs at the operating threshold,
    the sweetener veto removes 6 false merges and 74 TRUE MATCHES -- the only
    dimension whose veto costs more than it saves. So it is out of the veto and
    still in the audit columns, and putting it back is a config edit.
    """
    base = dict(
        pack={6}, volume={750}, package_type={"bottle"}, flavor={"cola"},
        carbonation={"carbonated"}, sweetener={"no_sugar"}, pulp={"no_pulp"},
    )
    gate = targeted_veto_gate(
        _info(**base),
        _info(**{**base, "sweetener": {"sugar"}}),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_route"] != "reject"
    assert "sweetener_mismatch" not in gate["targeted_gate_reason"]
    # ...but the conflict is still on the record
    assert "sweetener" in gate["targeted_critical_conflicts"]
    assert "sweetener" not in gate["targeted_vetoed_conflicts"]

    # selecting it back restores the old hard block
    restored = targeted_veto_gate(
        _info(**base),
        _info(**{**base, "sweetener": {"sugar"}}),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config={**SETTINGS, "veto_dimensions": [*SETTINGS["veto_dimensions"], "sweetener"]},
    )
    assert restored["targeted_gate_route"] == "reject"
    assert "sweetener_mismatch" in restored["targeted_gate_reason"]


def test_absent_evidence_never_vetoes_in_any_dimension():
    """The doctrine, pinned across every dimension: absent is not a conflict.

    A gate that fired on absence would block true matches en masse -- the same
    failure mode as the text-level removal, one level up.
    """
    full = dict(
        pack={6}, volume={750}, package_type={"bottle"}, flavor={"cola"},
        carbonation={"carbonated"}, sweetener={"sugar"}, pulp={"no_pulp"},
    )
    for dimension in (
        "pack", "volume", "package_type", "flavor", "carbonation", "sweetener", "pulp",
    ):
        absent = dict(full)
        absent[dimension] = set()
        gate = targeted_veto_gate(
            _info(**full),
            _info(**absent),
            sku_brand="Acme",
            candidate_brand="Acme",
            exact_gtin=False,
            config=SETTINGS,
        )
        assert gate["targeted_gate_route"] != "reject", dimension
        assert f"{dimension}_mismatch" not in gate["targeted_gate_reason"], dimension


def test_exact_gtin_lock_bypasses_conflicts_and_missingness():
    gate = targeted_veto_gate(
        _info(pack={12}, volume={750}),
        _info(),
        sku_brand="Acme",
        candidate_brand="Other",
        exact_gtin=True,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "exact_gtin_lock"
    assert gate["targeted_gate_route"] == "auto_merge"


def test_package_type_veto_has_explicit_audit_evidence():
    gate = targeted_veto_gate(
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        _info(pack={6}, volume={355}, package_type={"can"}),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_package_type_conflict"] == 1
    assert gate["targeted_package_type_a"] == '["bottle"]'
    assert gate["targeted_package_type_b"] == '["can"]'
    assert gate["targeted_gate_decision"] == "veto"


def test_bottle_sku_to_can_candidate_is_rejected_end_to_end():
    row = pd.Series(
        {
            "SKU_ID": "bottle-sku",
            "barcode": "",
            "title": "Acme soda 6 pack 12 oz bottles",
            "brand": "Acme",
        }
    )
    candidate = candidate_gate_fields(
        row,
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        "candidate-can",
        {
            "canonical": "Acme soda 6 pack 355 ml cans",
            "mode_brand": "Acme",
            "mode_flavor": "",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        },
        0.99,
        sku_id="bottle-sku",
        source_row_index="0",
    )
    assert candidate["attribute_conflict_type"] == "package_type"
    assert candidate["rule_ok"] == 0
    assert candidate["targeted_gate_route"] == "reject"

    trace = _annotate_candidates(pd.DataFrame([candidate]), 0.61)
    assert not bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["rejection_reason"] == "targeted_attribute_veto"


def test_exact_gtin_package_type_disagreement_uses_configured_lock():
    gtin = "4006381333931"
    row = pd.Series(
        {
            "SKU_ID": "exact-bottle",
            "barcode": gtin,
            "title": "Acme soda 6 pack 12 oz bottles",
            "brand": "Acme",
        }
    )
    candidate = candidate_gate_fields(
        row,
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        gtin,
        {
            "canonical": "Acme soda 6 pack 355 ml cans",
            "mode_brand": "Acme",
            "mode_flavor": "",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        },
        0.10,
        sku_id="exact-bottle",
        source_row_index="0",
    )
    assert candidate["targeted_gate_decision"] == "exact_gtin_lock"
    assert candidate["rule_ok"] == 0
    trace = _annotate_candidates(pd.DataFrame([candidate]), 0.99)
    assert bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["attribute_gate"] == "override_exact_gtin"


def _candidate(
    sku: str,
    score: float,
    *,
    route: str,
    exact: int = 0,
    status: str = "different",
) -> dict[str, object]:
    return {
        "SKU_ID": sku,
        "candidate_gtin": f"candidate-{sku}",
        "score": score,
        "gtin_status": status,
        "exact_gtin": exact,
        "rule_ok": 1,
        "attribute_matches": 3,
        "brand_conflict": 0,
        "targeted_gate_route": route,
    }


def test_weakest_06004_edge_fails_new_threshold():
    candidates = pd.DataFrame([_candidate("weak", 0.6004, route="auto_merge")])
    trace = _annotate_candidates(
        candidates,
        0.61,
    )
    assert float(trace.iloc[0]["effective_threshold"]) == pytest.approx(0.61)
    assert not bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["rejection_reason"] == "below_threshold"


def test_weak_missing_edge_is_routed_to_review_and_never_selected():
    candidates = pd.DataFrame([_candidate("weak", 0.6004, route="human_review")])
    predictions, trace = _assignments_with_trace(candidates, 0.60)
    assert trace.iloc[0]["assignment_gate"] == "human_review_candidate"
    assert trace.iloc[0]["rejection_reason"] == "human_review_missing_pack_or_volume"
    assert predictions.iloc[0]["ITEM_ID"].startswith("UNMATCHED_")


def test_061_is_monotonic_and_cannot_increase_overmerge_edges():
    candidates = pd.DataFrame(
        [
            _candidate("weak", 0.6004, route="auto_merge"),
            _candidate("veto", 0.90, route="reject"),
            _candidate("review", 0.95, route="human_review"),
            _candidate("strong", 0.82, route="auto_merge"),
        ]
    )
    at_060 = _annotate_candidates(candidates, 0.60)
    at_061 = _annotate_candidates(candidates, 0.61)
    accepted_060 = set(at_060.loc[at_060["accepted"], "SKU_ID"])
    accepted_061 = set(at_061.loc[at_061["accepted"], "SKU_ID"])
    assert accepted_061 <= accepted_060
    assert accepted_060 - accepted_061 == {"weak"}
    assert accepted_061 == {"strong"}


def test_060_override_is_bumped_without_lowering_other_strata():
    thresholds = rand_matching_cfg()["threshold_by_gtin_status"]
    assert thresholds["both_equal"] == pytest.approx(0.80)
    assert thresholds["different"] == pytest.approx(0.80)
    assert thresholds["one_missing"] == pytest.approx(0.61)
    assert thresholds["both_missing"] == pytest.approx(0.80)


def test_every_hard_no_gate_reason_maps_to_a_known_family():
    """Every reason the gate can emit on a HARD_NO must be classifiable.

    Only hard_no rows enter the balanced pool, so these are the reasons that
    must resolve. A new gate_reason string without a matching prefix makes
    sample_balanced_pairs raise "Unmatched hard-negative gate_reason values"
    and the whole balanced-pool lane hard-fails on fresh artifacts — which is
    exactly what the "Pack blocker:" reason did.
    """
    from training.sample_balanced_pairs import _reason_type

    hard_no_reasons = {
        # The composite blocker introduced with the shared critical-attribute
        # gate: highest-volume hard_no reason in the committed artifact
        # (87,804 of 135,769 rows) and the one that hard-failed the lane.
        "Pack blocker: pack size, package type, or volume mismatch",
        # Legacy single-dimension reasons, still emitted by the tail checks.
        "No volume overlap",
        "No pack overlap",
        "Package type mismatch",
        "Package material mismatch",
        "Flavor mismatch: lemon vs orange",
        # Categorical conflicts arrive as "Critical attribute mismatch: <dims>".
        "Critical attribute mismatch: flavor",
        "Critical attribute mismatch: carbonation,sweetener",
    }
    for reason in sorted(hard_no_reasons):
        assert _reason_type(reason) is not None, reason

    # Non-hard_no outcomes never reach the pool classifier, so they are
    # deliberately NOT required to resolve.
    for reason in ("Known critical attributes compatible", "Low raw volume confidence"):
        assert _reason_type(reason) is None, reason


def test_targeted_miner_never_emits_a_same_canonical_negative():
    """A same-canonical pair is a TRUE MATCH, never a label-0 row.

    build_training_data drops exactly those pairs for the baseline negatives
    and counts them as n_neg_same_canonical_dropped; the targeted miner must
    apply the same identity rule instead of re-adding them.
    """
    import numpy as np
    import pandas as pd

    from core.attribute_conflicts import canonical_attribute_info
    from core.hard_negatives import mine_targeted_attribute_negatives

    df = pd.DataFrame(
        {"title": ["Acme Cola 12 pack 355ml", "Acme Cola 12 pack 355ml"]}
    )
    canon_text = "acme cola still sugar"
    canonical_records = pd.DataFrame(
        [
            {
                "gtin": "A",
                "canonical": canon_text,
                "mode_brand": "acme",
                "mode_flavor": "",
                "volume_set": "[355]",
                "pack_set": "[12]",
                "package_type_set": "[]",
            },
            {
                "gtin": "B",
                "canonical": canon_text,
                "mode_brand": "acme",
                "mode_flavor": "",
                "volume_set": "[1000]",
                "pack_set": "[12]",
                "package_type_set": "[]",
            },
        ]
    )
    gates = pd.DataFrame(
        [{"gtin1": "A", "gtin2": "B", "similarity": 0.90, "gate_decision": "hard_no"}]
    )
    gtin_to_row = {"A": 0, "B": 1}
    gtin_to_canon_idx = {"A": 100, "B": 101}
    canonical_map = {"A": canon_text, "B": canon_text}

    # Sanity: the conflict evaluator DOES see a volume conflict here, so only
    # the same-canonical guard can prevent the pair from being mined.
    assert "volume" in critical_attribute_evaluation(
        canonical_attribute_info(canonical_records.iloc[0].to_dict()),
        canonical_attribute_info(canonical_records.iloc[1].to_dict()),
    )["conflicts"]

    guarded, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        canonical_map=canonical_map,
    )
    assert len(guarded) == 0

    unguarded, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
    )
    assert len(unguarded) == 2  # both directions — the label inversion


def test_targeted_miner_respects_the_gate_volume_tolerance():
    """A within-tolerance volume gap is agreement, not a mineable conflict."""
    import pandas as pd

    from core.hard_negatives import mine_targeted_attribute_negatives

    df = pd.DataFrame({"title": ["Acme Juice 1L", "Acme Juice 1L"]})
    base = {
        "canonical": "acme juice still",
        "mode_brand": "acme",
        "mode_flavor": "",
        "pack_set": "[6]",
        "package_type_set": "[]",
    }
    canonical_records = pd.DataFrame(
        [
            {**base, "gtin": "A", "volume_set": "[480]"},
            {**base, "gtin": "B", "volume_set": "[500]"},
        ]
    )
    gates = pd.DataFrame(
        [{"gtin1": "A", "gtin2": "B", "similarity": 0.90, "gate_decision": "proceed"}]
    )
    gtin_to_row = {"A": 0, "B": 1}
    gtin_to_canon_idx = {"A": 100, "B": 101}

    strict, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        volume_relative_tolerance=0.0,
    )
    assert len(strict) == 2  # 480 vs 500 is a conflict at exact-match tolerance

    tolerant, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        volume_relative_tolerance=0.05,  # the gate's own tolerance
    )
    assert len(tolerant) == 0
