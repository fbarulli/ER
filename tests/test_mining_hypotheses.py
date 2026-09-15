"""Hypothesis tests for the targeted-negative miner and the balanced pool.

Each test here is a REGRESSION PIN for a measured defect (or for a measured
non-defect that a future edit could silently reintroduce). The live-artifact
numbers behind every pin are in the audit report; the pins themselves use toy
worlds so the suite stays fast (base suite: 72 tests, ~24 s).

Measured on the live artifacts at commit 2cfd772 (135,769 gate rows,
43,780 candidates above the 0.50 similarity floor):

* strict name equality dropped 40,269 of 43,209 candidates reaching it (93.2%)
  and made flavour conflicts structurally unreachable — 5,549 flavour-conflict
  candidates above the floor, 0 emitted, while the gate itself labels 876 rows
  "Critical attribute mismatch: flavor";
* ``normalized_product_name`` leaked fused pack notation ("18x33cl"), which
  blocked 115 conflict candidates by name equality alone;
* the miner emitted 196 pairs against a configured target of 12,000.

The opt-in live check at the bottom re-measures the funnel on the real
artifacts: ``ER_LIVE_MINING_EVIDENCE=1 pytest tests/test_mining_hypotheses.py``.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.hard_negatives import (
    MiningFunnel,
    flavor_variant_product_name,
    mine_targeted_attribute_negatives,
    mine_targeted_attribute_negatives_with_funnel,
    normalized_product_name,
)
from training.sample_balanced_pairs import (
    REASON_PREFIX_TO_TYPE,
    _reason_type,
    build_outputs,
)

REPO = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO / "src" / "training" / "train.py"


# ── toy world ──────────────────────────────────────────────────────────────
def _records(brand: str, volume: float, flavor: str, canonical: str) -> dict:
    return {
        "mode_brand": brand,
        "volume_set": [float(volume)],
        "flavor_set": [flavor],
        "canonical": canonical,
        "mode_flavor": flavor,
    }


def _world(
    *,
    left_title: str,
    right_title: str,
    left: dict,
    right: dict,
    gate_decision: str = "hard_no",
    similarity: float = 0.70,
    left_gtin: str = "1111111111111",
    right_gtin: str = "2222222222222",
    left_canonical: str | None = None,
    right_canonical: str | None = None,
):
    """Build the minimal real input set the miner consumes."""
    df = pd.DataFrame(
        {
            "barcode": [left_gtin, right_gtin],
            "title": [left_title, right_title],
            "attributes": ["", ""],
        }
    )
    gates = pd.DataFrame(
        {
            "gtin1": [left_gtin],
            "gtin2": [right_gtin],
            "gate_decision": [gate_decision],
            "gate_reason": ["Pack blocker: pack size, package type, or volume mismatch"],
            "similarity": [float(similarity)],
        }
    )
    records = []
    for gtin, info in ((left_gtin, left), (right_gtin, right)):
        records.append({"gtin": gtin, **info})
    canonical_records = pd.DataFrame(records)
    canonical_map = {
        left_gtin: left_canonical if left_canonical is not None else str(left["canonical"]),
        right_gtin: right_canonical if right_canonical is not None else str(right["canonical"]),
    }
    return dict(
        df=df,
        gates=gates,
        canonical_records=canonical_records,
        gtin_to_row={left_gtin: 0, right_gtin: 1},
        gtin_to_canon_idx={left_gtin: 2, right_gtin: 3},
        canonical_map=canonical_map,
    )


def _mine(world, **kwargs):
    kwargs.setdefault("n_target", 100)
    kwargs.setdefault("min_similarity", 0.5)
    kwargs.setdefault("volume_relative_tolerance", 0.05)
    return mine_targeted_attribute_negatives_with_funnel(**world, **kwargs)


# ── H1: additive-vs-substitutive label conflict ────────────────────────────
def test_same_canonical_text_is_never_emitted_as_a_negative() -> None:
    """H1: the guard compares canonical STRINGS, so equal canonicals never leak.

    Two distinct GTINs whose canonical text is byte-identical are the same
    item; emitting one as the other's label-0 target is the contradiction the
    hypothesis describes. The miner must drop the candidate outright.
    """
    world = _world(
        left_title="acme cola cherry 500ml",
        right_title="acme cola cherry 500ml",
        left=_records("acme", 500.0, "cherry", "acme cola cherry 500"),
        right=_records("acme", 1500.0, "cherry", "acme cola cherry 1500"),
        left_canonical="acme cola cherry",  # identical canonical STRING
        right_canonical="acme cola cherry",
    )
    pairs, _scores, funnel = _mine(world)
    assert len(pairs) == 0
    assert funnel.dropped_candidates_same_canonical == 1
    assert funnel.emitted_pairs == 0


def test_emitted_negative_never_reuses_the_source_gtin_canonical() -> None:
    """Every emitted label-0 pair points at a DIFFERENT canonical string."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    pairs, _scores, _funnel = _mine(world)
    assert len(pairs) == 2
    cmap = world["canonical_map"]
    idx_to_gtin = {v: k for k, v in world["gtin_to_canon_idx"].items()}
    for anchor, target in pairs:
        source_gtin = world["df"]["barcode"].iloc[int(anchor)]
        target_gtin = idx_to_gtin[int(target)]
        assert cmap[source_gtin] != cmap[target_gtin]


# ── H2: miner starvation / unreachable dimensions ──────────────────────────
def test_strict_name_equality_makes_flavour_conflicts_unreachable() -> None:
    """H2: with name_match="exact" a flavour conflict can NEVER be emitted."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    pairs, _scores, funnel = _mine(world, name_match="exact")
    assert len(pairs) == 0
    assert funnel.dropped_candidates_name == 1
    assert "flavor" not in funnel.conflict_dimension_census


def test_flavor_variant_rule_emits_the_gate_confirmed_pair() -> None:
    """H2 fix: same name modulo flavour + gate hard_no + explicit flavour conflict."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    pairs, _scores, funnel = _mine(world)
    assert len(pairs) == 2
    assert funnel.flavor_variant_candidates == 1
    assert funnel.conflict_dimension_census.get("flavor") == 1
    assert funnel.emitted_pairs == 2


def test_flavor_variant_rule_never_relabels_a_proceed_row() -> None:
    """The gate is the label authority: a compatible pair stays out."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
        gate_decision="proceed",
    )
    pairs, _scores, funnel = _mine(world)
    assert len(pairs) == 0
    assert funnel.dropped_candidates_name == 1


def test_flavor_variant_rule_requires_a_real_flavour_conflict() -> None:
    """Names differing by a flavour word are not enough on their own."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        # SAME flavour evidence on both sides: the only real conflict is volume.
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 750.0, "pear", "acme soda pear 750"),
    )
    pairs, _scores, funnel = _mine(world)
    assert len(pairs) == 0
    assert funnel.dropped_candidates_name == 1


def test_flavor_variant_strip_only_removes_lexicon_words() -> None:
    assert flavor_variant_product_name("soda pear zero") == "soda zero"
    assert flavor_variant_product_name("soda") == "soda"
    assert flavor_variant_product_name(None) == ""


def test_unknown_name_match_mode_fails_loudly() -> None:
    world = _world(
        left_title="a",
        right_title="b",
        left=_records("acme", 330.0, "pear", "x"),
        right=_records("acme", 750.0, "pear", "y"),
    )
    with pytest.raises(ValueError, match="name_match must be"):
        _mine(world, name_match="fuzzy")


# ── H2b: fused pack notation leak in the name key ──────────────────────────
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("pear soda 18x33cl", "pear soda"),
        ("soda 12x330ml", "soda"),
        ("water 6x1.5l", "water"),
        ("water 6 x 1.5 l", "water"),
        ("acme cola 1.5 l pack of 6", "cola of 6"),  # pre-existing leftover, see report
    ],
)
def test_normalized_product_name_removes_fused_pack_notation(title: str, expected: str) -> None:
    assert normalized_product_name(title, "acme") == expected


def test_fused_pack_pair_is_reachable_by_name_equality() -> None:
    """The pack-conflict pair the fused token used to hide."""
    world = _world(
        left_title="acme mineral water 6x1.5l",
        right_title="acme mineral water 12x1.5l",
        left=_records("acme", 1500.0, "none", "acme water 1500"),
        right=_records("acme", 2000.0, "none", "acme water 2000"),
    )
    pairs, _scores, funnel = _mine(world)
    assert len(pairs) == 2
    assert funnel.conflict_dimension_census.get("volume") == 1


# ── H3: target unreachability ──────────────────────────────────────────────
def test_funnel_declares_an_unreachable_target() -> None:
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    _pairs, _scores, funnel = _mine(world, n_target=12000)
    readback = funnel.to_dict()
    assert readback["target"] == 12000
    assert readback["emitted_pairs"] == 2
    assert readback["target_reached"] is False
    # The knob's own capacity claim is now readable instead of implied: with a
    # single candidate the lane saturates at 2 of 12,000 and says so.
    assert readback["candidate_to_emitted_pct"] == 100.0
    assert readback["candidate_bottleneck"] in {
        step for step, _, _, _ in funnel.stages()
    }


def test_funnel_stages_are_cumulative_and_monotone() -> None:
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    stages = _mine(world)[2].stages()
    assert [stage[0] for stage in stages] == [
        "gate_similarity_floor",
        "canonical_records_resolution",
        "representative_row_resolution",
        "canonical_index_resolution",
        "same_canonical_guard",
        "brand_equality",
        "product_name_equality",
        "critical_attribute_conflict",
        "direction_expansion",
        "baseline_deduplication",
        "emitted",
    ]
    for step, incoming, outgoing, reason in stages:
        assert isinstance(step, str) and reason
        assert outgoing <= incoming or step == "direction_expansion"
        assert incoming >= 0
    for (_, _, previous_out, _), (_, next_in, _, _) in zip(stages, stages[1:]):
        assert previous_out == next_in


# ── H4: balanced-pool lane health ──────────────────────────────────────────
def _gate_csv(path: Path, positives: int, negatives: int, reason: str) -> Path:
    rows = [
        {
            "gtin1": f"{1000000 + i:07d}",
            "gtin2": f"{2000000 + i:07d}",
            "gate_decision": "proceed",
            "gate_reason": "Known critical attributes compatible",
            "similarity": 0.9,
        }
        for i in range(positives)
    ]
    rows += [
        {
            "gtin1": f"{3000000 + i:07d}",
            "gtin2": f"{4000000 + i:07d}",
            "gate_decision": "hard_no",
            "gate_reason": reason,
            "similarity": 0.9,
        }
        for i in range(negatives)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def _sku_csv(path: Path, gtins: set[str]) -> Path:
    pd.DataFrame(
        {
            "barcode": sorted(gtins),
            "product_id": [f"sku-{g}" for g in sorted(gtins)],
        }
    ).to_csv(path, index=False)
    return path


def _all_gtins(gate_path: Path) -> set[str]:
    gate = pd.read_csv(gate_path, dtype=str, keep_default_na=False)
    return set(gate["gtin1"]) | set(gate["gtin2"])


def _run_pool(tmp_path: Path, gate_path: Path, size: int = 3000) -> dict:
    return build_outputs(
        gate_path=gate_path,
        sku_path=_sku_csv(tmp_path / "sku.csv", _all_gtins(gate_path)),
        balanced_path=tmp_path / "balanced.csv",
        sample_path=tmp_path / "sample.csv",
        manifest_path=tmp_path / "manifest.json",
        sweep_path=tmp_path / "sweep.csv",
        sample_size=size,
        seed=42,
        allow_unmatched_types=False,
    )


def test_composite_pack_blocker_reason_is_registered() -> None:
    """"Pack blocker:" must map to a family, not abort the lane."""
    assert ("Pack blocker:", "pack_blocker") in REASON_PREFIX_TO_TYPE
    assert _reason_type("Pack blocker: pack size, package type, or volume mismatch") == (
        "pack_blocker"
    )


def test_balanced_pool_ceiling_is_bounded_by_the_positives(tmp_path: Path) -> None:
    """H4: the pool is 2*min(positives, typed negatives) — the POSITIVES bind."""
    gate = _gate_csv(tmp_path / "gate.csv", positives=3, negatives=40,
                     reason="Pack blocker: pack size, package type, or volume mismatch")
    with pytest.raises(ValueError, match="balanced pool has only 6") as failure:
        _run_pool(tmp_path, gate)
    # The message must name the binding side: the composite reason family is
    # NOT the constraint, and misattributing it sent the audit the wrong way.
    assert "bound by eligible positives" in str(failure.value)
    assert "reason families are not the constraint" in str(failure.value)
    # Adding negatives cannot move the ceiling: the positive side binds.
    gate2 = _gate_csv(tmp_path / "gate2.csv", positives=3, negatives=200,
                      reason="Pack blocker: pack size, package type, or volume mismatch")
    with pytest.raises(ValueError, match="balanced pool has only 6"):
        _run_pool(tmp_path, gate2)


def test_balanced_pool_builds_when_both_sides_are_sufficient(tmp_path: Path) -> None:
    gate = _gate_csv(tmp_path / "gate.csv", positives=6, negatives=40,
                     reason="Pack blocker: pack size, package type, or volume mismatch")
    manifest = _run_pool(tmp_path, gate, size=4)
    accounting = manifest["accounting"]
    assert accounting["balanced_rows"] == 12
    assert accounting["sample_rows"] == 4
    for pair_type, counts in accounting["balanced_by_type_and_label"].items():
        assert counts["0"] == counts["1"], pair_type


# ── H5: provenance integrity of the training append/guard blocks ───────────
def _provenance_blocks() -> dict[str, str]:
    """Extract train.py's REAL provenance statements (no hand copy)."""
    tree = ast.parse(TRAIN_PY.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_main_inner"
    )
    wanted: dict[str, ast.stmt] = {}
    for node in fn.body:
        text = ast.unparse(node)
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if target == "train_neg" and text == "train_neg = neg":
                wanted["alias"] = node
            elif target == "neg_sources":
                wanted["sources_init"] = node
            elif target == "train_neg_sources":
                wanted["sources_copy"] = node
            elif target == "balance_train_classes":
                wanted["balance_flag"] = node
        elif isinstance(node, ast.If):
            if ast.unparse(node.test) == "balance_train_classes":
                # first: this block also mentions train_neg_sources and raises
                wanted["balance"] = node
            elif "len(neg_sources) != len(neg)" in text:
                wanted["guard"] = node
            elif "_attr_neg = np.empty" in text:
                wanted["attr_mine"] = node
            elif "np.vstack([neg, targeted_attribute_neg])" in text:
                wanted["targeted"] = node
            elif "np.vstack([neg, _attr_neg])" in text:
                wanted["attr"] = node
    order = [
        "alias", "sources_init", "sources_copy", "targeted", "attr_mine",
        "attr", "guard", "balance_flag", "balance",
    ]
    missing = [key for key in order if key not in wanted]
    assert not missing, f"train.py provenance blocks not found: {missing}"
    return {key: ast.unparse(wanted[key]) for key in order}


def _run_provenance_case(
    blocks: dict[str, str], *, n_gate: int, n_targeted: int, n_attr: int,
    balance: bool, n_pos_pairs: int,
) -> dict:
    """Execute the real blocks in a sandbox namespace for one branch combo."""
    import core.hard_negatives as hard_negatives

    calls: list[int] = []

    def fake_attr_miner(_df, _payload, _row_bc, _emb, *, existing=None, n_target=0, **_kw):
        calls.append(len(existing) if existing is not None else -1)
        pairs = np.asarray(
            [(1000 + i, 2000 + i) for i in range(n_attr)], dtype=int
        ).reshape(-1, 2)
        return pairs, np.zeros(len(pairs))

    original = hard_negatives.mine_attribute_conflict_negatives
    hard_negatives.mine_attribute_conflict_negatives = fake_attr_miner
    namespace: dict[str, object] = {
        "np": np,
        "SEED": 42,
        "df": "df-sentinel",
        "payload": ["p"] * (2 * n_pos_pairs),
        "row_bc": np.arange(2 * n_pos_pairs),
        "emb0": np.zeros((4, 4)) if n_attr else np.zeros((0, 0)),
        "attribute_conflict_enabled": True,
        "attr_cfg": {"band": "0.50-0.95", "target": 12000},
        "targeted_attribute_neg": np.asarray(
            [(i, 100 + i) for i in range(n_targeted)], dtype=int
        ).reshape(-1, 2),
        "neg": np.asarray(
            [(i, 500 + i) for i in range(n_gate)], dtype=int
        ).reshape(-1, 2),
        "pos": np.arange(2 * n_pos_pairs).reshape(-1, 2),
        "load_config": lambda: {
            "pairs": {"balance_train_classes": balance},
            "gate": {"vol_tolerance": 0.05},
        },
        "mining_enabled": True,
        "print": lambda *a, **k: None,
    }
    try:
        for key in (
            "alias", "sources_init", "sources_copy", "targeted", "attr_mine",
            "attr", "guard", "balance_flag", "balance",
        ):
            exec(compile(blocks[key], f"<train.py:{key}>", "exec"), namespace)  # noqa: S102
    finally:
        hard_negatives.mine_attribute_conflict_negatives = original
    namespace["_attr_calls"] = calls
    return namespace


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("masking_only", dict(n_gate=6, n_targeted=0, n_attr=0, balance=False, n_pos_pairs=5)),
        ("targeted_only", dict(n_gate=6, n_targeted=3, n_attr=0, balance=False, n_pos_pairs=5)),
        ("attr_only", dict(n_gate=6, n_targeted=0, n_attr=2, balance=False, n_pos_pairs=5)),
        ("targeted_and_attr", dict(n_gate=6, n_targeted=3, n_attr=2, balance=False, n_pos_pairs=5)),
        ("empty_baseline", dict(n_gate=0, n_targeted=3, n_attr=0, balance=False, n_pos_pairs=5)),
        ("empty_all", dict(n_gate=0, n_targeted=0, n_attr=0, balance=False, n_pos_pairs=5)),
        ("balanced_short", dict(n_gate=6, n_targeted=3, n_attr=2, balance=True, n_pos_pairs=25)),
        ("balanced_long", dict(n_gate=6, n_targeted=3, n_attr=2, balance=True, n_pos_pairs=2)),
    ],
)
def test_negative_provenance_stays_aligned_and_labelled(label: str, kwargs: dict) -> None:
    """H5: every negative row carries its own source, in every branch combo."""
    namespace = _run_provenance_case(_provenance_blocks(), **kwargs)
    neg, sources = namespace["neg"], namespace["neg_sources"]
    train_neg, train_sources = namespace["train_neg"], namespace["train_neg_sources"]
    n_gate, n_targeted, n_attr = kwargs["n_gate"], kwargs["n_targeted"], kwargs["n_attr"]

    assert len(sources) == len(neg)
    assert len(train_sources) == len(train_neg)
    assert list(sources) == (
        ["gate"] * n_gate
        + ["targeted_attribute_conflict"] * n_targeted
        + ["attribute_conflict"] * n_attr
    )
    # Identity, not just length: each training row keeps the label of ITS pair.
    label_by_pair = {
        tuple(int(x) for x in pair): source for pair, source in zip(neg, sources)
    }
    assert [label_by_pair[tuple(int(x) for x in pair)] for pair in train_neg] == list(
        train_sources
    )


# ── H6: traceability of the miner ──────────────────────────────────────────
def test_funnel_wrapper_signature_matches_miner() -> None:
    """Drift guard: the wrapper must expose exactly the miner's parameters."""
    miner = inspect.signature(mine_targeted_attribute_negatives).parameters
    wrapper = inspect.signature(mine_targeted_attribute_negatives_with_funnel).parameters
    assert list(wrapper) == [name for name in miner if name != "funnel"]
    for name, parameter in wrapper.items():
        assert parameter.kind == miner[name].kind
        assert parameter.default == miner[name].default


def test_funnel_is_optional_and_does_not_change_the_output() -> None:
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    plain = mine_targeted_attribute_negatives(
        **world, n_target=100, min_similarity=0.5, volume_relative_tolerance=0.05
    )
    assert isinstance(plain, tuple) and len(plain) == 2
    with_funnel = _mine(world)
    assert np.array_equal(plain[0], with_funnel[0])
    assert np.array_equal(plain[1], with_funnel[1])
    assert isinstance(with_funnel[2], MiningFunnel)


def test_funnel_readback_is_trace_ready() -> None:
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    funnel = _mine(world)[2]
    detail = json.loads(json.dumps(funnel.to_dict()))
    assert detail["miner"] == "targeted_attribute_negatives"
    assert detail["name_match"] == "flavor_variant"
    assert set(detail["dropped_candidates"]) == {
        "gate_similarity_floor",
        "canonical_records_resolution",
        "representative_row_resolution",
        "canonical_index_resolution",
        "same_canonical_guard",
        "brand_equality",
        "product_name_equality",
        "critical_attribute_conflict",
    }
    assert detail["passed_candidates"] == 1
    steps = funnel.stages()
    assert len(steps) == 11
    assert steps[-1][0] == "emitted" and steps[-1][2] == detail["emitted_pairs"]


def test_funnel_name_blocked_census_exposes_the_starving_dimensions() -> None:
    """The census that answered H2: what the name filter silently blocks."""
    world = _world(
        left_title="acme soda pear 33cl",
        right_title="acme soda raspberry 33cl",
        left=_records("acme", 330.0, "pear", "acme soda pear 330"),
        right=_records("acme", 330.0, "raspberry", "acme soda raspberry 330"),
    )
    funnel = _mine(world, name_match="exact")[2]
    assert funnel.name_blocked_conflict_dimension_census.get("flavor") == 1
    assert funnel.conflict_dimension_census == {}


# ── opt-in live evidence ───────────────────────────────────────────────────
@pytest.mark.skipif(
    not os.environ.get("ER_LIVE_MINING_EVIDENCE"),
    reason="set ER_LIVE_MINING_EVIDENCE=1 to re-measure the funnel on live artifacts",
)
def test_live_funnel_accounting_closes() -> None:
    """Re-measure the real funnel; every count must close."""
    from core.common import DATA_DIR, RESULTS, F
    from core.hard_negatives import mine_targeted_attribute_negatives_with_funnel

    df = pd.read_csv(DATA_DIR / "dataset_deduped.csv", dtype=str, keep_default_na=False)
    gates = pd.read_csv(
        RESULTS / F["gate_results"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    canonical_records = pd.read_csv(
        RESULTS / F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
    )
    bc = df["barcode"].fillna("").astype(str).str.strip()
    best: dict[str, int] = {}
    for index, barcode in enumerate(bc):
        if barcode and (barcode not in best or len(df["title"].iloc[index]) > len(df["title"].iloc[best[barcode]])):
            best[barcode] = index
    gtin_to_canon_idx = {gtin: i for i, gtin in enumerate(sorted(best))}
    gtin_to_row = {gtin: row for gtin, row in best.items()}

    plain, _plain_scores, funnel = mine_targeted_attribute_negatives_with_funnel(
        df, gates, canonical_records, gtin_to_row, gtin_to_canon_idx,
        n_target=12000, min_similarity=0.5, volume_relative_tolerance=0.05,
    )
    strict = mine_targeted_attribute_negatives(
        df, gates, canonical_records, gtin_to_row, gtin_to_canon_idx,
        n_target=12000, min_similarity=0.5, volume_relative_tolerance=0.05,
        name_match="exact",
    )[0]
    stages = funnel.stages()
    for (_, _, previous_out, _), (_, next_in, _, _) in zip(stages, stages[1:]):
        assert previous_out == next_in
    assert funnel.emitted_pairs == len(plain)
    assert 2 * funnel.passed_candidates - funnel.dropped_pairs_already_in_baseline == len(plain)
    strict_pairs = {tuple(int(x) for x in pair) for pair in strict}
    relaxed_pairs = {tuple(int(x) for x in pair) for pair in plain}
    assert strict_pairs <= relaxed_pairs, "the name rule must only ADD pairs"


@pytest.mark.skipif(
    not os.environ.get("ER_LIVE_MINING_EVIDENCE"),
    reason="set ER_LIVE_MINING_EVIDENCE=1 to re-measure label integrity on live artifacts",
)
def test_live_no_text_pair_carries_both_labels() -> None:
    """H1 live measurement: no (text_a, text_b) tuple appears as both labels.

    Mirrors ``pipeline.build_training_data``'s payload construction and pair
    resolution (read-only) so the count is over the texts the MODEL ingests,
    not over canonical ids. Measured 0 at commit 2cfd772 and 0 again with the
    audit's miner changes, against 23,370 positives + 14,829 baseline
    negatives + the mined targeted pairs.
    """
    import pipeline as pipeline_module
    from core.common import DATA_DIR, RESULTS, F, load_config
    from core.hard_negatives import mine_targeted_attribute_negatives_with_funnel
    from core.model_input import (
        build_canonical_text,
        build_sku_text,
        model_input_info,
    )
    from core.structured_features import (
        canonical_info as canonical_structured_info,
        sku_info as sku_structured_info,
    )

    cfg = load_config()
    structured = bool(cfg["training"]["structured_features"]["enabled"])
    df = pd.read_csv(DATA_DIR / "dataset_deduped.csv", dtype=str, keep_default_na=False)
    gates = pd.read_csv(
        RESULTS / F["gate_results"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    canonical_records = pd.read_csv(
        RESULTS / F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
    )
    canon_map = pipeline_module.load_canonical_map()
    thr_neg = float(cfg["pairs"]["hardneg_sim_threshold"])

    # Built through the SAME shared builder the payload stage uses, so this
    # measurement keeps describing the texts the MODEL ingests when the
    # composition profile changes. It used to re-implement the composition,
    # which silently made the claim false after the default moved to `cleaned`.
    def sku_text(index: int) -> str:
        row = df.iloc[index]
        info = model_input_info(
            sku_structured_info(row["title"], row["attributes"])
        ) if structured else {}
        return build_sku_text(row, info)

    record_map = {str(r["gtin"]): r.to_dict() for _, r in canonical_records.iterrows()}
    canon_gtins = sorted(canon_map)
    gtin_to_canon_idx = {gtin: i for i, gtin in enumerate(canon_gtins)}
    canon_text = {
        gtin: build_canonical_text(
            record_map.get(gtin, {}),
            model_input_info(canonical_structured_info(record_map.get(gtin, {})))
            if structured
            else {},
        )
        for gtin in canon_gtins
    }

    bc = df["barcode"].fillna("").astype(str).str.strip()
    t_len = df["title"].fillna("").astype(str).str.len().to_numpy()
    order = np.lexsort((np.arange(len(t_len)), -t_len))
    gtin_to_row: dict[str, int] = {}
    for index in order:
        barcode = bc.iloc[index]
        if barcode and barcode not in gtin_to_row:
            gtin_to_row[barcode] = int(index)

    def key(left: int, right: int) -> tuple[str, str]:
        a, b = sku_text(left), canon_text[right]
        return (a, b) if a <= b else (b, a)

    labelled: dict[tuple[str, str], set[int]] = {}
    for index, barcode in enumerate(bc):
        if barcode in gtin_to_canon_idx:
            labelled.setdefault(key(index, canon_gtins[gtin_to_canon_idx[barcode]]), set()).add(1)
    same_canonical = (
        gates["gtin1"].map(canon_map).notna()
        & gates["gtin2"].map(canon_map).notna()
        & gates["gtin1"].map(canon_map).eq(gates["gtin2"].map(canon_map))
    )
    negatives = gates[
        (gates["gate_decision"] == "hard_no")
        & (pd.to_numeric(gates["similarity"], errors="coerce") >= thr_neg)
        & ~same_canonical
    ]
    for row in negatives.itertuples(index=False):
        left, right = str(row.gtin1), str(row.gtin2)
        if left in gtin_to_row and right in gtin_to_canon_idx:
            labelled.setdefault(key(gtin_to_row[left], right), set()).add(0)
        if right in gtin_to_row and left in gtin_to_canon_idx:
            labelled.setdefault(key(gtin_to_row[right], left), set()).add(0)

    targeted, _scores, _funnel = mine_targeted_attribute_negatives_with_funnel(
        df, gates, canonical_records, gtin_to_row, gtin_to_canon_idx,
        n_target=12000, min_similarity=0.5, volume_relative_tolerance=0.05,
        canonical_map=canon_map,
    )
    for anchor, target in targeted:
        labelled.setdefault(key(int(anchor), canon_gtins[int(target)]), set()).add(0)

    contradictions = [texts for texts, labels in labelled.items() if len(labels) > 1]
    assert contradictions == [], f"{len(contradictions)} text pairs carry both labels"
    assert len(targeted) > 0 and len(negatives) > 0

