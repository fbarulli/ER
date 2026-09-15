"""Tests for the ONE consolidated pipeline trace (src/core/tracing.py).

These lock the four properties the owner directive ("consolidate all csv's
generated ... 0 gap coverage") depends on:

1. the row contract, including the run axis (no anonymous rows);
2. consecutive runs never blur together and never duplicate rows;
3. the accounting identity — read from the trace FILE alone, every input row
   and every candidate pair is accounted for;
4. the destination comes from the ``training_trace`` layout in
   config/paths.yaml via core.common.artifact, never from __file__/__parents__
   or a hand-built "results/logs/..." literal.

The integration test drives BOTH real pipeline stages on a small real subset
of the live export, with every results path redirected into a tmp dir, so the
suite never touches the frozen results/ artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import core.common as common
import pipeline
from core.gtin import barcode_validity
from core.tracing import (
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    SCOPE_GROUP,
    TRACE_COLUMNS,
    TraceRun,
    accounting,
    assert_trace_frame,
    detail_json,
    read_trace,
    trace_path,
)

REPO = Path(__file__).resolve().parents[1]
OWNED_MODULES = (
    REPO / "src" / "core" / "tracing.py",
    REPO / "src" / "pipeline.py",
    REPO / "src" / "training" / "data_prep.py",
)


# ── helpers ────────────────────────────────────────────────────────────────
def _rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Index trace rows by step (last wins) for easy lookup."""
    return {str(row["step"]): row for _, row in frame.iterrows()}


def _flip_check_digit(barcode: str) -> str:
    """Return the same barcode with a check digit that must fail GS1."""
    body, last = barcode[:-1], int(barcode[-1])
    for candidate in range(10):
        if candidate == last:
            continue
        mutated = f"{body}{candidate}"
        if not bool(barcode_validity(pd.Series([mutated])).iloc[0]):
            return mutated
    raise AssertionError(f"no invalid mutation found for {barcode}")


@pytest.fixture(scope="module")
def live_slice() -> pd.DataFrame:
    """A small, REAL slice of the raw export: two brands, valid barcodes.

    Plus deliberately crafted populations so the row identity is non-trivial —
    a repeated gtin (collapses into a canonical), a missing gtin and a failed
    GS1 checksum (the two guard drops).
    """
    raw = pd.read_csv(common.DATA_PATH, dtype=str)
    valid = (
        raw["gtin"].notna()
        & (raw["gtin"].astype(str).str.strip() != "")
        & (raw["gtin"].astype(str).str.lower() != "nan")
    )
    raw = raw[valid]
    raw = raw[barcode_validity(raw["gtin"].fillna("").astype(str).str.strip())]
    per_brand = raw.groupby("brand")["gtin"].nunique().sort_values(ascending=False)
    brands = list(per_brand[per_brand >= 4].index[:2])
    assert len(brands) == 2, f"need two multi-gtin brands, got {brands}"
    subset = raw[raw["brand"].isin(brands)].groupby("brand", group_keys=False).head(12)

    collapsed = subset.head(1).copy()
    collapsed["sku_id"] = collapsed["sku_id"].astype(str) + "-dup"
    missing = subset.head(1).copy()
    missing["sku_id"] = missing["sku_id"].astype(str) + "-nogt"
    missing["gtin"] = ""
    checksum = subset.head(1).copy()
    checksum["sku_id"] = checksum["sku_id"].astype(str) + "-bad"
    checksum["gtin"] = _flip_check_digit(str(checksum["gtin"].iloc[0]))
    return pd.concat([subset, collapsed, missing, checksum], ignore_index=True)


@pytest.fixture()
def redirected_results(monkeypatch, tmp_path):
    """Point every results-rooted path (SSOT + the module-level RESULT names)
    at a tmp dir, so a real pipeline run is fully hermetic."""
    monkeypatch.setattr(pipeline, "RESULTS", tmp_path)
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setitem(common.F, "canonical_records", tmp_path / "canonical_records.csv")
    monkeypatch.setitem(common.F, "gate_results", tmp_path / "gate_results.csv")
    monkeypatch.delenv("EUROMONITOR_TRACE_RUN", raising=False)
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    return tmp_path


# ── 4. the path comes from the layout ──────────────────────────────────────
def test_trace_path_resolves_through_the_training_trace_layout(monkeypatch):
    """trace_path() must render the declared layout, not a literal."""
    calls: list[tuple[str, object]] = []
    real_artifact = common.artifact

    def spy(key, fields=None):
        calls.append((key, fields))
        return real_artifact(key, fields)

    monkeypatch.setattr(common, "artifact", spy)
    resolved = trace_path()

    assert calls == [("training_trace", None)]
    spec = common.LAYOUTS["training_trace"]
    assert spec.owner == "core.tracing"
    assert spec.root == "results"
    assert spec.template == "logs/training_trace.csv"
    assert resolved == common.artifact("training_trace")
    assert resolved == common.RESULTS / "logs" / "training_trace.csv"


def test_owned_modules_never_build_a_path_from_location_or_literal():
    """No __file__/__parents__, no "results/..." string literal in code.

    Docstrings are excluded (they legitimately NAME the files this directive
    consolidated); every other string constant is inspected.
    """
    for module in OWNED_MODULES:
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(body[0].value.value)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue  # docstrings NAME the consolidated files
                assert "__file__" not in node.value, (module, node.value)
                assert "results/" not in node.value, (module, node.value)
            elif isinstance(node, ast.Name):
                assert node.id not in {"__file__", "__parents__"}, (module, node.id)
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"__file__", "__parents__"}, (module, node.attr)


# ── 1. row contract ────────────────────────────────────────────────────────
def test_row_contract_accepts_what_the_writers_produce(tmp_path):
    target = tmp_path / "trace.csv"
    run = TraceRun("stage", run_id="run-contract")
    run.add("step", "sub", in_count=5, out_count=3, reason="why")
    run.add("step", "group", scope=SCOPE_GROUP, in_count=5, out_count=2, reason="why")
    run.write(target)

    frame = read_trace(target)
    assert list(frame.columns) == list(TRACE_COLUMNS)
    assert_trace_frame(frame)
    assert set(frame["run_id"]) == {"run-contract"}
    # the stage's first row states the run identity and how it was resolved
    assert list(frame["step"]) == ["run_identity", "step.sub", "step.group"]
    identity = detail_json(frame.iloc[0]["detail"])
    assert identity["run_id"] == "run-contract"
    assert "resolution" in identity and "policy" in identity
    # dropped_count is DERIVED, never hand-written
    assert int(frame[frame["step"] == "step.sub"].iloc[0]["dropped_count"]) == 2


def test_row_contract_rejects_anonymous_and_malformed_rows(tmp_path):
    target = tmp_path / "trace.csv"
    run = TraceRun("stage", run_id="run-x")
    run.add("step", "sub", in_count=1, out_count=1)
    run.write(target)
    good = read_trace(target)

    anonymous = good.copy()
    anonymous.loc[0, "run_id"] = ""
    with pytest.raises(ValueError, match="no.*run_id"):
        assert_trace_frame(anonymous)

    extra = good.copy()
    extra["undeclared"] = "x"
    with pytest.raises(ValueError, match="undeclared columns"):
        assert_trace_frame(extra)

    unknown_scope = good.copy()
    unknown_scope.loc[0, "scope"] = "banana"
    with pytest.raises(ValueError, match="unknown scope"):
        assert_trace_frame(unknown_scope)


# ── 2. run identity: no duplicates, no blurred runs ────────────────────────
def test_two_consecutive_identical_runs_do_not_duplicate_rows(tmp_path):
    """The anti-duplication guarantee: a deterministic pipeline re-run in the
    same run id commits its stage again, it does not append a second copy."""
    target = tmp_path / "trace.csv"

    def _one_run() -> pd.DataFrame:
        run = TraceRun("data_prep", run_id="run-fixed")
        run.add("gtin_guard", "identity_claims_evaluated", in_count=100, out_count=80)
        run.add("gate", "candidates_gated", in_count=50, out_count=50)
        run.write(target)
        return read_trace(target)

    first = _one_run()
    second = _one_run()

    assert len(first) == 3  # run_identity + two step rows
    assert len(second) == len(first)
    assert list(second["step"]) == list(first["step"])
    assert_trace_frame(second)


def test_a_stage_rerun_replaces_in_place_and_keeps_the_flow_order(tmp_path):
    target = tmp_path / "trace.csv"
    stage1 = TraceRun("data_prep", run_id="run-order")
    stage1.add("a", "first", in_count=1, out_count=1)
    stage1.write(target)
    stage2 = TraceRun("pairs", run_id="run-order")
    stage2.add("b", "second", in_count=1, out_count=1)
    stage2.write(target)

    # Re-run stage 1 (a second data-prep pass in the same run).
    again = TraceRun("data_prep", run_id="run-order")
    again.add("a", "first", in_count=2, out_count=2)
    again.add("a", "extra", in_count=2, out_count=1)
    again.write(target)

    frame = read_trace(target)
    steps = frame[frame["step"] != "run_identity"]
    assert list(steps["stage"]) == ["data_prep", "data_prep", "pairs"]
    assert list(steps["step"]) == ["a.first", "a.extra", "b.second"]
    assert int(steps.iloc[0]["in_count"]) == 2  # the re-run's numbers, not the stale ones


def test_two_different_runs_stay_distinguishable(tmp_path):
    target = tmp_path / "trace.csv"
    for run_id in ("run-one", "run-two"):
        run = TraceRun("data_prep", run_id=run_id)
        run.add("gtin_guard", "identity_claims_evaluated", in_count=7, out_count=5)
        run.write(target)

    frame = read_trace(target)
    assert set(frame["run_id"]) == {"run-one", "run-two"}
    assert len(frame) == 4  # a run_identity row + a step row per run
    for run_id in ("run-one", "run-two"):
        rows = frame[frame["run_id"] == run_id]
        assert list(rows["step"]) == ["run_identity", "gtin_guard.identity_claims_evaluated"]
    assert_trace_frame(frame)


def test_unlabelled_legacy_rows_are_dropped_not_mixed_in(tmp_path):
    """A file from before the run axis existed cannot be attributed, so it is
    dropped rather than sharing the file with labelled rows."""
    target = tmp_path / "trace.csv"
    legacy = pd.DataFrame(
        [{**{column: "" for column in TRACE_COLUMNS}, "stage": "old", "step": "old.step"}]
    )
    legacy.drop(columns=["run_id"]).to_csv(target, index=False)

    run = TraceRun("data_prep", run_id="run-new")
    run.add("gtin_guard", "identity_claims_evaluated", in_count=3, out_count=3)
    run.write(target)

    frame = read_trace(target)
    assert set(frame["run_id"]) == {"run-new"}
    assert set(frame["stage"]) == {"data_prep"}


def test_old_runs_age_out_but_whole_runs_at_a_time(tmp_path):
    target = tmp_path / "trace.csv"
    for index in range(8):
        run = TraceRun("data_prep", run_id=f"run-{index}")
        run.add("gtin_guard", "identity_claims_evaluated", in_count=1, out_count=1)
        run.write(target)

    frame = read_trace(target)
    kept = list(dict.fromkeys(frame["run_id"]))
    assert kept == [f"run-{index}" for index in range(3, 8)], kept
    assert len(frame) == 10  # a run_identity row + a step row per retained run


# ── 3. entity sampling: exact census + stratified sample ───────────────────
def test_entity_rows_are_censused_exactly_and_sampled_stratified(tmp_path):
    target = tmp_path / "trace.csv"
    records = (
        [{"key": f"a{i}", "bucket": "big"} for i in range(500)]
        + [{"key": f"b{i}", "bucket": "medium"} for i in range(50)]
        + [{"key": "c0", "bucket": "tiny"}]
    )
    run = TraceRun("pairs", run_id="run-sample")
    summary = run.add_entities(
        "pair_payload",
        records,
        key_of=lambda r: r["key"],
        reason_of=lambda r: r["bucket"],
        detail_of=lambda r: {"k": r["key"]},
        source="model payload",
        per_reason=64,
        total_cap=100,
    )
    run.write(target)

    frame = read_trace(target)
    census = frame[frame["step"] == "pair_payload.reason_census"]
    entities = frame[frame["scope"] == "entity"]

    # every bucket is censused with its EXACT population, including the tiny one
    assert dict(zip(census["reason"], census["in_count"].astype(int))) == {
        "big": 500,
        "medium": 50,
        "tiny": 1,
    }
    # the census sums back to the population — nothing is silently lost
    assert int(census["in_count"].astype(int).sum()) == len(records)
    assert summary["population"] == len(records)

    # the budget is respected and EVERY bucket still gets a representative row
    assert len(entities) <= 100
    assert set(entities["reason"]) == {"big", "medium", "tiny"}
    assert (entities["reason"] == "tiny").sum() == 1

    budget = frame[frame["step"] == "pair_payload.sample_budget"].iloc[0]
    assert int(budget["in_count"]) == len(records)
    assert int(budget["out_count"]) == len(entities)
    assert detail_json(budget["detail"])["omitted"] == len(records) - len(entities)
    # the sample never exceeds the declared policy caps
    assert ENTITY_SAMPLE_PER_REASON < ENTITY_ROW_CAP


# ── 3b. the accounting identity, on a real two-stage run ───────────────────
def test_two_stage_live_run_covers_every_row_and_every_pair(
    live_slice, redirected_results
):
    """Both stages on real data; the identity is recomputed FROM THE FILE."""
    raw = live_slice
    canon_df = raw.rename(columns=common.COLUMN_MAPPING)
    canon_df["product_id"] = raw["sku_id"]

    pairs, canon = pipeline.run_within_brand_pipeline(raw)
    bundle = pipeline.build_training_data(canon_df)

    frame = read_trace(trace_path())
    assert_trace_frame(frame)

    # ── both stages are present after a normal two-stage invocation ─────────
    assert set(frame["stage"]) == {"data_prep", "pairs"}
    assert len(frame[frame["stage"] == "pairs"]) > 0
    # … and each stage states its own run identity, with the fingerprint inputs
    identities = frame[frame["step"] == "run_identity"]
    assert set(identities["stage"]) == {"data_prep", "pairs"}
    for _, row in identities.iterrows():
        identity = detail_json(row["detail"])
        assert identity["run_id"] == row["run_id"]
        assert set(identity["sources"]) == {"canonical_records", "gate_results"}

    # ── no duplicate rows: no two rows are identical except for their stamp
    without_stamp = frame.drop(columns=["at"])
    assert len(without_stamp.drop_duplicates()) == len(frame)

    # ── the run axis: one run, stage 1 rows before stage 2 rows ────────────
    assert frame["run_id"].nunique() == 1
    assert list(dict.fromkeys(frame["stage"])) == ["data_prep", "pairs"]

    # ── the row identity, from the trace alone ─────────────────────────────
    measured = accounting(frame)
    valid_rows = len(raw) - 1 - 1  # one missing-gtin row, one bad-checksum row
    assert measured["rows_in"] == len(raw)
    assert measured["gtin_missing_or_nan"] == 1
    assert measured["gs1_checksum_failed"] == 1
    assert measured["rows_identity_valid"] == valid_rows
    assert (
        measured["rows_in"]
        == measured["canonical_records"]
        + measured["collapsed_same_gtin"]
        + measured["gtin_missing_or_nan"]
        + measured["gs1_checksum_failed"]
    )
    # the crafted repeated gtin really did collapse (so the term is exercised)
    assert measured["collapsed_same_gtin"] == 1
    assert measured["canonical_records"] == valid_rows - 1

    # ── the pair identity: every candidate pair carries one decision … ─────
    assert measured["gate_pairs"] == len(pairs)
    assert sum(measured["gate_decisions"].values()) == len(pairs)
    # … and exactly one label destiny
    assert measured["label_pairs"] == len(pairs)
    assert sum(measured["label_destiny"].values()) == len(pairs)
    rows = _rows(frame)
    assert int(rows["gate.candidates_gated"]["in_count"]) == len(pairs)
    assert int(rows["labels.every_pair_accounted"]["out_count"]) == len(pairs)

    # ── the negatives funnel closes: two directions per pair ───────────────
    band = detail_json(rows["negatives.gate_hard_no_band"]["detail"])
    assert band["hard_no_and_in_band"] == (
        int(rows["negatives.gate_hard_no_band"]["out_count"])
        + band["dropped_same_canonical"]
    )
    assert int(rows["negatives.index_resolution"]["in_count"]) == band["both_directions"]
    assert int(rows["negatives.index_resolution"]["out_count"]) == len(bundle["neg"])

    # ── the final label census equals the populations handed to training ───
    pair_census = _rows(frame)["payload.pair_census"]
    assert int(pair_census["out_count"]) == len(bundle["pos"]) + len(bundle["neg"]) + len(
        bundle["targeted_attribute_neg"]
    )
    kinds = detail_json(pair_census["detail"])
    assert kinds["pos"] == len(bundle["pos"])
    assert kinds["neg_hard"] == len(bundle["neg"])

    # ── the column-contract rows make the two-stage handoff visible ────────
    contracts = frame[frame["step"] == "column_contract.input_frame"]
    assert len(contracts) == 2
    by_stage = {str(row["stage"]): detail_json(row["detail"]) for _, row in contracts.iterrows()}
    assert by_stage["data_prep"]["missing_required"] == []
    assert "gtin" in by_stage["data_prep"]["columns"]
    assert "sku_name_eng" in by_stage["data_prep"]["columns"]
    assert by_stage["pairs"]["missing_required"] == []
    assert "barcode" in by_stage["pairs"]["columns"]
    assert "gtin" not in by_stage["pairs"]["columns"]  # a DIFFERENT contract


def test_identity_rejects_a_trace_with_an_unaccounted_row(
    live_slice, redirected_results
):
    """The identity is not vacuous: removing a destiny bucket breaks it."""
    raw = live_slice
    canon_df = raw.rename(columns=common.COLUMN_MAPPING)
    canon_df["product_id"] = raw["sku_id"]
    pairs, _ = pipeline.run_within_brand_pipeline(raw)
    pipeline.build_training_data(canon_df)

    frame = read_trace(trace_path())
    assert sum(accounting(frame)["label_destiny"].values()) == len(pairs)

    sabotaged = frame[frame["step"] != "labels.destiny_hard_no_below_similarity_floor"]
    assert sum(accounting(sabotaged)["label_destiny"].values()) != len(pairs)
