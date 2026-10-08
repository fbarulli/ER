"""Contracts for ``core.eval_trace``, validated against the REAL artifacts.

The point of this suite is that the second review's empirically demonstrated
false rejections are gone:

* D1 — ``MetricBlock(extra='forbid')`` used to reject the laya lane's own
  ``evaluate_records`` output because it carries ``by_type`` inside the dict;
* D2 — the report validator used to demand all three qtypes where the real
  corpus emits only ``{'noul'}``;
* D3 — the ablation flip predicate was a chained ``!=`` against a hardcoded
  ``0.0``: inert on the real rows, and it rejected a legitimate no-flip row
  whose two scores straddle the frozen threshold;
* D6 — ``Count(strict=True)``/``Literal[True]`` rejected every row read back
  from a persisted CSV.

Every test that can, reads the artifact on disk:
``results/attribute_ablation/report.json``, a real ``<track>__slice_metrics.csv``,
``results/laya_lane/kaggle/finetune/output/checkpoint/train_report.json``, the
corpus receipt and the question schema. No artifact is written here except into
tmp dirs.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

import core.common as common
from cli import laya_lane
from core.eval_trace import (
    AbstentionBlock,
    AttributeAttributionRow,
    Brier,
    CORE_RECORD_DIMENSIONS,
    EvalProvenance,
    EvalRowKey,
    MetricBlock,
    PersistedCount,
    QTypeName,
    SliceMetricsRow,
    TaggedRecord,
    TraceabilityCoverage,
    TraceabilityReport,
    UnmeasuredSliceRow,
    canonical_dimension,
    decision_flip,
    derived_dimension_policies,
    derived_record_census,
    row_from_csv,
    write_report,
)

REPO = Path(__file__).resolve().parents[1]
ABLATION_REPORT = REPO / "results/attribute_ablation/report.json"
CORPUS_RECEIPT = REPO / "data/laya/receipt.json"
QUESTION_SCHEMA = REPO / "config/laya.question.json"
LAYA_TRAIN_REPORT = (
    REPO / "results/laya_lane/kaggle/finetune/output/checkpoint/train_report.json"
)
SLICE_CSVS = (
    REPO / "results/model_tracks/1007T190357883306Z/gnn_only/gnn_only__local_completion"
           "/gnn_only__reports/gnn_only__slice_metrics.csv",
    REPO / "results/model_tracks/1007T193309196779Z/text/text__reports"
           "/text__slice_metrics.csv",
)


def _load(path: Path):
    if not path.is_file():
        pytest.skip(f"real artifact not present: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ablation_report():
    return _load(ABLATION_REPORT)


@pytest.fixture(scope="module")
def laya_train_report():
    return _load(LAYA_TRAIN_REPORT)


@pytest.fixture(scope="module")
def questions():
    return _load(QUESTION_SCHEMA)["questions"]


def _block(items: int = 10, **over) -> MetricBlock:
    fields = dict(items=items, loss=0.1, accuracy=0.5, mean_confidence=0.5,
                  ece=0.1, brier=0.2, brier_top1=0.1)
    fields.update(over)
    return MetricBlock(**fields)


def _coverage(*, total: int, items_total: int | None = None, source="finetune_corpus",
              **over) -> TraceabilityCoverage:
    fields = dict(
        records_total=total, items_total=items_total if items_total is not None else total,
        by_source={source: total}, dimension_values={}, dimension_multiplicity={},
        by_dimension={}, unknown_policy={}, slice_coverage="not_applicable",
        slice_coverage_reason="aggregate-only report",
        aggregate_reason="the fixture reports aggregates and carries no records")
    fields.update(over)
    return TraceabilityCoverage(**fields)


def _derived_coverage(records, items_total: int | None = None,
                      **over) -> TraceabilityCoverage:
    """The coverage a record-derived producer writes: EVERY census counted.

    Built from the same shared derivation the contract audits against, so this
    helper cannot encode a second opinion about what the records say.
    """
    derived = derived_record_census(records)
    policies = derived_dimension_policies(records)
    dimensions = {name: derived[name] for name in CORE_RECORD_DIMENSIONS}
    fields = dict(
        records_total=len(records),
        items_total=len(records) if items_total is None else items_total,
        by_source=dimensions["source"],
        dimension_values={name: set(counts) for name, counts in dimensions.items()},
        dimension_multiplicity={name: policies[name] for name in dimensions},
        by_dimension=dimensions,
        unknown_policy={name: "a record with no value is reported as 'unknown'"
                        for name in dimensions},
        slice_coverage=policies["slice"])
    fields.update(over)
    return TraceabilityCoverage(**fields)


def _provenance(source="finetune_corpus", **over) -> EvalProvenance:
    digests = {"corpus_sha256": "a" * 64, "decision_csv_sha256": "b" * 64}
    return EvalProvenance(source=source, model_id="fixture", digests=digests, **over)


def _straddle_row() -> dict:
    """A legitimate ablation row whose two scores straddle the hardcoded ``0.0``.

    ``baseline_score=-0.20`` and ``ablated_score=0.30`` with the emitter's
    decision_flip=False. Under the FROZEN threshold (0.7 here) both scores sit on
    the same side, so the row is a genuine no-flip and the corrected predicate
    agrees; the draft's chained ``!=`` against a hardcoded ``0.0`` called it a
    flip and rejected the row.
    """
    return {
        "sku_id1": "a", "sku_id2": "b", "label": "0", "split": "train",
        "attribute": None, "channel": None,
        "baseline_score": -0.20, "ablated_score": 0.30, "score_delta": 0.50,
        "decision_flip": False, "baseline_error": False, "ablated_error": True,
        "changed_listings": 1, "endpoint_input_changed": [False, False],
        "embedding_cosine_delta": [0.0, 0.0], "baseline_ranks": [1],
        "ablated_ranks": [1], "ann_baseline_hits": None, "ann_ablated_hits": None,
        "known_positive_recall_change": {},
        "current_attribute_evidence": {"x": None},
        "evidence_scope": "frozen payload; raw-row evidence unavailable",
    }


# ── D1: the shared metric contract accepts the lane's real producer output ──
def test_metric_block_accepts_the_real_laya_train_report(laya_train_report):
    for stage in ("before", "after"):
        block = MetricBlock.model_validate(laya_train_report[stage])
        assert block.items == 1259
        assert list(block.by_type) == ["noul"]  # the draft rejected this key
        assert sum(child.items for child in block.by_type.values()) == block.items
    assert MetricBlock.model_validate(laya_train_report["after"]).accuracy == 0.977


# ── C3: the Brier bound is 2, not 2*(1-1/k) ────────────────────────────────
def test_brier_bound_is_the_real_supremum():
    adapter = TypeAdapter(Brier)
    assert adapter.validate_python(2.0) == 2.0
    with pytest.raises(ValidationError):
        adapter.validate_python(2.0001)
    for k in (2, 3, 5, 10):
        # two disjoint point masses over k classes: sum((p - t)**2) == 2 exactly
        supremum = (1.0 - 0.0) ** 2 + (0.0 - 1.0) ** 2
        assert supremum == 2.0
        # the draft's formula is NOT a bound on this quantity
        assert supremum > 2 * (1 - 1 / k)


# ── C2: the calibration subset + accounting identities ─────────────────────
def test_metric_block_calibration_identities():
    fitted = _block(items=10, temperature=[1.0, 1.0, 2.0],
                    temperature_by_options={"noul:2": 1.5},
                    n_by_bucket={"noul:2": 6, "choice:3-5": 4})
    assert sum(fitted.n_by_bucket.values()) == fitted.items

    with pytest.raises(ValidationError, match="one scalar per qtype"):
        _block(items=10, temperature=[1.0])
    with pytest.raises(ValidationError, match="per-type sequence"):
        _block(items=10, temperature_by_options={"noul:2": 1.5},
               n_by_bucket={"noul:2": 10})
    with pytest.raises(ValidationError, match="fitted bucket"):
        _block(items=10, temperature=[1.0, 1.0, 1.0],
               temperature_by_options={"noul:2": 1.5},
               n_by_bucket={"choice:2": 10})
    with pytest.raises(ValidationError, match="account for every item"):
        _block(items=10, temperature=[1.0, 1.0, 1.0], n_by_bucket={"choice:2": 9})


# ── D2: the report accepts one question type and polices the item sum ──────
def _report(*, overall=None, by_type=None, total=None, source="finetune_corpus",
            records=(), keys=(), coverage=None, **prov) -> TraceabilityReport:
    overall = overall or _block()
    return TraceabilityReport(
        provenance=_provenance(source, **prov),
        overall=overall,
        by_type=by_type or {},
        records=records,
        keys=keys,
        coverage=coverage or _coverage(total=total or 10, source=source),
    )


def test_report_accepts_a_single_real_question_type(laya_train_report):
    block = MetricBlock.model_validate(laya_train_report["after"])
    report = _report(overall=block, by_type={"noul": block}, total=1259)
    assert list(report.by_type) == ["noul"]


def test_report_polices_the_per_type_item_sum():
    overall = _block(items=10)
    with pytest.raises(ValidationError, match="do not account"):
        _report(overall=overall, by_type={"noul": _block(items=9)})


def test_report_rejects_an_unknown_question_type():
    with pytest.raises(ValidationError):
        _report(overall=_block(items=10), by_type={"noul": _block(items=6),
                                                   "bogus": _block(items=4)})


def test_report_source_must_be_accounted_in_coverage():
    with pytest.raises(ValidationError, match="not accounted for in coverage"):
        _report(source="finetune_corpus",
                coverage=_coverage(total=10, source="tracks_suite"))


# ── D3: the ablation rows validate and the flip lives with the threshold ───
def test_every_real_ablation_row_validates_against_the_frozen_threshold(ablation_report):
    threshold = ablation_report["threshold"]
    flips = 0
    for row in ablation_report["rows"]:
        model = AttributeAttributionRow.model_validate(row)
        assert model.decision_flip == decision_flip(
            model.baseline_score, model.ablated_score, threshold)
        flips += model.decision_flip
    assert flips == 5  # the real report's flips, none silently skipped


def test_a_no_flip_row_with_straddling_scores_is_accepted():
    model = AttributeAttributionRow.model_validate(_straddle_row())
    assert model.decision_flip is False
    # the FROZEN threshold agrees with the row: both scores sit on the same side
    assert decision_flip(model.baseline_score, model.ablated_score, 0.7) is False
    # the draft's predicate (a chained `!=` against a hardcoded 0.0) did not
    draft_flip = (model.ablated_score >= 0.0) != (model.baseline_score >= 0.0)
    assert draft_flip is True


def test_the_row_model_does_not_police_the_frozen_threshold():
    """The model has no threshold, so it must not judge the flip: the adapter
    that owns the frozen threshold is the only authority (C5)."""
    row = _straddle_row()
    row["decision_flip"] = True  # the adapter would reject this; the model must not
    assert AttributeAttributionRow.model_validate(row).decision_flip is True


def test_absent_attribute_evidence_requires_an_explicit_scope():
    row = _straddle_row()
    row["current_attribute_evidence"] = {"attribute": None}
    row["evidence_scope"] = None
    with pytest.raises(ValidationError, match="absent attribute evidence"):
        AttributeAttributionRow.model_validate(row)
    row["evidence_scope"] = "frozen payload; raw-row evidence unavailable"
    AttributeAttributionRow.model_validate(row)


def test_the_real_null_scope_rows_carry_measured_evidence(ablation_report):
    """C6: the real absent-evidence signal is a None VALUE, not a None dict."""
    rows = ablation_report["rows"]
    null_scope = [row for row in rows if not (row.get("evidence_scope") or "").strip()]
    assert null_scope  # 2,220 of them in the real report
    for row in null_scope[:50]:
        evidence = row["current_attribute_evidence"] or {}
        assert any(value is not None for value in evidence.values())
        AttributeAttributionRow.model_validate(row)


# ── D6: rows read back from a persisted CSV validate ───────────────────────
@pytest.mark.parametrize("path", SLICE_CSVS, ids=lambda path: Path(path).name)
def test_real_slice_metrics_csv_rows_validate(path):
    if not Path(path).is_file():
        pytest.skip(f"real artifact not present: {path}")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kinds: Counter = Counter()
    for row in rows:
        coerced = row_from_csv(row)
        if coerced["evaluated"]:
            kinds[type(SliceMetricsRow.model_validate(coerced)).__name__] += 1
        else:
            kinds[type(UnmeasuredSliceRow.model_validate(coerced)).__name__] += 1
    assert set(kinds) == {"SliceMetricsRow", "UnmeasuredSliceRow"}


def test_raw_csv_cells_are_rejected_without_the_coercion():
    """The D6 false rejection: on disk every cell is a string."""
    path = SLICE_CSVS[0]
    if not path.is_file():
        pytest.skip(f"real artifact not present: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        first = next(row for row in csv.DictReader(handle) if row["evaluated"] == "True")
    with pytest.raises(ValidationError):
        SliceMetricsRow.model_validate(first)
    SliceMetricsRow.model_validate(row_from_csv(first))


def test_persisted_count_reads_back_python_and_numpy_scalars():
    adapter = TypeAdapter(PersistedCount)
    assert adapter.validate_python("7") == 7
    assert adapter.validate_python(7.0) == 7
    import numpy as np

    assert adapter.validate_python(np.int64(7)) == 7
    assert adapter.validate_python(np.int32(7)) == 7


# ── C9: coverage is derived from CARRIED records ───────────────────────────
def _carried():
    records = (
        TaggedRecord(record_id="r1", source="identity_decision_csv", split="dev",
                     population="pairs", difficulty="unknown",
                     difficulty_reason="no difficulty axis"),
        TaggedRecord(record_id="r2", source="identity_decision_csv", split="dev",
                     population="pairs", difficulty="unknown",
                     difficulty_reason="no difficulty axis"),
    )
    keys = (
        EvalRowKey(record_id="r1", question_id="identity_claim", question_type="noul"),
        EvalRowKey(record_id="r2", question_id="identity_claim", question_type="noul"),
    )
    return records, keys


def test_carried_records_are_authoritative_over_the_declared_strata():
    records, keys = _carried()
    report = _report(overall=_block(items=2), source="identity_decision_csv",
                     records=records, keys=keys,
                     coverage=_derived_coverage(records, 2))
    assert report.coverage.by_source == {"identity_decision_csv": 2}
    assert report.coverage.records_total == 2
    # the derivation covers EVERY carried axis, population and difficulty included
    assert set(report.coverage.by_dimension) == set(CORE_RECORD_DIMENSIONS)
    assert report.coverage.by_dimension["population"] == {"pairs": 2}
    assert report.coverage.by_dimension["difficulty"] == {"unknown": 2}
    assert report.coverage.by_dimension["slice"] == {"unknown": 2}


def test_declared_strata_disagreeing_with_the_records_are_rejected():
    records, keys = _carried()
    # the inflation is INTERNALLY consistent (every stratum sums to 3), so only
    # the carried records can expose it
    inflated = _derived_coverage(
        records, 3, records_total=3,
        by_source={"identity_decision_csv": 3},
        by_dimension={"source": {"identity_decision_csv": 3},
                      "split": {"dev": 3},
                      "population": {"pairs": 3},
                      "slice": {"unknown": 3},
                      "difficulty": {"unknown": 3}})
    with pytest.raises(ValidationError, match="disagree"):
        _report(overall=_block(items=3), source="identity_decision_csv",
                records=records, keys=keys, coverage=inflated)
    # a declared census no carried record tags is the same class of lie: the
    # declared key-set rises to the carried one (a zero count is still a claim)
    with pytest.raises(ValidationError, match="declared counts coverage mismatch"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records, keys=keys,
                coverage=_derived_coverage(
                    records, 2,
                    by_dimension={**_derived_coverage(records, 2).by_dimension,
                                  "population": {"pairs": 2, "minted": 0}},
                    dimension_values={**_derived_coverage(records, 2).dimension_values,
                                      "population": {"pairs", "minted"}}))


def test_carried_keys_must_cover_exactly_the_carried_records():
    records, _ = _carried()
    with pytest.raises(ValidationError, match="carried keys"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records,
                keys=(EvalRowKey(record_id="r1", question_id="identity_claim",
                                 question_type="noul"),),
                coverage=_derived_coverage(records, 2))


def test_the_item_total_is_the_carried_key_count_not_a_declared_number():
    """Falsified 2026-10-08: with records carried, the coverage could still claim
    ``items_total=7`` while the records' keys numbered two."""
    records, keys = _carried()
    with pytest.raises(ValidationError, match="item total disagrees"):
        _report(overall=_block(items=7), source="identity_decision_csv",
                records=records, keys=keys,
                coverage=_derived_coverage(records, 7))


def test_a_census_for_a_dimension_records_cannot_carry_is_refused():
    """The records are the authority for the vocabulary: a dimension a
    ``TaggedRecord`` cannot carry cannot be censused at all."""
    records, keys = _carried()
    coverage = _derived_coverage(
        records, 2,
        dimension_values={**_derived_coverage(records, 2).dimension_values,
                          "question_type": {"noul"}},
        dimension_multiplicity={**_derived_coverage(records, 2).dimension_multiplicity,
                                "question_type": "partition"},
        by_dimension={**_derived_coverage(records, 2).by_dimension,
                      "question_type": {"noul": 2}},
        unknown_policy={**_derived_coverage(records, 2).unknown_policy,
                        "question_type": "one type per question"})
    with pytest.raises(ValidationError, match="not carried by a TaggedRecord"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records, keys=keys, coverage=coverage)


def test_missing_core_dimension_censuses_are_refused():
    records, keys = _carried()
    full = _derived_coverage(records, 2)
    with pytest.raises(ValidationError, match="demand a census for"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records, keys=keys,
                coverage=full.model_copy(update={
                    "dimension_values": {"source": {"identity_decision_csv"}},
                    "by_dimension": {"source": {"identity_decision_csv": 2}},
                    "dimension_multiplicity": {"source": "partition"},
                    "unknown_policy": {"source": "the emitter tags the source"}}))


def test_a_multi_valued_axis_must_declare_overlap():
    """The multiplicity is a property of the carried data, not a claim: two
    records carrying two slices each cannot be said to partition."""
    records = (
        TaggedRecord(record_id="r1", source="laya_cli_eval", split="test",
                     population="p", slices=("a", "b"), difficulty="unknown",
                     difficulty_reason="no difficulty axis"),
        TaggedRecord(record_id="r2", source="laya_cli_eval", split="test",
                     population="p", slices=("a", "b"), difficulty="unknown",
                     difficulty_reason="no difficulty axis"),
    )
    keys = tuple(EvalRowKey(record_id=record.record_id, question_id="q",
                            question_type="noul") for record in records)
    derived = _derived_coverage(records, 2)
    assert derived.dimension_multiplicity["slice"] == "overlap"
    _report(overall=_block(items=2), source="laya_cli_eval", records=records,
            keys=keys, coverage=derived)
    # a partition claim is refused twice over: structurally, and by the records
    # themselves (the EVIDENCE layer, called directly so both are visible)
    partitioned = derived.model_copy(update={
        "dimension_multiplicity": {
            **derived.dimension_multiplicity, "slice": "partition"}})
    with pytest.raises(ValidationError, match="does not partition every record"):
        _report(overall=_block(items=2), source="laya_cli_eval", records=records,
                keys=keys, coverage=partitioned)
    rebuilt = TraceabilityCoverage.model_construct(**{
        **derived.__dict__,
        "dimension_multiplicity": {
            **derived.dimension_multiplicity, "slice": "partition"}})
    with pytest.raises(ValueError, match="multiplicity is 'overlap'"):
        rebuilt.require_carried_records(records, keys)


def test_the_unknown_policy_must_name_the_value_its_census_counts():
    """Falsified 2026-10-08: ``unknown_policy`` was any non-blank string, so a
    census counting the explicit unknown value could hand-wave it."""
    records, keys = _carried()
    with pytest.raises(ValidationError, match="must name the explicit"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records, keys=keys,
                coverage=_derived_coverage(
                    records, 2,
                    unknown_policy={name: "measured per record"
                                    for name in CORE_RECORD_DIMENSIONS}))


def test_an_unknown_difficulty_without_a_reason_is_refused():
    """The per-record reason the contract demands, checked on the CARRIED
    records (a record smuggled in without one cannot ride the coverage)."""
    records, keys = _carried()
    stripped = records[0].model_copy(update={"difficulty_reason": None})
    carried = (stripped, *records[1:])
    # the report re-validates the records themselves ...
    with pytest.raises(ValidationError, match="unknown difficulty requires an explicit reason"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=carried, keys=keys,
                coverage=_derived_coverage(carried, 2))
    # ... and the EVIDENCE layer checks the same thing on the raw records
    contract = _derived_coverage(carried, 2)
    with pytest.raises(ValueError, match="unknown difficulty requires a reason"):
        contract.require_carried_records(carried, keys)


def test_one_source_per_report_is_enforced_against_the_records():
    records, keys = _carried()
    coverage = _derived_coverage(
        records, 2, by_source={"tracks_suite": 2},
        dimension_values={**_derived_coverage(records, 2).dimension_values,
                          "source": {"tracks_suite"}},
        by_dimension={**_derived_coverage(records, 2).by_dimension,
                      "source": {"tracks_suite": 2}})
    with pytest.raises(ValidationError, match="disagree"):
        _report(overall=_block(items=2), source="tracks_suite",
                records=records, keys=keys, coverage=coverage)


def test_dimension_names_are_bound_to_one_shared_vocabulary():
    """Falsified 2026-10-08: names were free strings, so the ablation cohort's
    ``difficulty_slice`` and this module's ``difficulty`` were two axes."""
    assert canonical_dimension("difficulty_slice") == "difficulty"
    assert canonical_dimension("slices") == "slice"
    assert canonical_dimension("difficulty") == "difficulty"
    with pytest.raises(ValueError, match="unknown traceability dimension"):
        canonical_dimension("banana")
    # the alias resolves on the way IN, and the census is stored under one name
    coverage = _coverage(total=2, dimension_values={"difficulty_slice": {"unknown"}},
                         dimension_multiplicity={"difficulty_slice": "partition"},
                         by_dimension={"difficulty_slice": {"unknown": 2}},
                         unknown_policy={"difficulty_slice":
                                         "no difficulty axis: reported as 'unknown'"})
    assert set(coverage.dimension_values) == {"difficulty"}
    assert coverage.by_dimension["difficulty"] == {"unknown": 2}
    with pytest.raises(ValidationError, match="unknown traceability dimension"):
        _coverage(total=2, dimension_values={"banana": {"x"}},
                  dimension_multiplicity={"banana": "partition"},
                  by_dimension={"banana": {"x": 2}},
                  unknown_policy={"banana": "x"})


def test_the_tag_derivation_covers_the_record_vocabulary():
    """One mapping from a record to its tags: every name in the vocabulary is
    derived from the SAME record object the report carries."""
    from core.eval_trace import RecordDimension, record_dimension_tags
    records, _ = _carried()
    assert set(record_dimension_tags(records[0])) == set(RecordDimension.__args__)
    # every vocabulary name is a real TaggedRecord field (``slice`` is carried by
    # the plural ``slices``), so a name can never be vocabulary-only
    for name in RecordDimension.__args__:
        assert ('slices' if name == 'slice' else name) in TaggedRecord.model_fields, name


def test_an_aggregate_only_report_must_say_so():
    """Falsified 2026-10-08: a report could claim rows=999 vs items=7 with zero
    carried records and pass. Aggregate-only is now an explicit declaration."""
    records, keys = _carried()
    with pytest.raises(ValidationError, match="aggregate-only"):
        _report(overall=_block(items=7), source="finetune_corpus",
                coverage=_coverage(total=999, items_total=7,
                                   aggregate_reason=None))
    # ... and the declaration is refused when records DO back the coverage
    with pytest.raises(ValidationError, match="cannot declare aggregate-only"):
        _report(overall=_block(items=2), source="identity_decision_csv",
                records=records, keys=keys,
                coverage=_derived_coverage(records, 2,
                                           aggregate_reason="aggregate-only, honest"))
    # a report with neither metrics nor records is refused outright
    with pytest.raises(ValidationError, match="metrics, records, or both"):
        TraceabilityReport(provenance=_provenance("finetune_corpus"),
                           coverage=_coverage(total=10))


def test_an_identity_only_report_is_allowed_but_never_empty():
    """The per-row decision grain measures no metrics of its own: its report
    carries records and keys, and ``overall`` is null on purpose."""
    records, keys = _carried()
    document = TraceabilityReport(
        provenance=_provenance("identity_decision_csv"),
        records=records, keys=keys,
        coverage=_derived_coverage(records, 2))
    assert document.overall is None
    assert document.by_type == {}
    assert document.coverage.items_total == len(keys) == 2
    # the module docstring's decision rule: an identity-only report still states
    # no more than the records carry
    assert document.model_dump(mode="json")["overall"] is None


# ── C7: the abstention block carries three states and bucket thresholds ────
def test_abstention_census_and_bucket_thresholds():
    block = AbstentionBlock(items=10, passed=7, abstained=2, unevaluated=1,
                            thresholds_by_bucket={"noul:2": 0.6, "default": 0.5})
    assert block.thresholds_by_bucket["default"] == 0.5
    with pytest.raises(ValidationError, match="account for every item"):
        AbstentionBlock(items=10, passed=7, abstained=2, unevaluated=0)
    with pytest.raises(ValidationError, match="zero passed"):
        AbstentionBlock(items=0, passed=0, abstained=0, unevaluated=0,
                        accuracy_on_passed=0.5)
    with pytest.raises(ValidationError):
        AbstentionBlock(items=1, passed=1, abstained=0, unevaluated=0,
                        thresholds_by_bucket={"bogus": 0.5})


def test_real_fitted_abstention_thresholds_validate(laya_train_report):
    assert laya_lane.abstention_thresholds(
        laya_train_report["abstention_thresholds"]) == {"noul:2": 0.5996}


# ── C12: provenance digests, held-out and overlap facts ────────────────────
def test_provenance_requires_the_source_digest():
    with pytest.raises(ValidationError, match="corpus_sha256"):
        EvalProvenance(source="finetune_corpus", model_id="m")
    with pytest.raises(ValidationError, match="decision_csv_sha256"):
        EvalProvenance(source="identity_decision_csv", model_id="m")
    # an emitter that rests on no named digest of its own is fine
    assert EvalProvenance(source="tracks_suite", model_id="m").digests == {}


def test_real_corpus_receipt_digest_passes_the_digest_contract():
    receipt = _load(CORPUS_RECEIPT)
    provenance = EvalProvenance(source="finetune_corpus", model_id="laya",
                                digests={"corpus_sha256": receipt["corpus_sha256"]})
    assert provenance.digests["corpus_sha256"] == receipt["corpus_sha256"]


def test_real_question_schema_types_are_within_the_contract(questions):
    for qid, question in questions.items():
        assert question["type"] in QTypeName.__args__, qid


# ── C8/C13: the layout is declared and the writer resolves it ──────────────
def test_write_report_uses_the_declared_layout(tmp_path, monkeypatch):
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    stamped: list = []
    monkeypatch.setattr(common, "trace_artifact",
                        lambda key, path, producer="": stamped.append(
                            (key, Path(path), producer)))
    document = _report()
    path = write_report("traceability_report",
                        {"lane": "laya_lane", "track": "finetune"}, document)
    assert path == tmp_path / "logs" / "laya_lane" / "finetune__traceability.json"
    assert path.is_file()
    assert stamped == [("traceability_report", path, "core.eval_trace")]
    spec = common.LAYOUTS["traceability_report"]
    assert spec.owner == "core.eval_trace"
    assert spec.template == "logs/{lane}/{track}__traceability.json"


def test_layout_fields_are_declared_exactly(tmp_path, monkeypatch):
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    with pytest.raises(ValueError, match="requires fields"):
        common.artifact("traceability_report", {"lane": "laya_lane"})


# ── the three lane-side laya adapters ──────────────────────────────────────
def test_corpus_adapter_accepts_the_real_train_report(laya_train_report):
    document = laya_lane.corpus_traceability(
        laya_train_report, model_id="checkpoint",
        digests={"corpus_sha256": "2" * 64})
    assert document.provenance.source == "finetune_corpus"
    assert document.overall.items == 1259
    assert list(document.by_type) == ["noul"]
    assert document.provenance.is_held_out is False
    assert document.provenance.eval_overlap_items == 10  # from the report's note
    assert list(document.coverage.by_dimension["split"]) == ["dev"]
    assert document.coverage.slice_coverage == "not_applicable"


def test_corpus_adapter_carries_the_items_from_rows_skip_census():
    report = {
        "rows": 3, "items": 2, "eval_split": "dev",
        "after": _block(items=2).model_dump(),
        "skipped": {"empty_text": 2, "invalid_target": 1},
    }
    document = laya_lane.corpus_traceability(
        report, model_id="m", digests={"corpus_sha256": "3" * 64})
    assert document.skipped == {"empty_text": 2, "invalid_target": 1}
    assert document.coverage.records_total == 3
    assert document.coverage.items_total == 2


def test_corpus_adapter_rejects_a_report_without_a_metric_block():
    with pytest.raises(ValueError, match="before/after"):
        laya_lane.corpus_traceability({}, model_id="m",
                                      digests={"corpus_sha256": "4" * 64})


def test_decision_csv_records_join_the_bindings_record_columns(questions):
    rows = [
        {"gtin1": "1", "gtin2": "2", "true_label": "1", "attribute_pairs": "x"},
        {"_row": {"gtin1": "3", "gtin2": "4", "true_label": "0"}},
    ]
    records, keys = laya_lane.decision_csv_records(
        "identity", rows, split="dev", population="validated_pairs",
        questions=questions)
    assert [record.record_id for record in records] == ["1|2", "3|4"]
    assert {record.source for record in records} == {"identity_decision_csv"}
    assert len(keys) == len(records) * len(questions)
    assert {key.question_id for key in keys} == set(questions)


def test_decision_csv_records_reject_a_corpus_kind_and_a_duplicate_row(questions):
    with pytest.raises(ValueError, match="not a per-row decision CSV"):
        laya_lane.decision_csv_records("finetune", [], split="dev",
                                       population="p", questions=questions)
    with pytest.raises(ValueError, match="unknown decision kind"):
        laya_lane.decision_csv_records("nope", [], split="dev",
                                       population="p", questions=questions)
    with pytest.raises(ValueError, match="duplicate"):
        laya_lane.decision_csv_records(
            "identity", [{"gtin1": "1", "gtin2": "2"}] * 2, split="dev",
            population="p", questions=questions)


def test_eval_case_records_use_the_harness_case_grain(questions):
    cases = [
        {"qid": "identity_claim", "tags": ["en", "A"], "model": "m"},
        {"qid": "package_state", "tags": [], "model": "m"},
    ]
    records, keys = laya_lane.eval_case_records(
        cases, split="test", population="identity_pairs", questions=questions)
    assert [record.record_id for record in records] == ["case-00000", "case-00001"]
    assert records[0].slices == ("en", "A")
    assert [key.question_id for key in keys] == ["identity_claim", "package_state"]
    assert keys[1].question_type == "noul"


def test_records_traceability_derives_coverage_from_the_carried_records(questions):
    cases = [{"qid": "identity_claim", "tags": ["en"]},
             {"qid": "identity_claim", "tags": ["en", "fresh"]}]
    records, keys = laya_lane.eval_case_records(
        cases, split="test", population="identity_pairs", questions=questions)
    document = laya_lane.records_traceability(
        "laya_cli_eval", records, keys, model_id="m",
        digests={"decision_csv_sha256": "9" * 64},
        overall=_block(items=2), split="test")
    assert document.coverage.by_source == {"laya_cli_eval": 2}
    assert document.coverage.by_dimension["slice"] == {"en": 2, "fresh": 1}


def test_emit_traceability_writes_through_the_layout(tmp_path, monkeypatch):
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setattr(common, "trace_artifact", lambda *args, **kwargs: None)
    document = laya_lane.corpus_traceability(
        {"rows": 1, "items": 1, "eval_split": "dev",
         "after": _block(items=1).model_dump()},
        model_id="m", digests={"corpus_sha256": "5" * 64})
    path = laya_lane.emit_traceability("finetune", document)
    assert path == tmp_path / "logs" / "laya_lane" / "finetune__traceability.json"
    assert json.loads(path.read_text())["schema_id"] == "er-traceability-report-v1"


def test_fetched_traceability_validates_the_train_report(laya_train_report):
    receipt = {"corpus_sha256": {"train.jsonl": "6" * 64},
               "output_dir": "/kaggle/working/checkpoint"}
    documents, artifacts = laya_lane.fetched_traceability(
        receipt, {"train_report.json": laya_train_report, "notes.json": {"a": 1}})
    assert set(documents) == {"train_report.json"}
    assert documents["train_report.json"]["provenance"]["digests"]["corpus_sha256"] == "6" * 64
    assert artifacts == {}


def test_fetched_traceability_fails_loud_on_a_missing_digest(laya_train_report):
    """Falsified 2026-10-08: a receipt naming no corpus digest silently returned
    ``{}``, so a fetched evaluate_records payload vanished from the plan."""
    with pytest.raises(KeyError, match="names no corpus digest"):
        laya_lane.fetched_traceability(
            {}, {"train_report.json": laya_train_report})


def test_a_fetched_member_without_a_metric_block_is_not_silently_skipped():
    """Falsified 2026-10-08: such a member was ``continue``-ed away."""
    documents, _ = laya_lane.fetched_traceability(
        {"corpus_sha256": "7" * 64}, {"train_report.json": {"note": "no metrics"}})
    assert "not_applicable" in documents["train_report.json"]
    assert "evaluate_records" in documents["train_report.json"]["not_applicable"]


# ── the per-row grain: the three adapters' production emit path ─────────────
def _decision_receipt() -> dict:
    """The receipt the identity decision kernel writes (its real key set)."""
    return {"gpu_kind": "identity", "batch_size": 8, "min_confidence": 0.0,
            "decision_csv_sha256": "8" * 64, "split": "dev",
            "output_dir": "/kaggle/working"}


def test_the_fetched_identity_grain_is_emitted_through_the_layout(
        tmp_path, monkeypatch, questions):
    """The production caller the adapters never had: a fetched
    ``<kind>.decisions.jsonl`` becomes a record-derived report, WRITTEN through
    the declared ``traceability_report`` layout."""
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setattr(common, "trace_artifact", lambda *args, **kwargs: None)
    rows = [{"_row": {"gtin1": "1", "gtin2": "2", "true_label": "1"}},
            {"_row": {"gtin1": "3", "gtin2": "4", "true_label": "0"}}]
    documents, artifacts = laya_lane.fetched_traceability(
        _decision_receipt(), {}, decision_kind="identity",
        decision_rows=rows, questions=questions)
    document = documents["record_grain"]
    assert document["coverage"]["records_total"] == 2
    # every census is the records' own, not the kernel's word
    assert document["coverage"]["by_source"] == {"identity_decision_csv": 2}
    assert document["coverage"]["by_dimension"]["split"] == {"dev": 2}
    assert document["coverage"]["by_dimension"]["population"] == {"identity": 2}
    assert document["coverage"]["by_dimension"]["difficulty"] == {"unknown": 2}
    # identity-only: the decision grain measures no metrics of its own
    assert document["overall"] is None
    assert document["coverage"]["items_total"] == 2 * len(questions)
    path = Path(artifacts["record_grain"])
    assert path == tmp_path / "logs" / "laya_lane" / "identity__traceability.json"
    assert json.loads(path.read_text()) == document
    # the written document is self-describing: it re-validates from disk
    assert TraceabilityReport.model_validate(json.loads(path.read_text())) == \
        TraceabilityReport.model_validate(document)


def test_a_fetched_archive_without_the_record_grain_says_so(questions):
    """An archive that does not carry the per-row grain gets an explicit
    ``not_applicable`` entry, never a silent omission."""
    documents, artifacts = laya_lane.fetched_traceability(
        _decision_receipt(), {}, decision_kind="identity",
        decision_rows=(), questions=questions)
    assert artifacts == {}
    assert "decisions.jsonl" in documents["record_grain"]["not_applicable"]
    # a corpus decision kind has no per-row grain at all: also stated, not silent
    documents, _ = laya_lane.fetched_traceability(
        {"corpus_sha256": "a" * 64}, {}, decision_kind="finetune",
        questions=questions)
    assert "corpus grain" in documents["record_grain"]["not_applicable"]


def test_the_evals_case_grain_is_emitted_with_the_harness_aggregates(
        tmp_path, monkeypatch, questions):
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setattr(common, "trace_artifact", lambda *args, **kwargs: None)
    payload = {"cases": [{"qid": "identity_claim", "tags": ["en"]},
                         {"qid": "identity_claim", "tags": ["en", "fresh"]}],
               "accuracy": 0.5, "loss": 0.1, "mean_confidence": 0.5,
               "items": 2, "ece": 0.1, "brier": 0.2, "brier_top1": 0.1}
    documents, artifacts = laya_lane.fetched_traceability(
        {"gpu_kind": "laya-cli-eval", "evals_dataset_sha256": "9" * 64},
        {"report.json": payload}, decision_kind="laya-cli-eval",
        questions=questions)
    document = documents["record_grain"]
    assert document["provenance"]["source"] == "laya_cli_eval"
    assert document["coverage"]["by_dimension"]["slice"] == {"en": 2, "fresh": 1}
    assert document["overall"]["items"] == 2
    assert Path(artifacts["record_grain"]).is_file()


def test_the_fetched_record_grain_needs_the_staged_question_schema(
        tmp_path, monkeypatch):
    """The schema the remote run was fed is the one this box staged; without it
    the (row, qid) grain cannot be addressed, so it fails loud rather than
    guessing question types."""
    monkeypatch.setattr(laya_lane, "staging_dir", lambda: tmp_path / "staging")
    with pytest.raises(FileNotFoundError, match="staged"):
        laya_lane.staged_question_schema()


def test_provenance_accepts_the_digest_the_evals_receipt_really_carries():
    """The kaggle evals kernel writes ``evals_dataset_sha256`` (not
    ``decision_csv_sha256``): the requirement names the input it scored."""
    assert EvalProvenance(source="laya_cli_eval", model_id="m",
                          digests={"evals_dataset_sha256": "b" * 64}).digests
    assert EvalProvenance(source="laya_cli_eval", model_id="m",
                          digests={"decision_csv_sha256": "b" * 64}).digests
    with pytest.raises(ValidationError, match="requires one of digests"):
        EvalProvenance(source="laya_cli_eval", model_id="m")


def test_a_written_traceability_report_revalidates_from_disk(
        tmp_path, monkeypatch, questions):
    """Falsified 2026-10-08: ``dimension_values`` was ``exclude=True``, so the
    document ``write_report`` had just written could not be validated by its own
    contract (``by_dimension coverage mismatch``). A persisted artifact must be
    self-describing."""
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setattr(common, "trace_artifact", lambda *args, **kwargs: None)
    cases = [{"qid": "identity_claim", "tags": ["en", "zeta"]},
             {"qid": "identity_claim", "tags": ["en", "alpha"]}]
    records, keys = laya_lane.eval_case_records(
        cases, split="test", population="identity_pairs", questions=questions)
    document = laya_lane.records_traceability(
        "laya_cli_eval", records, keys, model_id="m",
        digests={"decision_csv_sha256": "9" * 64}, overall=_block(items=2),
        split="test")
    path = laya_lane.emit_traceability("cases", document)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    # the declared universe is part of the artifact, written in a stable order
    assert loaded["coverage"]["dimension_values"]["slice"] == ["alpha", "en", "zeta"]
    assert TraceabilityReport.model_validate(loaded) == document


def test_an_invisible_only_identity_tag_does_not_count_as_tagged():
    """Same visible-blankness rule on this module's own fields (falsified
    2026-10-08): a record cannot be "sliced" or an unknown difficulty "explained"
    with U+200B."""
    with pytest.raises(ValidationError, match="slice names must be nonempty"):
        TaggedRecord(record_id="r1", source="laya_cli_eval", split="test",
                     population="p", slices=("\u200b",), difficulty="unknown",
                     difficulty_reason="the evals harness carries no difficulty axis")
    with pytest.raises(ValidationError, match="unknown difficulty requires"):
        TaggedRecord(record_id="r1", source="laya_cli_eval", split="test",
                     population="p", difficulty="unknown", difficulty_reason="\u200b")
