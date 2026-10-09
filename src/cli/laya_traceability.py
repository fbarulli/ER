"""Laya lane evaluation-traceability adapters (lane-side; core owns the words).

core.eval_trace owns the contract; THIS lane owns its row shapes, the
record-id derivations and the corpus skip census. Every adapter is offline: it
consumes the dict the kernel already wrote, so a fetched report is validated
without laya installed and without re-deriving a single number.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from cli.laya_recipe import (
    DECISION_BINDINGS,
    FINETUNE_CORPUS_FILES,
    KINDS,
    QUESTION_SCHEMA_FILE,
)
from core.coverage_contracts import UNKNOWN_DIMENSION_VALUE
from core.eval_trace import (
    CORE_RECORD_DIMENSIONS,
    AbstentionBlock,
    EvalProvenance,
    EvalRowKey,
    MetricBlock,
    MinConfidenceKey,
    Share,
    TaggedRecord,
    TraceabilityCoverage,
    TraceabilityReport,
    derived_dimension_policies,
    derived_record_census,
    write_report,
)

# The one decision kind whose report is the laya-evals harness (its cases are the
# per-row grain); the rest of the decision CSV kinds report through the staged
# decision CSV itself.
_DECISION_EVAL_SOURCE = {"laya-cli-eval": "laya_cli_eval"}
# The kernel members that carry an evaluate_records payload.
_CORPUS_REPORT_NAMES = ("eval_report.json", "train_report.json")

# ── the FETCHED per-row grain (the record adapters' production caller) ──────
# `--fetch` is the one production path that sees both the remote run's own
# artifacts and this box's staged inputs.
_DECISION_ROWS_SUFFIX = ".decisions.jsonl"
_EVALS_DECISION_KIND = "laya-cli-eval"
_EVALS_REPORT_MEMBER = "report.json"
_RECORD_GRAIN_KEY = "record_grain"
_NOT_APPLICABLE = "not_applicable"
#: Every mandatory ``MetricBlock`` field: an aggregate block is only coerced into
#: one when it exposes the whole surface (see :func:`metric_block_or_none`).
_METRIC_BLOCK_FIELDS = ("items", "loss", "accuracy", "mean_confidence", "ece",
                        "brier", "brier_top1")


class RecordGrainGap(RuntimeError):
    """The fetched archive carries no per-row grain to validate (not an error in
    the run: an explicit ``not_applicable`` reason travels instead)."""


def decision_record_columns(decision_kind: str) -> tuple[str, ...]:
    """The columns that address one decision row (the kernel's ``_row`` tags)."""
    binding = DECISION_BINDINGS.get(decision_kind)
    if binding is None:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    columns = binding.get("record_columns")
    if not columns:
        raise ValueError(
            f"decision kind {decision_kind!r} is not a per-row decision CSV; "
            "its grain is the corpus row (see corpus_traceability)")
    return tuple(columns)


def decision_row_tags(row: dict) -> dict:
    """A decision row's tags: the kernel's ``_row`` when present, else the row."""
    tags = row.get("_row")
    return tags if isinstance(tags, dict) else row


def decision_source(decision_kind: str) -> str:
    """The eval source a per-row decision kind reports through."""
    decision_record_columns(decision_kind)  # fail loud on unknown/corpus kinds
    return _DECISION_EVAL_SOURCE.get(decision_kind, "identity_decision_csv")


def decision_csv_records(decision_kind: str, rows: list[dict], *,
                         split: str, population: str,
                         questions: dict[str, dict]) -> tuple[list, list]:
    """The ``identity_decision_csv`` grain: one TaggedRecord per CSV row, one
    EvalRowKey per (row, qid) the question schema declares."""
    columns = decision_record_columns(decision_kind)
    source = decision_source(decision_kind)
    qtypes = {qid: question["type"] for qid, question in questions.items()}
    records: list[TaggedRecord] = []
    keys: list[EvalRowKey] = []
    seen: set[str] = set()
    for row in rows:
        tags = decision_row_tags(row)
        record_id = "|".join(str(tags.get(column, "")) for column in columns)
        if record_id in seen:
            raise ValueError(f"duplicate decision record id {record_id!r}")
        seen.add(record_id)
        records.append(TaggedRecord(
            record_id=record_id, source=source, split=split,
            population=population, difficulty="unknown",
            difficulty_reason="the decision CSV carries no difficulty axis"))
        for qid, qtype in qtypes.items():
            keys.append(EvalRowKey(record_id=record_id, question_id=qid,
                                   question_type=qtype))
    return records, keys


def eval_case_records(cases: list[dict], *, split: str, population: str,
                      questions: dict[str, dict]) -> tuple[list, list]:
    """The ``laya_cli_eval`` grain: the ``laya.evals`` ``EvalReport.cases`` list.

    The harness carries no row id, so the case ordinal is the only stable
    identity; its free dimensions (language, model, tags) become slices.
    """
    qtypes = {qid: question["type"] for qid, question in questions.items()}
    records: list[TaggedRecord] = []
    keys: list[EvalRowKey] = []
    for index, case in enumerate(cases):
        record_id = f"case-{index:05d}"
        records.append(TaggedRecord(
            record_id=record_id, source="laya_cli_eval", split=split,
            population=population,
            slices=tuple(str(tag) for tag in (case.get("tags") or ())),
            difficulty="unknown",
            difficulty_reason="the evals harness carries no difficulty axis"))
        qid = str(case["qid"])
        keys.append(EvalRowKey(record_id=record_id, question_id=qid,
                               question_type=qtypes[qid]))
    return records, keys


def corpus_skip_census(report: dict) -> dict[str, int]:
    """``items_from_rows``' skip census: labelled questions that could not become
    items, counted by reason."""
    skipped = report.get("skipped") or {}
    if not isinstance(skipped, dict):
        raise ValueError(
            f"skipped census must be a mapping, got {type(skipped).__name__}")
    return {str(reason): int(count) for reason, count in skipped.items()}


@lru_cache(maxsize=1)
def _threshold_adapter():
    from pydantic import TypeAdapter

    return TypeAdapter(dict[MinConfidenceKey, Share])


def abstention_thresholds(mapping: dict | None) -> dict[str, float]:
    return dict(_threshold_adapter().validate_python(mapping or {}))


# The two shapes the finetune note uses to name the train/eval overlap:
# "10/1259 items overlap training data" and "10 items overlap".
_OVERLAP_PATTERNS = (r"(\d+)\s*/\s*(\d+)\s+items?", r"(\d+)\s+items?\s+overlap")


def overlap_items(note: Any) -> int | None:
    """The train/eval overlap count the finetune report names in its note."""
    text = str(note or "")
    for pattern in _OVERLAP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


# The unknown policy per CARRIED core dimension. Each one cites the explicit
# unknown value, because the census reports it: the shared contract refuses a
# policy that leaves the explicit value unexplained.
_CARRIED_UNKNOWN_POLICIES: dict[str, str] = {
    "source": ("the producer tags every record with the eval source it was "
               "measured under; 'unknown' is never a source a producer assigns"),
    "split": ("every record declares exactly one split; a record without one is "
              "tagged 'unknown' rather than omitted from the census"),
    "population": ("every record declares the population it was drawn from; a "
                   "record without one is tagged 'unknown', never dropped"),
    "slice": ("a record carrying no slice is counted under 'unknown' so the "
              "membership census still accounts for the whole population"),
    "difficulty": ("difficulty is measured per record; a record with no "
                   "measurement keeps 'unknown' and carries its own "
                   "difficulty_reason, never an invented easy/hard label"),
}


def coverage_from_records(records, *, items_total: int,
                          split: str | None = None) -> TraceabilityCoverage:
    """The carried-record coverage: EVERY census COUNTED from the records."""
    records = tuple(records)
    if not records:
        raise ValueError("coverage_from_records needs the records it censuses")
    derived = derived_record_census(records)
    policies = derived_dimension_policies(records)
    dimensions = {name: derived[name] for name in CORE_RECORD_DIMENSIONS}
    carried_splits = set(dimensions["split"])
    if split is not None and carried_splits != {split}:
        raise ValueError(
            f"declared split {split!r} is not the split the carried records "
            f"hold ({sorted(carried_splits)})")
    return TraceabilityCoverage(
        records_total=len(records), items_total=items_total,
        by_source=dimensions["source"],
        dimension_values={name: set(counts) for name, counts in dimensions.items()},
        dimension_multiplicity={name: policies[name] for name in dimensions},
        by_dimension=dimensions,
        unknown_policy=dict(_CARRIED_UNKNOWN_POLICIES),
        slice_coverage=policies["slice"])


def records_traceability(source: str, records, keys, *, model_id: str,
                         digests: dict, overall: MetricBlock | None = None,
                         split: str | None = None,
                         **provenance) -> TraceabilityReport:
    """A report whose identity is CARRIED and whose coverage is derived from it."""
    records, keys = tuple(records), tuple(keys)
    if not records:
        raise ValueError(
            "records_traceability needs the records it reports; an aggregate-only "
            "report is built by corpus_traceability, never here")
    carried_sources = {record.source for record in records}
    if carried_sources != {source}:
        raise ValueError(
            f"report source {source!r} is not what the carried records hold "
            f"({sorted(carried_sources)})")
    if not keys:
        raise ValueError(
            "carried records must carry their row keys: the item grain is the "
            "key population, never a declared number")
    items_total = len(keys) if overall is None else overall.items
    return TraceabilityReport(
        provenance=EvalProvenance(source=source, model_id=model_id,
                                  digests=digests, **provenance),
        overall=overall,
        by_type=dict(overall.by_type) if overall else {},
        records=records, keys=keys,
        coverage=coverage_from_records(records, items_total=items_total,
                                       split=split))


def corpus_traceability(report: dict, *, model_id: str, digests: dict,
                        split: str | None = None,
                        records: tuple = (), keys: tuple = ()) -> TraceabilityReport:
    """The fine-tune corpus producer -> the traceability contract."""
    metrics = dict(report.get("after") or report.get("before") or {})
    if not metrics:
        raise ValueError("corpus report carries no before/after metric block")
    overall = MetricBlock.model_validate(metrics)
    rows = int(report.get("rows", report.get("eval_items", overall.items)))
    items = int(report.get("items", overall.items))
    if split is None:
        split = (report.get("eval_split")
                 or Path(str(report.get("eval_source") or "")).stem
                 or UNKNOWN_DIMENSION_VALUE)
    if bool(records) != bool(keys):
        raise ValueError(
            "the corpus grain carries its records and their keys together, or "
            "neither: rows=" + str(len(records)) + " keys=" + str(len(keys)))
    thresholds = abstention_thresholds(report.get("abstention_thresholds"))
    census = report.get("abstention")
    # The eval path may pin the runtime scalar explicitly (`min_confidence`);
    # otherwise the fitted map's "default" sentinel is the gate.
    min_confidence = report.get("min_confidence")
    if min_confidence is None:
        min_confidence = thresholds.get("default")
    if records:
        coverage = coverage_from_records(records, items_total=items, split=split)
    else:
        # Aggregate-only: the declared numbers are the producer's own measured
        # aggregates and the reason is mandatory, so this mode is never silent.
        coverage = TraceabilityCoverage(
            records_total=rows, items_total=items,
            by_source={"finetune_corpus": rows},
            dimension_values={"split": {split}},
            dimension_multiplicity={"split": "partition"},
            by_dimension={"split": {split: rows}},
            unknown_policy={"split": (
                f"the corpus split is declared by the producer ({split!r}); "
                f"a report that declares none is reported as "
                f"{UNKNOWN_DIMENSION_VALUE!r} rather than guessed")},
            slice_coverage="not_applicable",
            slice_coverage_reason=("the corpus producer reports aggregates; "
                                   "per-slice rows are not emitted yet"),
            aggregate_reason=("the corpus receipt carries measured aggregate "
                              "counts and no per-row population; the record-grain "
                              "census arrives only when the rows are carried"))
    return TraceabilityReport(
        provenance=EvalProvenance(
            source="finetune_corpus", model_id=model_id, digests=digests,
            is_held_out=report.get("is_held_out"),
            eval_overlap_items=overlap_items(report.get("note")),
            min_confidence=min_confidence,
            batch_size=int(report.get("batch_size") or 0)),
        overall=overall,
        by_type=dict(overall.by_type),
        skipped=corpus_skip_census(report),
        abstention=(AbstentionBlock.model_validate(census) if census else None),
        records=tuple(records),
        keys=tuple(keys),
        coverage=coverage)


def corpus_digest(receipt: dict) -> str:
    """The corpus digest a fetched lane receipt carries."""
    corpus = receipt.get("corpus_sha256")
    if isinstance(corpus, dict):
        for name in FINETUNE_CORPUS_FILES:
            if isinstance(corpus.get(name), str):
                return corpus[name]
    if isinstance(corpus, str):
        return corpus
    for key in ("eval_jsonl_sha256", "eval_data_sha256"):
        value = receipt.get(key)
        if isinstance(value, str):
            return value
    raise KeyError("receipt carries no corpus digest (corpus_sha256 / "
                   "eval_jsonl_sha256 / eval_data_sha256)")


def metric_block_or_none(payload: dict) -> MetricBlock | None:
    """The harness's aggregate block, ONLY when the whole surface is there."""
    fields = {name: (payload or {})[name] for name in MetricBlock.model_fields
              if name in (payload or {})}
    if not set(_METRIC_BLOCK_FIELDS) <= set(fields):
        return None
    return MetricBlock.model_validate(fields)


def read_decision_rows(body: bytes) -> list[dict]:
    """The kernel's ``<kind>.decisions.jsonl`` bytes -> the per-row answer dicts."""
    rows: list[dict] = []
    for number, line in enumerate(body.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"decision row {number} is a {type(payload).__name__}, not an object")
        rows.append(payload)
    return rows


def staged_question_schema(staging_root: Path) -> dict:
    """The ``questions`` dict of the schema THIS BOX staged, never re-derived."""
    for kind in KINDS:
        path = Path(staging_root) / kind / "question" / QUESTION_SCHEMA_FILE
        if not path.is_file():
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        questions = schema.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(
                f"staged question schema {path} carries no 'questions' dict")
        return questions
    raise FileNotFoundError(
        f"no staged {QUESTION_SCHEMA_FILE} under {staging_root}; stage the payload "
        "before fetching, or the per-row grain cannot be addressed")


def record_grain_traceability(decision_kind: str, *, receipt: dict,
                              reports: dict, decision_rows=(),
                              questions=None, staging_root: Path | None = None
                              ) -> TraceabilityReport:
    """Build the PER-ROW report from the fetched grain (identity CSV / evals cases)."""
    binding = DECISION_BINDINGS.get(decision_kind)
    if binding is None:
        # An external lane kind (e.g. laya-hpo) is not a per-row decision kind:
        # state the gap so `fetched_traceability` records `not_applicable`
        # instead of crashing the harvest of an otherwise valid archive.
        raise RecordGrainGap(
            f"decision kind {decision_kind!r} has no per-row decision binding "
            "(see corpus_traceability)")
    if not binding.get("record_columns"):
        raise RecordGrainGap(
            f"decision kind {decision_kind!r} reports through the corpus grain, "
            "not a per-row decision CSV (see corpus_traceability)")
    source = decision_source(decision_kind)
    digests = {key: receipt[key] for key in
               ("decision_csv_sha256", "evals_dataset_sha256")
               if isinstance(receipt.get(key), str)}
    # What the receipt can name, and the explicit unknown when it names nothing:
    # a guessed 'test' split would be a fabricated tag.
    split = str(receipt.get("split") or receipt.get("eval_split")
                or UNKNOWN_DIMENSION_VALUE)
    population = str(receipt.get("population") or decision_kind)
    model_id = str(receipt.get("checkpoint") or receipt.get("checkpoint_hub")
                   or receipt.get("output_dir") or receipt.get("gpu_kind")
                   or decision_kind)
    if questions is None:
        if staging_root is None:
            raise FileNotFoundError(
                "record_grain_traceability needs the staged question schema "
                "(questions= or staging_root=)")
        questions = staged_question_schema(staging_root)
    if decision_kind == _EVALS_DECISION_KIND:
        payload = reports.get(_EVALS_REPORT_MEMBER)
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list) or not cases:
            raise RecordGrainGap(
                f"the fetched {_EVALS_REPORT_MEMBER} carries no 'cases' list, so the "
                "laya.evals per-case grain cannot be validated")
        records, keys = eval_case_records(
            cases, split=split, population=population, questions=questions)
        overall = metric_block_or_none(payload)
    else:
        if not decision_rows:
            raise RecordGrainGap(
                f"the fetched archive carries no {decision_kind}"
                f"{_DECISION_ROWS_SUFFIX}, so the per-row decision grain cannot be "
                "validated")
        records, keys = decision_csv_records(
            decision_kind, list(decision_rows), split=split,
            population=population, questions=questions)
        overall = None
    return records_traceability(
        source, records, keys, model_id=model_id, digests=digests,
        overall=overall, split=split,
        batch_size=int(receipt.get("batch_size") or 0))


def fetched_traceability(receipt: dict, reports: dict, *,
                         decision_kind: str | None = None,
                         decision_rows=(), questions=None,
                         staging_root: Path | None = None
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a fetched kernel's reports and EMIT the per-row grain it carries."""
    model_id = str(receipt.get("output_dir") or receipt.get("checkpoint")
                   or receipt.get("gpu_kind") or "laya")
    documents: dict[str, Any] = {}
    metric_members = {
        name: payload for name, payload in reports.items()
        if isinstance(payload, dict) and (payload.get("before") or payload.get("after"))}
    if metric_members:
        try:
            digest = corpus_digest(receipt)
        except KeyError as error:
            raise KeyError(
                f"fetched members {sorted(metric_members)} are evaluate_records "
                f"payloads but the receipt names no corpus digest, so their "
                f"provenance cannot be stamped: {error}") from error
        for name in sorted(metric_members):
            documents[name] = corpus_traceability(
                metric_members[name], model_id=model_id,
                digests={"corpus_sha256": digest}).model_dump(mode="json")
    else:
        for name in _CORPUS_REPORT_NAMES:
            if isinstance(reports.get(name), dict):
                documents[name] = {_NOT_APPLICABLE: (
                    f"{name} carries no before/after evaluate_records block to "
                    "validate")}
    artifacts: dict[str, Any] = {}
    if decision_kind is not None:
        try:
            document = record_grain_traceability(
                decision_kind, receipt=receipt, reports=reports,
                decision_rows=decision_rows, questions=questions,
                staging_root=staging_root)
        except RecordGrainGap as gap:
            documents[_RECORD_GRAIN_KEY] = {_NOT_APPLICABLE: str(gap)}
        else:
            artifacts[_RECORD_GRAIN_KEY] = str(
                emit_traceability(decision_kind, document))
            documents[_RECORD_GRAIN_KEY] = document.model_dump(mode="json")
    return documents, artifacts


def emit_traceability(track: str, document: TraceabilityReport, *,
                      lane: str = "laya_lane") -> Path:
    """Write a lane traceability document to the declared ``traceability_report``
    layout (core.common owns the template; core.eval_trace stamps the write)."""
    return write_report("traceability_report", {"lane": lane, "track": track},
                        document)
