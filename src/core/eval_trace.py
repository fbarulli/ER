"""General evaluation-traceability contracts (per-slice / per-attribute grain).

This module owns the *vocabulary* every evaluation producer reports through. It
is deliberately distinct from its neighbours:

* ``core.tracing`` — the ONE consolidated pipeline trace (data flow, one CSV);
* ``core.step_trace`` — function-level timing on ``timings.log``;
* ``core.coverage_contracts`` — completeness of *measured attributes*;
* ``core.audit_json`` — a row-level audit CSV streamed into one JSON array.

GRAINS (three producers, one contract):

* ``finetune_corpus``        — one corpus row may label zero, one or several
                               qids; the laya trainer's ``items_from_rows``
                               turns it into one item per labelled (row, qid)
                               and returns a ``skipped`` census by reason. The
                               row key travels BESIDE the item, never inside it.
* ``identity_decision_csv``  — the staged decision CSV already carries row tags
                               (``cli.laya_lane.DECISION_BINDINGS``) and the
                               decision kernel attaches ``answer['_row']``.
* ``laya_cli_eval``          — the ``laya.evals`` harness reports aggregates
                               PLUS a per-case record list (``EvalReport.cases``).
* ``tracks_suite``           — emitter-aligned rows, validated verbatim.

``cascade`` is deliberately NOT a source here. Verified 2026-10-08: the cascade
lane emits no per-row eval records (only ``cascade_report.json``), so a declared
``EvalSource`` for it would be aspirational, and a source no producer emits is
exactly what this contract exists to prevent. Cascade completeness is validated
by the GENERAL per-record contract instead -
``core.coverage_contracts.ReportCoverageContract``, adopted by
``graph_tracks.report``.

The contract is a PREDICATE over the emitter's own dicts: producers keep
producing exactly what they produce and the emitted bytes are untouched. Lane
knowledge (column allowlists, frozen thresholds, row-id derivations) lives in
the owning lane module, never here; this module only owns the shared vocabulary
plus the two primitives a lane adapter needs to read a persisted row back
(``PersistedCount`` / ``row_from_csv``) and the declared destination
(``write_report``).

The design was independently reviewed twice; corrections C1-C13 of the second
review are applied here. The rejected draft's broken ``decision_flip`` chained
comparison is gone: a row model has no threshold, so the flip is decided by
``decision_flip(...)`` inside the lane adapter that owns the frozen threshold.

WHICH CONTRACT A NEW PRODUCER MUST USE (the decision rule):

* A producer that CAN carry its scored rows - one ``TaggedRecord`` per row, with
  a stable record id - reports through THIS module: ``TraceabilityReport`` plus
  ``TraceabilityCoverage``. Records are authoritative: whenever they are carried,
  EVERY census in the coverage (``records_total``, ``by_source``, the item count
  behind ``keys`` and each ``by_dimension`` map) is RE-DERIVED from them and any
  disagreement fails loud (``TraceabilityCoverage.require_carried_records``). A
  report that carries no records at all must say so explicitly
  (``coverage.aggregate_reason``); there is no silent "assume the numbers are
  fine" path, and a dimension a record cannot carry cannot be censused here.
* A producer measuring ONE axis over a population it cannot carry per row uses
  ``core.coverage_contracts.ReportCoverageContract`` (records mandatory there
  too: the ablation cohort's per-pair strata and the attribute-separation axis
  both adopt it).
* The per-artifact contracts (``CohortCoverage``, ``AttributeCensus``, ...) also
  live in ``core.coverage_contracts``, which owns the multiplicity policy and
  the explicit unknown value. This module IMPORTS that vocabulary and does not
  re-declare it.

This module does NOT delegate to ``ReportCoverageContract``: no laya-lane code
constructs one today, so the two must not be described as one contract.

THE SHARED DIMENSION VOCABULARY. One name per axis: ``Dimension`` is the
report's declared vocabulary, ``RecordDimension`` the subset a carried record
actually tags, and ``DIMENSION_ALIASES`` resolves the producer-side spellings of
an axis (``difficulty_slice`` is ``difficulty``, ``slices`` is ``slice``). A name
outside the vocabulary (``banana``) fails loud in ``canonical_dimension``, and a
declared census is canonicalized through it, so the ablation cohort's
``difficulty_slice`` column and this module's ``difficulty`` dimension cannot
drift into two axes.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from core.coverage_contracts import (
    UNKNOWN_DIMENSION_VALUE,
    Count,
    CoverageModel,
    Difficulty,
    DimensionPolicy,
    Share,
    blank,
    require_keys,
)

# ── shared scalars ──────────────────────────────────────────────────────────
# QTYPES / QTYPE_NAMES are laya's own axis; never re-declared here, only pinned
# as the accepted names.
QTypeName = Literal["choice", "score", "noul"]
N_QTYPES = len(QTypeName.__args__)

# A fitted temperature is PINNED to the invariant (>0), not to laya's clamp: a
# relaxed upstream clamp must not fail our contract.
Temperature = Annotated[float, Field(gt=0, allow_inf_nan=False)]

# NOT a Share. The multi-class Brier score is mean(sum((p - t) ** 2)); its
# supremum is 2 for EVERY k >= 2 (two disjoint point masses), so a Share
# (le=1) would reject real output. (2 * (1 - 1/k) is NOT this bound: it gives
# 1.0 at k = 2, where the real sup is 2.)
Brier = Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]

# laya's temp_bucket() keys: "<qtype>:<size>" with size in {2,3-5,6-10,11+}.
BucketKey = Annotated[str, Field(pattern=r"^(choice|score|noul):(2|3-5|6-10|11\+)$")]
# The confidence gate's fitted map carries a runtime "default" sentinel beside
# the bucket keys (the block that answers when no bucket matched).
MinConfidenceKey = BucketKey | Literal["default"]

# A count read back FROM DISK. ``coverage_contracts.Count`` is strict=True by
# design (in-process counters): it rejects "7", 7.0 and numpy ints, i.e. every
# pandas/csv-sourced row. Emitter-aligned rows may legitimately be validated
# from a persisted artifact, so they use this instead.
PersistedCount = Annotated[int, Field(ge=0)]

EvalSource = Literal[
    "finetune_corpus", "identity_decision_csv", "laya_cli_eval",
    "tracks_suite",
]
# The multiplicity policy vocabulary ("partition" | "overlap" |
# "not_applicable") and the ``"unknown"`` dimension value are OWNED by
# ``core.coverage_contracts`` and imported above: one declaration, two
# consumers. ``DimensionPolicy`` stays in ``__all__`` as a re-export.
# The dimensions the emitters actually populate. ``gate_decision``/``gate_reason``
# are kept because the sampling ledger does carry them; the rest were missing
# from the draft while real rows populate them.
Dimension = Literal[
    "source", "split", "population", "slice", "difficulty",
    "evaluation_scope",  # training_diagnostic | bundle_diagnostic | ... | heldout
    "eval_half",         # EVAL_SUMMARY_COLUMNS in core.schemas
    "gate_decision", "gate_reason", "attribute", "family", "question_type",
    "channel",           # text | graph | both
    "masking_profile",   # the ablation intervention axis (what was masked)
    "abstention",        # passed | abstained | unevaluated
    "skip_reason",       # the corpus builder's items_from_rows reasons
]

# ── the record-carried dimension vocabulary (ONE, shared) ───────────────────
# ``Dimension`` above is what a REPORT may declare; ``RecordDimension`` is the
# subset a CARRIED ``TaggedRecord`` actually carries, and therefore the only
# subset whose census can be re-derived from the records (``record_dimension_tags``
# is the ONE mapping from a record to its tags, never a second registry).
RecordDimension = Literal[
    "source", "split", "population", "slice", "difficulty",
    "gate_decision", "gate_reason", "attribute", "family",
]
#: The dimensions EVERY TaggedRecord populates, so a carried report must census
#: all of them: source/split/population/difficulty are mandatory fields and
#: ``slice`` is the (possibly empty) membership tuple.
CORE_RECORD_DIMENSIONS: tuple[str, ...] = (
    "source", "split", "population", "slice", "difficulty",
)
#: ONE name per axis: the producer-side spelling resolves to the shared name.
#: The ablation cohort frame names its difficulty column ``difficulty_slice``;
#: a census declares the canonical name, so the two cannot drift into two axes.
DIMENSION_ALIASES: dict[str, str] = {
    "difficulty_slice": "difficulty",
    "slices": "slice",
}


def canonical_dimension(name: str) -> str:
    """The ONE shared name for a traceability dimension.

    An alias resolves (``difficulty_slice`` -> ``difficulty``); a name outside
    the vocabulary (``banana``) fails loud rather than being carried into a
    census nobody can read.
    """
    resolved = DIMENSION_ALIASES.get(name, name)
    if resolved not in Dimension.__args__:
        raise ValueError(
            f"unknown traceability dimension {name!r}; expected one of "
            f"{list(Dimension.__args__)} (aliases: {sorted(DIMENSION_ALIASES)})")
    return resolved


def canonical_dimension_map(mapping):
    """Canonicalize a declared dimension map, refusing an alias collision.

    ``{'difficulty': ..., 'difficulty_slice': ...}`` names the SAME axis twice
    with different content, so it is an error, never a silent overwrite.
    """
    canonical: dict = {}
    for name, value in mapping.items():
        resolved = canonical_dimension(name)
        if resolved in canonical:
            raise ValueError(
                f"dimension {resolved!r} is declared twice (alias collision)")
        canonical[resolved] = value
    return canonical


def record_dimension_tags(record: "TaggedRecord") -> dict[str, tuple[str, ...]]:
    """Every dimension tag a record carries, with the ABSENT case explicit.

    A dimension the record does not carry (no gate decision, no attribute, no
    slice) is tagged with the explicit unknown value instead of being omitted, so
    the derived census always accounts for the whole population and an absent
    value can never masquerade as a measured stratum.
    """
    return {
        "source": (record.source,),
        "split": (record.split,),
        "population": (record.population,),
        "slice": record.slices or (UNKNOWN_DIMENSION_VALUE,),
        "difficulty": (record.difficulty,),
        "gate_decision": (record.gate_decision or UNKNOWN_DIMENSION_VALUE,),
        "gate_reason": (record.gate_reason or UNKNOWN_DIMENSION_VALUE,),
        "attribute": (record.attribute or UNKNOWN_DIMENSION_VALUE,),
        "family": (record.family or UNKNOWN_DIMENSION_VALUE,),
    }


def _tally_dimensions(records) -> tuple[dict[str, dict[str, int]],
                                        dict[str, int]]:
    """(value -> records carrying it) per dimension, plus each dimension's
    widest tag tuple: the ONE derivation every consumer shares."""
    tally = {name: Counter() for name in RecordDimension.__args__}
    widths = {name: 0 for name in RecordDimension.__args__}
    for record in records:
        for name, values in record_dimension_tags(record).items():
            tally[name].update(values)
            widths[name] = max(widths[name], len(values))
    return ({name: dict(counts) for name, counts in tally.items()}, widths)


def derived_record_census(records) -> dict[str, dict[str, int]]:
    """The census of EVERY record-carried dimension, re-derived from the records.

    The ONE re-derivation: the report validator and ``coverage_from_records``
    read the same numbers, so an audited census can never be recomputed
    differently by a consumer.
    """
    return _tally_dimensions(tuple(records))[0]


def derived_dimension_policies(records) -> dict[str, DimensionPolicy]:
    """Each dimension's multiplicity, DERIVED from how the records tag it.

    A dimension some record carries several values of is a MEMBERSHIP axis
    (``overlap``); one every record carries exactly once is a ``partition``. The
    policy is therefore a property of the carried data, not a producer's claim.
    """
    widths = _tally_dimensions(tuple(records))[1]
    return {name: ("overlap" if widths[name] > 1 else "partition")
            for name in RecordDimension.__args__}


# ── metrics ─────────────────────────────────────────────────────────────────
class MetricBlock(BaseModel):
    """The metric surface an evaluation producer reports for one population.

    Field names are the emitter's own. ``by_type`` is declared RECURSIVELY
    because ``evaluate_records`` returns it inside the same dict; an
    ``extra='forbid'`` model without it rejects the laya lane's real output.

    ``temperature`` / ``temperature_by_options`` / ``n_by_bucket`` come from
    ``fit_temperature_map`` and are optional: a producer may merge the fitted
    calibration into the block it reports, or omit it and report the
    ``evaluate_records`` surface alone.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    items: Count
    loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    accuracy: Share
    mean_confidence: Share
    # ece is None when fewer than two scored records exist.
    ece: Share | None
    brier: Brier | None
    brier_top1: Share | None
    by_type: dict[QTypeName, "MetricBlock"] = Field(default_factory=dict)
    # per-type sequence, one scalar per qtype id.
    temperature: list[Temperature] | None = None
    temperature_by_options: dict[BucketKey, Temperature] = Field(default_factory=dict)
    n_by_bucket: dict[BucketKey, Count] = Field(default_factory=dict)

    @model_validator(mode="after")
    def temperature_shape(self):
        if self.temperature is not None and len(self.temperature) != N_QTYPES:
            raise ValueError(
                "temperature must carry one scalar per qtype (choice/score/noul)")
        if self.temperature is None and (self.temperature_by_options or self.n_by_bucket):
            raise ValueError("per-bucket temperatures require the per-type sequence")
        # The fit omits sub-MIN_BUCKET_N buckets from temperature_by_options while
        # n_by_bucket counts the FULL input, so only this direction holds.
        if not set(self.temperature_by_options) <= set(self.n_by_bucket):
            raise ValueError("a fitted bucket must be one of the counted buckets")
        if self.n_by_bucket and sum(self.n_by_bucket.values()) != self.items:
            raise ValueError("calibration buckets do not account for every item")
        return self


MetricBlock.model_rebuild()


class AbstentionBlock(BaseModel):
    """The confidence gate's census and its per-bucket thresholds.

    Three states on purpose: the gate distinguishes ``unevaluated`` (the gate
    could not decide) from ``passed``, and a two-state block could not represent
    it without making the producer lie. ``thresholds_by_bucket`` is a map, not a
    scalar: the fitted thresholds are keyed by ``temp_bucket`` plus a runtime
    ``"default"``.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_confidence: Share | None = None
    thresholds_by_bucket: dict[MinConfidenceKey, Share] = Field(default_factory=dict)
    items: Count
    passed: Count
    abstained: Count
    unevaluated: Count
    accuracy_on_passed: Share | None = None

    @model_validator(mode="after")
    def accounted(self):
        if self.passed + self.abstained + self.unevaluated != self.items:
            raise ValueError("abstention states do not account for every item")
        if self.accuracy_on_passed is not None and self.passed == 0:
            raise ValueError("accuracy_on_passed with zero passed items")
        return self


# ── identity ────────────────────────────────────────────────────────────────
class TaggedRecord(BaseModel):
    """The identity contract every evaluated record must CARRY.

    ``slices`` is plural on purpose: producers assign one row to several slices
    at once, so a singular ``slice`` would silently drop memberships.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    record_id: str = Field(min_length=1)
    source: EvalSource
    split: str = Field(min_length=1)
    population: str = Field(min_length=1)
    slices: tuple[str, ...] = ()
    difficulty: Difficulty = "unknown"
    difficulty_reason: str | None = None
    gate_decision: Literal["proceed", "hard_no", "fallback"] | None = None
    gate_reason: str | None = None
    attribute: str | None = None
    family: str | None = None

    @model_validator(mode="after")
    def identity_complete(self):
        if any(blank(value) for value in self.slices):
            raise ValueError("slice names must be nonempty")
        if self.difficulty == "unknown" and blank(self.difficulty_reason or ""):
            raise ValueError("unknown difficulty requires an explicit reason")
        if self.gate_decision is None and not blank(self.gate_reason or ""):
            raise ValueError("gate_reason without a gate_decision")
        return self


class EvalRowKey(BaseModel):
    """The true metric grain: one metric instance per (record, question)."""
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: QTypeName


class EvalProvenance(BaseModel):
    """What produced this report and which inputs it rests on.

    Inputs live in ONE declared mapping rather than N hard-coded field names:
    the real receipts key their inputs differently per artifact (the corpus
    receipt carries ``corpus_size`` and a nested size map; the decision
    receipt carries ``decision_csv_size``). The per-source requirement below
    names only the key the contract demands for that source. Every value is a
    structural byte size — no content is fingerprinted anywhere.

    The mapping is named ``digests`` because that is the name the writer, the
    fetched-lane adapters and this module's own contract suite all use, and it
    is the persisted JSON key; renaming the field to ``inputs`` stranded every
    producer with ``extra_forbidden`` (falsified 2026-10-09: the fetched
    record-grain artifact failed to build).
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source: EvalSource
    model_id: str = Field(min_length=1)
    laya_package: str | None = None
    digests: dict[str, str] = Field(default_factory=dict)
    # A fine-tune eval's most important traceability fact: was the scored split
    # really held out, and how many items overlapped training data.
    is_held_out: bool | None = None
    eval_overlap_items: Count | None = None
    min_confidence: Share | None = None
    batch_size: Count = 0
    revision: str | None = None

    @model_validator(mode="after")
    def source_inputs_present(self):
        # The input keys a source may name for its scored INPUT; at
        # least one must be present. ``laya_cli_eval`` accepts either the
        # decision CSV the harness derived its dataset from or that dataset
        # itself: the kaggle evals kernel writes ``evals_dataset_size``, so
        # demanding a key that producer cannot emit would make the requirement
        # unfalsifiable-but-unsatisfiable.
        required = {
            "finetune_corpus": ("corpus_size",),
            "identity_decision_csv": ("decision_csv_size",),
            "laya_cli_eval": ("decision_csv_size", "evals_dataset_size"),
        }.get(self.source)
        if required and not (set(required) & set(self.digests)):
            raise ValueError(
                f"source={self.source} requires one of digests[{', '.join(required)}]")
        return self


# ── strict aggregates (producers that own their field names) ─────────────────
class SliceMetrics(BaseModel):
    """Per-slice aggregate for a producer that owns its field names."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    slice: str = Field(min_length=1)
    split: str = Field(min_length=1)
    population: str | None = None
    rows: Count
    items: Count
    metrics: MetricBlock


class AttributeAttribution(BaseModel):
    """Per-attribute aggregate (the flip / baseline / ablated surface).

    Deliberately NOT the emitter row shape: the tracks emitter's rows are
    validated verbatim by ``AttributeAttributionRow`` below.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    attribute: str = Field(min_length=1)
    family: str | None = None
    affected_records: Count
    baseline_accuracy: Share
    ablated_accuracy: Share
    delta_accuracy: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
    flip_rate: Share
    mean_signed_delta_p: Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)]
    metrics: MetricBlock | None = None

    @model_validator(mode="after")
    def delta_is_measured(self):
        if abs(self.delta_accuracy - (self.ablated_accuracy - self.baseline_accuracy)) > 1e-9:
            raise ValueError("delta_accuracy disagrees with baseline/ablated")
        return self


# ── emitter-aligned rows (validate verbatim; never rename, never reserialize) ─
class SliceMetricsRow(BaseModel):
    """One POPULATED slice-metrics row.

    ``extra='allow'`` carries the dynamic metric columns (``p_at_r*``,
    ``pooled_*``): nothing is enumerated here, so retuning ``operating_recall``
    or ``retrieval_ks`` cannot break validation — and no second metric registry
    is maintained in ``core``.
    """
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    model: str = Field(min_length=1)
    split: str = Field(min_length=1)
    slice: str = Field(min_length=1)
    evaluated: Literal[True]
    rows: PersistedCount
    positive_pairs: PersistedCount
    negative_pairs: PersistedCount
    both_classes: bool

    @model_validator(mode="after")
    def pair_counts_account(self):
        if self.positive_pairs + self.negative_pairs != self.rows:
            raise ValueError("slice pair counts do not account for every row")
        return self


class UnmeasuredSliceRow(BaseModel):
    """One EMPTY slice row: the emitter writes a DIFFERENT key set for it, so it
    is a distinct class, never merged into the populated one.

    ``extra='allow'`` because the row is also read back from disk, where the CSV
    carries the UNION header: pandas writes the empty row with empty-string cells
    for every metric column it does not populate. The emptiness is pinned by the
    ``Literal`` fields, not by the absence of unknown keys.
    """
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    model: str = Field(min_length=1)
    split: str = Field(min_length=1)
    slice: str = Field(min_length=1)
    rows: Literal[0]
    positive_pairs: Literal[0]
    negative_pairs: Literal[0]
    both_classes: Literal[False]
    evaluated: Literal[False]


class AttributeAttributionRow(BaseModel):
    """One row of the attribute-ablation report; field names are the emitter's.

    ``extra='allow'`` carries the ``**pair`` columns and the configured slice
    columns. Container shapes owned by the ablation lane are typed loosely on
    purpose: this contract pins presence and the accounting identities, not the
    internals of another lane's containers.

    There is deliberately NO ``decision_flip`` predicate here: the model has no
    threshold, and the draft's chained ``!=`` comparison with a hardcoded ``0.0``
    was both inverted and untestable. The flip is decided by the lane adapter
    that owns the frozen threshold, via :func:`decision_flip`.
    """
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    sku_id1: str
    sku_id2: str
    label: Literal["0", "1"]
    split: str = Field(min_length=1)
    attribute: str | None
    channel: str | None
    baseline_score: float
    ablated_score: float
    score_delta: float
    decision_flip: bool
    baseline_error: bool
    ablated_error: bool
    changed_listings: PersistedCount
    endpoint_input_changed: list[bool]
    embedding_cosine_delta: list[float]
    baseline_ranks: list[Any]
    ablated_ranks: list[Any]
    ann_baseline_hits: Any
    ann_ablated_hits: Any
    known_positive_recall_change: dict[str, list[int | None]]
    current_attribute_evidence: dict[str, Any] | None
    evidence_scope: str | None = None

    @model_validator(mode="after")
    def accounting(self):
        if abs(self.score_delta - (self.ablated_score - self.baseline_score)) > 1e-9:
            raise ValueError("score_delta disagrees with ablated-baseline")
        if len(self.endpoint_input_changed) != 2 or len(self.embedding_cosine_delta) != 2:
            raise ValueError("endpoint evidence must cover both endpoints")
        # The emitter ALWAYS writes a dict; the absent-evidence case is a dict
        # whose entries are None, paired with an explicit evidence_scope string.
        evidence = self.current_attribute_evidence or {}
        if (not evidence or all(value is None for value in evidence.values())) \
                and blank(self.evidence_scope or ""):
            raise ValueError("absent attribute evidence requires an explicit evidence_scope")
        return self


def decision_flip(baseline_score: float, ablated_score: float, threshold: float) -> bool:
    """The frozen-threshold flip predicate.

    The row model has no threshold, so the lane adapter that owns it (and only
    that adapter) decides whether the baseline's verdict and the ablated verdict
    disagree. Kept as a named primitive so ``>= threshold`` is written once.
    """
    return (ablated_score >= threshold) != (baseline_score >= threshold)


# ── persisted-artifact entry points ─────────────────────────────────────────
def row_from_csv(row: dict) -> dict:
    """A CSV cell dict -> the emitter's dict shape, explicitly and once.

    pandas writes booleans as ``'True'``/``'False'`` and every count as a
    string, so a persisted row cannot be validated by ``Literal[True]``/strict
    ``Count`` without this coercion. It is the ONE place the disk -> emitter
    mapping lives; callers then feed the result to the row model.

    ``both_classes`` is coerced too: the empty slice row pins it with
    ``Literal[False]``, which (unlike plain ``bool``) does not accept the string
    ``'False'``.
    """
    coerced = dict(row)
    for key in ("evaluated", "both_classes"):
        value = coerced.get(key)
        if isinstance(value, str):
            coerced[key] = value.strip().lower() == "true"
    for key in ("rows", "positive_pairs", "negative_pairs"):
        value = coerced.get(key)
        if isinstance(value, str):
            coerced[key] = int(value) if value.strip() else 0
    return coerced


# ── coverage / completeness ─────────────────────────────────────────────────
class TraceabilityCoverage(CoverageModel):
    """Every dimension value accounted; identity is RE-DERIVED, never asserted.

    Two layers of accounting, and neither one is a producer's word:

    1. STRUCTURE (this validator, always): the declared dimension key-sets agree
       (``dimension_values`` / ``dimension_multiplicity`` / ``by_dimension`` /
       ``unknown_policy``), every declared name is canonical (``banana`` fails
       loud, ``difficulty_slice`` resolves to ``difficulty``), every census
       respects its declared multiplicity, and the ``not_applicable`` idiom keys
       on the explicit unknown value. An ``unknown_policy`` is never vacuous: a
       dimension whose census counts the explicit ``unknown`` value must state
       how that value is treated; ``not_applicable`` requires an explicit reason.
    2. EVIDENCE (``require_carried_records``, whenever records are carried): the
       record population, the source strata, the item count behind the carried
       ``keys`` and EVERY ``by_dimension`` map are re-derived from the records
       themselves, so a report cannot claim rows=999 while carrying seven, and
       cannot declare a stratum census no record tags.

    ``dimension_values`` is the declared universe and IS serialized, so a written
    artifact is self-describing: a re-reader validates the document it loaded
    without re-deriving the universe. ``aggregate_reason`` is the explicit
    declaration that makes the aggregate-only case (no carried records) a stated
    fact rather than a silent gap; the report validator requires it.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    records_total: Count
    items_total: Count
    by_source: dict[EvalSource, Count]
    dimension_values: dict[Dimension, set[str]] = Field(default_factory=dict)
    dimension_multiplicity: dict[Dimension, DimensionPolicy]
    by_dimension: dict[Dimension, dict[str, Count]]
    unknown_policy: dict[Dimension, str]
    slice_coverage: DimensionPolicy
    slice_coverage_reason: str | None = None
    #: Why this coverage carries no records to derive from. Mandatory for an
    #: aggregate-only report (see ``TraceabilityReport.contracts``); a report
    #: that carries records must leave it unset, and IS re-derived instead.
    aggregate_reason: str | None = None

    @field_validator("dimension_values", "dimension_multiplicity", "by_dimension",
                     "unknown_policy", mode="before")
    @classmethod
    def _canonical_dimension_names(cls, value):
        """Bind every declared dimension name to the shared vocabulary.

        This is the ONLY place a declared name enters the model, so an alias
        (``difficulty_slice``) resolves once and a name outside the vocabulary
        (``banana``) fails loud instead of riding into a census.
        """
        if isinstance(value, dict):
            return canonical_dimension_map(value)
        return value

    @field_serializer("dimension_values")
    def _canonical_dimension_values(self, value):
        """Sort each declared value set so a written artifact's bytes are
        stable: a ``set`` would otherwise serialize in process-dependent order
        and make two identical runs differ."""
        return {dimension: sorted(values) for dimension, values in value.items()}

    @model_validator(mode="after")
    def complete(self):
        if not self.by_source:
            raise ValueError("coverage must name at least one source stratum")
        if sum(self.by_source.values()) != self.records_total:
            raise ValueError("source strata do not account for every record")
        for mapping, name in ((self.by_dimension, "by_dimension"),
                              (self.dimension_multiplicity, "multiplicity"),
                              (self.unknown_policy, "unknown policy")):
            require_keys(mapping, self.dimension_values, name)
        for dimension, counts in self.by_dimension.items():
            require_keys(counts, self.dimension_values[dimension], f"dimension {dimension}")
            policy_text = self.unknown_policy[dimension]
            if blank(policy_text):
                raise ValueError(f"unknown policy for {dimension} must be explicit")
            policy = self.dimension_multiplicity[dimension]
            # Non-vacuous: a census that counts the explicit unknown value must
            # say how that value is treated. ``not_applicable`` is exempt because
            # its census is already pinned to the unknown value alone below.
            if (policy != "not_applicable"
                    and UNKNOWN_DIMENSION_VALUE in counts
                    and UNKNOWN_DIMENSION_VALUE not in policy_text):
                raise ValueError(
                    f"unknown policy for {dimension} must name the explicit "
                    f"{UNKNOWN_DIMENSION_VALUE!r} value its census counts")
            total = sum(counts.values())
            if policy == "partition" and total != self.records_total:
                raise ValueError(f"{dimension} does not partition every record")
            if policy == "overlap" and total < self.records_total:
                raise ValueError(f"{dimension} under-accounts the record population")
            if policy == "not_applicable" and not (
                    counts.get(UNKNOWN_DIMENSION_VALUE, 0) == self.records_total
                    and all(v == 0 for k, v in counts.items()
                            if k != UNKNOWN_DIMENSION_VALUE)):
                raise ValueError(f"{dimension} is not_applicable but carries measured values")
        if self.slice_coverage == "not_applicable" and blank(
                self.slice_coverage_reason or ""):
            raise ValueError("not_applicable slice coverage requires a reason")
        if self.aggregate_reason is not None and blank(self.aggregate_reason):
            raise ValueError("an aggregate-only reason must carry visible text")
        return self

    # ── the EVIDENCE layer: every census re-derived from the carried records ──
    def require_carried_records(self, records, keys) -> None:
        """Re-derive EVERY census from ``records`` and fail on any disagreement.

        Called by ``TraceabilityReport`` whenever records are carried, so the
        coverage is a property of the carried data:

        * ``records_total``        == ``len(records)``;
        * ``by_source``            == the carried source census;
        * the key ids              == exactly the carried record ids, and
          ``items_total``          == the number of carried keys;
        * every declared dimension is one a ``TaggedRecord`` can carry, is
          declared (the five core axes are mandatory), and its census AND its
          declared value universe equal what the records tag;
        * the declared multiplicity matches how the records tag the axis (a
          record carrying several values of a dimension makes it ``overlap``);
        * an ``unknown`` difficulty carries its own reason, per record.

        The records are also the authority for the vocabulary: a dimension name
        the records cannot carry cannot be censused at all.
        """
        records = tuple(records)
        keys = tuple(keys)
        if not records:
            raise ValueError("coverage cannot be checked against zero carried records")
        derived = derived_record_census(records)
        if self.records_total != len(records):
            raise ValueError(
                "coverage record count disagrees with the carried records: "
                f"declared {self.records_total}, carried {len(records)}")
        if dict(self.by_source) != derived["source"]:
            raise ValueError("coverage source strata disagree with the carried records")
        if {key.record_id for key in keys} != {record.record_id for record in records}:
            raise ValueError("carried keys do not cover exactly the carried records")
        if len(keys) != self.items_total:
            raise ValueError(
                "coverage item total disagrees with the carried keys: "
                f"declared {self.items_total}, carried {len(keys)}")
        missing = [name for name in CORE_RECORD_DIMENSIONS
                   if name not in self.dimension_values]
        if missing:
            raise ValueError(
                f"carried records demand a census for {missing}; declared "
                f"dimensions: {sorted(self.dimension_values)}")
        widths = _tally_dimensions(records)[1]
        for dimension, declared in self.by_dimension.items():
            if dimension not in RecordDimension.__args__:
                raise ValueError(
                    f"{dimension!r} is not carried by a TaggedRecord, so its "
                    "census cannot be re-derived from the carried records")
            carried = derived[dimension]
            require_keys(declared, carried, f"{dimension} declared counts")
            if dict(declared) != carried:
                raise ValueError(
                    f"{dimension}: declared counts disagree with the carried records")
            if set(self.dimension_values[dimension]) != set(carried):
                raise ValueError(
                    f"{dimension}: declared value universe disagrees with the "
                    "carried records")
            policy = self.dimension_multiplicity[dimension]
            if widths[dimension] > 1 and policy != "overlap":
                raise ValueError(
                    f"{dimension}: records carry several values each, so the "
                    f"multiplicity is 'overlap', not {policy!r}")
        unexplained = [record.record_id for record in records
                       if record.difficulty == UNKNOWN_DIMENSION_VALUE
                       and blank(record.difficulty_reason or "")]
        if unexplained:
            raise ValueError(
                "an unknown difficulty requires a reason per record; "
                f"{len(unexplained)} carried record(s) carry none "
                f"(e.g. {unexplained[:3]})")


class TraceabilityReport(BaseModel):
    """One traceability report per (producer, lane, track, split).

    ``overall`` is the metric surface; it is OPTIONAL only so an IDENTITY-ONLY
    report (the per-row decision grain, which measures no metrics of its own) can
    carry its records without inventing a metric block. A report must carry
    metrics, records, or both, and a report carrying no records must DECLARE that
    its coverage is aggregate-only (``coverage.aggregate_reason``), so "no
    evidence at all" is never the silent default.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_id: Literal["er-traceability-report-v1"] = "er-traceability-report-v1"
    provenance: EvalProvenance
    overall: MetricBlock | None = None
    by_type: dict[QTypeName, MetricBlock] = Field(default_factory=dict)
    by_slice: list[SliceMetrics | SliceMetricsRow | UnmeasuredSliceRow] = Field(default_factory=list)
    by_attribute: list[AttributeAttribution | AttributeAttributionRow] = Field(default_factory=list)
    abstention: AbstentionBlock | None = None
    # The corpus builder's skip census: labelled questions that could not become
    # items, by reason. Counted at the (row, qid) grain, so it is NOT a coverage
    # dimension (coverage counts records).
    skipped: dict[str, Count] = Field(default_factory=dict)
    # The identity contract, CARRIED. When records are carried they are
    # authoritative: EVERY coverage census is re-derived from them, and their keys
    # must be carried too (the item population is then the key count, never a
    # second declared number). A report that carries none must declare
    # ``coverage.aggregate_reason``.
    records: tuple[TaggedRecord, ...] = ()
    keys: tuple[EvalRowKey, ...] = ()
    coverage: TraceabilityCoverage

    @model_validator(mode="after")
    def contracts(self):
        # No require_keys on by_type: a producer emits only the types that have
        # items, e.g. {'noul'} for a single-type corpus. The accounting identity
        # is the real constraint.
        if self.by_type:
            if self.overall is None:
                raise ValueError("per-type metrics require the overall metric block")
            if sum(block.items for block in self.by_type.values()) != self.overall.items:
                raise ValueError("per-type items do not account for the overall item count")
        if self.overall is None and not self.records:
            raise ValueError("a report must carry metrics, records, or both")
        if any(blank(reason) for reason in self.skipped):
            raise ValueError("skip reasons must be nonempty")
        if self.provenance.source not in self.coverage.by_source:
            raise ValueError("report source is not accounted for in coverage")
        if self.records:
            if self.coverage.aggregate_reason is not None:
                raise ValueError(
                    "a report carrying records cannot declare aggregate-only coverage")
            self.coverage.require_carried_records(self.records, self.keys)
        elif blank(self.coverage.aggregate_reason or ""):
            raise ValueError(
                "a report carrying no records must declare why its coverage is "
                "aggregate-only (coverage.aggregate_reason)")
        return self


# ── declared destination ────────────────────────────────────────────────────
def write_report(key: str, fields: dict, document: TraceabilityReport) -> Path:
    """Write ``document`` to the layout ``key`` resolves and stamp the write.

    The destination is never spelled here: ``core.common.artifact`` renders the
    declared ``traceability_report`` template (owners fail loud on drift) and
    ``core.common.trace_artifact`` records the write in the artifacts trace.
    """
    from core.common import artifact, trace_artifact

    path = artifact(key, fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    trace_artifact(key, path, producer="core.eval_trace")
    return path


__all__ = [
    "AbstentionBlock", "AttributeAttribution", "AttributeAttributionRow",
    "Brier", "BucketKey", "CORE_RECORD_DIMENSIONS", "Count",
    "DIMENSION_ALIASES", "Dimension", "DimensionPolicy",
    "EvalProvenance", "EvalRowKey", "EvalSource", "MetricBlock",
    "MinConfidenceKey", "N_QTYPES", "PersistedCount", "QTypeName",
    "RecordDimension", "Share",
    "SliceMetrics", "SliceMetricsRow", "TaggedRecord", "Temperature",
    "TraceabilityCoverage", "TraceabilityReport", "UnmeasuredSliceRow",
    "canonical_dimension", "canonical_dimension_map", "decision_flip",
    "derived_dimension_policies", "derived_record_census", "record_dimension_tags",
    "row_from_csv", "write_report",
]
