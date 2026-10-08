"""Validated completeness contracts for measured coverage and augmentation.

Producers supply their existing registry explicitly. No second attribute list
is maintained here; missing evidence is recorded, never inferred.

This module is the project's coverage SSOT. Alongside the per-artifact contracts
it owns :class:`ReportCoverageContract`, the GENERAL per-record / per-dimension
completeness contract: records are mandatory and authoritative (the population
is ``len(records)``, never a declared second number) and each traceability
dimension is accounted under an explicit multiplicity policy plus an explicit
unknown policy.

WHICH CONTRACT DOES A NEW PRODUCER USE? (the single decision rule)

1. A new producer with a per-record population and per-dimension completeness
   claims uses :class:`ReportCoverageContract`. Carry one record per record,
   declare your own dimension key-set, and account EVERY dimension under an
   explicit multiplicity policy and an explicit unknown policy
   (``not_applicable`` requires the whole population under ``unknown`` plus a
   reason). Do not add a new per-artifact contract for a new producer.
2. A producer whose emitted artifact shape is already FROZEN keeps that
   artifact's own contract (``CohortCoverage``, ``AttributeSeparationCoverage``).
   Those are historical instances of the same idiom, pinned by bytes already
   shipped -- not a second home for new work.
3. A laya-lane producer uses ``core.eval_trace.TraceabilityCoverage``: the laya
   traceability artifact's own contract (fixed ``Dimension`` literal, checked
   against persisted reports). It imports this module's ``DimensionPolicy`` /
   ``UNKNOWN_DIMENSION_VALUE`` vocabulary so those words have ONE home, but it
   does NOT consume ``ReportCoverageContract``.

Consumers of :class:`ReportCoverageContract` today:
``training.attribute_separation`` (the track pair/attribute census) and
``model_tracks.ablation_cohort`` (the ablation cohort's dimensions).
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Count = Annotated[int, Field(ge=0, strict=True)]
Share = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

# The ONE difficulty vocabulary, shared by the ablation cohort strata
# (``CohortCoverage.by_difficulty``) and by per-record traceability tags
# (``core.eval_trace.TaggedRecord.difficulty``). ``unknown`` is a DECLARED value
# that carries an explicit reason, never an omission; it is NOT
# ``UNKNOWN_DIMENSION_VALUE`` below, which answers the different question "no
# value was measured for this dimension at all".
Difficulty = Literal['easy', 'medium', 'hard', 'unknown']

# Unicode categories that carry no visible glyph: separators, controls and
# format characters (zero-width space U+200B, BOM U+FEFF, soft hyphen U+00AD,
# word joiner U+2060) plus lone combining marks (U+034F).
_INVISIBLE_CATEGORIES = frozenset({'Cc', 'Cf', 'Mn', 'Me', 'Zl', 'Zp', 'Zs'})


def blank(value: str) -> bool:
    """True when ``value`` carries no VISIBLE character.

    ``str.strip()`` alone is not blankness: a tag or policy built only from
    zero-width/format characters or lone combining marks is invisible yet not
    whitespace, so a ``strip()`` check would accept it as a real value and let
    an untagged record (or an unanswered unknown policy) through. A value that
    merely CONTAINS such a character beside a visible one is untouched.
    """
    return all(unicodedata.category(character) in _INVISIBLE_CATEGORIES
               for character in value)


class CoverageModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


def require_keys(actual, expected, name):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f'{name} coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}')


class UnclassifiedCoverage(CoverageModel):
    rows_populated: Count
    distinct_value_sets: Count


class AttributeCensusEntry(UnclassifiedCoverage):
    distinct_raw_strings: Count
    same_gtin_pairs_both_populated: Count
    conflict_pairs: Count
    conflict_rate: Share
    top_sets: list[tuple[list[str], Annotated[int, Field(ge=1, strict=True)]]]

    @model_validator(mode='after')
    def accounting(self):
        if self.conflict_pairs > self.same_gtin_pairs_both_populated:
            raise ValueError('conflict count exceeds observed pairs')
        expected = round(self.conflict_pairs / self.same_gtin_pairs_both_populated, 4) if self.same_gtin_pairs_both_populated else 0
        if abs(self.conflict_rate - expected) > 1e-8:
            raise ValueError('conflict rate disagrees with counts')
        if self.distinct_value_sets > self.rows_populated or sum(n for _, n in self.top_sets) > self.rows_populated:
            raise ValueError('value-set support exceeds populated rows')
        return self


class AttributeCensus(CoverageModel):
    expected_attributes: set[str] = Field(exclude=True)
    rows: Count
    valid_gtin_rows: Count
    same_gtin_pairs_total: Count
    keys: dict[str, AttributeCensusEntry]
    unclassified_keys: dict[str, UnclassifiedCoverage]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.keys, self.expected_attributes, 'attribute census')
        if self.valid_gtin_rows > self.rows or set(self.keys) & set(self.unclassified_keys):
            raise ValueError('invalid census population accounting')
        for entry in [*self.keys.values(), *self.unclassified_keys.values()]:
            if entry.rows_populated > self.rows:
                raise ValueError('attribute support exceeds catalog rows')
        if any(e.same_gtin_pairs_both_populated > self.same_gtin_pairs_total for e in self.keys.values()):
            raise ValueError('attribute pair support exceeds total pairs')
        return self


class AttributeGenerationBudgetEntry(CoverageModel):
    kind: Literal['SET_NUMERIC', 'SET_CATEGORICAL', 'SET_ENUM', 'CONSTANT', 'NUMERIC_BAND', 'unclassified']
    rows_populated: Count
    conflict_rate: Share
    distsets: Count
    headroom_share: Share
    expected_pair_coverage: Count
    expected_mint: Count
    masking_donor: bool
    veto_candidate: bool
    eval_slice: bool
    eval_supported_sets: Count
    structured_channel: Literal['numeric', 'text', 'none', 'CONSTANT']
    mint_at_target: Count | None = None

    @model_validator(mode='after')
    def eligible(self):
        if not self.masking_donor and (self.expected_mint or self.mint_at_target):
            raise ValueError('ineligible attribute assigned mint supply')
        if self.eval_slice != (self.eval_supported_sets > 0):
            raise ValueError('evaluation eligibility disagrees with support')
        return self


class AttributeGenerationBudget(CoverageModel):
    expected_attributes: set[str] = Field(exclude=True)
    entries: dict[str, AttributeGenerationBudgetEntry]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.entries, self.expected_attributes, 'attribute generation budget')
        return self


class BatchPopulationCoverage(CoverageModel):
    batch_size: Annotated[int, Field(ge=1, strict=True)]
    composition: dict[str, Annotated[int, Field(ge=1, strict=True)]] = Field(min_length=1)
    observed_counts: dict[str, Annotated[int, Field(ge=1, strict=True)]] = Field(min_length=1)
    dataset_rows: Count

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.composition, self.observed_counts, 'batch population')
        if sum(self.composition.values()) != self.batch_size:
            raise ValueError('batch size differs from composition total')
        if sum(self.observed_counts.values()) != self.dataset_rows:
            raise ValueError('population support does not account for every dataset row')
        return self


class NegativeSupplyCoverage(CoverageModel):
    anchors_total: Count
    anchors_with_real_partner: Count
    coverage_share: Share

    @model_validator(mode='after')
    def complete(self):
        if self.anchors_with_real_partner > self.anchors_total:
            raise ValueError('covered anchors exceed anchor population')
        expected = round(self.anchors_with_real_partner / max(self.anchors_total, 1), 4)
        if abs(self.coverage_share - expected) > 1e-8:
            raise ValueError('anchor coverage share disagrees with counts')
        return self


class CohortCoverage(CoverageModel):
    cohort_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    pair_rows: Count
    minted_endpoints_total: Count
    minted_endpoints_covered: Count
    by_scope: dict[str, Count]
    by_population: dict[str, Count]
    by_difficulty: dict[Difficulty, Count]
    unknown_difficulty_policy: str = Field(min_length=1)

    @model_validator(mode='after')
    def complete(self):
        if self.minted_endpoints_total != self.minted_endpoints_covered:
            raise ValueError('ablation cohort dropped minted endpoints')
        require_keys(self.by_difficulty, set(Difficulty.__args__), 'difficulty')
        for counts in (self.by_scope, self.by_population, self.by_difficulty):
            if sum(counts.values()) != self.pair_rows:
                raise ValueError('cohort strata do not account for every pair')
        return self


class VendorVariationRow(CoverageModel):
    gtin: str
    brand: str
    volume_within_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    pack_within_diversity: Count
    n_listings: Annotated[int, Field(ge=1, strict=True)]


class ProductVariationRow(CoverageModel):
    brand: str
    similarity: Annotated[float, Field(allow_inf_nan=False)]
    volume_delta: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    pack_conflict: bool
    flavor_conflict: bool


class SamplingArtifact(CoverageModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class SamplingInput(SamplingArtifact):
    rows: Count
    source_schema: dict[str, str | None] | None = Field(default=None, alias='schema')


class PairTypeDefinition(CoverageModel):
    name: str
    families: dict[str, str]
    positive_assignment: str
    negative_assignment: str


class CandidateGeneration(CoverageModel):
    positive: str
    negative: str
    excluded: str
    threshold_source: str


class DistributionContract(CoverageModel):
    sku: str
    attribute: str
    time: str
    selection: str


class ThresholdSweepRow(CoverageModel):
    threshold: Annotated[float, Field(allow_inf_nan=False)]
    tp: Count
    fp: Count
    fn: Count
    tn: Count
    precision: Share
    recall: Share
    fpr: Share


class SamplingAccounting(CoverageModel):
    eligible_positive: Count
    eligible_typed_negative: Count
    excluded_fallback: Count
    excluded_below_positive_threshold: Count
    excluded_below_negative_threshold: Count
    excluded_other_gate_decision: Count
    unmatched_negative_reasons: dict[str, Count]
    balanced_rows: Count
    sample_rows: Count
    balanced_by_type_and_label: dict[str, dict[Literal['0', '1'], Count]]
    sample_by_type_and_label: dict[str, dict[Literal['0', '1'], Count]]
    threshold_sweep: list[ThresholdSweepRow]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.sample_by_type_and_label, self.balanced_by_type_and_label, 'sample pair families')
        for groups, total in ((self.balanced_by_type_and_label, self.balanced_rows),
                              (self.sample_by_type_and_label, self.sample_rows)):
            if sum(sum(counts.values()) for counts in groups.values()) != total:
                raise ValueError('pair-family counts do not account for all rows')
            for counts in groups.values():
                require_keys(counts, {'0', '1'}, 'pair labels')
                if counts['0'] != counts['1']:
                    raise ValueError('pair family is not label balanced')
        if self.sample_rows > self.balanced_rows:
            raise ValueError('sample exceeds balanced population')
        for row in self.threshold_sweep:
            if row.tp + row.fp + row.fn + row.tn != self.sample_rows:
                raise ValueError('threshold sweep does not account for every sample row')
        for family, counts in self.sample_by_type_and_label.items():
            if any(counts[label] > self.balanced_by_type_and_label[family][label] for label in ('0', '1')):
                raise ValueError('sample stratum exceeds its source population')
        return self


class BalancedSamplingManifest(CoverageModel):
    schema_version: Literal[1]
    seed: int
    requested_sample_size: Count
    pair_type_definition: PairTypeDefinition
    candidate_generation: CandidateGeneration
    distribution_contract: DistributionContract
    inputs: dict[Literal['gate_results', 'sku_data'], SamplingInput]
    accounting: SamplingAccounting
    outputs: dict[Literal['balanced', 'sample', 'threshold_sweep'], SamplingArtifact]

    @model_validator(mode='after')
    def complete(self):
        require_keys(self.inputs, {'gate_results', 'sku_data'}, 'sampling inputs')
        require_keys(self.outputs, {'balanced', 'sample', 'threshold_sweep'}, 'sampling outputs')
        if self.requested_sample_size != self.accounting.sample_rows:
            raise ValueError('sample size differs from requested size')
        return self


# ── project-wide per-record / per-dimension coverage ────────────────────────
# THE RULE FOR A NEW PRODUCER IS IN THE MODULE DOCSTRING ("which contract does a
# new producer use?"); this block only records why the older contracts stay.
#
# * ``CohortCoverage`` — the ablation cohort's three FIXED strata
#   (scope/population/difficulty) over a producer-supplied ``pair_rows``. It is
#   one hard-coded instance of what ``ReportCoverageContract`` generalizes; it is
#   deliberately NOT refactored, because its emitted artifact shape is frozen.
# * ``AttributeSeparationCoverage`` — the attribute axis as a ``partition``
#   (n_positive + n_negative + n_unobservable == pair_rows) plus per-value
#   support rows. The same completeness idiom over ONE dimension, at a different
#   grain; left as-is for the same reason.
# * ``TraceabilityCoverage`` (``core.eval_trace``) — laya-first and additive: a
#   fixed ``Dimension`` literal, an OPT-IN carried record list, and a
#   producer-supplied ``records_total`` checked against ``Counter(records)`` only
#   when records happen to be present. It stays the laya vocabulary contract
#   (laya does NOT consume ``ReportCoverageContract``); it can delegate to this
#   class in a later additive step, never by having its frozen artifact shape
#   edited.
#
# ``ReportCoverageContract`` is the producer-neutral generalization: records are
# MANDATORY and authoritative (the population is ``len(records)``, never a second
# declared number flagged against them), and the dimension key-set is the
# producer's own explicit declaration — no second registry is maintained here,
# mirroring ``CoverageModel``'s rule.
DimensionPolicy = Literal['partition', 'overlap', 'not_applicable']
UNKNOWN_DIMENSION_VALUE = 'unknown'


class TaggedDimensionRecord(CoverageModel):
    """A record that CARRIES its own dimension tags.

    ``dimensions`` maps each declared dimension to the value(s) the record
    belongs to. A dimension the record does not carry is MISSING, not empty: the
    parent contract requires the key-set to equal its declared dimensions, which
    is what makes "every record tagged" a property of the CARRIED data rather
    than of two producer-supplied numbers agreeing.

    A tuple (not a scalar) is the value shape because one record legitimately
    belongs to several values of an overlap dimension (e.g. several slices),
    while a partition dimension is enforced by the parent's sum check.
    """
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    record_id: str = Field(min_length=1)
    dimensions: dict[str, tuple[str, ...]] = Field(min_length=1)

    @model_validator(mode='after')
    def tagged(self):
        for name, values in self.dimensions.items():
            if blank(name):
                raise ValueError('dimension names must be nonempty')
            if not values:
                raise ValueError(f'record {self.record_id!r} carries no tag for {name!r}')
            if any(blank(value) for value in values):
                raise ValueError(f'record {self.record_id!r} carries a blank tag for {name!r}')
        return self


class DimensionAccounting(CoverageModel):
    """One dimension's declared census, multiplicity policy and unknown policy.

    ``counts`` is the producer's own census (value -> records carrying it). The
    parent checks its arithmetic against the multiplicity policy and then
    re-derives it from the carried records, so a census no record supports cannot
    pass: ``partition`` must sum to the record population, ``overlap`` must be at
    least it, and ``not_applicable`` must report the whole population under
    ``unknown`` with no measured value and an explicit reason.

    ``unknown_policy`` is mandatory and must be explicit for EVERY dimension,
    whatever its multiplicity: "how records whose value is unknown are treated"
    is never answered by omission.
    """
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    policy: DimensionPolicy
    counts: dict[str, Count]
    unknown_policy: str
    reason: str | None = None

    @model_validator(mode='after')
    def declared(self):
        if blank(self.unknown_policy):
            raise ValueError('unknown policy must be explicit')
        if any(blank(value) for value in self.counts):
            raise ValueError('dimension value names must be nonempty')
        if self.policy == 'not_applicable' and blank(self.reason or ''):
            raise ValueError('not_applicable coverage requires an explicit reason')
        return self


def derived_dimension_counts(
    records: Iterable[TaggedDimensionRecord],
    dimensions: Iterable[str],
) -> dict[str, dict[str, int]]:
    """Count each declared dimension's tags over the CARRIED records.

    The ONE derivation, shared by the contract's validator and by consumers that
    want the audited census, so a consumer can never recompute it differently.
    """
    names = list(dimensions)
    records = list(records)
    counts: dict[str, dict[str, int]] = {}
    for name in names:
        tally: Counter[str] = Counter()
        for record in records:
            tally.update(record.dimensions.get(name, ()))
        counts[name] = dict(tally)
    return counts


class ReportCoverageContract(CoverageModel):
    """Every carried record tagged; every declared dimension fully accounted.

    The GENERAL, producer-neutral coverage contract (see the module's decision
    rule for which producer uses it). Two assertions, both derived from data:

    1. Every record is ACCOUNTED/TAGGED. ``records`` is mandatory and the record
       population is ``len(records)``; each record must carry exactly the
       producer's declared dimension key-set, each dimension with a non-blank
       value. A missing dimension, an empty tag or a blank tag fails loud, so a
       report cannot claim full tagging while carrying untagged records.
    2. Every dimension is FULLY ACCOUNTED under an explicit multiplicity policy:
       ``partition`` (declared counts sum to the population), ``overlap`` (they
       are at least the population) or ``not_applicable`` (the whole population
       is ``unknown``, with a mandatory reason). The declared census is then
       required to EQUAL the census re-derived from the carried records, so the
       accounting rests on carried evidence, not on two numbers agreeing.
    """
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    records: tuple[TaggedDimensionRecord, ...] = Field(min_length=1)
    dimensions: dict[str, DimensionAccounting] = Field(min_length=1)

    @property
    def records_total(self) -> int:
        """The population, DERIVED from the carried records (never declared)."""
        return len(self.records)

    def derived_counts(self) -> dict[str, dict[str, int]]:
        """The audited census, recomputed from the carried records."""
        return derived_dimension_counts(self.records, self.dimensions)

    @model_validator(mode='after')
    def complete(self):
        if any(blank(name) for name in self.dimensions):
            raise ValueError('dimension names must be nonempty')
        expected = set(self.dimensions)
        # 1. Every record is TAGGED: it carries exactly the declared dimensions,
        #    each with at least one non-blank value (see TaggedDimensionRecord).
        for record in self.records:
            require_keys(record.dimensions, expected, f'record {record.record_id!r} tagging')
        # 2. Every dimension is ACCOUNTED under its declared multiplicity policy.
        carried_by_dimension = self.derived_counts()
        for name, accounting in self.dimensions.items():
            declared, policy = accounting.counts, accounting.policy
            total = sum(declared.values())
            if policy == 'partition' and total != self.records_total:
                raise ValueError(
                    f'{name}: partition accounting does not account for every record'
                )
            if policy == 'overlap' and total < self.records_total:
                raise ValueError(
                    f'{name}: overlap accounting under-accounts the record population'
                )
            if policy == 'not_applicable' and (
                declared.get(UNKNOWN_DIMENSION_VALUE, 0) != self.records_total
                or any(value and key != UNKNOWN_DIMENSION_VALUE
                       for key, value in declared.items())
            ):
                raise ValueError(f'{name}: not_applicable but carries measured values')
            # 3. The declared census must EQUAL what the carried records tag.
            carried = carried_by_dimension[name]
            require_keys(declared, carried, f'{name} declared counts')
            if dict(declared) != carried:
                raise ValueError(f'{name}: declared counts disagree with the carried records')
        return self
