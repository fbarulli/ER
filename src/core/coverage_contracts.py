"""Validated completeness contracts for measured coverage and augmentation.

Producers supply their existing registry explicitly. No second attribute list
is maintained here; missing evidence is recorded, never inferred.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Count = Annotated[int, Field(ge=0, strict=True)]
Share = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


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
    by_difficulty: dict[Literal['easy', 'hard', 'unknown'], Count]
    unknown_difficulty_policy: str = Field(min_length=1)

    @model_validator(mode='after')
    def complete(self):
        if self.minted_endpoints_total != self.minted_endpoints_covered:
            raise ValueError('ablation cohort dropped minted endpoints')
        require_keys(self.by_difficulty, {'easy', 'hard', 'unknown'}, 'difficulty')
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
