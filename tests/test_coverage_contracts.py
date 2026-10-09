"""Coverage omissions must fail at producer boundaries."""
import pandas as pd
import pytest
from pydantic import ValidationError

from core.attribute_universe import AttributeUniverse
from core.coverage_contracts import (
    AttributeCensus,
    BatchPopulationCoverage,
    CohortCoverage,
    Difficulty,
    DimensionAccounting,
    DimensionPolicy,
    ReportCoverageContract,
    TaggedDimensionRecord,
    UNKNOWN_DIMENSION_VALUE,
)


def test_census_and_budget_reject_missing_registry_attribute():
    universe = AttributeUniverse(pd.DataFrame({'attribute': [''], 'gtin': ['']}))
    census = universe.census()
    missing = next(iter(universe.registry))
    del census['keys'][missing]
    with pytest.raises(ValidationError, match='coverage mismatch'):
        AttributeCensus.model_validate({**census, 'expected_attributes': set(universe.registry)})
    with pytest.raises(ValidationError, match='coverage mismatch'):
        universe.datagen_budget(census=census, min_value_support=20)


def test_batch_contract_rejects_unrepresented_population():
    with pytest.raises(ValidationError, match='coverage mismatch'):
        BatchPopulationCoverage(batch_size=2, composition={'original': 2},
            observed_counts={'original': 3, 'minted': 2}, dataset_rows=5)


def test_cohort_requires_unknown_difficulty_and_complete_mint_lineage():
    fields = dict(cohort_key='clean:bundle:2', pair_rows=2,
        minted_endpoints_total=1, minted_endpoints_covered=1,
        by_scope={'training_diagnostic': 2}, by_population={'real': 2},
        by_difficulty={'easy': 1, 'medium': 0, 'hard': 0, 'unknown': 1},
        unknown_difficulty_policy='retain unknown')
    CohortCoverage(**fields)
    with pytest.raises(ValidationError, match='difficulty.*coverage mismatch'):
        CohortCoverage(**{**fields, 'by_difficulty': {'easy': 2, 'hard': 0}})
    with pytest.raises(ValidationError, match='dropped minted'):
        CohortCoverage(**{**fields, 'minted_endpoints_covered': 0})


# ── ReportCoverageContract: the general, project-wide per-dimension contract ─
def _record(record_id: str, **dimensions) -> TaggedDimensionRecord:
    """Build a tagged record; a scalar value is the single-value case."""
    return TaggedDimensionRecord(
        record_id=record_id,
        dimensions={
            name: tuple(values) if isinstance(values, (list, tuple)) else (values,)
            for name, values in dimensions.items()
        },
    )


def _tracks_case() -> ReportCoverageContract:
    """A realistic tracks case: split/population/difficulty/slice/attribute."""
    records = (
        _record('pair-1', split='dev', population='real', difficulty='easy',
                slice=['unseen', 'sparse_neighborhood'], attribute='brand'),
        _record('pair-2', split='dev', population='minted', difficulty='unknown',
                slice=['isolated'], attribute='unknown'),
        _record('pair-3', split='test', population='real', difficulty='hard',
                slice=['unseen'], attribute='volume'),
        _record('pair-4', split='test', population='real', difficulty='medium',
                slice=['missing_field', 'isolated'], attribute='unknown'),
    )
    dimensions = {
        'split': DimensionAccounting(
            policy='partition', counts={'dev': 2, 'test': 2},
            unknown_policy='a scored pair belongs to exactly one split'),
        'population': DimensionAccounting(
            policy='partition', counts={'real': 3, 'minted': 1},
            unknown_policy='minted rows are supplied data, never unknown'),
        'difficulty': DimensionAccounting(
            policy='partition',
            counts={'easy': 1, 'medium': 1, 'hard': 1, 'unknown': 1},
            unknown_policy='unknown difficulty is retained and reported'),
        'slice': DimensionAccounting(
            policy='overlap',
            counts={'unseen': 2, 'sparse_neighborhood': 1, 'isolated': 2,
                    'missing_field': 1},
            unknown_policy='slices are memberships; a pair may carry several'),
        'attribute': DimensionAccounting(
            policy='partition', counts={'brand': 1, 'volume': 1, 'unknown': 2},
            unknown_policy='no-attribute pairs are reported as unknown'),
    }
    return ReportCoverageContract(records=records, dimensions=dimensions)


def _laya_case() -> ReportCoverageContract:
    """A realistic laya case: question_type/eval_half/abstention."""
    records = (
        _record('q-1', question_type='choice', eval_half='A', abstention='passed'),
        _record('q-2', question_type='score', eval_half='B', abstention='abstained'),
        _record('q-3', question_type='noul', eval_half='B', abstention='unevaluated'),
    )
    dimensions = {
        'question_type': DimensionAccounting(
            policy='partition', counts={'choice': 1, 'score': 1, 'noul': 1},
            unknown_policy='every scored question declares one type'),
        'eval_half': DimensionAccounting(
            policy='partition', counts={'A': 1, 'B': 2},
            unknown_policy='a question belongs to exactly one half'),
        'abstention': DimensionAccounting(
            policy='partition',
            counts={'passed': 1, 'abstained': 1, 'unevaluated': 1},
            unknown_policy='the gate has three exhaustive states'),
    }
    return ReportCoverageContract(records=records, dimensions=dimensions)


def test_tracks_coverage_case_is_accepted_and_census_is_derived():
    contract = _tracks_case()
    assert contract.records_total == 4  # derived from the carried records
    assert contract.derived_counts()['slice'] == {
        'unseen': 2, 'sparse_neighborhood': 1, 'isolated': 2, 'missing_field': 1,
    }


def test_laya_coverage_case_is_accepted_by_the_same_contract():
    contract = _laya_case()
    assert contract.records_total == 3
    assert contract.derived_counts()['abstention'] == {
        'passed': 1, 'abstained': 1, 'unevaluated': 1,
    }


def test_record_population_is_never_a_declared_second_number():
    case = _tracks_case()
    with pytest.raises(ValidationError):
        ReportCoverageContract(
            records=case.records, dimensions=case.dimensions, records_total=999)
    with pytest.raises(ValidationError):  # zero carried records cannot claim coverage
        ReportCoverageContract(records=(), dimensions=case.dimensions)


def test_an_untagged_record_is_rejected():
    case = _tracks_case()
    untagged = TaggedDimensionRecord(
        record_id='pair-1',
        dimensions={k: v for k, v in case.records[0].dimensions.items() if k != 'difficulty'},
    )
    with pytest.raises(ValidationError, match='tagging coverage mismatch'):
        ReportCoverageContract(records=(untagged, *case.records[1:]), dimensions=case.dimensions)
    with pytest.raises(ValidationError, match='carries no tag'):
        TaggedDimensionRecord(record_id='pair-1', dimensions={'split': ()})


def test_a_partition_that_does_not_sum_is_rejected():
    records = (
        _record('p1', split=['dev', 'test'], population='real'),
        _record('p2', split='dev', population='real'),
    )
    dimensions = {
        'split': DimensionAccounting(
            policy='partition', counts={'dev': 2, 'test': 1},
            unknown_policy='one split per pair'),
        'population': DimensionAccounting(
            policy='partition', counts={'real': 2}, unknown_policy='one population per pair'),
    }
    with pytest.raises(ValidationError, match='does not account for every record'):
        ReportCoverageContract(records=records, dimensions=dimensions)


def test_an_overlap_that_under_accounts_is_rejected():
    records = (_record('p1', slice='unseen'), _record('p2', slice='unseen'))
    dimensions = {'slice': DimensionAccounting(
        policy='overlap', counts={'unseen': 1}, unknown_policy='slices are memberships')}
    with pytest.raises(ValidationError, match='under-accounts the record population'):
        ReportCoverageContract(records=records, dimensions=dimensions)


def test_not_applicable_requires_unknown_only_and_a_reason():
    records = (_record('p1', slice='unknown'), _record('p2', slice='unknown'))
    with pytest.raises(ValidationError, match='carries measured values'):
        ReportCoverageContract(
            records=records,
            dimensions={'slice': DimensionAccounting(
                policy='not_applicable', counts={'isolated': 2},
                unknown_policy='no slices here', reason='aggregate-only report')})
    with pytest.raises(ValidationError, match='requires an explicit reason'):
        DimensionAccounting(
            policy='not_applicable', counts={'unknown': 2}, unknown_policy='no slices here')
    accepted = ReportCoverageContract(
        records=records,
        dimensions={'slice': DimensionAccounting(
            policy='not_applicable', counts={'unknown': 2},
            unknown_policy='no slices here', reason='aggregate-only report')})
    assert accepted.records_total == 2


def test_the_unknown_policy_is_mandatory_and_never_blank():
    with pytest.raises(ValidationError, match='must be explicit'):
        DimensionAccounting(policy='partition', counts={'dev': 2}, unknown_policy='   ')
    with pytest.raises(ValidationError):  # missing entirely
        DimensionAccounting(policy='partition', counts={'dev': 2})


def test_declared_counts_must_equal_the_carried_records():
    records = (_record('p1', split='dev'), _record('p2', split='dev'))
    with pytest.raises(ValidationError, match='declared counts coverage mismatch'):
        ReportCoverageContract(
            records=records,
            dimensions={'split': DimensionAccounting(
                policy='partition', counts={'dev': 1, 'test': 1},
                unknown_policy='one split per pair')})


def test_an_invisible_only_tag_does_not_count_as_tagged():
    """Falsified 2026-10-08: ``str.strip()`` is not blankness. U+200B is invisible
    but NOT whitespace, so a record could satisfy "every record carries a
    non-blank tag" while tagging nothing. The public contract must reject it and
    accept the same record with a real tag."""
    dims = {'split': DimensionAccounting(
        policy='partition', counts={'dev': 1}, unknown_policy='one split per record')}
    with pytest.raises(ValidationError, match='carries a blank tag'):
        ReportCoverageContract(
            records=(TaggedDimensionRecord(
                record_id='r1', dimensions={'split': ('\u200b',)}),),
            dimensions=dims)
    accepted = ReportCoverageContract(
        records=(TaggedDimensionRecord(
            record_id='r1', dimensions={'split': ('dev',)}),),
        dimensions=dims)
    assert accepted.records_total == 1


def test_the_policy_vocabulary_and_unknown_key_have_exactly_one_home():
    """De-duplicated 2026-10-08: ``DimensionPolicy`` and the ``'unknown'``
    dimension value were declared in BOTH this module and ``core.eval_trace``.
    They now live here and laya imports them, so the two contracts must read the
    SAME object -- and laya's ``not_applicable`` branch must still key on it."""
    import core.eval_trace as eval_trace

    assert eval_trace.DimensionPolicy is DimensionPolicy
    assert eval_trace.UNKNOWN_DIMENSION_VALUE is UNKNOWN_DIMENSION_VALUE
    assert eval_trace.Difficulty is Difficulty

    accepted = eval_trace.TraceabilityCoverage(
        records_total=2, items_total=2, by_source={'laya_cli_eval': 2},
        dimension_values={'split': {'unknown'}},
        dimension_multiplicity={'split': 'not_applicable'},
        by_dimension={'split': {UNKNOWN_DIMENSION_VALUE: 2}},
        unknown_policy={'split': 'no split was measured for these records'},
        slice_coverage='not_applicable', slice_coverage_reason='aggregate-only')
    assert accepted.by_dimension['split'] == {UNKNOWN_DIMENSION_VALUE: 2}
    with pytest.raises(ValidationError, match='not_applicable but carries measured values'):
        eval_trace.TraceabilityCoverage(
            records_total=2, items_total=2, by_source={'laya_cli_eval': 2},
            dimension_values={'split': {'train'}},
            dimension_multiplicity={'split': 'not_applicable'},
            by_dimension={'split': {'train': 2}},
            unknown_policy={'split': 'no split was measured'},
            slice_coverage='not_applicable', slice_coverage_reason='aggregate-only')
