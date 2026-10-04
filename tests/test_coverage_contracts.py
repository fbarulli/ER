"""Coverage omissions must fail at producer boundaries."""
import pandas as pd
import pytest
from pydantic import ValidationError

from core.attribute_universe import AttributeUniverse
from core.coverage_contracts import AttributeCensus, BatchPopulationCoverage, CohortCoverage


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
    fields = dict(cohort_sha256='0' * 64, pair_rows=2,
        minted_endpoints_total=1, minted_endpoints_covered=1,
        by_scope={'training_diagnostic': 2}, by_population={'real': 2},
        by_difficulty={'easy': 1, 'hard': 0, 'unknown': 1},
        unknown_difficulty_policy='retain unknown')
    CohortCoverage(**fields)
    with pytest.raises(ValidationError, match='difficulty.*coverage mismatch'):
        CohortCoverage(**{**fields, 'by_difficulty': {'easy': 2, 'hard': 0}})
    with pytest.raises(ValidationError, match='dropped minted'):
        CohortCoverage(**{**fields, 'minted_endpoints_covered': 0})
