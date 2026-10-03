"""Dataset parsing and dedupe policy share the validated YAML contract."""

import pytest
from pydantic import ValidationError

from core.common import data_cfg
from core.schemas import DataConfig


def test_descriptor_partition_follows_config():
    from core.sku_identity import DESCRIPTOR_COLUMNS, NON_DESCRIPTOR_COLUMNS

    cfg = data_cfg()
    assert DESCRIPTOR_COLUMNS == tuple(cfg.descriptor_columns)
    assert set(DESCRIPTOR_COLUMNS).isdisjoint(NON_DESCRIPTOR_COLUMNS)
    assert set(DESCRIPTOR_COLUMNS) | NON_DESCRIPTOR_COLUMNS == set(cfg.column_mapping.values())


@pytest.mark.parametrize('descriptors', [['sku_name_eng', 'sku_name_eng'], ['not_a_column']])
def test_descriptor_policy_rejects_duplicate_or_unknown_columns(descriptors):
    raw = data_cfg().model_dump()
    raw['descriptor_columns'] = descriptors
    with pytest.raises(ValidationError, match='descriptor_columns'):
        DataConfig.model_validate(raw)


def test_adjudication_policy_rejects_overlapping_verdicts():
    raw = data_cfg().model_dump()
    duplicate = {**raw['dedupe_adjudications'][0], 'decision': 'keep'}
    raw['dedupe_adjudications'].append(duplicate)
    with pytest.raises(ValidationError, match='duplicate retailer/gtin'):
        DataConfig.model_validate(raw)


def test_dataset_policy_rejects_numeric_coercion():
    raw = data_cfg().model_dump()
    raw['dataset_csv_read']['dtype'] = 'float64'
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)
