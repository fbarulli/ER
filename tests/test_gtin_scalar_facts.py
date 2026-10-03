import pandas as pd
import pytest
from core.gtin import normalize_and_validate_gtin, normalize_gtin_value, gtin_validity
from core.sku_identity import _gtin_facts


def test_scalar_facts_match_vector_contract_and_real_catalog():
    from core.common import TRAIN_ROOT
    cases = [None, pd.NA, float('nan'), '', '00000000', 'garbage',
             '4006381333931', '4006381333932', '1-735143004010',
             '6 pack 4006381333931', '12345678 87654321', '123456789012',
             '00012345600012', '1234567', '123456789012345']
    catalog = TRAIN_ROOT/'data/track_setup/eligible_catalog.csv'
    if catalog.is_file():
        cases += pd.read_csv(catalog,dtype=str,keep_default_na=False,usecols=['gtin']).gtin.iloc[::71].tolist()
    expected = normalize_and_validate_gtin(pd.Series(cases))
    valid = gtin_validity(pd.Series(cases))
    for n, value in enumerate(cases):
        key, structural = normalize_gtin_value(value)
        expected_key = expected.gtin_clean.iloc[n]
        assert key == (None if pd.isna(expected_key) else expected_key)
        assert structural == bool(expected.gtin_structurally_valid.iloc[n])
        assert _gtin_facts(value) == (bool(valid.iloc[n]), key if valid.iloc[n] else '')


def test_scalar_identity_reads_current_policy_without_cache(monkeypatch):
    from core import identity_policy
    from core.identity_policy import Hold, ListingHold
    gtin = '4006381333931'
    policy = identity_policy.review_policy().model_copy(deep=True)
    policy.quarantined_gtins.pop(gtin, None)
    monkeypatch.setattr(identity_policy,'review_policy',lambda:policy)
    assert _gtin_facts(gtin) == (True,gtin)
    assert identity_policy.review_reason(gtin) == ''
    policy.quarantined_gtins[gtin] = Hold(reason='updated hold',evidence_sku_ids=['row'])
    assert _gtin_facts(gtin) == (False,'')
    assert identity_policy.review_reason(gtin) == 'updated hold'
    policy.quarantined_listings['row'] = ListingHold(reason='listing hold',evidence_sku_ids=['row'],expected_gtin=gtin)
    assert identity_policy.listing_review_reason('row',gtin) == 'listing hold'
    policy.quarantined_listings['row'].expected_gtin = '735143004010'
    assert identity_policy.listing_review_reason('row',gtin) == ''
