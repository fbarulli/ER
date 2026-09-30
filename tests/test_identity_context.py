"""Audit cues must stay bounded and preserve uncertainty."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('identity_context_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_identity_context.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(title='', description='', attributes='', gtin='868784000346', brand='3D'):
    return {'sku_id': '1', 'gtin': gtin, 'brand': brand, 'retailer': 'Shop',
            'sku_name_eng': title, 'description_short_eng': description, 'attribute': attributes}


def test_numeric_claim_requires_caffeine_and_retains_basis_uncertainty():
    frame = pd.DataFrame([row(description='200 mg sodium. Caffeine free.', attributes='Caffeine: 0-15 mg'),
                          row(description='200 mg caffeine per 16 oz can.', attributes='Caffeine: 25-50 mg'),
                          row(description='Contains 100mg caffeine.', attributes='Caffeine: 50-100 mg')])
    report = module.audit(frame)
    assert report['counts']['rows_with_numeric_caffeine_in_text'] == 2
    assert report['counts']['rows_with_numeric_caffeine_and_basis_cue'] == 1
    claims = report['examples']['numeric_caffeine_text']
    assert claims[0]['caffeine_claims'][0]['mg'] == 200
    assert not claims[1]['caffeine_claims'][0]['explicit_basis_in_snippet']
    assert report['counts']['rows_with_caffeine_attribute'] == 3


def test_multipack_title_and_missing_count_are_separate_from_identity():
    frame = pd.DataFrame([row(title='Tea 4-pack', attributes='Pack Type: Can; Volume: 355'),
                          row(title='Tea 4-pack', attributes='Pack Type: Can; Count per Unit: 4'),
                          row(title='Tea 355 ml', attributes='Pack Type: Can')])
    report = module.audit(frame)
    assert report['counts']['rows_with_multipack_title_signal'] == 2
    assert report['counts']['multipack_title_rows_without_count_per_unit'] == 1
    assert 'decision' not in report


def test_brand_groups_require_valid_gtin_and_nonempty_different_names():
    frame = pd.DataFrame([row(brand='3D'), row(brand='3d'), row(brand=''),
                          row(brand='Olvi', gtin='invalid'), row(brand='Kevyt Olo', gtin='invalid')])
    assert module.audit(frame)['counts']['valid_gtin_groups_with_multiple_nonempty_normalized_brands'] == 0
    frame.loc[len(frame)] = row(brand='Blue Energy')
    report = module.audit(frame)
    assert report['counts']['valid_gtin_groups_with_multiple_nonempty_normalized_brands'] == 1
    assert report['brand_variation_groups'][0]['brands'] == ['3d', 'blue energy']
