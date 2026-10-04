"""Preserve terminal retail pack counts such as the audited Hip Pop 12x."""
import pytest

from pipeline import extract_pack_evidence, extract_pack_from_title


@pytest.mark.parametrize('title', [
    'Hip Pop - Blueberry Ginger - kombucha - 12x',
    'Hip Pop - Tropical Peach - Living Soda - 12x',
    'Kombucha (12×)', 'Kombucha - 12 X ',
])
def test_terminal_pack_count_retains_evidence_and_reaches_model_inputs(title):
    from core.model_input import build_sku_texts
    from core.sku_identity import row_identity
    import pandas as pd
    count, confidence = extract_pack_from_title(title)
    assert count == 12 and confidence > 0
    evidence = extract_pack_evidence(title)
    assert len(evidence) == 1
    entry = evidence[0]
    assert title[entry['start']:entry['end']] == entry['raw_match']
    row = {'sku_name_eng': title, 'brand': 'Hip Pop', 'attribute': '',
           'description_short_eng': '', 'sku_url': '', 'image_url': '', 'category': ''}
    assert 12. in row_identity(row).pack
    texts, _ = build_sku_texts(pd.DataFrame([row]), structured_enabled=True)
    assert 'pack_qty_12' in texts[0]


@pytest.mark.parametrize('title', [
    'ZX12x', 'Model A12x', 'Kombucha 0.12x', '$ 12x', 'Kombucha 0x',
    '12x1 mineralwasser', '12x1 pet bottles', 'Kombucha 12x stronger',
])
def test_model_codes_prices_and_unfinished_multipliers_remain_unknown(title):
    assert extract_pack_from_title(title) == (1, 0.)
