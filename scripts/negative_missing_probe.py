"""Manually verify MISSING descriptor fields in the 37 deep-agreement negative pairs.

For every (row, dimension) that the frozen residual cards mark missing, search ALL
13 original dataset columns one by one with the corpus' own extractors and report
exactly which column(s) carry the un-extracted value (extraction gap) or none
(genuinely absent).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from core.common import audit_finding
from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd

from core.text import extract_volume_evidence
from core.critical_attributes import flavor_tokens_from_text
from core.url_evidence import url_text
from pipeline import extract_pack_from_title

COLUMNS = ('sku_id', 'retailer', 'country', 'sku_name_eng',
           'description_short_eng', 'breadcrumbs_eng', 'sku_url', 'image_url',
           'sku_last_price', 'gtin', 'brand', 'category', 'attribute')


def carbonation_hits(text: str):
    hits = sorted({w for w in ('sparkling', 'carbonated', 'kohlensaeure',
                               'kohlensäure', 'con gas', 'still', 'stilles',
                               'no bubbles', 'sin gas', 'no bubbles')
                   if w in text.lower()})
    return hits


def sweetener_hits(text: str):
    low = text.lower()
    return sorted({w for w in ('sugar', 'sucralose', 'stevia', 'aspartame',
                               'acesulfame', 'cyclamate', 'saccharin', 'syrup',
                               'no sugar', 'sugar free', 'zero sugar',
                               'unsweetened', 'monk fruit', 'diet', 'miel',
                               'nuoc')  # nuoc = false positive guard needed?
                   if w in low})


def package_type_hits(text: str):
    return sorted({w for w in ('bottle', 'flasche', 'can', 'dose', 'carton',
                               'tetra', 'jar', 'glass', 'blister', 'box',
                               'pouch', 'cup', 'tub', 'tin', 'sachet',
                               'pet ', ' aluminium', 'doppio')
                   if w in text.lower()})


def package_material_hits(text: str):
    return sorted({w for w in ('glass', 'plastic', 'pet', 'metal', 'aluminium',
                               'aluminum', 'alloy', 'paperboard', 'carton',
                               'tin ', 'monster', 'poly', 'tins')
                   if w in text.lower()})


def pulp_hits(text: str):
    return sorted({w for w in ('pulp', 'pulpa', 'con pulpa', 'no pulp',
                               'voxels') if w in text.lower()})


def volume_weight_hits(text: str):
    low = text.lower()
    fluid = sorted(set(re.findall(
        r'(\d+(?:[.,]\d+)?)\s*(?:ml|cl|fl oz|fluid ounces?|liters?|litres?|l\b)', low)))
    other = sorted(set(re.findall(
        r'(\d+(?:[.,]\d+)?)\s*(?:oz|ounce|g\b|gr\b|gram|mg\b)', low)))
    return fluid, other


def probe(dimension, row):
    """Return {column: finding} for one missing (row, dimension)."""
    out = {}
    norm_row = dict(row)
    norm_row['sku_url'] = url_text(row.get('sku_url'))
    for col in COLUMNS:
        text = str(norm_row.get(col, '') or '')
        if not text or text.lower() in ('nan', 'na'):
            continue
        if dimension == 'flavor':
            toks = flavor_tokens_from_text(text)
            if toks:
                out[col] = sorted(toks)
        elif dimension == 'carbonation':
            h = carbonation_hits(text)
            strip_terms = {'still', 'stilles', 'untreated'}
            if h and h - strip_terms:
                out[col] = [w for w in h if w not in strip_terms]
            elif h:
                out[col] = 'only-' + ','.join(h)
        elif dimension == 'sweetener' or dimension == 'sweetener_type':
            h = sweetener_hits(text)
            # 'sugar alone' is name noise (e.g. brand Sugar); return with care
            if h:
                out[col] = h
        elif dimension == 'package_type':
            h = package_type_hits(text)
            if h:
                out[col] = h
        elif dimension == 'package_material':
            h = package_material_hits(text)
            if h:
                out[col] = h
        elif dimension == 'pulp':
            h = pulp_hits(text)
            if h:
                out[col] = h
        elif dimension == 'sweetening':
            h = sorted({w for w in ('sweetened', 'unsweetened', 'artificial sweetener',
                                    'no artificial sweeteners') if w in text.lower()})
            if h:
                out[col] = h
        elif dimension == 'volume_ml':
            fluid, other = volume_weight_hits(text)
            if fluid or other:
                out[col] = ('fluid=' + ','.join(fluid) if fluid else '') + \
                           (' weight=' + ','.join(other) if other else '')
        elif dimension == 'pack':
            n, conf = extract_pack_from_title(text)
            # n==1 is the extraction sentinel here; only report confident hits
            if n and n != 1 and conf > 0.0:
                out[col] = f'pack={n} conf={conf}'
    return out


def main():
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    raw = {r['sku_id']: dict(r) for r in df[list(COLUMNS)].to_dict('records')}

    j = json.load(open(audit_finding('residual_cases.json')))
    neg = [c for c in j if c['kind'] == 'different_gtin']
    cards = []
    for c in neg:
        L, R = c['left'], c['right']
        same_name = L['sku_name_eng'] == R['sku_name_eng']
        same_img = L['image_url'] == R['image_url'] and bool(L['image_url'])
        same_url = L['sku_url'].split('?')[0] == R['sku_url'].split('?')[0]
        if same_name and (same_img or same_url):
            cards.append((c, L, R))

    report = []
    col_hit = Counter()
    stats = Counter()
    for c, L, R in cards:
        pair = {'id': c['id'], 'title': L['sku_name_eng'][:70],
                'gtins': [L.get('gtin'), R.get('gtin')], 'ga': []}
        for side, rowc in (('left', L), ('right', R)):
            missing_dims = [d for d, who in c['missing'].items()
                            if who in ('both', side)]
            for dim in missing_dims:
                hits = probe(dim, raw.get(rowc['sku_id'], {}))
                if not hits:
                    verdict = 'absent'
                    stats['absent'] += 1
                else:
                    verdict = 'gap'
                    stats['gap'] += 1
                    for col in hits:
                        col_hit[col] += 1
                pair['ga'].append({'side': side, 'sku': rowc['sku_id'],
                                   'dim': dim, 'verdict': verdict,
                                   'hits': {k: v[:6] for k, v in hits.items()}})
        if pair['ga']:
            report.append(pair)

    json.dump({'pairs': report, 'stats': {'absent': stats['absent'],
                                          'gap': stats['gap'],
                                          'gap_by_column': col_hit}},
              open(audit_finding('negative_missing_probe.json'), 'w'),
              indent=1)
    print('missing dims verified:', stats['absent'] + stats['gap'])
    print('genuinely absent everywhere (all 13 columns):', stats['absent'])
    print('present in raw columns but NOT extracted (extraction gaps):', stats['gap'])
    print('gap columns:', col_hit.most_common())
    print('pairs with at least one finding:', len(report))


if __name__ == '__main__':
    main()
