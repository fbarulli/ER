#!/usr/bin/env python3
"""Read-only audit of measurement basis, packaging levels, and brand variation.

Signals select listings for review. They neither repair metadata nor establish
identity. Counts use all original listings; valid-GTIN group counts are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re

import pandas as pd

from core.common import DATA_PATH
from core.gtin import normalize_and_validate_gtin
from core.product_dimensions import row_dimensions
from core.text import normalized_attribute_text
from core.progress import tracked

logger = logging.getLogger(__name__)
MG = re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\s*(?:of\s+)?caffeine\b|\bcaffeine\s*(?:content\s*)?(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\b', re.I)
BASIS = re.compile(r'(?:per|each|in\s+(?:a|each))\s+(?:\d+(?:\.\d+)?\s*(?:fl\.?\s*)?(?:oz|ounce|ml|milliliter|l|litre|liter)s?\s*)?(?:can|bottle|serving|100\s*ml)\b', re.I)
PACK = re.compile(r'\b(?:\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(?:ml|cl|l|oz)|\d+\s*[- ]?\s*(?:pack|pk|ct|count)\b|pack\s+of\s+\d+\b|\d+\s*[- ]?\s*(?:fl\.?\s*)?oz\s*(?:cans|bottles)\s*/\s*\d+\s*pk)', re.I)


def audit(frame: pd.DataFrame) -> dict:
    required = {'sku_id', 'gtin', 'brand', 'retailer', 'sku_name_eng', 'description_short_eng', 'attribute'}
    if not required <= set(frame.columns):
        raise ValueError(f'missing original dataset columns: {sorted(required - set(frame.columns))}')
    facts = normalize_and_validate_gtin(frame.gtin)
    counts = Counter()
    examples = defaultdict(list)
    groups = defaultdict(list)
    for i, row in enumerate(tracked(frame.to_dict('records'), 'identity context', len(frame))):
        attributes = row_dimensions({'attribute': row['attribute']}).attributes
        text = row['sku_name_eng'] + '\n' + row['description_short_eng']
        claims = []
        for match in MG.finditer(text):
            snippet = text[max(0, match.start()-90):match.end()+160]
            claims.append({'mg': float(match[1] or match[2]), 'snippet': snippet,
                           'explicit_basis_in_snippet': bool(BASIS.search(snippet))})
        has_caffeine = bool(attributes.get('Caffeine'))
        pack_match = PACK.search(row['sku_name_eng'])
        flags = []
        if has_caffeine:
            counts['rows_with_caffeine_attribute'] += 1
            # The raw enum stores mg but provides no denominator field.
            flags.append('caffeine_attribute_without_structured_basis')
        if claims:
            counts['rows_with_numeric_caffeine_in_text'] += 1
            if has_caffeine:
                counts['rows_with_caffeine_attribute_and_numeric_text'] += 1
            if any(c['explicit_basis_in_snippet'] for c in claims):
                counts['rows_with_numeric_caffeine_and_basis_cue'] += 1
            flags.append('numeric_caffeine_text')
        if pack_match:
            counts['rows_with_multipack_title_signal'] += 1
            if attributes.get('Pack Type'):
                counts['multipack_title_rows_with_pack_type'] += 1
            if not attributes.get('Count per Unit'):
                counts['multipack_title_rows_without_count_per_unit'] += 1
            flags.append('multipack_title')
        if 'can' in attributes.get('Pack Type', ()) and attributes.get('Pack Material Type', frozenset()) & {'paper/carton', 'glass', 'plastic'}:
            counts['can_with_nonmetal_material_rows'] += 1
            flags.append('can_nonmetal_material_review')
        record = {'sku_id': row['sku_id'], 'gtin': row['gtin'], 'retailer': row['retailer'],
                  'sku_name_eng': row['sku_name_eng'], 'raw_attributes': row['attribute'],
                  'caffeine_claims': claims, 'multipack_title_match': pack_match[0] if pack_match else None}
        for flag in flags:
            if len(examples[flag]) < 8:
                examples[flag].append(record)
        if facts.gtin_structurally_valid.iat[i]:
            groups[str(facts.gtin_clean.iat[i])].append(row)
    brand_groups = []
    for gtin, rows in groups.items():
        brands = {normalized_attribute_text(r['brand']) for r in rows if r['brand'].strip()}
        if len(brands) > 1:
            brand_groups.append({'validated_gtin': gtin, 'brands': sorted(brands),
                                 'rows': rows})
    counts['valid_gtin_groups_with_multiple_nonempty_normalized_brands'] = len(brand_groups)
    counts['listing_rows_in_brand_variation_groups'] = sum(len(g['rows']) for g in brand_groups)
    return {'source_population': 'all original dataset listings', 'rows': len(frame),
            'method': 'Heuristic text cues select review cases; no identity decisions or corrections. Caffeine denominator is absent from the registered raw attribute schema. Multipack detection uses title only. Brand comparisons use normalized nonempty names and checksum-valid cleaned GTIN groups.',
            'counts': dict(counts), 'examples': dict(examples), 'brand_variation_groups': brand_groups,
            'limitations': ['A per-serving cue near a caffeine number is not a verified binding between that number and that serving.',
                           'A multipack title does not establish a reliable numeric pack count; nested packaging must be inspected.',
                           'Nonmetal material with Can may describe an outer package, or incorrect metadata.',
                           'Brand variation does not prove a parent/subbrand relationship.',
                           'Full original-row counts differ from the capped deduped pair sample in finding 01.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA_PATH)
    parser.add_argument('--output', type=Path, default=Path('dashboard/evidence/identity/03_measurement_and_packaging_context.json'))
    args = parser.parse_args()
    report = audit(pd.read_csv(args.dataset, dtype=str, keep_default_na=False, low_memory=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info('identity context audit: %s', report['counts'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
