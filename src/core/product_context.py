"""Evidence with measurement basis and packaging scope; unknown is explicit.

Title volume/count reuse pipeline extractors. Caffeine conversion requires an
explicit linked denominator. Reviewed image evidence is scoped to one listing
and its GTIN. Raw source attributes are never rewritten.
"""
from functools import lru_cache
import re
from typing import Mapping

from core.identity_policy import review_policy
from core.unit_canonicalization import canonical_volume_ml

_CAFFEINE = re.compile(
    r'(?P<mg>\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\s*(?:of\s+)?caffeine\s*'
    r'(?:per|in\s+(?:each|a)|in|each)\s*'
    r'(?:(?P<volume>\d+(?:\.\d+)?)\s*(?P<unit>fl\.?\s*oz|fluid\s+ounces?|oz|ml|l)\s*)?'
    r'(?P<container>can|bottle|serving)?\b', re.I)
_CONTAINER = re.compile(r'\b(cans?|bottles?|cartons?)\b', re.I)

@lru_cache(maxsize=8192)
def title_quantities(title: str):
    from pipeline import extract_volume_from_title, extract_pack_from_title
    volume = extract_volume_from_title(title)
    count, confidence = extract_pack_from_title(title)
    return volume, count if confidence > 0 else None


def resolve_context(row: Mapping, attributes: Mapping) -> dict:
    title = str(row.get('title', '') or '')
    description = str(row.get('description', '') or '')
    volume, count = title_quantities(title)
    unit_ml = volume['volume_ml'] or None
    inner = sorted({m[0].casefold().rstrip('s') for m in _CONTAINER.finditer(title)})
    context = {
        'unit_volume': {'value': unit_ml, 'unit': 'ml', 'source': 'title', 'evidence': volume.get('raw_match', '')},
        'pack_count': {'value': count, 'source': 'title', 'evidence': title if count else ''},
        'inner_packaging': {'types': inner, 'materials': [], 'source': 'title' if inner else 'unknown'},
        'outer_packaging': {'types': [], 'materials': [], 'source': 'unknown'},
        'unscoped_packaging': {'types': sorted(attributes.get('Pack Type', ())),
                              'materials': sorted(attributes.get('Pack Material Type', ())), 'source': 'raw attributes'},
        'caffeine': {'raw_mg': sorted(attributes.get('Caffeine', ())), 'raw_basis': 'unknown',
                     'claims': [], 'comparison_basis': 'mg_per_100ml'},
    }
    raw_types = set(context['unscoped_packaging']['types'])
    raw_materials = context['unscoped_packaging']['materials']
    if len(inner) == 1 and raw_types == set(inner) and len(raw_materials) == 1:
        context['inner_packaging']['materials'] = raw_materials
        context['inner_packaging']['source'] = 'explicit title container + consistent raw packaging'
    for source, text in (('title', title), ('description', description)):
        for match in _CAFFEINE.finditer(text):
            mg = float(match['mg'])
            basis_ml = canonical_volume_ml(match['volume'], match['unit']) if match['volume'] and float(match['volume']) > 0 else None
            container = (match['container'] or '').casefold()
            # A linked "per can" can use the title's volume only if the title
            # identifies that same container. A serving without size stays unknown.
            if basis_ml is None and not match['volume'] and container in inner:
                basis_ml = unit_ml
            if not container and basis_ml is None:
                continue
            if basis_ml is not None and basis_ml <= 0:
                basis_ml = None
            context['caffeine']['claims'].append({'mg': mg, 'basis': container or 'explicit_volume',
                'basis_ml': basis_ml, 'mg_per_100ml': mg / basis_ml * 100 if basis_ml else None,
                'source': source, 'evidence': match[0]})
    sku = str(row.get('product_id', '') or '')
    reviewed = review_policy().listing_context.get(sku)
    if reviewed:
        from core.gtin import normalize_and_validate_gtin
        import pandas as pd
        actual = normalize_and_validate_gtin(pd.Series([row.get('barcode', '')])).gtin_clean.iat[0]
        if pd.notna(actual) and str(actual).zfill(14) == reviewed.gtin.zfill(14):
            context['inner_packaging'] = {'types': [reviewed.inner_type] if reviewed.inner_type else [],
                'materials': [reviewed.inner_material] if reviewed.inner_material else [], 'source': reviewed.source}
            context['outer_packaging'] = {'types': [reviewed.outer_type] if reviewed.outer_type else [],
                'materials': [reviewed.outer_material] if reviewed.outer_material else [], 'source': reviewed.source}
            if reviewed.pack_count:
                context['pack_count'] = {'value': reviewed.pack_count, 'source': reviewed.source, 'evidence': reviewed.source}
    return context


def compare_context(left: dict, right: dict) -> dict:
    def comparison(a, b):
        if not a or not b:
            return {'status': 'unknown', 'left': a, 'right': b}
        return {'status': 'equal' if a == b else 'different', 'left': a, 'right': b}
    result = {'pack_count': comparison(left['pack_count']['value'], right['pack_count']['value'])}
    for level in ('inner_packaging', 'outer_packaging'):
        result[level] = {field: comparison(left[level][field], right[level][field]) for field in ('types', 'materials')}
    a = [v['mg_per_100ml'] for v in left['caffeine']['claims'] if v['mg_per_100ml'] is not None]
    b = [v['mg_per_100ml'] for v in right['caffeine']['claims'] if v['mg_per_100ml'] is not None]
    # Conflicting quantities inside one listing are not a reliable normalization.
    status = 'unknown_basis'
    if a and b:
        consistent = max(a)-min(a) <= max(a)*.05 and max(b)-min(b) <= max(b)*.05
        status = ('equal' if abs(a[0]-b[0]) <= max(a[0],b[0])*.05 else 'different') if consistent else 'ambiguous'
    result['caffeine'] = {'status': status, 'left': a, 'right': b, 'unit': 'mg_per_100ml',
                          'raw_ranges_compared': False}
    return result
