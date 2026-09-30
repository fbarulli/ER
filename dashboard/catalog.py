"""Original-listing lookup using the shared ER column map and dimension evaluator."""
from functools import lru_cache
from itertools import combinations, islice

import pandas as pd

from core.common import COLUMN_MAPPING, DATA_PATH
from core.gtin import normalize_and_validate_gtin
from core.product_dimensions import evaluate_dimensions, row_dimensions
from core.product_context import compare_context
from core.identity_policy import review_reason
from core.text import normalize_retailer

SOURCE = DATA_PATH

@lru_cache(maxsize=1)
def _catalog(modified_ns: int):
    frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False, low_memory=False)
    return frame, normalize_and_validate_gtin(frame.gtin)


def lookup(gtin: str) -> dict:
    if not gtin.isdigit() or len(gtin) not in (8, 12, 13, 14):
        raise ValueError('Enter a GTIN with 8, 12, 13, or 14 digits.')
    query = normalize_and_validate_gtin(pd.Series([gtin]))
    if not query.gtin_structurally_valid.iat[0]:
        raise ValueError('GTIN check digit is invalid.')
    frame, facts = _catalog(SOURCE.stat().st_mtime_ns)
    selected = frame[facts.gtin_structurally_valid & facts.gtin_clean.eq(query.gtin_clean.iat[0])]
    rows = selected.to_dict('records')
    canonical = selected.rename(columns=COLUMN_MAPPING).to_dict('records')
    evidence = [row_dimensions(row) for row in canonical]
    pairs = []
    candidates = ((i,j) for i,j in combinations(range(len(rows)), 2)
                  if normalize_retailer(rows[i]['retailer']) != normalize_retailer(rows[j]['retailer']))
    for i,j in islice(candidates,20):
        dimensions = evaluate_dimensions(evidence[i], evidence[j])
        pairs.append({'sku_id1': rows[i]['sku_id'], 'sku_id2': rows[j]['sku_id'],
                      'retailer1': rows[i]['retailer'], 'retailer2': rows[j]['retailer'],
                      'dimensions': dimensions, 'context_comparison': compare_context(evidence[i].context, evidence[j].context)})
    return {'query_gtin': gtin, 'validated_gtin': str(query.gtin_clean.iat[0]),
            'columns': list(frame.columns), 'rows': rows, 'pairs': pairs,
            'pair_cap': 20, 'source': 'dataset.csv', 'identity_review_reason': review_reason(gtin),
            'interpretation': 'Raw dimension evidence only. Disjoint values are review signals, not identity decisions. Leading-zero UPC normalization is used for lookup.'}
