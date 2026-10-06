"""Verdicts for the biggest retained same-GTIN merges: per-dimension dissent.

For every big same-GTIN family (top-N by row count, excluding held GTINs),
compare each descriptor dimension across the family's rows. A dimension value
supported by a strong majority with a MINORITY dissent is a feed attribute
error candidate (merge stays TRUE; dissent row needs a scoped repair); values
splitting the family are genuine-variant candidates (merge needs scope
adjudication). Writes identity/findings/biggest_merge_verdicts.json.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT/'src'))
from core.common import audit_finding
sys.path.insert(0, str(ROOT / 'src'))
from core.gtin import gtin_validity
from core.identity_policy import held_keys, reviewed_row_mask
from core.sku_identity import (
    PACK_SENTINEL,
    brand_conflict,
    categorical_conflict,
    row_identity,
)
from core.critical_attributes import volumes_compatible
from core.text import normalize_retailer

DIMS = ('brand', 'volume_ml', 'pack', 'flavor', 'carbonation', 'sweetener',
        'sweetener_type', 'sweetening', 'pulp', 'package_type', 'package_material')


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        return sorted([plain(v) for v in value], key=str)
    return value


def as_set(value):
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    if isinstance(value, (list, tuple)):
        return frozenset(value)
    return frozenset({value}) if value is not None else frozenset()


def dimension_conflict(dimension, left, right) -> bool:
    """Mirrors identity_conflict for ONE dimension, with the same predicates."""
    if not left or not right:
        return False
    if dimension == 'brand':
        return brand_conflict(as_set(left), as_set(right))
    if dimension == 'volume_ml':
        return not volumes_compatible(as_set(left), as_set(right))
    if dimension == 'pack':
        l_pack = {p for p in as_set(left) if p != PACK_SENTINEL}
        r_pack = {p for p in as_set(right) if p != PACK_SENTINEL}
        return bool(l_pack and r_pack and not (l_pack & r_pack))
    return categorical_conflict(dimension, as_set(left), as_set(right))


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    valid = gtin_validity(df.gtin) & ~reviewed_row_mask(df)
    el = df[valid].copy()
    el['g_key'] = el.gtin.str.strip().str.zfill(14)
    sizes = el.groupby('g_key').size().sort_values(ascending=False)
    held = held_keys()

    verdicts = []
    for g_key in sizes.index:
        if len(verdicts) >= top_n:
            break
        if g_key in held:
            continue
        fam = el[el.g_key == g_key].sort_values('sku_id')
        rows = fam.to_dict('records')
        if len(rows) < 3:
            continue
        idents = {r['sku_id']: row_identity(dict(r)) for r in rows}
        # One feed can speak several times (country domains, casing). The
        # verdict is decided per normalized FEED with the predicate itself:
        # feed representatives (most complete row) are pairwise
        # identity_conflict-checked, so a duplicated feed can neither
        # manufacture a majority nor hide a conflict.
        retailer_of = {r['sku_id']: normalize_retailer(r['retailer']) for r in rows}
        feed_rows = defaultdict(list)
        for r in rows:
            feed_rows[retailer_of[r['sku_id']]].append(r)
        representative = {}
        for feed, frows in feed_rows.items():
            representative[feed] = max(
                frows, key=lambda r: idents[r['sku_id']].completeness)['sku_id']
        feeds = sorted(feed_rows)
        dims = {}
        for d in DIMS:
            per_feed = {f: plain(getattr(idents[representative[f]], d)) for f in feeds}
            populated = {f: v for f, v in per_feed.items() if v}
            if len(populated) < 2:
                continue
            counts = Counter(str(v) for v in populated.values())
            if len(counts) < 2:
                continue
            majority_value = counts.most_common(1)[0][0]
            majority_feeds = [f for f in populated
                              if str(populated[f]) == majority_value]
            dissent_feeds = [
                f for f, v in populated.items()
                if any(dimension_conflict(d, populated[m], v) for m in majority_feeds)]
            dims[d] = {
                'feed_value_counts': dict(counts),
                'dissent_feeds': dissent_feeds,
                'dissent_sku_ids': [representative[f] for f in dissent_feeds],
                'consensus_feeds': [f for f in populated if f not in dissent_feeds],
                'value_counts': {
                    str(v): c for v, c in Counter(
                        str(v) for v in
                        {r['sku_id']: plain(getattr(idents[r['sku_id']], d)) for r in rows}
                        .values() if v).items()},
            }
        if dims:
            verdicts.append({
                'gtin': g_key, 'rows': len(rows),
                'titles': sorted({r['sku_name_eng'][:90] for r in rows})[:4],
                'dimension_verdicts': dims,
            })

    out = {'verdicts': verdicts}
    audit_finding('biggest_merge_verdicts.json').write_text(
        json.dumps(out, indent=2) + '\n')
    for v in verdicts:
        print(f"== {v['gtin']} rows={v['rows']} {v['titles'][:2]}")
        for d, info in v['dimension_verdicts'].items():
            print(f"   {d}: {info['value_counts']} dissent={info['dissent_sku_ids']}")


if __name__ == '__main__':
    main()
