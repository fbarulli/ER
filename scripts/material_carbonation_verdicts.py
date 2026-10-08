"""Material + carbonation same-GTIN pair conflicts, feed-level verdicts.

Rebuilds the NEXT_STEPS item-2/3 cohorts (material pair conflicts,
carbonated-vs-still pair conflicts) on the CURRENT eligible catalog (after the
census holds), then decides every same-GTIN family per normalized feed with the
production predicates, plus a per-feed country tally for dissenting claims.
Output: identity/findings/material_carbonation_verdicts.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT/'src'))
from core.common import audit_finding
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd

from core.gtin import gtin_validity
from core.identity_policy import reviewed_row_mask
from core.sku_identity import PACK_SENTINEL, categorical_conflict, row_identity
from core.text import normalize_retailer

DIMS = ('package_material', 'package_type', 'carbonation')


def as_set(value):
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    if isinstance(value, (list, tuple)):
        return frozenset(value)
    return frozenset({value}) if value is not None else frozenset()


def dimension_conflict(dimension, left, right) -> bool:
    """Predicate-exact per-dimension conflict (mirrors identity_conflict)."""
    l_set, r_set = as_set(left), as_set(right)
    if not l_set or not r_set:
        return False
    if dimension == 'pack':
        l_pack = {p for p in l_set if p != PACK_SENTINEL}
        r_pack = {p for p in r_set if p != PACK_SENTINEL}
        return bool(l_pack and r_pack and not (l_pack & r_pack))
    return categorical_conflict(dimension, l_set, r_set)


def main():
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    el = df[gtin_validity(df.gtin) & ~reviewed_row_mask(df)].copy()
    el['g_key'] = el.gtin.str.strip().str.zfill(14)

    family_verdicts = []
    feed_dissent = Counter()      # feed -> dissenting dims authored
    feed_majority = Counter()     # feed -> majority dims held
    feed_countries = defaultdict(set)
    dissent_types = {d: Counter() for d in DIMS}

    for g_key, fam in el.groupby('g_key'):
        rows = fam.sort_values('sku_id').to_dict('records')
        idents = {r['sku_id']: row_identity(dict(r)) for r in rows}
        feed_rows = defaultdict(list)
        for r in rows:
            feed_rows[normalize_retailer(r['retailer'])].append(r)
        if len(feed_rows) < 2:
            continue
        rep = {f: max(fr, key=lambda r: idents[r['sku_id']].completeness)
               for f, fr in feed_rows.items()}
        feeds = sorted(feed_rows)
        for f in feeds:
            feed_countries[f].add(rep[f]['country'])

        conflicts = {}
        for d in DIMS:
            values = {f: sorted(getattr(idents[rep[f]['sku_id']], d)) for f in feeds}
            values = {f: frozenset(v) for f, v in values.items() if v}
            if len(values) < 2 or len({str(sorted(v)) for v in values.values()}) < 2:
                continue
            counts = Counter(str(sorted(v)) for v in values.values())
            if len(counts) < 2:
                continue
            majority_value = counts.most_common(1)[0][0]
            majority_feeds = [f for f in values
                              if str(sorted(values[f])) == majority_value]
            dissent = [f for f, v in values.items()
                       if any(dimension_conflict(d, values[m], v)
                              for m in majority_feeds)]
            if not dissent:
                continue
            conflicts[d] = {
                'feed_value_counts': {f: str(sorted(values[f])) for f in values},
                'majority_feeds': majority_feeds,
                'dissent_feeds': dissent,
                'dissent_rep_sku_ids': [rep[f]['sku_id'] for f in dissent],
            }
            for f in dissent:
                feed_dissent[f] += 1
                pair = tuple(sorted([str(sorted(values[f])), majority_value]))[:2]
                dissent_types[d][pair] += 1
            for f in majority_feeds:
                feed_majority[f] += 1
        if conflicts:
            family_verdicts.append({'gtin': g_key, 'rows': len(rows),
                                    'feeds': feeds,
                                    'conflicts': conflicts,
                                    'title': rows[0]['sku_name_eng'][:80]})

    out = {
        'eligible_rows': len(el),
        'families_with_feed_conflicts': len(family_verdicts),
        'verdicts': family_verdicts,
        'feed_dissent_authored': dict(feed_dissent.most_common(40)),
        'feed_majority_holdings': dict(feed_majority.most_common(40)),
        'feed_country_tallies': {f: sorted(c) for f, c in feed_countries.items()},
        'dissent_pair_types': {d: c.most_common(15) for d, c in dissent_types.items()},
    }
    audit_finding('material_carbonation_verdicts.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('eligible rows:', len(el))
    print('families with feed conflicts:', len(family_verdicts))
    print('feeds authoring most dissent:', feed_dissent.most_common(15))
    for d, c in dissent_types.items():
        print(f"{d}: top dissent pair shapes: {c.most_common(8)}")


if __name__ == '__main__':
    main()
