"""Corpus-wide bundle-scope census: every same-GTIN family whose rows
advertise different explicit retail bundle counts.

Background: BUNDLE_SCOPE_FINDINGS.md / bundle_scope_holds.json inventoried 34
GTIN families, but the supply was the FROZEN residual cohort only (the 56
held positive pairs). The biggest same-GTIN merge families in the corpus
(Cocofina 44 rows, Equinox 28, Actiph 25, What A Melon 19/12, duskin Golden
Delicious 13, Hi Ball 11, Cherry Bay 10, ...) have the same shape and were
never observed by that cohort-limited census. This script re-derives the
census from the full eligible corpus using the SAME pack evidence the
identity predicate sees (pipeline.extract_pack_from_title semantics via
row_identity), so a hold decision never rests on a sample.

Read-only against the corpus; writes identity/findings/corpus_bundle_scope.json
and prints a summary. It does NOT apply holds — applying them is a separate,
reviewed config change.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from core.common import audit_finding
from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / 'src'))
from core.gtin import gtin_validity
from core.identity_policy import held_keys, reviewed_row_mask
from core.sku_identity import PACK_SENTINEL, identity_conflict, row_identity


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        return sorted([plain(v) for v in value], key=str)
    return value


def main():
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    valid = gtin_validity(df.gtin) & ~reviewed_row_mask(df)
    eligible = df[valid].copy()
    eligible['g_key'] = eligible.gtin.str.strip().str.zfill(14)
    families = {k: g for k, g in eligible.groupby('g_key')}

    identities = {}
    for g_key, fam in families.items():
        for row in fam.to_dict('records'):
            identities[row['sku_id']] = row_identity(row)

    already_held = held_keys()
    census = []
    held_collision_families = 0
    for g_key, fam in sorted(families.items()):
        if g_key in already_held:
            # Held families stay in the record so the artifact documents the
            # corpus state (applied holds included), not just the open tail.
            held_collision_families += 1
            continue
        rows = fam.sort_values('sku_id').to_dict('records')
        if len(rows) < 2:
            continue
        packs = {r['sku_id']: {p for p in identities[r['sku_id']].pack
                               if p != PACK_SENTINEL} for r in rows}
        # Bundle-count collision: at least one pair of rows advertises
        # disjoint non-trivial pack counts. Missing counts are not evidence.
        collision_pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = packs[rows[i]['sku_id']], packs[rows[j]['sku_id']]
                if a and b and not (a & b):
                    collision_pairs.append([rows[i]['sku_id'], rows[j]['sku_id']])
        if not collision_pairs:
            continue
        conflict_dims = Counter()
        for a_id, b_id in collision_pairs:
            a, b = identities[a_id], identities[b_id]
            conflict_dims.update(identity_conflict(
                dataclasses.replace(a, gtin_trusted=False, gtin_key=''),
                dataclasses.replace(b, gtin_trusted=False, gtin_key='')))
        title_counts = Counter()
        for r in rows:
            for count in packs[r['sku_id']]:
                title_counts[str(count)] += 1
        census.append({
            'gtin': g_key,
            'rows': len(rows),
            'collision_pairs': len(collision_pairs),
            'advertised_counts': dict(sorted(title_counts.items(), key=lambda kv: float(kv[0]))),
            'conflict_dimensions': dict(conflict_dims),
            'sample_collision_pairs': collision_pairs[:8],
            'rows_detail': [{
                'sku_id': r['sku_id'],
                'retailer': r['retailer'],
                'country': r['country'],
                'title': r['sku_name_eng'],
                'url': r['sku_url'],
                'packs': sorted(packs[r['sku_id']]),
            } for r in rows],
        })

    census.sort(key=lambda f: (-f['collision_pairs'], -f['rows']))
    out = {
        'census': 'corpus-wide bundle-count collisions; disjoint title-advertised pack counts within one eligible GTIN family',
        'eligible_rows': len(eligible),
        'eligible_gtins': len(families),
        'already_held_gtins': len(already_held),
        'held_collision_families': held_collision_families,
        'families': len(census),
        'affected_rows': sum(f['rows'] for f in census),
        'families_detail': census,
    }
    (audit_finding('corpus_bundle_scope.json')).write_text(
        json.dumps(out, indent=2) + '\n')

    print(json.dumps({k: v for k, v in out.items() if k != 'families_detail'}, indent=2))
    print('\nTop families:')
    for f in census[:30]:
        print(f"  {f['gtin']} rows={f['rows']} collision_pairs={f['collision_pairs']} "
              f"counts={f['advertised_counts']} dims={f['conflict_dimensions']}")


if __name__ == '__main__':
    main()
