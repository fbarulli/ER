"""Reproduce scorecard residuals and retain individual, source-linked evidence.

Read-only against corpus/policy; outputs only identity/findings/residual_*.
Sibling suggestions are explicitly not applied to the source or identity rules.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from core.common import AUDIT_FINDINGS_DIR
sys.path.insert(0, str(ROOT / 'src'))
from core.gtin import gtin_validity
from core.identity_policy import reviewed_row_mask
from core.sku_identity import row_identity, identity_conflict, evaluate_sku_identity
from dedupe_predicate_scorecard import sha256_of, EXPECTED_SHA

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


def text_identity(identity):
    return dataclasses.replace(identity, gtin_trusted=False, gtin_key='')


def main():
    out = AUDIT_FINDINGS_DIR
    digest = sha256_of(ROOT / 'dataset.csv')
    assert digest == EXPECTED_SHA, 'Scorecard dataset has changed'
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    valid = gtin_validity(df.gtin) & ~reviewed_row_mask(df)
    eligible = df[valid].copy()
    eligible['g_key'] = eligible.gtin.str.strip().str.zfill(14)
    records = eligible.to_dict('records')
    rows = {r['sku_id']: r for r in records}
    assert len(rows) == len(records), 'SKU IDs must uniquely identify evidence'
    families = defaultdict(list)
    for r in records:
        families[r['g_key']].append(r)
    pairs = {}
    # The original scorecard reads blank titles as NA; groupby excludes them.
    multi = (eligible.groupby(['retailer', 'sku_name_eng']).g_key.transform('nunique') > 1) & eligible.sku_name_eng.ne('')
    negative = []
    for _, group in eligible[multi].groupby(['retailer', 'sku_name_eng'], sort=False):
        group = group.sort_values('sku_id').to_dict('records')
        batch = [(a, b) for a, b in zip(group, group[1:]) if a['g_key'] != b['g_key']]
        negative.extend([batch[i] for i in np.linspace(0, len(batch)-1, 8).astype(int)] if len(batch)>8 else batch)
    positive = []
    for _, group in eligible[eligible.groupby('g_key').sku_id.transform('size')>1].groupby('g_key', sort=False):
        group = group.sort_values(['retailer', 'sku_id']).to_dict('records')
        batch = [(group[0], b) for b in group[1:]]
        positive.extend([batch[i] for i in np.linspace(0,len(batch)-1,8).astype(int)] if len(batch)>8 else batch)
    for kind, supply in [('different_gtin', negative), ('same_gtin', positive)]:
        if len(supply)>4000:
            supply = [supply[i] for i in np.linspace(0,len(supply)-1,4000).astype(int)]
        pairs[kind] = sorted(supply, key=lambda p: (p[0]['sku_id'],p[1]['sku_id']))
    cache = {}
    def identity(row):
        key = row['sku_id']
        if key not in cache:
            cache[key] = row_identity(row)
        return cache[key]
    cases = []
    for kind, supply in pairs.items():
        for a, b in supply:
            ai, bi = identity(a), identity(b)
            conflicts = identity_conflict(text_identity(ai), text_identity(bi))
            if (kind == 'different_gtin' and conflicts) or (kind == 'same_gtin' and not conflicts):
                continue
            case = {'id': f'R{len(cases)+1:04d}', 'kind': kind, 'left': a, 'right': b,
                    'descriptor_conflicts': conflicts, 'production': evaluate_sku_identity(ai, bi),
                    'left_dimensions': {d: plain(getattr(ai,d)) for d in DIMS},
                    'right_dimensions': {d: plain(getattr(bi,d)) for d in DIMS},
                    'missing': {d: ('both' if not getattr(ai,d) and not getattr(bi,d) else 'left' if not getattr(ai,d) else 'right')
                                for d in DIMS if not getattr(ai,d) or not getattr(bi,d)},
                    'siblings': {}, 'proposed_repairs': {}}
            repaired = []
            for side, row, ident in [('left',a,ai), ('right',b,bi)]:
                siblings = [r for r in families[row['g_key']] if r['sku_id'] != row['sku_id']]
                case['siblings'][side] = [dict(r, dimensions={d: plain(getattr(identity(r),d)) for d in DIMS}) for r in siblings]
                updates = {}
                for d in DIMS:
                    if getattr(ident,d):
                        continue
                    evidence = [(r['sku_id'],getattr(identity(r),d)) for r in siblings if getattr(identity(r),d)]
                    values = {v for _, v in evidence}
                    if len(values)==1:
                        value = next(iter(values))
                        updates[d] = value
                        case['proposed_repairs'][side+':'+d] = {'value':plain(value), 'source_skus':[s for s,_ in evidence], 'status':'candidate_requires_source_validation'}
                repaired.append(dataclasses.replace(ident, **updates))
            case['sibling_candidate_conflicts'] = identity_conflict(*map(text_identity,repaired))
            case['disposition'] = ('keep_distinct_gtins' if kind=='different_gtin' else 'retain_trusted_match_audit_descriptors')
            cases.append(case)
        print(kind, 'pairs', len(supply), 'residuals', sum(c['kind']==kind for c in cases), flush=True)
    summary = {'dataset_sha256':digest, 'dataset_rows':len(df), 'trusted_rows':len(eligible),
        'pairs':{k:len(v) for k,v in pairs.items()}, 'residuals':dict(Counter(c['kind'] for c in cases)),
        'actual_decisions':dict(Counter(c['production']['decision'] for c in cases)),
        'same_gtin_conflict_dimensions':dict(Counter(d for c in cases if c['kind']=='same_gtin' for d in c['descriptor_conflicts'])),
        'different_gtin_raw_review_dimensions':dict(Counter(d for c in cases if c['kind']=='different_gtin' for d in c['production']['review_dimensions'])),
        'negative_sibling_candidate_conflicts':sum(bool(c['sibling_candidate_conflicts']) for c in cases if c['kind']=='different_gtin')}
    (out/'residual_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (out/'residual_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# Individual residual case evidence', '', 'Generated by `scripts/investigate_identity_residuals.py`. Source rows are frozen in `residual_cases.json`; live URLs may change. Sibling repairs are candidates, not adjudications.', '']
    for c in cases:
        lines += [f"## {c['id']} — {c['kind']}", '',f"Disposition: **{c['disposition']}**. Actual predicate: `{c['production']['decision']}`.", '']
        for side in ('left','right'):
            r=c[side]
            lines += [f"- {side}: SKU {r['sku_id']}, {r['retailer']}, GTIN `{r['gtin']}` — {r['sku_name_eng']}",f"  Source: {r['sku_url']}", f"  Attributes: {r['attribute']}"]
        lines += ['', 'Descriptor conflicts: '+(', '.join(c['descriptor_conflicts']) or 'none')+'.',
                  'Missing dimensions: '+json.dumps(c['missing'])+'.',
                  'Additional raw dimensions requiring review: '+(', '.join(c['production']['review_dimensions']) or 'none')+'.',
                  'Same-GTIN comparison SKUs: '+json.dumps({s:[r['sku_id'] for r in rs] for s,rs in c['siblings'].items()})+'.',
                  'Candidate missing-field repairs (source validation required): '+json.dumps(c['proposed_repairs'])+'.',
                  'Conflicts after candidate sibling repair: '+(', '.join(c['sibling_candidate_conflicts']) or 'none')+'.', '']
    (out/'residual_cases.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
