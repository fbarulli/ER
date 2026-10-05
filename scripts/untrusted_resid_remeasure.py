"""Re-measure the untrusted-title collision census after the 86 scope holds.

Groups the frozen census (959 groups) against the CURRENT eligible catalog:
rows whose gtin is now held (or otherwise review-pinned) leave the group; a
group whose every row leaves is retired. Output:
identity/findings/untrusted_resid_remeasure.json + md appendix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd

from core.gtin import gtin_validity
from core.identity_policy import reviewed_row_mask


def main():
    census = json.load(open(ROOT / 'identity/findings/untrusted_title_conflicts.json'))
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    facts = gtin_validity(df.gtin)
    held = ~facts & (df.gtin.str.strip() != '')
    pinned = reviewed_row_mask(df)
    eligible = facts & ~pinned
    elig_by_sid = {sid: bool(e) for sid, e in zip(df.sku_id, eligible)}
    gtin_by_sid = dict(zip(df.sku_id, df.gtin.str.strip()))

    before_rows = before_pairs = after_rows = after_groups = 0
    group_states = {'retired': 0, 'reduced': 0, 'intact': 0}
    reduced = []
    for g in census:
        rows = w = g.get('rows', []) or []
        ids = [r['sku_id'] for r in rows]
        liv = [sid for sid in ids if elig_by_sid.get(sid, False)]
        dead_held = [sid for sid in ids
                     if not elig_by_sid.get(sid, False)
                     and (held[df.index[df.sku_id == sid][0]]
                          if (df.sku_id == sid).any() else False)]
        before_rows += len(ids)
        after_rows += len(liv)
        if not liv:
            group_states['retired'] += 1
        elif len(liv) < len(ids):
            group_states['reduced'] += 1
            reduced.append({'retailer': g['retailer'], 'title': g['title'],
                            'before': len(ids), 'after': len(liv),
                            'held_out_gtins': sorted({(gtin_by_sid.get(s) or '')
                                                      for s in dead_held})[:4]})
        else:
            group_states['intact'] += 1

    # rough pair estimate: conflicting pairs were census-computed as pairs with
    # conflicting descriptors; proportional shrink is rows*(rows-1)/2-ish only
    # for reporting clarity we compute row counts, not re-inferred pair counts.
    out = {'before_rows': before_rows, 'after_rows': after_rows,
           'rows_retired': before_rows - after_rows,
           'groups': len(census), 'states': group_states,
           'reduced_examples': reduced[:40]}
    (ROOT / 'identity/findings/untrusted_resid_remeasure.json').write_text(
        json.dumps(out, indent=1) + '\n')
    md = ['## Untrusted-title census re-measured after the scope holds', '',
          json.dumps({'groups': group_states, 'rows': out}, indent=2), '']
    md += ['Reduced groups:']
    md += [f"- {r['retailer']} \u2014 {r['title']} ({r['before']} rows \u2192 {r['after']})"
           for r in reduced[:15]]
    (ROOT / 'identity/findings/UNTRUSTED_TITLE_CONFLICTS.md').write_text('')
    p = ROOT / 'identity/findings/UNTRUSTED_TITLE_CONFLICTS.md'
    p.write_text(p.read_text() + '\n' + '\n'.join(md) + '\n')
    print(json.dumps({'states': group_states, 'before_rows': before_rows,
                      'after_rows': after_rows}, indent=2))


if __name__ == '__main__':
    main()
