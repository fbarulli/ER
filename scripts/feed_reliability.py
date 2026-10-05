"""Per-feed reliability aggregation over the full verdict span.

Reads identity/findings/biggest_merge_verdicts.json (produced by
verdict_biggest_merges.py with a large top-N, i.e. ALL multi-feed same-GTIN
families) and tallies, per normalized feed: families spoken in, decisive
dissents authored per dimension, majority holdings per dimension, and the
dissent/majority ratio that scores systematic feed noise.
Output: identity/findings/feed_reliability.json + FEED_RELIABILITY.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIMS = ('brand', 'volume_ml', 'pack', 'flavor', 'carbonation', 'sweetener',
        'sweetener_type', 'sweetening', 'pulp', 'package_type', 'package_material')

fam_spoken = Counter()
dissent = {d: Counter() for d in DIMS}
majority = {d: Counter() for d in DIMS}

for v in json.load(open(ROOT / 'identity/findings/biggest_merge_verdicts.json'))['verdicts']:
    feeds_spoken = set()
    for d, info in v['dimension_verdicts'].items():
        for f in info.get('consensus_feeds', []):
            feeds_spoken.add(f)
            majority[d][f] += 1
        for f in info.get('dissent_feeds', []):
            feeds_spoken.add(f)
            dissent[d][f] += 1
    for f in feeds_spoken:
        fam_spoken[f] += 1

rows = []
for f, spoken in fam_spoken.items():
    d_total = sum(dissent[d][f] for d in DIMS)
    m_total = sum(majority[d][f] for d in DIMS)
    ratio = d_total / m_total if m_total else None
    top_dims = sorted(((d, dissent[d][f]) for d in DIMS if dissent[d][f]),
                      key=lambda t: -t[1])[:4]
    rows.append({
        'feed': f, 'families_spoken': spoken,
        'dissent_authored': d_total, 'majority_holdings': m_total,
        'dissent_ratio': ratio,
        'top_dissent_dims': top_dims,
        'per_dim_dissent': {d: dissent[d][f] for d in DIMS if dissent[d][f]},
    })
rows.sort(key=lambda r: -(r['dissent_authored'] / max(r['majority_holdings'], 1)))
json.dump({'feeds': rows, 'families': len(json.load(open(ROOT / 'identity/findings/biggest_merge_verdicts.json'))['verdicts'])},
          open(ROOT / 'identity/findings/feed_reliability.json', 'w'), indent=1)

lines = ['# Per-feed reliability (all multi-feed same-GTIN families)', '',
         "Ratio = dissents authored / majority holdings over descriptor verdicts.",
         'High ratio = systematically contradicting feed; low = consistently',
         'in the consensus or mostly silent.', '',
         '| Feed | families | dissents | majority | ratio | top dissent dims |',
         '|---|---:|---:|---:|---:|---|']
for r in rows[:40]:
    dims = ', '.join(f'{d} x{n}' for d, n in r['top_dissent_dims'])
    ratio = f"{r['dissent_ratio']:.2f}" if r['dissent_ratio'] is not None else 'silent-majority'
    lines.append(f"| {r['feed']} | {r['families_spoken']} | {r['dissent_authored']} "
                 f"| {r['majority_holdings']} | {ratio} | {dims} |")
(ROOT / 'identity/findings/FEED_RELIABILITY.md').write_text('\n'.join(lines) + '\n')

print('feeds ranked:', len(rows))
for r in rows[:25]:
    ratio = r['dissent_ratio']
    print(f"{r['feed'][:18]:18} spoken {r['families_spoken']:4} | dissent {r['dissent_authored']:4} "
          f"| majority {r['majority_holdings']:5} | ratio "
          f"{ratio and round(ratio, 2)}")
