"""Corpus-wide pseudo-GTIN census via brand-prefix family scans.

Every distinct brand cell maps to the multiset of GS1 3-digit prefixes of its
GTINs. A brand whose GTIN prefixes split into a strong consensus family plus
straggler outliers gets those outlier rows flagged; retailer attribution shows
where scraped pseudo-codes concentrate. Marketplace-attributed outliers get a
stronger reading than national-chain multi-country variants. Output:
identity/findings/pseudo_gtin_census.json + PSEUDO_GTIN_CENSUS.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.common import audit_finding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd

from core.gtin import gtin_validity, normalize_and_validate_gtin


def main():
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    facts = normalize_and_validate_gtin(df.gtin)
    el = df[facts.gtin_structurally_valid].copy()
    el['gkey'] = facts.loc[el.index, 'gtin_clean']
    el['c13'] = el.gkey.map(lambda g: ('0' + g)[:13] if len(g) == 12
                            else (g[1:] if len(g) == 14 and g[0] == '0' else g))
    el = el[el.c13.str.isdigit()].copy()
    el['prefix'] = el.c13.str[:3]
    el['brand_cell'] = (el.brand.fillna('')).str.strip().str.lower()

    rows_e = el[['brand_cell', 'prefix', 'retailer', 'gkey', 'sku_id', 'sku_name_eng']].drop_duplicates('gkey')
    by_brand = defaultdict(Counter)
    rows_by_brand = defaultdict(list)
    for r in rows_e.to_dict('records'):
        b = r['brand_cell']
        if not b:
            continue
        by_brand[b][r['prefix']] += 1
        rows_by_brand[b].append(r)

    flagged = []
    for b, counts in by_brand.items():
        if len(counts) < 2:
            continue
        total = sum(counts.values())
        top3, topn = counts.most_common(1)[0]
        if topn / total >= 0.8 or topn < 4:
            continue
        outliers = [(p, n) for p, n in counts.most_common() if p != top3]
        outlier_rows = [r for r in rows_by_brand[b] if r['prefix'] != top3]
        retailers = Counter(o['retailer'] for o in outlier_rows)
        marketplaces = {'amazon', 'walmart', 'ebay', 'aliexpress', 'gittigidiyor',
                        'trendyol'}
        mp = sum(v for name, v in retailers.items()
                 if name.lower() in marketplaces)
        flagged.append({
            'brand': b, 'total_gtins': total,
            'family_prefix': top3, 'family_count': topn,
            'outlier_prefixes': dict(outliers),
            'outlier_gtins': len(outlier_rows),
            'outlier_retailers': dict(retailers.most_common(6)),
            'sample_outlier_gtins': sorted({o['gkey'] for o in outlier_rows})[:8],
            'marketplace_outliers': mp,
            'reading': ('marketplace pseudo-suspicion' if mp >= 3 else
                        'multi-country brand variants (national chains)'),
        })
    flagged.sort(key=lambda f: -f['outlier_gtins'])

    out = {
        'brands_with_gtins': len([b for b in by_brand if b]),
        'distinct_gtins': int(el.gkey.nunique()),
        'brands_flagged': len(flagged),
        'flagged': flagged,
    }
    (audit_finding('pseudo_gtin_census.json')).write_text(
        json.dumps(out, indent=1) + '\n')

    lines = ['# Pseudo-GTIN census (corpus-wide brand-prefix family scan)', '',
             f"{out['distinct_gtins']} distinct checksum-valid gtins across "
             f"{out['brands_with_gtins']} brand cells; {len(flagged)} brands split on "
             "prefixes (<80% family share with >=4 gtins).", '',
             'Reading per brand: outlier GTINs attributed to marketplaces are the',
             'scraped pseudo-code pattern (see NEGATIVE_AGREEMENT_AUDIT §5);',
             'outliers held together by national retail chains are genuine',
             'multi-country brand variants.', '',
             '| Brand | family | outliers | top outlier prefixes | retailers | reading |',
             '|---|---|---|---|---|---|']
    for f in flagged[:60]:
        lines.append('| {brand} | {family_prefix} x{family_count} | {outlier_gtins} | {pfx} | {ret} | {rdg} |'
                     .format(brand=f['brand'] or '(blank)', family_prefix=f['family_prefix'],
                             family_count=f['family_count'], outlier_gtins=f['outlier_gtins'],
                             pfx=', '.join(f'{p} x{n}' for p, n in
                                           list(f['outlier_prefixes'].items())[:4]),
                             ret=', '.join(f'{k} x{v}' for k, v in
                                           list(f['outlier_retailers'].items())[:3]),
                             rdg=f['reading']))
    (audit_finding('PSEUDO_GTIN_CENSUS.md')).write_text('\n'.join(lines) + '\n')
    mp_flagged = [f for f in flagged if 'marketplace' in f['reading']]
    print('brands flagged:', len(flagged),
          '| marketplace-attributed:', len(mp_flagged),
          '| marketplace outlier gtins:', sum(f['outlier_gtins'] for f in mp_flagged))
    for f in flagged[:12]:
        print(f"{f['brand'][:24]:24} family {f['family_prefix']} x{f['family_count']} | "
              f"outliers {f['outlier_gtins']} {list(f['outlier_prefixes'].items())[:3]} "
              f"| {list(f['outlier_retailers'].items())[:2]} | {f['reading']}")


if __name__ == '__main__':
    main()
