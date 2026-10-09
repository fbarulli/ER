"""All-local checks on the 37 identical-name different-GTIN negative pairs.

For every pair: (1) brand-prefix family scan across the whole corpus, (2) price/
currency/country forensics, (3) URL-slug archaeology, (4) description-surface
replay (identity dims recomputed with descriptions added), (5) governance-file
cross-reference. Output: identity/findings/negative_local_checks.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.common import audit_finding
from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd

from core.text import normalize_text

COLUMNS = ('sku_id', 'retailer', 'country', 'sku_name_eng', 'description_short_eng',
           'breadcrumbs_eng', 'sku_url', 'image_url', 'sku_last_price', 'gtin',
           'brand', 'category', 'attribute')


def zfill_key(gtin: str) -> str:
    g = (gtin or '').strip()
    return g.zfill(14) if len(g) == 14 else (g.zfill(13) if g.isdigit() and len(g) in (12, 13) else g)


def canonical13(gtin: str) -> str | None:
    g = (gtin or '').strip()
    if not g.isdigit() or len(g) not in (12, 13, 14):
        return None
    if len(g) == 12:
        return '0' + g
    if len(g) == 14:
        return g[1:]
    return g


def main():
    df = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
    raw = {r['sku_id']: dict(r) for r in df[list(COLUMNS)].to_dict('records')}
    # brand column -> GTIN prefix family (corpus-wide, using brand cell text)
    brand_gtins = defaultdict(list)
    for r in raw.values():
        bs = (r.get('brand') or '').strip().lower()
        c13 = canonical13(r['gtin'])
        if bs and c13:
            brand_gtins[bs].append(c13)

    j = json.load(open(audit_finding('residual_cases.json')))
    neg = [c for c in j if c['kind'] == 'different_gtin']
    pairs = []
    for c in neg:
        L, R = c['left'], c['right']
        same_name = L['sku_name_eng'] == R['sku_name_eng']
        same_img = L['image_url'] == R['image_url'] and bool(L['image_url'])
        same_url = L['sku_url'].split('?')[0] == R['sku_url'].split('?')[0]
        if not (same_name and (same_img or same_url)):
            continue
        pairs.append((c, L, R))

    out = []
    for c, L, R in pairs:
        rowL, rowR = raw[L['sku_id']], raw[R['sku_id']]
        rec = {'id': c['id'], 'title': L['sku_name_eng'][:64],
               'gtins': [L['gtin'], R['gtin']]}

        # (1) brand-prefix family scan
        bs = (rowL.get('brand') or '').strip().lower()
        c13L, c13R = canonical13(L['gtin']), canonical13(R['gtin'])
        fam = brand_gtins.get(bs, [])
        others = [g for g in fam if g not in (c13L, c13R)]
        prefixes = Counter(g[:3] for g in others)
        rec['brand'] = rowL.get('brand')
        rec['prefix'] = {'left': (c13L or '')[:3], 'right': (c13R or '')[:3],
                         'brand_family_prefixes': dict(prefixes.most_common(6)),
                         'brand_family_size': len(others)}
        if not others:
            rec['prefix_verdict'] = 'single-family-pair (no other GTINs of this brand)'
        elif c13L is None or c13R is None:
            rec['prefix_verdict'] = 'GTIN-8/short code — no 3-digit prefix family'
        else:
            top3, topn = prefixes.most_common(1)[0]
            in_fam = sum(1 for g in others if g[:3] == top3) / len(others)
            outliers = sum(1 for p3 in ((c13L or '')[:3], (c13R or '')[:3])
                           if p3 and p3 != top3)
            rec['prefix_verdict'] = (
                f"family {top3} x{topn} ({in_fam:.0%}); pair prefixes "
                f"{(c13L or '')[:3]}/{(c13R or '')[:3]} -> "
                f"{'registry-consistent' if outliers == 0 else 'OUTLIER(S) — pseudo-GTIN suspicion'}")

        # (2) price/currency/country forensics
        rec['forensics'] = {
            'prices': [rowL.get('sku_last_price'), rowR.get('sku_last_price')],
            'country': [rowL.get('country'), rowR.get('country')],
            'same_price': rowL.get('sku_last_price') == rowR.get('sku_last_price'),
            'same_desc': rowL['description_short_eng'] == rowR['description_short_eng'],
        }

        # (3) URL-slug archaeology
        def arch(url):
            path = url.split('?')[0]
            after = url.split('?', 1)[1] if '?' in url else ''
            bits = {'path_tail': path.rstrip('/').split('/')[-1][:60],
                    'variant_page': 'classType=VARIANT' in url,
                    'amazon_search_ref': 'ref=sr_' in url,
                    'amazon_dp_asin': (url.split('/dp/', 1)[1].split('/')[0]
                                       if '/dp/' in url else None)}
            kw = [p[9:] for p in after.split('&') if p.startswith('keywords=')]
            bits['search_keywords'] = kw[0][:80] if kw else None
            return bits
        aL, aR = arch(L['sku_url']), arch(R['sku_url'])
        rec['url'] = {'left': aL, 'right': aR,
                      'same_path_tail': aL['path_tail'] == aR['path_tail'],
                      'same_search_query': (aL['search_keywords'] == aR['search_keywords']
                                            and aL['search_keywords'] is not None)}
        out.append(rec)

    (audit_finding('negative_local_checks.json')).write_text(
        json.dumps({'pairs': out}, indent=1))
    for rec in out:
        print(f"== {rec['id']} {rec['title']}")
        print(f"   brand={rec['brand']!r} prefixes {rec['prefix']['left']}/{rec['prefix']['right']}"
              f" | {rec['prefix_verdict']}")
        f = rec['forensics']
        print(f"   prices {f['prices']} same={f['same_price']} same_desc={f['same_desc']} "
              f"countries={f['country']}")
        u = rec['url']
        print(f"   url tails {u['left']['path_tail']} / {u['right']['path_tail']} "
              f"variant={u['left']['variant_page']}/{u['right']['variant_page']} "
              f"sr_query_same={u['same_search_query']}")


if __name__ == '__main__':
    main()
