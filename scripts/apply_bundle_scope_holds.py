"""Apply the corpus-wide bundle-scope census as reviewed identity holds.

Reads identity/findings/corpus_bundle_scope.json (produced by
census_bundle_scope.py) and, for every unheld family it contains:

- adds the GTIN to config/identity_reviews.json `quarantined_gtins` with the
  same reason wording the cohort-derived holds use;
- appends the family to identity/findings/bundle_scope_holds.json in the
  established schema (evidence_cases empty: these families were observed
  corpus-wide, not inside the frozen residual cohort);
- appends the family section to identity/findings/BUNDLE_SCOPE_FINDINGS.md.

The census file lists families explicitly; this script never re-derives them,
so re-running with the same census is idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / 'identity/findings'
CONFIG = ROOT / 'config/identity_reviews.json'
REASON = ('Source titles advertise different explicit retail bundle counts '
          'under one GTIN; identifier scope (inner item versus complete '
          'offer) is unresolved. See identity/findings/BUNDLE_SCOPE_FINDINGS.md.')

SECTION_PREAMBLE = ('Source titles advertise different explicit bundle counts '
                    'under one GTIN; identifier scope (inner item versus '
                    'complete offer) is unresolved. See '
                    'identity/findings/BUNDLE_SCOPE_FINDINGS.md.')


def main():
    census = json.loads((FINDINGS / 'corpus_bundle_scope.json').read_text())
    config = json.loads(CONFIG.read_text())
    held = {key.zfill(14) for key in config['quarantined_gtins']}
    holds_doc = json.loads((FINDINGS / 'bundle_scope_holds.json').read_text())
    doc_text = (FINDINGS / 'BUNDLE_SCOPE_FINDINGS.md').read_text()

    applied = []
    for family in census['families_detail']:
        raw_gtin = family['gtin']
        zfilled = raw_gtin.zfill(14)
        if zfilled in held:
            continue
        rows = family['rows_detail']
        evidence_sku_ids = [r['sku_id'] for r in rows]
        config['quarantined_gtins'][raw_gtin.lstrip('0') or raw_gtin] = {
            'reason': REASON,
            'evidence_sku_ids': evidence_sku_ids,
        }
        holds_doc.append({
            'gtin': raw_gtin.lstrip('0') or raw_gtin,
            'evidence_cases': [],
            'evidence_sku_ids': evidence_sku_ids,
            'census_source': 'scripts/census_bundle_scope.py (corpus-wide)',
            'affected_rows': [{
                'sku_id': r['sku_id'],
                'retailer': r['retailer'],
                'country': r['country'],
                'sku_name_eng': r['title'],
                'sku_url': r['url'],
            } for r in rows],
        })
        lines = [f'## GTIN {raw_gtin.lstrip("0") or raw_gtin} — {family["rows"]} affected rows',
                 '', SECTION_PREAMBLE, '']
        lines += [f"- SKU {r['sku_id']} ({r['retailer']}): {r['title']}. "
                  f"Source: {r['url']}" for r in rows]
        doc_text = doc_text.rstrip('\n') + '\n\n' + '\n'.join(lines) + '\n'
        applied.append(raw_gtin)

    CONFIG.write_text(json.dumps(config, indent=2) + '\n')
    (FINDINGS / 'bundle_scope_holds.json').write_text(
        json.dumps(holds_doc, indent=2) + '\n')
    (FINDINGS / 'BUNDLE_SCOPE_FINDINGS.md').write_text(doc_text)
    print(f'applied {len(applied)} new holds; config total '
          f"{len(config['quarantined_gtins'])}")


if __name__ == '__main__':
    main()
