#!/usr/bin/env python3
"""Read-only original-data audit of numbers explicitly labelled GLN and mixed flavor feeds."""
import json
from pathlib import Path
import re
import pandas as pd
from core.gtin import normalize_and_validate_gtin
from core.product_dimensions import row_dimensions

ROOT = Path(__file__).resolve().parents[1]
frame = pd.read_csv(ROOT / 'dataset.csv', dtype=str, keep_default_na=False)
facts = normalize_and_validate_gtin(frame.gtin)
pattern = re.compile(r'\bgln\s*[:#-]?\s*(\d{13})\b', re.I)
explicit = []
for row in frame.to_dict('records'):
    matches = pattern.findall(row['description_short_eng'])
    if row['gtin'] in matches:
        explicit.append(row)
keys = sorted({r['gtin'] for r in explicit})
affected = frame[frame.gtin.isin(keys)]
flavor_groups = []
for gtin, group in frame[facts.gtin_structurally_valid].groupby('gtin'):
    attrs = [row_dimensions({'attributes': s}).attributes.get('Flavour', frozenset()) for s in group.attribute]
    if any(a and b and a.isdisjoint(b) for a in attrs for b in attrs):
        flavor_groups.append({'gtin': gtin, 'rows': len(group)})
selected_keys = keys + ['8713300049748', '8713300049779', '5601607074866', '8713300449227', '8713300049786']
output = {
    'source': 'dataset.csv',
    'explicit_gln_equal_gtin_rows': len(explicit),
    'explicit_gln_equal_gtin_keys': keys,
    'all_rows_sharing_explicit_gln_keys': len(affected),
    'retailers': affected.retailer.value_counts().to_dict(),
    'groups': {gt: {'rows': int((frame.gtin == gt).sum()),
                    'explicit_gln_rows': sum(r['gtin'] == gt for r in explicit)} for gt in keys},
    'same_valid_gtin_disjoint_raw_flavor_groups': flavor_groups,
    'note': 'Disjoint raw flavors are review signals, not confirmed different products. Explicit GLN provenance does not recover true item GTINs.',
    'selected_original_rows': frame[frame.gtin.isin(selected_keys)].to_dict('records'),
}
target = ROOT / 'dashboard/evidence/identity/05_06_gtin_discovery_followup.json'
target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k:v for k,v in output.items() if k not in ['selected_original_rows','same_valid_gtin_disjoint_raw_flavor_groups']}, indent=2))
print('Disjoint flavor groups:',len(flavor_groups),'rows:',sum(g['rows'] for g in flavor_groups))

# Human-readable result pages show source comparisons rather than raw evidence dumps.
from html import escape

def text(value):
    return escape(str(value))

def source(sku):
    selected = frame[frame.sku_id == sku]
    assert len(selected) == 1, sku
    return selected.iloc[0].to_dict()

def table(rows, extra=None):
    fields = [('retailer', 'Retailer'), ('sku_id', 'Source SKU'), ('gtin', 'Filed GTIN'),
              ('sku_name_eng', 'Title')]
    cells = []
    for key, label in fields:
        cells.append('<tr><th>' + label + '</th>' + ''.join('<td>' + text(r[key]) + '</td>' for r in rows) + '</tr>')
    for key in ['Flavour','Volume','Pack Type','Pack Material Type']:
        cells.append('<tr><th>' + key + '</th>' + ''.join('<td>' + text(', '.join(sorted(row_dimensions({'attributes':r['attribute']}).attributes.get(key, ())))) + '</td>' for r in rows) + '</tr>')
    if extra:
        cells.append('<tr><th>Clarifying source evidence</th>' + ''.join('<td>' + text(v) + '</td>' for v in extra) + '</tr>')
    cells.append('<tr><th>Original product</th>' + ''.join('<td><a target="_blank" rel="noopener" href="' + escape(r['sku_url'],quote=True) + '">Source listing</a> · <a href="/catalog?gtin=' + text(r['gtin']) + '">All original columns</a></td>' for r in rows) + '</tr>')
    return '<table><tbody>' + ''.join(cells) + '</tbody></table>'

def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+text(title)+'</title><style>body{font:16px system-ui;color:#202329;margin:24px;max-width:1200px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ddd;padding:12px;text-align:left;vertical-align:top}th{width:150px;background:#f5f6f8}td{min-width:180px}.status{display:inline-block;background:#fff1c9;padding:6px 12px;border-radius:6px}section{margin:32px 0}h2{font-size:20px}a{color:#2457a7}</style><h1>'+text(title)+'</h1>'+body+'</html>'

result_dir = ROOT / 'dashboard/experiments/results/identity'
body = '<p><strong>61 listings explicitly label their filed number “GLN”; 98 listings share those 21 numbers.</strong> All are from Alcampo. These numbers pass the existing checksum.</p><p class="status">Open — identifier-type correction and split protection pending</p>'
body += '<section><h2>Same filed number, different KAS products</h2>' + table([source('160474885'),source('160722770')], ['Description: “gln 8410408010808”; net volume 4l','Description: “gln 8410408010808”; net volume 500ml']) + '<p>Result: lemon 2 × 2 L and orange 500 ml are different sellable items. The shared number is explicitly presented as GLN in both descriptions, so checksum validity does not establish item identity.</p></section>'
body += '<section><h2>Same filed number, tea versus water</h2>' + table([source('160495724'),source('160767321')], ['Description explicitly labels “gln 8480011000022”','Description explicitly labels “gln 8480011000022”']) + '<p>Result: a sugar-free lemon tea and a 5 L mineral-water carafe cannot share item identity on this evidence. Their true item GTINs remain unknown; source records are preserved.</p></section>'
body += '<p>Checked all 71,623 original rows, explicit GLN wording, all 13 columns for affected records, and shared parser dimensions. Do not relabel these records to invented GTINs. Runtime enforcement is not claimed by this result.</p>'
(result_dir / '05_gln_in_gtin_findings.html').write_text(page('Finding 05 · GLN copied into product GTIN',body))
body = '<p class="status">Open — mixed source fields; no automatic repair</p><p>Compare each suspect listing with the same GTIN and a separately numbered flavor from the same brand.</p>'
body += '<section><h2>Passionfruit title attached to an orange product URL</h2>' + table([source('512665846'),source('519464460'),source('70182030')], ['URL path: premium-orange-pulp-free.html; title says Passionfruit Heaven','Title, URL and description agree on Orange Pulp Free','Separate flavor: Passionfruit Heaven, separately numbered GTIN']) + '<p>Result: Jan Linders SKU 512665846 internally mixes passionfruit title/attributes with an orange URL. A real Passionfruit Heaven listing exists under another GTIN, 8713300449227. This supports a mixed-record diagnosis; it does not identify which source field should be replaced.</p></section>'
body += '<section><h2>Strawberry title attached to a Mango Dream URL and description</h2>' + table([source('512465527'),source('519335533'),source('133029943')], ['URL: mango-dream.html; description: “description MANGO DREAM coolbest”; title: Strawberry Hill','Title, description and URL agree on Mango Dream','Separate flavor: Strawberry Hill, separately numbered GTIN']) + '<p>Result: Jan Linders SKU 512465527 has a Strawberry Hill title while its URL and supplier-description name Mango Dream. Coop shows a separately numbered Strawberry Hill, 8713300049786. Correcting the title would require reliable source adjudication; the snapshot establishes the contradiction, not a verified repair.</p></section>'
body += '<p>Checked original titles, descriptions, URLs, GTIN validation and shared-parser flavors. Numbers and flavor distinctions shown are source facts; no listing was edited or excluded by this audit.</p>'
(result_dir / '06_mixed_flavor_source_fields_findings.html').write_text(page('Finding 06 · Flavor title versus source context',body))
for number, name, summary in [
    ('05','gln_in_gtin','Do explicit GLN source labels explain product-barcode collisions?\n\n61 source listings label their filed GTIN as GLN. 21 numbers are shared by 98 Alcampo rows. Compare lemon and orange KAS, and tea versus water. The correction remains open pending enforcement.'),
    ('06','mixed_flavor_source_fields','Can another listing of the same flavor clarify a conflicting GTIN group?\n\nTwo Jan Linders Cool Best records disagree internally: Passionfruit title/orange URL, and Strawberry title/Mango Dream URL and description. Side-by-side original data includes separately numbered real flavor variants. No automatic repair is claimed.')]:
    (ROOT / f'dashboard/experiments/identity/{number}_{name}.py').write_text('"""'+summary+'\n"""\n')
print('Wrote Findings 05/06 HTML comparisons and experiment entries.')
