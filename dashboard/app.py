"""ER composition root and discovery view for the imported Broadway dashboard."""
import html
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
os.environ['BROADWAY_EXPERIMENTS_ROOT'] = str(ROOT / 'experiments')
os.environ['BROADWAY_DEFAULT_EXPERIMENT_SERIES'] = 'identity'
os.environ['BROADWAY_OBSERVATIONS_DIR'] = str(ROOT / 'observations')
os.environ['BROADWAY_DIAGRAMS_DIR'] = str(ROOT / 'diagrams')

from vendor.experiments_dashboard import app
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from catalog import lookup
from fastapi.staticfiles import StaticFiles

app.title = 'ER discovery'
app.mount('/images', StaticFiles(directory=ROOT / 'images'), name='identity-images')
# Preserve the upstream overview and put the discovery at the entrance.
overview = next(route for route in app.routes if getattr(route, 'path', None) == '/')
app.routes.remove(overview)
app.add_api_route('/experiments', overview.endpoint, response_model=None)

DATA = ROOT / 'evidence' / 'identity'

def read(name):
    return json.loads((DATA / ('01_identity_discovery_' + name)).read_text())

def escape(value):
    return html.escape(str(value), quote=True)

def link(url, label):
    if str(url).startswith(('https://', 'http://')):
        return f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
    return escape(url)

@app.get('/', response_class=HTMLResponse)
def discovery():
    from vendor.experiments_dashboard import census_rows
    from core.identity_policy import review_policy
    findings = census_rows('identity')
    counts, audit = read('pair_disagreement_counts.json'), read('identity_dimensions.json')
    links = ''.join(f'<li><a href="/experiments/{escape(item["experiment"])}?focus=identity">{escape(item["experiment"].replace("_", " "))}</a></li>' for item in findings)
    dimensions = ''.join(f'<tr><td>{escape(d["dimension"])}</td><td>{d["coverage"]:.1%}</td><td>{d["same_gtin_both_observed"]:,}</td><td>{d["same_gtin_different"]:,}</td></tr>' for d in sorted(audit['dimension_evaluation'],key=lambda d:d['same_gtin_different'],reverse=True))
    cases = [('860000322966','Halfday honey / ginseng'),('860000322935','Halfday lemon'),('4018852010371','Bebivita banana'),('868784000346','3D blue energy'),('851107003629','Brew Dr ginger / turmeric'),('6419806053204','Kevyt Olo grapefruit')]
    originals = ''.join(f'<li><a href="/catalog?gtin={gtin}">{escape(title)} · {gtin}</a></li>' for gtin,title in cases)
    n_held = len(review_policy().quarantined_gtins)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER discovery</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}h1{{font-size:1.5rem}}li{{margin:.5rem 0}}iframe{{width:100%;height:1100px;border:1px solid #ddd}}.status{{padding:8px;background:#dff4e8;display:inline-block}}.metrics{{display:flex;gap:1rem;flex-wrap:wrap}}.metric{{border:1px solid #ddd;padding:1rem}}.metric strong{{display:block;font-size:1.5rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.5rem;text-align:left}}details{{margin:1rem 0}}input{{padding:.5rem}}summary{{cursor:pointer}}</style></head><body><h1>ER · Identity findings and fixes</h1><p class="status">{n_held} reviewed identifiers blocked from labels and splits · source data retained</p><h2>Original audit</h2><div class="metrics"><div class="metric"><strong>{counts['pairs_with_at_least_one_disjoint_dimension']:,} / {counts['sampled_cross_retailer_same_gtin_pairs']:,}</strong>pairs with raw disagreements</div><div class="metric"><strong>{counts['affected_gtins']:,}</strong>affected GTINs</div><div class="metric"><strong>{audit['observed_dimensions']}</strong>dimensions checked</div><div class="metric"><strong>71,623</strong>original listings</div></div><p>Audit snapshot · capped at 20 cross-retailer pairs per GTIN · raw disagreements, not confirmed identity errors.</p><h2>Findings · comparisons · results</h2><ul>{links}</ul><iframe title="Finding 01 — comparison and fix" src="/results/identity/01_identity_discovery_findings.html" sandbox="allow-same-origin allow-popups"></iframe><h2>Original listing evidence</h2><form action="/catalog"><label for="gtin">GTIN</label> <input id="gtin" name="gtin" placeholder="868784000346" required> <button>Compare listings</button></form><ul>{originals}</ul><details><summary>All 37 dimensions · coverage and raw disagreements</summary><table><tr><th>Dimension</th><th>Coverage</th><th>Pairs with both observed</th><th>Disjoint pairs</th></tr>{dimensions}</table></details><p><a href="/experiments">Experiment dashboard</a> · <a href="/canvas">Maps</a></p></body></html>'''


@app.get('/findings/02', response_class=HTMLResponse)
def exact_title_variants():
    return (ROOT / 'experiments/results/identity/02_exact_title_variants_findings.html').read_text()


@app.get('/api/catalog')
def catalog_api(gtin: str):
    try:
        return lookup(gtin.strip())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get('/catalog', response_class=HTMLResponse)
def catalog_page(gtin: str):
    evidence = catalog_api(gtin)
    body = f'<h1>Original GTIN group · {escape(evidence["validated_gtin"])}</h1><p>{len(evidence["rows"])} listings. {escape(evidence["interpretation"])}</p>'
    if evidence['identity_review_reason']:
        body += f'<p style="background:#fff0cf;padding:1rem">Identity blocked · excluded from labels and splits: {escape(evidence["identity_review_reason"])}</p>'
    for row in evidence['rows']:
        fields = ''.join(f'<tr><th>{escape(c)}</th><td>{link(row[c], "source") if c in ("sku_url", "image_url") else escape(row[c])}</td></tr>' for c in evidence['columns'])
        local = ROOT / 'images' / f'{row["sku_id"]}.png'
        picture = f'<img style="height:160px;max-width:100%;object-fit:contain" src="/images/{escape(row["sku_id"])}.png" alt="{escape(row["sku_name_eng"])}">' if local.exists() else ''
        body += f'<section><h2>{escape(row["retailer"])} · {escape(row["sku_id"])}</h2>{picture}<p>{escape(row["sku_name_eng"])}</p><details><summary>All 13 original columns</summary><table>{fields}</table></details><p>{link(row["image_url"], "Source image")}</p></section>'
    body += '<h2>Cross-retailer dimension comparisons</h2><p>At most 20 pairs, in original listing order. All registered dimensions remain visible; missing evidence stays unknown.</p>'
    for pair in evidence['pairs']:
        rows = ''.join(f'<tr><td>{escape(name)}</td><td>{escape(result["status"])}</td><td>{escape(", ".join(result["left"]))}</td><td>{escape(", ".join(result["right"]))}</td></tr>' for name,result in pair['dimensions'].items())
        changed = sum(r['review'] for r in pair['dimensions'].values())
        context_rows = []
        for name, value in pair['context_comparison'].items():
            scoped = [(name, value)] if 'status' in value else [(name + ' ' + field, result) for field,result in value.items()]
            for field, result in scoped:
                display = lambda v: ', '.join(map(str,v)) if isinstance(v,list) else str(v) if v is not None else 'Unknown'
                context_rows.append(f'<tr><th>{escape(field)}</th><td>{escape(result["status"])}</td><td>{escape(display(result["left"]))}</td><td>{escape(display(result["right"]))}</td></tr>')
        contextual = ''.join(context_rows)
        body += f'<details><summary>{escape(pair["retailer1"])} ({escape(pair["sku_id1"])}) ↔ {escape(pair["retailer2"])} ({escape(pair["sku_id2"])}) · {changed} dimensions needing review</summary><table><tr><th>Dimension</th><th>Status</th><th>Left</th><th>Right</th></tr>{rows}</table><p>Resolved context</p><table><tr><th>Field</th><th>Status</th><th>Left</th><th>Right</th></tr>{contextual}</table></details>'
    return evidence_page('Original catalog evidence', body)


def evidence_page(title, body):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>' + escape(title) + '</title><style>body{font-family:system-ui;margin:2rem;color:#222}section{border-top:1px solid #ccc;padding:1rem 0}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #ccc;padding:.4rem;text-align:left;overflow-wrap:anywhere}details{margin:1rem 0}summary{cursor:pointer}</style></head><body><p><a href="/">Finding 01</a> · <a href="/experiments">All findings</a></p>' + body + '</body></html>'

@app.get('/findings/03', response_class=HTMLResponse)
def context_finding():
    return (ROOT / 'experiments/results/identity/03_measurement_and_packaging_context_findings.html').read_text()

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('ER_DASHBOARD_PORT', '8001')))
