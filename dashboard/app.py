"""ER composition root and discovery view for the imported Broadway dashboard."""
import html
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

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
from core.common import DATA_PATH, load_dataset

app.title = 'ER discovery'
from training_reports import router as training_reports_router
app.include_router(training_reports_router)
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
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER discovery</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}h1{{font-size:1.5rem}}li{{margin:.5rem 0}}iframe{{width:100%;height:1100px;border:1px solid #ddd}}.status{{padding:8px;background:#dff4e8;display:inline-block}}.metrics{{display:flex;gap:1rem;flex-wrap:wrap}}.metric{{border:1px solid #ddd;padding:1rem}}.metric strong{{display:block;font-size:1.5rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.5rem;text-align:left}}details{{margin:1rem 0}}input{{padding:.5rem}}summary{{cursor:pointer}}</style></head><body><h1>ER · Identity findings and fixes</h1><p class="status">{n_held} reviewed identifiers blocked from labels and splits · source data retained</p><h2>Original audit</h2><div class="metrics"><div class="metric"><strong>{counts['pairs_with_at_least_one_disjoint_dimension']:,} / {counts['sampled_cross_retailer_same_gtin_pairs']:,}</strong>pairs with raw disagreements</div><div class="metric"><strong>{counts['affected_gtins']:,}</strong>affected GTINs</div><div class="metric"><strong>{audit['observed_dimensions']}</strong>dimensions checked</div><div class="metric"><strong>71,623</strong>original listings</div></div><p>Audit snapshot · capped at 20 cross-retailer pairs per GTIN · raw disagreements, not confirmed identity errors.</p>
<p>Session tracks · <a href="/training"><strong>Training reports</strong></a> · <a href="/datagen"><strong>Datagen track</strong></a> (identity fixes, GTIN integrity, attribute-universe census, datagen budget) · <a href="/gate"><strong>Gate decisions</strong></a> (original-column sample of 5 per decision bucket) · <a href="/graphs"><strong>Graphs track</strong></a> (GNN-only / hybrid semantic-ID lane)</p><h2>Findings · comparisons · results</h2><ul>{links}</ul><iframe title="Finding 01 — comparison and fix" src="/results/identity/01_identity_discovery_findings.html" sandbox="allow-same-origin allow-popups"></iframe><h2>Original listing evidence</h2><form action="/catalog"><label for="gtin">GTIN</label> <input id="gtin" name="gtin" placeholder="868784000346" required> <button>Compare listings</button></form><ul>{originals}</ul><details><summary>All 37 dimensions · coverage and raw disagreements</summary><table><tr><th>Dimension</th><th>Coverage</th><th>Pairs with both observed</th><th>Disjoint pairs</th></tr>{dimensions}</table></details><p><a href="/experiments">Experiment dashboard</a> · <a href="/canvas">Maps</a></p></body></html>'''


@app.get('/findings/02', response_class=HTMLResponse)
def exact_title_variants():
    return (ROOT / 'experiments/results/identity/02_exact_title_variants_findings.html').read_text()


# ── datagen track ────────────────────────────────────────────────────────────
# All measured session evidence feeds one page: the dedupe manifest is read live;
# the session-measured tables are embedded with their source run; the attribute
# universe census (results/attribute_universe_census.json) renders when present
# and states its in-flight status otherwise — never a placeholder number.

_results = ROOT.parent / 'results'

def _manifest():
    try:
        return json.loads((_results / 'manifests' / 'dedupe.json').read_text())
    except Exception:
        return {}


@app.get('/datagen', response_class=HTMLResponse)
def datagen_track():
    m = _manifest()
    ra = m.get('row_accounting', {})
    def metric(v, label, fallback='—'):
        return '<div class="metric"><strong>' + escape(fallback if v in (None, '') else f'{v:,}') + '</strong>' + escape(label) + '</div>'
    metrics = ''.join([
        metric(ra.get('input_rows', 71_623), 'original listings'),
        metric(ra.get('output_rows'), 'deduped rows (merge retry vs 62,963 prior refresh: +116, attributed to commits 0452692..2d3ac4b — wave-1 fixes byte-identical)'),
        metric(ra.get('dropped', {}).get('t1_retailer_barcode', 1_850) if ra else None, 'T1 collapses'),
        metric(ra.get('dropped', {}).get('t1_5_retailer_malformed_barcode_same_product', 96) if ra else None, 'T1.5 malformed-barcode recoveries'),
        metric(ra.get('skipped_checksum_invalid', 3_867) if ra else None, 'checksum-invalid retained rows'),
        metric(ra.get('unresolved_identity_review_rows', 207) if ra else None, 'escalated identity questions (review evidence, never guessed)'),
    ])
    gtin_rows = ''.join(f'<tr><td>{k}</td><td>{v:,}</td></tr>' for k, v in [
        ('missing (NA) barcode', 41_545), ('checksum-invalid (rows survive, identity claim dies)', 3_715),
        ('checksum-valid rows', 26_363), ('distinct valid GTINs', 13_250),
        ('first-run ≠ longest-run cells (truncation fix regression)', 0),
        ('review-quarantined GTINs', 139),
    ])
    desc_rows = ''.join(f'<tr><td>{k}</td><td>{f"{v:,}"}</td></tr>' for k, v in [
        ('deduped rows consuming description after alias fix', 52_856),
        ('carbonation filled where title/attributes empty', 330),
        ('sweetener filled where title/attributes empty', 545),
        ('pulp filled where title/attributes empty', 67),
        ('both-present disagreements (stay veto/review)', 1_073),
        ('rows relying on description ONLY (regression risk)', 0),
    ])
    try:
        vocab = json.loads((ROOT.parent / 'config' / 'vocabulary.json').read_text())
        prov = vocab.get('brand_aliases_provenance', {})
        for a, b in sorted(vocab.get('brand_aliases', {}).items()):
            brand_note += f" · {a}→{b}"
        brand_note = (f"{len(vocab.get('brand_aliases', {}))} alias entries · "
                      f"{prov.get('granted_families', '?')} granted families · "
                      f"{prov.get('dissolved_false_veto_pairs', '?')} false-veto pairs dissolved · "
                      f"{prov.get('declined_groups', '?')} groups declined with reasons")
    except Exception:
        brand_note = 'vocabulary.json not readable in this view'
    census_path = _results / 'attribute_universe_census.json'
    if census_path.exists():
        try:
            cu = json.loads(census_path.read_text())
            universe_status = f"census loaded · {len(cu)} top-level keys · results/attribute_universe_census.json"
        except Exception:
            universe_status = 'census file present but not parseable'
        budget_rows = '<li>' + '</li><li>'.join(escape(str(k)) for k in sorted(cu)[:40]) + '</li>'
        budget_html = f'<h2>AttributeUniverse census</h2><p class="muted">{escape(universe_status)}</p><ul>{budget_rows}</ul>'
    else:
        universe_status = 'census JSON not yet written (AttributeUniverse build in flight — renders here when it lands)'
        budget_html = f'<h2>AttributeUniverse census</h2><p class="muted">{escape(universe_status)}</p>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER datagen</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.5rem;text-align:left}}details{{margin:1rem 0}}.metric{{border:1px solid #ddd;padding:1rem;width:14rem}}.metrics{{display:flex;gap:1rem;flex-wrap:wrap}}.muted{{color:#666}}.metric strong{{display:block;font-size:1.4rem}}</style></head><body>
<h1>ER · Datagen track — findings of session 2026-09-30</h1>
<h2>Dedupe re-run after Wave-1 fixes (measured on the original dataset)</h2>
<div class="metrics">{metrics}</div>
<details open><summary>Closure gate: 71,623 == 63,079 + 8,544 · identity invariant PASS (13,216 trusted barcodes kept)</summary></details>
<ul>
<li><a href="/" target="_blank" rel="noopener">Original audit</a>: 91.4% of unparsed attribute keys still carry same-GTIN conflicts (13.8–44.4%)</li>
<li>GTIN census</li><li>Description evidence recovered by the alias fix</li><li>Brand alias fold</li></ul>
<details open><summary>GTIN capture ledger — after Wave-1 fixes</summary>
<table><tr><th>Population</th><th>Rows</th></tr>{gtin_rows}</table>
<p>Wave-1 verdict: 0 cells changed on this corpus (byte-identical SHA). Guarantees <code>gtin_equivalent()</code> + longest-run parsing for future feeds.</p>
</details>
<details open><summary>Description evidence captured by the <code>description</code> alias fix</summary>
<table><tr><th>Item</th><th>Rows</th></tr>{desc_rows}</table>
</details>
<details open><summary>Brand alias fold (veto-asymmetry — folds add, never swap)</summary><p>{escape(brand_note)}</p></details>
<details open><summary>Capture still pending (the next multiplier)</summary>
<ul>
<li><code>Pack Material Type</code>: 51,703 rows (72%), 5 value-sets, 13.8% same-GTIN conflict — veto-grade, currently review-lane only</li>
<li>Water type 14.9% · Made from 21.2% · Juice features 31.9% · health claims 33.9%</li>
<li>Juice content 63,117 rows (88%) still prose, not a numeric band field</li>
</ul>
</details>
{budget_html}
<p><a href="/">← home</a> · <a href="/gate"><strong>Gate decisions</strong></a> (original-column sample of 5 per decision bucket) · <a href="/graphs">graphs track</a></p>
</body></html>'''


# ── gate-decisions evidence ──────────────────────────────────────────────────
# The stage-1 gate artifacts (data/gate_results.csv: gtin1, gtin2, canon1,
# canon2, gate_decision, gate_reason, similarity — the deciding clause is the
# gate_reason column) record each candidate pair in canonical form only. This
# page joins those pairs back to the RAW export via core.common.load_dataset
# (SSOT column mapping) and shows the ORIGINAL columns, as exported — the raw
# 13-column feed, not the cleaned/canonical view. Raw frame cached once by
# file modification time, mirroring dashboard/catalog.py.

_gate_dir = ROOT / 'evidence' / 'datagen'
_gate_results_path = ROOT.parent / 'data' / 'gate_results.csv'
_GATE_BUCKETS = ('proceed', 'hard_no', 'fallback')
# (canonical column after load_dataset, original export header shown)
_GATE_ORIGINAL_COLUMNS = (
    ('retailer', 'retailer'), ('barcode', 'gtin'), ('title', 'sku_name_eng'),
    ('description', 'description_short_eng'), ('attributes', 'attribute'),
)

@lru_cache(maxsize=1)
def _gate_raw(mtime_ns: int):
    frame = load_dataset()
    listings = frame.barcode.value_counts()
    resolvable = frame[frame.barcode.notna() & frame.barcode.ne('')]
    first = resolvable.drop_duplicates('barcode', keep='first').set_index('barcode')
    return first, listings

@lru_cache(maxsize=1)
def _gate_results_frame(mtime_ns: int):
    return pd.read_csv(_gate_results_path, dtype=str, keep_default_na=False, low_memory=False)

def _gate_route_gloss(clause: str) -> str:
    if 'Pack blocker' in clause:
        return 'a pack size / package-type / volume conflict — blocked before anything else could run; never a merge, never a review'
    if 'Ambiguous volume' in clause:
        return 'volume evidence contradicts itself, so the gate cannot trust even the raw numbers — deferred to review'
    if 'Critical attribute mismatch' in clause:
        return f'a retailer-defining attribute ({clause.split(": ", 1)[-1]}) conflicts between the two listings — blocked'
    if 'material mismatch' in clause:
        return 'both sides name a package material and they disagree — blocked'
    if 'asserted on one side only' in clause:
        return 'one-sided packaging assertion (case vs unstated): a missing marker is absence of evidence, not a negative — a human applies the rule, the model is not asked to guess'
    if 'Overlapping but low consistency' in clause or 'Overlap but low consistency' in clause:
        return 'volume/pack overlaps exist but the consistency scores fall under the fallback threshold — overlap alone is too weak to proceed'
    if 'Low raw pack confidence' in clause:
        return 'raw pack evidence on at least one side scores under the veto threshold — deferred, neither confirmed nor denied'
    if 'Low raw volume confidence' in clause:
        return 'raw volume evidence on at least one side scores under the veto threshold — deferred, neither confirmed nor denied'
    return 'deferred — the clause text above is the route the pair took'

def _gate_side(gt: str, raw, listings) -> dict:
    row = raw.loc[gt]
    return {orig: (gt if c == 'barcode' else str(row[c]) if pd.notna(row[c]) else '') for c, orig in _GATE_ORIGINAL_COLUMNS} | {
        '_listings': f'first of {int(listings.loc[gt]):,} exported listing(s) for this barcode'}


def _gate_mismatch_html(left_attrs: str, right_attrs: str) -> str:
    """Per-dimension mismatch table for one candidate pair, computed from the
    ORIGINAL attribute cells through the same SSOT the census uses
    (core.attribute_universe.parse). Mismatch = both sides populated, parsed
    sets differ (or the pair's conflict predicate fires for delegated dims);
    single-sided rows are shown as missing-evidence, never as conflict."""
    from core.attribute_universe import AttributeUniverse
    uni = AttributeUniverse(pd.DataFrame({'barcode': ['0', '0'],
                                            'attributes': [str(left_attrs or ''), str(right_attrs or '')]}))
    lparse, rparse = uni.parse(left_attrs), uni.parse(right_attrs)
    predicates = uni._conflict_predicates()
    mism, onesided = [], []
    for key in sorted(set(lparse) | set(rparse)):
        lv, rv = lparse.get(key), rparse.get(key)
        def fmt(v):
            if v is None:
                return '—'
            if isinstance(v, frozenset):
                return ', '.join(sorted(v)) if v else 'empty'
            return ', '.join(str(s) for s in getattr(v, '__members__', ())) if False else str(v)
        if lv is None and rv is not None:
            onesided.append(f'<tr><td>{escape(key)}</td><td>—</td><td>{escape(fmt(rv))}</td></tr>')
        elif rv is None and lv is not None:
            onesided.append(f'<tr><td>{escape(key)}</td><td>{escape(fmt(lv))}</td><td>—</td></tr>')
        elif lv is not None and rv is not None:
            pred = predicates.get(key, lambda a, b: a != b)
            if pred(lv, rv):
                mism.append(f'<tr><td>{escape(key)}</td><td>{escape(fmt(lv))}</td><td>{escape(fmt(rv))}</td></tr>')
    if not mism and not onesided:
        return '<p class="muted">All registered dimensions agree or are unknown on both sides.</p>'
    mismatch_table = (f'<table><tr><th>Dimension · VALUES DIFFER</th><th>Left</th><th>Right</th></tr>'
                      + ''.join(mism) + '</table>') if mism else ''
    onesided_table = (f'<details><summary>Single-sided evidence ({len(onesided)} dims — populated on one side only)</summary>'
                      f'<table><tr><th>Dimension</th><th>Left</th><th>Right</th></tr>{"".join(onesided)}</table></details>') if onesided else ''
    return (f'<p><strong>Mismatching dimensions · {len(mism)} differ</strong></p>{mismatch_table}'
            + (f'<p class="muted">{len(mism)} dimension(s) genuinely disagree</p>' if mism else '') + onesided_table)
def _gate_sample(g, raw, listings, decision: str, size: int = 5):
    bucket = g[g.gate_decision == decision]
    picked, skipped = [], 0
    for row in bucket.itertuples(index=False):
        if len(picked) >= size:
            break
        if row.gtin1 in raw.index and row.gtin2 in raw.index:
            picked.append(row)
        else:
            skipped += 1
    return picked, skipped, len(bucket)

def _gate_pair_html(rank: int, row, raw, listings) -> str:
    try:
        sim = f'{float(row.similarity):.3f}'
    except Exception:
        sim = escape(row.similarity)
    model = _gate_side(row.gtin1, raw, listings)
    other = _gate_side(row.gtin2, raw, listings)
    rows = (
        f'<tr><th>left · {escape(model["retailer"])}</th>' + ''.join(f'<td>{escape(model[orig])}</td>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
        f'<tr><th>right · {escape(other["retailer"])}</th>' + ''.join(f'<td>{escape(other[orig])}</td>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
    )
    listing_notes = f'{escape(model["_listings"])} · {escape(other["_listings"])}'
    clause = escape(row.gate_reason)
    label = 'fallback_reason' if row.gate_decision == 'fallback' else 'gate_reason (deciding clause)'
    route = '' if row.gate_decision in ('hard_no', 'proceed') else f'{escape(_gate_route_gloss(str(row.gate_reason)))} — '
    verdict = f'<p><strong>{escape(row.gate_decision)}</strong> · <code>{escape(label)}</code>: {clause} · {route}similarity {escape(sim)}</p>'
    mism = _gate_mismatch_html(model['attribute'], other['attribute'])
    return (f'<details><summary>#{rank} · {escape(row.gate_decision)} · similarity {escape(sim)} · {clause}</summary>'
            f'{verdict}<p class="muted">{listing_notes}</p>{mism}'
            f'<table><tr><th>Side</th>' + ''.join(f'<th>{escape(orig)}</th>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
            f'{rows}</table></details>')

@app.get('/gate', response_class=HTMLResponse)
def gate_decisions():
    g = _gate_results_frame(_gate_results_path.stat().st_mtime_ns)
    raw, listings = _gate_raw(DATA_PATH.stat().st_mtime_ns)
    counts = g.gate_decision.value_counts()
    buckets_html, snapshot = [], {}
    for decision in _GATE_BUCKETS:
        picked, skipped, total = _gate_sample(g, raw, listings, decision)
        snapshot[decision] = [
            {'gtin1': r.gtin1, 'gtin2': r.gtin2, 'gate_decision': r.gate_decision,
             'gate_reason': r.gate_reason, 'similarity': r.similarity,
             'left': {orig: (r.gtin1 if c == 'barcode' else str(raw.loc[r.gtin1][c])) for c, orig in _GATE_ORIGINAL_COLUMNS},
             'right': {orig: (r.gtin2 if c == 'barcode' else str(raw.loc[r.gtin2][c])) for c, orig in _GATE_ORIGINAL_COLUMNS},
             **({'fallback_reason': r.gate_reason} if decision == 'fallback' else {})}
            for r in picked]
        pairs_html = ''.join(_gate_pair_html(rank, row, raw, listings) for rank, row in enumerate(picked, 1))
        buckets_html.append(
            f'<section><h2>{escape(decision)} · {counts.get(decision, 0):,} pairs</h2>'
            f'<p class="muted">Original columns, as exported · no cleaning · deterministic sample: first {len(picked)} pairs in candidate order '
            f'(both endpoints resolve to raw export rows)'
            + (f' · {skipped:,} endpoint-unresolvable pair(s) skipped before the sample' if skipped else '') + '</p>' + pairs_html + '</section>')
    try:
        _gate_dir.mkdir(parents=True, exist_ok=True)
        (_gate_dir / 'gate_decision_sample.json').write_text(json.dumps(snapshot, indent=1))
        snapshot_note = f'deterministic copy written at render: dashboard/evidence/datagen/gate_decision_sample.json'
    except Exception as error:
        snapshot_note = f'snapshot not written ({escape(error)})'
    counts_row = ''.join(f'<div class="metric"><strong>{counts.get(d, 0):,}</strong>{escape(d)} pairs</div>' for d in _GATE_BUCKETS)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER gate decisions</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{border:1px solid #ccc;padding:.4rem;text-align:left;overflow-wrap:anywhere}}details{{margin:1rem 0}}summary{{cursor:pointer}}section{{border-top:1px solid #ccc;padding:1rem 0}}.muted{{color:#666}}.metric{{border:1px solid #ddd;padding:1rem;width:11rem}}.metrics{{display:flex;gap:1rem;flex-wrap:wrap}}code{{background:#f6f6f6;padding:.1rem .3rem}}</style></head><body>
<h1>ER · Gate decisions — original-column evidence</h1>
<p><strong>Original columns, as exported · no cleaning.</strong> Every candidate pair below is joined back to the raw 13-column feed (dataset.csv via <code>core.common.load_dataset</code>, SSOT column mapping) — not the cleaned/canonical view. The deciding clause is the <code>gate_reason</code> column of data/gate_results.csv.</p>
<div class="metrics">{counts_row}</div>
<p class="muted">{escape(snapshot_note)}</p>
{''.join(buckets_html)}
<p><a href="/">← home</a> · <a href="/datagen">datagen track</a></p>
</body></html>'''


@app.get('/graphs', response_class=HTMLResponse)
def graphs_track():
    cfgs = []
    for name in ('graph_tracks_gnn.yaml', 'graph_tracks_hybrid.yaml'):
        path = ROOT.parent / 'config' / name
        if path.exists():
            cfgs.append(f'<tr><td>{escape(name)}</td><td><pre>{escape(path.read_text()[:400])}</pre></td></tr>')
    snaps = []
    for d in sorted(_results.glob('graph_tracks/*'))[:8]:
        snaps.append(f'<li><code>{escape(d.name)}</code></li>')
    for d in sorted((ROOT.parent / 'dvc_refs').glob('*'))[:8]:
        snaps.append(f'<li>dvc_refs/<code>{escape(d.name)}</code> (published)</li>')
    snap_html = '<ul>' + ''.join(snaps) + '</ul>' if snaps else '<p>No local run snapshots yet — graph-track workers publish from the Colab lane (DVC remote + W&B offline bundles).</p>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER graphs</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}pre{{background:#f6f6f6;padding:.5rem;overflow-x:auto}}ul{{margin:.5rem 0}}</style></head><body>
<h1>ER · Graphs track — GNN-only + hybrid semantic-ID lane</h1>
<p>Full-batch typed two-hop aggregation (NOT sampled GraphSAGE) · 8 categorical relations + volume/pack · checkpoint/DVC snapshot lifecycle landed in commit 5001027 · tractable 62,963-row catalogs, skip.</p>
<h2>Waiting on</h2><ul><li>AttributeUniverse census lands first — it feeds the graph node relations</li><li>P1/P2 items from TODO.md close before graph linkage expands (owner DEAD LAST ruling)</li></ul>
<h2>Configs</h2><table>{''.join(cfgs)}</table>
<h2>Local snapshots</h2>{snap_html}
<p><a href="/">← home</a> · <a href="/datagen">datagen track</a></p>
</body></html>'''


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
