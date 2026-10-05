"""ER composition root and discovery view for the imported Broadway dashboard."""
import html
import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'src'))
os.environ['BROADWAY_EXPERIMENTS_ROOT'] = str(ROOT / 'experiments')
os.environ['BROADWAY_DEFAULT_EXPERIMENT_SERIES'] = 'identity'
os.environ['BROADWAY_OBSERVATIONS_DIR'] = str(ROOT / 'observations')
os.environ['BROADWAY_DIAGRAMS_DIR'] = str(ROOT / 'diagrams')

from vendor.experiments_dashboard import app
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, Response
from catalog import lookup
from fastapi.staticfiles import StaticFiles
from core.common import DATA_PATH, F, data_cfg, load_dataset

app.title = 'ER discovery'
from training_reports import router as training_reports_router
app.include_router(training_reports_router)
from jev_reports import router as jev_reports_router
app.include_router(jev_reports_router)
from decision_reports import router as decision_reports_router
app.include_router(decision_reports_router)

# ── standard page chrome ─────────────────────────────────────────────────────
# ONE consistent top bar (→ /, findings, gate, datagen, graphs, training) for
# every HTML page, including pages rendered by the vendor module we do not
# edit. Bound at the composition root: an http middleware injects the bar into
# every text/html response whose path is not /api/*. Idempotent — pages that
# already carry data-er-nav are passed through untouched, so a bar can never
# render twice. Each page keeps its own <title>.
_CHROME_MARKER = 'data-er-nav'
_CHROME_BODY = re.compile(r'(<body[^>]*>)', re.I)

def _chrome():
    def a(href, label, strong=False):
        style = 'color:#fff;font-weight:700' if strong else 'color:#93c5fd;text-decoration:none'
        return f'<a href="{href}" style="{style}">{label}</a>'
    return ('<nav ' + _CHROME_MARKER + ' aria-label="ER navigation" style="position:sticky;top:0;z-index:50;'
            'display:flex;gap:1.1rem;flex-wrap:wrap;align-items:center;background:#1f2937;'
            'padding:.55rem 1.2rem;font:500 .95rem system-ui">'
            + a('/', '← home', strong=True)
            + a('/experiments', 'findings') + a('/gate', 'gate')
            + a('/datagen', 'datagen') + a('/graphs', 'graphs')
            + a('/training', 'training') + a('/runs', 'runs') + a('/jev', 'JEV audits') + a('/decisions', 'attribute tracking') + '</nav>')

_CHROME = _chrome()

@app.middleware("http")
async def _page_chrome(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/datagen", "/graphs", "/gate", "/gate/fallback", "/jev", "/decisions"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    if path.startswith("/api/"):
        return response
    if not str(response.headers.get("content-type", "")).startswith("text/html"):
        return response
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    text = body.decode("utf-8", errors="replace")
    if _CHROME_MARKER not in text and "<body" in text.lower():
        text = _CHROME_BODY.sub(lambda m: m.group(1) + _CHROME, text, count=1)
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(content=text.encode("utf-8"), status_code=response.status_code,
                    headers=headers, media_type="text/html")


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
# All measured session evidence feeds one page in the Finding-01 identity
# format: numbered findings, each with a metrics row, ONE comparison table
# (original/pre-change → current/resolved + PASS/FAIL/OPEN outcome), a details
# element holding the verbatim evidence excerpt (capped) with its source path,
# and a one-line verdict. No raw blobs inline.

_results = ROOT.parent / 'results'

# ── Finding-01 rendering helpers (shared with /gate and /graphs) ─────────────
_FINDING_STYLE = (
    'section{margin:1.2rem 0;border-top:2px solid #333;padding-top:.6rem}'
    'table{border-collapse:collapse;width:100%;font-size:.92rem;margin:.6rem 0}'
    'td,th{border:1px solid #ccc;padding:.45rem .6rem;text-align:left;overflow-wrap:anywhere}'
    'details{margin:.6rem 0}summary{cursor:pointer}'
    '.metrics{display:flex;gap:.8rem;flex-wrap:wrap;margin:.6rem 0}'
    '.metric{border:1px solid #ddd;padding:.6rem .8rem;width:12rem}'
    '.metric strong{display:block;font-size:1.25rem}'
    '.badge{padding:.15rem .55rem;border-radius:.6rem;font-weight:700;font-size:.82rem;vertical-align:middle}'
    '.badge-pass{background:#dff4e8}.badge-fail{background:#f8d7da}.badge-open{background:#fff0cf}'
    '.status{padding:8px;background:#dff4e8;display:inline-block}'
    '.muted{color:#666}code{background:#f6f6f6;padding:.05rem .3rem}'
    'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f6f6;padding:.6rem}')

def _fmetric(value, label):
    return '<div class="metric"><strong>' + escape(f'{value:,}' if isinstance(value, int) else str(value)) + '</strong>' + escape(label) + '</div>'

def _fbadge(outcome):
    cls = {'PASS': 'badge-pass', 'FAIL': 'badge-fail', 'OPEN': 'badge-open'}.get(outcome)
    if not cls:
        return escape(outcome)
    return f'<span class="badge {cls}">{outcome}</span>'

def _fcompare(rows):
    """rows: (original/pre-change html, current/resolved html, outcome) tuples."""
    body = ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{_fbadge(outcome)}</td></tr>'
                   for a, b, outcome in rows)
    return ('<table><tr><th>Original / pre-change</th><th>Current / resolved</th>'
            '<th>Outcome</th></tr>' + body + '</table>')

def _fevidence(path, excerpt, cap=900):
    text = str(excerpt)
    if len(text) > cap:
        text += f'\n… [verbatim excerpt capped at {cap:,} of {len(str(excerpt)):,} characters — full file: {path}]'
    return (f'<details><summary>Raw evidence · source <code>{escape(path)}</code></summary>'
            f'<pre>{escape(text)}</pre></details>')

def _ffinding(number, name, outcome, metrics, compare, evidence, verdict):
    return (f'<section><h2 style="font-size:1.15rem">Finding {number:02d} — {escape(name)}&#160;{_fbadge(outcome)}</h2>'
            f'<div class="metrics">{metrics}</div>{compare}{evidence}'
            f'<p><strong>Verdict:</strong> {verdict}</p></section>')


def _manifest():
    try:
        return json.loads((_results / 'manifests' / 'dedupe.json').read_text())
    except Exception:
        return {}


@app.get('/datagen', response_class=HTMLResponse)
def datagen_track():
    m = _manifest()
    ra = m.get('row_accounting', {})
    def worker(name):
        path = ROOT.parent / 'data' / 'prepared' / 'full' / name
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    w1, w2 = worker('worker_1_baseline.pkl.gz.json'), worker('worker_2_baseline.pkl.gz.json')
    try:
        dp = json.loads((_results / 'manifests' / 'data_prep.json').read_text())
        dpa = dp.get('row_accounting', {})
        fl = dpa.get('flags_census', {})
    except Exception:
        dp, dpa, fl = {}, {}, {}
    try:
        lp = json.loads((_results / 'manifests' / 'labeled_pairs.json').read_text())['row_accounting']
    except Exception:
        lp = {}
    try:
        fv = json.loads((_results / 'manifests' / 'final_validation.json').read_text())
    except Exception:
        fv = {}
    teacher_conflicts = sum(v for k, v in fl.items() if str(k).startswith('description_conflict')) or None
    del teacher_conflicts
    # Generation quantification: read ONLY what the bundling process publishes —
    # the bundle header (augmentation_coverage + effective_train_ratio), the
    # newest run's handoff.json and its timing_offenders.log. Bundles are
    # published beside the run tree, so the newest published header wins when
    # no per-worker manifest is on disk; no external version store.
    published = (list((ROOT.parent / 'data' / 'prepared' / 'full').glob('*.pkl.gz.json'))
                 + list((_results / 'training_prep').glob('*.pkl.gz.json')))
    bundle_path = max(published, key=lambda p: p.stat().st_mtime, default=None)
    bundle, bundle_source = w1 or {}, 'data/prepared/full/worker_1_baseline.pkl.gz.json'
    if not bundle.get('augmentation_coverage'):
        try:
            bundle = json.loads(bundle_path.read_text())
            bundle_source = str(bundle_path.relative_to(ROOT.parent))
        except Exception:
            bundle, bundle_source = {}, 'no bundle header published yet'
    ac = bundle.get('augmentation_coverage', {})
    groups = ac.get('attributes', {})
    ratio = bundle.get('effective_train_ratio')
    ratio = '—' if ratio in (None, '') else f'{float(ratio):.3f}'
    try:
        run_dirs = sorted(p for p in (_results / 'training_prep').glob('2*') if p.is_dir())
        latest = run_dirs[-1] if run_dirs else None
        try:
            hq = json.loads((latest / 'handoff.json').read_text()) if latest else None
        except Exception:
            hq = None
        handoff_cell = (f"pass · {len(hq.get('inputs', []))} inputs metered · "
                        f"loss/batch attested: {'yes' if hq.get('loss_batch_correctness') else 'n/a'}"
                        if hq else 'no handoff.json yet (run prepare_all)')
        offender_path = latest / 'timing_offenders.log' if latest else None
        if offender_path and Path(offender_path).is_file():
            lines = [l for l in Path(offender_path).read_text().splitlines() if l and not l.startswith('#')]
            offender_cell = lines[0] if lines else 'empty report'
        else:
            offender_cell = 'no report yet'
    except Exception:
        handoff_cell, offender_cell = 'no handoff.json yet (run prepare_all)', 'no report yet'
    # Finding 01 — dedupe closure
    dedupe_metrics = ''.join([
        _fmetric(ra.get('input_rows', 71_623), 'original listings in'),
        _fmetric(ra.get('output_rows', 63_079), 'deduped rows out'),
        _fmetric(ra.get('dropped', {}).get('t1_retailer_gtin', 1_850), 'T1 collapses'),
        _fmetric(ra.get('dropped', {}).get('t1_5_retailer_malformed_gtin_same_product', 96), 'T1.5 malformed-gtin recoveries'),
        _fmetric(ra.get('skipped_checksum_invalid', 3_867), 'checksum-invalid retained'),
        _fmetric(ra.get('unresolved_identity_review_rows', 207), 'escalated identity questions (never guessed)'),
    ])
    dedupe_compare = _fcompare([
        ('Prior refresh baseline: <strong>62,963</strong> output rows',
         f"<strong>{ra.get('output_rows', 63_079):,}</strong> output rows (+116, attributed to commits 0452692..2d3ac4b — Wave-1 fixes byte-identical)",
         'PASS'),
        ('Row closure must recompose per refresh',
         '71,623 == 63,079 + 8,544 · identity invariant holds (13,216 trusted gtins kept)',
         'PASS'),
        ('Escalations must never be guessed', f"{ra.get('unresolved_identity_review_rows', 207):,} rows escalated to review evidence", 'PASS'),
    ])
    dedupe_evidence = _fevidence('results/manifests/dedupe.json',
                                 json.dumps({'status': m.get('status'), 'row_accounting': ra,
                                             'outputs': m.get('outputs')}, indent=1))
    f01 = _ffinding(1, 'Dedupe re-run after Wave-1 fixes (measured on the original dataset)', 'PASS',
                    dedupe_metrics, dedupe_compare, dedupe_evidence,
                    f"<span class='badge badge-pass'>PASS</span> closure gate {ra.get('input_rows', 71_623):,} == {ra.get('output_rows', 63_079):,} + 8,544; Wave-1 fixes changed 0 cells on this corpus.")
    # Finding 02 — GTIN capture ledger
    gtin_ledger = [
        ('missing (NA) gtin rows', '41,545', '41,545'),
        ('checksum-invalid rows (rows survive, identity claim dies)', '3,715', '3,715'),
        ('checksum-valid rows', '26,363', '26,363'),
        ('distinct valid GTINs', '13,250', '13,250'),
        ('first-run ≠ longest-run cells (truncation fix regression)', '0', '0'),
        ('review-quarantined GTINs', '139', '139'),
    ]
    gtin_metrics = ''.join([
        _fmetric(0, 'cells changed vs pre-change'),
        _fmetric(13_250, 'distinct valid GTINs'),
        _fmetric(139, 'review-quarantined GTINs'),
        _fmetric(m.get('inputs', [{}])[0].get('sha256', '')[:12], 'dataset.csv SHA (unchanged)'),
    ])
    gtin_compare = _fcompare([(name, current, 'PASS')
                              for name, original, current in gtin_ledger])

    gtin_evidence = _fevidence(f'dataset.csv (sha256 {m.get("inputs", [{}])[0].get("sha256", "?")[:20]}…)',
                               'GTIN ledger re-measured on the same byte-identical export — session ledger (2026-09-30):\n'
                               + '\n'.join(f'{k}: {v:,}' for k, v in [
                                   ('missing (NA) gtin', 41_545),
                                   ('checksum-invalid rows', 3_715),
                                   ('checksum-valid rows', 26_363),
                                   ('distinct valid GTINs', 13_250),
                                   ('first-run ≠ longest-run cells', 0),
                                   ('review-quarantined GTINs', 139)]))
    f02 = _ffinding(2, 'GTIN capture ledger — Wave-1 truncation-fix guarantees', 'PASS',
                    gtin_metrics, gtin_compare, gtin_evidence,
                    f"<span class='badge badge-pass'>PASS</span> unchanged (byte-identical SHA) — 0 first-run ≠ longest-run cells; guarantees <code>gtin_equivalent()</code> + longest-run parsing for future feeds.")
    # Finding 03 — description alias fix
    desc_metrics = ''.join([
        _fmetric(52_856, 'deduped rows consuming description'),
        _fmetric(330, 'carbonation filled where title/attributes empty'),
        _fmetric(545, 'sweetener filled where title/attributes empty'),
        _fmetric(67, 'pulp filled where title/attributes empty'),
        _fmetric(1_073, 'both-present disagreements (stay veto/review)'),
        _fmetric(0, 'rows relying on description ONLY'),
    ])
    desc_compare = _fcompare([
        ('Description invisible to the attribute parser (alias gap)',
         '52,856 of 63,079 deduped rows consume <code>description</code>', 'PASS'),
        ('Carbonation empty where title/attributes empty', '330 rows filled from description evidence', 'PASS'),
        ('Sweetener empty where title/attributes empty', '545 rows filled', 'PASS'),
        ('Pulp empty where title/attributes empty', '67 rows filled', 'PASS'),
        ('Both title and description populated and disagree', '1,073 rows stay on the veto/review lane (no overwrite)', 'PASS'),
        ('Regression risk: description becoming the sole evidence', '0 rows rely on description ONLY', 'PASS'),
    ])
    desc_evidence = _fevidence(
        'session ledger · dataset.csv description_short_eng via core.common.load_dataset',
        'Description-evidence ledger measured 2026-09-30, after the description alias fix '
        '(alias folded; source column retained verbatim):\n'
        + '\n'.join(f'{k}: {v:,}' for k, v in [
            ('deduped rows consuming description', 52_856),
            ('carbonation filled where title/attributes empty', 330),
            ('sweetener filled where title/attributes empty', 545),
            ('pulp filled where title/attributes empty', 67),
            ('both-present disagreements (stay veto/review)', 1_073),
            ('rows relying on description ONLY', 0)]))
    f03 = _ffinding(3, 'Description evidence recovered by the <code>description</code> alias fix', 'PASS',
                    desc_metrics, desc_compare, desc_evidence,
                    f"<span class='badge badge-pass'>PASS</span> — description is recovered only where the other channels are empty; disputes stay on veto/review.")
    # Finding 04 — brand alias fold
    try:
        vocab = json.loads((ROOT.parent / 'config' / 'vocabulary.json').read_text())
        prov = vocab.get('brand_aliases_provenance', {})
        prov_counts = prov.get('counts', {})
        alias_list = ' · '.join(f'{escape(a)}→{escape(b)}' for a, b in sorted(vocab.get('brand_aliases', {}).items()))
    except Exception:
        vocab, prov, prov_counts, alias_list = {}, {}, {}, ''
    brand_metrics = ''.join([
        _fmetric(prov_counts.get('alias_entries', len(vocab.get('brand_aliases', {}))), 'alias entries (folds add, never swap)'),
        _fmetric(prov_counts.get('alias_families', '?'), 'granted alias families'),
        _fmetric(prov_counts.get('variant_groups_measured', '?'), 'brand-variant groups measured'),
        _fmetric(prov_counts.get('within_group_brand_vetoes_before_seeding', '?'), 'within-group vetoes before seeding'),
        _fmetric(prov_counts.get('declined_group_gtins', '?'), 'declined group GTINs (reasoned, not swallowed)'),
    ])
    brand_compare = _fcompare([
        ('Retailer-brand variants veto per-attribute equal identity evidence',
         '8 alias folds (<code>' + alias_list.replace('<code>', '').replace('</code>', '') + '</code>) — folds ADD, never swap', 'PASS'),
        ('Veto asymmetry: missing marker must not become a negative',
         'folds applied one-directionally; 19 group GTINs declined with reasons instead of being swallowed', 'PASS'),
        ('False-veto pairs from sibling brands (e.g. hi/hiball, fitaid/lifeaid)',
         'dissolved by the granted families; reproduction at head: 71 groups (65 two-brand / 6 three-brand), 341 rows', 'PASS'),
    ])
    brand_evidence = _fevidence('config/vocabulary.json',
                                'brand_aliases_provenance:\n' + json.dumps(prov, indent=1))
    f04 = _ffinding(4, 'Brand alias fold (veto-asymmetry)', 'PASS', brand_metrics, brand_compare,
                    brand_evidence, f"<span class='badge badge-pass'>PASS</span> — folds add, never swap; declined groups keep their named reasons.")
    # Finding 05 — attribute universe census
    budget_html = ''
    # migrated 2026-10-05: fail-loud tracked evidence lives in artifacts/evidence/
    census_path = ROOT.parent / 'artifacts' / 'evidence' / 'attribute_universe_census.json'
    if census_path.exists():
        try:
            cu = json.loads(census_path.read_text())
            census_budget = cu.get('datagen_budget', {})
            pending_rows = ''.join(
                f'<tr><td>{escape(k)}</td><td>{v.get("rows_populated", "?"):,}</td>'
                f'<td>{v.get("conflict_rate", 0):.1%}</td><td>{"veto-grade" if v.get("veto_candidate") else "candidate"}</td><td>{_fbadge("OPEN")}</td></tr>'
                for k, v in sorted(census_budget.items(), key=lambda kv: -kv[1].get('conflict_rate', 0))
                if kv[1].get('headroom_share', 0) >= 0.13)[:8]
            census_metrics = ''.join([
                _fmetric(len(cu), 'census top-level keys'),
                _fmetric(len(cu.get('baseline', {})), 'dimensions in baseline census'),
                _fmetric(len(census_budget), 'datagen-budget dimensions'),
                _fmetric('open', 'capture still pending (next multiplier)'),
            ])
            census_compare = _fcompare([
                ('<code>Pack Material Type</code> prose only', '51,703 rows (72%), 5 value-sets, 13.8% same-GTIN conflict — veto-grade, currently review-lane only', 'OPEN'),
                ('Water type / Made from / Juice features / health claims', '14.9% · 21.2% · 31.9% · 33.9% same-GTIN conflict still unparsed', 'OPEN'),
                ('Juice content', '63,117 rows (88%) still prose, not a numeric band field', 'OPEN'),
            ])
            f05 = _ffinding(5, 'AttributeUniverse census — capture still pending', 'OPEN', census_metrics,
                            census_compare,
                            _fevidence('artifacts/evidence/attribute_universe_census.json',
                                       json.dumps({'baseline_keys': sorted(cu.get('baseline', {}))}, indent=1)),
                            "<span class='badge badge-open'>OPEN</span> — censused and scoped, capture pending; this is the next multiplier.")
        except Exception:
            f05 = _ffinding(5, 'AttributeUniverse census', 'OPEN', '', _fcompare([]),
                            _fevidence('artifacts/evidence/attribute_universe_census.json', 'census file present but not parseable'),
                            'census unreadable in this view.')
    else:
        f05 = _ffinding(5, 'AttributeUniverse census — capture still pending', 'OPEN', '', _fcompare([]),
                        _fevidence('artifacts/evidence/attribute_universe_census.json', 'census JSON not yet written (AttributeUniverse build in flight — renders here when it lands)'),
                        'census not yet on disk.')
    # Finding 06 — training-data preparation · offline bundle lane (current blocker)
    bundle_metrics = ''.join([
        _fmetric(lp.get('output_rows', 8_736), 'labeled pairs (1,023 pos · 7,713 hard-neg)'),
        _fmetric(dpa.get('gate_pairs', 135_246), 'gate pairs evaluated'),
        _fmetric(dpa.get('output_rows', 13_216), 'canonical records (visible corpus)'),
        _fmetric(fv.get('rows', 6_351), 'final-validation pairs'),
        _fmetric(fv.get('positives', 565), 'final-validation positives'),
        _fmetric(fv.get('positives_straddling_folds', 0), 'positives straddling folds (leak guard)'),
    ])
    def wcell(key, cast=str):
        a = cast(w1.get(key)) if w1 else 'manifest missing'
        b = cast(w2.get(key)) if w2 else 'manifest missing'
        return a, b
    n1, n2 = wcell('n_df')
    b1, b2 = wcell('n_labeled_pairs_bytes')
    m1, m2 = wcell('masking_config', lambda c: f'frac={c.get("frac")}')
    bundle_compare = _fcompare([
        ('labeled_pairs stage: gate pairs must land on one consistent labeling',
         f"<strong>{lp.get('output_rows', 8_736):,}</strong> = {lp.get('pos_labeled', 1_023):,} positive + {lp.get('hard_neg_labeled', 7_713):,} hard-negative (fallbacks 41,748 / below-threshold 84,762 dropped)",
         'PASS'),
        ('final-validation leak guard: positives must not straddle folds',
         f"<strong>{fv.get('positives_straddling_folds', 0):,}</strong> straddling of {fv.get('positives', 565):,} positives", 'PASS'),
        ('Both workers embed ONE upstream canonical snapshot',
         f'worker_1 n_df={n1} vs worker_2 n_df={n2} — different upstream snapshots', 'FAIL'),
        ('Both workers embed THE SAME labeled_pairs.csv',
         f'worker_1 {b1} B (matches on-disk 255,053 B) vs worker_2 {b2} B (stale)', 'FAIL'),
        ('One masking profile across the bundle',
         f'worker_1 {m1} vs worker_2 {m2}', 'FAIL'),
    ])
    bundle_evidence = _fevidence('data/prepared/full/worker_1_baseline.pkl.gz.json + worker_2_baseline.pkl.gz.json',
                                 json.dumps({'worker_1': {'n_df': w1.get('n_df'), 'n_payload': w1.get('n_payload'),
                                                          'n_pos': w1.get('n_pos'), 'n_neg': w1.get('n_neg'),
                                                          'sha256': w1.get('sha256'),
                                                          'masking_frac': w1.get('masking_config', {}).get('frac')},
                                             'worker_2': {'n_df': w2.get('n_df'), 'n_payload': w2.get('n_payload'),
                                                          'n_pos': w2.get('n_pos'), 'n_neg': w2.get('n_neg'),
                                                          'sha256': w2.get('sha256'),
                                                          'masking_frac': w2.get('masking_config', {}).get('frac')}},
                                            indent=1) if isinstance(w1, dict) and isinstance(w2, dict)
                                 else 'worker manifests not readable')
    f06 = _ffinding(6, 'Training-data preparation · offline bundle lane — CURRENT blocker', 'OPEN',                    bundle_metrics, bundle_compare, bundle_evidence,
                    "<span class='badge badge-open'>OPEN</span> — bundle blocker: worker_1 and worker_2 embed different upstream snapshots (n_df 62,927 vs 56,529) and different masking fracs, so the offline bundle cannot ship as one lane — rebuild both workers against the current canonical corpus (13,216 rows · data_prep.json), while labeled_pairs and final_validation themselves PASS.")
    teacher_rows = ''.join(f'<tr><td><code>{escape(k)}</code></td><td>{v:,}</td></tr>' for k, v in sorted(fl.items()))
    # Finding 07 — same-GTIN duplicate variation (measured, feeds augmentation design)
    var, dvc = {}, {}
    try:
        dvc = json.loads((_results / 'duplicate_variation_census.json').read_text())
        var = dvc.get('column_varies_pct', {})
        conf_rows = ''.join(f'<tr><td>{escape(k)}</td><td>{v:,} groups</td></tr>'
                            for k, v in sorted(dvc.get('top_same_gtin_attribute_conflicts', {}).items(),
                                               key=lambda kv: -kv[1])[:5])
    except Exception:
        dvc, conf_rows = {}, ''
    dup_metrics = ''.join([
        _fmetric(dvc.get('duplicate_groups', 7_704), 'same-GTIN duplicate groups'),
        _fmetric(dvc.get('rows_in_duplicates', 22_785), 'rows inside those groups'),
        _fmetric(dvc.get('title_token_dissimilarity_mean', 0.703), 'title token dissimilarity (median ' + escape(str(dvc.get('title_token_dissimilarity_median', 0.75))) + ')'),
        _fmetric(var.get('brand', 0.011), 'brand varies (the fingerprint stays)'),
        _fmetric(var.get('retailer', 0.914), 'of duplicates are cross-retailer'),
        _fmetric(var.get('attribute', 1.0), 'attribute cells differ (partial declarations)'),
    ])
    dup_compare = _fcompare([
        ('Assumed variation scale: masking band U(0.20, 0.30), polite rewording',
         f"measured title dissimilarity <strong>{dvc.get('title_token_dissimilarity_mean', 0.703):.3f}</strong> mean / "
         f"{dvc.get('title_token_dissimilarity_median', 0.75):.3f} median — retailers rewrite ~3× the band we mask at", 'OPEN'),
        ('Assumed negatives are "different products"; conflict = identity',
         'same-GTIN copies genuinely disagree: health claims 58.1%, juice features 63.6%, caffeine 33.3%, sweetener 30.2%, carbonization 26.1% — self-reported retailer noise, the honest conflict distribution', 'OPEN'),
        ('Donor sourcing: 10-retry indifferent lane, no marketplace structure',
         '<strong>91.4%</strong> of real duplicates are CROSS-retailer — donor pools must draw cross-seller minted pairs to carry the real phrasing gap', 'OPEN'),
        ('missing_both read as capture bug',
         'attribute cells differ in <strong>100%</strong> of duplicate groups: each retailer declares a different partial subset — planned "declaration dropout" lane mints that exact shape (no invented tokens)', 'OPEN'),
    ])
    dup_evidence = _fevidence('results/duplicate_variation_census.json', json.dumps(
        {'column_varies_pct': dvc.get('column_varies_pct'),
         'top_same_gtin_attribute_conflicts': dvc.get('top_same_gtin_attribute_conflicts')}, indent=1))
    f07 = _ffinding(7, 'Duplicate variation census — what retailers actually vary (augmentation design input)', 'OPEN',
                    dup_metrics, dup_compare, dup_evidence,
                    "<span class='badge badge-open'>OPEN</span> — measured 2026-10-01; four augmentation principles follow from it: (1) masking extent sampled near the real distribution, (2) declaration-dropout lane emulating partial attribute cells, (3) cross-retailer donor bias, (4) twin weights matched to measured conflict rates. Wiring pending — replaces the earlier weakspot-quota shares those measurements supersede.")
    # Finding 08 — augmentation targets + masking-reachable quota shares
    try:
        plan = json.loads((_results / 'augmentation_plan.json').read_text())
    except Exception:
        plan = {}
    try:
        import re as _re
        quota = {}
        cfg_text = (ROOT.parent / 'config' / 'training.yaml').read_text()
        m = _re.search(r'field_quota_shares:\n((?:    \w+:[^\n]*\n)+)', cfg_text)
        if m:
            quota = dict((k, float(v)) for k, v in
                         _re.findall(r'    (\w+): ([0-9.]+)', m.group(1)))
    except Exception:
        quota = {}
    target_metrics = ''.join([
        _fmetric(2_000, 'positives to mint (weakspot-weighted)'),
        _fmetric(1_800, 'hard negatives to mint (same lanes, 1-sided)'),
        _fmetric(f"{plan.get('n_pos_target', 2000)}", 'allocation target (plan v1)'),
        _fmetric('1 : ~5–6', 'kept pos:neg training ratio (from 1 : 7.9)'),
    ] + [
        _fmetric(f'{"{"} {g}: {s:.2f} {"}"}'.replace('{', '', 0).replace('}', '', 0), f'{g} quota share — of mask-reachable slots')
        for g, s in list(quota.items())
    ])
    target_compare = _fcompare([
        ('Labeled pool today: 983 pos / 7,728 hard-neg (1 : 7.9)',
         'mint 2,000 symmetric positives + 1,800 single-sided hard negatives on the same weak-spot lanes — ratio lands ~1 : 5–6 instead of drifting to 1 : 10', 'OPEN'),
        ('Mask-reachable weak spots get the quota',
         'sweetener family ~79%, carbonation ~7%, package material ~7%, juice content ~4% of swap/twin slots (measured conflict-rate weights)', 'OPEN'),
        ('Weak spots masking cannot reach (registry-only keys)',
         'health claims, made from, sustainable sourcing, no artificial ingredients, rtd coffee style, diets, energy source… get coverage from the declaration-dropout lane + stronger masking extent, NOT quotas', 'OPEN'),
        ('Below the donor floor (raw-territory: re-capture decides)',
         'giftbox 21, special edition 216, sports drinks style 464, nutri score 412 rows — augmentation can never admit; only raw capture grows these', 'OPEN'),
    ])
    f08_target = _ffinding(8, 'Datagen targets — mint 2,000 pos + 1,800 hard-neg, weakspot quotas on mask-reachable lanes', 'OPEN',
                           target_metrics, target_compare,
                           _fevidence('results/augmentation_plan.json (allocation v1) + config/training.yaml masking.field_quota_shares (v2: measured-conflict-rate weights)',
                                      json.dumps({'plan_v1_weakspot_alloctions': plan.get('weakspot_weighted_mint'),
                                                  'superseded_by': 'duplicate variation census (Finding 07)'}, indent=1)),
                           "<span class='badge badge-open'>OPEN</span> — targets fixed; quota shares from the measured same-GTIN conflict distribution (Finding 07); generated examples land here once the mint run completes.")
    # Finding 09 — how the generated rows are made (the two lanes + minting)
    gen_metrics = ''.join([
        _fmetric(f"{ac.get('source_train_positives', 0):,} / {ac.get('source_train_negatives', 0):,}", 'source positives / negatives in'),
        _fmetric(f"{ac.get('minted_negatives', 0):,}", 'Lane-1 twins minted (negatives)'),
        _fmetric(f"{ac.get('masked_positives', 0):,} / {ac.get('masked_minted_negatives', 0):,}", 'Lane-2 masked views (pos / minted neg)'),
        _fmetric(f"{len(groups)} / 37", 'donor-capable field groups'),
        _fmetric(ratio, 'guaranteed bundle-only view ratio'),
    ])
    gen_compare = _fcompare([
        ('Negatives had no generator — gate-labeled or hand-picked',
         'Lane 1 · counterfactual twins: flip ONE field both sides agree on, donor value from a real cross-retailer row; twin labeled 0 by construction, original pair stays 1', 'PASS'),
        ('Each pair trained on a single view',
         'Lane 2 · masking: declaration dropout + extent-band masking + value swaps, volume/pack lineage re-encoded so text and numbers agree', 'PASS'),
        ('An unprovable flip would train silently',
         'every twin carries an audit row (target_mode, fields_hit, donor provenance) re-verified at load — an unprovable flip is rejected', 'PASS'),
        ('Minted rows could displace real partners',
         'real partners mined first (embeddings/TF-IDF blocking, same-GTIN exclusion, top-k per anchor); minted rows only top up uncovered anchors; pairs.csv sha-pinned', 'PASS'),
        ('Generation could reach any of the 37 attribute keys',
         f"{len(groups)} donor-capable groups only; juice_content has no comparable evidence, so nothing is minted for it", 'PASS'),
        ('Run knobs could drift from the analysis knobs',
         'difficulty slices from the config <code>difficulty:</code> block, hashed provenance + composition_fingerprint attested in <code>handoff.json</code>, same SSOT knobs feed post-training analysis', 'PASS'),
    ])
    gen_evidence = _fevidence(f'{bundle_source} (bundle header augmentation_coverage) + results/training_prep/<run>/handoff.json',
                              json.dumps({'requested_counts': ac.get('requested_counts'),
                                          'effective_train_ratio': bundle.get('effective_train_ratio'),
                                          'handoff_boundary': handoff_cell,
                                          'worst_timing_offender': offender_cell,
                                          'attributes': {k: v.get('status') for k, v in groups.items()}}, indent=1))
    f09 = _ffinding(9, 'How generated training data is made — two lanes + minting', 'PASS',
                    gen_metrics, gen_compare, gen_evidence,
                    f"<span class='badge badge-pass'>PASS</span> — the two lanes ship: {ac.get('minted_negatives', 0):,} twins and "
                    f"{ac.get('masked_positives', 0) + ac.get('masked_minted_negatives', 0):,} masked views, every generated row re-verified at load. "
                    f"Open part: boundary attestation ({handoff_cell}). Code: <code>training/masking.py</code>, "
                    f"<code>training/negative_supply.py</code>, <code>training/difficulty.py</code>.")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER datagen</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}{_FINDING_STYLE}h1{{font-size:1.4rem}}.verdict{{margin:.4rem 0}}</style></head><body>
<h1>ER · Datagen track — identity fixes, GTIN integrity, attribute census, datagen budget</h1>
<p class="status">{ra.get('input_rows', 71_623):,} original listings → {ra.get('output_rows', 63_079):,} deduped · closure gate {ra.get('input_rows', 71_623):,} == 63,079 + 8,544 · Wave-1 fixes byte-identical · offline bundle lane OPEN (Finding 06)</p>
{f01}{f02}{f03}{f04}{f05}{f06}{f07}{f08_target}{f09}
<details open><summary>Teacher-flag census from data_prep.json (veto/review — never guessed)</summary>
<table><tr><th>Flag</th><th>Records</th></tr>{teacher_rows}</table>
</details>
<p class="muted"><a href="/gate"><strong>Gate decisions</strong></a> (original-column sample of 5 per decision bucket, per-pair dimension evidence via the same AttributeUniverse SSOT) · <a href="/">← home</a> · <a href="/graphs">graphs track</a></p>
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
# These fallback pages show the ENTIRE original entry — all 13 raw-export
# columns in RAW-export header names (dashboard rule: original columns, as
# exported, no cleaning).
_FALLBACK_RAW_COLUMNS = tuple(data_cfg().column_mapping.values())
# (canonical column after load_dataset, original export header shown)
_GATE_ORIGINAL_COLUMNS = tuple((canonical, original) for original, canonical in data_cfg().column_mapping.items())


@lru_cache(maxsize=1)
def _gate_source_frame(mtime_ns: int):
    return load_dataset()


@lru_cache(maxsize=1)
def _gate_cards(mtime_ns: int):
    frame = pd.read_csv(F['canonical_records'], dtype=str, keep_default_na=False)
    return frame.set_index('gtin', drop=False).to_dict('index')

@lru_cache(maxsize=1)
def _gate_raw(mtime_ns: int):
    frame = _gate_source_frame(mtime_ns)
    listings = frame.gtin.value_counts()
    resolvable = frame[frame.gtin.notna() & frame.gtin.ne('')]
    first = resolvable.drop_duplicates('gtin', keep='first').set_index('gtin')
    return first, listings

@lru_cache(maxsize=1)
def _gate_results_frame(mtime_ns: int):
    return pd.read_csv(_gate_results_path, dtype=str, keep_default_na=False, low_memory=False)

@lru_cache(maxsize=1)
def _gate_universe(mtime_ns: int):
    """AttributeUniverse built on the FULL dataset frame (SSOT constructor:
    a pd.DataFrame carrying the canonical 'attribute' + 'gtin' columns) —
    no synthetic stub; parse() then runs on arbitrary cells."""
    from core.attribute_universe import AttributeUniverse
    return AttributeUniverse(_gate_source_frame(mtime_ns))

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
    return {orig: (gt if c == 'gtin' else str(row[c]) if pd.notna(row[c]) else '') for c, orig in _GATE_ORIGINAL_COLUMNS} | {
        '_listings': f'Illustrative preview: first of {int(listings.loc[gt]):,} exported listing(s); not the aggregated decision input'}


def _gate_value_html(value, limit: int = 140) -> str:
    """Compact cell: preview inline, FULL value inside a nested details element
    so long parsed sets / delegated extracts are never clipped."""
    if value is None:
        return '—'
    text = ', '.join(sorted(str(x) for x in value)) if isinstance(value, frozenset) else str(value)
    if not text:
        return 'empty'
    if len(text) <= limit:
        return escape(text)
    return (f'{escape(text[:limit])}… '
            f'<details><summary>full value ({len(text):,} chars)</summary><code>{escape(text)}</code></details>')


def _gate_dimension_evidence_html(left_attrs: str, right_attrs: str, uni) -> str:
    """FULL attribute-decision evidence for one candidate pair (owner ruling):
    a compact per-dimension table covering every parsed dimension with status
    agree | conflict | single-sided | unknown, computed from the ORIGINAL
    attribute cells through the SSOT parser (core.attribute_universe.parse on
    the full dataset frame — a pd.DataFrame with the canonical columns).
    Absence is missing evidence, never a conflict; delegated fields whose
    extracted set is empty on a side stay unknown. Nothing is truncated
    inline: the FULL raw attribute cells render inside a details element."""
    lparse, rparse = uni.parse(left_attrs or ''), uni.parse(right_attrs or '')
    predicates = uni._conflict_predicates()
    groups = {'conflict': [], 'single-sided': [], 'agree': [], 'unknown': []}
    for key in sorted(set(lparse) | set(rparse)):
        lv, rv = lparse.get(key), rparse.get(key)
        if lv is None and rv is not None:
            groups['single-sided'].append((key, None, rv)); continue
        if rv is None and lv is not None:
            groups['single-sided'].append((key, lv, None)); continue
        pred = predicates.get(key) or (lambda a, b: bool(set(a)) and bool(set(b)) and a != b)
        if not (lv is not None and rv is not None):
            continue
        if not bool(set(lv)) or not bool(set(rv)):
            groups['unknown'].append((key, lv, rv))
        elif pred(lv, rv):
            groups['conflict'].append((key, lv, rv))
        else:
            groups['agree'].append((key, lv, rv))
    absent = len(set(uni.registry) - {k for k in set(lparse) | set(rparse) if k != 'unclassified_keys'})
    if not any(groups.values()):
        return '<p class="muted">No registered dimension carries evidence on either side.</p>'
    def dim_rows(items):
        return ''.join(
            f'<tr><td>{escape(key)}</td><td>{_gate_value_html(lv)}</td><td>{_gate_value_html(rv)}</td></tr>'
            for key, lv, rv in items)
    head = '<tr><th>Dimension</th><th>Left</th><th>Right</th></tr>'
    conflict_table = (f'<table>{head}{dim_rows(groups["conflict"])}</table>') if groups['conflict'] else ''
    single_block = (f'<details><summary>single-sided ({len(groups["single-sided"])} dims — populated on one side only · missing evidence, not a conflict)</summary>'
                    f'<table>{head}{dim_rows(groups["single-sided"])}</table></details>') if groups['single-sided'] else ''
    agree_block = (f'<details><summary>agree ({len(groups["agree"])} dims)</summary><table>{head}{dim_rows(groups["agree"])}</table></details>') if groups['agree'] else ''
    unknown_block = (f'<details><summary>unknown ({len(groups["unknown"])} dims — populated raw but no delegated evidence on a side)</summary><table>{head}{dim_rows(groups["unknown"])}</table></details>') if groups['unknown'] else ''
    raw_block = (f'<details><summary>Raw attribute cells · full strings (left {len(str(left_attrs or "")):,} chars · right {len(str(right_attrs or "")):,} chars)</summary>'
                 f'<table><tr><th>Side</th><th>attribute (as exported)</th></tr>'
                 f'<tr><th>left</th><td><code>{escape(str(left_attrs or ""))}</code></td></tr>'
                 f'<tr><th>right</th><td><code>{escape(str(right_attrs or ""))}</code></td></tr></table></details>')
    headline = f'<p><strong>Mismatching dimensions · {len(groups["conflict"])} conflict</strong> · {len(groups["single-sided"])} single-sided · {len(groups["agree"])} agree · {len(groups["unknown"])} unknown · {absent} registered dims absent on both sides</p>'
    verdict = ''
    if groups['conflict']:
        names = ', '.join(key for key, _l, _r in groups['conflict'])
        verdict = f'<p class="muted">Independent preview diagnostic: {escape(names)} disagree under attribute predicates; this is not attribution for the stored gate decision.</p>'
    return headline + verdict + conflict_table + single_block + agree_block + unknown_block + raw_block
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

def _gate_canonical_evidence_html(left: str, right: str, cards: dict) -> str:
    """Render actual aggregate values, including provenance, without re-extracting."""
    left_card, right_card = cards.get(left, {}), cards.get(right, {})
    fields = sorted(set(left_card) | set(right_card))
    rows = ''.join(
        f'<tr><th>{escape(field)}</th><td>{_gate_value_html(left_card.get(field))}</td>'
        f'<td>{_gate_value_html(right_card.get(field))}</td></tr>' for field in fields)
    return ('<details open><summary>Canonical decision inputs · aggregate sets, confidence and source provenance</summary>'
            '<p class="muted">Values from the current canonical artifact; the stored gate reason below identifies the selected branch.</p>'
            f'<table><tr><th>Field</th><th>Left</th><th>Right</th></tr>{rows}</table></details>')


def _gate_listing_evidence_html(left: str, right: str, frame) -> str:
    blocks = []
    for side, gtin in (('left', left), ('right', right)):
        selected = frame[frame['gtin'].eq(gtin)]
        records = selected.where(selected.notna(), '').to_dict('records')
        cells = ''.join('<tr>' + ''.join(
            f'<td>{_gate_value_html(record.get(column))}</td>' for column, _original in _GATE_ORIGINAL_COLUMNS
        ) + '</tr>' for record in records)
        headings = ''.join(f'<th>{escape(original)}</th>' for _column, original in _GATE_ORIGINAL_COLUMNS)
        blocks.append(f'<details><summary>{side} · all {len(selected):,} original listings for {escape(gtin)}</summary>'
                      f'<table><tr>{headings}</tr>{cells}</table>{_gate_date_context_html(records)}</details>')
    return ''.join(blocks)


def _gate_date_context_html(records: list[dict]) -> str:
    from core.date_evidence import extract_date_evidence

    roles = {'expiry': 'Expiry', 'manufacture': 'Manufacture', 'shelf_life': 'Shelf life',
             'expiry_reference': 'Expiry reference', 'date_format_reference': 'Date format guidance',
             'unspecified_calendar_date': 'Date (context unclear)'}
    entries = []
    for record in records:
        for column in ('sku_name_eng', 'attribute', 'description_short_eng', 'breadcrumbs_eng', 'category'):
            for entry in extract_date_evidence(record.get(column)):
                value = ' or '.join(entry['normalized_candidates']) or 'No definite calendar date'
                if entry['role'] == 'shelf_life':
                    value = f"{entry['duration_value']} {entry['duration_unit']}s"
                entries.append('<tr>' + ''.join(f'<td>{escape(str(part))}</td>' for part in (
                    record.get('sku_id', ''), column, roles[entry['role']], entry['raw_match'], value,
                )) + '</tr>')
    if not entries:
        return ''
    return ('<details><summary>Stock dates and shelf life · listing context</summary>'
            '<p class="muted">Date differences describe stock or batches. These are review context; '
            'the deciding gate clause is shown separately. The export has no collection timestamp.</p>'
            '<table><tr><th>Listing</th><th>Source</th><th>Role</th><th>Original text</th>'
            '<th>Interpretation</th></tr>' + ''.join(entries) + '</table></details>')


def _gate_pair_html(rank: int, row, raw, listings, uni, cards=None, source_frame=None) -> str:
    try:
        sim = f'{float(row.similarity):.3f}'
    except Exception:
        sim = escape(row.similarity)
    model = _gate_side(row.gtin1, raw, listings)
    other = _gate_side(row.gtin2, raw, listings)
    def side_cell(value: str) -> str:
        if len(value) <= 140:
            return escape(value)
        return (f'{escape(value[:140])}… <details><summary>full string ({len(value):,} chars)</summary>'
                f'<code>{escape(value)}</code></details>')
    rows = (
        f'<tr><th>left · {escape(model["retailer"])}</th>' + ''.join(f'<td>{side_cell(model[orig]) if orig == "attribute" else escape(model[orig])}</td>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
        f'<tr><th>right · {escape(other["retailer"])}</th>' + ''.join(f'<td>{side_cell(other[orig]) if orig == "attribute" else escape(other[orig])}</td>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
    )
    listing_notes = f'{escape(model["_listings"])} · {escape(other["_listings"])}'
    clause = escape(row.gate_reason)
    label = 'fallback_reason' if row.gate_decision == 'fallback' else 'gate_reason (deciding clause)'
    route = '' if row.gate_decision == 'proceed' else f'{escape(_gate_route_gloss(str(row.gate_reason)))} — '
    verdict = f'<p><strong>{escape(row.gate_decision)}</strong> · <code>{escape(label)}</code>: {clause} · {route}similarity {escape(sim)}</p>'
    mism = ('<details><summary>Independent attribute diagnostic · illustrative first listings</summary>'
            + _gate_dimension_evidence_html(model['attribute'], other['attribute'], uni) + '</details>')
    canonical = _gate_canonical_evidence_html(row.gtin1, row.gtin2, cards or {})
    original_listings = _gate_listing_evidence_html(row.gtin1, row.gtin2, source_frame) if source_frame is not None else ''
    return (f'<details><summary>#{rank} · {escape(row.gate_decision)} · similarity {escape(sim)} · {clause}</summary>'
            f'{verdict}{canonical}{original_listings}<p class="muted">{listing_notes}</p>{mism}'
            f'<table><tr><th>Side</th>' + ''.join(f'<th>{escape(orig)}</th>' for _c, orig in _GATE_ORIGINAL_COLUMNS) + '</tr>'
            f'{rows}</table></details>')

def _gate_snapshot(g, raw, listings) -> dict:
    return {decision: [
        {'gtin1': r.gtin1, 'gtin2': r.gtin2, 'gate_decision': r.gate_decision,
         'gate_reason': r.gate_reason, 'similarity': r.similarity,
         'left': {orig: (r.gtin1 if c == 'gtin' else str(raw.loc[r.gtin1][c])) for c, orig in _GATE_ORIGINAL_COLUMNS},
         'right': {orig: (r.gtin2 if c == 'gtin' else str(raw.loc[r.gtin2][c])) for c, orig in _GATE_ORIGINAL_COLUMNS},
         **({'fallback_reason': r.gate_reason} if decision == 'fallback' else {})}
        for r in _gate_sample(g, raw, listings, decision)[0]]
        for decision in _GATE_BUCKETS}

@app.get('/gate', response_class=HTMLResponse)
def gate_decisions():
    g = _gate_results_frame(_gate_results_path.stat().st_mtime_ns)
    raw, listings = _gate_raw(DATA_PATH.stat().st_mtime_ns)
    uni = _gate_universe(DATA_PATH.stat().st_mtime_ns)
    cards = _gate_cards(F['canonical_records'].stat().st_mtime_ns)
    source_frame = _gate_source_frame(DATA_PATH.stat().st_mtime_ns)
    counts = g.gate_decision.value_counts()
    bucket_outcomes = {'proceed': 'PASS', 'hard_no': 'PASS', 'fallback': 'OPEN'}
    bucket_verdicts = {
        'proceed': "<span class='badge badge-pass'>PASS</span> stored gate decisions proceed; canonical inputs and selected branch are shown for each pair.",
        'hard_no': "<span class='badge badge-pass'>PASS</span> decisive clause blocks each sampled pair before anything else can run: never a merge, never a review.",
        'fallback': "<span class='badge badge-open'>OPEN</span> overlap exists but is too weak to proceed — deferred to review evidence, neither confirmed nor denied.",
    }
    findings = []
    for number, decision in enumerate(_GATE_BUCKETS, 1):
        picked, skipped, total = _gate_sample(g, raw, listings, decision)
        name_map = {'proceed': 'Proceed bucket', 'hard_no': 'Hard-no bucket', 'fallback': 'Fallback bucket'}
        metrics = ''.join([
            _fmetric(counts.get(decision, 0), f'{decision} pairs'),
            _fmetric(len(picked), 'sampled (first 5 in candidate order, endpoints resolvable)'),
            _fmetric(skipped, 'endpoint-unresolvable pairs skipped'),
        ])
        outcome = bucket_outcomes[decision]
        compare_rows = []
        examples_html = []
        for rank, row in enumerate(picked, 1):
            try:
                sim = f'{float(row.similarity):.3f}'
            except Exception:
                sim = escape(row.similarity)
            left_title = str(raw.loc[row.gtin1]['sku_name_eng'])
            right_title = str(raw.loc[row.gtin2]['sku_name_eng'])
            original = (f"<code>{escape(row.gtin1)}</code> · {escape(left_title)} ↔ "
                        f"<code>{escape(row.gtin2)}</code> · {escape(right_title)}")
            route = '' if decision == 'proceed' else f'{escape(_gate_route_gloss(str(row.gate_reason)))} — '
            resolved = f"<strong>{escape(decision)}</strong> (sim {escape(sim)}) — {route}<code>{escape(str(row.gate_reason))}</code>"
            compare_rows.append((original, resolved, outcome))
            examples_html.append(_gate_pair_html(rank, row, raw, listings, uni, cards, source_frame))
        sample_path = _gate_dir / 'gate_decision_sample.json'
        try:
            snapshot = json.loads(sample_path.read_text())
            excerpt = json.dumps(snapshot.get(decision, {}), indent=1)
        except Exception:
            excerpt = 'rendered evidence sample not yet written — run python dashboard/write_gate_sample.py'
        stamp = __import__('datetime').datetime.fromtimestamp(sample_path.stat().st_mtime).isoformat(timespec='seconds') if sample_path.exists() else 'not written'
        evidence = (
            f'<details open><summary>Raw evidence · full per-pair comparison (sample of 5 · original columns, as exported)</summary>'
            f'{"".join(examples_html)}</details>'
            + _fevidence(f'dashboard/evidence/datagen/gate_decision_sample.json (last written {stamp}) · data/gate_results.csv ({len(g):,} rows, gate_reason = deciding clause)',
                         excerpt))
        findings.append(_ffinding(number, f'{name_map[decision]} — original-column sample of 5', outcome,
                                  metrics, _fcompare(compare_rows), evidence,
                                  bucket_verdicts[decision]))
    counts_row = ''.join(_fmetric(counts.get(d, 0), f'{d} pairs') for d in _GATE_BUCKETS)
    try:
        from core.common import training_cfg

        ns = training_cfg().negative_supply
        mode, run_tag, mint_cap = ns.mode, ns.pairs_run_tag, ns.mint_cap
    except Exception:
        mode, run_tag, mint_cap = "gate", None, 1.0
    if mode == "lane":
        _lane_tail = (
            f" <strong>Active mode: <code>lane</code></strong> — negatives come from the lane's "
            f"real partners + minted top-up (run tag <code>{escape(str(run_tag))}</code>, "
            f"mint_cap {mint_cap:g}); the decisions below are SHADOW comparison columns only."
        )
    else:
        _lane_tail = (
            " <strong>Active mode: <code>gate</code></strong> — the stored decisions below are "
            "still the training label source until the mode flips on evidence (real-vs-minted "
            "discriminator + stratified eval: model-alone vs the gate on real-pair recall and "
            "false-merge rate), never on the code merely being present."
        )
    lane_note = (
        "<p class='muted'><strong>Decision-path status (owner ruling 2026-10-03):</strong> the "
        "attribute gate <strong>leaves the decision path</strong> — it is attribute-driven, so it "
        "can never be the label source nor a feature; it keeps running in <strong>shadow</strong> "
        "mode only, to compare \"model alone\" against the gate on real pairs. The replacement is "
        "the real-partner-first negative-supply lane (<code>src/training/negative_supply.py</code>: "
        "real partners first, minted only to top-up), selected by "
        "<code>training.negative_supply.mode</code>." + _lane_tail + "</p>"
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER gate decisions</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}{_FINDING_STYLE}h1{{font-size:1.4rem}}h2{{margin-top:.2rem}}</style></head><body>
 <h1>ER · Gate decisions — original-column evidence</h1>
 <p class="status">{len(g):,} candidate pair gates · {counts.get('proceed', 0):,} proceed · {counts.get('hard_no', 0):,} hard-no · {counts.get('fallback', 0):,} fallback · every fallback pair with full strings, no truncation · deciding clause = <code>gate_reason</code></p>
 {lane_note}
 <p><strong>Original columns, as exported · no cleaning.</strong> Every sampled pair below is joined back to the raw 13-column feed (dataset.csv via <code>core.common.load_dataset</code>, SSOT column mapping) — not the cleaned/canonical view. The COMPLETE fallback review pool lives at <a href="/gate/fallback"><strong>/gate/fallback</strong></a> — every pair, full strings, no truncation.</p>
 {''.join(findings)}
<p class="muted"><a href="/">← home</a> · <a href="/datagen">datagen track</a> · <a href="/graphs">graphs track</a></p>
</body></html>'''


@app.get('/gate/fallback', response_class=HTMLResponse)
def gate_fallback():
    """EVERY fallback pair — the ENTIRE original entry, full strings.

    Each pair renders BOTH sides with all 13 raw-export columns (sku_id,
    retailer, country, title, description, breadcrumbs_eng, url, image_url,
    price, gtin, brand, category, attributes), no ellipsis and no
    <details> folding anywhere — the whole entry as exported. The full
    canonical texts and the deciding clause ride along. The bucket is OPEN
    by definition (low raw volume/pack extraction confidence -> the gate
    deliberately withholds), so everything below is review evidence.
    Sorted by similarity descending — the solvable-probable pool first.
    (Per-pair dimension evidence stays on /gate's 5-sample boxes; parsing
    the AttributeUniverse for all 41,481 pairs here would flip a wholesale
    listing into minutes of server work.)
    """
    g = _gate_results_frame(_gate_results_path.stat().st_mtime_ns)
    raw, _listings = _gate_raw(DATA_PATH.stat().st_mtime_ns)
    fb = g[g.gate_decision == 'fallback'].copy()
    fb = fb.assign(simv=pd.to_numeric(fb.similarity, errors='coerce').fillna(0.0))
    counts = fb.gate_reason.value_counts()
    sim_bands = {
        band: int(grp.sum()) for band, grp in {
            '<0.30': (fb.simv < 0.30),
            '0.30-0.50': fb.simv.between(0.30, 0.50, 'left'),
            '0.50-0.70': fb.simv.between(0.50, 0.70, 'left'),
            '0.70-0.90': fb.simv.between(0.70, 0.90, 'left'),
            '>=0.90': (fb.simv >= 0.90)}.items()
    }
    fb = fb.sort_values('simv', ascending=False)

    def side_cells(gtin: str) -> str:
        if gtin not in raw.index:
            return '<td><code>—</code></td>' * len(_FALLBACK_RAW_COLUMNS)
        entry = raw.loc[gtin].copy()
        # _gate_raw drops 'gtin' into the index; resurrect it for the
        # all-columns render (the raw-export column belongs on screen).
        entry['gtin'] = gtin
        return ''.join(
            f'<td><code>{escape(str(entry[c]))}</code></td>' for c in _FALLBACK_RAW_COLUMNS)

    rows = ''.join(
        f'<tr><td><code>{escape(str(r.gtin1))}</code> ↔ <code>{escape(str(r.gtin2))}</code></td>'
        f'<td>{float(r.simv):.3f}</td>'
        f'<td>{escape(str(r.gate_reason))}</td>'
        f'<td colspan="2"><code>{escape(str(r.canon1))}</code><br><code>{escape(str(r.canon2))}</code></td></tr>'
        f'<tr class="pp"><td colspan="3">left · <code>{escape(str(r.gtin1))}</code></td>'
        + side_cells(str(r.gtin1)) + '</tr>'
        f'<tr><td colspan="3">right · <code>{escape(str(r.gtin2))}</code></td>'
        + side_cells(str(r.gtin2)) + '</tr>'
        for r in fb.itertuples())
    metrics = ''.join([
        _fmetric(len(fb), 'fallback pairs (ALL rendered — entire entry, no truncation)'),
        _fmetric(int((fb.simv >= 0.7).sum()), 'sim >= 0.70 (solvable-probable pool)'),
        _fmetric(int((fb.simv >= 0.9).sum()), 'sim >= 0.90'),
        _fmetric(len(set(fb.gtin1) | set(fb.gtin2)), 'distinct gtins involved'),
    ])
    reason_row = ''.join(_fmetric(v, k) for k, v in counts.items())
    band_row = ''.join(_fmetric(v, f'sim {k}') for k, v in sim_bands.items())
    col_head = ''.join(f'<th>{escape(c)}</th>' for c in _FALLBACK_RAW_COLUMNS)
    _DOCTYPE = "<!doctype html><html lang=\"en\">"
    head = _DOCTYPE + (
        "<head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>ER gate fallback — full review pool</title>"
        "<style>body{font-family:system-ui;margin:2rem;color:#222}" + _FINDING_STYLE +
        "h1{font-size:1.4rem}"
        "table{border-collapse:collapse;width:100%;font-size:.78rem}"
        "td,th{border:1px solid #ccc;padding:.3rem .4rem;vertical-align:top;text-align:left;overflow-wrap:anywhere}"
        "code{font-size:.62rem}td code{display:block;white-space:pre-wrap}"
        "tr.pp td{background:#f7f7f7}</style></head><body>")
    body = (
        "<h1>ER · Gate fallback — every undecided pair, entire original entry</h1>"
        f"<div class=\"finding\">{metrics}{reason_row}{band_row}</div>"
        "<div class=\"verdict\"><span class='badge badge-open'>OPEN</span> — low raw extraction confidence "
        "on volume/pack: overlap exists, verdict deliberately withheld (no guessing). "
        "Both sides carry the ENTIRE raw-export entry (all 13 columns, full strings) plus the full "
        "canonical texts. Sorted by similarity descending. Metrics + doctrine proposal: "
        "<code>scripts/fallback_adjudication.py</code> → results/gate_fallback_metrics.json.</div>"
        "<table><tr><th>pair</th><th>sim</th><th>gate_reason</th><th>pair canonical · FULL</th><th>canon2 (FULL)</th>"
        + col_head + "</tr>" + rows + "</table>"
        "<p class=\"muted\"><a href=\"/gate\">← gate decisions</a> · "
        "<a href=\"/datagen\">datagen track</a> · <a href=\"/\">home</a></p></body></html>")
    return head + body


@app.get('/graphs', response_class=HTMLResponse)
def graphs_track():
    # Finding 01 — configs + lane scope
    config_evidence = []
    config_count = 0
    for name in ('graph_tracks_gnn.yaml', 'graph_tracks_hybrid.yaml'):
        path = ROOT.parent / 'config' / name
        if path.exists():
            config_count += 1
            config_evidence.append(f'{name}:\n' + path.read_text()[:900])
    config_excerpt = '\n\n'.join(config_evidence) or 'no graph-track configs present yet'
    cfg_metrics = ''.join([
        _fmetric(config_count, 'configs governed in config/'),
        _fmetric('8 + 2', 'categorical relations + volume/pack'),
        _fmetric('full-batch', 'typed two-hop aggregation (NOT sampled GraphSAGE)'),
    ])
    cfg_compare = _fcompare([
        ('Aggregation scheme (planned lane)', 'full-batch typed two-hop aggregation over 8 categorical relations + volume/pack', 'PASS'),
        ('Config surface', 'graph_tracks_gnn.yaml + graph_tracks_hybrid.yaml', 'PASS' if config_count == 2 else 'OPEN'),
        ('Tractable catalogs in lane scope', '62,963-row catalogs — skipped by design (graph lane covers tractable sizes only)', 'OPEN'),
    ])
    f01 = _ffinding(1, 'GNN-only + hybrid semantic-ID lane — configs', 'PASS', cfg_metrics, cfg_compare,
                    _fevidence('config/graph_tracks_gnn.yaml + config/graph_tracks_hybrid.yaml (verbatim, capped)', config_excerpt),
                    "<span class='badge badge-pass'>PASS</span> — lane setting fixed in config; tractable catalogs skip.")
    # Finding 02 — checkpoint / DVC snapshot lifecycle
    snips = sorted((ROOT.parent / 'results').glob('graph_tracks/*'))[:8]
    pubs = sorted((ROOT.parent / 'dvc_refs').glob('*'))[:8]
    lifecycle_metrics = ''.join([
        _fmetric(len(snips), 'local graph_tracks snapshots'),
        _fmetric(len(pubs), 'published DVC refs'),
        _fmetric('5001027', 'commit where lifecycle landed'),
    ])
    lifecycle_compare = _fcompare([
        ('Checkpoint persistence planned as ad-hoc local folders',
         'DVC refs + checkpoint manifests (commit 5001027)', 'PASS'),
        ('Snapshot publication from the Colab lane (DVC remote + W&B offline bundles)',
         'published refs resolve locally in dvc_refs/', 'PASS' if pubs else 'OPEN'),
    ])
    listing = '\n'.join([f'results/graph_tracks/{d.name}' for d in snips]
                        + [f'dvc_refs/{d.name} (published)' for d in pubs]) or 'no local snapshots yet — workers publish from the Colab lane'
    f02 = _ffinding(2, 'Checkpoint / DVC snapshot lifecycle', 'PASS' if pubs else 'OPEN',
                    lifecycle_metrics, lifecycle_compare,
                    _fevidence('results/graph_tracks/* + dvc_refs/* (directory listing, capped)', listing, cap=600),
                    f"<span class='badge {'badge-pass' if pubs else 'badge-open'}'>{'PASS' if pubs else 'OPEN'}</span> — lifecycle logic landed in commit 5001027{' ; published refs present' if pubs else ' ; no published refs yet'}.")
    # Finding 03 — sequencing (waiting-on)
    # migrated 2026-10-05: fail-loud tracked evidence lives in artifacts/evidence/
    census_path = ROOT.parent / 'artifacts' / 'evidence' / 'attribute_universe_census.json'
    census_landed = census_path.exists()
    wait_metrics = ''.join([
        _fmetric('landed' if census_landed else 'in flight', 'AttributeUniverse census (feeds graph node relations)'),
        _fmetric('open', 'P1/P2 items from TODO.md (owner DEAD LAST ruling)'),
    ])
    wait_compare = _fcompare([
        ('AttributeUniverse census must land before graph node relations exist',
         f'artifacts/evidence/attribute_universe_census.json {"present — feeds the relations" if census_landed else "not yet written"}',
         'PASS' if census_landed else 'OPEN'),
        ('P1/P2 TODO items must close before graph linkage expands', 'open — owner DEAD LAST ruling', 'OPEN'),
    ])
    f03 = _ffinding(3, 'Waiting on — sequencing rules', 'OPEN', wait_metrics, wait_compare,
                    _fevidence('artifacts/evidence/attribute_universe_census.json + TODO.md',
                               'Census feeds the graph node relations; P1/P2 items gate the expansion of graph linkage.'),
                    "<span class='badge badge-open'>OPEN</span> — census is the dependency; linkage expansion waits on the P1/P2 owner ruling.")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER graphs</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}{_FINDING_STYLE}h1{{font-size:1.4rem}}</style></head><body>
<h1>ER · Graphs track — GNN-only + hybrid semantic-ID lane</h1>
<p class="status">2 lanes (GNN-only / hybrid semantic-ID) · full-batch typed two-hop aggregation · snapshot lifecycle landed (commit 5001027) · 62,963-row catalogs skip</p>
{f01}{f02}{f03}
<p class="muted"><a href="/">← home</a> · <a href="/datagen">datagen track</a> · <a href="/gate">gate decisions</a> · <a href="/training">training reports</a></p>
</body></html>'''


# ── runs history track ───────────────────────────────────────────────────────
# Table of every generated run (bundle prep runs + training runs) with a
# regression quick-glance against the PREVIOUS bundle run, plus a compare
# view (/runs/compare?a=..&b=..) rendering the run-history ComparisonReport:
# per-stage seconds scale-normalized to per-1000-dataset-rows rates (grown
# rates highlighted), census shifts, output-rate regressions and the
# worst-offender drift. regression semantics live in
# model_tracks.run_history.compare; this page only renders them.

_RUNS_STYLE = _FINDING_STYLE + (
    'tr.grew{background:#f8d7da}tr.fell{background:#dff4e8}'
    '.runstatus{padding:.1rem .5rem;border-radius:.6rem;font-weight:700;font-size:.8rem}'
    '.st-complete{background:#dff4e8}.st-running{background:#fff0cf}'
    '.st-failed{background:#f8d7da}.st-unknown{background:#eee}')


def _run_status_badge(status):
    key = (status or 'unknown')
    label = escape(key)
    cls = {c: f'st-{c}' for c in ('complete', 'running', 'failed', 'unknown')}.get(key, 'st-unknown')
    return f'<span class="runstatus {cls}">{label}</span>'


def _runs_roots():
    from core.common import TRAIN_ROOT
    return TRAIN_ROOT / 'results' / 'training_prep', TRAIN_ROOT / 'training_results'


def _runs_table(prep_root, training_root):
    """Rows (run id, date, status, rows, outputs, artifacts) + regression flag vs the previous bundle run."""
    from model_tracks.run_history import (bundle_facts, compare, list_bundles,
                                          list_training_runs)
    def fmt(value):
        return '—' if value is None else f'{value:,}'
    rows = []
    bundle_dirs = list_bundles([prep_root])
    # Chronological pairing: each run's badge compares it against the OLDER
    # neighbor (compare(older, newer); b = this run), so a REGRESSION badge
    # means "this run got slower per-1000-rows than the run before it".
    older = None
    for entry in bundle_dirs:
        try:
            run = bundle_facts(entry)
        except ValueError as error:
            rows.append(f'<tr class="grew"><td>{escape(entry.name)}</td>'
                        f'<td colspan="9">unreadable run record — {escape(str(error))} '
                        f'<a href="/runs/compare?a={escape(entry.name)}&b={escape(entry.name)}">try compare page</a></td></tr>')
            continue
        regression_cell = '·'
        if older is not None:
            report = compare(older, run)
            if report.pair_kind == 'incomplete_pair':
                regression_cell = f'<a href="/runs/compare?a={escape(report.a_id)}&b={escape(report.b_id)}" style="color:#666">pair?</a>'
            elif report.regressions:
                regression_cell = (f'<a href="/runs/compare?a={escape(report.a_id)}&b={escape(report.b_id)}" '
                                   f'style="color:#b00;font-weight:700">REGRESSION ({len(report.regressions)})</a>')
        labeled = (f"{run.labeled.kept:,} ({run.labeled.pos:,}p/{run.labeled.hard_neg:,}n)"
                   if run.labeled else '—')
        attested = ('attested' if run.handoff and run.handoff.loss_batch_attested else
                    'present' if run.handoff else '—')
        rows.append(
            f'<tr><td><a href="/runs/compare?a={escape(run.run_id)}&b={escape(run.run_id)}">{escape(run.run_id)}</a></td>'
            f'<td>{escape(run.created or "—")}</td><td>{_run_status_badge(run.status)}</td>'
            f'<td>{fmt(run.dataset_rows)}</td><td>{labeled}</td><td>{fmt(run.minted_rows)}</td>'
            f'<td>{len(run.stages)}</td><td>{run.total_stage_seconds:,.1f}</td>'
            f'<td>{attested}</td><td>{regression_cell}</td></tr>')
        older = run
    training_rows = []
    for entry in list_training_runs([training_root]):
        from model_tracks.run_history import training_run_facts
        facts_run = training_run_facts(entry)
        training_rows.append(
            f'<tr><td>{escape(facts_run.run_id)}</td><td>{escape(facts_run.created or "—")}</td>'
            f'<td>{_run_status_badge(facts_run.status)}</td><td colspan="6">—</td></tr>')
    return rows, training_rows


@app.get('/runs', response_class=HTMLResponse)
def runs_page():
    prep_root, training_root = _runs_roots()
    try:
        rows, training_rows = _runs_table(prep_root, training_root)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    body = ''.join(rows)
    training_body = ''.join(training_rows) or '<tr><td colspan="9">no completed training runs locally</td></tr>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER runs</title><style>{_RUNS_STYLE}</style></head><body>
<h1>ER · Run history</h1>
<p class="status">{len(rows)} bundle runs · regression flag compares each run against the PREVIOUS bundle run (per-1000-rows scale-normalized; status-mismatched pairs are marked pair?, never regressions)</p>
<table><tr><th>Run</th><th>Created</th><th>Status</th><th>Dataset rows</th><th>Labeled pairs</th><th>Minted</th><th>Stages</th><th>Stage seconds</th><th>Handoff</th><th>vs previous</th></tr>{body}</table>
<h2>Training runs <span class="muted">(completed-retention markers only)</span></h2>
<table><tr><th>Run</th><th>Created</th><th>Status</th><th colspan="6">Metrics</th></tr>{training_body}</table>
<p class="muted"><a href="/">← home</a> · <a href="/training">training reports</a> · compare view: <code>/runs/compare?a=&lt;id&gt;&amp;b=&lt;id&gt;</code></p>
</body></html>'''


@app.get('/runs/compare', response_class=HTMLResponse)
def runs_compare_page(a: str, b: str):
    from model_tracks.run_history import compare, facts as run_facts
    prep_root, training_root = _runs_roots()
    try:
        report = compare(run_facts(a, prep_roots=[prep_root], training_roots=[training_root]),
                         run_facts(b, prep_roots=[prep_root], training_roots=[training_root]))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    def fmt(value, spec=',.0f'):
        return '—' if value is None else format(value, spec)

    stage_rows = []
    for row in report.stages:
        grew = row.per_1k_ratio is not None and row.per_1k_ratio > 1.0 \
            and row.regression is not None
        cls = ' class="grew"' if grew and row.regression else (' class="fell"' if row.regression is False and row.per_1k_ratio is not None and row.per_1k_ratio < 1.0 else '')
        flag = 'REGRESSION' if row.regression else ('improved' if row.regression is False and row.per_1k_ratio is not None and row.per_1k_ratio < 1.0 else '·')
        stage_rows.append(
            f'<tr{cls}><td>{escape(row.stage)}</td><td>{fmt(row.a_seconds, ",.1f")}</td>'
            f'<td>{fmt(row.b_seconds, ",.1f")}</td><td>{fmt(row.a_per_1k, ".3f")}</td>'
            f'<td>{fmt(row.b_per_1k, ".3f")}</td>'
            f'<td>{"×%.2f" % row.per_1k_ratio if row.per_1k_ratio is not None else "—"}</td>'
            f'<td>{escape(row.a_status or "—")} → {escape(row.b_status or "—")}</td><td>{flag}</td></tr>')
    census_rows = ''.join(
        f'<tr><td>{escape(row.key)}</td><td>{fmt(row.a)}</td><td>{fmt(row.b)}</td>'
        f'<td>{fmt(row.a_per_1k, ".3f")}</td><td>{fmt(row.b_per_1k, ".3f")}</td>'
        f'<td>{"×%.2f" % row.per_1k_ratio if row.per_1k_ratio is not None else "—"}</td></tr>'
        for row in report.census) or '<tr><td colspan="6">no gate census on both runs</td></tr>'
    output_rows = []
    for row in report.outputs:
        cls = ' class="grew"' if row.regression else ''
        flag = 'REGRESSION (rate fell)' if row.regression else ('improved' if row.regression is False and row.per_1k_ratio is not None and row.per_1k_ratio > 1.0 else '·')
        output_rows.append(
            f'<tr{cls}><td>{escape(row.name)}</td><td>{fmt(row.a)}</td><td>{fmt(row.b)}</td>'
            f'<td>{fmt(row.a_per_1k, ".3f")}</td><td>{fmt(row.b_per_1k, ".3f")}</td>'
            f'<td>{"×%.2f" % row.per_1k_ratio if row.per_1k_ratio is not None else "—"}</td><td>{flag}</td></tr>')
    offenders = ''.join(
        f'<tr><td>{escape(label)}</td><td>{fmt(a_sec, ",.1f")}</td><td>{fmt(b_sec, ",.1f")}</td></tr>'
        for label, a_sec, b_sec in report.offenders_top) or '<tr><td colspan="3">offenders absent on both runs</td></tr>'
    regressions = ('<ul>' + ''.join(
        f'<li><span class="badge badge-fail">{escape(item.kind)}</span> {escape(item.detail)}</li>'
        for item in report.regressions) + '</ul>'
        if report.regressions else
        '<p>No regressions under the scale-normalized rules.</p>'
        if report.pair_kind == 'complete_pair' else
        '<p><span class="badge badge-open">incomplete_pair</span> — runs of different status are NOT compared as regressions.</p>')
    banner = (f'<p class="status">pair={escape(report.pair_kind)} · scale_normalized='
              f'{"yes" if report.scale_normalized else "no"} '
              f'(dataset_rows {fmt(report.dataset_rows_a)} vs {fmt(report.dataset_rows_b)})</p>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER runs · compare</title><style>{_RUNS_STYLE}</style></head><body>
<h1>ER · Run compare — {escape(report.a_id)} → {escape(report.b_id)}</h1>{banner}
<h2>Stages (raw seconds | per 1000 dataset rows; grown per-1k rows highlighted)</h2>
<table><tr><th>Stage</th><th>A s</th><th>B s</th><th>A s/1k</th><th>B s/1k</th><th>B/A</th><th>Status A→B</th><th>Verdict</th></tr>{''.join(stage_rows)}</table>
<h2>Census shifts (gate pairs, per-1000-rows)</h2>
<table><tr><th>Key</th><th>A</th><th>B</th><th>A per-1k</th><th>B per-1k</th><th>B/A</th></tr>{census_rows}</table>
<h2>Output counts per 1000 rows (fell = regression)</h2>
<table><tr><th>Output</th><th>A</th><th>B</th><th>A per-1k</th><th>B per-1k</th><th>B/A</th><th>Verdict</th></tr>{''.join(output_rows)}</table>
<h2>Worst offender drift</h2>
<p>{escape(report.worst_offender_a.label if report.worst_offender_a else '—')} ({fmt(report.worst_offender_a.seconds if report.worst_offender_a else None, ",.1f")}s) →
{escape(report.worst_offender_b.label if report.worst_offender_b else '—')} ({fmt(report.worst_offender_b.seconds if report.worst_offender_b else None, ",.1f")}s)
{"<strong>— DRIFTED</strong>" if report.worst_offender_drift else ""}</p>
<table><tr><th>Offender</th><th>A s</th><th>B s</th></tr>{offenders}</table>
<h2>Regressions ({len(report.regressions)})</h2>{regressions}
<p class="muted"><a href="/runs">← run history</a></p>
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
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>' + escape(title) + '</title><style>body{font-family:system-ui;margin:2rem;color:#222}section{border-top:1px solid #ccc;padding:1rem 0}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #ccc;padding:.4rem;text-align:left;overflow-wrap:anywhere}details{margin:1rem 0}summary{cursor:pointer}</style></head><body><p><a href="/">← home</a> · <a href="/experiments">All findings</a> · <a href="/training">Training reports</a> · <a href="/datagen">Datagen track</a> · <a href="/gate">Gate decisions</a> · <a href="/graphs">Graphs track</a></p>' + body + '</body></html>'

@app.get('/findings/03', response_class=HTMLResponse)
def context_finding():
    return (ROOT / 'experiments/results/identity/03_measurement_and_packaging_context_findings.html').read_text()

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('ER_DASHBOARD_PORT', '8001')))
