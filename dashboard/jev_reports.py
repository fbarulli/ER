"""Live, read-only views of staged JEV samples and saved checkpoints."""
import html
import json
from collections import Counter
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

PROJECT = Path(__file__).resolve().parents[1]
router = APIRouter()
e = lambda value: html.escape(str(value), quote=True)


def load(name, default):
    path = PROJECT / 'jev' / name
    return json.loads(path.read_text()) if path.exists() else default


def checkpoint(name):
    path = PROJECT / 'jev' / name
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    return rows


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{e(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{e(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


@router.get('/jev', response_class=HTMLResponse)
def jev(round: int | None = None, stratum: str = ''):
    ledger = load('sample_ledger.json', [])
    if round is None:
        round = max((x['round'] for x in ledger if x.get('kind') != 'controlled_repeat'), default=0)
    selected = next((x for x in ledger if x['round'] == round), None)
    if selected is None: raise HTTPException(404, 'Sample round not found')
    summaries = []
    for item in ledger:
        sample = load(Path(item['sample']).name, [])
        staged = {(x.get('input_scope',''),x['gtin1'],x['gtin2']) for x in sample}
        done = {(x.get('input_scope',''),x.get('gtin1'),x.get('gtin2')) for x in checkpoint(Path(item['checkpoint']).name) if x.get('status') == 'ok'} & staged
        status = 'Tested' if staged and done == staged else 'Partially tested' if done else 'Staged, not tested'
        summaries.append([item['round'], status, item['unique_pairs'], len(staged), len(done)])
    body = '<h1>JEV matching audits</h1><p>Fresh samples, attribute coverage, and saved evaluation results. Each pair is staged in both orders.</p>'
    body += '<h2>Sample ledger</h2>' + table(['Round','Status','Unique pairs','Staged calls','Completed calls'], summaries)
    body += '<p>' + ' · '.join(f'<a href="/jev?round={x["round"]}">Round {x["round"]}</a>' for x in ledger) + '</p>'
    body += f'<h2>Round {round}</h2>'
    if round >= 3:
        summary = load(f'sample_{round}_summary.json', {})
        if selected.get('kind') == 'controlled_repeat':
            body += '<p>Intentional repeat of one previously tested pair, using both input formats and both orders.</p>'
        else:
            body += '<p>Seed ' + e(summary.get('seed','')) + '; excludes ' + e(summary.get('excluded_previously_staged_or_tested_pairs',0)) + ' previously staged or tested pairs. Positive = proceed; negative = hard no; uncertain = fallback. High similarity ≥ 0.8; lower similarity &lt; 0.8.</p>'
        body += table(['Stratum','Candidate pool','Selected pairs'], [[k,v['candidate_pool'],v['selected']] for k,v in summary.get('allocations',{}).items()])
        body += '<details><summary>Attribute coverage (unique pairs)</summary>'
        coverage = summary.get('attribute_coverage',{})
        body += table(['Stratum','Attribute','State','Pairs'], [[*k.split('|'),v] for k,v in coverage.items() if not stratum or k.split('|')[0]==stratum]) + '</details>'
    if round >= 4:
        body += '<h3>Input split</h3>' + table(['Input cohort','Stratum','Pairs'], [[scope,k,v] for scope,allocation in summary.get('input_cohorts',{}).items() for k,v in allocation.items()])
        body += '<p>' + e(summary.get('comparison','')) + '</p>'
        body += table(['Input cohort','Evidence'], summary.get('input_definitions',{}).items())
    sample = [x for x in load(Path(selected['sample']).name,[]) if x.get('copy') == 'a_order']
    strata = sorted({x['stratum'] for x in sample})
    body += f'<form><input type="hidden" name="round" value="{round}"><label>Stratum <select name="stratum"><option value="">All</option>' + ''.join(f'<option value="{e(x)}" {"selected" if x==stratum else ""}>{e(x)}</option>' for x in strata) + '</select></label> <button>Filter</button></form>'
    filtered = [x for x in sample if not stratum or x['stratum']==stratum]
    body += f'<h3>Pairs ({len(filtered)})</h3><p>Preview shows up to 100 pairs. Download the sample for all pairs and both orders.</p>'
    body += table(['GTIN A','GTIN B','Input','Stratum','Gate','Similarity','Attribute states'], [[x['gtin1'],x['gtin2'],x.get('input_scope','first source listing'),x['stratum'],x.get('gate',''),x['similarity'], '; '.join(f'{k}: {v}' for k,v in x.get('attribute_states',{}).items())] for x in filtered[:100]])
    body += f'<p><a href="/jev/artifact?name={e(Path(selected["sample"]).name)}">Download sample</a> · <a href="/jev/artifact?name=sample_ledger.json">Download ledger</a></p>'
    checkpoint_name = Path(selected['checkpoint']).name
    if (PROJECT/'jev'/checkpoint_name).exists():
        body += f'<p><a href="/jev/artifact?name={e(checkpoint_name)}">Download completed results</a></p>'
    run_name = f'audit_run_{round}.json'
    if (PROJECT/'jev'/run_name).exists():
        run = load(run_name, {})
        body += '<p>Adapter: ' + e(run.get('adapter','')) + '; model: ' + e(run.get('model','')) + '; completed: ' + e(run.get('completed_utc','')) + '.</p><p><a href="/jev/artifact?name=' + e(run_name) + '">Download run metadata</a></p>'
    reports = load('verification_results.json', [])
    body += '<h2>Verified rounds</h2>' + table(['Checkpoint','Pairs','Low-score proceeds','High-score rejections','Decision order differences'], [[x['checkpoint'],x['unique_pairs'],len(x['low_score_proceeds']),len(x['high_score_rejections']),len(x['gate_asymmetries'])] for x in reports])
    selected_report = next((x for x in reports if x['checkpoint'] == checkpoint_name), None)
    if selected_report and round >= 4:
        body += '<h3>Judgments by input cohort</h3>' + table(['Input','Gate decision','Pairs','Mean score','Both scores < 0.2','Both scores > 0.8'], [[scope,decision,v['pairs'],f"{v['mean_score']:.3f}" if v['mean_score'] is not None else '',v['both_below_0_2'],v['both_above_0_8']] for scope,decisions in selected_report.get('input_cohorts',{}).items() for decision,v in decisions.items()])
    control = load('control_comparison_5.json', {})
    if control:
        body += '<h3>Same-pair input comparison</h3><p>' + e(' ↔ '.join(control['pair'])) + '</p>'
        body += table(['Input','Original order score','Swapped order score'], [[scope,v['a_order'],v['b_swapped']] for scope,v in control['input_scores'].items()])
        body += '<p>' + e(control['interpretation']) + '</p>'
    body += '<p>Low-score proceeds have JEV scores below 0.2 in both orders; high-score rejections score above 0.8 in both. These are saved replay findings, not a live replay of future gate changes. Rounds 1–3 used the first source listing. Round 4 compares merged gate evidence with all original listings. Balanced discovery samples do not measure population accuracy.</p>'
    body += '<p><a href="/jev/artifact?name=verification_results.json">Download verification details</a></p>'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JEV audits</title><style>body{font:15px system-ui;color:#222;margin:2rem}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #ddd;padding:.5rem;text-align:left;overflow-wrap:anywhere}th{background:#f3f4f6}details,form{margin:1rem 0}a{color:#2563eb}</style></head><body>' + body + '</body></html>'


@router.get('/jev/artifact')
def artifact(name: str):
    allowed = {'sample_ledger.json','sample_3_summary.json','verification_results.json','SAMPLE_LEDGER.md','VERIFICATION.md','audit_run_3.json','audit_errors_3.jsonl','control_comparison_5.json'}
    for item in load('sample_ledger.json',[]):
        allowed.update((Path(item['sample']).name,Path(item['checkpoint']).name))
    for item in load('sample_ledger.json',[]):
        n=item['round']
        allowed.update((f'audit_run_{n}.json',f'sample_{n}_summary.json',f'input_states_{n}.json'))
    path = PROJECT/'jev'/name
    if name not in allowed or not path.is_file() or path.is_symlink():
        raise HTTPException(404, 'Artifact not found')
    return FileResponse(path, filename=name)
