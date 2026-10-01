"""Read training report plots from local run folders and downloaded suites."""
from html import escape
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode
import csv
import io
import json
import stat
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

router = APIRouter()
PROJECT = Path(__file__).resolve().parents[1]
REPORT_ERRORS = (OSError, ValueError, zipfile.BadZipFile, RuntimeError, EOFError)
METRIC_SUFFIXES = ('model_evaluation_summary.csv', 'retrieval_summary.csv', 'fold_metrics.csv')


def safe_member(member):
    relative = PurePosixPath(member)
    return bool(member) and not relative.is_absolute() and '..' not in relative.parts and '\\' not in member


def runs():
    found = {}
    for base in (PROJECT / 'results/model_tracks', PROJECT / 'results/graph_tracks',
                 PROJECT / 'training_results'):
        if not base.exists():
            continue
        for path in sorted(base.iterdir(), reverse=True):
            if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
                continue
            if path.is_dir() or (path.suffix == '.zip' and '__inputs' not in path.name):
                key = path.relative_to(PROJECT).as_posix()
                if path.is_dir() and not any(member.endswith(('.png',) + METRIC_SUFFIXES)
                                             for member in entries(path)):
                    continue
                found[key] = path
    return dict(sorted(found.items(), key=lambda item: item[1].stat().st_mtime, reverse=True))


def entries(path):
    if path.is_dir():
        return [p.relative_to(path).as_posix() for p in path.rglob('*')
                if p.is_file() and p.resolve().is_relative_to(path.resolve())
                and safe_member(p.relative_to(path).as_posix())]
    with zipfile.ZipFile(path) as archive:
        return [info.filename for info in archive.infolist() if safe_member(info.filename)
                and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)]


def read(path, member):
    if not safe_member(member):
        raise HTTPException(404, 'Report artifact not found')
    if path.is_dir():
        target = (path / member).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise HTTPException(404, 'Report artifact not found')
        return target.read_bytes()
    with zipfile.ZipFile(path) as archive:
        try:
            if stat.S_ISLNK(archive.getinfo(member).external_attr >> 16):
                raise HTTPException(404, 'Report artifact not found')
            return archive.read(member)
        except KeyError as error:
            raise HTTPException(404, 'Report artifact not found') from error


def table(data):
    rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
    if not rows:
        return '<p>No metric rows.</p>'
    return '<div class="scroll"><table>' + ''.join(
        '<tr>' + ''.join(f'<{tag}>{escape(cell)}</{tag}>' for cell in row) + '</tr>'
        for tag, row in [('th', rows[0])] + [('td', row) for row in rows[1:21]]) + '</table></div>'


@router.get('/training/plot')
def plot(run: str, artifact: str):
    path = runs().get(run)
    if path is None or not artifact.endswith('.png'):
        raise HTTPException(404, 'Report plot not found')
    try:
        if artifact not in entries(path):
            raise HTTPException(404, 'Report plot not found')
        return Response(read(path, artifact), media_type='image/png')
    except REPORT_ERRORS as error:
        raise HTTPException(404, 'Report plot unavailable') from error


@router.get('/training', response_class=HTMLResponse)
def training(run: str | None = None):
    available = runs()
    if run is not None and run not in available:
        raise HTTPException(404, 'Training run not found')
    selected = run or next(iter(available), None)
    options = ''.join(f'<option value="{escape(key, quote=True)}"'
                      f'{" selected" if key == selected else ""}>{escape(key)}</option>'
                      for key in available)
    body = '<h1>Training reports</h1><p>Saved metrics and plots for text, GNN-only and hybrid runs. Smoke runs verify the workflow; their scores are not model quality benchmarks.</p>'
    if selected is None:
        body += '<p>No downloaded training reports yet.</p>'
    else:
        body += f'<form><label>Run <select name="run">{options}</select></label> <button>Show reports</button></form>'
        path = available[selected]
        try:
            members = entries(path)
        except REPORT_ERRORS:
            members = []
            body += '<p>This report archive could not be read. Download the completed results again.</p>'
        if 'suite_manifest.json' in members:
            try:
                manifest = json.loads(read(path, 'suite_manifest.json'))
                if not isinstance(manifest, dict) or not isinstance(manifest.get('config', {}), dict):
                    raise ValueError('Invalid suite manifest')
            except REPORT_ERRORS:
                manifest = {}
                body += '<p>Suite metadata could not be read.</p>'
            cfg = manifest.get('config', {})
            complete = 'suite_result.json' in members
            body += f'<p>Device: {escape(str(cfg.get("device", "unknown")))} · Epochs: {escape(str(cfg.get("epochs", "unknown")))} · Test reporting: {escape(str(cfg.get("report_test", "unknown")))} · {"Complete suite" if complete else "Incomplete suite"}</p>'
        plots = sorted(m for m in members if m.endswith('.png'))
        summaries = sorted(m for m in members if m.endswith(METRIC_SUFFIXES))
        for member in summaries:
            try:
                metrics = table(read(path, member))
            except REPORT_ERRORS:
                metrics = '<p>Metrics could not be read.</p>'
            body += f'<details open><summary>{escape(member)}</summary>{metrics}</details>'
        body += '<div class="plots">'
        for member in plots:
            url = '/training/plot?' + urlencode({'run': selected, 'artifact': member})
            body += f'<figure><figcaption>{escape(member)}</figcaption><a href="{escape(url, quote=True)}" target="_blank"><img loading="lazy" src="{escape(url, quote=True)}" alt="{escape(Path(member).stem, quote=True)}"></a></figure>'
        body += '</div>'
        if not plots:
            body += '<p>This run has no saved plots.</p>'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER training reports</title><style>body{font-family:system-ui;margin:2rem;color:#222}select{max-width:75vw}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:1rem}figure{margin:0;border:1px solid #ddd;padding:1rem}img{width:100%;height:auto}figcaption{overflow-wrap:anywhere;margin-bottom:.5rem}.scroll{overflow:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}details{margin:1rem 0}</style></head><body><p><a href="/">Home</a> · <a href="/graphs">Graph tracks</a></p>' + body + '</body></html>'
