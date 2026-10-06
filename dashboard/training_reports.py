"""Read training report plots from local run folders and downloaded suites."""
from html import escape
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode
import csv
import io
import json
import stat
import zipfile
import tarfile
from core.archive_reader import open_archive, archive_sidecar
from model_tracks import archive_verification

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse

router = APIRouter()
PROJECT = Path(__file__).resolve().parents[1]
REPORT_ERRORS = (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError, RuntimeError, EOFError)
METRIC_SUFFIXES = (
    'model_evaluation_summary.csv',
    'retrieval_summary.csv',
    'fold_metrics.csv',
    # Coverage/generalization slices and attribute separation were being
    # written on every track run and never displayed anywhere.
    'slice_metrics.csv',
    'attribute_separation_summary.csv',
    'attribute_separation_values.csv',
)
# Provenance contracts. These carry the honesty statement of a run
# (whether test labels were used for selection, what retrieval protocol was
# used, whether unlabeled pairs were treated as negatives), so they are part
# of the report surface, not internal bookkeeping.
MANIFEST_SUFFIXES = ('report_manifest.json', 'completion_manifest.json')
PERFORMANCE_SUFFIXES = ('performance.json',)
# ``report.json`` is written by TWO unrelated producers with different
# schemas: the legacy lane (confusion / attribute_errors / random_easy /
# score_overlap) and the attribute-ablation lane
# (er-attribute-ablation-report-v1). Both are kept here and dispatched by
# their declared ``schema`` at render time.
ERROR_SUFFIXES = ('report.json',)
# Artifacts kept for provenance/download but not rendered as a table.
AUDIT_ONLY_SUFFIXES = ('scored_pairs.csv',)
PROFILER_SUFFIXES = ('operator_summary.txt', 'profile_manifest.json')
LOG_PREVIEW_BYTES = 64 * 1024
COLLECTED_EVENTS = '__collected__/suite_events.jsonl'
# Directory-name suffix marking a DVC payload snapshot of a sibling tree.
# graph_tracks/dvc.py snapshots the whole completion directory into
# ``<track>__payload`` so `dvc add` sees the outputs, and that copy lives
# inside the run directory. The two copies are byte-identical, so without
# this every graph-track table and plot rendered twice and the artifact count
# doubled. Verified by SHA-256 on both smoke runs.
DUPLICATE_TREE_SUFFIX = '__payload'


def collected_events(path):
    if path.is_dir():
        return None
    sibling = archive_sidecar(path, '.events.jsonl')
    return sibling if sibling.is_file() and not sibling.is_symlink() else None


def is_log(member):
    relative = PurePosixPath(member)
    return (relative.name.endswith(('.log', '.jsonl')) or relative.name in
            {'live_status.json', 'trainer_state.json', 'checkpoint_manifest.json', 'best_checkpoint.json'}) and not any(
                part in {'wandb', 'mlruns', '.git', '.dvc'} for part in relative.parts)


def local_target(path, member):
    target = path / member
    if (not safe_member(member) or any(parent.is_symlink() for parent in
        [target, *target.parents] if parent.is_relative_to(path))
            or not target.resolve().is_relative_to(path.resolve()) or not target.is_file()):
        raise HTTPException(404, 'Report artifact not found')
    return target


def safe_member(member):
    relative = PurePosixPath(member)
    return bool(member) and not relative.is_absolute() and '..' not in relative.parts and '\\' not in member


def is_duplicate_copy(member):
    """True for members living inside a DVC payload snapshot of a sibling tree.

    ``graph_tracks/dvc.py`` copies the whole completion directory into
    ``<track>__payload`` so ``dvc add`` sees the outputs, and that copy sits
    inside the run directory. The two copies are byte-identical, so listing
    both doubled every table and plot on the page and inflated the artifact
    count. Verified by SHA-256 on both smoke runs.
    """
    parts = PurePosixPath(member).parts
    # The marker is a directory name; the sibling ``<track>__payload.dvc``
    # pointer file is metadata, not a duplicate copy, so only match the
    # directory form.
    return any(part.endswith(DUPLICATE_TREE_SUFFIX) and not part.endswith('.dvc')
               for part in parts)


def runs():
    found = {}
    for base in (PROJECT / 'results/model_tracks', PROJECT / 'results/graph_tracks',
                 PROJECT / 'training_results'):
        if not base.exists():
            continue
        for path in sorted(base.iterdir(), reverse=True):
            if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
                continue
            if path.is_dir() or (path.name.endswith(('.zip', '.tar.zst')) and '__inputs' not in path.name):
                key = path.relative_to(PROJECT).as_posix()
                if path.is_dir() and not any(
                        member.endswith(('.png',) + METRIC_SUFFIXES + ERROR_SUFFIXES
                                        + MANIFEST_SUFFIXES + PERFORMANCE_SUFFIXES
                                        + PROFILER_SUFFIXES)
                        or is_log(member)
                        for member in entries(path)):
                    continue
                found[key] = path
    return dict(sorted(found.items(), key=lambda item: item[1].stat().st_mtime, reverse=True))


def default_run(available):
    """The run to show when the user has not chosen one.

    Sorting by mtime alone puts the newest in-flight run first, and an
    in-flight run is a ``__logs`` directory with no metrics at all — so
    /training landed on an empty page while the only run carrying a complete
    three-track report sat fifth in the list. Prefer the most recent run that
    actually has something to show, then fall back to the newest.
    """
    if not available:
        return None
    scored = []
    for key, path in available.items():
        try:
            members = entries(path)
        except REPORT_ERRORS:
            continue
        has_metrics = any(m.endswith(METRIC_SUFFIXES + (".png",)) for m in members)
        scored.append((has_metrics, key))
    for has_metrics, key in scored:
        if has_metrics:
            return key
    return next(iter(available))


def entries(path):
    if path.is_dir():
        return [p.relative_to(path).as_posix() for p in path.rglob('*')
                if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(path.resolve())
                and not any(parent.is_symlink() for parent in p.parents if parent.is_relative_to(path))
                and safe_member(p.relative_to(path).as_posix())]
    with open_archive(path) as archive:
        members = [info.filename for info in archive.infolist() if safe_member(info.filename)
                and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)]
    if collected_events(path) is not None and COLLECTED_EVENTS not in members:
        members.append(COLLECTED_EVENTS)
    return members


@contextmanager
def open_artifact(path, member):
    if not safe_member(member):
        raise HTTPException(404, 'Report artifact not found')
    if member == COLLECTED_EVENTS and (sibling := collected_events(path)) is not None:
        with local_target(sibling.parent, sibling.name).open('rb') as handle:
            yield handle
        return
    if path.is_dir():
        with local_target(path, member).open('rb') as handle:
            yield handle
        return
    with open_archive(path) as archive:
        try:
            if stat.S_ISLNK(archive.getinfo(member).external_attr >> 16):
                raise HTTPException(404, 'Report artifact not found')
            with archive.open(member) as handle:
                yield handle
        except KeyError as error:
            raise HTTPException(404, 'Report artifact not found') from error


def read(path, member, limit=None):
    with open_artifact(path, member) as handle:
        return handle.read() if limit is None else handle.read(limit)


def log_preview(path, member):
    content = read(path, member, LOG_PREVIEW_BYTES + 1)
    truncated = len(content) > LOG_PREVIEW_BYTES
    text = content[:LOG_PREVIEW_BYTES].decode('utf-8', errors='replace')
    if truncated:
        text += f'\n\n[Preview truncated after {LOG_PREVIEW_BYTES:,} bytes. Download the full log.]'
    return text


@router.get('/training/log')
def log(run: str, artifact: str, download: bool = False):
    path = runs().get(run)
    if path is None or not (is_log(artifact) or is_downloadable_audit(artifact)
                        or is_profiler_artifact(artifact)):
        raise HTTPException(404, 'Training log not found')
    try:
        if artifact not in entries(path) or is_duplicate_copy(artifact):
            raise HTTPException(404, 'Training log not found')
        if not download:
            return Response(log_preview(path, artifact), media_type='text/plain')
        # Open once to validate before returning the response. Stream all bytes
        # on download instead of buffering an unbounded log in memory.
        with open_artifact(path, artifact):
            pass
        def chunks():
            with open_artifact(path, artifact) as handle:
                while chunk := handle.read(64 * 1024):
                    yield chunk
        return StreamingResponse(chunks(), media_type='text/plain', headers={
            'Content-Disposition': "attachment; filename*=UTF-8''" + quote(PurePosixPath(artifact).name, safe='')})
    except REPORT_ERRORS as error:
        raise HTTPException(404, 'Training log unavailable') from error


def is_downloadable_audit(member):
    """Row-level audit artifacts, offered for download but not rendered.

    ``scored_pairs.csv`` is the row-level evidence behind every metric table,
    so it belongs to the report surface — but it is one row per pair and would
    swamp the page if tabulated.
    """
    return PurePosixPath(member).name.endswith(AUDIT_ONLY_SUFFIXES)


def is_profiler_artifact(member):
    """Opt-in torch profiler output; audit evidence, served as a download."""
    return PurePosixPath(member).name.endswith(PROFILER_SUFFIXES)


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


@router.get('/training/error_analysis')
def error_analysis(run: str):
    path = runs().get(run)
    if path is None:
        raise HTTPException(404, 'Training run not found')
    try:
        members = entries(path)
    except REPORT_ERRORS:
        raise HTTPException(404, 'Training run unavailable')
    # Two unrelated producers write report.json with different schemas. Return
    # them all with their declared schema intact instead of whichever one
    # rglob happened to hit first, which silently hid the ablation lane
    # whenever the legacy report was also present.
    found = []
    for member in sorted(members):
        if not member.endswith(ERROR_SUFFIXES) or is_duplicate_copy(member):
            continue
        try:
            report = json.loads(read(path, member))
        except REPORT_ERRORS:
            continue
        if isinstance(report, dict):
            found.append({
                'artifact': member,
                'schema': report.get('schema', 'er-legacy-training-report'),
                'renderable': report.get('schema') in report_schemas() or bool(
                    set(report) & {'confusion', 'attribute_errors', 'random_easy',
                                   'score_overlap'}),
                'report': report,
            })
    if not found:
        raise HTTPException(404, 'No report.json found in this run')
    return found if len(found) > 1 else found[0]['report']


def _cell(value):
    """Render one JSON value for an HTML table cell.

    Containers are JSON-encoded rather than str()'d so a dict/list shows its
    structure instead of Python repr, and ``None`` stays visibly empty
    (an unmeasured metric) rather than becoming the string "None".
    """
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _csv_frame(payload):
    """Flatten a list of flat dicts into CSV bytes for the existing table()."""
    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows:
        return b''
    columns = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        if isinstance(row, dict):
            writer.writerow([_csv_value(row.get(column)) for column in columns])
        else:
            writer.writerow([_csv_value(row)])
    return buffer.getvalue().encode('utf-8')


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return ''
    return value


def _ablation_report_html(report):
    """Render er-attribute-ablation-report-v1.

    This schema was previously unreachable from the dashboard: the renderer
    only knew the legacy keys (confusion / attribute_errors / random_easy /
    score_overlap) and the two key sets have an EMPTY intersection, so the
    page printed the "Post-training error analysis" heading and then nothing
    at all -- silently discarding score_delta, decision_flip,
    embedding_cosine_delta, known_positive_recall_change and the threshold
    provenance.
    """
    out = [f'<h3>Attribute ablation: {escape(str(report.get("track", "?")))} '
           f'/ {escape(str(report.get("attribute", "?")))}</h3>']
    facts = [
        ('Schema', report.get('schema')),
        ('Intervention', report.get('intervention')),
        ('Retrieval intervention', report.get('retrieval_intervention')),
        ('Retrieval scope', report.get('retrieval_scope')),
        ('Retrieval catalog count', report.get('retrieval_catalog_count')),
        ('Split', report.get('split')),
        ('Composition', report.get('composition')),
        ('Changed listings', report.get('changed_listings')),
        ('Decision flip', report.get('decision_flip')),
        ('Endpoint input changed', report.get('endpoint_input_changed')),
        ('Embedding dtype', report.get('embedding_dtype')),
        ('Baseline / ablated score', f'{report.get("baseline_score")} / {report.get("ablated_score")}'),
        ('Score delta', report.get('score_delta')),
        ('Embedding cosine delta', report.get('embedding_cosine_delta')),
        ('Missing axes', ', '.join(map(str, report['missing_axes']))
            if isinstance(report.get('missing_axes'), list) else report.get('missing_axes')),
        ('Threshold', report.get('threshold')),
        ('Threshold binding', report.get('threshold_binding')),
        ('Threshold source', report.get('threshold_source')),
        ('Threshold provenance', report.get('threshold_provenance')),
        ('Ann baseline hits', report.get('ann_baseline_hits')),
        ('Ann ablated hits', report.get('ann_ablated_hits')),
        ('Checkpoint role', report.get('checkpoint_role')),
        ('Current attribute evidence', ', '.join(map(str, report['current_attribute_evidence']))
            if isinstance(report.get('current_attribute_evidence'), list)
            else report.get('current_attribute_evidence')),
    ]
    out.append('<table><tr><th>Field</th><th>Value</th></tr>')
    for label, value in facts:
        out.append(f'<tr><td>{escape(label)}</td><td>{escape(_cell(value))}</td></tr>')
    out.append('</table>')
    provenance = report.get('threshold_provenance')
    if isinstance(provenance, dict) and provenance:
        out.append('<h4>Threshold provenance</h4><table><tr><th>Key</th><th>Value</th></tr>')
        for key, value in sorted(provenance.items()):
            out.append(f'<tr><td>{escape(str(key))}</td><td>{escape(_cell(value))}</td></tr>')
        out.append('</table>')
    recall = report.get('known_positive_recall_change')
    if isinstance(recall, dict) and recall:
        out.append('<h4>Known-positive recall change</h4>'
                   '<table><tr><th>k</th><th>Baseline</th><th>Ablated</th><th>Change</th></tr>')
        for key in sorted(recall, key=lambda item: (len(str(item)), str(item))):
            row = recall[key]
            if isinstance(row, dict):
                out.append(f'<tr><td>{escape(str(key))}</td><td>{escape(_cell(row.get("baseline")))}</td>'
                           f'<td>{escape(_cell(row.get("ablated")))}</td>'
                           f'<td>{escape(_cell(row.get("change", row.get("delta"))))}</td></tr>')
            else:
                out.append(f'<tr><td>{escape(str(key))}</td><td colspan="3">{escape(_cell(row))}</td></tr>')
        out.append('</table>')
    rows = report.get('rows')
    if isinstance(rows, list) and rows:
        flips = [r for r in rows if isinstance(r, dict) and r.get('decision_flip')]
        if flips:
            out.append(f'<h4>Decision flips ({len(flips)} of {len(rows)} rows)</h4>'
                       '<p>Pairs whose decision changed when the declared attribute was '
                       'removed. Gate reason is unrecorded on these rows in the current '
                       'reporter; evidence, masking and variant are the available context.</p>'
                       '<div class="scroll"><table><tr><th>attribute</th><th>channel</th>'
                       '<th>pair (gtin)</th><th>pair (sku)</th><th>label</th><th>split</th>'
                       '<th>score delta</th><th>baseline / ablated</th><th>evidence</th>'
                       '<th>masking</th><th>variant</th><th>changed listings</th>'
                       '<th>target mode</th></tr>')
            for r in sorted(flips, key=lambda item: (str(item.get('attribute')),
                                                     str(item.get('gtin1')),
                                                     str(item.get('gtin2')))):
                out.append(
                    f'<tr><td>{escape(str(r.get("attribute")))}</td>'
                    f'<td>{escape(str(r.get("channel")))}</td>'
                    f'<td>{escape(str(r.get("gtin1")))}/{escape(str(r.get("gtin2")))}</td>'
                    f'<td>{escape(str(r.get("sku_id1")))}/{escape(str(r.get("sku_id2")))}</td>'
                    f'<td>{escape(str(r.get("label")))}</td>'
                    f'<td>{escape(str(r.get("split")))}</td>'
                    f'<td>{escape(_cell(r.get("score_delta")))}</td>'
                    f'<td>{escape(_cell(r.get("baseline_score")))}/{escape(_cell(r.get("ablated_score")))}</td>'
                    f'<td>{escape(_cell(r.get("current_attribute_evidence")))}</td>'
                    f'<td>{escape(str(r.get("masking_profile")))}</td>'
                    f'<td>{escape(str(r.get("generation_variant")))}</td>'
                    f'<td>{escape(str(r.get("changed_listings")))}</td>'
                    f'<td>{escape(str(r.get("target_mode")))}</td></tr>')
            out.append('</table></div>')
        out.append(f'<h4>Per-row deltas ({len(rows)})</h4><div class="scroll">')
        out.append(table(_csv_frame(json.dumps(rows))))
        out.append('</div>')
    return ''.join(out)


def _legacy_report_html(report):
    """Render the legacy generate_training_report report.json."""
    out = []
    if 'confusion' in report:
        out.append('<h3>Confusion matrices</h3><table><tr><th>Operating point</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>')
        for op, cm in report['confusion'].items():
            out.append(f'<tr><td>{escape(op)}</td><td>{cm.get("tp", "?")}</td><td>{cm.get("fp", "?")}</td><td>{cm.get("fn", "?")}</td><td>{cm.get("tn", "?")}</td><td>{cm.get("accuracy", "?")}</td><td>{cm.get("precision", "?")}</td><td>{cm.get("recall", "?")}</td><td>{cm.get("f1", "?")}</td></tr>')
        out.append('</table>')
    if 'attribute_errors' in report:
        out.append('<h3>Attribute error rates</h3><table><tr><th>Attribute</th><th>Error rate</th><th>Errors</th><th>Mean score</th><th>n</th></tr>')
        for attr, ae in sorted(report['attribute_errors'].items()):
            out.append(f'<tr><td>{escape(attr)}</td><td>{ae.get("error_rate", "?")}</td><td>{ae.get("errors", "?")}</td><td>{ae.get("mean_score", "?")}</td><td>{ae.get("n", "?")}</td></tr>')
        out.append('</table>')
    if 'random_easy' in report:
        out.append('<h3>Score distributions</h3><table><tr><th>Population</th><th>Mean</th><th>Median</th><th>n</th></tr>')
        for pop, stats in report['random_easy'].items():
            out.append(f'<tr><td>{escape(pop)}</td><td>{stats.get("mean", "?")}</td><td>{stats.get("median", "?")}</td><td>{stats.get("n", "?")}</td></tr>')
        out.append('</table>')
    if 'score_overlap' in report:
        out.append('<h3>Score overlap</h3><table><tr><th>Split</th><th>Label 0 mean</th><th>Label 0 n</th><th>Label 1 mean</th><th>Label 1 n</th><th>Overlap coefficient</th></tr>')
        for split, stats in report['score_overlap'].items():
            out.append(f'<tr><td>{escape(split)}</td><td>{stats.get("label_0_mean", "?")}</td><td>{stats.get("label_0_n", "?")}</td><td>{stats.get("label_1_mean", "?")}</td><td>{stats.get("label_1_n", "?")}</td><td>{stats.get("overlap_coefficient", "?")}</td></tr>')
        out.append('</table>')
    return ''.join(out)


def report_schemas():
    """``report.json`` producers, dispatched by the schema they declare.

    Built on demand rather than at import: the renderers are defined below
    this point in the module.
    """
    return {
        'er-attribute-ablation-report-v1': _ablation_report_html,
        'er-track-report-manifest-v1': _manifest_html,
    }


def render_report(report):
    """Render any known report.json, by declared schema.

    Falls back to a generic key/value dump for an unrecognised schema so a new
    producer is visibly unhandled rather than silently rendering as empty --
    which is exactly the failure mode this dispatch exists to remove.
    """
    if not isinstance(report, dict):
        return '<p>Report is not a JSON object.</p>'
    schema = report.get('schema')
    renderer = report_schemas().get(schema) if isinstance(schema, str) else None
    if renderer is not None:
        return renderer(report)
    if any(key in report for key in ('confusion', 'attribute_errors', 'random_easy',
                                     'score_overlap')):
        return _legacy_report_html(report)
    keys = sorted(report)
    return ('<p>This report schema is not rendered yet; its fields are listed below.</p>'
            '<table><tr><th>Field</th><th>Value</th></tr>'
            + ''.join(f'<tr><td>{escape(str(key))}</td><td>{escape(_cell(report[key]))}</td></tr>'
                      for key in keys) + '</table>')


def _manifest_html(report):
    """Render a track report/completion manifest: the honesty contract."""
    out = ['<h3>Report manifest</h3>', _honesty_html(report)]
    summary = report.get('summary')
    if isinstance(summary, list) and summary:
        out.append('<h4>Pair metrics</h4><div class="scroll">')
        out.append(table(_csv_frame(json.dumps(summary))))
        out.append('</div>')
    slices = report.get('slices')
    if isinstance(slices, list) and slices:
        out.append('<h4>Generalization slices</h4><div class="scroll">')
        out.append(table(_csv_frame(json.dumps(slices))))
        out.append('</div>')
    intervals = report.get('confidence_intervals')
    if isinstance(intervals, dict) and intervals:
        out.append(_intervals_html(intervals))
    performance = report.get('performance')
    if isinstance(performance, dict) and performance:
        out.append(_performance_html(performance))
    return ''.join(out)


#: Fields that state what a run is and is NOT allowed to have done.
HONESTY_FIELDS = (
    'track', 'checkpoint', 'threshold', 'threshold_source',
    'test_used_for_selection', 'test_reported', 'model_selection',
    'metrics_scope', 'graph_context', 'trained_endpoints_scored',
    'unlabeled_pairs_are_negatives', 'identity_conflict_policy_applied',
    'retrieval_protocol', 'retrieval_ks', 'schema',
)


def _honesty_html(report):
    rows = ''.join(
        f'<tr><td>{escape(str(key))}</td><td>{escape(_cell(report.get(key)))}</td></tr>'
        for key in HONESTY_FIELDS if key in report)
    return ('<table><caption>What this run measured</caption>'
            '<tr><th>Field</th><th>Value</th></tr>' + rows + '</table>')


def _intervals_html(intervals):
    out = ['<h4>Paired bootstrap confidence intervals</h4>']
    for split in sorted(intervals):
        block = intervals[split]
        if not isinstance(block, dict):
            continue
        note = block.get('repeated_training_seeds_note')
        seeds = block.get('repeated_training_seeds')
        out.append(f'<p><strong>{escape(split)}</strong> — method {escape(str(block.get("method")))}, '
                   f'resamples {escape(str(block.get("resamples_requested")))}, '
                   f'confidence {escape(str(block.get("confidence")))}, '
                   f'repeated training seeds: {escape(str(seeds))}</p>')
        if note:
            out.append(f'<p><em>{escape(str(note))}</em></p>')
        metrics = block.get('metrics')
        if isinstance(metrics, dict) and metrics:
            out.append('<table><tr><th>Metric</th><th>Point</th><th>Low</th><th>High</th><th>Resamples used</th></tr>')
            for metric in sorted(metrics):
                row = metrics[metric]
                if not isinstance(row, dict):
                    continue
                out.append(f'<tr><td>{escape(str(metric))}</td><td>{escape(_cell(row.get("point")))}</td>'
                           f'<td>{escape(_cell(row.get("low")))}</td><td>{escape(_cell(row.get("high")))}</td>'
                           f'<td>{escape(_cell(row.get("resamples_used")))}</td></tr>')
            out.append('</table>')
    return ''.join(out)


def _performance_html(performance):
    """Render operational cost: latency, memory, refresh."""
    out = ['<h4>Operational cost</h4>']
    torch_profile = performance.get('torch_profile')
    if isinstance(torch_profile, dict):
        out.append('<p>Opt-in PyTorch trace profile (covers at most '
                   f'{escape(str(torch_profile.get("covers_at_most_steps", 3)))} steps, '
                   'includes profiling overhead):</p>')
        top = torch_profile.get('top_self_time_seconds')
        if isinstance(top, dict) and top:
            out.append('<table><tr><th>Operator</th><th>Self seconds</th></tr>')
            for name, value in top.items():
                out.append(f'<tr><td>{escape(str(name))}</td><td>{escape(_cell(value))}</td></tr>')
            out.append('</table>')
    out.append('<table><tr><th>Peak RSS (MB)</th><th>Peak CUDA allocated (MB)</th></tr>')
    out.append(f'<tr><td>{escape(_cell(performance.get("peak_rss_mb")))}</td>'
               f'<td>{escape(_cell(performance.get("peak_cuda_allocated_mb")))}</td></tr></table>')
    sections = performance.get('sections')
    if isinstance(sections, dict) and sections:
        out.append('<table><tr><th>Phase</th><th>Calls</th><th>Total s</th>'
                   '<th>Median s</th><th>P95 s</th><th>Max s</th></tr>')
        for label in sorted(sections):
            row = sections[label]
            if not isinstance(row, dict):
                continue
            out.append(f'<tr><td>{escape(str(label))}</td><td>{escape(_cell(row.get("calls")))}</td>'
                       f'<td>{escape(_cell(row.get("total_seconds")))}</td>'
                       f'<td>{escape(_cell(row.get("median_seconds")))}</td>'
                       f'<td>{escape(_cell(row.get("p95_seconds")))}</td>'
                       f'<td>{escape(_cell(row.get("max_seconds")))}</td></tr>')
        out.append('</table>')
    missing = performance.get('missing_required_sections')
    if missing:
        out.append('<p>Not measured in this run: '
                   f'{escape(", ".join(map(str, missing)))}. '
                   'Latency, memory and refresh time are required by '
                   'the model plan.</p>')
    return ''.join(out)


def _verification_html(result):
    """The archived post-download verification outcome, with the failure inline."""
    status = str(result.get('status', 'unknown'))
    color = {'verified': '#1a7f37', 'failed': '#cf222e', 'unreadable': '#9a6700'}.get(status, '#57606a')
    body = (f'<details open><summary style="color:{color};font-weight:600">'
            f'Archive verification: {escape(status)} '
            f'({escape(str(result.get("verified_at", "unknown")))})</summary>')
    sha = result.get('zip_sha256') or {}
    if sha.get('match') is True:
        body += '<p>sha256 sidecar: match</p>'
    elif sha.get('match') is False:
        body += (f'<p style="color:#cf222e">sha256 sidecar: MISMATCH — expected '
                 f'{escape(str(sha.get("expected")))} · actual {escape(str(sha.get("actual")))}</p>')
    else:
        body += '<p>sha256 sidecar: not present</p>'
    if result.get('error'):
        body += f'<pre style="overflow:auto">{escape(str(result["error"]))}</pre>'
    tracks = result.get('tracks') or {}
    if tracks:
        body += ('<table><tr><th>track</th><th>report</th><th>threshold</th>'
                 '<th>checkpoint sha256</th><th>test reported</th><th>ablation</th></tr>')
        for track, entry in tracks.items():
            ablation = 'bound' if entry.get('ablation') else '—'
            body += (f'<tr><td>{escape(str(track))}</td><td>{escape(str(entry.get("report")))}</td>'
                     f'<td>{escape(str(entry.get("threshold")))}</td>'
                     f'<td>{escape(str(entry.get("checkpoint_sha256")))}</td>'
                     f'<td>{escape(str(entry.get("test_reported")))}</td><td>{ablation}</td></tr>')
        body += '</table>'
    body += '</details>'
    return body


@router.get('/training/verify')
def verify_run(run: str):
    """Re-verify a downloaded suite archive and record the outcome as a sidecar."""
    available = runs()
    if run not in available:
        raise HTTPException(404, 'Training run not found')
    path = available[run]
    if path.is_dir():
        raise HTTPException(400, 'Verification applies to downloaded suite archives, not local run folders')
    result = archive_verification.verification_result(path)
    archive_verification.write_verification(path, result)
    return RedirectResponse('/training?' + urlencode({'run': run}), status_code=302)


@router.get('/training', response_class=HTMLResponse)
def training(run: str | None = None):
    available = runs()
    if run is not None and run not in available:
        raise HTTPException(404, 'Training run not found')
    selected = run or default_run(available)
    options = ''.join(f'<option value="{escape(key, quote=True)}"'
                      f'{" selected" if key == selected else ""}>{escape(key)}</option>'
                      for key in available)
    body = '<h1>Training reports</h1><p>Saved metrics and plots for text, GNN-only and hybrid runs. Smoke runs verify the workflow; their scores are not model quality benchmarks.</p>'
    if selected is None:
        body += '<p>No downloaded training reports yet.</p>'
    else:
        body += f'<form><label>Run <select name="run">{options}</select></label> <button>Show reports</button></form>'
        path = available[selected]
        if not path.is_dir():
            # Post-download verification outcome: the sealing-time contract
            # was checked on the machine that wrote the archive; this sidecar
            # records the re-check of the bytes that actually arrived.
            verification = archive_verification.load_verification(path)
            if verification is None:
                verify_url = '/training/verify?' + urlencode({'run': selected})
                body += (f'<p><a href="{escape(verify_url, quote=True)}">Verify archive</a> — re-checks the '
                         f'sha256 sidecar and the sealing-time contract, then records the outcome as '
                         f'{escape(archive_sidecar(path, ".verification.json").name)}.</p>')
            else:
                body += _verification_html(verification)
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
        # Duplicate DVC payload copies are excluded everywhere below: they are
        # byte-identical snapshots, and listing them doubled every table.
        visible = [m for m in members if not is_duplicate_copy(m)]
        hidden = len(members) - len(visible)
        if hidden:
            body += (f'<p>Excluded {hidden} byte-identical DVC payload copies '
                     'of these artifacts.</p>')
        plots = sorted(m for m in visible if m.endswith('.png'))
        summaries = sorted(m for m in visible if m.endswith(METRIC_SUFFIXES))
        manifests = sorted(m for m in visible if m.endswith(MANIFEST_SUFFIXES))
        performances = sorted(m for m in visible if m.endswith(PERFORMANCE_SUFFIXES))
        audit_only = sorted(m for m in visible if m.endswith(AUDIT_ONLY_SUFFIXES))
        error_reports = sorted(m for m in visible if m.endswith(ERROR_SUFFIXES))
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
        for member in manifests:
            try:
                manifest = json.loads(read(path, member))
            except REPORT_ERRORS:
                body += f'<p>{escape(member)} could not be read.</p>'
                continue
            body += f'<details open><summary>{escape(member)}</summary>{_manifest_html(manifest)}</details>'
        for member in performances:
            try:
                performance = json.loads(read(path, member))
            except REPORT_ERRORS:
                body += f'<p>{escape(member)} could not be read.</p>'
                continue
            body += f'<details open><summary>{escape(member)}</summary>{_performance_html(performance)}</details>'
        profiles = sorted(m for m in visible if m.endswith(PROFILER_SUFFIXES))
        if profiles:
            body += ('<h2>Operator profile</h2><p>Opt-in torch profiler output. It covers '
                     'at most three active steps and carries profiling overhead, so it is '
                     'reported separately from the wall-clock sections above.</p>')
            for member in profiles:
                url = '/training/log?' + urlencode({'run': selected, 'artifact': member,
                                                    'download': 'true'})
                if member.endswith('profile_manifest.json'):
                    try:
                        body += (f'<details><summary>{escape(member)}</summary>'
                                 f'<a href="{escape(url, quote=True)}">Download</a>'
                                 f'<pre>{escape(json.dumps(json.loads(read(path, member)), indent=2))}</pre></details>')
                        continue
                    except REPORT_ERRORS:
                        pass
                try:
                    table_text = read(path, member).decode('utf-8', errors='replace')
                except REPORT_ERRORS:
                    table_text = 'Profile could not be read.'
                body += (f'<details><summary>{escape(member)}</summary>'
                         f'<a href="{escape(url, quote=True)}">Download</a>'
                         f'<pre style="overflow:auto">{escape(table_text)}</pre></details>')
        if audit_only:
            body += ('<h2>Scored pairs</h2><p>Row-level scores behind the metrics '
                     'above, kept for audit rather than rendered inline.</p><ul>')
            for member in audit_only:
                query = {'run': selected, 'artifact': member}
                url = '/training/log?' + urlencode({**query, 'download': 'true'})
                body += f'<li>{escape(member)} · <a href="{escape(url, quote=True)}">Download</a></li>'
            body += '</ul>'
        for member in error_reports:
            try:
                report = json.loads(read(path, member))
            except REPORT_ERRORS:
                body += f'<p>{escape(member)} could not be read.</p>'
                continue
            body += '<h2>Post-training error analysis</h2>'
            rendered = render_report(report)
            if not rendered.strip():
                body += (f'<p>{escape(member)} declared no renderable sections.</p>')
            else:
                body += rendered
        logs = sorted(member for member in visible if is_log(member))
        body += '<h2>Training logs</h2><p>Durable structured events and raw worker output. Refresh to see newly saved events in local runs.</p>'
        for member in logs:
            query = {'run': selected, 'artifact': member}
            preview_url = '/training/log?' + urlencode(query)
            download_url = '/training/log?' + urlencode({**query, 'download': 'true'})
            try:
                preview = log_preview(path, member)
            except REPORT_ERRORS:
                preview = 'Log could not be read.'
            body += (f'<details><summary>{escape(member)}</summary><p>'
                     f'<a href="{escape(preview_url, quote=True)}">Open preview</a> · '
                     f'<a href="{escape(download_url, quote=True)}">Download full log</a></p>'
                     f'<pre style="overflow:auto;max-height:30rem;white-space:pre-wrap">{escape(preview)}</pre></details>')
        if not logs:
            body += '<p>No saved training logs in this run.</p>'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ER training reports</title><style>body{font-family:system-ui;margin:2rem;color:#222}select{max-width:75vw}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:1rem}figure{margin:0;border:1px solid #ddd;padding:1rem}img{width:100%;height:auto}figcaption{overflow-wrap:anywhere;margin-bottom:.5rem}.scroll{overflow:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}details{margin:1rem 0}</style></head><body><p><a href="/">Home</a> · <a href="/graphs">Graph tracks</a></p>' + body + '</body></html>'
