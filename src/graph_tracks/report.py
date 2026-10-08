"""Dev-fit threshold, final test metrics and known-positive retrieval reports.

Uses the ER evaluation metric names and pooled ranking implementation. Pair
ranking is explicitly pooled: it is not mislabeled as catalog retrieval.
"""
from __future__ import annotations
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve,
                             roc_auc_score, roc_curve)
from core.common import plot_dpi, precision_at_recall_key
from core.coverage_contracts import (
    UNKNOWN_DIMENSION_VALUE, DimensionAccounting, ReportCoverageContract,
    TaggedDimensionRecord,
)
from core.ranking_metrics import POOLED_METRIC_PREFIX, ranking_at_k
from graph_tracks.artifacts import name
from graph_tracks.data import file_size, load_records, load_text_cache

#: The cascade is a combinator: text retrieves, gnn_only decides, and neither
#: role owns an encoded listing catalog. It therefore scores no
#: generalization-slice or attribute-separation population of its own. These
#: notes make that explicit so the cascade manifest carries the same
#: ``slices``/``attributes`` keys as the trained lanes without pretending a
#: population was measured.
CASCADE_TRACEABILITY = {
    'slices': ('not applicable: the cascade scores only retrieved candidate '
               'pairs; it has no encoded listing catalog to classify into the '
               'unseen / sparse-neighborhood / isolated / missing-field slices'),
    'attributes': ('not applicable: the cascade composes the trained lanes and '
                   'carries no listing-attribute table of its own; per-attribute '
                   'separation is reported by the text and gnn_only lanes'),
}


def dev_threshold(labels, scores):
    if set(labels) != {0., 1.}:
        raise ValueError('threshold calibration needs both classes on dev')
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def recall_at_precision(precision, recall, target):
    """Highest recall reachable while holding precision at or above ``target``.

    ``precision_recall_curve`` returns the sentinel ``(precision=1,
    recall=0)`` as its last element, so the body is sliced off first: without
    that, an unreachable target would always match the sentinel and report a
    confident 0.0 instead of "no such operating point".

    Returns None when no operating point clears the target, which is the
    honest answer -- the agreement is unmeetable on this split.
    """
    body = slice(0, len(precision) - 1)
    ok = precision[body] >= target
    return float(recall[body][ok].max()) if ok.any() else None


def pair_metrics(labels, scores, threshold, ks):
    from core.common import operating_precision, operating_recall, precision_at_recall_key

    labels = np.asarray(labels, dtype=int)
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions,
        average='binary', zero_division=0)
    supported = set(labels) == {0, 1}
    p, r, _ = precision_recall_curve(labels, scores) if supported else (None, None, None)
    pooled = ranking_at_k(labels, scores, tuple(ks)) if labels.any() else {}
    agreed = operating_precision()
    target_recall = operating_recall()
    precision_at_target = float(p[:-1][r[:-1] >= target_recall].max()) if supported else None
    return {'rows': len(labels), 'positive_pairs': int(labels.sum()),
        'negative_pairs': int((labels == 0).sum()), 'threshold': threshold,
        'roc_auc': float(roc_auc_score(labels, scores)) if supported else None,
        'pr_auc': float(average_precision_score(labels, scores)) if supported else None,
        'agreed_recall': target_recall,
        'precision_at_recall': precision_at_target,
        precision_at_recall_key(): precision_at_target,
        # The model plan asks for "recall at an agreed precision". The
        # agreement is config SSOT (evaluation.operating_precision) and is
        # echoed into every row so the number is self-documenting.
        'agreed_precision': agreed,
        'recall_at_precision': recall_at_precision(p, r, agreed) if supported else None,
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision_defined': bool(tp + fp), 'both_classes': supported,
        **{key: value for key, value in pooled.items() if key.startswith('pooled_')}}


#: The metric columns ``pair_metrics`` emits under a FIXED, name-stable header.
#: DERIVED from a probe CALL, never retyped: the emitter itself is the registry,
#: so a metric added or dropped next to ``pair_metrics`` moves every consumer at
#: once. A hand-typed second registry is the defect pinned by
#: tests/test_column_ssot.py, and a *validated* copy of the emitter's header is
#: exactly what silently drifts from it.
#:
#: The two DYNAMIC families are excluded because their names are decided by
#: config at emission time, not by the emitter's own shape:
#:
#:   * ``precision_at_recall_key()`` (``p_at_r<recall>``) -- retuning
#:     ``evaluation.operating_recall`` renames the column;
#:   * ``pooled_*`` from ``ranking_at_k`` -- the ladder is ``retrieval_ks``, and
#:     a slice with no positive pair emits none of them.
#:
#: The slice adapter matches those two families by PATTERN (see
#: ``graph_tracks.report_slices.DYNAMIC_METRIC_PATTERNS``); consumers that need
#: the complete emitted header should call ``pair_metrics`` itself.
PAIR_METRIC_KEYS: tuple[str, ...] = tuple(
    key for key in pair_metrics(np.array([0, 1]), np.array([0.1, 0.9]), 0.5, ())
    if not key.startswith(POOLED_METRIC_PREFIX)
    and key != precision_at_recall_key()
)


def retrieval_report(records, vectors, pairs, output, track, cfg, *, perf=None):
    """Same-split eligible catalogs; no trained endpoints; incomplete truth explicit.

    The per-split catalogs are built into a temporary directory: they exist
    only to answer the queries below, and persisting them used to write a
    second full HNSW index per split into the report directory alongside the
    track's real ``<track>__index`` -- double the index bytes on disk and a
    second object the DVC snapshot then copied into ``<track>__payload``.
    """
    from training.hnsw_index import PersistentHnswIndex
    rows = []
    with tempfile.TemporaryDirectory(prefix=f'{track}_retrieval_') as scratch:
        scratch_path = Path(scratch)
        for split in ('dev', 'test'):
            if split == 'test' and not cfg.report_test:
                continue
            indices, labels = pairs[split]
            positive = indices[labels == 1]
            if not len(positive):
                continue
            from model_tracks.training_data import retrieval_indices
            targets = [i for i in retrieval_indices(records) if records[i]['split'] == split]
            relevant = {}
            for left, right in positive:
                relevant.setdefault(int(left), set()).add(int(right))
                relevant.setdefault(int(right), set()).add(int(left))
            index_started = time.monotonic()
            index = PersistentHnswIndex(scratch_path / f'{split}_index',
                ef_construction=cfg.hnsw_ef_construction, M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
            # No encoder checkpoint claim: report provenance sizes are supplied separately.
            checkpoint = cfg.checkpoint
            index.build(vectors[targets], [records[i]['sku_id'] for i in targets],
                        checkpoint=checkpoint, model_name=track,
                        preprocessing_fingerprint=cfg.listings_size)
            build_seconds = time.monotonic() - index_started
            query_started = time.monotonic()
            query_ids = sorted(relevant)
            rankings, _ = index.query(vectors[query_ids], top_k=min(max(cfg.retrieval_ks) + 1, len(targets)))
            query_seconds = time.monotonic() - query_started
            if perf is not None:
                perf.record('index_build', build_seconds)
                perf.record('query', query_seconds)
                perf.count('queries', len(query_ids))
            for query, ranking in zip(query_ids, rankings):
                candidates = [targets[int(label)] for label in ranking if targets[int(label)] != query]
                for k in cfg.retrieval_ks:
                    recovered = len(set(candidates[:k]) & relevant[query])
                    rows.append({'split': split, 'sku_id': records[query]['sku_id'], 'k': k,
                        'eligible_catalog': len(targets) - 1, 'known_relevant': len(relevant[query]),
                        'retrieved': len(candidates[:k]), 'recovered': recovered,
                        'known_positive_recall': recovered / len(relevant[query]),
                        'known_positive_hit': int(recovered > 0)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / name(track, 'retrieval_queries.csv'), index=False)
    summary = []
    if len(frame):
        for (split, k), group in frame.groupby(['split', 'k']):
            summary.append({'split': split, 'k': int(k), 'queries': len(group),
                'known_positive_recall': float(group.known_positive_recall.mean()),
                'hits': float(group.known_positive_hit.mean()),
                'minimum_eligible_catalog': int(group.eligible_catalog.min()),
                'maximum_eligible_catalog': int(group.eligible_catalog.max()),
                'budget_covers_entire_catalog': bool((group.eligible_catalog <= k).all())})
    pd.DataFrame(summary).to_csv(output / name(track, 'retrieval_summary.csv'), index=False)
    return summary


def attribute_summary_rows(report_dir: Path, track: str) -> list:
    """Read back the attribute-separation table the lane just wrote.

    ``report_attributes.write_reports`` owns the CSV; the manifest only records
    the same rows so the per-track contract carries the attribute traceability
    table alongside the generalization slices. ``to_json``/``json.loads``
    round-trips the frame so numpy scalars and NaN become JSON-native values in
    one place (the manifest forbids inf/nan).
    """
    path = report_dir / name(track, 'attribute_separation_summary.csv')
    if not path.is_file():
        return []
    return json.loads(pd.read_csv(path).to_json(orient='records'))


def complete(checkpoint: Path, listings: Path, pair_path: Path, output: Path, cfg, *, text_cache=None, saved_inference=None):
    from graph_tracks.train import load_pairs
    records = load_records(listings)
    pairs = load_pairs(pair_path, records)
    from graph_tracks.artifacts import checkpoint_track
    from core.bootstrap_ci import paired_bootstrap
    from core.performance import PerformanceRecorder
    track = checkpoint_track(checkpoint)
    perf = PerformanceRecorder(track)
    def progress(phase, **details):
        print(f'[postprocess/{track}] ' + json.dumps({'phase': phase, **details}, default=str), flush=True)
    if saved_inference is None:
        from graph_tracks.infer import forward_outputs
        inference = forward_outputs(checkpoint, listings, pair_path,
            output / name(track, 'inference'), cfg, text_cache=text_cache)
    else:
        inference = Path(saved_inference)
    from graph_tracks.artifacts import GraphForwardManifest
    manifest = GraphForwardManifest.model_validate_json(
        (inference / name(track, 'export_manifest.json')).read_text()).model_dump(by_alias=True)
    if manifest.get('track') != track or not manifest.get('forward_only'):
        raise ValueError('saved graph inference track/forward contract mismatch')
    for key, path in [('checkpoint_size', checkpoint), ('listings_size', listings),
                      ('pairs_size', pair_path), ('vectors_size', inference / name(track, 'vectors.npz')),
                      ('split_scores_size', inference / name(track, 'split_scores.npz'))]:
        if manifest.get(key) != file_size(path):
            raise ValueError(f'saved graph inference mismatch: {key}')
    if cfg.report_test and not manifest['report_test']:
        raise ValueError('saved graph forward omitted requested test scores')
    perf.adopt('encode', manifest.get('performance', {}).get('sections', {}).get('encode', {}))
    with np.load(inference / name(track, 'vectors.npz'), allow_pickle=False) as cache:
        if cache['ids'].astype(str).tolist() != [r['sku_id'] for r in records]:
            raise ValueError('saved graph inference ID order mismatch')
        vectors = cache['embeddings']
    if vectors.shape != (manifest['count'], manifest['dimension']):
        raise ValueError('saved graph catalog vector shape differs from manifest')
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('saved graph inference vector dtype/norm mismatch')
    scores = {}
    with np.load(inference / name(track, 'split_scores.npz'), allow_pickle=False) as cache:
        for split in ('dev', 'test'):
            if split == 'test' and not cfg.report_test:
                continue
            if len(pairs[split][0]):
                if split not in cache or cache[split].shape != (len(pairs[split][0]),):
                    raise ValueError('saved graph pair score population mismatch')
                scores[split] = cache[split]
                if scores[split].dtype != np.float32 or not np.isfinite(scores[split]).all() or np.any(scores[split] < 0) or np.any(scores[split] > 1):
                    raise ValueError('saved graph pair score dtype/finite mismatch')
    if cfg.build_index:
        from training.hnsw_index import PersistentHnswIndex
        index_path = output / name(track, 'index')
        index_started = time.monotonic()
        index = PersistentHnswIndex(index_path, ef_construction=cfg.hnsw_ef_construction,
            M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
        from model_tracks.training_data import retrieval_indices
        catalog_indices = retrieval_indices(records)
        index.build(vectors[catalog_indices], [records[i]['sku_id'] for i in catalog_indices], checkpoint=checkpoint,
                    model_name=track, preprocessing_fingerprint=file_size(listings))
        perf.record('index_build', time.monotonic() - index_started)
    progress('saved_forward_validated', shape=list(vectors.shape), inference=str(inference))
    threshold = dev_threshold(pairs['dev'][1], scores['dev'])
    progress('threshold_selected', source='dev_youden', threshold=float(threshold), test_used=False)
    summary, scored_rows = [], []
    intervals = {}
    for split, values in scores.items():
        indices, labels = pairs[split]
        summary.append({'model': track, 'split': split, 'threshold_source': 'dev_youden',
                        'checkpoint': checkpoint.name, **pair_metrics(labels, values, threshold, cfg.retrieval_ks)})
        intervals[split] = paired_bootstrap(labels, values, track=track, split=split)
        for (left, right), label, score in zip(indices, labels, values):
            scored_rows.append({'sku_id1': records[left]['sku_id'],
                'sku_id2': records[right]['sku_id'], 'true_label': int(label),
                'split': split, 'score': float(score), 'prediction': int(score >= threshold)})
    report_dir = output / name(track, 'reports')
    report_dir.mkdir()
    pd.DataFrame(summary).to_csv(report_dir / name(track, 'model_evaluation_summary.csv'), index=False)
    scored = pd.DataFrame(scored_rows)
    scored.to_csv(report_dir / name(track, 'scored_pairs.csv'), index=False)
    from graph_tracks.report_attributes import write_reports as write_attribute_reports
    write_attribute_reports(listings, records, pairs, scores, report_dir, track)
    progress('attribute_reports_complete', output=str(report_dir))
    # The model plan requires unseen / sparse-neighborhood / isolated /
    # missing-field slices; only the attribute slices existed before this.
    # ``report`` validates its own rows ONCE, before persisting them; the
    # manifest below carries those already-validated rows verbatim, so no
    # second check of the same list runs here or in the manifest writer.
    from graph_tracks.report_slices import report as slice_report
    slices = slice_report(records, scored, track=track, output=report_dir,
                          pair_metrics=pair_metrics, threshold=threshold,
                          ks=tuple(cfg.retrieval_ks))
    progress('slice_reports_complete', slices=[r['slice'] for r in slices])
    # Pass immutable provenance separately rather than adding undeclared config fields.
    from graph_tracks.config import RetrievalReportContext
    retrieval_cfg = RetrievalReportContext.from_config(cfg, checkpoint, file_size(listings))
    progress('retrieval_started', ks=list(cfg.retrieval_ks), protocol='within-split, self excluded')
    retrieval = retrieval_report(records, vectors, pairs, report_dir, track, retrieval_cfg, perf=perf)
    progress('retrieval_complete', summary=retrieval)
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    # TrainingProfiler is opt-in and writes next to the run; fold it in so the
    # operator table is not left sitting unread in the run directory.
    from core.performance import summarize_profiler_directory, summarize_refresh_timings
    # Refresh is measured by the trainer, in another process; adopt its
    # aggregate so it stops being reported as a missing required section.
    perf.adopt('refresh', summarize_refresh_timings(output.parent).get('refresh', {}))
    performance = perf.summary()
    # TrainingProfiler is opt-in and writes next to the run; fold it in so the
    # operator table is not left sitting unread in the run directory.
    profile = next((parent / name(track, 'profile') for parent in checkpoint.parents
                    if (parent / name(track, 'profile')).is_dir()), None)
    if profile is not None:
        performance.update(summarize_profiler_directory(profile))
    write_manifest(report_dir / name(track, 'report_manifest.json'), build_manifest(
        track=track, checkpoint=checkpoint, checkpoint_size=file_size(checkpoint),
        listings_size=file_size(listings), pairs_size=file_size(pair_path),
        threshold=threshold, threshold_source='dev_youden',
        test_reported='test' in scores, model_selection='dev_pr_auc',
        retrieval_ks=cfg.retrieval_ks, summary=summary, retrieval=retrieval,
        slices=slices, attributes=attribute_summary_rows(report_dir, track),
        performance=performance, confidence_intervals=intervals))
    _plots(scored, report_dir, track, threshold)
    progress('plots_complete', plots=[str(path) for path in sorted(report_dir.glob('*.png'))])
    perf.write_payload(report_dir / name(track, 'performance.json'), performance)
    progress('performance_complete', performance=performance)
    report = output / name(track, 'training_report.md')
    lines = [f'# {track} model report', '', f'Selected checkpoint: `{checkpoint.name}`.',
             f'Dev-fit Youden threshold: {threshold:.6f}. Test labels were not used for selection.', '',
             '## Pair metrics', '', '```text', pd.DataFrame(summary).to_string(index=False), '```', '',
             '## Retrieval', '', '```json', json.dumps(retrieval, indent=2), '```', '',
             '## Generalization slices', '', '```text', pd.DataFrame(slices).to_string(index=False), '```', '',
             '## Paired bootstrap confidence intervals', '',
             'Pair-resampling intervals, not repeated-training-seed intervals: this lane '
             'trains one checkpoint per split.', '',
             '```json', json.dumps(intervals, indent=2), '```', '',
             '## Operational cost', '', '```json', json.dumps(performance, indent=2), '```', '',
             'Retrieval truth includes direct confirmed positive pairs only; unlabeled candidates are not negatives.',
             'Small catalogs where K covers every target do not demonstrate useful retrieval quality.',
             'Graph context comes from training listings; dev/test queries do not communicate.',
             'Reported scores are model-only, without the shared identity conflict policy.',
             'Dev metrics are calibration/selection diagnostics; test is the held-out quality report.']
    report.write_text('\n'.join(lines) + '\n')
    progress('reports_complete', report=str(report), metrics=summary)
    return {'inference': inference, 'reports': report_dir, 'report': report,
            'summary': summary, 'retrieval': retrieval, 'slices': slices,
            'performance': performance, 'confidence_intervals': intervals}


def cascade_traceability_coverage(
        query_ids, *, measured_slices, measured_attributes) -> ReportCoverageContract:
    """The cascade's ``not_applicable`` traceability claim, machine-checked.

    ``report_cascade`` scores retrieved candidate pairs for a set of query
    records and owns no encoded listing catalog, so no scored record can carry a
    generalization-slice or attribute-separation tag. That is a DECLARED
    ``not_applicable`` for both dimensions over the whole scored-query
    population, with an explicit reason and unknown policy -- expressed in
    ``core.coverage_contracts.ReportCoverageContract``, the project's general
    per-record contract, so the claim is checked rather than aspirational.

    The claim is only honest while the tables it talks about stay empty, so the
    emitted ``slices``/``attributes`` rows are part of the check: a
    ``not_applicable`` dimension carries the whole population under
    ``unknown`` and NO measured value. Nothing new is emitted here; the cascade
    report derives its ``traceability`` prose FROM the validated contract and
    its bytes are unchanged.
    """
    records = tuple(query_ids)
    if not records:
        raise ValueError('cascade traceability needs at least one scored query record')
    if measured_slices or measured_attributes:
        raise ValueError(
            'the cascade declares slice/attribute coverage not_applicable but emits rows')
    return ReportCoverageContract(
        records=tuple(
            TaggedDimensionRecord(
                record_id=f'query:{identifier}',
                dimensions={name: (UNKNOWN_DIMENSION_VALUE,)
                            for name in CASCADE_TRACEABILITY})
            for identifier in records),
        dimensions={name: DimensionAccounting(
            policy='not_applicable',
            counts={UNKNOWN_DIMENSION_VALUE: len(records)},
            unknown_policy=('every scored query record is unknown for this dimension: the '
                            'cascade composes trained artifacts and owns no listing-attribute '
                            'table to derive a value from'),
            reason=reason)
            for name, reason in CASCADE_TRACEABILITY.items()},
    )


def report_cascade(ranked, relevant, decisions, output, *, track='cascade',
                   ks=(), recall_targets=(0.95,), bins=10, threshold=None):
    """Cascade report: ranker recall AND decider precision/calibration.

    Consumes the two role outputs of :func:`model_tracks.cascade.cascade` at a
    fixed candidate budget and evaluates each role on its own terms -- the
    ranker by candidate recall, the decider on the RETRIEVED candidate set by
    PR-AUC, precision at the requested recalls and ECE. It is a combinator
    report: no fused embedding or fused score is computed here.

    The cascade composes trained artifacts and never sees a listing catalog, so
    its slice/attribute coverage is a declared ``not_applicable`` that
    :func:`cascade_traceability_coverage` validates (see it for the rule).
    """
    from model_tracks.cascade import decider_report, ranker_report, retrieved_relevance
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not ks:
        ks = tuple(range(1, ranked.candidate_ids.shape[1] + 1))
    ranker = ranker_report(list(ranked.candidate_ids), relevant, ks)
    labels, scores = retrieved_relevance(decisions, relevant)
    decider = decider_report(labels, scores, recall_targets=recall_targets, bins=bins)
    # The cascade declares no slice/attribute population of its own; that claim
    # is validated as a per-record coverage contract over the scored queries
    # (see above). It emits nothing: the traceability prose below comes FROM the
    # validated contract, so the report's bytes are unchanged.
    slices: list = []
    attributes: list = []
    coverage = cascade_traceability_coverage(
        decisions.query_ids, measured_slices=slices, measured_attributes=attributes)
    report = {'schema': 'er-cascade-report-v1', 'track': track,
              'roles': {'ranker': ranker, 'decider': decider},
              # Same traceability keys as every trained-lane manifest; empty
              # with a documented reason because the cascade has no listing
              # catalog of its own to slice or attribute-score.
              'slices': slices,
              'attributes': attributes,
              'traceability': {name: coverage.dimensions[name].reason
                               for name in CASCADE_TRACEABILITY},
              'retrieval_ks': [int(k) for k in ks],
              'decision_threshold': threshold,
              'composed_from': ['text ranker (ANN candidates)',
                                'gnn_only pair scorer (decisions)']}
    (output / name(track, 'cascade_report.json')).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    lines = [f'# {track} cascade report', '',
             'The cascade is a combinator: the text ranker retrieves, the '
             'gnn_only pair scorer decides, and both roles are reported separately.', '',
             '## Ranker (candidate recall)', '', '```json',
             json.dumps(ranker, indent=2, sort_keys=True), '```', '',
             '## Decider (retrieved candidate set)', '', '```json',
             json.dumps(decider, indent=2, sort_keys=True), '```', '',
             '## Traceability', '',
             'The cascade emits no generalization-slice or attribute-separation '
             'population of its own: it scores only the retrieved candidate pairs '
             'and owns no encoded listing catalog.', '', '```json',
             json.dumps({'slices': report['slices'], 'attributes': report['attributes'],
                         'traceability': report['traceability']},
                        indent=2, sort_keys=True), '```', '']
    (output / name(track, 'cascade_report.md')).write_text('\n'.join(lines) + '\n')
    return report


def _plots(scored, output, track, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split, group in scored.groupby('split'):
        axes[0].hist(group.score, bins=20, alpha=.5, label=split)
        if group.true_label.nunique() == 2:
            p, r, _ = precision_recall_curve(group.true_label, group.score)
            axes[1].plot(r, p, label=split)
    axes[0].axvline(threshold, color='red', linestyle='--', label='dev threshold')
    axes[0].set(xlabel='cosine similarity' if track == 'text' else 'pair match probability',
                ylabel='pairs', title=track)
    axes[1].set(xlabel='recall', ylabel='precision', title='Precision–recall')
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / name(track, 'score_distribution_and_pr.png'), dpi=plot_dpi())
    plt.close(fig)


def main():
    import argparse
    from graph_tracks.config import load_config
    from core.common import TRAIN_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    complete(args.checkpoint, (TRAIN_ROOT / cfg.listings).resolve(), (TRAIN_ROOT / cfg.pairs).resolve(),
             args.output, cfg, text_cache=(TRAIN_ROOT / cfg.text_cache).resolve() if cfg.text_cache else None)

if __name__ == '__main__':
    main()
