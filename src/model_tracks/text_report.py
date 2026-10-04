"""Report the selected text checkpoint on the same held-out listing pairs."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from graph_tracks.config import load_text_config, RetrievalReportContext


def complete(output: Path, setup: Path, *, device: str, report_test: bool):
    from model_tracks.text_export import validate as validate_export
    from graph_tracks.data import file_hash, load_records
    from graph_tracks.report import dev_threshold, pair_metrics, retrieval_report
    from graph_tracks.report_slices import report as slice_report
    from graph_tracks.train import load_pairs
    from training.validation_inference import resolve_best_checkpoint
    from core.bootstrap_ci import paired_bootstrap
    from core.performance import PerformanceRecorder
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    # Validate the lane before checkpoint loading, encoding or report writes.
    settings = load_text_config(setup / 'text.yaml')
    settings.report_test = report_test
    started = time.monotonic()
    print(f"[text-postprocess] start output={output} setup={setup} device={device} report_test={report_test}", flush=True)
    checkpoint, selection = resolve_best_checkpoint(output)
    print(f"[text-selection] checkpoint={checkpoint} reason=trainer_recorded_best "
          f"best_metric={selection.get('best_metric')} recorded_step={selection.get('global_step')}", flush=True)
    perf = PerformanceRecorder('text')
    reports = output / 'text__reports'
    reports.mkdir(exist_ok=True)
    listings = setup / 'prepared/listings.json'
    print(f"[text-phase] inputs start listings={listings} pairs={setup / 'prepared/pairs.csv'}", flush=True)
    records = load_records(listings)
    pairs = load_pairs(setup / 'prepared/pairs.csv', records)
    print(f"[text-phase] inputs complete listings={len(records)} split_pairs="
          f"{ {split: {'positive': int(labels.sum()), 'negative': int((labels == 0).sum())} for split, (_, labels) in pairs.items()} }", flush=True)
    cache_started = time.monotonic()
    print(f"[text-phase] vector_export start catalog={setup / 'eligible_catalog.csv'} checkpoint={checkpoint} "
          f"output={output / 'text__vectors.npz'}", flush=True)
    cache = output / 'text__vectors.npz'
    vectors, metadata = validate_export(cache, checkpoint, setup)
    perf.record('cache_validation', time.monotonic() - cache_started)
    perf.adopt('encode', metadata.get('performance', {}).get('sections', {}).get('encode', {}))
    print(f"[text-phase] vector_export complete path={cache} shape={vectors.shape} "
          f"seconds={time.monotonic() - cache_started:.3f}", flush=True)
    cfg = settings
    retrieval_cfg = RetrievalReportContext.from_config(cfg, checkpoint, file_hash(listings))
    if cfg.build_index:
        index_started = time.monotonic()
        print(f"[text-phase] index_build start path={output / 'text__index'} vectors={len(vectors)} "
              f"M={cfg.hnsw_m} ef_construction={cfg.hnsw_ef_construction} ef_search={cfg.hnsw_ef_search}", flush=True)
        from training.hnsw_index import PersistentHnswIndex
        index = PersistentHnswIndex(output / 'text__index', ef_construction=cfg.hnsw_ef_construction,
                                   M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
        from model_tracks.training_data import retrieval_indices
        catalog_indices = retrieval_indices(records)
        index.build(vectors[catalog_indices], [records[i]['sku_id'] for i in catalog_indices], checkpoint=checkpoint,
                    model_name='text', preprocessing_fingerprint=file_hash(listings))
        perf.record('index_build', time.monotonic() - index_started)
        print(f"[text-phase] index_build complete path={output / 'text__index'} seconds={time.monotonic() - index_started:.3f}", flush=True)
    else:
        print('[text-phase] index_build skipped reason=build_index_false', flush=True)
    scores = {}
    for split in ('dev', 'test'):
        if split == 'test' and not report_test:
            print("[text-evaluation] test skipped reason=report_test_false", flush=True)
            continue
        indices = pairs[split][0]
        print(f"[text-evaluation] start split={split} pairs={len(indices)} score=embedding_cosine", flush=True)
        if len(indices):
            scores[split] = (vectors[indices[:, 0]] * vectors[indices[:, 1]]).sum(-1)
        else:
            print(f"[text-evaluation] skipped split={split} reason=no_pairs", flush=True)
    if 'dev' not in scores:
        raise ValueError('text calibration requires non-empty dev pairs')
    threshold = dev_threshold(pairs['dev'][1], scores['dev'])
    print(f"[text-calibration] threshold={threshold} source=dev_youden dev_pairs={len(pairs['dev'][1])}", flush=True)
    summary, scored, intervals = [], [], {}
    for split, values in scores.items():
        indices, labels = pairs[split]
        summary.append({'model': 'text', 'split': split, 'threshold_source': 'dev_youden',
                        'checkpoint': checkpoint.name,
                        **pair_metrics(labels, values, threshold, cfg.retrieval_ks)})
        intervals[split] = paired_bootstrap(labels, values, track='text', split=split)
        print(f"[text-evaluation] complete summary={json.dumps(summary[-1], sort_keys=True, default=str)}", flush=True)
        for (a, b), label, score in zip(indices, labels, values):
            scored.append({'sku_id1':records[a]['sku_id'], 'sku_id2':records[b]['sku_id'],
                           'true_label':int(label), 'split':split, 'score':float(score),
                           'prediction':int(score >= threshold)})
    print(f"[text-phase] reports start directory={reports} kinds=summary,scored_pairs,plots,attributes,slices,retrieval", flush=True)
    pd.DataFrame(summary).to_csv(reports / 'text__model_evaluation_summary.csv', index=False)
    scored_frame = pd.DataFrame(scored)
    scored_frame.to_csv(reports / 'text__scored_pairs.csv', index=False)
    from graph_tracks.report import _plots
    _plots(scored_frame, reports, 'text', threshold)
    from graph_tracks.report_attributes import write_reports as write_attribute_reports
    write_attribute_reports(listings, records, pairs, scores, reports, 'text')
    slices = slice_report(records, scored_frame, track='text', output=reports,
                          pair_metrics=pair_metrics, threshold=threshold,
                          ks=tuple(cfg.retrieval_ks))
    print(f"[text-phase] retrieval start splits={list(scores)} ks={cfg.retrieval_ks}", flush=True)
    retrieval = retrieval_report(records, vectors, pairs, reports, 'text', retrieval_cfg, perf=perf)
    from core.performance import summarize_profiler_tree, summarize_refresh_timings
    # Refresh is measured by the trainer; adopt it so it is not reported missing.
    perf.adopt('refresh', summarize_refresh_timings(output / 'logs').get('refresh', {}))
    performance = perf.summary()
    performance.update(summarize_profiler_tree(output / 'profiles'))
    perf.write_payload(reports / 'text__performance.json', performance)
    (output / 'text__training_report.md').write_text(
        '# Text track\n\nSelected checkpoint: ' + str(checkpoint) + '\n\n'
        'Held-out listing pairs use the shared split. Threshold fitted on dev only.\n'
        'Scores are model-only; known-positive retrieval truth is incomplete.\n\n'
        + '\n'.join(f"- {r['split']}: PR-AUC {r['pr_auc']}, F1 {r['f1']}" for r in summary) + '\n\n'
        '## Generalization slices\n\n```text\n'
        + pd.DataFrame(slices).to_string(index=False) + '\n```\n\n'
        '## Paired bootstrap confidence intervals\n\n'
        'Pair-resampling intervals, not repeated-training-seed intervals: this lane\n'
        'trains one checkpoint per split.\n\n```json\n'
        + json.dumps(intervals, indent=2, default=str) + '\n```\n\n'
        '## Operational cost\n\n```json\n'
        + json.dumps(performance, indent=2, default=str) + '\n```\n')
    # The text lane used to hand-write a manifest missing every honesty field
    # the graph lanes emit (test_used_for_selection, metrics_scope,
    # retrieval_protocol, ...). Both lanes now share one contract.
    write_manifest(output / 'text__completion_manifest.json', build_manifest(
        track='text', checkpoint=checkpoint,
        checkpoint_sha256=file_hash(checkpoint),
        listings_sha256=file_hash(listings),
        pairs_sha256=file_hash(setup/'prepared/pairs.csv'),
        threshold=threshold, threshold_source='dev_youden',
        test_reported='test' in scores, model_selection='dev_pr_auc',
        retrieval_ks=cfg.retrieval_ks, vectors_metadata=metadata,
        summary=summary, retrieval=retrieval, slices=slices,
        performance=performance, confidence_intervals=intervals,
        report_test=report_test,
        extra={'checkpoint_source': 'selected_best',
               'best_metric': selection.get('best_metric'),
               'recorded_step': selection.get('global_step')}))
    print(f"[text-phase] reports complete artifacts={[str(path) for path in sorted(reports.rglob('*')) if path.is_file()]}", flush=True)
    print(f"[text-postprocess] complete checkpoint={checkpoint} manifest={output / 'text__completion_manifest.json'} "
          f"report={output / 'text__training_report.md'} seconds={time.monotonic() - started:.3f}", flush=True)
