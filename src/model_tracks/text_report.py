"""Report the selected text checkpoint on the same held-out listing pairs."""
from pathlib import Path
import json
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml


def complete(output: Path, setup: Path, *, device: str, report_test: bool):
    from model_tracks.text_export import validate as validate_export
    from graph_tracks.data import file_hash, load_records, load_text_cache
    from graph_tracks.report import dev_threshold, pair_metrics, retrieval_report
    from graph_tracks.train import load_pairs
    from training.validation_inference import resolve_best_checkpoint
    from training.hnsw_index import PersistentHnswIndex
    started = time.monotonic()
    print(f"[text-postprocess] start output={output} setup={setup} device={device} report_test={report_test}", flush=True)
    checkpoint, selection = resolve_best_checkpoint(output)
    print(f"[text-selection] checkpoint={checkpoint} reason=trainer_recorded_best "
          f"best_metric={selection.get('best_metric')} recorded_step={selection.get('global_step')}", flush=True)
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
    print(f"[text-phase] vector_export complete path={cache} shape={vectors.shape} "
          f"seconds={time.monotonic() - cache_started:.3f}", flush=True)
    settings = yaml.safe_load((setup / 'gnn_only.yaml').read_text())
    settings.update(report_test=report_test, _checkpoint=str(checkpoint), _listings_sha256=file_hash(listings))
    cfg = SimpleNamespace(**settings)
    index_started = time.monotonic()
    print(f"[text-phase] index_build start path={output / 'text__index'} vectors={len(vectors)} "
          f"M={cfg.hnsw_m} ef_construction={cfg.hnsw_ef_construction} ef_search={cfg.hnsw_ef_search}", flush=True)
    index = PersistentHnswIndex(output / 'text__index', ef_construction=cfg.hnsw_ef_construction,
                               M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
    index.build(vectors, [r['sku_id'] for r in records], checkpoint=checkpoint,
                model_name='text', preprocessing_fingerprint=file_hash(listings))
    print(f"[text-phase] index_build complete path={output / 'text__index'} seconds={time.monotonic() - index_started:.3f}", flush=True)
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
    threshold = dev_threshold(pairs['dev'][1], scores['dev'])
    print(f"[text-calibration] threshold={threshold} source=dev_youden dev_pairs={len(pairs['dev'][1])}", flush=True)
    summary, scored = [], []
    for split, values in scores.items():
        indices, labels = pairs[split]
        summary.append({'model': 'text', 'split': split, 'threshold_source': 'dev_youden',
                        **pair_metrics(labels, values, threshold, cfg.retrieval_ks)})
        print(f"[text-evaluation] complete summary={json.dumps(summary[-1], sort_keys=True)}", flush=True)
        for (a, b), label, score in zip(indices, labels, values):
            scored.append({'sku_id1':records[a]['sku_id'], 'sku_id2':records[b]['sku_id'],
                           'true_label':int(label), 'split':split, 'score':float(score),
                           'prediction':int(score >= threshold)})
    print(f"[text-phase] reports start directory={reports} kinds=summary,scored_pairs,plots,attributes,retrieval", flush=True)
    pd.DataFrame(summary).to_csv(reports / 'text__model_evaluation_summary.csv', index=False)
    pd.DataFrame(scored).to_csv(reports / 'text__scored_pairs.csv', index=False)
    from graph_tracks.report import _plots
    _plots(pd.DataFrame(scored), reports, 'text', threshold)
    from graph_tracks.report_attributes import write_reports as write_attribute_reports
    write_attribute_reports(listings, records, pairs, scores, reports, 'text')
    print(f"[text-phase] retrieval start splits={list(scores)} ks={cfg.retrieval_ks}", flush=True)
    retrieval = retrieval_report(records, vectors, pairs, reports, 'text', cfg)
    (output / 'text__training_report.md').write_text(
        '# Text track\n\nSelected checkpoint: ' + str(checkpoint) + '\n\n'
        'Held-out listing pairs use the shared split. Threshold fitted on dev only.\n'
        'Scores are model-only; known-positive retrieval truth is incomplete.\n\n'
        + '\n'.join(f"- {r['split']}: PR-AUC {r['pr_auc']}, F1 {r['f1']}" for r in summary) + '\n')
    (output / 'text__completion_manifest.json').write_text(json.dumps({
        'checkpoint':str(checkpoint), 'vectors_metadata':metadata, 'report_test':report_test,
        'listings_sha256':file_hash(listings), 'pairs_sha256':file_hash(setup/'prepared/pairs.csv'),
        'summary':summary, 'retrieval':retrieval}, indent=2) + '\n')
    print(f"[text-phase] reports complete artifacts={[str(path) for path in sorted(reports.rglob('*')) if path.is_file()]}", flush=True)
    print(f"[text-postprocess] complete checkpoint={checkpoint} manifest={output / 'text__completion_manifest.json'} "
          f"report={output / 'text__training_report.md'} seconds={time.monotonic() - started:.3f}", flush=True)
