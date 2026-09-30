"""Report the selected text checkpoint on the same held-out listing pairs."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml


def complete(output: Path, setup: Path, *, device: str, report_test: bool):
    from graph_tracks.text_cache import create_cache
    from graph_tracks.data import file_hash, load_records, load_text_cache
    from graph_tracks.report import dev_threshold, pair_metrics, retrieval_report
    from graph_tracks.train import load_pairs
    from training.validation_inference import resolve_best_checkpoint
    from training.hnsw_index import PersistentHnswIndex
    checkpoint, _ = resolve_best_checkpoint(output)
    reports = output / 'text__reports'
    reports.mkdir()
    listings = setup / 'prepared/listings.json'
    records = load_records(listings)
    pairs = load_pairs(setup / 'prepared/pairs.csv', records)
    cache = create_cache(setup / 'eligible_catalog.csv', checkpoint, output / 'text__vectors.npz', device=device)
    vectors, metadata = load_text_cache(cache, [r['product_id'] for r in records])
    settings = yaml.safe_load((setup / 'gnn_only.yaml').read_text())
    settings.update(report_test=report_test, _checkpoint=str(checkpoint), _listings_sha256=file_hash(listings))
    cfg = SimpleNamespace(**settings)
    index = PersistentHnswIndex(output / 'text__index', ef_construction=cfg.hnsw_ef_construction,
                               M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
    index.build(vectors, [r['product_id'] for r in records], checkpoint=checkpoint,
                model_name='text', preprocessing_fingerprint=file_hash(listings))
    scores = {}
    for split in ('dev', 'test'):
        if split == 'test' and not report_test:
            continue
        indices = pairs[split][0]
        if len(indices):
            scores[split] = (vectors[indices[:, 0]] * vectors[indices[:, 1]]).sum(-1)
    threshold = dev_threshold(pairs['dev'][1], scores['dev'])
    summary, scored = [], []
    for split, values in scores.items():
        indices, labels = pairs[split]
        summary.append({'model': 'text', 'split': split, 'threshold_source': 'dev_youden',
                        **pair_metrics(labels, values, threshold, cfg.retrieval_ks)})
        for (a, b), label, score in zip(indices, labels, values):
            scored.append({'product_id1':records[a]['product_id'], 'product_id2':records[b]['product_id'],
                           'true_label':int(label), 'split':split, 'score':float(score),
                           'prediction':int(score >= threshold)})
    pd.DataFrame(summary).to_csv(reports / 'text__model_evaluation_summary.csv', index=False)
    pd.DataFrame(scored).to_csv(reports / 'text__scored_pairs.csv', index=False)
    from graph_tracks.report_attributes import write_reports as write_attribute_reports
    write_attribute_reports(listings, records, pairs, scores, reports, 'text')
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
