"""Dev-fit threshold, final test metrics and known-positive retrieval reports.

Uses the ER evaluation metric names and pooled ranking implementation. Pair
ranking is explicitly pooled: it is not mislabeled as catalog retrieval.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve,
                             roc_auc_score, roc_curve)
from core.ranking_metrics import ranking_at_k
from graph_tracks.artifacts import name
from graph_tracks.data import file_hash, load_records, load_text_cache


def dev_threshold(labels, scores):
    if set(labels) != {0., 1.}:
        raise ValueError('threshold calibration needs both classes on dev')
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def pair_metrics(labels, scores, threshold, ks):
    labels = np.asarray(labels, dtype=int)
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions,
        average='binary', zero_division=0)
    supported = set(labels) == {0, 1}
    p, r, _ = precision_recall_curve(labels, scores) if supported else (None, None, None)
    pooled = ranking_at_k(labels, scores, tuple(ks)) if labels.any() else {}
    return {'rows': len(labels), 'positive_pairs': int(labels.sum()),
        'negative_pairs': int((labels == 0).sum()), 'threshold': threshold,
        'roc_auc': float(roc_auc_score(labels, scores)) if supported else None,
        'pr_auc': float(average_precision_score(labels, scores)) if supported else None,
        'p_at_r95': float(p[r >= .95].max()) if supported else None,
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision_defined': bool(tp + fp), 'both_classes': supported,
        **{key: value for key, value in pooled.items() if key.startswith('pooled_')}}


def retrieval_report(records, vectors, pairs, output, track, cfg):
    """Same-split eligible catalogs; no trained endpoints; incomplete truth explicit."""
    from training.hnsw_index import PersistentHnswIndex
    rows = []
    for split in ('dev', 'test'):
        if split == 'test' and not cfg.report_test:
            continue
        indices, labels = pairs[split]
        positive = indices[labels == 1]
        if not len(positive):
            continue
        targets = [i for i, r in enumerate(records) if r['split'] == split]
        relevant = {}
        for left, right in positive:
            relevant.setdefault(int(left), set()).add(int(right))
            relevant.setdefault(int(right), set()).add(int(left))
        index = PersistentHnswIndex(output / name(track, f'{split}_retrieval_index'),
            ef_construction=cfg.hnsw_ef_construction, M=cfg.hnsw_m, ef_search=cfg.hnsw_ef_search)
        # No encoder checkpoint claim: report provenance hashes are supplied separately.
        checkpoint = Path(cfg._checkpoint)
        index.build(vectors[targets], [records[i]['product_id'] for i in targets],
                    checkpoint=checkpoint, model_name=track,
                    preprocessing_fingerprint=cfg._listings_sha256)
        query_ids = sorted(relevant)
        rankings, _ = index.query(vectors[query_ids], top_k=min(max(cfg.retrieval_ks) + 1, len(targets)))
        for query, ranking in zip(query_ids, rankings):
            candidates = [targets[int(label)] for label in ranking if targets[int(label)] != query]
            for k in cfg.retrieval_ks:
                recovered = len(set(candidates[:k]) & relevant[query])
                rows.append({'split': split, 'product_id': records[query]['product_id'], 'k': k,
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


def complete(checkpoint: Path, listings: Path, pair_path: Path, output: Path, cfg, *, text_cache=None):
    from graph_tracks.infer import GraphEncoder, export
    from graph_tracks.train import load_pairs, write_json
    records = load_records(listings)
    pairs = load_pairs(pair_path, records)
    encoder = GraphEncoder(checkpoint, cfg.device)
    track = encoder.manifest['track']
    if file_hash(listings) != encoder.manifest['listings_sha256'] or file_hash(pair_path) != encoder.manifest['pairs_sha256']:
        raise ValueError('post-training report must use checkpoint-bound listing/pair inputs')
    inference = export(checkpoint, listings, output / name(track, 'inference'), text_cache=text_cache,
                       build_index=cfg.build_index, device=cfg.device, batch_size=cfg.inference_batch_size)
    cache = np.load(inference / name(track, 'vectors.npz'), allow_pickle=False)
    vectors = cache['embeddings']
    text = None if text_cache is None else load_text_cache(text_cache, [r['product_id'] for r in records])[0]
    scores = {}
    with torch.no_grad():
        embeddings = torch.as_tensor(vectors, device=cfg.device)
        text_tensor = None if text is None else torch.as_tensor(text, device=cfg.device)
        for split in ('dev', 'test'):
            if split == 'test' and not cfg.report_test:
                continue
            indices = pairs[split][0]
            if len(indices):
                scores[split] = encoder.scorer(embeddings, torch.as_tensor(indices, device=cfg.device),
                                               text_tensor).sigmoid().cpu().numpy()
    threshold = dev_threshold(pairs['dev'][1], scores['dev'])
    summary, scored_rows = [], []
    for split, values in scores.items():
        indices, labels = pairs[split]
        summary.append({'model': track, 'split': split, 'threshold_source': 'dev_youden',
                        'checkpoint': checkpoint.name, **pair_metrics(labels, values, threshold, cfg.retrieval_ks)})
        for (left, right), label, score in zip(indices, labels, values):
            scored_rows.append({'product_id1': records[left]['product_id'],
                'product_id2': records[right]['product_id'], 'true_label': int(label),
                'split': split, 'score': float(score), 'prediction': int(score >= threshold)})
    report_dir = output / name(track, 'reports')
    report_dir.mkdir()
    pd.DataFrame(summary).to_csv(report_dir / name(track, 'model_evaluation_summary.csv'), index=False)
    scored = pd.DataFrame(scored_rows)
    scored.to_csv(report_dir / name(track, 'scored_pairs.csv'), index=False)
    # Attribute availability slices are diagnostic; no automatic identity vetoes.
    slices = []
    for split in scores:
        indices, labels = pairs[split]
        for relation in ('flavor', 'sweetener', 'package_type'):
            observed = np.asarray([bool(records[a]['attributes'].get(relation)) and
                                   bool(records[b]['attributes'].get(relation)) for a, b in indices])
            for present in (True, False):
                mask = observed == present
                if mask.any():
                    slices.append({'split': split, 'slice': f'{relation}_both_observed={present}',
                        **pair_metrics(labels[mask], scores[split][mask], threshold, cfg.retrieval_ks)})
    pd.DataFrame(slices).to_csv(report_dir / name(track, 'slice_metrics.csv'), index=False)
    # Pass immutable provenance separately rather than adding undeclared config fields.
    from types import SimpleNamespace
    retrieval_cfg = SimpleNamespace(**cfg.model_dump(), _checkpoint=str(checkpoint),
                                    _listings_sha256=file_hash(listings))
    retrieval = retrieval_report(records, vectors, pairs, report_dir, track, retrieval_cfg)
    write_json(report_dir / name(track, 'report_manifest.json'), {
        'track': track, 'checkpoint_sha256': file_hash(checkpoint),
        'listings_sha256': file_hash(listings), 'pairs_sha256': file_hash(pair_path),
        'threshold': threshold, 'threshold_source': 'dev_youden',
        'test_used_for_selection': False, 'model_selection': 'dev_pr_auc',
        'graph_context': 'training-listings-only', 'trained_endpoints_scored': False,
        'retrieval_protocol': 'within-split catalog, self excluded, direct known positives only',
        'unlabeled_pairs_are_negatives': False, 'identity_conflict_policy_applied': False,
        'metrics_scope': 'model-only', 'test_reported': 'test' in scores})
    _plots(scored, report_dir, track, threshold)
    report = output / name(track, 'training_report.md')
    lines = [f'# {track} model report', '', f'Selected checkpoint: `{checkpoint.name}`.',
             f'Dev-fit Youden threshold: {threshold:.6f}. Test labels were not used for selection.', '',
             '## Pair metrics', '', '```text', pd.DataFrame(summary).to_string(index=False), '```', '',
             '## Retrieval', '', '```json', json.dumps(retrieval, indent=2), '```', '',
             'Retrieval truth includes direct confirmed positive pairs only; unlabeled candidates are not negatives.',
             'Small catalogs where K covers every target do not demonstrate useful retrieval quality.',
             'Graph context comes from training listings; dev/test queries do not communicate.',
             'Reported scores are model-only, without the shared identity conflict policy.',
             'Dev metrics are calibration/selection diagnostics; test is the held-out quality report.']
    report.write_text('\n'.join(lines) + '\n')
    return {'inference': inference, 'reports': report_dir, 'report': report,
            'summary': summary, 'retrieval': retrieval}


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
    axes[0].set(xlabel='pair match probability', ylabel='pairs', title=track)
    axes[1].set(xlabel='recall', ylabel='precision', title='Precision–recall')
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / name(track, 'score_distribution_and_pr.png'), dpi=150)
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
