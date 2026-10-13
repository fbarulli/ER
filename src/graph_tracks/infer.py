"""Checkpoint-bound inductive vectors, pair scores, and optional HNSW export.

Only load trusted checkpoints: optimizer/RNG payloads use Python pickle.
Hybrid exported vectors are graph-informed vectors; the scorer's direct text
path is separate and is not reproduced by cosine ANN search.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core.perf_switches import perf_enabled
from graph_tracks.artifacts import name, checkpoint_track
from graph_tracks.data import file_size, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN, PairScorer
from graph_tracks.train import write_json

# CUDA-only: keep encoded batches on device and transfer once instead of
# syncing the host after every query batch. CPU keeps the original path.
_INFER_BATCHED_TRANSFER = perf_enabled("graph.infer_batched_transfer")


class GraphEncoder:
    def __init__(self, checkpoint: Path, device='cpu', *, prepared_support=None):
        self.checkpoint = checkpoint
        self.checkpoint_size = file_size(checkpoint)
        track = checkpoint_track(checkpoint)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get('schema') != 'er-graph-checkpoint-v1':
            raise ValueError('unsupported checkpoint schema')
        if payload['manifest']['track'] != track:
            raise ValueError('checkpoint filename/payload track mismatch')
        self.manifest, self.vocabulary = payload['manifest'], payload['vocabulary']
        self.device = device
        from graph_tracks.config import GraphConfig
        cfg = GraphConfig.model_validate(self.manifest['config']).model_dump()
        self.manifest['config'] = cfg
        self.model = AttributeGNN(self.vocabulary, cfg['hidden_dim'], cfg['output_dim'],
                                 payload['text_dim'], cfg['graph_enabled'], cfg['aggregation_backend']).to(device)
        self.scorer = PairScorer(bool(payload['text_dim'])).to(device)
        self.model.load_state_dict(payload['model'])
        self.scorer.load_state_dict(payload['scorer'])
        self.model.eval()
        self.scorer.eval()
        support = prepared_support if prepared_support is not None else tensorize(payload['support_records'], self.vocabulary, device)
        support_text = payload['support_text']
        if support_text is not None:
            support_text = support_text.to(device)
        with torch.no_grad():
            self.states = self.model.context(support, support_text)

    def encode(self, records, text=None, batch_size=1024):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not records:
            raise ValueError('empty inference population')
        if self.device == 'cuda':
            raise ValueError('CUDA inference requires locally prepared graph batches')
        batches = (tensorize(records[start:start + batch_size], self.vocabulary, self.device)
                   for start in range(0, len(records), batch_size))
        return self.encode_prepared(batches, text)

    def encode_prepared(self, batches, text=None):
        """Forward already tensorized batches; preparation may run locally."""
        chunks, pending, start = [], [], 0
        batched_transfer = _INFER_BATCHED_TRANSFER and torch.device(self.device).type == 'cuda'
        with torch.no_grad():
            for batch in batches:
                end = start + len(batch.numeric)
                vectors = None if text is None else torch.as_tensor(text[start:end], device=self.device)
                encoded = self.model.encode(batch, self.states, vectors)
                if batched_transfer:
                    pending.append(encoded)
                else:
                    chunks.append(encoded.cpu().numpy())
                start = end
        if pending:
            chunks.append(torch.cat(pending).cpu().numpy())
        if not chunks or (text is not None and start != len(text)):
            raise ValueError('prepared graph batch population mismatch')
        vectors = np.concatenate(chunks)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
            raise ValueError('encoder produced invalid vectors')
        return vectors


def export(checkpoint: Path, listings: Path, output: Path, *, text_cache=None,
           pairs=None, build_index=False, device='cpu', batch_size=1024,
           encoder=None, prepared_plan=None, prepared_arrays=None):
    if output.exists():
        raise FileExistsError(output)
    records = load_records(listings, require_training=False)
    ids = [r['sku_id'] for r in records]
    from graph_tracks.prepared_inputs import PLAN, load_plan, load_batch, PreparedGraphInputs
    owned_arrays = None
    if prepared_plan is None and (listings.parent / PLAN).is_file():
        prepared_plan, prepared_arrays = load_plan(listings)
        owned_arrays = prepared_arrays
    if prepared_plan is not None:
        PreparedGraphInputs(plan=prepared_plan, arrays=prepared_arrays).validate_catalog(records)
        if prepared_plan['ids'] != ids or prepared_plan['listings_size'] != file_size(listings):
            raise ValueError('prepared inference population mismatch')
        support = (load_batch(prepared_arrays, 'train', device, prepared_plan['vocabulary'])
                   if encoder is None else None)
    else:
        support = None
        if device == 'cuda':
            raise ValueError('CUDA inference requires locally prepared graph tensors')
    if encoder is not None and (encoder.checkpoint_size != file_size(checkpoint)
                                or encoder.device != device):
        raise ValueError('shared graph encoder differs from export checkpoint/device')
    encoder = encoder or GraphEncoder(checkpoint, device, prepared_support=support)
    if prepared_plan is not None and prepared_plan['vocabulary'] != encoder.vocabulary:
        raise ValueError('prepared inference vocabulary differs from checkpoint')
    if prepared_plan is not None and prepared_plan.get('support_listings_size', prepared_plan['listings_size']) != encoder.manifest['listings_size']:
        raise ValueError('prepared training support differs from checkpoint')
    if prepared_plan is not None and prepared_plan.get('checkpoint_size', file_size(checkpoint)) != file_size(checkpoint):
        raise ValueError('prepared query checkpoint mismatch')
    text, metadata = None, None
    if text_cache:
        raise ValueError('graph export forbids a fused text cache; the hybrid fusion is retired')
    from core.performance import PerformanceRecorder
    perf = PerformanceRecorder(encoder.manifest['track'])
    if prepared_plan is not None:
        batches = (load_batch(prepared_arrays, prefix, device, encoder.vocabulary)
                   for prefix in prepared_plan['query_batches'])
        with perf.section("encode"):
            vectors = encoder.encode_prepared(batches, text)
    else:
        with perf.section("encode"):
            vectors = encoder.encode(records, text, batch_size)
    if len(vectors) != len(ids):
        raise ValueError('graph export embedding/ID population mismatch')
    if owned_arrays is not None:
        owned_arrays.close()
    track = encoder.manifest['track']
    output.mkdir(parents=True)
    np.savez_compressed(output / name(track, 'vectors.npz'), ids=np.asarray(ids), embeddings=vectors)
    if pairs:
        frame = pd.read_csv(pairs, dtype=str, keep_default_na=False)
        if set(frame.columns) != {'sku_id1', 'sku_id2'}:
            raise ValueError('inference pairs require exactly sku_id1,sku_id2; labels are excluded')
        lookup = {key: i for i, key in enumerate(ids)}
        if any(key not in lookup for key in list(frame.sku_id1) + list(frame.sku_id2)):
            raise ValueError('unknown inference pair endpoint')
        indices = np.asarray([(lookup[a], lookup[b]) for a, b in
                              zip(frame.sku_id1, frame.sku_id2)], dtype=np.int64).reshape(-1, 2)
        with torch.no_grad():
            scores = encoder.scorer(torch.as_tensor(vectors, device=device),
                torch.as_tensor(indices, device=device),
                None if text is None else torch.as_tensor(text, device=device)).sigmoid().cpu().numpy()
        frame['score'] = scores
        frame.to_csv(output / name(track, 'pair_scores.csv'), index=False)
    if build_index:
        from training.hnsw_index import PersistentHnswIndex
        index = PersistentHnswIndex(output / name(track, 'index'), ef_construction=encoder.manifest['config']['hnsw_ef_construction'],
                                    M=encoder.manifest['config']['hnsw_m'],
                                    ef_search=encoder.manifest['config']['hnsw_ef_search'])
        index.build(vectors, ids, checkpoint=checkpoint, model_name=encoder.manifest['track'],
                    preprocessing_fingerprint=file_size(listings))
    from graph_tracks.artifacts import GraphExportManifest
    write_json(output / name(track, 'export_manifest.json'), GraphExportManifest.model_validate({
        'schema': 'er-graph-export-v1', 'checkpoint_size': file_size(checkpoint),
        'listings_size': file_size(listings), 'vectors_size': file_size(output / name(track, 'vectors.npz')),
        'text_cache_size': file_size(text_cache) if text_cache else None,
        'track': encoder.manifest['track'], 'graph_context': 'training-listings-only',
        'vector_kind': 'graph-informed', 'ann_reproduces_pair_scorer': False,
        'count': len(ids), 'dimension': vectors.shape[1], 'index_built': build_index,
        'embedding_dtype': str(vectors.dtype), 'performance': perf.summary(),
        'id_kind': 'listing_sku_id'}).model_dump(by_alias=True))
    return output


def forward_outputs(checkpoint, listings, pair_path, output, cfg, *, text_cache=None,
                    prepared_plan=None, prepared_arrays=None, return_encoder=False):
    """Save GPU forward results for CPU-only analysis after session teardown."""
    from graph_tracks.train import load_pairs
    from graph_tracks.prepared_inputs import load_plan, PLAN
    records = load_records(listings)
    owned_arrays = None
    if prepared_plan is None and (listings.parent / PLAN).is_file():
        prepared_plan, prepared_arrays = load_plan(listings, pair_path)
        owned_arrays = prepared_arrays
    if prepared_plan is not None:
        if prepared_plan.get('pairs_size') != file_size(pair_path):
            raise ValueError('prepared forward pair source mismatch')
        # Prepared local endpoint indices map to catalog vectors without CSV
        # parsing or building a new endpoint lookup in the GPU worker.
        pair_data = {}
        for split in ('dev', 'test'):
            indices = prepared_arrays[split+'/catalog_pairs']
            if indices.dtype != np.int64 or indices.ndim != 2 or indices.shape[1] != 2:
                raise ValueError('prepared catalog pair dtype/shape mismatch')
            pair_data[split] = (indices, None)
    elif cfg.device == 'cuda':
        raise ValueError('CUDA forward export requires locally prepared pair indices')
    else:
        pair_data = load_pairs(pair_path, records)
    from graph_tracks.prepared_inputs import load_batch
    support = (load_batch(prepared_arrays, 'train', cfg.device, prepared_plan['vocabulary'])
               if prepared_plan is not None else None)
    encoder = GraphEncoder(checkpoint, cfg.device, prepared_support=support)
    del support
    for key, path in [('listings_size', listings), ('pairs_size', pair_path)]:
        if encoder.manifest.get(key) != file_size(path):
            raise ValueError(f'forward export must use checkpoint-bound inputs: {key}')
    inference = export(checkpoint, listings, output, text_cache=text_cache,
                       device=cfg.device, batch_size=cfg.inference_batch_size,
                       encoder=encoder, prepared_plan=prepared_plan, prepared_arrays=prepared_arrays)
    if owned_arrays is not None:
        owned_arrays.close()
    track = checkpoint_track(checkpoint)
    scorer = encoder.scorer
    with np.load(inference / name(track, 'vectors.npz'), allow_pickle=False) as cache:
        vectors = torch.as_tensor(cache['embeddings'], device=cfg.device)
    text = None if text_cache is None else torch.as_tensor(
        load_text_cache(text_cache, [r['sku_id'] for r in records])[0], device=cfg.device)
    scores = {}
    with torch.no_grad():
        for split in ('dev', 'test'):
            if split == 'test' and not cfg.report_test:
                continue
            indices = pair_data[split][0]
            scores[split] = scorer(vectors, torch.as_tensor(indices, device=cfg.device), text).sigmoid().cpu().numpy()
    score_path = inference / name(track, 'split_scores.npz')
    np.savez_compressed(score_path, **scores)
    manifest_path = inference / name(track, 'export_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest.update(pairs_size=file_size(pair_path), split_scores_size=file_size(score_path),
                    report_test=cfg.report_test, forward_only=True)
    from graph_tracks.artifacts import GraphForwardManifest
    write_json(manifest_path, GraphForwardManifest.model_validate(manifest).model_dump(by_alias=True))
    return (inference, encoder) if return_encoder else inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'listings', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--text-cache', type=Path)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--build-index', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    export(args.checkpoint, args.listings, args.output, text_cache=args.text_cache,
           pairs=args.pairs, build_index=args.build_index, device=args.device, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
