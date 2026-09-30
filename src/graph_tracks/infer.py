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
from graph_tracks.data import file_hash, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN, PairScorer
from graph_tracks.train import write_json


class GraphEncoder:
    def __init__(self, checkpoint: Path, device='cpu'):
        self.checkpoint = checkpoint
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get('schema') != 'er-graph-checkpoint-v1':
            raise ValueError('unsupported checkpoint schema')
        marker = checkpoint.parent / 'checkpoint_manifest.json'
        if not marker.is_file() or json.loads(marker.read_text())['files']['graph_model.pt'] != file_hash(checkpoint):
            raise ValueError('checkpoint missing completion marker or hash mismatch')
        self.manifest, self.vocabulary = payload['manifest'], payload['vocabulary']
        self.device = device
        cfg = self.manifest['config']
        self.model = AttributeGNN(self.vocabulary, cfg['hidden_dim'], cfg['output_dim'],
                                 payload['text_dim'], cfg['graph_enabled']).to(device)
        self.scorer = PairScorer(bool(payload['text_dim'])).to(device)
        self.model.load_state_dict(payload['model'])
        self.scorer.load_state_dict(payload['scorer'])
        self.model.eval()
        self.scorer.eval()
        support = tensorize(payload['support_records'], self.vocabulary, device)
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
        chunks = []
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = tensorize(records[start:start + batch_size], self.vocabulary, self.device)
                vectors = None if text is None else torch.as_tensor(
                    text[start:start + batch_size], device=self.device)
                chunks.append(self.model.encode(batch, self.states, vectors).cpu().numpy())
        vectors = np.concatenate(chunks)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
            raise ValueError('encoder produced invalid vectors')
        return vectors


def export(checkpoint: Path, listings: Path, output: Path, *, text_cache=None,
           pairs=None, build_index=False, device='cpu', batch_size=1024):
    if output.exists():
        raise FileExistsError(output)
    records = load_records(listings, require_training=False)
    ids = [r['product_id'] for r in records]
    encoder = GraphEncoder(checkpoint, device)
    text, metadata = None, None
    hybrid = encoder.manifest['track'] == 'hybrid'
    if hybrid != bool(text_cache):
        raise ValueError('hybrid requires text cache; gnn_only forbids it')
    if text_cache:
        text, metadata = load_text_cache(text_cache, ids)
        for key in ('checkpoint_sha256', 'composition'):
            if metadata[key] != encoder.manifest['text_metadata'][key]:
                raise ValueError(f'inference text cache mismatch: {key}')
    vectors = encoder.encode(records, text, batch_size)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'vectors.npz', ids=np.asarray(ids), embeddings=vectors)
    if pairs:
        frame = pd.read_csv(pairs, dtype=str, keep_default_na=False)
        if set(frame.columns) != {'product_id1', 'product_id2'}:
            raise ValueError('inference pairs require exactly product_id1,product_id2; labels are excluded')
        lookup = {key: i for i, key in enumerate(ids)}
        if any(key not in lookup for key in list(frame.product_id1) + list(frame.product_id2)):
            raise ValueError('unknown inference pair endpoint')
        indices = np.asarray([(lookup[a], lookup[b]) for a, b in
                              zip(frame.product_id1, frame.product_id2)], dtype=np.int64).reshape(-1, 2)
        with torch.no_grad():
            scores = encoder.scorer(torch.as_tensor(vectors, device=device),
                torch.as_tensor(indices, device=device),
                None if text is None else torch.as_tensor(text, device=device)).sigmoid().cpu().numpy()
        frame['score'] = scores
        frame.to_csv(output / 'pair_scores.csv', index=False)
    if build_index:
        from training.hnsw_index import PersistentHnswIndex
        index = PersistentHnswIndex(output / 'index', ef_construction=200, M=16, ef_search=100)
        index.build(vectors, ids, checkpoint=checkpoint, model_name=encoder.manifest['track'],
                    preprocessing_fingerprint=file_hash(listings))
    write_json(output / 'export_manifest.json', {
        'schema': 'er-graph-export-v1', 'checkpoint_sha256': file_hash(checkpoint),
        'listings_sha256': file_hash(listings), 'vectors_sha256': file_hash(output / 'vectors.npz'),
        'text_cache_sha256': file_hash(text_cache) if text_cache else None,
        'track': encoder.manifest['track'], 'graph_context': 'training-listings-only',
        'vector_kind': 'graph-informed', 'ann_reproduces_pair_scorer': False,
        'count': len(ids), 'dimension': vectors.shape[1], 'index_built': build_index})
    return output


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
