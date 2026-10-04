"""CPU preparation and hash-bound transfer of immutable graph topology."""
import json
from pydantic import BaseModel, ConfigDict, model_validator
from pathlib import Path
import numpy as np
import torch
from graph_tracks.data import GraphBatch, RELATIONS, NUMERIC, file_hash, fit_vocabulary, load_records, tensorize
from graph_tracks.pooling import topology

PLAN = 'graph_plan.json'
ARRAYS = 'graph_inputs.npz'


class PreparedGraphInputs(BaseModel):
    """Trace catalog rows through immutable query batches and split pair indices."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra='forbid')

    plan: dict
    arrays: object

    @model_validator(mode='after')
    def validate_batches(self):
        self.validate_catalog()
        return self

    def validate_catalog(self, records=None, pair_data=None):
        ids = self.plan['ids']
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('prepared graph IDs must be unique and nonempty')
        if records is not None and ids != [r['sku_id'] for r in records]:
            raise ValueError('prepared graph ID order mismatch')
        start = 0
        for prefix in self.plan['query_batches']:
            if prefix != f'query/{start}':
                raise ValueError('prepared graph query batch order/coverage mismatch')
            numeric = self.arrays[prefix+'/numeric']
            if numeric.ndim != 2 or len(numeric) == 0:
                raise ValueError('prepared graph query batch must contain rows')
            start += len(numeric)
        if start != len(ids):
            raise ValueError('prepared graph query batch population mismatch')
        if records is None or 'populations' not in self.plan:
            return
        for split in ('train', 'dev', 'test'):
            population = self.plan['populations'][split]
            if population != [i for i, r in enumerate(records) if r['split'] == split]:
                raise ValueError('prepared graph split population mismatch')
            local = self.arrays[split+'/pairs']
            catalog = self.arrays[split+'/catalog_pairs']
            labels = self.arrays[split+'/labels']
            if (local.dtype != np.int64 or local.ndim != 2 or local.shape[1] != 2
                    or catalog.dtype != np.int64 or catalog.shape != local.shape
                    or labels.dtype != np.float32 or labels.shape != (len(local),)
                    or not np.isin(labels, [0., 1.]).all()
                    or np.any(local < 0) or np.any(local >= len(population))):
                raise ValueError('prepared graph pair dtype/shape/bounds mismatch')
            mapped = np.asarray(population, dtype=np.int64)[local]
            if not np.array_equal(mapped, catalog):
                raise ValueError('prepared graph local/catalog pair mapping mismatch')
            if pair_data is not None and (not np.array_equal(catalog, pair_data[split][0])
                                         or not np.array_equal(labels, pair_data[split][1])):
                raise ValueError('prepared graph pairs differ from source labels/endpoints')


def save_batch(arrays, prefix, batch, vocabulary):
    arrays[prefix+'/numeric'] = batch.numeric.numpy()
    for relation, (listing, value) in batch.edges.items():
        arrays[prefix+'/'+relation+'/listing'] = listing.numpy()
        arrays[prefix+'/'+relation+'/value'] = value.numpy()
        for attribute in (False, True):
            count = len(vocabulary[relation])+1 if attribute else len(batch.numeric)
            source, target, sizes = topology(batch, relation, attribute=attribute, count=count)
            stem = prefix+'/'+relation+('/attribute' if attribute else '/listing_pool')
            for key, tensor in zip(('source', 'target', 'sizes'), (source, target, sizes)):
                arrays[stem+'/'+key] = tensor.numpy()


def load_batch(arrays, prefix, device, vocabulary):
    numeric = arrays[prefix+'/numeric']
    if numeric.dtype != np.float32 or numeric.ndim != 2 or numeric.shape[1] != len(NUMERIC)*3 or not np.isfinite(numeric).all():
        raise ValueError('prepared graph numeric dtype/schema mismatch')
    batch = GraphBatch(torch.as_tensor(numeric, device=device), {})
    batch._pool_topology = {}
    for relation in RELATIONS:
        edge_arrays = [arrays[prefix+'/'+relation+'/'+key] for key in ('listing', 'value')]
        if any(a.dtype != np.int64 for a in edge_arrays):
            raise ValueError('prepared graph edge dtype mismatch')
        left, right = edge_arrays
        if left.ndim != 1 or right.shape != left.shape or np.any(left < 0) or np.any(left >= len(numeric)) or np.any(right < 0) or np.any(right > len(vocabulary[relation])):
            raise ValueError('prepared graph edge shape/bounds mismatch')
        listing, value = [torch.as_tensor(a, device=device) for a in edge_arrays]
        batch.edges[relation] = (listing, value)
        signature = (id(listing), listing._version, id(value), value._version)
        for attribute in (False, True):
            count = len(vocabulary[relation])+1 if attribute else len(numeric)
            stem = prefix+'/'+relation+('/attribute' if attribute else '/listing_pool')
            tensors = [torch.as_tensor(arrays[stem+'/'+key], device=device) for key in ('source', 'target', 'sizes')]
            if tensors[0].dtype != torch.long or tensors[1].dtype != torch.long or tensors[2].dtype != torch.float32:
                raise ValueError('prepared graph pooling dtype mismatch')
            source, target, sizes = [arrays[stem+'/'+key] for key in ('source', 'target', 'sizes')]
            source_count = len(numeric) if attribute else len(vocabulary[relation])+1
            if source.ndim != 1 or target.shape != source.shape or sizes.shape != (count, 1) or not np.isfinite(sizes).all() or np.any(sizes < 1) or np.any(source < 0) or np.any(source >= source_count) or np.any(target < 0) or np.any(target >= count):
                raise ValueError('prepared graph pooling shape/bounds mismatch')
            batch._pool_topology[(relation, attribute, count, torch.float32)] = (signature, *tensors, listing, value)
    return batch


def prepare_training(listings: Path, pairs: Path, output: Path | None = None, *, batch_size=1024):
    """Run locally before packaging; never perform a model forward."""
    from graph_tracks.train import load_pairs, write_json
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = output or listings.parent
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(listings)
    pair_data = load_pairs(pairs, records)
    vocabulary = fit_vocabulary(records)
    arrays, populations = {}, {}
    for split in ('train', 'dev', 'test'):
        indices = [i for i, r in enumerate(records) if r['split'] == split]
        populations[split] = indices
        lookup = {i: n for n, i in enumerate(indices)}
        arrays[split+'/pairs'] = np.asarray([(lookup[int(a)], lookup[int(b)]) for a, b in pair_data[split][0]], dtype=np.int64).reshape(-1, 2)
        arrays[split+'/labels'] = pair_data[split][1]
        arrays[split+'/catalog_pairs'] = pair_data[split][0]
        if indices and split != 'test':
            save_batch(arrays, split, tensorize([records[i] for i in indices], vocabulary, 'cpu'), vocabulary)
    batches = []
    for start in range(0, len(records), batch_size):
        prefix = f'query/{start}'
        save_batch(arrays, prefix, tensorize(records[start:start+batch_size], vocabulary, 'cpu'), vocabulary)
        batches.append(prefix)
    with (output/ARRAYS).open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    plan = {'schema': 'er-graph-prepared-v1', 'listings_sha256': file_hash(listings),
            'pairs_sha256': file_hash(pairs), 'arrays_sha256': file_hash(output/ARRAYS),
            'support_listings_sha256': file_hash(listings),
            'relations': list(RELATIONS), 'numeric': list(NUMERIC), 'vocabulary': vocabulary,
            'ids': [r['sku_id'] for r in records], 'populations': populations, 'query_batches': batches}
    write_json(output/PLAN, plan)
    return output/PLAN


def prepare_inference(listings: Path, checkpoint: Path, output: Path | None = None, *, batch_size=1024):
    """Prepare new inductive queries locally with checkpoint-native support."""
    from graph_tracks.artifacts import checkpoint_track
    from graph_tracks.train import write_json
    checkpoint_track(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    records = load_records(listings, require_training=False)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = output or listings.parent
    output.mkdir(parents=True, exist_ok=True)
    vocabulary = payload['vocabulary']
    arrays = {}
    save_batch(arrays, 'train', tensorize(payload['support_records'], vocabulary, 'cpu'), vocabulary)
    batches = []
    for start in range(0, len(records), batch_size):
        prefix = f'query/{start}'
        save_batch(arrays, prefix, tensorize(records[start:start+batch_size], vocabulary, 'cpu'), vocabulary)
        batches.append(prefix)
    with (output/ARRAYS).open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    plan = {'schema': 'er-graph-prepared-v1', 'listings_sha256': file_hash(listings),
            'checkpoint_sha256': file_hash(checkpoint),
            'support_listings_sha256': payload['manifest']['listings_sha256'],
            'arrays_sha256': file_hash(output/ARRAYS), 'relations': list(RELATIONS),
            'numeric': list(NUMERIC), 'vocabulary': vocabulary,
            'ids': [r['sku_id'] for r in records], 'query_batches': batches}
    write_json(output/PLAN, plan)
    return output/PLAN


def load_plan(listings: Path, pairs: Path | None = None):
    path = listings.parent/PLAN
    plan = json.loads(path.read_text())
    if plan.get('schema') != 'er-graph-prepared-v1' or plan.get('relations') != list(RELATIONS) or plan.get('numeric') != list(NUMERIC):
        raise ValueError('prepared graph schema mismatch')
    for key, source in [('listings_sha256', listings), ('arrays_sha256', path.parent/ARRAYS)]:
        if plan.get(key) != file_hash(source):
            raise ValueError(f'prepared graph mismatch: {key}')
    if pairs is not None and plan.get('pairs_sha256') != file_hash(pairs):
        raise ValueError('prepared graph pair mismatch')
    arrays = np.load(path.parent/ARRAYS, allow_pickle=False)
    try:
        records = load_records(listings, require_training=pairs is not None)
        pair_data = None
        if pairs is not None:
            from graph_tracks.train import load_pairs
            pair_data = load_pairs(pairs, records)
        PreparedGraphInputs(plan=plan, arrays=arrays).validate_catalog(records, pair_data)
    except Exception:
        arrays.close()
        raise
    return plan, arrays


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listings', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    print(prepare_training(args.listings, args.pairs, batch_size=args.batch_size))


if __name__ == '__main__':
    main()
