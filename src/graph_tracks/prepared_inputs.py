"""CPU preparation and hash-bound transfer of immutable graph topology.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`BatchCodec` — the persisted per-batch array layout (save/load of
  numeric, edges and the pooled segment topology bundles)
  (:func:`save_batch` / :func:`load_batch` stay the public faces).
- :class:`QueryBatches` — the query/{start} chunk plan shared by
  :func:`prepare_training` and :func:`prepare_inference`.
- :class:`SplitPopulation` — one split's listing indices, local pair remap
  and catalog-pair arrays (:func:`prepare_training`'s split loop).
- :class:`PreparedGraphInputs` / :class:`PreparedPoolingInputs` — the
  pydantic validation models (responsibilities unchanged).
"""
import json
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.run_log import RunLogger
from training.prepare_all_trace import timed
from graph_tracks.data import (
    GraphBatch,
    NUMERIC,
    RELATIONS,
    file_hash,
    fit_vocabulary,
    load_records,
    tensorize,
)
from graph_tracks.pooling import SegmentTopology, move_batch, topology

_LOG = RunLogger(__name__)

PLAN = 'graph_plan.json'
ARRAYS = 'graph_inputs.npz'


class BatchCodec:
    """The persisted per-batch array layout (dense + graph + pooled topology)."""

    @staticmethod
    def save(arrays, prefix, batch, vocabulary):
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
                segment = target._er_segment_topology
                arrays[stem+'/segment_order'] = segment.order.numpy()
                arrays[stem+'/segment_offsets'] = segment.offsets.numpy()

    @classmethod
    def load(cls, arrays, prefix, device, vocabulary):
        numeric = arrays[prefix+'/numeric']
        if (numeric.dtype != np.float32 or numeric.ndim != 2
                or numeric.shape[1] != len(NUMERIC)*3 or not np.isfinite(numeric).all()):
            raise ValueError('prepared graph numeric dtype/schema mismatch')
        batch = cls._edges(arrays, prefix, numeric, vocabulary)
        return move_batch(batch, device)

    @staticmethod
    def _edges(arrays, prefix, numeric, vocabulary):
        batch = GraphBatch(torch.as_tensor(numeric), {})
        batch._pool_topology = {}
        for relation in RELATIONS:
            edge_arrays = [arrays[prefix+'/'+relation+'/'+key] for key in ('listing', 'value')]
            if any(a.dtype != np.int64 for a in edge_arrays):
                raise ValueError('prepared graph edge dtype mismatch')
            left, right = edge_arrays
            if (left.ndim != 1 or right.shape != left.shape
                    or np.any(left < 0) or np.any(left >= len(numeric))
                    or np.any(right < 0) or np.any(right > len(vocabulary[relation]))):
                raise ValueError('prepared graph edge shape/bounds mismatch')
            listing, value = [torch.as_tensor(a) for a in edge_arrays]
            batch.edges[relation] = (listing, value)
            signature = (id(listing), None if torch.is_inference(listing) else listing._version,
                         id(value), None if torch.is_inference(value) else value._version)
            for attribute in (False, True):
                count = len(vocabulary[relation])+1 if attribute else len(numeric)
                stem = prefix+'/'+relation+('/attribute' if attribute else '/listing_pool')
                prepared = PreparedPoolingInputs.from_arrays(arrays, stem, count)
                valid = right != 0
                expected_source, expected_target = ((left[valid], right[valid]) if attribute
                                                    else (right, left))
                if (not np.array_equal(prepared.source, expected_source)
                        or not np.array_equal(prepared.target, expected_target)):
                    raise ValueError('prepared graph pooling direction differs from membership edges')
                source, target, sizes = [torch.as_tensor(a) for a in
                                         (prepared.source, prepared.target, prepared.sizes)]
                target._er_segment_topology = SegmentTopology(
                    torch.as_tensor(prepared.order), torch.as_tensor(prepared.offsets),
                    None if torch.is_inference(target) else target._version, count)
                batch._pool_topology[(relation, attribute, count, torch.float32)] = (
                    signature, source, target, sizes, listing, value)
        return batch


class QueryBatches:
    """The query/{start} chunk plan shared by prepare_training/prepare_inference."""

    def __init__(self, records: list[dict], vocabulary, arrays, *, batch_size: int, name: str):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self._records, self._vocabulary, self._arrays = records, vocabulary, arrays
        self._size, self._name = batch_size, name

    def run(self) -> list[str]:
        """Persist every query chunk; returns the plan's batch prefixes."""
        batches: list[str] = []
        starts = list(range(0, len(self._records), self._size))
        for start in _LOG.progress(
            starts, desc=f'query_batch_save_{self._name}', unit='batch',
            total=len(starts),
        ):
            prefix = f'query/{start}'
            batch = tensorize(self._records[start:start+self._size], self._vocabulary, 'cpu')
            BatchCodec.save(self._arrays, prefix, batch, self._vocabulary)
            batches.append(prefix)
        return batches


class SplitPopulation:
    """One split's listing indices, local pair remap and catalog pairs."""

    def __init__(self, name: str, records: list[dict], pair_data):
        self.name = name
        self.indices = [i for i, r in enumerate(records) if r['split'] == name]
        self._pairs, self._labels = pair_data

    @property
    def lookup(self) -> dict[int, int]:
        return {i: n for n, i in enumerate(self.indices)}

    def arrays(self) -> dict[str, np.ndarray]:
        lookup = self.lookup
        return {
            self.name+'/pairs': np.asarray(
                [(lookup[int(a)], lookup[int(b)]) for a, b in self._pairs],
                dtype=np.int64).reshape(-1, 2),
            self.name+'/labels': self._labels,
            self.name+'/catalog_pairs': self._pairs,
        }


def save_batch(arrays, prefix, batch, vocabulary):
    """Persist one batch's dense/edge/pooling arrays (see :class:`BatchCodec`)."""
    BatchCodec.save(arrays, prefix, batch, vocabulary)


def load_batch(arrays, prefix, device, vocabulary):
    """Load one batch with its pooled topology re-attached (see :class:`BatchCodec`)."""
    return BatchCodec.load(arrays, prefix, device, vocabulary)


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
        self._check_ids()
        start = self._check_query_batches()
        if start != len(self.plan['ids']):
            raise ValueError('prepared graph query batch population mismatch')
        if records is None or 'populations' not in self.plan:
            return
        self._check_populations(records, pair_data)

    def _check_ids(self) -> None:
        ids = self.plan['ids']
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('prepared graph IDs must be unique and nonempty')

    def _check_query_batches(self) -> int:
        start = 0
        for prefix in self.plan['query_batches']:
            if prefix != f'query/{start}':
                raise ValueError('prepared graph query batch order/coverage mismatch')
            numeric = self.arrays[prefix+'/numeric']
            if numeric.ndim != 2 or len(numeric) == 0:
                raise ValueError('prepared graph query batch must contain rows')
            start += len(numeric)
        return start

    def _check_populations(self, records, pair_data) -> None:
        for split in ('train', 'dev', 'test'):
            population = self.plan['populations'][split]
            if population != [i for i, r in enumerate(records) if r['split'] == split]:
                raise ValueError('prepared graph split population mismatch')
            local = self.arrays[split+'/pairs']
            catalog = self.arrays[split+'/catalog_pairs']
            labels = self.arrays[split+'/labels']
            self._check_pair_arrays(population, local, catalog, labels)
            mapped = np.asarray(population, dtype=np.int64)[local]
            if not np.array_equal(mapped, catalog):
                raise ValueError('prepared graph local/catalog pair mapping mismatch')
            if pair_data is not None and (not np.array_equal(catalog, pair_data[split][0])
                                          or not np.array_equal(labels, pair_data[split][1])):
                raise ValueError('prepared graph pairs differ from source labels/endpoints')

    @staticmethod
    def _check_pair_arrays(population, local, catalog, labels) -> None:
        if (local.dtype != np.int64 or local.ndim != 2 or local.shape[1] != 2
                or catalog.dtype != np.int64 or catalog.shape != local.shape
                or labels.dtype != np.float32 or labels.shape != (len(local),)
                or not np.isin(labels, [0., 1.]).all()
                or np.any(local < 0) or np.any(local >= len(population))):
            raise ValueError('prepared graph pair dtype/shape/bounds mismatch')


class PreparedPoolingInputs(BaseModel):
    """CPU-validated pooling direction and stable, exact segment partition."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra='forbid')
    source: np.ndarray
    target: np.ndarray
    sizes: np.ndarray
    order: np.ndarray
    offsets: np.ndarray
    count: int = Field(ge=0, strict=True)

    @model_validator(mode='after')
    def validate_topology(self):
        self._check_shapes()
        lengths = np.bincount(self.target, minlength=self.count)
        self._check_denominators(lengths)
        self._check_order(lengths)
        return self

    def _check_shapes(self) -> None:
        source, target, sizes, count = self.source, self.target, self.sizes, self.count
        if (source.dtype != np.int64 or target.dtype != np.int64
                or source.ndim != 1 or target.shape != source.shape
                or sizes.dtype != np.float32 or sizes.shape != (count, 1)
                or np.any(target < 0) or np.any(target >= count)):
            raise ValueError('prepared graph pooling dtype/shape/bounds mismatch')

    def _check_denominators(self, lengths) -> None:
        if not np.array_equal(self.sizes[:, 0], np.maximum(lengths, 1)):
            raise ValueError('prepared graph pooling denominators differ from edge counts')

    def _check_order(self, lengths) -> None:
        order, target, offsets = self.order, self.target, self.offsets
        if (order.dtype != np.int64 or order.shape != target.shape
                or offsets.dtype != np.int64 or offsets.shape != (self.count + 1,)
                or np.any(order < 0) or np.any(order >= len(target))
                or not np.array_equal(np.bincount(order, minlength=len(target)),
                                      np.ones(len(target), dtype=np.int64))):
            raise ValueError('prepared graph segment order must be an exact edge permutation')
        ordered = target[order]
        if (np.any(ordered[1:] < ordered[:-1])
                or np.any((ordered[1:] == ordered[:-1]) & (order[1:] < order[:-1]))
                or not np.array_equal(offsets, np.concatenate(([0], np.cumsum(lengths))))):
            raise ValueError('prepared graph segments must preserve stable edge order and exact offsets')

    @classmethod
    def from_arrays(cls, arrays, stem: str, count: int) -> "PreparedPoolingInputs":
        source, target, sizes = [arrays[stem+'/'+key] for key in ('source', 'target', 'sizes')]
        segment_keys = [stem+'/'+key for key in ('segment_order', 'segment_offsets')]
        present = [key in arrays for key in segment_keys]
        if any(present) and not all(present):
            raise ValueError('prepared graph segment metadata is incomplete')
        if all(present):
            order, offsets = [arrays[key] for key in segment_keys]
        else:
            # Older hash-bound packages are compatible: derive their missing
            # integer metadata on CPU, before uploading any topology.
            segment = SegmentTopology.prepare(torch.as_tensor(target), count)
            order, offsets = segment.order.numpy(), segment.offsets.numpy()
        return cls(source=source, target=target, sizes=sizes, order=order,
                   offsets=offsets, count=count)


@timed
def prepare_training(listings: Path, pairs: Path, output: Path | None = None, *, batch_size=1024):
    """Run locally before packaging; never perform a model forward.

    Threading: :class:`SplitPopulation` (split loop) -> :class:`QueryBatches`
    (the query/{start} chunks) -> the hash-bound plan JSON. Validation lives
    on the PreparedGraphInputs loader (load_plan), not here.
    """
    from graph_tracks.train import load_pairs, write_json
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = output or listings.parent
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(listings)
    pair_data = load_pairs(pairs, records)
    vocabulary = fit_vocabulary(records)
    arrays, populations = {}, {}
    for split in _LOG.progress(('train', 'dev', 'test'), desc='prepared_split_arrays', unit='split'):
        population = SplitPopulation(split, records, pair_data[split])
        populations[split] = population.indices
        arrays.update(population.arrays())
        if population.indices and split != 'test':
            save_batch(arrays, split,
                       tensorize([records[i] for i in population.indices], vocabulary, 'cpu'),
                       vocabulary)
    batches = QueryBatches(
        records, vocabulary, arrays, batch_size=batch_size, name='training',
    ).run()
    with (output/ARRAYS).open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    plan = {'schema': 'er-graph-prepared-v1', 'listings_sha256': file_hash(listings),
            'pairs_sha256': file_hash(pairs), 'arrays_sha256': file_hash(output/ARRAYS),
            'support_listings_sha256': file_hash(listings),
            'relations': list(RELATIONS), 'numeric': list(NUMERIC), 'vocabulary': vocabulary,
            'ids': [r['sku_id'] for r in records], 'populations': populations,
            'query_batches': batches}
    write_json(output/PLAN, plan)
    return output/PLAN


@timed
def prepare_inference(listings: Path, checkpoint: Path, output: Path | None = None, *, batch_size=1024):
    """Prepare new inductive queries locally with checkpoint-native support.

    Threading: the checkpoint's support records persist under 'train' (the
    checkpoint-native support set), then the inductive queries are chunked by
    :class:`QueryBatches` exactly like training preparation.
    """
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
    arrays: dict[str, np.ndarray] = {}
    save_batch(arrays, 'train', tensorize(payload['support_records'], vocabulary, 'cpu'), vocabulary)
    batches = QueryBatches(
        records, vocabulary, arrays, batch_size=batch_size, name='inference',
    ).run()
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


@timed
def load_plan(listings: Path, pairs: Path | None = None):
    """Hash-check the prepared package, then CPU-validate its topology."""
    path = listings.parent/PLAN
    plan = json.loads(path.read_text())
    if (plan.get('schema') != 'er-graph-prepared-v1'
            or plan.get('relations') != list(RELATIONS)
            or plan.get('numeric') != list(NUMERIC)):
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
