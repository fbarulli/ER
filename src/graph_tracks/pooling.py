"""Reusable topology metadata for immutable, full-batch graph inputs.

Only integer topology and detached denominators are cached on the batch;
trainable values and autograd graphs are never retained between steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from core.perf_switches import perf_enabled
from graph_tracks.data import GraphBatch


def _segment_degrees(target: torch.Tensor, count: int,
                     dtype: torch.dtype) -> torch.Tensor:
    """Raw per-row degrees for a fixed topology, scattered at most once.

    The degree ``index_add_`` is cached on the immutable index tensor and keyed
    by its version/count/dtype, so a fixed topology pays for the degree scatter
    only once. Callers apply ``clamp_min(1).unsqueeze(1)`` themselves, which
    keeps the exact denominator identity semantics of :func:`topology`.
    Invalidated automatically when the index is mutated in place.
    """
    cached = getattr(target, '_er_segment_degrees', None)
    version = None if torch.is_inference(target) else target._version
    if cached is not None and cached[0] == version and cached[1] == count and cached[2] == dtype:
        return cached[3]
    sizes = torch.zeros(count, dtype=dtype, device=target.device)
    sizes.index_add_(0, target, torch.ones(len(target), dtype=dtype, device=target.device))
    target._er_segment_degrees = (version, count, dtype, sizes)
    return sizes


def _segment_denominators(target: torch.Tensor, count: int,
                          dtype: torch.dtype) -> torch.Tensor:
    """Clamped ``(count, 1)`` denominators, derived from the cached degrees.

    Same identity as ``topology(...)[2]`` (``clamp_min(1).unsqueeze(1)``, so an
    empty relation row divides by ``1``), but cached alongside the degrees so a
    repeated :func:`fused_pool` neither re-scatters nor re-allocates.
    """
    cached = getattr(target, '_er_segment_denominators', None)
    version = None if torch.is_inference(target) else target._version
    if cached is not None and cached[0] == version and cached[1] == count and cached[2] == dtype:
        return cached[3]
    denominators = _segment_degrees(target, count, dtype).clamp_min(1).unsqueeze(1)
    target._er_segment_denominators = (version, count, dtype, denominators)
    return denominators


def topology(batch: GraphBatch, relation: str, *, attribute: bool = False,
             count: int | None = None, dtype: torch.dtype = torch.float32):
    listing, value = batch.edges[relation]
    count = len(batch.numeric) if count is None else count
    cache = getattr(batch, '_pool_topology', None)
    if cache is None:
        cache = batch._pool_topology = {}
    key = (relation, attribute, count, dtype)
    # Compiled callers prime immutable topology eagerly before capture.
    # Avoid tensor version reads and Python cache mutation inside the graph.
    if torch.compiler.is_compiling() and key in cache:
        return cache[key][1:4]
    signature = (id(listing), None if torch.is_inference(listing) else listing._version,
                 id(value), None if torch.is_inference(value) else value._version)
    cached = cache.get(key)
    if cached is None or cached[0] != signature:
        # Reuse the same prepared indices (and their segment metadata) when
        # autocast changes only the denominator dtype.
        prepared = next((entry for cache_key, entry in cache.items()
                         if cache_key[:3] == key[:3] and entry[0] == signature), None)
        if prepared is not None:
            cached = (signature, prepared[1], prepared[2], prepared[3].to(dtype), listing, value)
            cache[key] = cached
            return cached[1:4]
        if listing.device.type != 'cpu':
            raise RuntimeError('graph topology must be prepared on CPU before device transfer')
        if attribute:
            valid = value != 0
            source, target = listing[valid], value[valid]
        else:
            source, target = value, listing
        sizes = _segment_degrees(target, count, dtype)
        cached = (signature, source, target, sizes.clamp_min(1).unsqueeze(1), listing, value)
        target._er_segment_topology = SegmentTopology.prepare(target, count)
        cache[key] = cached
    return cached[1:4]


def pool(values: torch.Tensor, target: torch.Tensor, sizes: torch.Tensor):
    total = values.new_zeros((len(sizes), values.shape[-1]))
    total.index_add_(0, target, values)
    return total / sizes


def fused_pool(values: torch.Tensor, target: torch.Tensor, count: int,
               *, dtype: torch.dtype | None = None,
               single_scatter: bool | None = None):
    """Mean pooling that fuses the value and degree scattering passes.

    The value sum is the only ``index_add_`` executed per call: the clamped
    denominators come from :func:`_segment_denominators`, which scatters
    ``count`` degrees once per fixed ``(target, count, dtype)`` topology and
    reuses them afterwards. Semantics -- including the ``clamp_min(1)``
    empty-segment rule -- are bit-identical to
    ``pool(values, target, topology(...)[2])`` and the result stays
    differentiable with respect to ``values``.

    ``single_scatter`` (default ``perf_enabled('graph.fused_pool_single_scatter')``,
    off) collapses the two passes into literally one segmented scatter by
    appending a column of ones to ``values`` and scattering the combined
    ``(E, F + 1)`` block. That saves a kernel launch but copies the whole
    payload, so it only pays off for tall, narrow messages; the default keeps
    the cached-degree fusion, which moves far less memory for wide feature
    blocks. The single-scatter variant is restricted to float32/float64 because
    the appended ones column must represent every degree exactly (float32 is
    exact to 2**24 edges). Both settings are bit-identical to ``pool``.
    """
    if values.ndim != 2 or target.ndim != 1 or values.shape[0] != target.shape[0]:
        raise ValueError('fused pool values must match the edge population')
    if values.device != target.device:
        raise ValueError('fused pool values and topology must share a device')
    dtype = values.dtype if dtype is None else dtype
    if single_scatter is None:
        single_scatter = perf_enabled('graph.fused_pool_single_scatter', default=False)
    if single_scatter and values.dtype in (torch.float32, torch.float64):
        ones = torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)
        combined = values.new_zeros((count, values.shape[-1] + 1))
        combined.index_add_(0, target, torch.cat([values, ones], dim=1))
        total, degrees = combined[:, :-1], combined[:, -1].unsqueeze(1)
        if degrees.dtype != dtype:
            degrees = degrees.to(dtype)
        return total / degrees.clamp_min(1)
    total = values.new_zeros((count, values.shape[-1]))
    total.index_add_(0, target, values)
    return total / _segment_denominators(target, count, dtype)


@dataclass(frozen=True)
class SegmentTopology:
    """Stable grouping of immutable edges, prepared once outside autograd."""
    order: torch.Tensor
    offsets: torch.Tensor
    version: int | None
    count: int

    @classmethod
    def prepare(cls, target: torch.Tensor, count: int) -> SegmentTopology:
        if target.device.type != 'cpu':
            raise RuntimeError('segment topology must be prepared on CPU before device transfer')
        indices = target.detach()
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError('segment targets must be a vector of int64 indices')
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= count):
            raise ValueError('segment target outside pooling population')
        if count < 0:
            raise ValueError('segment population must be nonnegative')
        lengths = torch.bincount(indices, minlength=count)
        order = torch.argsort(indices, stable=True)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        return cls(order.to(target.device), offsets.to(target.device),
                   None if torch.is_inference(target) else target._version, count)


def segment_pool(values: torch.Tensor, target: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or target.ndim != 1 or values.shape[0] != target.shape[0]:
        raise ValueError('segment values must match the prepared edge population')
    if sizes.ndim != 2 or sizes.shape[1] != 1:
        raise ValueError('segment denominators must have one value per pooled row')
    if values.device != target.device or sizes.device != values.device:
        raise ValueError('segment values, topology, and denominators must share a device')
    cached = getattr(target, '_er_segment_topology', None)
    if not torch.compiler.is_compiling():
        version = None if torch.is_inference(target) else target._version
        if cached is None or cached.version != version or cached.count != len(sizes):
            cached = SegmentTopology.prepare(target, len(sizes))
            target._er_segment_topology = cached
    elif cached is None:
        raise RuntimeError('segment topology must be prepared before graph capture')
    # Preparation validates integer bounds and constructs exact offsets from
    # every edge. Skipping repeated offset validation avoids CUDA scalar reads.
    total = torch.segment_reduce(values[cached.order], 'sum', offsets=cached.offsets,
                                 axis=0, unsafe=True, initial=0)
    return total / sizes


def prime_batch(batch: GraphBatch, vocabulary: dict[str, list[str]]) -> None:
    """Build both detached pooling directions before any device upload."""
    if batch.numeric.device.type != 'cpu':
        raise ValueError('graph batch preparation requires CPU tensors')
    for relation in batch.edges:
        topology(batch, relation)
        topology(batch, relation, attribute=True, count=len(vocabulary[relation]) + 1)


def move_batch(batch: GraphBatch, device: str | torch.device) -> GraphBatch:
    """Transfer prepared metadata together with edges, without device readback."""
    device = torch.device(device)
    if batch.numeric.device == device:
        return batch
    if batch.numeric.device.type != 'cpu':
        raise ValueError('graph batches must transfer from their CPU preparation')
    moved = GraphBatch(batch.numeric.to(device), {
        relation: (listing.to(device), value.to(device))
        for relation, (listing, value) in batch.edges.items()})
    moved._pool_topology = {}
    for key, entry in batch._pool_topology.items():
        _, source, target, sizes, _, _ = entry
        segment = getattr(target, '_er_segment_topology', None)
        if segment is None:
            raise ValueError('prepared graph target is missing segment metadata')
        source, target, sizes = source.to(device), target.to(device), sizes.to(device)
        target._er_segment_topology = SegmentTopology(
            segment.order.to(device), segment.offsets.to(device),
            None if torch.is_inference(target) else target._version, segment.count)
        listing, value = moved.edges[key[0]]
        signature = (id(listing), None if torch.is_inference(listing) else listing._version,
                     id(value), None if torch.is_inference(value) else value._version)
        moved._pool_topology[key] = (signature, source, target, sizes, listing, value)
    return moved
