"""Reusable topology metadata for immutable, full-batch graph inputs.

Only integer topology and detached denominators are cached on the batch;
trainable values and autograd graphs are never retained between steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from graph_tracks.data import GraphBatch


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
        if attribute:
            valid = value != 0
            source, target = listing[valid], value[valid]
        else:
            source, target = value, listing
        sizes = torch.zeros(count, dtype=dtype, device=listing.device)
        sizes.index_add_(0, target, torch.ones(len(target), dtype=dtype, device=listing.device))
        cached = (signature, source, target, sizes.clamp_min(1).unsqueeze(1), listing, value)
        cache[key] = cached
    return cached[1:4]


def pool(values: torch.Tensor, target: torch.Tensor, sizes: torch.Tensor):
    total = values.new_zeros((len(sizes), values.shape[-1]))
    total.index_add_(0, target, values)
    return total / sizes


@dataclass(frozen=True)
class SegmentTopology:
    """Stable grouping of immutable edges, prepared once outside autograd."""
    order: torch.Tensor
    lengths: torch.Tensor
    version: int | None
    count: int

    @classmethod
    def prepare(cls, target: torch.Tensor, count: int) -> SegmentTopology:
        indices = target.detach().cpu()
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError('segment targets must be a vector of int64 indices')
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= count):
            raise ValueError('segment target outside pooling population')
        lengths = torch.bincount(indices, minlength=count)
        order = torch.argsort(indices, stable=True)
        return cls(order.to(target.device), lengths.to(target.device),
                   None if torch.is_inference(target) else target._version, count)


def segment_pool(values: torch.Tensor, target: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    cached = getattr(target, '_er_segment_topology', None)
    if not torch.compiler.is_compiling():
        version = None if torch.is_inference(target) else target._version
        if cached is None or cached.version != version or cached.count != len(sizes):
            cached = SegmentTopology.prepare(target, len(sizes))
            target._er_segment_topology = cached
    elif cached is None:
        raise RuntimeError('segment topology must be prepared before graph capture')
    # Preparation validates integer bounds and constructs exact lengths from
    # every edge. Skipping repeated length validation avoids CUDA scalar reads.
    total = torch.segment_reduce(values[cached.order], 'sum', lengths=cached.lengths,
                                 axis=0, unsafe=True, initial=0)
    return total / sizes
