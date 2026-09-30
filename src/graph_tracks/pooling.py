"""Reusable topology metadata for immutable, full-batch graph inputs.

Only integer topology and detached denominators are cached on the batch;
trainable values and autograd graphs are never retained between steps.
"""
from __future__ import annotations

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
