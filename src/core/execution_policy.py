"""Shared config vocabulary for device-aware execution, without importing Torch."""
from typing import Literal

OptimizerBackend = Literal['auto', 'foreach', 'fused', 'cuda_fused']
AggregationBackend = Literal['index_add', 'segment', 'cuda_segment']

ResolvedOptimizerBackend = Literal['auto', 'foreach', 'fused']
ResolvedAggregationBackend = Literal['index_add', 'segment']


def resolve_aggregation(backend: AggregationBackend, device: str) -> ResolvedAggregationBackend:
    if backend == 'cuda_segment':
        return 'segment' if device.split(':', 1)[0] == 'cuda' else 'index_add'
    return backend
