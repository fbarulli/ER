"""Fused pooling must be bit-identical to the two-pass ``pool``.

A3: the value scatter and the degree scatter are fused into one pass per call
(the degree scatter is computed once per fixed topology and cached, or, with
``single_scatter``, folded into the value scatter itself). Both settings are
checked here against ``graph_tracks.pooling.pool`` -- forward *and* gradient --
on CPU fixtures that include duplicated targets, an empty segment and both
float32/float64.
"""
import pytest
import torch

from graph_tracks.data import GraphBatch
from graph_tracks.model import mean_pool
from graph_tracks.pooling import _segment_degrees, _segment_denominators, fused_pool, pool, topology

# listing = relation edges, value = attribute ids; row 3 of the 4-row population
# receives no edge (empty segment), row 0 receives two.
LISTING = torch.tensor([0, 0, 1, 2, 2, 2, 0])
VALUE = torch.tensor([1, 2, 0, 2, 2, 3, 4])


def _edge_fixture(dtype, *, attribute=False, count=4):
    batch = GraphBatch(torch.zeros(count, 6), {'brand': (LISTING, VALUE)})
    source, target, sizes = topology(batch, 'brand', attribute=attribute, count=count, dtype=dtype)
    values = torch.randn(int(source.max()) + 1, 8, dtype=dtype)
    return values[source], target, sizes


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('single_scatter', [False, True])
def test_fused_pool_is_bit_identical_to_pool(dtype, single_scatter):
    edges, target, sizes = _edge_fixture(dtype)
    expected = pool(edges, target, sizes)
    actual = fused_pool(edges, target, len(sizes), dtype=dtype, single_scatter=single_scatter)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('single_scatter', [False, True])
def test_fused_pool_preserves_the_attribute_filtered_topology(single_scatter):
    edges, target, sizes = _edge_fixture(torch.float32, attribute=True, count=5)
    expected = pool(edges, target, sizes)
    actual = fused_pool(edges, target, len(sizes), dtype=torch.float32,
                        single_scatter=single_scatter)
    assert torch.equal(actual, expected)
    # an empty segment divides by the clamp, not by zero
    assert sizes.flatten().min() == 1.0


@pytest.mark.parametrize('single_scatter', [False, True])
def test_fused_pool_gradients_match_pool_bit_for_bit(single_scatter):
    edges, target, sizes = _edge_fixture(torch.float32)
    required = edges.clone().requires_grad_(True)
    actual = fused_pool(required, target, len(sizes), single_scatter=single_scatter)
    expected = pool(required, target, sizes)
    actual_grad = torch.autograd.grad(actual.square().sum(), required)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), required, retain_graph=True)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert bool(actual_grad.abs().sum() > 0)


def test_single_scatter_switch_is_read_per_call(monkeypatch):
    edges, target, sizes = _edge_fixture(torch.float32)
    expected = pool(edges, target, sizes)
    monkeypatch.setenv('ER_PERF_GRAPH_FUSED_POOL_SINGLE_SCATTER', '1')
    assert torch.equal(fused_pool(edges, target, len(sizes), dtype=torch.float32), expected)
    monkeypatch.setenv('ER_PERF_GRAPH_FUSED_POOL_SINGLE_SCATTER', '0')
    assert torch.equal(fused_pool(edges, target, len(sizes), dtype=torch.float32), expected)


def test_half_precision_keeps_the_cached_degree_path():
    edges, target, sizes = _edge_fixture(torch.float16)
    # float16 cannot represent every degree exactly in an appended ones column,
    # so the single-scatter request degrades to the cached-degree fusion.
    actual = fused_pool(edges, target, len(sizes), dtype=torch.float16, single_scatter=True)
    assert torch.equal(actual, pool(edges, target, sizes))


def test_degree_scatter_and_denominators_are_cached_until_the_index_moves():
    target = LISTING.clone()
    denominators = _segment_denominators(target, 4, torch.float32)
    assert _segment_denominators(target, 4, torch.float32) is denominators
    assert _segment_degrees(target, 4, torch.float32) is _segment_degrees(target, 4, torch.float32)
    assert torch.equal(denominators.flatten(), torch.tensor([3.0, 1.0, 3.0, 1.0]))
    target[0] = 1                                   # in-place mutation invalidates
    resolved = _segment_denominators(target, 4, torch.float32)
    assert resolved is not denominators
    assert torch.equal(resolved.flatten(), torch.tensor([2.0, 2.0, 3.0, 1.0]))
    assert torch.equal(resolved.flatten(), torch.bincount(target, minlength=4).clamp_min(1).float())


def test_topology_denominators_agree_with_the_fused_cache():
    batch = GraphBatch(torch.zeros(4, 6), {'brand': (LISTING.clone(), VALUE.clone())})
    _, target, sizes = topology(batch, 'brand', count=4)
    fused = fused_pool(target.new_zeros((len(target), 8)), target, 4)
    assert torch.equal(fused, pool(target.new_zeros((len(target), 8)), target, sizes))
    assert torch.equal(_segment_denominators(target, 4, torch.float32), sizes)


def test_repeated_fused_pool_reuses_the_same_denominators():
    edges, target, sizes = _edge_fixture(torch.float32)
    first = fused_pool(edges, target, len(sizes), dtype=torch.float32)
    cached = _segment_denominators(target, len(sizes), torch.float32)
    second = fused_pool(edges, target, len(sizes), dtype=torch.float32)
    assert _segment_denominators(target, len(sizes), torch.float32) is cached
    assert torch.equal(first, second)


def test_wired_mean_pool_helper_matches_the_reference_pool():
    row = torch.tensor([0, 0, 1, 2, 2, 2, 0])
    values = torch.randn(5, 8)
    sizes = torch.bincount(row, minlength=4).clamp_min(1).unsqueeze(1).float()
    assert torch.equal(mean_pool(values[row], row, 4), pool(values[row], row, sizes))


def test_legacy_two_pass_helper_is_reachable_and_identical(monkeypatch):
    """``ER_PERF_LEGACY=1`` still reaches the un-fused helper, bit-identically."""
    model_module = __import__('graph_tracks.model', fromlist=['mean_pool'])
    monkeypatch.setattr(model_module, '_FUSE_POOL_PASSES', False)
    row = torch.tensor([0, 0, 1, 2, 2, 2, 0])
    values = torch.randn(5, 8)
    sizes = torch.bincount(row, minlength=4).clamp_min(1).unsqueeze(1).float()
    legacy = model_module.mean_pool(values[row], row, 4)
    assert torch.equal(legacy, pool(values[row], row, sizes))


def test_fused_pool_rejects_population_mismatches():
    edges, target, sizes = _edge_fixture(torch.float32)
    with pytest.raises(ValueError, match='edge population'):
        fused_pool(edges, target[:-1], len(sizes))
    with pytest.raises(ValueError, match='edge population'):
        fused_pool(edges[:, 0], target, len(sizes))
    with pytest.raises(ValueError, match='share a device'):
        fused_pool(edges, target.to('meta'), len(sizes))
