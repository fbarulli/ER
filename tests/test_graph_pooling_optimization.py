"""Static topology reuse must preserve reduction values and gradients."""
import pytest
import torch

from graph_tracks.data import GraphBatch
from graph_tracks.model import mean_pool
from graph_tracks.pooling import pool, topology


@pytest.mark.parametrize('attribute', [False, True])
def test_cached_pool_matches_values_and_gradients(attribute):
    listing = torch.tensor([0, 0, 1, 2, 2])
    value = torch.tensor([1, 2, 0, 2, 2])
    batch = GraphBatch(torch.zeros(3, 6), {'brand': (listing, value)})
    source, target, sizes = topology(batch, 'brand', attribute=attribute, count=4)
    x = torch.randn(4, 8, requires_grad=True)
    valid = value != 0 if attribute else torch.ones_like(value, dtype=torch.bool)
    expected_source = listing[valid] if attribute else value
    expected_target = value[valid] if attribute else listing
    expected = mean_pool(x[expected_source], expected_target, 4)
    actual = pool(x[source], target, sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert topology(batch, 'brand', attribute=attribute, count=4)[2] is sizes
    assert not sizes.requires_grad
    # In-place changes or replacing topology must invalidate static metadata.
    value[0] = 0
    changed = topology(batch, 'brand', attribute=attribute, count=4)
    assert changed[2] is not sizes
    batch.edges['brand'] = (listing.clone(), value.clone())
    assert topology(batch, 'brand', attribute=attribute, count=4)[2] is not changed[2]


def test_primed_graph_compiles_without_topology_graph_breaks():
    from graph_tracks.data import RELATIONS, tensorize
    from graph_tracks.model import AttributeGNN
    vocabulary = {r: ['a'] for r in RELATIONS}
    batch = tensorize([{'attribute': {r: ['a'] for r in RELATIONS},
                        'numeric': {}}], vocabulary, 'cpu')
    model = AttributeGNN(vocabulary, hidden=8, output=8)
    expected = model.encode(batch, model.context(batch))
    # The eager backend verifies graph capture/parity without Inductor setup.
    context = torch.compile(model.context, backend='eager', fullgraph=True)
    encode = torch.compile(model.encode, backend='eager', fullgraph=True)
    actual = encode(batch, context(batch))
    torch.testing.assert_close(actual, expected)


def test_inference_mode_topology():
    with torch.inference_mode():
        batch = GraphBatch(torch.zeros(1, 6),
                           {'brand': (torch.tensor([0]), torch.tensor([1]))})
        source, target, sizes = topology(batch, 'brand')
        assert source.tolist() == [1]
        assert target.tolist() == [0]
        assert sizes.tolist() == [[1.0]]
