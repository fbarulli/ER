"""TASK B item 12: graph architecture switches are additive and default-off.

Regularization flags (dropout, edge_dropout) must be EXACTLY a no-op in eval
mode, so inference built from a checkpoint is unchanged when they are enabled.
Structural flags (residual, two_hop, gated_pool) change the parameterization
and are checked to forward with the right output shape.
"""
import torch

from graph_tracks.data import RELATIONS, tensorize
from graph_tracks.model import AttributeGNN

VOCABULARY = {relation: ['a', 'b'] for relation in RELATIONS}


def _batch():
    def row(first, second):
        return {
            'attribute': {
                relation: (first, second)[index % 2]
                for index, relation in enumerate(RELATIONS)
            },
            'numeric': {},
        }

    return tensorize(
        [row('a', 'b'), row('b', 'b'), row('a', 'a')], VOCABULARY, 'cpu'
    )


def _base(seed=0):
    torch.manual_seed(seed)
    return AttributeGNN(VOCABULARY, hidden=8, output=8)


def test_regularization_flags_are_eval_noops():
    batch = _batch()
    base = _base()
    torch.manual_seed(0)
    regularized = AttributeGNN(
        VOCABULARY, hidden=8, output=8, dropout=0.5, edge_dropout=0.5
    )
    regularized.load_state_dict(base.state_dict())
    base.eval()
    regularized.eval()
    with torch.no_grad():
        b = base.encode(batch, base.context(batch))
        r = regularized.encode(batch, regularized.context(batch))
    assert torch.equal(b, r)


def test_regularization_flags_instantiate_modules():
    model = AttributeGNN(
        VOCABULARY, hidden=8, output=8, dropout=0.5, edge_dropout=0.25
    )
    assert model.dropout is not None
    assert model.edge_dropout == 0.25


def test_structural_flags_forward_with_expected_shape():
    batch = _batch()
    for kwargs in (
        {"residual": True},
        {"two_hop": True},
        {"gated_pool": True},
        {"residual": True, "two_hop": True, "gated_pool": True},
    ):
        model = AttributeGNN(VOCABULARY, hidden=8, output=8, **kwargs)
        model.eval()
        out = model.encode(batch, model.context(batch))
        assert out.shape[0] == len(batch.numeric)


def test_structural_flags_add_parameters():
    base = _base()
    structural = AttributeGNN(
        VOCABULARY, hidden=8, output=8, residual=True, two_hop=True, gated_pool=True
    )
    base_params = sum(p.numel() for p in base.parameters())
    structural_params = sum(p.numel() for p in structural.parameters())
    assert structural_params > base_params


def test_invalid_dropout_rejected():
    import pytest

    with pytest.raises(ValueError):
        AttributeGNN(VOCABULARY, hidden=8, output=8, dropout=1.0)
    with pytest.raises(ValueError):
        AttributeGNN(VOCABULARY, hidden=8, output=8, edge_dropout=-0.1)
