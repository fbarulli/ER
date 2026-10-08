"""A4: the accelerator wiring in ``graph_tracks/model.py`` is additive.

Every wired path is checked to leave the model's output bit-identical to the
un-wired path on CPU, the switches are checked to default to *off*, and the
compile hook is checked to be applied lazily, once, and never to change the
forward result.
"""
import importlib
from contextlib import contextmanager

import torch

import graph_tracks.model as model_module
from core.perf_switches import legacy_mode
from graph_tracks.data import RELATIONS, tensorize
from graph_tracks.model import AttributeGNN
from graph_tracks.pooling import pool

VOCABULARY = {relation: ['a', 'b'] for relation in RELATIONS}


def _batch():
    def row(first, second):
        return {'attribute': {relation: (first, second)[index % 2]
                              for index, relation in enumerate(RELATIONS)},
                'numeric': {}}

    return tensorize([row('a', 'b'), row('b', 'b'), row('a', 'a')], VOCABULARY, 'cpu')


def _model(seed=0):
    torch.manual_seed(seed)
    return AttributeGNN(VOCABULARY, hidden=8, output=8)


def _wire_all(monkeypatch, *, segment_reduce=True, autocast=True, compile_=False):
    monkeypatch.setattr(model_module, '_WIRE_SEGMENT_REDUCE', segment_reduce)
    monkeypatch.setattr(model_module, '_WIRE_AUTOCAST', autocast)
    monkeypatch.setattr(model_module, '_WIRE_COMPILE', compile_)


def test_accelerator_wiring_defaults_to_off(monkeypatch):
    for name in ('ER_PERF_ACCEL_GRAPH_WIRE_SEGMENT_REDUCE',
                 'ER_PERF_ACCEL_GRAPH_WIRE_AUTOCAST', 'ER_PERF_ACCEL_GRAPH_WIRE_COMPILE'):
        monkeypatch.delenv(name, raising=False)
    reloaded = importlib.reload(model_module)
    try:
        assert reloaded._WIRE_SEGMENT_REDUCE is False
        assert reloaded._WIRE_AUTOCAST is False
        assert reloaded._WIRE_COMPILE is False
        # Fused pooling is the ordinary graph.* convention: on unless legacy mode
        # was exported for the whole run (ER_PERF_LEGACY=1 is read at import).
        assert reloaded._FUSE_POOL_PASSES is (not legacy_mode())
    finally:
        importlib.reload(model_module)


def test_index_add_pool_wiring_is_bit_identical_on_cpu(monkeypatch):
    values = torch.randn(12, 5)
    target = torch.tensor([0, 0, 1, 2, 2, 2, 0, 3, 3, 1, 0, 2])
    sizes = torch.bincount(target, minlength=4).clamp_min(1).unsqueeze(1).float()
    expected = pool(values, target, sizes)
    _wire_all(monkeypatch, segment_reduce=False)
    assert torch.equal(model_module._index_add_pool(values, target, sizes), expected)
    _wire_all(monkeypatch, segment_reduce=True)
    assert torch.equal(model_module._index_add_pool(values, target, sizes), expected)


def test_index_add_pool_keeps_pool_promotion_when_dtypes_differ(monkeypatch):
    values = torch.randn(6, 4, dtype=torch.float32)
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    sizes = torch.ones(3, 1, dtype=torch.float64)
    expected = pool(values, target, sizes)
    _wire_all(monkeypatch, segment_reduce=True)
    actual = model_module._index_add_pool(values, target, sizes)
    assert actual.dtype == expected.dtype == torch.float64
    assert torch.equal(actual, expected)


def test_attribute_gnn_output_is_unchanged_with_every_accelerator_wired(monkeypatch):
    batch = _batch()
    gnn = _model()
    baseline_model = _model()
    baseline_context = baseline_model.context(batch)
    baseline = baseline_model.encode(batch, baseline_context)
    _wire_all(monkeypatch)
    context = gnn.context(batch)
    wired = gnn.encode(batch, context)
    assert torch.equal(wired, baseline)
    for relation, state in context.items():
        assert torch.equal(state, baseline_context[relation])


def test_pool_backend_cache_and_legacy_path_agree(monkeypatch):
    batch = _batch()
    torch.manual_seed(3)
    gnn = AttributeGNN(VOCABULARY, hidden=8, output=8)
    with_cache = gnn.encode(batch, gnn.context(batch))
    monkeypatch.setattr(model_module, '_POOL_BACKEND_CACHE', False)
    without_cache = gnn.encode(batch, gnn.context(batch))
    assert torch.equal(with_cache, without_cache)


def test_compile_hook_is_lazy_idempotent_and_named(monkeypatch):
    calls = []

    def spy(module, *, name, mode='reduce-overhead', fullgraph=False):
        calls.append(name)
        return module

    monkeypatch.setattr(model_module, 'compile_model', spy)
    _wire_all(monkeypatch, compile_=True)
    gnn = _model()
    batch = _batch()
    assert calls == []                       # never compiled at construction
    gnn.context(batch)
    assert calls == ['gnn.context', 'gnn.encode']
    gnn.encode(batch, gnn.context(batch))
    assert calls == ['gnn.context', 'gnn.encode']   # applied once
    assert gnn.compile_accelerated_methods() is gnn


def test_compile_hook_is_not_applied_when_the_switch_is_off(monkeypatch):
    calls = []
    monkeypatch.setattr(model_module, 'compile_model',
                        lambda module, **kwargs: calls.append(kwargs) or module)
    _wire_all(monkeypatch, compile_=False)
    gnn = _model()
    gnn.encode(_batch(), gnn.context(_batch()))
    assert calls == []


def test_autocast_wiring_enters_the_shared_context_once_per_forward(monkeypatch):
    entered = []

    @contextmanager
    def recorder(device):
        entered.append(str(device))
        yield

    batch = _batch()
    baseline = _model(5).encode(batch, _model(5).context(batch))
    monkeypatch.setattr(model_module, 'autocast_context', recorder)
    _wire_all(monkeypatch, autocast=True)
    gnn = _model(5)
    context = gnn.context(batch)
    result = gnn.encode(batch, context)
    assert entered == ['cpu', 'cpu']
    assert torch.equal(result, baseline)


def test_autocast_wiring_is_skipped_when_switched_off(monkeypatch):
    def explode(device):  # pragma: no cover - must never be reached
        raise AssertionError('autocast must not be consulted while switched off')

    monkeypatch.setattr(model_module, 'autocast_context', explode)
    _wire_all(monkeypatch, autocast=False)
    gnn = _model()
    gnn.encode(_batch(), gnn.context(_batch()))


def test_wired_path_survives_fullgraph_compilation(monkeypatch):
    """The wired pooling/autocast paths must not introduce a graph break."""
    _wire_all(monkeypatch)
    batch = _batch()
    gnn = _model()
    baseline = gnn.encode(batch, gnn.context(batch))
    compiled_context = torch.compile(gnn.context, backend='eager', fullgraph=True)
    compiled_encode = torch.compile(gnn.encode, backend='eager', fullgraph=True)
    assert torch.equal(compiled_encode(batch, compiled_context(batch)), baseline)


def test_graph_disabled_model_still_returns_a_message_free_encoding(monkeypatch):
    _wire_all(monkeypatch)
    batch = _batch()
    torch.manual_seed(1)
    gnn = AttributeGNN(VOCABULARY, hidden=8, output=8, graph_enabled=False)
    encoded = gnn.encode(batch, gnn.context(batch))
    assert encoded.shape == (3, 8)
    assert torch.allclose(encoded.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_real_compilation_preserves_the_forward_pass(monkeypatch):
    """Real Inductor compilation of both methods (the switch's worst case)."""
    monkeypatch.delenv('ER_PERF_LEGACY', raising=False)
    monkeypatch.delenv('ER_PERF_ACCEL_COMPILE', raising=False)
    _wire_all(monkeypatch, compile_=True)
    batch = _batch()
    gnn = _model()
    baseline = _model()
    expected = baseline.encode(batch, baseline.context(batch))
    gnn.compile_accelerated_methods()
    actual = gnn.encode(batch, gnn.context(batch))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
