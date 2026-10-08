"""Guard, fallback and equivalence harness for the opt-in Triton reduction.

No GPU on this host: the real grouped kernel is *wired and guarded* here and can
only be validated numerically on a GPU session (see the skipped GPU test). What
these tests do exercise on CPU is the whole guard machine -- opt-in default,
capability refusal, the value/derivative safety refusal, the one-time numeric
validation against ``index_add_``, the permanent disable on disagreement or
launch error, and the equivalence harness itself.
"""
import pytest
import torch

import core.fast_kernels as fast_kernels
from core.perf_switches import legacy_mode
from core.fast_kernels import (
    check_segment_reduce_equivalence,
    cuda_graph_wrap,
    reference_segment_sum,
    reset_segment_guard,
    segment_reduce_capability,
    segment_reduce_fast,
    segment_reduce_status,
)
from graph_tracks.pooling import pool

# ``ER_PERF_LEGACY=1`` is the global kill switch: every ``perf_enabled`` read is
# forced off, so the opt-in defaults cannot be observed in a legacy-mode process.
pytestmark = pytest.mark.skipif(
    legacy_mode(),
    reason='ER_PERF_LEGACY=1 forces every accelerator switch off; run without it to '
           'observe the opt-in defaults',
)


@pytest.fixture(autouse=True)
def _isolated_guard():
    reset_segment_guard()
    yield
    reset_segment_guard()


def _fixture(edges=64, features=8, count=12, *, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(edges, features, generator=generator, dtype=dtype)
    index = torch.randint(0, count, (edges,), generator=generator)
    sizes = torch.bincount(index, minlength=count).clamp_min(1).unsqueeze(1).to(dtype)
    return values, index, sizes


def _force_eligible(monkeypatch):
    """Pretend the host is CUDA-capable so the launch path runs on CPU tensors."""
    monkeypatch.setattr(fast_kernels, '_segment_fast_eligibility',
                        lambda values: (True, 'forced for the CPU harness'))


def _inject_cpu_kernel(monkeypatch, *, corrupt=False):
    """Stand in for the Triton launch with an explicit per-segment sum.

    ``corrupt`` makes it disagree with ``index_add_`` so the correctness guard
    must reject it.
    """
    launched = []

    def launch(grouped, offsets, out, features, count, block_f):
        launched.append(int(count))
        for segment in range(count):
            start, end = int(offsets[segment]), int(offsets[segment + 1])
            if end > start:
                out[segment] = grouped[start:end].sum(0, dtype=torch.float32)
            if corrupt:
                out[segment] = out[segment] + 1.0

    if fast_kernels.triton is None:  # pragma: no cover - triton is installed here
        from types import SimpleNamespace
        monkeypatch.setattr(fast_kernels, 'triton', SimpleNamespace(
            next_power_of_2=lambda value: 1 << max(0, int(value) - 1).bit_length(),
            cdiv=lambda a, b: -(-int(a) // int(b))), raising=False)
    monkeypatch.setattr(fast_kernels, '_grouped_kernel_launch', launch)
    return launched


# ---------------------------------------------------------------------------
# strictly opt-in + capability report
# ---------------------------------------------------------------------------
def test_segment_switch_is_off_unless_explicitly_opted_in(monkeypatch):
    monkeypatch.delenv('ER_PERF_LEGACY', raising=False)
    monkeypatch.delenv('ER_PERF_ACCEL_SEGMENT_REDUCE', raising=False)
    assert segment_reduce_capability()['opt_in'] is False
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '1')
    assert segment_reduce_capability()['opt_in'] is True
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '0')
    assert segment_reduce_capability()['opt_in'] is False


def test_capability_names_every_missing_precondition():
    capability = segment_reduce_capability()
    assert set(capability) >= {'opt_in', 'triton', 'cuda', 'device', 'eligible', 'reason'}
    if not torch.cuda.is_available():
        assert capability['eligible'] is False
        assert 'CUDA' in capability['reason']
        assert capability['device'] is None
    else:  # pragma: no cover - GPU host
        assert capability['reason']


def test_experimental_cuda_graph_wrapper_is_a_no_op_unless_opted_in(monkeypatch):
    monkeypatch.delenv('ER_PERF_LEGACY', raising=False)
    monkeypatch.delenv('ER_PERF_ACCEL_CUDA_GRAPH', raising=False)
    marker = lambda value: value  # noqa: E731 - identity stand-in for a step
    assert cuda_graph_wrap(marker) is marker


# ---------------------------------------------------------------------------
# CPU fallback: bit-identical to pooling.pool, and the guard says why
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('reduce', ['mean', 'sum'])
def test_fast_path_falls_back_to_index_add_on_a_cpu_host(monkeypatch, reduce):
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '1')
    values, index, sizes = _fixture()
    total = values.new_zeros((len(sizes), values.shape[-1]))
    total.index_add_(0, index, values)
    expected = total if reduce == 'sum' else total / sizes
    actual = segment_reduce_fast(values, index, sizes, reduce=reduce)
    assert torch.equal(actual, expected)
    status = segment_reduce_status()
    assert status.active is False
    assert status.attempts == 0                 # nothing was ever launched
    assert 'CUDA' in status.reason


def test_grad_enabled_values_are_never_eligible(monkeypatch):
    """A raw kernel write cannot build an autograd edge: refuse it up front."""
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '1')
    monkeypatch.setattr(fast_kernels, '_TRITON_AVAILABLE', True)

    class _CudaLookingTensor:
        is_cuda = True
        requires_grad = True

        @staticmethod
        def is_floating_point():
            return True

    eligible, reason = fast_kernels._segment_fast_eligibility(_CudaLookingTensor())
    assert eligible is False
    assert 'autograd' in reason

    _CudaLookingTensor.requires_grad = False
    eligible, reason = fast_kernels._segment_fast_eligibility(_CudaLookingTensor())
    assert eligible is True and reason == 'eligible'


def test_grad_enabled_call_stays_differentiable_through_the_fallback():
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    values = torch.randn(8, 4, requires_grad=True)
    sizes = torch.bincount(index, minlength=3).clamp_min(1).unsqueeze(1).float()
    result = segment_reduce_fast(values, index, sizes)
    assert result.requires_grad
    result.square().mean().backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())


def test_float16_values_fall_back_because_the_kernel_accumulates_float32():
    values, index, sizes = _fixture(dtype=torch.float16)
    actual = segment_reduce_fast(values, index, sizes)
    assert torch.equal(actual, pool(values, index, sizes))


# ---------------------------------------------------------------------------
# equivalence harness (CPU-testable)
# ---------------------------------------------------------------------------
def test_harness_is_positive_on_cpu_and_reports_the_kernel_skip(monkeypatch):
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '1')
    values, index, sizes = _fixture(edges=200, features=5, count=30)
    verdict = check_segment_reduce_equivalence(values, index, len(sizes))
    assert verdict.fallback_equal is True
    assert verdict.fallback_max_abs_diff == 0.0
    assert verdict.fast_checked is False
    assert 'CUDA' in verdict.reason


def test_harness_catches_a_disagreeing_trial_kernel():
    values, index, sizes = _fixture(edges=32, features=3, count=6)
    verdict = check_segment_reduce_equivalence(
        values, index, len(sizes),
        trial_fn=lambda v, i, c: reference_segment_sum(v, i, c) + 1.0)
    assert verdict.fast_checked is True
    assert verdict.fast_equal is False
    assert verdict.fast_max_abs_diff == pytest.approx(1.0)


def test_harness_reports_a_raising_trial_instead_of_raising():
    values, index, sizes = _fixture(edges=16, features=2, count=4)

    def boom(values, index, count):
        raise RuntimeError('no kernel image is available for execution on the device')

    verdict = check_segment_reduce_equivalence(values, index, len(sizes), trial_fn=boom)
    assert verdict.fast_checked is False
    assert verdict.fallback_equal is True
    assert 'no kernel image' in verdict.reason


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reference_matches_the_production_sum_with_and_without_empty_segments(dtype):
    for empty_rows in (0, 5):
        values, index, sizes = _fixture(edges=150, features=4, count=20, dtype=dtype)
        if empty_rows:
            # Rows 20..24 have no incoming edge: denominators stay at the clamp.
            sizes = torch.bincount(index, minlength=20 + empty_rows).clamp_min(1).unsqueeze(1).to(dtype)
        reference = reference_segment_sum(values, index, len(sizes))
        fallback = segment_reduce_fast(values, index, sizes, reduce='sum')
        if dtype is torch.float64:
            assert torch.equal(fallback, reference)
        else:
            torch.testing.assert_close(fallback, reference, rtol=1e-6, atol=1e-6)
        if empty_rows:
            assert torch.equal(reference[20:], torch.zeros(empty_rows, 4, dtype=dtype))
            assert torch.equal(fallback[20:], torch.zeros(empty_rows, 4, dtype=dtype))


def test_harness_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='equivalence harness'):
        check_segment_reduce_equivalence(torch.zeros(3, 2), torch.zeros(2, dtype=torch.long), 4)


# ---------------------------------------------------------------------------
# the guard machine, driven through an injected CPU kernel
# ---------------------------------------------------------------------------
def test_validated_kernel_is_used_and_revalidated_only_once(monkeypatch):
    _force_eligible(monkeypatch)
    launched = _inject_cpu_kernel(monkeypatch)
    values, index, sizes = _fixture()
    actual = segment_reduce_fast(values, index, sizes)
    expected = pool(values, index, sizes)
    status = segment_reduce_status()
    assert status.state == 'validated' and status.attempts == 1
    assert 'validated' in status.reason
    # the guard's declared budget is what reconciles grouped vs scatter order
    torch.testing.assert_close(actual, expected, rtol=fast_kernels._GUARD_RTOL,
                              atol=fast_kernels._GUARD_ATOL)
    assert len(launched) == 2  # caller values + deterministic probe, once
    segment_reduce_fast(values, index, sizes)
    assert len(launched) == 3  # reused: no second validation
    assert segment_reduce_status().attempts == 1


def test_a_changed_shape_is_revalidated_not_inherited(monkeypatch):
    """The guard keys on the edge count too, so a new population re-validates."""
    _force_eligible(monkeypatch)
    launched = _inject_cpu_kernel(monkeypatch)
    values, index, sizes = _fixture(edges=64, features=8, count=12)
    segment_reduce_fast(values, index, sizes)
    assert segment_reduce_status().attempts == 1
    other_values, other_index, other_sizes = _fixture(edges=40, features=8, count=12)
    segment_reduce_fast(other_values, other_index, other_sizes)
    status = segment_reduce_status()
    assert status.attempts == 2 and status.state == 'validated'
    assert len(status.validated) == 2
    assert len(launched) == 4           # two validations (values + probe) each


def test_disagreeing_kernel_disables_the_fast_path_permanently(monkeypatch):
    _force_eligible(monkeypatch)
    launched = _inject_cpu_kernel(monkeypatch, corrupt=True)
    values, index, sizes = _fixture()
    actual = segment_reduce_fast(values, index, sizes)
    assert len(launched) == 1           # rejected on the caller values, no probe
    assert torch.equal(actual, pool(values, index, sizes))
    status = segment_reduce_status()
    assert status.state == 'disabled'
    assert 'disagrees with index_add_' in status.reason
    segment_reduce_fast(values, index, sizes)
    assert len(launched) == 1           # never launched again
    assert segment_reduce_status().state == 'disabled'


def test_raising_kernel_disables_the_fast_path(monkeypatch):
    _force_eligible(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError('no kernel image is available for execution on the device')

    monkeypatch.setattr(fast_kernels, '_grouped_kernel_launch', boom)
    values, index, sizes = _fixture()
    actual = segment_reduce_fast(values, index, sizes)
    assert torch.equal(actual, pool(values, index, sizes))
    status = segment_reduce_status()
    assert status.state == 'disabled'
    assert 'launch failed' in status.reason


def test_guard_gives_up_on_a_non_finite_kernel_result(monkeypatch):
    _force_eligible(monkeypatch)

    def poisoned(grouped, offsets, out, features, count, block_f):
        out.fill_(float('nan'))

    monkeypatch.setattr(fast_kernels, '_grouped_kernel_launch', poisoned)
    values, index, sizes = _fixture()
    actual = segment_reduce_fast(values, index, sizes)
    assert torch.equal(actual, pool(values, index, sizes))
    assert segment_reduce_status().state == 'disabled'


def test_reset_segment_guard_clears_the_verdict(monkeypatch):
    _force_eligible(monkeypatch)
    _inject_cpu_kernel(monkeypatch)
    values, index, sizes = _fixture()
    segment_reduce_fast(values, index, sizes)
    assert segment_reduce_status().active is True
    reset_segment_guard()
    status = segment_reduce_status()
    assert status.state == 'unvalidated' and status.validated == () and status.attempts == 0


def test_invalid_reduce_and_shapes_are_rejected():
    values, index, sizes = _fixture()
    with pytest.raises(ValueError, match='unknown segment reduce'):
        segment_reduce_fast(values, index, sizes, reduce='median')
    with pytest.raises(ValueError, match='2-D'):
        segment_reduce_fast(values[:, 0], index, sizes)
    with pytest.raises(ValueError, match='index the edge dimension'):
        segment_reduce_fast(values, index[:-1], sizes)
    with pytest.raises(ValueError, match='share a device'):
        segment_reduce_fast(values, index, torch.ones(len(sizes), 1, device='meta'))
    with pytest.raises(ValueError, match='denominators'):
        segment_reduce_fast(values, index, sizes.repeat(1, 2))


# ---------------------------------------------------------------------------
# GPU-only: the real kernel, gated and documented rather than guessed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not fast_kernels.triton_available(),
    reason='no CUDA/Triton on this host: the grouped kernel is wired and guarded but '
           'its numeric equivalence to index_add_ can only be validated on a GPU session',
)
def test_real_triton_kernel_matches_index_add_on_a_gpu(monkeypatch):  # pragma: no cover
    monkeypatch.setenv('ER_PERF_ACCEL_SEGMENT_REDUCE', '1')
    generator = torch.Generator().manual_seed(7)
    # 96 features exceeds BLOCK_F (64), so the multi-tile grid path is covered.
    values = torch.randn(4096, 96, generator=generator).cuda()
    index = torch.randint(0, 256, (4096,), generator=generator).cuda()
    sizes = torch.bincount(index, minlength=256).clamp_min(1).unsqueeze(1).float()
    verdict = check_segment_reduce_equivalence(values, index, 256)
    assert verdict.fallback_equal is True
    assert verdict.fast_checked is True and verdict.fast_equal is True
    assert segment_reduce_status().state == 'validated'
    fast = segment_reduce_fast(values, index, sizes)
    reference = pool(values, index, sizes)
    torch.testing.assert_close(fast, reference, rtol=fast_kernels._GUARD_RTOL,
                              atol=fast_kernels._GUARD_ATOL)
