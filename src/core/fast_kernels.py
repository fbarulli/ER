"""Optional accelerator primitives shared by the text and graph trainers.

Every primitive here is strictly opt-in and must stay importable on a CPU-only
host without Triton or a GPU. Availability is *probed*, never assumed, and the
whole module is deactivated by ``ER_PERF_LEGACY=1`` through
:func:`core.perf_switches.perf_enabled`:

============================  ===================  ========================
switch (``perf_enabled``)     default when unset   gate
============================  ===================  ========================
``accel.compile``             on                   :func:`compile_model`
``accel.segment_reduce``      **off**              Triton segment-add kernel
``accel.cuda_graph``          **off**              :func:`cuda_graph_wrap`
``accel.autocast``            on                   :func:`autocast_context`
============================  ===================  ========================

The two primitives that no host here could validate numerically (the Triton
segment kernel and the experimental CUDA-graph wrapper) default to **off**:
they only run when ``ER_PERF_ACCEL_SEGMENT_REDUCE=1`` /
``ER_PERF_ACCEL_CUDA_GRAPH=1`` is set explicitly on a host that can run them.

Nothing in this module imports Triton at module scope unless it is installed;
``torch.compile`` is only ever *called* behind the gate, never at import time.

The Triton reduction additionally carries a runtime guard:

  * *capability* -- opt-in AND CUDA AND Triton importable AND floating dtype
    AND ``not values.requires_grad`` (a raw kernel write cannot build an autograd
    edge, so a grad-enabled call would silently detach -- it always falls back);
  * *correctness* -- the first eligible call for a given
    ``(device, dtype, feature-width, edge-count)`` signature is compared against
    the ``index_add_`` fallback, on the caller's own values **and** on a
    deterministic synthetic probe; any mismatch (or any launch error) disables
    the fast path for the rest of the process and logs once.

:func:`check_segment_reduce_equivalence` is the CPU-testable harness for that
reduction: it always compares the production fallback against
:func:`reference_segment_sum` (an obviously-correct explicit loop) and reports
the Triton comparison with an explicit skip reason on hosts without CUDA.
"""
from __future__ import annotations

import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch

from core.perf_switches import perf_enabled

__all__ = [
    'compile_model',
    'segment_reduce_fast',
    'segment_reduce_capability',
    'segment_reduce_status',
    'check_segment_reduce_equivalence',
    'reference_segment_sum',
    'reset_segment_guard',
    'autocast_context',
    'cuda_graph_wrap',
    'triton_available',
]

# ---------------------------------------------------------------------------
# logging (RunLogger only; no ad-hoc prints in src/)
# ---------------------------------------------------------------------------
_LOGGER = None
_LOGGED: set[str] = set()


def _log_once(key: str, message: str) -> None:
    """Emit ``message`` at most once per process for ``key``."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    global _LOGGER
    if _LOGGER is None:
        from core.run_log import RunLogger
        _LOGGER = RunLogger('core.fast_kernels')
    _LOGGER.info('[accel] ' + message)


def _torch_major() -> int:
    try:
        return int(str(torch.__version__).split('.', 1)[0])
    except (ValueError, AttributeError):
        return 0


def _same_callable(left, right) -> bool:
    """Identity, or equality for bound methods handed out fresh each access."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------
_COMPILE_CACHE: dict[tuple[str, str, bool], tuple[Any, Any]] = {}


def compile_model(module, *, name: str, mode: str = 'reduce-overhead',
                  fullgraph: bool = False):
    """Wrap ``module`` with ``torch.compile`` behind ``accel.compile``.

    Intended call sites (both trainers already do the equivalent inline):

      * ``graph_tracks/model.py`` / ``graph_tracks/benchmark.py``: compile the
        ``AttributeGNN.context`` / ``encode`` bound methods, e.g.
        ``model.context = compile_model(model.context, name='gnn.context')``.
      * ``model_tracks`` training loop: compile the text encoder forward.

    A compiled handle is cached per ``(name, mode, fullgraph)`` so repeated
    calls return the same object; if the underlying module changes, it is
    recompiled. Any unsupported platform (torch < 2, missing ``torch.compile``,
    Inductor failure) logs once and returns the **original** module, so callers
    never need a try/except and can always call the result.
    """
    if not perf_enabled('accel.compile'):
        return module
    if _torch_major() < 2 or not callable(getattr(torch, 'compile', None)):
        _log_once('compile.unsupported',
                  f'compile_model({name}): torch.compile unavailable; using eager')
        return module
    key = (name, mode, fullgraph)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None and _same_callable(cached[0], module):
        return cached[1]
    try:
        compiled = torch.compile(module, mode=mode, fullgraph=fullgraph)
    except Exception as exc:  # never raise on unsupported platforms
        _log_once(f'compile.failed.{name}',
                  f'compile_model({name}): torch.compile failed ({exc!r}); using eager')
        _COMPILE_CACHE[key] = (module, module)
        return module
    _COMPILE_CACHE[key] = (module, compiled)
    _log_once(f'compile.ok.{name}',
              f'compile_model({name}): compiled mode={mode} fullgraph={fullgraph}')
    return compiled


# ---------------------------------------------------------------------------
# Triton grouped segment-add (runtime-guarded, opt-in, UNVALIDATED ON THIS HOST)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import probe, environment dependent
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton absent on most CI/CPU hosts
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# Declared equivalence budget of the runtime guard. The kernel accumulates in
# float32 and casts back, so a grouping that differs from ``index_add_``'s
# accumulation order may drift by a few ULPs; a difference beyond this budget is
# a kernel bug, not rounding, and disables the fast path for the process.
_GUARD_RTOL = 1e-4
_GUARD_ATOL = 1e-5
# Deterministic probe seed: the guard re-runs the reduction on a fixed synthetic
# tensor so a grouping bug cannot hide behind degenerate (e.g. all-zero) inputs.
_PROBE_SEED = 20261008


def triton_available() -> bool:
    """Whether a Triton kernel *could* be selected (installed + CUDA)."""
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


if _TRITON_AVAILABLE:

    @triton.jit
    def _segment_add_kernel(values_ptr, offsets_ptr, out_ptr, n_features,
                            BLOCK_F: tl.constexpr):
        """out[s] = sum(values[offsets[s]:offsets[s + 1]]) for one segment tile.

        Grouped/segmented reduction: the caller sorts edges by target and
        passes the ``(count + 1,)`` segment offsets, so every program owns one
        complete output row and reads its edges contiguously. No ``atomic_add``
        is issued, which removes the contention the previous scatter kernel hit
        whenever many edges share a target.

        NOTE: written but NOT validated on real GPU hardware (no GPU on this
        host). It is launched only when ``accel.segment_reduce`` is explicitly
        opted in AND CUDA AND Triton are available AND the runtime guard below
        has accepted it; the arithmetic (float32 accumulator, cast back to the
        caller dtype) must stay equivalent to :func:`_torch_segment_sum`.
        """
        segment = tl.program_id(0)
        feature_block = tl.program_id(1)
        offs_f = feature_block * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < n_features
        start = tl.load(offsets_ptr + segment)
        end = tl.load(offsets_ptr + segment + 1)
        accumulator = tl.zeros((BLOCK_F,), dtype=tl.float32)
        for edge in range(start, end):
            row = tl.load(values_ptr + edge * n_features + offs_f,
                          mask=f_mask, other=0.0)
            accumulator += row
        tl.store(out_ptr + segment * n_features + offs_f, accumulator, mask=f_mask)


# The single launch indirection: the guard and the CPU test harness both drive
# this, so the kernel itself never has to be importable to exercise the logic.
_TRITON_LAUNCH = _segment_add_kernel if _TRITON_AVAILABLE else None


def _grouped_kernel_launch(grouped: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor,
                           features: int, count: int, block_f: int) -> None:
    """Launch the grouped Triton kernel (one program per (segment, feature tile))."""
    grid = (count, triton.cdiv(features, block_f))
    _TRITON_LAUNCH[grid](grouped, offsets, out, features, BLOCK_F=block_f)


def _triton_segment_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Grouped segment sum through the Triton kernel (float32 accumulator).

    Edges are stably sorted by target once; the kernel then reduces each
    contiguous group without atomics. ``values`` is only cast when it is not
    already float32, and ``index`` is never copied.

    NOTE: the kernel writes into a fresh ``out`` buffer, so the returned tensor
    carries no autograd edge. Callers must keep grad-enabled tensors on the
    ``index_add_`` fallback; :func:`_segment_fast_eligibility` enforces that.
    """
    edges, features = values.shape
    out = torch.zeros((count, features), dtype=torch.float32, device=values.device)
    if count == 0 or edges == 0:
        return out.to(values.dtype)
    order = torch.argsort(index, stable=True)
    grouped = values[order]
    if grouped.dtype != torch.float32:
        grouped = grouped.float()
    lengths = torch.bincount(index, minlength=count)
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    block_f = min(64, triton.next_power_of_2(max(features, 1)))
    _grouped_kernel_launch(grouped, offsets, out, features, count, block_f)
    return out.to(values.dtype)


def _torch_segment_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Reference/fallback sum, identical to ``graph_tracks.pooling.pool``."""
    total = values.new_zeros((count, values.shape[-1]))
    total.index_add_(0, index, values)
    return total


def reference_segment_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Obviously-correct reference: one explicit accumulation per edge.

    ``O(E * F)`` in Python, but it is the kernel of truth for the equivalence
    harness: no scatter, no sort, no atomics, no vectorisation, no dtype change.
    """
    out = values.new_zeros((count, values.shape[-1]))
    for row, target in enumerate(index.tolist()):
        out[int(target)] += values[row]
    return out


# ---------------------------------------------------------------------------
# runtime guard: capability + one-time correctness validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentFastPathStatus:
    """Outcome of the runtime guard for the Triton segment reduction."""
    state: str                      # 'unvalidated' | 'validated' | 'disabled'
    reason: str
    validated: tuple[str, ...] = ()  # device|dtype|features|edges signatures accepted
    attempts: int = 0                # signatures actually launched + validated

    @property
    def active(self) -> bool:
        return self.state == 'validated'


_GUARD = {'state': 'unvalidated', 'reason': 'no eligible call yet',
          'validated': (), 'attempts': 0}


def reset_segment_guard() -> None:
    """Forget the guard outcome (a new run, or a test injecting a fake kernel)."""
    _GUARD.update(state='unvalidated', reason='no eligible call yet',
                  validated=(), attempts=0)


def segment_reduce_status() -> SegmentFastPathStatus:
    """The guard's current verdict, including the reason for a fallback."""
    return SegmentFastPathStatus(state=_GUARD['state'], reason=_GUARD['reason'],
                                 validated=tuple(_GUARD['validated']),
                                 attempts=int(_GUARD['attempts']))


def segment_reduce_capability() -> dict[str, Any]:
    """Whether this host could run the Triton reduction, and what is missing."""
    opt_in = perf_enabled('accel.segment_reduce', default=False)
    cuda = bool(torch.cuda.is_available())
    missing = []
    if not opt_in:
        missing.append('opt-in: set ER_PERF_ACCEL_SEGMENT_REDUCE=1')
    if not _TRITON_AVAILABLE:
        missing.append('triton is not importable')
    if not cuda:
        missing.append('no CUDA device on this host')
    device = None
    if cuda:
        try:
            device = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver quirk on a CUDA host
            device = 'cuda'
    return {
        'opt_in': opt_in,
        'triton': _TRITON_AVAILABLE,
        'triton_version': getattr(triton, '__version__', None) if _TRITON_AVAILABLE else None,
        'cuda': cuda,
        'device': device,
        'eligible': opt_in and _TRITON_AVAILABLE and cuda,
        'reason': '; '.join(missing) if missing else 'eligible on this host',
    }


def _segment_fast_eligibility(values: torch.Tensor) -> tuple[bool, str]:
    """Capability guard: everything that must hold before a launch is attempted."""
    if not perf_enabled('accel.segment_reduce', default=False):
        return False, ('accel.segment_reduce is opt-in and off '
                       '(set ER_PERF_ACCEL_SEGMENT_REDUCE=1 to enable)')
    if not _TRITON_AVAILABLE:
        return False, 'triton is not importable'
    if not values.is_cuda:
        return False, 'values are not on CUDA'
    if not values.is_floating_point():
        return False, f'values are {values.dtype}, not floating point'
    if values.requires_grad:
        return False, ('values require grad: index_add_ keeps the autograd edge, '
                       'the raw kernel write would detach the graph')
    return True, 'eligible'


def _guard_signature(values: torch.Tensor) -> tuple[str, str, int, int]:
    """Everything the kernel's correctness depends on and that can change.

    The edge count is part of the key: a grouping/offset bug that only shows up
    at another population size must be re-validated, not inherited from a
    smaller node.
    """
    return (str(values.device), str(values.dtype), int(values.shape[-1]), int(values.shape[0]))


def _synthetic_probe(values: torch.Tensor) -> torch.Tensor:
    """Deterministic same-shape probe tensor, so a grouping bug cannot hide."""
    generator = torch.Generator(device=values.device).manual_seed(_PROBE_SEED)
    return torch.rand(values.shape, generator=generator, dtype=values.dtype,
                      device=values.device)


def _close_enough(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        return False
    if not bool(torch.isfinite(candidate).all()):
        return False
    # A non-finite reference value is never "close": fail safe onto index_add_.
    if not bool(torch.isfinite(reference).all()):
        return False
    return bool(torch.allclose(candidate, reference, rtol=_GUARD_RTOL, atol=_GUARD_ATOL))


def _maximum_absolute_difference(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        return float('inf')
    return float((candidate - reference).abs().max())


def _validate_kernel_result(values: torch.Tensor, index: torch.Tensor, count: int,
                            candidate: torch.Tensor) -> tuple[bool, str]:
    """Compare a fast-path result against ``index_add_`` on real and probe input."""
    reference = _torch_segment_sum(values, index, count)
    if not _close_enough(candidate, reference):
        return False, ('kernel result disagrees with index_add_ on the caller values '
                       f'(rtol={_GUARD_RTOL}, atol={_GUARD_ATOL})')
    probe = _synthetic_probe(values)
    probe_reference = _torch_segment_sum(probe, index, count)
    probe_candidate = _triton_segment_sum(probe, index, count)
    if not _close_enough(probe_candidate, probe_reference):
        return False, ('kernel result disagrees with index_add_ on the deterministic '
                       f'probe (rtol={_GUARD_RTOL}, atol={_GUARD_ATOL})')
    return True, f'validated against index_add_ on caller values and probe (rtol={_GUARD_RTOL})'


def _disable_segment_fast_path(reason: str) -> None:
    _GUARD.update(state='disabled', reason=reason)
    _log_once('segment.triton_disabled',
              f'segment_reduce_fast: fast path disabled ({reason}); using index_add_ from now on')


def _fast_segment_sum(values: torch.Tensor, index: torch.Tensor,
                      count: int) -> torch.Tensor | None:
    """Guarded launch. ``None`` means: use the ``index_add_`` fallback."""
    # The disable latch is consulted before any capability probe, so a rejected
    # kernel can never be launched again -- not even by a caller that overrides
    # the capability check.
    if _GUARD['state'] == 'disabled':
        return None
    eligible, reason = _segment_fast_eligibility(values)
    if not eligible:
        if _GUARD['state'] == 'unvalidated':
            _GUARD['reason'] = reason
        return None
    label = '|'.join(map(str, _guard_signature(values)))
    known = label in _GUARD['validated']
    if not known:
        _GUARD['attempts'] = int(_GUARD['attempts']) + 1
    try:
        candidate = _triton_segment_sum(values, index, count)
        if known:
            return candidate
        accepted, why = _validate_kernel_result(values, index, count, candidate)
    except Exception as exc:  # never raise from an accelerator path
        _disable_segment_fast_path(f'triton launch failed ({exc!r})')
        return None
    if not accepted:
        _disable_segment_fast_path(why)
        return None
    _GUARD.update(state='validated', reason=why, validated=tuple(_GUARD['validated']) + (label,))
    _log_once('segment.triton.validated', f'segment_reduce_fast: Triton kernel {why}')
    return candidate


@dataclass(frozen=True)
class SegmentEquivalence:
    """Result of :func:`check_segment_reduce_equivalence`."""
    fallback_equal: bool
    fallback_max_abs_diff: float
    fast_checked: bool
    fast_equal: bool | None
    fast_max_abs_diff: float | None
    reason: str


def check_segment_reduce_equivalence(values: torch.Tensor, index: torch.Tensor,
                                     count: int | None = None, *,
                                     trial_fn: Callable | None = None) -> SegmentEquivalence:
    """CPU-testable equivalence harness for the segment reduction.

    Always compares the production fallback (``index_add_``) against
    :func:`reference_segment_sum`. Additionally compares a fast-path trial
    (``trial_fn``, or the guarded Triton path when this host is eligible)
    against that same reference, so the harness yields a positive result on a
    CPU-only host and an explicit skip reason for the unvalidated kernel.

    A rejected trial is reported, not raised; when the trial is the guarded
    Triton path it also latches the guard off (that is the fallback behaviour
    under test).
    """
    if values.ndim != 2 or index.ndim != 1 or values.shape[0] != index.shape[0]:
        raise ValueError('equivalence harness needs (E, F) values indexed by (E,) rows')
    if count is None:
        count = int(index.max()) + 1 if index.numel() else 0
    reference = reference_segment_sum(values, index, count)
    fallback = _torch_segment_sum(values, index, count)
    fallback_diff = _maximum_absolute_difference(fallback, reference)
    verdict = SegmentEquivalence(
        fallback_equal=_close_enough(fallback, reference),
        fallback_max_abs_diff=fallback_diff,
        fast_checked=False, fast_equal=None, fast_max_abs_diff=None,
        reason='fallback compared against the explicit reference',
    )
    if trial_fn is None:
        if _GUARD['state'] == 'disabled':
            return SegmentEquivalence(**{**verdict.__dict__, 'reason':
                                         f'kernel not checked: fast path disabled ({_GUARD["reason"]})'})
        eligible, why = _segment_fast_eligibility(values)
        if not eligible:
            return SegmentEquivalence(**{**verdict.__dict__,
                                         'reason': f'kernel not checked: {why}'})
        trial_fn = _triton_segment_sum
    try:
        trial = trial_fn(values, index, count)
    except Exception as exc:
        return SegmentEquivalence(**{**verdict.__dict__,
                                     'reason': f'kernel trial raised: {exc!r}'})
    diff = _maximum_absolute_difference(trial, reference)
    return SegmentEquivalence(**{**verdict.__dict__, 'fast_checked': True,
                                 'fast_equal': _close_enough(trial, reference),
                                 'fast_max_abs_diff': diff,
                                 'reason': f'kernel trial compared (rtol={_GUARD_RTOL})'})


def segment_reduce_fast(values: torch.Tensor, index: torch.Tensor, sizes: torch.Tensor,
                        *, reduce: str = 'mean') -> torch.Tensor:
    """Fast typed-relation pooling with ``pool``/``segment_pool`` semantics.

    Intended call site: ``graph_tracks/model.py`` ``AttributeGNN.pool``
    (``model.py:46``), as a drop-in for both the ``index_add`` and ``segment``
    aggregation backends::

        # model.py:47-49
        from core.fast_kernels import segment_reduce_fast
        return segment_reduce_fast(values, target, sizes, reduce='mean')

    Arguments match that call site exactly:

      * ``values``  -- ``(E, F)`` edge messages (float);
      * ``index``   -- ``(E,)`` int64 target rows in ``[0, count)``;
      * ``sizes``   -- ``(count,)`` or ``(count, 1)`` denominators.

    ``reduce`` is ``"mean"`` (segment sum / ``sizes``) or ``"sum"``.

    Correctness note -- empty segments
    ----------------------------------
    The existing pipeline builds ``sizes`` in ``pooling.topology`` and clamps
    it with ``clamp_min(1)`` before it reaches ``pool``/``segment_pool``, so a
    relation row with zero incoming edges yields ``0 / 1 = 0``. This function
    divides by ``sizes`` *exactly as supplied* (no hidden clamp), so passing
    ``topology()``'s denominators reproduces that empty-segment behaviour
    bit-for-bit. A caller that supplies raw zero denominators gets the same
    ``0/0`` NaNs the existing ``pool`` would produce -- the fallback is a
    literal ``index_add_`` sum divided by ``sizes``.

    Selection: the Triton kernel runs only when ``accel.segment_reduce`` is
    explicitly opted in (``ER_PERF_ACCEL_SEGMENT_REDUCE=1``; the switch defaults
    to **off**) AND ``values.is_cuda`` AND Triton is importable AND the values do
    not require grad AND the one-time correctness guard has accepted this
    ``(device, dtype, feature-width, edge-count)`` signature. Otherwise -- and
    after any launch error or numeric disagreement -- it dispatches to
    ``index_add_``. :func:`segment_reduce_status` reports the guard's verdict.
    """
    if reduce not in {'mean', 'sum'}:
        raise ValueError(f'unknown segment reduce {reduce!r}')
    if values.ndim != 2:
        raise ValueError('segment values must be 2-D (edges, features)')
    if index.ndim != 1 or values.shape[0] != index.shape[0]:
        raise ValueError('segment index must index the edge dimension')
    if index.device != values.device or sizes.device != values.device:
        raise ValueError('segment values, index and denominators must share a device')
    count = sizes.shape[0]
    if sizes.ndim not in (1, 2) or (sizes.ndim == 2 and sizes.shape[1] != 1):
        raise ValueError('segment denominators must be (count,) or (count, 1)')

    total = _fast_segment_sum(values, index, count)
    if total is None:
        total = _torch_segment_sum(values, index, count)
    else:
        _log_once('segment.triton', 'segment_reduce_fast: using the validated Triton segment kernel')

    if reduce == 'sum':
        return total
    # Divide by the denominators exactly as supplied; only cast when the dtype
    # actually differs so the common float32 case copies nothing.
    denominator = sizes.reshape(count, 1)
    if denominator.dtype != total.dtype:
        denominator = denominator.to(total.dtype)
    return total / denominator


# ---------------------------------------------------------------------------
# autocast
# ---------------------------------------------------------------------------
def autocast_context(device):
    """Autocast context for ``device`` behind ``accel.autocast``.

    Intended call sites: the training step bodies in
    ``graph_tracks/benchmark.py`` (``with torch.autocast('cuda', ...)``) and the
    text-training loop, e.g. ``with autocast_context(device): ...``.

    Returns bf16 on a bf16-capable CUDA device, else fp16 on CUDA, and a
    ``nullcontext`` on CPU or when the switch is off. Never raises.
    """
    if not perf_enabled('accel.autocast'):
        return nullcontext()
    try:
        if torch.device(device).type != 'cuda':
            return nullcontext()
        if not torch.cuda.is_available():
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _log_once(f'autocast.{torch.device(device).type}', f'autocast_context: {dtype}')
        return torch.autocast(device_type='cuda', dtype=dtype, enabled=True)
    except Exception as exc:
        _log_once('autocast.failed', f'autocast_context: falling back to no autocast ({exc!r})')
        return nullcontext()


# ---------------------------------------------------------------------------
# CUDA graphs (EXPERIMENTAL)
# ---------------------------------------------------------------------------
def cuda_graph_wrap(fn: Callable) -> Callable:
    """EXPERIMENTAL static-shape CUDA-graph capture/replay of ``fn``.

    Intended call site: wrapping one pure, shape-static training step (for
    example the closure built in ``graph_tracks/benchmark.py``). Gated by
    ``accel.cuda_graph``, which defaults to **off** (this helper has not been
    exercised on GPU hardware on this host): on CPU, without CUDA, or when the
    switch is off this returns ``fn`` unchanged (a genuine no-op).

    Positional tensor arguments are copied into static capture buffers each
    replay. Keyword arguments, changing shapes/dtypes, or non-tensor arguments
    fall back to an eager ``fn`` call rather than raising. Returned tensors
    alias the capture buffers -- treat them read-only. This helper has NOT been
    exercised on GPU hardware on this host.
    """
    if not (perf_enabled('accel.cuda_graph', default=False) and torch.cuda.is_available()):
        return fn

    state: dict[str, Any] = {'static': None, 'graph': None, 'outputs': None}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs or not args or not all(torch.is_tensor(a) for a in args):
            return fn(*args, **kwargs)
        try:
            if state['graph'] is None:
                static = tuple(a.detach().clone() for a in args)
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        fn(*static)
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = fn(*static)
                state.update(static=static, graph=graph, outputs=outputs)
            static = state['static']
            for destination, source in zip(static, args):
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    return fn(*args, **kwargs)
                destination.copy_(source)
            state['graph'].replay()
            return state['outputs']
        except Exception as exc:  # pragma: no cover - GPU only
            _log_once('cuda_graph.failed',
                      f'cuda_graph_wrap: capture/replay failed ({exc!r}); using eager')
            state.update(static=None, graph=None, outputs=None)
            return fn(*args, **kwargs)

    return wrapped
