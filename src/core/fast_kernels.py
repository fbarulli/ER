"""Optional accelerator primitives shared by the text and graph trainers.

Every primitive here is strictly opt-in and must stay importable on a CPU-only
host without Triton or a GPU. Availability is *probed*, never assumed, and the
whole module is deactivated by ``ER_PERF_LEGACY=1`` through
:func:`core.perf_switches.perf_enabled`:

============================  =========================================
switch (``perf_enabled``)     gate
============================  =========================================
``accel.compile``             :func:`compile_model`
``accel.segment_reduce``      Triton segment-add kernel
``accel.autocast``            :func:`autocast_context`
``accel.cuda_graph``          :func:`cuda_graph_wrap` (experimental)
============================  =========================================

Nothing in this module imports Triton at module scope unless it is installed;
``torch.compile`` is only ever *called* behind the gate, never at import time.
"""
from __future__ import annotations

import functools
from contextlib import nullcontext
from typing import Any, Callable

import torch

from core.perf_switches import perf_enabled

__all__ = [
    'compile_model',
    'segment_reduce_fast',
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
# Triton segment-add (UNVALIDATED ON HARDWARE -- see note below)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import probe, environment dependent
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton absent on most CI/CPU hosts
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    """Whether a Triton kernel *could* be selected (installed + CUDA)."""
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


if _TRITON_AVAILABLE:

    @triton.jit
    def _segment_add_kernel(values_ptr, index_ptr, out_ptr, n_edges, n_features,
                            BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr):
        """out[index[e]] += values[e] for one (edge-block, feature-block) tile.

        NOTE: this kernel is written but has NOT been validated on real GPU
        hardware (no GPU on this host). It is only ever launched when
        ``accel.segment_reduce`` is on AND CUDA AND Triton are all available,
        and any launch failure silently falls back to the torch path.
        """
        pid_e = tl.program_id(0)
        pid_f = tl.program_id(1)
        offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        edge_mask = offs_e < n_edges
        index = tl.load(index_ptr + offs_e, mask=edge_mask, other=0).to(tl.int64)
        values = tl.load(values_ptr + offs_e[:, None] * n_features + offs_f[None, :],
                         mask=edge_mask[:, None], other=0.0)
        tl.atomic_add(out_ptr + index[:, None] * n_features + offs_f[None, :],
                      values, mask=edge_mask[:, None])


def _triton_segment_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Segment sum through the Triton atomic-add kernel (float32 accumulator)."""
    edges, features = values.shape
    out = torch.zeros((count, features), dtype=torch.float32, device=values.device)
    block_e = 128
    block_f = min(64, triton.next_power_of_2(max(features, 1)))
    grid = (triton.cdiv(edges, block_e), triton.cdiv(features, block_f))
    _segment_add_kernel[grid](
        values.contiguous().float(), index.to(torch.int32).contiguous(), out,
        edges, features, BLOCK_E=block_e, BLOCK_F=block_f,
    )
    return out.to(values.dtype)


def _torch_segment_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Reference/fallback sum, identical to ``graph_tracks.pooling.pool``."""
    total = values.new_zeros((count, values.shape[-1]))
    total.index_add_(0, index, values)
    return total


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

    Selection: the Triton kernel runs only when ``perf_enabled(
    'accel.segment_reduce')`` AND ``values.is_cuda`` AND Triton is importable;
    otherwise (and on any Triton error) it dispatches to ``index_add_``.
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

    if perf_enabled('accel.segment_reduce') and values.is_cuda and _TRITON_AVAILABLE:
        try:
            total = _triton_segment_sum(values, index, count)
            _log_once('segment.triton', 'segment_reduce_fast: using Triton segment kernel')
        except Exception as exc:  # never raise from an accelerator path
            _log_once('segment.triton_failed',
                      f'segment_reduce_fast: Triton failed ({exc!r}); using index_add_')
            total = _torch_segment_sum(values, index, count)
    else:
        total = _torch_segment_sum(values, index, count)

    if reduce == 'sum':
        return total
    denominator = sizes.to(dtype=values.dtype).reshape(count, 1)
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
    ``accel.cuda_graph``; on CPU, without CUDA, or when the switch is off this
    returns ``fn`` unchanged (a genuine no-op).

    Positional tensor arguments are copied into static capture buffers each
    replay. Keyword arguments, changing shapes/dtypes, or non-tensor arguments
    fall back to an eager ``fn`` call rather than raising. Returned tensors
    alias the capture buffers -- treat them read-only. This helper has NOT been
    exercised on GPU hardware on this host.
    """
    if not (perf_enabled('accel.cuda_graph') and torch.cuda.is_available()):
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
