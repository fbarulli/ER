"""Training-loop speed classes for the laya perf patch (injected VERBATIM).

Same injection contract as ``core.laya_controls`` / ``core.laya_eval_fast``:
cohesive classes, no module-level ``torch`` import (the loop passes its local
``torch`` in, so these are unit-testable on the host and exec'd byte-for-byte
into the staged kernels, which cannot import the repo). Listed in
``cli.laya_train_patch_ddp.FINETUNE_CONTROL_LOGIC_SOURCE``.

Why this module exists: the stopped HPO run's T4 profile is CPU-DISPATCH-BOUND
— Self CPU 17.2 s vs Self CUDA 4.4 s per 3 optimizer steps, GPU only ~40% busy.
The forward is a dispatch storm (6528 ``linear``/``mm`` calls/step at
micro_batch 4) and the autocast path a cast/copy storm (46,476 ``_to_copy`` +
54,828 ``copy_`` per 3 steps). These classes fuse/overlap that work; the
eval-path siblings live in ``core.laya_eval_fast``.
"""
from __future__ import annotations

from typing import Any


class ModelCompiler:
    """``torch.compile`` the training module (offenders #2 dispatch + #4 cast).

    Fuses the per-op Python dispatch and the autocast weight recasts into fewer,
    larger Inductor kernels. CPU-proxy measured on the real laya checkpoint
    (mb=4): calls/step 19,395 -> 8,542 (-56%), ``aten::linear``/``matmul`` fused
    to 0, ``_to_copy`` 34 -> 4 (-88%), ``copy_`` 442 -> 112 (-75%). On the
    dispatch-bound T4 that removes the Self-CPU ceiling the 40%-GPU profile hit.

    ``mode`` is the control dial (``FinetuneSpec.compile_mode``): ``default`` is
    the safe fusion; ``reduce-overhead``/``max-autotune`` trade compile time for
    GPU throughput (their CUDA-graph path only fully engages on STATIC shapes —
    laya's collate pads to the batch-longest, so with ``dynamic=True`` they
    still fuse but may not capture graphs; HPO measures the net, never assumed).

    ``dynamic=True`` handles variable lengths; Laya's ``independent_dims``
    also prevents coincidentally equal row/token/marker dimensions from being
    tied together at trace time. The context covers lazy forward execution,
    not just construction. Torch can still specialize zero/one dimensions.

    Applied AFTER the DDP wrap so ``torch.compile`` sees the
    ``DistributedDataParallel`` module and its DDPOptimizer splits the graph at
    the allreduce buckets (the documented DDP+compile order). The compile
    wrapper (``_orig_mod``) is stripped by ``ControlCheckpointer.unwrap`` before
    any ``state_dict`` is persisted. Construction failures log a full traceback
    and return the eager module. Lazy compilation failures still fail the trial.
    """

    #: Pinned: variable per-batch sequence length => one dynamic compile.
    DYNAMIC = True

    @staticmethod
    def compile(torch_module: "Any", model: "Any", mode: str | None) -> "Any":  # noqa: UP037 -- injected without module imports
        resolved = str(mode or "default")
        try:
            compiled = torch_module.compile(
                model, mode=resolved, dynamic=ModelCompiler.DYNAMIC)
            print(f"[perf-patch] torch.compile enabled mode={resolved} "
                  f"dynamic={ModelCompiler.DYNAMIC}", flush=True)
        except Exception:  # noqa: BLE001 - fail-soft to eager, recorded
            import traceback

            print("[perf-patch] torch.compile skipped (eager fallback): "
                  + traceback.format_exc(), flush=True)
            return model
        try:
            from laya._compile import independent_dims
        except ImportError:
            import traceback

            print("[perf-patch] independent shape tracing unavailable; "
                  "keeping ordinary dynamic compilation: "
                  + traceback.format_exc(), flush=True)
        else:
            compiled.forward = independent_dims()(compiled.forward)
        return compiled


class BatchMover:
    """ONE pinned, ``non_blocking`` host->device move of a collated batch.

    Offender #6: the loop moved mask/target/qtype with blocking ``.to(device)``
    on PAGEABLE memory, and ``_forward`` re-moved input_ids/attention_mask the
    same way. A pageable H2D copy serialises the host against the stream — the
    smoke harvest attributed **326 s/epoch** to ``collate+batch_move`` at
    micro_batch 1 (each of ~3240 small transfers waited on the prior step's GPU
    work, starving the card to ~40% busy). Pinning the batch and copying with
    ``non_blocking=True`` lets the transfer overlap the compute stream (the
    DataLoader idiom; torch's caching pinned allocator keeps the source alive
    until the async copy lands, and the same-stream forward sees the data in
    order). Numerically identical — same bytes, same dtype, only the transfer
    timing moves.

    The WHOLE batch moves once here, so the loop's mask/target/qtype reads and
    ``_forward``'s own ``.to(device)`` are then no-ops (never a second copy).

    Fail-soft: a ``pin_memory`` failure (no CUDA pinned pool, host pinned-OOM
    under the 3-parallel-trial budget) falls back to a blocking move for that
    tensor and records the reason ONCE — never a silent recipe change, never a
    crash. On a non-CUDA device (or ``enabled=False``) it is a plain move.
    """

    #: Warn-once latch so a pinned-pool OOM logs one traceback, not one/batch.
    _warned = False

    @staticmethod
    def to_device(torch_module, batch, device, enabled):
        use_pin = (bool(enabled) and getattr(device, "type", "") == "cuda"
                   and torch_module.cuda.is_available())
        moved = {}
        for key, value in batch.items():
            if not torch_module.is_tensor(value):
                moved[key] = value
            elif not use_pin:
                moved[key] = value.to(device)
            else:
                moved[key] = BatchMover._pin_move(torch_module, value, device)
        return moved

    @staticmethod
    def _pin_move(torch_module, value, device):
        try:
            source = value if value.is_pinned() else value.pin_memory()
            return source.to(device, non_blocking=True)
        except Exception:  # noqa: BLE001 - fail-soft to a blocking copy
            if not BatchMover._warned:
                import traceback

                print("[perf-patch] pin_memory unavailable; blocking H2D "
                      "fallback: " + traceback.format_exc(), flush=True)
                BatchMover._warned = True
            return value.to(device)

