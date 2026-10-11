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

    ``dynamic=True`` is pinned (not a knob): the sequence length varies per
    batch, so a static compile would thrash recompile-per-shape. One dynamic
    compile serves every batch length.

    Applied AFTER the DDP wrap so ``torch.compile`` sees the
    ``DistributedDataParallel`` module and its DDPOptimizer splits the graph at
    the allreduce buckets (the documented DDP+compile order). The compile
    wrapper (``_orig_mod``) is stripped by ``ControlCheckpointer.unwrap`` before
    any ``state_dict`` is persisted. Fail-soft: a compile error (backend miss,
    unsupported op) logs the FULL traceback and returns the eager module, so a
    run never silently downshifts the recipe NOR dies on a compiler bug.
    """

    #: Pinned: variable per-batch sequence length => one dynamic compile.
    DYNAMIC = True

    @staticmethod
    def compile(torch_module, model, mode):
        resolved = str(mode or "default")
        try:
            compiled = torch_module.compile(
                model, mode=resolved, dynamic=ModelCompiler.DYNAMIC)
            print("[perf-patch] torch.compile enabled mode=%s dynamic=%s"
                  % (resolved, ModelCompiler.DYNAMIC), flush=True)
            return compiled
        except Exception:  # noqa: BLE001 - fail-soft to eager, recorded
            import traceback

            print("[perf-patch] torch.compile skipped (eager fallback): "
                  + traceback.format_exc(), flush=True)
            return model
