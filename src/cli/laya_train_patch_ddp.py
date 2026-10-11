"""DDP + control-logic half of the fine-tune PERF_PATCH kernel text.

The fine-tune kernel's `@PERF_PATCH@` injection is one template split at the
`class ControlCheckpointer` boundary so each half stays small: this module owns
the control-logic injection and the distributed/data-parallel helpers, the
sibling `laya_train_patch_loop` owns the training loop.
"""
from __future__ import annotations

import inspect

from core import laya_controls, laya_eval_fast, laya_train_fast, laya_train_parts

# Movement/redundancy PERF_PATCH for the finetune kernel (laya>=0.3.29).
# Replaces laya.train.train_model with a faithful copy carrying exactly three
# recipe-neutral changes:
#   (1) the running loss is accumulated as a 0-dim CUDA tensor and synced with
#       ONE .item() per epoch (the stock loop synced twice per micro-step,
#       train.py:696 and :698);
#   (2) the batch tensors the loop consumes are moved to the device ONCE (the
#       stock loop re-moved marker_mask and qtype after _forward had already
#       moved them, train.py:608 vs :671);
#   (3) encode_item is memoized per (item id, option order) so steady-state
#       epochs skip re-tokenizing (train.py:665). draw_option_order is still
#       called in the same per-step order, so the RNG stream is identical.
# The SINGLE-PROCESS path is byte-for-byte the stock loop (ITEM order included);
# when WORLD_SIZE>1 the same loop additionally DDP-wraps the model and shards
# the items via `build_distributed_sampler` (see the DDP helpers below). The
# recipe flags, grad-accum window, clipping, scheduler and seed paths are
# unchanged. Opt out (patch AND sampler) with ER_LAYA_PERF_PATCH=0; force the
# single-process fallback with ER_LAYA_DDP=0. Injected at `@PERF_PATCH@`.
#
# The pure control logic lives in `core.laya_controls` as COHESIVE CLASSES and
# is injected here VERBATIM via `inspect.getsource`: one source of truth,
# unit-tested on the host and executed byte-for-byte in the kernel (which
# cannot import the repo). Injection order is the class dependency order.
FINETUNE_CONTROL_LOGIC_SOURCE = "\n\n".join(
    inspect.getsource(klass) for klass in (
        laya_controls.ControlBlock,
        laya_controls.EarlyStopStep,
        laya_controls.EarlyStopPolicy,
        laya_controls.LrSchedulerFactory,
        laya_controls.MetricFlattener,
        laya_controls.AbstainCoverage,
        laya_controls.DevReport,
        laya_controls.TrainingControls,
        laya_controls.ParamGroupBuilder,
        laya_controls.OptimizerStateCaster,
        laya_controls.LrScaler,
        laya_controls.BatchRamp,
        laya_controls.LossWeightSchedule,
        laya_controls.RDrop,
        laya_controls.DropPath,
        laya_controls.DynamicPadder,
        laya_controls.TqdmLaneWriter,
        laya_controls.LaneProgress,
        laya_controls.PhaseTimings,
        laya_controls.CalibrationTimingHook,
        laya_controls.ProfilerSession,
        laya_train_parts.AdversarialPerturber,
        laya_train_parts.TrainingOptimizer,
        laya_train_parts.LossBuilder,
        laya_train_parts.Forwarder,
        laya_train_fast.ModelCompiler,
        laya_train_fast.BatchMover,
        laya_eval_fast.EvalRecordsPolicy,
        laya_eval_fast.EvalRecordsHook,
        laya_eval_fast.EncodeMemo,
        laya_eval_fast.CollateMemo,
        laya_eval_fast.DevEvaluator,
    )
)


DDP_PATCH_TEMPLATE = '''\
@CONTROL_LOGIC@

# The kernel template defines the wandb helpers BEFORE this patch is injected;
# the offline DDP unit test execs this source alone, so fall back to no-ops.
try:
    wandb_log_epoch
except NameError:
    WANDB_RUN = None

    class _OfflineLogSink:
        """No-op log sink so the offline PERF-patch exec (unit tests) runs
        without the kernel's shared wandb block."""

        @staticmethod
        def add(line):
            return None

        @staticmethod
        def flush():
            return None

    LOG_SINK = _OfflineLogSink()

    def wandb_log_epoch(epoch, mean, extra=None):
        return None

    def wandb_log_control_summary(result):
        return None

    def wandb_log_event(name, **fields):
        return None

PERF_PATCH_ENV = "ER_LAYA_PERF_PATCH"


def perf_patch_enabled():
    # one env flag disables BOTH the movement patch and the GPU sampler.
    return os.environ.get(PERF_PATCH_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


# Distributed-data-parallel wiring (2xT4). `finetune` is a black box that
# calls train_model once per rank, so the wrap + the per-rank shard live IN
# the patched loop (DDP averages the gradients for us). One env flag forces
# the single-process fallback: ER_LAYA_DDP=0.
DDP_ENV = "ER_LAYA_DDP"

# Post-destroy exit-rendezvous bound (seconds): how long a non-rank0 process
# keeps its CUDA context alive waiting for rank 0's destroy to land (see
# destroy_if_distributed). Bounded + fail-soft: on timeout the reason is
# recorded and the worker exits anyway.
DDP_TEARDOWN_RENDEZVOUS_SECONDS = 300.0


def ddp_enabled():
    return os.environ.get(DDP_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def dist_env():
    # LOCAL_RANK drives the per-rank CUDA device; RANK the global rank; the
    # spawn/torchrun launcher sets both (world_size <= 1 => single process).
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    return local_rank, rank, world_size


def is_distributed():
    return ddp_enabled() and dist_env()[2] > 1


def is_rank0():
    return dist_env()[1] == 0


def build_distributed_sampler(items, seed):
    # One disjoint shard per rank, reseeded every epoch via set_epoch(). The
    # pad-to-even behaviour keeps every rank's step count equal (the DDP
    # allreduce needs matching backward counts); a corpus whose item count
    # divides world_size then covers every item exactly once per epoch.
    import torch
    local_rank, rank, world_size = dist_env()
    return torch.utils.data.DistributedSampler(
        items, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)


def init_distributed(backend=None):
    # nccl on the T4 pair, gloo for the CPU test; idempotent per process.
    if not is_distributed():
        return False
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    local_rank, rank, world_size = dist_env()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # env:// rendezvous needs a master; torchrun sets these, plain
        # torch.multiprocessing.spawn does not (single box => localhost).
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if backend == "nccl":
            # The NCCL HeartbeatMonitor is a SECOND, independent abort timer
            # (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, torch default 480s) that no
            # collective timeout reaches: it fires when the watchdog thread
            # makes no progress for that window, typically because the main
            # thread holds the GIL inside a CUDA call (e.g. CudaEventDestroy)
            # during teardown. That is EXACTLY the 2xT4 smoke abort
            # (laya_1011T044654Z, rev c5e13e6): rank 0 trained + evaluated
            # fine, then "watchdog got stuck for 480 seconds" SIGABRT'd it
            # before the tail ran. The generous collective timeout below does
            # NOT bound it. Size the heartbeat from the SAME SSOT
            # (laya.ddp_timeout_minutes) so a transient GIL stall in teardown
            # can never trip it; a real hang is still bounded by the collective
            # timeout + the rendezvous. Set BEFORE init_process_group reads it.
            os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(
                max(1, int(DDP_TIMEOUT_MINUTES * 60)))
        # Explicit GENEROUS collective timeout (config SSOT: the baked
        # DDP_TIMEOUT_MINUTES literal = laya.ddp_timeout_minutes). Torch's
        # 10-min default is smaller than rank 0's legitimate slow collectives
        # (the final checkpoint save runs while the other ranks wait at the
        # post-training barrier), and the NCCL watchdog then aborts a healthy
        # job: checkTimeout -> DistBackendError, the 2026-10-10 2xT4 run.
        # Defense-in-depth only; the teardown reorder (teardown_after_training)
        # is what removes the divergence.
        init_kwargs = {"timeout": timedelta(minutes=DDP_TIMEOUT_MINUTES)}
        if backend == "nccl":
            # EAGER per-rank device binding (torch>=2.2). The lazy binding is
            # what the hung 2xT4 smoke (2026-10-10T17:36, rev 98eb6084) warned
            # about — "barrier(): using the device under current context" —
            # right before rank 0's watchdog reported "stuck ... another
            # thread holding the GIL inside a CUDA api" and the rank-0 tail
            # (held-out eval + receipt) never ran. Binding the group to THIS
            # rank's device makes every collective and the teardown
            # deterministic. Older torch without the kwarg: retry without it,
            # loudly (full traceback, never silent).
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        try:
            torch.distributed.init_process_group(backend, **init_kwargs)
        except TypeError:
            import traceback
            print("[perf-patch] init_process_group rejected device_id; "
                  "retrying without it\\n" + traceback.format_exc(), flush=True)
            init_kwargs.pop("device_id", None)
            torch.distributed.init_process_group(backend, **init_kwargs)
    if rank == 0:
        # Drop a stale exit-rendezvous marker from a PREVIOUS session/trial so
        # a peer can never mistake it for this teardown's confirmation.
        marker = _teardown_marker_path()
        if marker is not None and os.path.exists(marker):
            try:
                os.remove(marker)
            except OSError:
                import traceback
                print("[perf-patch] stale teardown marker removal failed\\n"
                      + traceback.format_exc(), flush=True)
    return True


class DeterministicDdp:
    """DDP reductions that never depend on a rank's used-parameter graph.

    Letting DDP search for unused parameters makes each rank derive its
    reduction behaviour from ITS OWN used-parameter graph; with
    per-rank-different data the ranks then enqueue a different number of
    collectives and the next barrier (the post-training staging barrier) times
    out. We disable that unused search and instead pin every trainable
    parameter into the loss with an exactly-zero term, so the used set is
    static and identical on every rank and the reduction order cannot diverge.
    ``static_graph=True`` is deliberately NOT used: it also requires the used
    set to be static, which the laya model does not guarantee per rank.
    """

    @staticmethod
    def wrap(torch, model, device, local_rank):
        kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            kwargs.update(device_ids=[local_rank], output_device=local_rank)
        return torch.nn.parallel.DistributedDataParallel(model, **kwargs)

    @staticmethod
    def accumulate(model: "object", synchronize: bool) -> "object":
        from contextlib import nullcontext

        return nullcontext() if synchronize else model.no_sync()

    @staticmethod
    def loss_guard(params):
        # 0 * sum(p) pulls every trainable param into the autograd graph with a
        # zero loss and zero gradient, so find_unused_parameters=False never
        # sees a missing grad, without changing the loss value or the grads.
        return 0.0 * sum(p.sum() for p in params if p.requires_grad)


def barrier_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()


def _teardown_marker_path():
    """The post-destroy exit-rendezvous marker under the kernel's WORKING.

    ``None`` when WORKING is absent (the offline unit-test exec of this
    source), which disables the rendezvous without any caller branch."""
    root = globals().get("WORKING")
    if root is None:
        return None
    return os.path.join(str(root), ".ddp_teardown_done")


def _teardown_rendezvous():
    """Hold every non-rank0 process until rank 0's destroy has landed.

    The peer-exit race (2xT4 smoke, 2026-10-10T17:36, rev 98eb6084 — the
    teardown reorder was already in): rank 0 passed the barrier, then its
    watchdog reported "stuck for 480 seconds ... another thread holding the
    GIL inside a CUDA api" and the rank-0 tail (held-out eval + receipt +
    tar) NEVER ran. A spawned peer that returns from its worker immediately
    after its own destroy starts interpreter finalization and tears its CUDA
    context down while rank 0 is still inside NCCL teardown. The marker keeps
    the peers' contexts alive (bounded, fail-soft, reason recorded) until
    rank 0 confirms its destroy returned."""
    import time
    marker = _teardown_marker_path()
    if marker is None:
        return
    if is_rank0():
        try:
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("destroyed")
        except OSError:
            import traceback
            print("[perf-patch] teardown marker write failed\\n"
                  + traceback.format_exc(), flush=True)
        return
    deadline = time.time() + DDP_TEARDOWN_RENDEZVOUS_SECONDS
    while time.time() < deadline:
        if os.path.exists(marker):
            return
        time.sleep(1.0)
    print("[perf-patch] teardown rendezvous timed out after %.0fs (rank 0 "
          "marker absent); exiting anyway" % DDP_TEARDOWN_RENDEZVOUS_SECONDS,
          flush=True)


def destroy_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
        # AFTER the destroy actually ran: rank 0 publishes the confirmation,
        # the peers hold their contexts until it lands (the exit race fix).
        _teardown_rendezvous()


def teardown_after_training():
    # The 2xT4 NCCL watchdog class (checkTimeout -> DistBackendError, the
    # 2026-10-10 run): rank 0's single-writer tail (held-out eval + receipt +
    # tar of /kaggle/working; 512.7 s measured) ran while every other rank
    # parked on the group's next collective -- the final barrier inside
    # destroy_process_group -- and a tail longer than the watchdog timeout
    # aborted a healthy job. Reordered so NO rank ever parks on a collective
    # during the tail: every rank confirms training done (barrier), every rank
    # destroys the group TOGETHER right now, and only THEN does rank 0 run its
    # tail as a plain single process with no process group alive anywhere.
    # destroy_if_distributed carries the post-destroy exit rendezvous, so a
    # peer's CUDA context cannot die mid-teardown on rank 0 either (the SECOND
    # exposure of the class: the 17:36 smoke hung inside rank 0's NCCL
    # teardown itself, before the tail ever started).
    barrier_if_distributed()
    destroy_if_distributed()


def resolve_nprocs():
    # 2 ranks only when DDP AND the perf loop are on and 2+ cuda devices
    # exist; an already-launched torchrun-style WORLD_SIZE is honoured as-is.
    if not (ddp_enabled() and perf_patch_enabled()):
        return 1
    env_world = int(os.environ.get("WORLD_SIZE", "0") or "0")
    if env_world > 1:
        return env_world
    try:
        import torch
    except Exception:
        return 1
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return torch.cuda.device_count()
    except Exception:
        return 1
    return 1


def launch_finetune(worker):
    # Already launched torchrun-style (WORLD_SIZE>1): this process IS a rank,
    # run it directly. Otherwise spawn one process per visible device (nccl);
    # the caller falls back to the in-process single-device session when this
    # returns 1. spawn (not fork) because resolve_nprocs touched CUDA.
    if is_distributed():
        _, rank, world_size = dist_env()
        worker(rank, world_size)
        return world_size
    nprocs = resolve_nprocs()
    if nprocs > 1:
        import torch.multiprocessing as mp
        print("[perf-patch] distributed launch: %d ranks (nccl)" % nprocs,
              flush=True)
        mp.spawn(worker, args=(nprocs,), nprocs=nprocs, join=True,
                 start_method="spawn")
    return nprocs


'''
