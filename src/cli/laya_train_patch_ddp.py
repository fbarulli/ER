"""DDP + control-logic half of the fine-tune PERF_PATCH kernel text.

The fine-tune kernel's `@PERF_PATCH@` injection is one template split at the
`class ControlCheckpointer` boundary so each half stays small: this module owns
the control-logic injection and the distributed/data-parallel helpers, the
sibling `laya_train_patch_loop` owns the training loop.
"""
from __future__ import annotations

import inspect

from core import laya_controls


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
        laya_controls.ProfilerSession,
    )
)


DDP_PATCH_TEMPLATE = '''\
@CONTROL_LOGIC@

# The kernel template defines the wandb helpers BEFORE this patch is injected;
# the offline DDP unit test execs this source alone, so fall back to no-ops.
try:
    wandb_log_epoch
except NameError:
    def wandb_log_epoch(epoch, mean, extra=None):
        return None

    def wandb_log_control_summary(result):
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
        torch.distributed.init_process_group(backend)
    return True


def barrier_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()


def destroy_if_distributed():
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def run_on_rank0(fn):
    # rank 0 ONLY stages: the barrier first guarantees every rank finished
    # training before the single writer touches /kaggle/working.
    barrier_if_distributed()
    if is_rank0():
        return fn()
    return None


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
