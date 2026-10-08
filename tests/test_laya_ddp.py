"""Offline DDP wiring pins for the laya finetune kernel (no GPU required).

The real kernel launches ``torch.multiprocessing.spawn`` with one process per
visible CUDA device and runs ``_perf_train_model`` from
``FINETUNE_PERF_PATCH_SOURCE`` inside a ``DistributedDataParallel`` model over
a per-rank ``DistributedSampler`` shard. These tests exec that same source on
CPU with the ``gloo`` backend and ``world_size=2`` to prove:

  * the sampler shards are DISJOINT and cover every item once per epoch;
  * the rank-0 gate runs the save/eval function on rank 0 ONLY;
  * a 2-process training run completes with a finite, rank-averaged loss;
  * ``resolve_nprocs``/``launch_finetune`` spawn 2 ranks on 2 GPUs and fall
    back to a single in-process run on CPU (and force 1 for ER_LAYA_DDP=0).
"""
from __future__ import annotations

import json
import math
import os
import random
import types
from pathlib import Path

import torch

from cli import laya_lane

WORLD = 2
N_ITEMS = 8  # divisible by WORLD -> exact disjoint cover with no padding
SEED = 1729


def _load_ddp_namespace():
    """Exec the finetune PERF/DDP source exactly as the rendered kernel does."""
    namespace = {"os": os, "math": math, "random": random}
    exec(laya_lane.FINETUNE_PERF_PATCH_SOURCE, namespace)
    return namespace


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(6, 3)
        self.head_checkpointing = False

    def forward(self, x):
        return self.head(torch.relu(self.encoder(x)))


class _Cfg:
    epochs = 2
    micro_batch = 2
    grad_accum = 1
    encoder_lr = 1e-2
    head_lr = 1e-2
    min_lr = 1e-3
    weight_decay = 0.0
    grad_clip = 1.0
    loss = "soft-ce"
    label_smoothing = 0.0
    rl_samples = 4
    sigma_start = 0.4
    sigma_end = 0.1
    w_sph = 0.75
    w_rps = 1.0
    shuffle_options = ()
    freeze_encoder = False
    seed = SEED
    amp = False
    gradient_checkpointing = False
    log_every = 0
    target_error = 0.1
    min_abstain_n = 10

    def validate(self):
        pass


def _make_fake_laya(processed):
    """A minimal ``laya.train`` so the real loop runs with no laya install."""
    train = types.ModuleType("laya.train")

    def sigma_at(epoch, epochs, sigma_start, sigma_end):
        return sigma_start

    def draw_option_order(it, rng, shuffle_options):
        return None

    def encode_item(tok, it, max_len, head_max_len, order, parallel):
        processed.append(it["i"])
        x = torch.zeros(4)
        x[it["label"]] = 1.0
        x[3] = it["i"] / float(N_ITEMS)
        return {"x": x, "y": it["label"]}

    def collate_items(chunks, pad_token_id):
        flat = [enc for chunk in chunks for enc in chunk]
        features = torch.stack([enc["x"] for enc in flat])
        labels = torch.tensor([enc["y"] for enc in flat], dtype=torch.long)
        return {"x": features,
                "marker_mask": torch.ones(len(flat)),
                "target": labels,
                "qtype": torch.zeros(len(flat), dtype=torch.long)}

    def _forward(model, batch, device, amp, freeze_encoder):
        return model(batch["x"].to(device))

    def soft_ce_loss(logits, target, mask):
        return torch.nn.functional.cross_entropy(logits, target)

    train.sigma_at = sigma_at
    train.draw_option_order = draw_option_order
    train.encode_item = encode_item
    train.collate_items = collate_items
    train._forward = _forward
    train.soft_ce_loss = soft_ce_loss
    laya = types.ModuleType("laya")
    laya.train = train
    return laya


def _ddp_worker(rank, world_size, out_path):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.pop("ER_LAYA_DDP", None)
    os.environ.pop("ER_LAYA_PERF_PATCH", None)
    namespace = _load_ddp_namespace()
    assert namespace["init_distributed"](backend="gloo") is True
    try:
        import torch.distributed as dist
        items = [{"i": i, "label": i % 3} for i in range(N_ITEMS)]
        sampler = namespace["build_distributed_sampler"](items, SEED)
        sampler.set_epoch(0)
        local_indices = list(sampler)
        shards = [None for _ in range(world_size)]
        dist.all_gather_object(shards, local_indices)

        # rank-0 gate: the save/eval callback must fire on rank 0 only.
        gate_calls = []
        gate_results = [None for _ in range(world_size)]
        result = namespace["run_on_rank0"](
            lambda: gate_calls.append(rank) or "staged")
        dist.all_gather_object(gate_results, result)

        processed = []
        laya = _make_fake_laya(processed)
        import sys
        sys.modules["laya"] = laya
        sys.modules["laya.train"] = laya.train
        history = namespace["_perf_train_model"](
            _TinyNet(), types.SimpleNamespace(pad_token_id=0), items, _Cfg(),
            torch.device("cpu"), 8, 4)
        histories = [None for _ in range(world_size)]
        dist.all_gather_object(histories, history)

        if rank == 0:
            Path(out_path).write_text(json.dumps({
                "shards": shards,
                "gate_results": gate_results,
                "gate_calls": gate_calls,
                "history": history,
                "histories": histories,
            }), encoding="utf-8")
    finally:
        namespace["destroy_if_distributed"]()


def test_ddp_two_rank_shards_gate_and_finite_loss(tmp_path):
    out_path = str(tmp_path / "ddp.json")
    torch.multiprocessing.spawn(
        _ddp_worker, args=(WORLD, out_path), nprocs=WORLD, join=True,
        start_method="spawn")
    result = json.loads(Path(out_path).read_text(encoding="utf-8"))

    # disjoint shards that together cover every item exactly once
    rank0, rank1 = result["shards"]
    assert len(rank0) == N_ITEMS // WORLD
    assert len(rank1) == N_ITEMS // WORLD
    assert set(rank0).isdisjoint(set(rank1))
    assert sorted(rank0 + rank1) == list(range(N_ITEMS))

    # rank-0-only gate: only rank 0 returned the staged result
    assert result["gate_results"] == ["staged", None]
    assert result["gate_calls"] == [0]

    # 2-process run completed and the loss is finite (finite on both ranks)
    assert result["history"]
    assert all(math.isfinite(value) for value in result["history"])
    # the epoch loss is rank-averaged: every rank reports the same numbers
    assert result["histories"][0] == result["histories"][1]
    assert all(
        all(math.isfinite(value) for value in history)
        for history in result["histories"])


def test_dist_env_reads_rank_vars_and_ddp_optout(monkeypatch):
    namespace = _load_ddp_namespace()
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("ER_LAYA_DDP", raising=False)
    assert namespace["dist_env"]() == (1, 1, 2)
    assert namespace["is_distributed"]() is True
    assert namespace["is_rank0"]() is False
    # ER_LAYA_DDP=0 forces the single-process fallback even under WORLD_SIZE>1
    monkeypatch.setenv("ER_LAYA_DDP", "0")
    assert namespace["is_distributed"]() is False


def test_resolve_nprocs_fallbacks_and_optout(monkeypatch):
    namespace = _load_ddp_namespace()
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("ER_LAYA_DDP", raising=False)
    monkeypatch.delenv("ER_LAYA_PERF_PATCH", raising=False)
    # no cuda on this box -> single process
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert namespace["resolve_nprocs"]() == 1
    # two GPUs -> two ranks
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert namespace["resolve_nprocs"]() == 2
    # the ER_LAYA_DDP=0 opt-out beats two GPUs
    monkeypatch.setenv("ER_LAYA_DDP", "0")
    assert namespace["resolve_nprocs"]() == 1
    # the perf loop is required for the sharded loop
    monkeypatch.delenv("ER_LAYA_DDP")
    monkeypatch.setenv("ER_LAYA_PERF_PATCH", "0")
    assert namespace["resolve_nprocs"]() == 1
    monkeypatch.delenv("ER_LAYA_PERF_PATCH")
    # an already-launched torchrun-style world size is honoured
    monkeypatch.setenv("WORLD_SIZE", "3")
    assert namespace["resolve_nprocs"]() == 3


def test_launch_finetune_spawns_two_ranks_when_two_gpus(monkeypatch):
    namespace = _load_ddp_namespace()
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("ER_LAYA_DDP", raising=False)
    monkeypatch.delenv("ER_LAYA_PERF_PATCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    calls = {}

    def fake_spawn(fn, args=(), nprocs=1, join=True, daemon=False,
                   start_method=None):
        calls["fn"] = fn
        calls["nprocs"] = nprocs
        calls["start_method"] = start_method

    monkeypatch.setattr(torch.multiprocessing, "spawn", fake_spawn)

    def worker(rank, world_size):
        return None

    assert namespace["launch_finetune"](worker) == 2
    assert calls == {"fn": worker, "nprocs": 2, "start_method": "spawn"}


def test_launch_finetune_runs_in_process_when_already_launched(monkeypatch):
    namespace = _load_ddp_namespace()
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("ER_LAYA_DDP", raising=False)
    monkeypatch.delenv("ER_LAYA_PERF_PATCH", raising=False)
    spawned = []
    monkeypatch.setattr(torch.multiprocessing, "spawn",
                        lambda *args, **kwargs: spawned.append(1))
    seen = {}

    def worker(rank, world_size):
        seen["rank"] = rank
        seen["world_size"] = world_size

    assert namespace["launch_finetune"](worker) == 2
    assert spawned == []
    assert seen == {"rank": 1, "world_size": 2}


def test_launch_finetune_single_process_on_cpu(monkeypatch):
    namespace = _load_ddp_namespace()
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("ER_LAYA_DDP", raising=False)
    monkeypatch.delenv("ER_LAYA_PERF_PATCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    spawned = []
    monkeypatch.setattr(torch.multiprocessing, "spawn",
                        lambda *args, **kwargs: spawned.append(1))
    assert namespace["launch_finetune"](lambda rank, world: None) == 1
    assert spawned == []


def test_rendered_kernel_bakes_ddp_wiring_without_leftover_tokens():
    import re

    values = {
        "LAYA_PACKAGE": laya_lane.FINETUNE_LAYA_PACKAGE,
        "BASE_MODEL_ARCHIVE": "convaiinnovations-laya.tar.zst",
        "BASE_MODEL_DIR": "convaiinnovations-laya",
        "RUN_TAG": "gpu_test",
        "TRAIN_JSONL": "train.jsonl",
        "DEV_JSONL": "dev.jsonl",
        "TEST_JSONL": "test.jsonl",
        "FINETUNE_CONFIG": repr(laya_lane.finetune_config()),
        "FINETUNE_DEVICE": "auto",
        "HELD_OUT_BATCH": str(laya_lane._spec().laya_decision_batch_size),
        "REPOSITORY": "anomalyco/er",
        "BRANCH": "main",
        "REVISION": "deadbeef",
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
    }
    preflight = laya_lane._template(
        laya_lane.FINETUNE_RUNTIME_PREFLIGHT, values)
    script = laya_lane._template(
        laya_lane.FINETUNE_KERNEL_SCRIPT,
        {**values, "RUNTIME_PREFLIGHT": preflight})
    laya_lane._kernel_script_gate(script)
    laya_lane._module_scope_gate(script)
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)
    assert "def launch_finetune" in script
    assert "def finetune_worker" in script
    assert 'DistributedDataParallel' in script
    assert "build_distributed_sampler" in script
    assert "run_on_rank0(rank0_work)" in script
    assert "destroy_if_distributed()" in script
    # the single-process branch is preserved byte-for-byte
    assert "random.Random(config.seed + epoch).shuffle(epoch_items)" in script


def test_ddp_find_unused_parameters_is_pinned():
    """The 2xT4 run crashed with 'Expected to have finished reduction in the
    prior iteration' — laya's model leaves parameters unused under DDP, so the
    wrapper MUST pass find_unused_parameters=True. Pin it so it cannot regress."""
    from cli import laya_lane as L

    src = L.FINETUNE_PERF_PATCH_SOURCE
    assert "DistributedDataParallel" in src
    assert src.count("find_unused_parameters=True") >= 2, (
        "both the CUDA and CPU DDP wrappers must set find_unused_parameters=True")
