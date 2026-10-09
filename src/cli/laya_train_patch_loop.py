"""Training-loop half of the fine-tune PERF_PATCH kernel text.

Concatenated after `laya_train_patch_ddp.DDP_PATCH_TEMPLATE` and joined with
the control logic to rebuild ``FINETUNE_PERF_PATCH_SOURCE`` byte-for-byte.
"""
from __future__ import annotations

from cli.laya_train_patch_ddp import (
    DDP_PATCH_TEMPLATE,
    FINETUNE_CONTROL_LOGIC_SOURCE,
)


LOOP_PATCH_TEMPLATE = '''\
class ControlCheckpointer:
    """Per-epoch checkpoint save + resume with optimizer/scheduler state.

    The checkpoint dir is the CANONICAL, shared path on EVERY rank (rank 0
    writes, all ranks resume), deliberately separate from the per-rank final
    save dir. A ``run_tag`` is stored with each file so a resume can never
    continue from another run's checkpoint, and the best-metric weights are
    persisted separately so a resumed run that never improves ends on the
    best weights, not the last epoch.
    """

    def __init__(self, torch_module, checkpoint_dir, run_tag):
        self._torch = torch_module
        self._checkpoint_dir = checkpoint_dir
        self._run_tag = run_tag

    @classmethod
    def for_training(cls, torch_module):
        # FINETUNE_CHECKPOINT_DIR is the shared canonical dir (all ranks);
        # FINETUNE_OUTPUT_DIR is the per-rank fallback for single-process.
        directory = (globals().get("FINETUNE_CHECKPOINT_DIR")
                     or globals().get("FINETUNE_OUTPUT_DIR"))
        return cls(torch_module, directory, globals().get("FINETUNE_RUN_TAG"))

    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model

    def directory(self):
        if not self._checkpoint_dir:
            return None
        return os.path.join(str(self._checkpoint_dir), "checkpoints")

    def _matches_run(self, state):
        return self._run_tag is None or state.get("run_tag") == self._run_tag

    def save(self, model, optimizer, scheduler, epoch, best, bad_epochs):
        if not is_rank0():
            return
        path = self.directory()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        self._torch.save({
            "epoch": int(epoch),
            "model": self.unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best,
            "bad_epochs": int(bad_epochs),
            "run_tag": self._run_tag,
        }, os.path.join(path, "epoch_%d.pt" % int(epoch)))

    def save_best(self, model, metric, accuracy, epoch):
        """Persist the best early-stop-metric weights (rank 0 only)."""
        if not is_rank0():
            return
        path = self.directory()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        self._torch.save({
            "model": self.unwrap(model).state_dict(),
            "metric": metric, "accuracy": accuracy, "epoch": int(epoch),
            "run_tag": self._run_tag,
        }, os.path.join(path, "best.pt"))

    def _load_best(self, path, device):
        best_path = os.path.join(path, "best.pt")
        if not os.path.isfile(best_path):
            return None
        try:
            state = self._torch.load(best_path, map_location=device,
                                     weights_only=False)
        except Exception:
            return None
        return state if self._matches_run(state) else None

    def resume(self, model, optimizer, scheduler, device):
        """Return ``(start_epoch, best, bad_epochs, best_record)``.

        Every rank resumes from the SAME shared file so ranks stay in
        lockstep; a checkpoint from a different ``run_tag`` is ignored. The
        best record carries the saved best weights + metric so a resumed run
        that never improves still ends on the best epoch.
        """
        path = self.directory()
        if not path or not os.path.isdir(path):
            return 0, None, 0, None
        names = [name for name in os.listdir(path)
                 if name.startswith("epoch_") and name.endswith(".pt")
                 and name[len("epoch_"):-3].isdigit()]
        names.sort(key=lambda name: int(name[len("epoch_"):-3]))
        for name in reversed(names):
            try:
                state = self._torch.load(os.path.join(path, name),
                                         map_location=device,
                                         weights_only=False)
            except Exception as error:
                print("[perf-patch] resume skipped " + name + ": "
                      + str(error)[:160], flush=True)
                continue
            if not self._matches_run(state):
                continue
            self.unwrap(model).load_state_dict(state["model"])
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
            except Exception as error:
                print("[perf-patch] resume state skipped: "
                      + str(error)[:160], flush=True)
            best_record = self._load_best(path, device)
            print("[perf-patch] resumed " + name + " (next epoch "
                  + str(int(state.get("epoch", 0)) + 2) + ")", flush=True)
            return (int(state.get("epoch", 0)) + 1, state.get("best"),
                    int(state.get("bad_epochs", 0)), best_record)
        return 0, None, 0, None


class AdversarialPerturber:
    """FGM (embedding-direction) / AWP (weight-direction) perturbation.

    ``fgm`` perturbs the embedding parameters (the classic Fast Gradient
    Method input perturbation); ``awp`` perturbs every trainable parameter.
    Both step along the gradient direction and are restored after the extra
    forward/backward, so the recipe is unchanged when ``adv_eps == 0``.
    """

    @staticmethod
    def targets(model, params, adv_kind):
        if str(adv_kind).lower() != "fgm":
            return list(params)
        named = dict(model.named_parameters())
        picked = [param for name, param in named.items()
                  if "embed" in name.lower() and param.requires_grad]
        return picked or list(params)

    @staticmethod
    def perturb(torch, params, adv_eps, norm):
        deltas = []
        with torch.no_grad():
            for param in params:
                if param.grad is None:
                    continue
                delta = (adv_eps / norm) * param.grad.detach()
                param.add_(delta)
                deltas.append((param, delta))
        return deltas

    @staticmethod
    def restore(torch, deltas):
        with torch.no_grad():
            for param, delta in deltas:
                param.sub_(delta)


class TrainingOptimizer:
    """Optimizer construction + best-effort layer-wise LR decay."""

    @staticmethod
    def make(torch, model, groups, config, control):
        kind = str(control.get("optimizer") or "").lower()
        groups = ParamGroupBuilder.apply(
            model, groups, bool(control.get("no_decay_bias_norm")))
        fused = any(param.is_cuda for group in groups
                    for param in group["params"])
        if kind == "adafactor":
            try:
                return torch.optim.Adafactor(
                    groups, lr=config.head_lr,
                    weight_decay=config.weight_decay)
            except Exception:
                print("[perf-patch] Adafactor unavailable; AdamW", flush=True)
        if kind == "lamb":
            try:
                from torch_optimizer import Lamb
                return Lamb(groups, lr=config.head_lr,
                            weight_decay=config.weight_decay)
            except Exception:
                try:
                    from transformers.optimization import Lamb
                    return Lamb(groups, lr=config.head_lr,
                                weight_decay=config.weight_decay)
                except Exception:
                    print("[perf-patch] Lamb unavailable; AdamW", flush=True)
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay,
                                 fused=fused)

    @staticmethod
    def apply_layer_decay(model, groups, layer_decay):
        import re as _re

        depths = {}
        for name, param in model.named_parameters():
            match = _re.search(r"(?:layers?|layer)[.]([0-9]+)[.]", name)
            if match:
                depths[id(param)] = int(match.group(1))
        if not depths:
            return groups
        deepest = max(depths.values())
        decayed = []
        for group in groups:
            base_lr = float(group.get("lr"))
            for param in group["params"]:
                depth = depths.get(id(param))
                lr = base_lr if depth is None else base_lr * (
                    layer_decay ** (deepest - depth))
                decayed.append({"params": [param], "lr": lr})
        return decayed


class LossBuilder:
    """Class weighting, weighted soft-CE and per-row loss for hard mining."""

    @staticmethod
    def class_weights(torch, items, device):
        counts = {}
        for item in items:
            try:
                target = item["target"]
                label = max(range(len(target)),
                            key=lambda i: float(target[i]))
            except Exception:
                continue
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return None
        classes = max(counts) + 1
        total = sum(counts.values())
        weights = [0.0] * classes
        for label in range(classes):
            count = counts.get(label, 0)
            weights[label] = (total / (classes * count)) if count else 0.0
        return torch.tensor(weights, dtype=torch.float32, device=device)

    @staticmethod
    def weighted(torch, logits, target, mask, weights):
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
        per_row = -(target * logp * mask).sum(-1)
        labels = target.argmax(dim=-1)
        row_w = weights.to(logits.device)[labels]
        return (per_row * row_w).mean()

    @staticmethod
    def per_row(torch, logits, target, mask):
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), -1)
        return (-(target * logp * mask).sum(-1)).detach()

    @staticmethod
    def compute(torch, laya_train, config, logits, target, mask, qtype, sigma,
                w_sph, w_rps, class_weights, margin):
        """The objective for one forward: laya's rlcd / weighted / soft-CE.

        With no class weights and margin==0 this returns laya's
        ``soft_ce_loss`` unchanged (byte-identical default). The contrastive
        margin is applied to the soft-CE objective only (laya's ``rlcd_loss``
        exposes no margin); class_weight takes precedence over margin.
        """
        if config.loss == "rlcd":
            return laya_train.rlcd_loss(logits, target, mask, qtype, sigma,
                                        config.rl_samples, w_sph, w_rps)
        if margin and float(margin) > 0.0:
            logits = logits - float(margin) * (1.0 - target)
        if class_weights is not None:
            return LossBuilder.weighted(torch, logits, target, mask,
                                        class_weights)
        return laya_train.soft_ce_loss(logits, target, mask)


class Forwarder:
    """The bf16-aware forward wrapper (fp16 defers to laya's own _forward)."""

    @staticmethod
    def run(torch, laya_train, model, batch, device, amp, freeze_encoder,
            amp_dtype):
        if (not amp) or amp_dtype != "bf16" or not hasattr(torch, "autocast"):
            return laya_train._forward(model, batch, device, amp,
                                       freeze_encoder)
        args = (batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device))
        kwargs = {"detach_encoder": freeze_encoder}
        if "option_ids" in batch:
            kwargs.update(position_ids=batch["position_ids"].to(device),
                          option_ids=batch["option_ids"].to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, _act = model(*args, **kwargs)
        return logits.float()


class DistributedBroadcast:
    """Bring a vector of scalars into lockstep across DDP ranks (src=0)."""

    @staticmethod
    def values(torch, values, device):
        if not is_distributed():
            return [float(value) for value in values]
        import torch.distributed as dist

        tensor = torch.tensor([float(value) for value in values],
                              device=device)
        dist.broadcast(tensor, src=0)
        return [float(value) for value in tensor.tolist()]


class EpochItemSelector:
    """Curriculum / balanced sampling / hard-example mining (single-process)."""

    @staticmethod
    def select(control, items, epoch, config, seed):
        selected = list(items)
        if control.get("curriculum"):
            keep = max(1, int(round(
                len(selected) * (epoch + 1) / config.epochs)))
            rng = random.Random(seed + 7919 * epoch)
            rng.shuffle(selected)
            selected = selected[:keep]
        if control.get("balanced_sample"):
            selected = EpochItemSelector._balanced(selected, seed, epoch)
        frac = control.get("hard_example_frac")
        scores = globals().get("FINETUNE_HARD_SCORES")
        if frac and float(frac) > 0.0 and isinstance(scores, dict) and scores:
            ranked = sorted(
                selected, key=lambda item: scores.get(id(item), 0.0),
                reverse=True)
            selected = ranked[:max(1, int(round(len(ranked) * frac)))]
        return selected

    @staticmethod
    def _balanced(selected, seed, epoch):
        buckets = {}
        for item in selected:
            try:
                label = max(range(len(item["target"])),
                            key=lambda i: float(item["target"][i]))
            except Exception:
                label = -1
            buckets.setdefault(label, []).append(item)
        if len(buckets) <= 1:
            return selected
        per = max(len(bucket) for bucket in buckets.values())
        rng = random.Random(seed + 104729 * epoch)
        balanced = []
        for bucket in buckets.values():
            balanced.extend(rng.choice(bucket) for _ in range(per))
        rng.shuffle(balanced)
        return balanced


class DevEvaluator:
    """Rank-0 dev scoring + end-of-run calibration/confusion artifacts."""

    @staticmethod
    def metrics(laya_train, model, tok, dev_items, device, max_len,
                head_max_len, parallel, control, annotate=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, annotate)
        metrics = laya_train.evaluate_records(records)
        return DevReport.from_metrics(
            metrics, records=records,
            confidence_threshold=control.get("abstain_confidence"))

    @staticmethod
    def artifacts(laya_train, model, tok, dev_items, device, max_len,
                  head_max_len, parallel, control, annotate=None):
        records = DevEvaluator._records(laya_train, model, tok, dev_items,
                                        device, max_len, head_max_len,
                                        parallel, annotate)
        metrics = laya_train.evaluate_records(records)
        confusion = {}
        for qtype, logits, target, k in records:
            try:
                import numpy as np
                z = np.asarray(logits, dtype=float)[:int(k)]
                pred = int(z.argmax())
                gold = int(np.asarray(target, dtype=float)[:int(k)].argmax())
            except Exception:
                continue
            bucket = confusion.setdefault(str(qtype), {})
            row = bucket.setdefault(str(gold), {})
            row[str(pred)] = row.get(str(pred), 0) + 1
        temperature = None
        if control.get("temperature_scale"):
            try:
                temperature = laya_train.fit_temperature_map(
                    records).get("temperature")
            except Exception as error:
                print("[perf-patch] temperature fit skipped: "
                      + str(error)[:120], flush=True)
        return {
            "items": len(records),
            "accuracy": metrics.get("accuracy"),
            "ece": metrics.get("ece"),
            "brier": metrics.get("brier"),
            "brier_top1": metrics.get("brier_top1"),
            "confusion": confusion,
            "temperature": temperature,
        }

    @staticmethod
    def _records(laya_train, model, tok, dev_items, device, max_len,
                 head_max_len, parallel, annotate=None):
        import contextlib

        scope = annotate("calibration") if annotate else contextlib.nullcontext()
        with scope:
            return laya_train.calibration_records(
                model, tok, dev_items, device, max_len, head_max_len,
                parallel=parallel)


class WandbProfileSink:
    """Mirrors the ProfilerSession top-ops table to the live wandb run."""

    @staticmethod
    def log(rows, epoch, trace_path):
        try:
            if WANDB_RUN is None:
                return
            import wandb
            table = wandb.Table(
                columns=["op", "cuda_ms", "cpu_ms", "count"],
                data=[[key, cuda_ms, cpu_ms, count]
                      for key, cuda_ms, cpu_ms, count in rows])
            WANDB_RUN.log({"profile/top_ops": table, "profile/epoch": epoch,
                           "profile/trace": str(trace_path)})
        except Exception:
            pass


class WandbTimingSink:
    """Mirrors the per-phase wall-clock seconds to the live wandb run."""

    @staticmethod
    def log(payload):
        try:
            if WANDB_RUN is None:
                return
            WANDB_RUN.log(payload)
        except Exception as error:
            print("[perf-patch] timing wandb skipped: "
                  + type(error).__name__ + ": " + str(error)[:200], flush=True)


def _perf_train_model(model, tok, items, config, device, max_len, head_max_len,
                      on_epoch_end=None, parallel=False):
    # Faithful copy of laya.train.train_model (0.3.29) carrying the three
    # recipe-neutral perf changes AND the YAML-driven training controls read
    # from the baked FINETUNE_CONTROL block (never a TrainConfig kwarg).
    import time
    import torch
    from laya import train as laya_train

    # laya ran the pre-train dev evaluation inside finetune() before calling
    # this loop; log the transition so the quiet stretch is visible remotely.
    wandb_log_event("pretrain_eval_done")

    controls = TrainingControls.parse(globals().get("FINETUNE_CONTROL"), {})
    control = controls.block
    config.validate()
    if not items:
        raise ValueError("no training items")
    amp = (device.type == "cuda") if config.amp is None else bool(config.amp)
    amp_dtype = str(control.get("amp_dtype") or "").lower()
    if control.get("tf32") and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if control.get("deterministic"):
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as error:
            print("[perf-patch] determinism unavailable: "
                  + str(error)[:120], flush=True)
    checkpointing = (amp if config.gradient_checkpointing is None
                     else bool(config.gradient_checkpointing))
    unfreeze_after = control.get("unfreeze_after_epoch")
    gradual = (unfreeze_after is not None) and (not config.freeze_encoder)
    if config.freeze_encoder or gradual:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif checkpointing and hasattr(model.encoder,
                                   "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    if config.freeze_encoder or gradual:
        model.encoder.eval()

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": head_params, "lr": config.head_lr}]
    if not config.freeze_encoder:
        encoder_params = ([p for _n, p in model.named_parameters()
                           if _n.startswith("encoder.")] if gradual else
                          [p for _n, p in model.named_parameters()
                           if _n.startswith("encoder.") and p.requires_grad])
        groups.insert(0, {"params": encoder_params, "lr": config.encoder_lr})
    layer_decay = control.get("layer_decay")
    if layer_decay and 0.0 < float(layer_decay) < 1.0:
        groups = TrainingOptimizer.apply_layer_decay(model, groups, float(layer_decay))
    optimizer = TrainingOptimizer.make(torch, model, groups, config, control)
    # LR scaling from the effective batch (explicit encoder_lr/head_lr win when
    # lr_scaling == "none"): scaled BEFORE the scheduler so its base LRs and
    # onecycle max_lr see the scaled peak.
    world_size = dist_env()[2] if is_distributed() else 1
    effective_batch = LrScaler.effective_batch(
        config.micro_batch, config.grad_accum, world_size)
    lr_factor = LrScaler.factor(effective_batch, control.get("base_batch"),
                                control.get("lr_scaling"))
    LrScaler.apply(optimizer, lr_factor)
    print("[perf-patch] lr_scaling=%s effective_batch=%d world_size=%d "
          "factor=%.4g"
          % (control.get("lr_scaling"), effective_batch, world_size,
             lr_factor), flush=True)
    # DDP: wrap the model (grads averaged across ranks) and shard the items
    # with a per-rank DistributedSampler. The shard length is equal on every
    # rank (pad-to-even), so the grad-accum window and the optimizer steps
    # stay in lockstep across the DDP allreduce.
    ddp_sampler = None
    if is_distributed():
        local_rank, rank, world_size = dist_env()
        ddp_sampler = build_distributed_sampler(items, config.seed)
        model = DeterministicDdp.wrap(torch, model, device, local_rank)
        print("[perf-patch] ddp: rank %d/%d, %d local items"
              % (rank, world_size, len(ddp_sampler)), flush=True)
    elif control.get("compile_model"):
        try:
            model = torch.compile(model)
            print("[perf-patch] torch.compile enabled", flush=True)
        except Exception as error:
            print("[perf-patch] torch.compile skipped: "
                  + str(error)[:120], flush=True)
    epoch_len = len(ddp_sampler) if ddp_sampler is not None else len(items)
    steps_per_epoch = math.ceil(epoch_len / config.micro_batch)
    updates = max(1, math.ceil(steps_per_epoch / config.grad_accum)
                  * config.epochs)
    scheduler_factory = controls.scheduler(optimizer, updates, config.min_lr)
    scheduler = scheduler_factory.build()
    scheduler_kind = scheduler_factory.kind
    warmup = scheduler_factory.warmup
    plateau = scheduler_factory.is_plateau
    stop_policy = controls.early_stop_policy()
    lower_is_better = stop_policy.lower_is_better
    early_metric = control.get("early_stop_metric")
    use_scaler = (amp and device.type == "cuda" and amp_dtype != "bf16")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    print("[perf-patch] amp=%s amp_dtype=%s checkpointing=%s scaler=%s scheduler=%s warmup=%d updates=%d steps_per_epoch=%d"
          % (amp, amp_dtype or "none", checkpointing, use_scaler, scheduler_kind, warmup, updates, steps_per_epoch), flush=True)

    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g["params"]]
    history = []
    cache = {}
    hits = lookups = 0
    # ── control state ────────────────────────────────────────────────────
    # `dev_enabled` is computed identically on every rank (the control block
    # and the stashed rows are the same everywhere), so the per-epoch
    # broadcast collective is entered by ALL ranks or none — never a subset.
    dev_rows = globals().get("FINETUNE_DEV_ROWS")
    dev_enabled = bool(control.get("eval_dev")) and dev_rows is not None
    dev_items = None
    if dev_enabled:
        dev_items, dev_skipped = laya_train.items_from_rows(
            tok, dev_rows, max_len, head_max_len, label_smoothing=0.0)
        print("[perf-patch] dev eval items %d (skipped %r)"
              % (len(dev_items), dev_skipped), flush=True)
    class_weights = (LossBuilder.class_weights(torch, items, device)
                     if control.get("class_weight") else None)
    loss_schedule = LossWeightSchedule.from_control(control, config)
    r_drop_requested = bool(control.get("r_drop"))
    r_drop_on = r_drop_requested and RDrop.available(model)
    if r_drop_requested and not r_drop_on:
        print("[perf-patch] r_drop auto-disabled: model carries no dropout>0",
              flush=True)
    r_drop_alpha = control.get("r_drop_alpha")
    drop_path_on = bool(control.get("drop_path"))
    dynamic_padding_on = bool(control.get("dynamic_padding"))
    pad_to_multiple = control.get("pad_to_multiple")
    ramp_on = bool(control.get("batch_size_ramp"))
    target_grad_accum = int(config.grad_accum)
    optim_state_dtype = control.get("optim_state_dtype")
    adv_eps = control.get("adv_eps")
    adv_kind = control.get("adv_kind")
    ema_decay = (float(control.get("ema_decay"))
                 if control.get("ema") else None)
    ema_state = ({id(p): p.detach().clone() for p in params}
                 if control.get("ema") else None)
    swa_on = bool(control.get("swa"))
    swa_lr = control.get("swa_lr")
    swa_state = ({id(p): p.detach().clone() for p in params}
                 if swa_on else None)
    swa_count = 0
    swa_start = (int(round(float(control.get("swa_start_frac"))
                           * config.epochs)) if swa_on else None)
    checkpointer = ControlCheckpointer.for_training(torch)
    start_epoch, best, bad_epochs, best_record = 0, None, 0, None
    if control.get("resume"):
        start_epoch, best, bad_epochs, best_record = checkpointer.resume(
            model, optimizer, scheduler, device)
    best_acc = (best_record or {}).get("accuracy")
    best_state = (best_record or {}).get("model")
    stopped = False
    stopped_epoch = None
    profiler = ProfilerSession.for_training(
        torch, control, device, is_rank0(),
        globals().get("FINETUNE_OUTPUT_DIR"), on_metrics=WandbProfileSink.log,
        timings=globals().get("PHASE_TIMINGS"))
    if profiler.enabled:
        profiler.start()

    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.time()
        profiler.epoch = epoch + 1
        if gradual and epoch >= int(unfreeze_after):
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            model.encoder.train()
        if swa_on and epoch == swa_start and swa_lr:
            for group in optimizer.param_groups:
                group["lr"] = float(swa_lr)
            print("[perf-patch] SWA phase: lr -> %.6g" % float(swa_lr),
                  flush=True)
        if drop_path_on:
            DropPath.apply(model, DropPath.rate(
                epoch, config.epochs, control.get("drop_path_rate"),
                control.get("drop_path_schedule")))
        # batch-size ramp ramps grad_accum (rank-symmetric => DDP lockstep);
        # steps_per_epoch/updates stay on the TARGET so schedules keep length.
        epoch_grad_accum = (BatchRamp.grad_accum(
            epoch, target_grad_accum, control.get("batch_ramp_start_frac"),
            control.get("batch_ramp_epochs")) if ramp_on else target_grad_accum)
        if ddp_sampler is not None:
            # Per-epoch reseed: every rank shuffles identically then takes a
            # disjoint stride slice, so no item is trained twice per epoch.
            ddp_sampler.set_epoch(epoch)
            epoch_items = [items[i] for i in ddp_sampler]
        else:
            epoch_items = list(items)
            random.Random(config.seed + epoch).shuffle(epoch_items)
            epoch_items = EpochItemSelector.select(control, epoch_items, epoch,
                                              config, config.seed)
        hard_scores = ({} if control.get("hard_example_frac") else None)
        # ONE loss-weight schedule: mode "laya" reproduces sigma_at exactly and
        # keeps w_sph/w_rps/margin constant (byte-identical default).
        weights = loss_schedule.at(epoch)
        sigma = weights["sigma"]
        epoch_w_sph, epoch_w_rps = weights["w_sph"], weights["w_rps"]
        epoch_margin = weights["margin"]
        total, n_steps = None, 0
        grad_norm_sum, grad_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk_items = epoch_items[start:start + config.micro_batch]
            chunk = []
            with profiler.phase("data.encode"):
                for it in chunk_items:
                    order = laya_train.draw_option_order(
                        it, order_rng, config.shuffle_options)
                    key = (id(it), tuple(order) if order is not None else None,
                           max_len, head_max_len, parallel)
                    encoded = cache.get(key)
                    if encoded is None:
                        encoded = laya_train.encode_item(
                            tok, it, max_len, head_max_len, order, parallel)
                        cache[key] = encoded
                    else:
                        hits += 1
                    lookups += 1
                    chunk.append(encoded)
            with profiler.phase("collate+batch_move"):
                batch = laya_train.collate_items([chunk], tok.pad_token_id)
                # dynamic padding: laya's collate already pads to the batch
                # longest (max_len only TRUNCATES); this rounds the collated
                # batch length to pad_to_multiple.
                if dynamic_padding_on:
                    batch = DynamicPadder.pad(torch, batch, pad_to_multiple)
                # (2) one device move for what the loop consumes; _forward's own
                # .to(device) on the same device is then a no-op.
                mask = batch["marker_mask"].to(device)
                target = batch["target"].to(device)
                qtype = batch["qtype"].to(device)
            with profiler.phase("forward"):
                logits = Forwarder.run(torch, laya_train, model, batch, device,
                                        amp, config.freeze_encoder, amp_dtype)
            with profiler.phase("loss"):
                loss = LossBuilder.compute(
                    torch, laya_train, config, logits, target, mask, qtype,
                    sigma, epoch_w_sph, epoch_w_rps, class_weights,
                    epoch_margin)
                if ddp_sampler is not None:
                    loss = loss + DeterministicDdp.loss_guard(params)
                if r_drop_on:
                    logits_two = Forwarder.run(
                        torch, laya_train, model, batch, device, amp,
                        config.freeze_encoder, amp_dtype)
                    loss_two = LossBuilder.compute(
                        torch, laya_train, config, logits_two, target, mask,
                        qtype, sigma, epoch_w_sph, epoch_w_rps, class_weights,
                        epoch_margin)
                    kl = RDrop.kl(torch, logits, logits_two, mask)
                    loss = RDrop.combine(loss, loss_two, kl, r_drop_alpha)
            window_start = (n_steps // epoch_grad_accum) * epoch_grad_accum
            window_size = min(epoch_grad_accum,
                              steps_per_epoch - window_start)
            scaled = loss / window_size
            with profiler.phase("backward"):
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            if adv_eps and float(adv_eps) > 0.0:
                # FGM perturbs the embedding params, AWP every trainable param;
                # both take an extra backward then restore the weights.
                targets = AdversarialPerturber.targets(model, params, adv_kind)
                total_sq = 0.0
                for p in targets:
                    if p.grad is not None:
                        total_sq += float(p.grad.detach().pow(2).sum())
                norm = math.sqrt(total_sq) + 1e-12
                deltas = AdversarialPerturber.perturb(
                    torch, targets, float(adv_eps), norm)
                adv_logits = Forwarder.run(
                    torch, laya_train, model, batch, device, amp,
                    config.freeze_encoder, amp_dtype)
                adv_loss = laya_train.soft_ce_loss(adv_logits, target, mask)
                adv_scaled = adv_loss / window_size
                if scaler is not None:
                    scaler.scale(adv_scaled).backward()
                else:
                    adv_scaled.backward()
                AdversarialPerturber.restore(torch, deltas)
            n_steps += 1
            if (n_steps % epoch_grad_accum == 0
                    or start + config.micro_batch >= len(epoch_items)):
                with profiler.phase("optimizer_step"):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                    if control.get("log_grad_norm") and grad_norm is not None:
                        grad_norm_sum += grad_norm.detach().clamp(max=config.grad_clip)
                        grad_steps += 1
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if not plateau:
                        scheduler.step()
                    if str(optim_state_dtype or "fp32").lower() == "bf16":
                        OptimizerStateCaster.apply(torch, optimizer,
                                                   optim_state_dtype)
                    if ema_state is not None:
                        with torch.no_grad():
                            for p in params:
                                shadow = ema_state[id(p)]
                                shadow.mul_(ema_decay).add_(
                                    p.detach(), alpha=1.0 - ema_decay)
                    optimizer.zero_grad(set_to_none=True)
            if hard_scores is not None:
                per_row = LossBuilder.per_row(torch, logits, target, mask)
                for item, value in zip(chunk_items, per_row.tolist()):
                    hard_scores[id(item)] = float(value)
            # (1) stay on-GPU: accumulate the loss, sync once per epoch.
            detached = loss.detach()
            total = detached if total is None else total + detached
            if config.log_every and n_steps % config.log_every == 0:
                # Per-step loss: ONE extra GPU sync every log_every steps
                # (negligible); it was dropped when the loop switched to
                # on-GPU accumulation, which left the log loss-less.
                print("epoch %d/%d step %d loss %.4f" % (
                    epoch + 1, config.epochs, n_steps,
                    float(detached.item())), flush=True)
            # Bounded schedule: one profiler "step" per micro-batch.
            profiler.step()
        if hard_scores is not None:
            globals()["FINETUNE_HARD_SCORES"] = hard_scores
        mean = (float(total.item() / max(1, n_steps))
                if total is not None else 0.0)
        # DDP already averages the gradients; report the rank-averaged scalar
        # too so every rank logs the same honest epoch loss.
        if is_distributed():
            import torch.distributed as dist
            mean_tensor = torch.tensor(mean, device=device)
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            mean = float(mean_tensor.item()) / dist_env()[2]
        history.append(mean)
        epoch_time = max(0.0, time.time() - epoch_started)
        lr_now = float(optimizer.param_groups[0]["lr"])
        print("epoch %d/%d mean loss %.4f lr %.6g grad_accum %d hits %d/%d"
              % (epoch + 1, config.epochs, mean, lr_now, epoch_grad_accum, hits, lookups), flush=True)
        grad_norm_mean = ((grad_norm_sum / grad_steps).item() if grad_steps else None)
        # ── per-epoch dev evaluation (rank 0) + broadcast ───────────────
        dev = None
        if dev_enabled:
            # Every rank flips to eval/train together (DDP mode consistency);
            # only rank 0 runs the forward.
            model.eval()
            if is_rank0():
                with profiler.phase("dev_eval"):
                    dev = DevEvaluator.metrics(
                        laya_train, model, tok, dev_items, device,
                        max_len, head_max_len, parallel, control,
                        annotate=profiler.phase)
            payload = dev.payload() if dev is not None else [0.0, 0.0, 0.0,
                                                             -1.0, -1.0]
            has_dev, dev_acc, dev_loss, dev_abstain, dev_cov = DistributedBroadcast.values(
                torch, payload, device)
            dev = DevReport.from_payload(has_dev, dev_acc, dev_loss,
                                         dev_abstain, dev_cov)
            if dev is not None and is_rank0():
                print("epoch %d/%d dev_acc=%.4f dev_loss=%.4f"
                      % (epoch + 1, config.epochs, dev.accuracy, dev.loss),
                      flush=True)
            # restore training mode (and the frozen-encoder eval that train()
            # would otherwise undo) on every rank.
            model.train()
            if config.freeze_encoder or gradual:
                model.encoder.eval()
        # ── best tracking + early stop + plateau + checkpointing ────────
        extra = {"epoch": epoch + 1, "train/lr": lr_now,
                 "epoch_time_s": epoch_time}
        if grad_norm_mean is not None:
            extra["train/grad_norm"] = grad_norm_mean
        stop_flag = False
        if dev is not None:
            extra.update(dev.to_wandb("dev"))
            metric_value = (dev.loss if lower_is_better else dev.accuracy)
            step = stop_policy.update(metric_value, best, bad_epochs)
            best, bad_epochs = step.best, step.bad_epochs
            # keep_best tracks the EARLY-STOP optimum (not raw accuracy): the
            # saved/deployed weights are the same epoch selection picked.
            if step.improved:
                best_acc = dev.accuracy
                if control.get("keep_best") and is_rank0():
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in
                                  ControlCheckpointer.unwrap(model).state_dict().items()}
                    checkpointer.save_best(model, best, best_acc, epoch)
            stop_flag = step.stop and bool(control.get("early_stop"))
            if plateau:
                scheduler.step(metric_value)
        # EVERY rank broadcasts the stop decision every epoch (fixed shape),
        # even when dev is disabled on every rank; skipping it on some ranks
        # would desync the post-training staging barrier.
        stop_flag = bool(DistributedBroadcast.values(
            torch, [1.0 if stop_flag else 0.0], device)[0] >= 0.5)
        extra["select/best_dev_accuracy"] = best_acc
        extra["select/bad_epochs"] = bad_epochs
        wandb_log_epoch(epoch, mean, extra)
        profiler.flush_epoch(epoch + 1)
        with profiler.phase("checkpoint_save"):
            if on_epoch_end is not None:
                on_epoch_end(epoch, mean)
            if control.get("save_each_epoch"):
                checkpointer.save(model, optimizer, scheduler, epoch, best,
                                  bad_epochs)
        if swa_state is not None and epoch >= swa_start:
            with torch.no_grad():
                for p in params:
                    swa_state[id(p)].add_(p.detach())
                swa_count += 1
        if stop_flag:
            stopped = True
            stopped_epoch = epoch + 1
            print("[perf-patch] early stop at epoch %d (%s=%s best=%s)"
                  % (epoch + 1, early_metric, metric_value, best), flush=True)
            break
    profiler.close()
    # ── end-of-training weight selection (rank 0 only) ──────────────────
    if is_rank0():
        if control.get("keep_best") and best_state is not None:
            ControlCheckpointer.unwrap(model).load_state_dict(best_state)
            print("[perf-patch] restored best dev_accuracy=%.4f" % best_acc,
                  flush=True)
        elif swa_state is not None and swa_count > 0:
            with torch.no_grad():
                for p in params:
                    p.copy_(swa_state[id(p)] / float(swa_count))
            print("[perf-patch] restored SWA average (%d epochs)" % swa_count,
                  flush=True)
        elif ema_state is not None:
            with torch.no_grad():
                for p in params:
                    p.copy_(ema_state[id(p)])
            print("[perf-patch] restored EMA weights", flush=True)
    result = {
        "epochs_run": len(history),
        "stopped": bool(stopped),
        "stopped_epoch": stopped_epoch,
        "best_dev_accuracy": best_acc,
        "best_metric": best,
        "bad_epochs": bad_epochs,
        "scheduler": scheduler_kind,
        "warmup_steps": warmup,
        "history": [float(value) for value in history],
    }
    if dev_items and is_rank0() and (
            control.get("write_error_artifacts")
            or control.get("temperature_scale")):
        try:
            model.eval()
            result["error_artifacts"] = DevEvaluator.artifacts(
                laya_train, model, tok, dev_items, device, max_len,
                head_max_len, parallel, control,
                annotate=profiler.phase)
        except Exception as error:
            result["error_artifacts_error"] = (
                type(error).__name__ + ": " + str(error)[:200])
    globals()["FINETUNE_CONTROL_RESULT"] = result
    try:
        wandb_log_control_summary(result)
    except Exception:
        pass
    wandb_log_event("training_done", epochs_run=len(history),
                    best_dev_accuracy=best_acc)
    model.eval()
    return history


def apply_perf_patch():
    # Apply BEFORE the device patch so the device wrapper closes over (and
    # preserves) this loop; opt out with ER_LAYA_PERF_PATCH=0.
    if not perf_patch_enabled():
        print("[perf-patch] disabled via " + PERF_PATCH_ENV, flush=True)
        return False
    from laya import train as laya_train
    globals()["PHASE_TIMINGS"] = CalibrationTimingHook.install_for_rank(
        laya_train, is_rank0(), sink=WandbTimingSink.log)
    laya_train.train_model = _perf_train_model
    print("[perf-patch] laya.train.train_model patched: on-GPU loss (1 sync/"
          "epoch), single device move, encode memoization", flush=True)
    return True


def start_gpu_sampler():
    """Start the ONE 1 Hz GPU poller: append gpu_usage.log AND mirror to wandb.

    The existing sampler loop is reused (same 1 Hz cadence, same log file) but
    runs in-process so it reaches the live wandb run; there is still exactly one
    poller. A wandb write failure is fail-soft and never stops the samples.
    """
    if not perf_patch_enabled():
        return None
    if shutil.which("nvidia-smi") is None:
        log("gpu sampler: nvidia-smi absent; skipping")
        return None
    import threading

    path = WORKING / "gpu_usage.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    thread = threading.Thread(target=_gpu_sampler_loop, args=(path, stop_event),
                              daemon=True)
    thread.start()
    log("gpu sampler: 1 Hz -> " + str(path))
    return thread, stop_event


def _gpu_sampler_loop(path, stop_event):
    command = ["nvidia-smi",
               "--query-gpu=utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader"]
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        while not stop_event.wait(1.0):
            try:
                output = subprocess.run(command, capture_output=True, text=True,
                                        check=False, timeout=5).stdout
            except Exception as error:
                print("[gpu-sampler] query skipped: " + type(error).__name__,
                      flush=True)
                continue
            for line in output.splitlines():
                handle.write(line.strip() + "\\n")
                _wandb_log_gpu(line)


def _wandb_log_gpu(line):
    if WANDB_RUN is None:
        return
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return
    try:
        util = float(parts[0].rstrip("%").strip())
        mem_used = float(parts[1].split()[0])
    except (ValueError, IndexError):
        return
    try:
        WANDB_RUN.log({"gpu/util_pct": util, "gpu/mem_used_mb": mem_used})
    except Exception as error:
        print("[gpu-sampler] wandb skipped: " + type(error).__name__ + ": "
              + str(error)[:200], flush=True)


def stop_gpu_sampler(handle):
    if not handle:
        return
    thread, stop_event = handle
    stop_event.set()
    thread.join(timeout=5)


def summarize_gpu_usage(path):
    if not path.is_file():
        return None
    utils, mems, total_mb = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util = float(parts[0].rstrip("%").strip())
            used = float(parts[1].split()[0])
            total = float(parts[2].split()[0])
        except (ValueError, IndexError):
            continue
        utils.append(util)
        mems.append(used)
        total_mb = total
    if not utils:
        return None
    return {
        "samples": len(utils),
        "util_min_pct": min(utils),
        "util_max_pct": max(utils),
        "util_mean_pct": sum(utils) / len(utils),
        "mem_used_peak_mb": max(mems),
        "mem_total_mb": total_mb,
    }
'''

FINETUNE_PERF_PATCH_SOURCE = (
    DDP_PATCH_TEMPLATE + LOOP_PATCH_TEMPLATE).replace(
    "@CONTROL_LOGIC@", FINETUNE_CONTROL_LOGIC_SOURCE)
