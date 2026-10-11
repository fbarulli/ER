"""Training-loop half of the fine-tune PERF_PATCH kernel text.

Concatenated after `laya_train_patch_ddp.DDP_PATCH_TEMPLATE` and joined with
the control logic to rebuild ``FINETUNE_PERF_PATCH_SOURCE`` byte-for-byte.
"""
from __future__ import annotations

from cli.laya_train_patch_checkpoint import CHECKPOINTER_TEMPLATE
from cli.laya_train_patch_ddp import (
    DDP_PATCH_TEMPLATE,
    FINETUNE_CONTROL_LOGIC_SOURCE,
)

LOOP_PATCH_TEMPLATE = CHECKPOINTER_TEMPLATE + '''\
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
        except Exception as error:
            print("[perf-patch] profiler wandb skipped: "
                  + type(error).__name__ + ": " + str(error) + "\\n"
                  + traceback.format_exc(), flush=True)


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
                  + type(error).__name__ + ": " + str(error) + "\\n"
                  + traceback.format_exc(), flush=True)


def _perf_train_model(model, tok, items, config, device, max_len, head_max_len,
                      on_epoch_end=None, parallel=False):
    # Faithful copy of laya.train.train_model (0.3.29) carrying the three
    # recipe-neutral perf changes AND the YAML-driven training controls read
    # from the baked FINETUNE_CONTROL block (never a TrainConfig kwarg).
    import sys
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
    # torch.compile AFTER the DDP wrap so DDPOptimizer sees the DDP module
    # (the documented order); in the single-process path it compiles the raw
    # model. Fail-soft to eager inside ModelCompiler (offenders #2/#4).
    if control.get("compile_model"):
        model = ModelCompiler.compile(torch, model, control.get("compile_mode"))
    epoch_len = len(ddp_sampler) if ddp_sampler is not None else len(items)
    steps_per_epoch = math.ceil(epoch_len / config.micro_batch)
    # The schedule length MUST equal the ACTUAL optimizer-step count. With
    # batch_size_ramp the per-epoch grad_accum (hence the step count) is below
    # the target on the ramp epochs, so a length computed from the target makes
    # OneCycleLR raise "Tried to step N+1 times. The specified number of total
    # steps is N" on the first ramped epoch. Sum the real per-epoch steps.
    ramp_on = bool(control.get("batch_size_ramp"))
    target_grad_accum = int(config.grad_accum)
    epoch_grad_accums = [
        (BatchRamp.grad_accum(epoch, target_grad_accum,
                              control.get("batch_ramp_start_frac"),
                              control.get("batch_ramp_epochs"))
         if ramp_on else target_grad_accum)
        for epoch in range(config.epochs)]
    updates = max(1, sum(math.ceil(steps_per_epoch / grad_accum)
                         for grad_accum in epoch_grad_accums))
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
    memo = globals().get("ENCODE_MEMO") or EncodeMemo()
    collate_memo = globals().get("COLLATE_MEMO") or CollateMemo()
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
    pin_memory_on = control.get("pin_memory", True)
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
    best_dev_loss = None
    best_state = (best_record or {}).get("model")
    stopped = False
    stopped_epoch = None
    profiler = ProfilerSession.for_training(
        torch, control, device, is_rank0(),
        globals().get("FINETUNE_OUTPUT_DIR"), on_metrics=WandbProfileSink.log,
        timings=globals().get("PHASE_TIMINGS"))
    if profiler.enabled:
        profiler.start()

    # rank-0 progress bars (epochs + steps within an epoch): stderr carries
    # the CR frames to the console capture and the tee mirrors them into the
    # live wandb log channel, so the bar reaches the lane log either way.
    progress = LaneProgress(is_rank0(), LOG_SINK, sys.stderr)
    epoch_bar = progress.bar(range(start_epoch, config.epochs),
                             desc="epochs", unit="epoch",
                             initial=start_epoch, total=config.epochs)
    for epoch in epoch_bar:
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
        # the per-epoch value is the one the schedule length was summed from.
        epoch_grad_accum = epoch_grad_accums[epoch]
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
        for start in progress.bar(range(0, len(epoch_items),
                                        config.micro_batch),
                                  desc="epoch %d/%d" % (epoch + 1,
                                                        config.epochs),
                                  unit="batch", leave=False):
            chunk_items = epoch_items[start:start + config.micro_batch]
            chunk = []
            with profiler.phase("data.encode"):
                for it in chunk_items:
                    order = laya_train.draw_option_order(
                        it, order_rng, config.shuffle_options)
                    chunk.append(laya_train.encode_item(
                        tok, it, max_len, head_max_len, order, parallel))
            with profiler.phase("collate+batch_move"):
                batch = laya_train.collate_items([chunk], tok.pad_token_id)
                # dynamic padding: laya's collate already pads to the batch
                # longest (max_len only TRUNCATES); this rounds the collated
                # batch length to pad_to_multiple.
                if dynamic_padding_on:
                    batch = DynamicPadder.pad(torch, batch, pad_to_multiple)
                # (2) ONE pinned non_blocking move of the WHOLE batch (offender
                # #6: blocking pageable H2D serialized the host against the
                # stream — 326 s/epoch in the smoke harvest, GPU starved to
                # ~40%). The loop's reads below and _forward's own .to(device)
                # are then no-ops, so nothing is copied twice.
                batch = BatchMover.to_device(torch, batch, device,
                                             pin_memory_on)
                mask = batch["marker_mask"]
                target = batch["target"]
                qtype = batch["qtype"]
            update_due = ((n_steps + 1) % epoch_grad_accum == 0
                          or start + config.micro_batch >= len(epoch_items))
            # Adversarial perturbations consume synchronized gradients.
            synchronize = (ddp_sampler is None or update_due
                           or bool(adv_eps and float(adv_eps) > 0.0))
            with DeterministicDdp.accumulate(model, synchronize):
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
            if update_due:
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
              % (epoch + 1, config.epochs, mean, lr_now, epoch_grad_accum,
                 memo.hits, memo.lookups), flush=True)
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
                        collate_memo)
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
                best_dev_loss = dev.loss
                # keep_best restores the early-stop optimum into the FINAL
                # model; the best.pt FILE only serves a resume, so a
                # configs+results run (save_each_epoch=False) skips it.
                if control.get("keep_best") and is_rank0():
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in
                                  ControlCheckpointer.unwrap(model).state_dict().items()}
                    if control.get("save_each_epoch"):
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
            # save_each_epoch owns EVERY per-epoch artifact: the torch resume
            # checkpoint (model+optimizer+scheduler) AND the caller's HF
            # snapshot. A configs+results run writes neither.
            if control.get("save_each_epoch"):
                if on_epoch_end is not None:
                    on_epoch_end(epoch, mean)
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
    _close_bar = getattr(epoch_bar, "close", None)
    if _close_bar is not None:
        _close_bar()
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
        "best_dev_loss": best_dev_loss,
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
                head_max_len, parallel, control, collate_memo)
        except Exception as error:
            result["error_artifacts_error"] = (
                type(error).__name__ + ": " + str(error)[:200])
    globals()["FINETUNE_CONTROL_RESULT"] = result
    try:
        wandb_log_control_summary(result)
    except Exception as error:
        print("[perf-patch] control-summary wandb skipped: "
              + type(error).__name__ + ": " + str(error) + "\\n"
              + traceback.format_exc(), flush=True)
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
    control = TrainingControls.parse(globals().get("FINETUNE_CONTROL"), {}).block
    policy = globals().setdefault("EVAL_RECORDS_POLICY", EvalRecordsPolicy())
    EvalRecordsHook.install(laya_train, policy.update(control))
    EncodeMemo.install(laya_train,
                       globals().setdefault("ENCODE_MEMO", EncodeMemo()).reset())
    CollateMemo.install(
        laya_train,
        globals().setdefault("COLLATE_MEMO", CollateMemo()).reset())
    laya_train.train_model = _perf_train_model
    print("[perf-patch] laya.train.train_model patched: on-GPU loss (1 sync/"
          "epoch), pinned non_blocking batch move, encode + dev-collate "
          "memoization; eval records amp=%s batch=%d"
          % (policy.eval_amp, policy.eval_batch), flush=True)
    return True


def start_gpu_sampler(path=None, devices=None):
    """Start the ONE 1 Hz GPU poller: append a log AND mirror to wandb.

    The existing sampler loop is reused (same 1 Hz cadence) but runs in-process
    so it reaches the live wandb run; there is still exactly one poller per
    process. ``path`` defaults to the session-wide ``gpu_usage.log`` (the
    finetune kernel's single rank-0 sampler); a MULTI-worker lane (HPO) passes a
    PER-WORKER path so parallel workers never clobber one file, and its
    ``devices`` (its own ``CUDA_VISIBLE_DEVICES``) so ``nvidia-smi -i`` reports
    ONLY the GPU that worker owns — each worker records its OWN gpu/util_pct. A
    wandb write failure is fail-soft and never stops the samples.
    """
    if not perf_patch_enabled():
        return None
    if shutil.which("nvidia-smi") is None:
        log("gpu sampler: nvidia-smi absent; skipping")
        return None
    import threading

    path = Path(path) if path else (WORKING / "gpu_usage.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    thread = threading.Thread(target=_gpu_sampler_loop,
                              args=(path, stop_event, devices), daemon=True)
    thread.start()
    log("gpu sampler: 1 Hz -> " + str(path)
        + (" (nvidia-smi -i " + devices + ")" if devices else ""))
    return thread, stop_event


def _gpu_sampler_loop(path, stop_event, devices=None):
    command = ["nvidia-smi"]
    if devices:
        # Pin the query to the GPU(s) THIS process owns (CUDA_VISIBLE_DEVICES);
        # nvidia-smi ignores that env var, so without -i every parallel worker
        # would log every card and gpu/util_pct could not be attributed.
        command += ["-i", devices]
    command += ["--query-gpu=utilization.gpu,memory.used,memory.total",
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
              + str(error) + "\\n" + traceback.format_exc(), flush=True)


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


# The DEFAULT-ON held-out gate (owner order), declared ONCE for both kernels.
HELD_OUT_ENV = "ER_LAYA_HELD_OUT"


def held_out_enabled():
    """DEFAULT-ON held-out validation; opt out with ER_LAYA_HELD_OUT=0."""
    return os.environ.get(HELD_OUT_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def evaluate_held_out(test_path, checkpoint, device, split="test"):
    """Score a trained checkpoint on the HELD-OUT split (both kernels).

    DEFAULT-ON (owner order, gate: held_out_enabled): the training-time eval is
    the dev split and overlaps training/calibration, so this is the run's
    honest generalization number. The raw (uncalibrated) metrics are reported —
    the calibration was already fitted on the calibration split during
    training. An empty/unusable split raises RuntimeError (NOT SystemExit):
    the callers' fail-soft handlers catch Exception, so a held-out failure is
    recorded with its reason and NEVER discards the checkpoint/receipt.

    ``split`` names the split being scored ("test" on the full corpus; the
    50/50 HPO carve passes "dev" — its zero-leak validate half — because its
    test split is empty by construction).
    """
    import torch
    from laya import train as laya_train

    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(test_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise RuntimeError(
            "held-out split " + str(Path(test_path).name) + " produced no "
            "usable items (skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=int(globals().get("HELD_OUT_BATCH", 8)), parallel=parallel)
    return {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": str(Path(test_path).name),
        "eval_split": str(split),
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "metrics": laya_train.evaluate_records(records),
    }
'''

FINETUNE_PERF_PATCH_SOURCE = (
    DDP_PATCH_TEMPLATE + LOOP_PATCH_TEMPLATE).replace(
    "@CONTROL_LOGIC@", FINETUNE_CONTROL_LOGIC_SOURCE)
