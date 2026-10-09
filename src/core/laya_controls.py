"""src/core/laya_controls.py — cohesive, GPU-free classes for the laya controls.

The fine-tune kernel runs from ATTACHED datasets (it never clones the repo), so
it cannot import this package: ``cli.laya_lane`` bakes these exact CLASS sources
into the staged perf patch via ``inspect.getsource``. That keeps ONE source of
truth — the classes below are unit-tested here with plain Python (no CUDA) and
executed byte-for-byte inside the remote kernel.

Every dial VALUE comes from the baked control block (built from
``core.laya_config.FinetuneSpec``); this module never repeats a spec default.
Heavy dependencies (torch) are imported lazily inside the methods that need
them so the module stays importable without a GPU.
"""

from __future__ import annotations

import math
import os
from typing import Any


class ControlBlock:
    """A parsed/merged control block with defensive typed access.

    The staged kernel bakes the FULL block (built from ``FinetuneSpec``), so a
    missing key only ever appears in an offline unit test: absent keys read as
    ``None`` (feature off), never as a duplicated spec default.
    """

    def __init__(self, raw=None, defaults=None):
        self._values = dict(defaults or {})
        if isinstance(raw, dict):
            for key, value in raw.items():
                if value is not None:
                    self._values[key] = value

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __getitem__(self, key):
        return self._values[key]

    def __contains__(self, key):
        return key in self._values

    def as_dict(self):
        return dict(self._values)


class EarlyStopStep:
    """The result of one early-stop patience step (a value object)."""

    def __init__(self, best, bad_epochs, improved, stop):
        self.best = best
        self.bad_epochs = bad_epochs
        self.improved = improved
        self.stop = stop


class EarlyStopPolicy:
    """Patience/min-delta early stopping (direction-aware, pure).

    No dial default is restated here: `from_control` reads every value from
    the baked block (built from ``FinetuneSpec``); direct construction requires
    the caller to pass them.
    """

    def __init__(self, patience, min_delta, lower_is_better):
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta)
        self.lower_is_better = bool(lower_is_better)

    @classmethod
    def from_control(cls, control):
        return cls(
            patience=control["early_stop_patience"],
            min_delta=control["early_stop_min_delta"],
            lower_is_better=control["early_stop_metric"] == "dev_loss")

    def is_improvement(self, value, best):
        if best is None:
            return True
        if self.lower_is_better:
            return value < best - self.min_delta
        return value > best + self.min_delta

    def update(self, value, best, bad_epochs):
        if self.is_improvement(value, best):
            return EarlyStopStep(value, 0, True, False)
        bad = int(bad_epochs) + 1
        stop = self.patience is not None and bad >= self.patience
        return EarlyStopStep(best, bad, False, stop)


class LrSchedulerFactory:
    """Builds the per-update LR scheduler from an explicit control block.

    ``kind == "cosine"`` with zero warmup reproduces the landed
    ``CosineAnnealingLR(optimizer, T_max=updates, eta_min=min_lr)`` exactly.
    ``plateau`` is stepped per EPOCH by the caller, never per update.
    """

    def __init__(self, optimizer, kind, total_updates, min_lr, warmup,
                 plateau_mode, plateau_factor, plateau_patience,
                 onecycle_pct_start):
        self.optimizer = optimizer
        self.kind = str(kind).lower()
        self.total_updates = max(1, int(total_updates))
        self.min_lr = float(min_lr)
        self.warmup = max(0, int(warmup or 0))
        self.plateau_mode = plateau_mode
        self.plateau_factor = float(plateau_factor)
        self.plateau_patience = int(plateau_patience)
        self.onecycle_pct_start = float(onecycle_pct_start)
        self.is_plateau = self.kind == "plateau"
        if self.is_plateau and self.warmup > 0:
            raise ValueError(
                "lr_scheduler='plateau' does not support warmup (it steps on "
                "the per-epoch dev metric); set warmup_steps/warmup_frac to 0")

    @classmethod
    def from_control(cls, control, optimizer, total_updates, min_lr,
                     lower_is_better=False):
        warmup = cls.effective_warmup_steps(
            control["warmup_steps"], control["warmup_frac"], total_updates)
        return cls(
            optimizer=optimizer,
            kind=control["lr_scheduler"],
            total_updates=total_updates,
            min_lr=min_lr,
            warmup=warmup,
            plateau_mode="min" if lower_is_better else "max",
            plateau_factor=control["plateau_factor"],
            plateau_patience=control["plateau_patience"],
            onecycle_pct_start=control["onecycle_pct_start"])

    @staticmethod
    def effective_warmup_steps(warmup_steps, warmup_frac, total_updates):
        total = max(1, int(total_updates))
        if warmup_steps and int(warmup_steps) > 0:
            return max(0, min(int(warmup_steps), total - 1))
        if warmup_frac and float(warmup_frac) > 0.0:
            return max(0, min(int(round(float(warmup_frac) * total)),
                              total - 1))
        return 0

    def build(self):
        import torch

        total, warmup = self.total_updates, self.warmup
        if self.kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.plateau_mode,
                factor=float(self.plateau_factor),
                patience=int(self.plateau_patience),
                min_lr=self.min_lr)
        if self.kind == "onecycle":
            base = [float(group.get("lr", self.min_lr))
                    for group in self.optimizer.param_groups]
            pct = (float(warmup) / float(total)) if warmup > 0 \
                else float(self.onecycle_pct_start)
            pct = min(max(pct, 1e-8), 0.9)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=base, total_steps=total, pct_start=pct)
        main_steps = max(1, total - warmup) if warmup > 0 else total
        main = self._build_main(main_steps)
        if warmup > 0:
            warm = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-8, end_factor=1.0,
                total_iters=warmup)
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warm, main], milestones=[warmup])
        return main

    def _build_main(self, main_steps):
        import torch

        if self.kind == "constant":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda _step: 1.0)
        if self.kind == "linear":
            def _decay(step):
                if main_steps <= 1:
                    return 1.0
                return max(0.0,
                           1.0 - float(step) / float(main_steps - 1))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, _decay)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=main_steps, eta_min=self.min_lr)


class MetricFlattener:
    """Flattens a metric mapping into ``prefix/key`` entries, dropping None."""

    @staticmethod
    def flatten(prefix, mapping):
        flat = {}
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                if value is None:
                    continue
                flat[str(prefix) + "/" + str(key)] = value
        return flat


class AbstainCoverage:
    """Best-effort abstain-rate / coverage from calibration records.

    ``laya.train.evaluate_records`` emits neither key, so the lane derives them
    from the records it already has: a record abstains when its top-1 softmax
    probability falls below the configured confidence threshold.
    """

    @staticmethod
    def estimate(records, confidence_threshold):
        if confidence_threshold is None:
            return None, None
        total = abstained = 0
        for record in records or ():
            try:
                _qtype, logits, _target, k = record
                z = [float(value) for value in list(logits)[:int(k)]]
                top = max(z)
                exps = [math.exp(value - top) for value in z]
                conf = max(exps) / sum(exps)
            except Exception:
                continue
            total += 1
            if conf < float(confidence_threshold):
                abstained += 1
        if total == 0:
            return None, None
        rate = abstained / float(total)
        return rate, 1.0 - rate


class DevReport:
    """The per-epoch dev metrics value object (rank-0 + broadcast payloads)."""

    def __init__(self, accuracy, loss, abstain_rate=None, coverage=None,
                 ece=None, brier=None, brier_top1=None, mean_confidence=None):
        self.accuracy = float(accuracy)
        self.loss = float(loss)
        self.abstain_rate = abstain_rate
        self.coverage = coverage
        self.ece = ece
        self.brier = brier
        self.brier_top1 = brier_top1
        self.mean_confidence = mean_confidence

    @classmethod
    def from_metrics(cls, metrics, records=None, confidence_threshold=None):
        rate, coverage = AbstainCoverage.estimate(records, confidence_threshold)
        return cls(
            accuracy=float(metrics.get("accuracy") or 0.0),
            loss=float(metrics.get("loss") or 0.0),
            abstain_rate=rate,
            coverage=coverage,
            ece=metrics.get("ece"),
            brier=metrics.get("brier"),
            brier_top1=metrics.get("brier_top1"),
            mean_confidence=metrics.get("mean_confidence"))

    @classmethod
    def from_payload(cls, has_dev, accuracy, loss, abstain_rate, coverage):
        if has_dev < 0.5:
            return None
        return cls(accuracy=accuracy, loss=loss,
                   abstain_rate=None if abstain_rate < 0 else abstain_rate,
                   coverage=None if coverage < 0 else coverage)

    def payload(self):
        return [1.0, self.accuracy, self.loss,
                -1.0 if self.abstain_rate is None else self.abstain_rate,
                -1.0 if self.coverage is None else self.coverage]

    def to_wandb(self, prefix="dev"):
        return MetricFlattener.flatten(prefix, {
            "accuracy": self.accuracy, "loss": self.loss,
            "abstain_rate": self.abstain_rate, "coverage": self.coverage,
            "ece": self.ece, "brier": self.brier,
            "brier_top1": self.brier_top1,
            "mean_confidence": self.mean_confidence})


class TrainingControls:
    """Facade over a parsed control block: typed dials + policy/factory access.

    The block is the YAML-driven ``FinetuneSpec`` surface (baked as
    ``FINETUNE_CONTROL``); this class never restates a dial default.
    """

    def __init__(self, block):
        self.block = block

    @classmethod
    def parse(cls, raw, defaults=None):
        return cls(ControlBlock(raw, defaults))

    def get(self, key, default=None):
        return self.block.get(key, default)

    def flag(self, key):
        return bool(self.block.get(key))

    def early_stop_policy(self):
        return EarlyStopPolicy.from_control(self.block)

    def scheduler(self, optimizer, total_updates, min_lr):
        policy = self.early_stop_policy()
        return LrSchedulerFactory.from_control(
            self.block, optimizer, total_updates, min_lr,
            lower_is_better=policy.lower_is_better)


class ParamGroupBuilder:
    """Splits optimizer groups into decay / no-decay (bias + norm, ndim<=1).

    Composes with (never replaces) ``weight_decay``: when disabled the groups
    pass through untouched and the optimizer's global weight_decay applies.
    """

    @staticmethod
    def is_no_decay(name, param):
        return param.ndim <= 1 or str(name).endswith("bias")

    @staticmethod
    def apply(model, groups, enabled):
        if not enabled:
            return groups
        named = {id(param): name for name, param in model.named_parameters()}
        decay_groups, no_decay_groups = [], []
        for group in groups:
            decay, no_decay = [], []
            for param in group["params"]:
                target = no_decay if ParamGroupBuilder.is_no_decay(
                    named.get(id(param), ""), param) else decay
                target.append(param)
            base = {key: value for key, value in group.items()
                    if key != "params"}
            if decay:
                decay_groups.append({**base, "params": decay})
            if no_decay:
                no_decay_groups.append({**base, "params": no_decay,
                                        "weight_decay": 0.0})
        return decay_groups + no_decay_groups


class OptimizerStateCaster:
    """Best-effort bf16 optimizer states (composes with AMP forward dtype)."""

    @staticmethod
    def apply(torch_module, optimizer, dtype_name):
        if str(dtype_name or "fp32").lower() != "bf16":
            return False
        target = torch_module.bfloat16
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if (torch_module.is_tensor(value)
                        and value.is_floating_point()
                        and value.dtype != target):
                    state[key] = value.to(target)
        return True


class LrScaler:
    """Scale the peak LR from the effective batch (explicit LRs win at none)."""

    @staticmethod
    def factor(effective_batch, base_batch, rule):
        if str(rule or "none").lower() == "none" or not base_batch:
            return 1.0
        ratio = float(effective_batch) / float(base_batch)
        if str(rule).lower() == "linear":
            return ratio
        if str(rule).lower() == "sqrt":
            return math.sqrt(ratio)
        return 1.0

    @staticmethod
    def effective_batch(micro_batch, grad_accum, world_size):
        return max(1, int(micro_batch)) * max(1, int(grad_accum)) * max(
            1, int(world_size))

    @staticmethod
    def apply(optimizer, factor):
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * factor
        return factor


class BatchRamp:
    """Linear micro-batch/grad-accum ramp (grad_accum here: rank-symmetric)."""

    @staticmethod
    def grad_accum(epoch, target, start_frac, ramp_epochs):
        target = max(1, int(target))
        if not ramp_epochs or ramp_epochs <= 0 or epoch >= ramp_epochs:
            return target
        start = max(1, int(round(target * float(start_frac))))
        frac = float(epoch + 1) / float(ramp_epochs)
        return max(1, int(round(start + (target - start) * frac)))


class LossWeightSchedule:
    """ONE schedule for sigma / w_sph / w_rps / contrastive margin.

    ``mode == "laya"`` reproduces ``laya.train.sigma_at`` exactly (linear sigma,
    constant weights) so a default run is byte-identical; ``linear``/``cosine``
    interpolate every term with the same curve (never double-scheduled).
    """

    def __init__(self, mode, epochs, sigma_start, sigma_end, w_sph, w_sph_end,
                 w_rps, w_rps_end, margin, margin_end):
        self.mode = str(mode or "laya").lower()
        self.epochs = max(1, int(epochs))
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.w_sph = float(w_sph)
        self.w_sph_end = w_sph_end
        self.w_rps = float(w_rps)
        self.w_rps_end = w_rps_end
        self.margin = float(margin or 0.0)
        self.margin_end = margin_end

    @classmethod
    def from_control(cls, control, config):
        # The START values (sigma/w_sph/w_rps) are TRAINCONFIG fields; only the
        # new end values + margin live in the control block. No value is baked
        # twice.
        return cls(
            mode=control["loss_schedule"], epochs=config.epochs,
            sigma_start=config.sigma_start, sigma_end=config.sigma_end,
            w_sph=config.w_sph, w_sph_end=control.get("w_sph_end"),
            w_rps=config.w_rps, w_rps_end=control.get("w_rps_end"),
            margin=control["contrastive_margin"],
            margin_end=control.get("contrastive_margin_end"))

    @staticmethod
    def _curve(frac, mode):
        if mode == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * frac))
        return frac

    @staticmethod
    def _interp(start, end, curve):
        if end is None:
            return start
        return start + (float(end) - start) * curve

    def at(self, epoch):
        frac = float(epoch) / float(max(1, self.epochs - 1))
        curve = self._curve(frac, "linear" if self.mode == "laya" else self.mode)
        return {
            "sigma": self.sigma_start + (self.sigma_end - self.sigma_start) * curve,
            "w_sph": self._interp(self.w_sph, self.w_sph_end, curve),
            "w_rps": self._interp(self.w_rps, self.w_rps_end, curve),
            "margin": self._interp(self.margin, self.margin_end, curve),
        }


class RDrop:
    """R-Drop: two dropout-masked forwards + symmetric KL (dropout-gated)."""

    @staticmethod
    def available(model):
        import torch

        for module in model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p > 0:
                return True
        return False

    @staticmethod
    def kl(torch_module, logits_one, logits_two, mask):
        fill = -1e9
        logp = torch_module.log_softmax(logits_one.masked_fill(~mask, fill), -1)
        logq = torch_module.log_softmax(logits_two.masked_fill(~mask, fill), -1)
        p, q = logp.exp(), logq.exp()
        sym = 0.5 * ((p * (logp - logq)).sum(-1)
                     + (q * (logq - logp)).sum(-1))
        return sym.mean()

    @staticmethod
    def combine(loss_one, loss_two, kl, alpha):
        return 0.5 * (loss_one + loss_two) + float(alpha) * kl


class DropPath:
    """Stochastic depth over transformer blocks (norm-first residual form)."""

    @staticmethod
    def rate(epoch, epochs, base_rate, schedule):
        base = float(base_rate or 0.0)
        if base <= 0.0:
            return 0.0
        if str(schedule).lower() == "linear" and epochs > 1:
            return base * float(epoch + 1) / float(epochs)
        return base

    @staticmethod
    def apply(model, rate, blocks=None):
        import torch

        rate = float(rate or 0.0)
        if rate <= 0.0:
            return 0
        targets = blocks
        if targets is None:
            head = getattr(model, "head", None)
            targets = list(getattr(head, "layers", []) or [])
        wrapped = 0
        for module in targets:
            if getattr(module, "_laya_drop_path", False):
                module.drop_path_rate = rate
                continue
            original = module.forward

            def forward_with_drop_path(*args, _original=original, **kwargs):
                x = args[0] if args else None
                out = _original(*args, **kwargs)
                current = float(getattr(module, "drop_path_rate", 0.0))
                if (current > 0.0 and module.training and x is not None
                        and torch.is_tensor(out) and out.shape == x.shape):
                    keep = 1.0 - current
                    shape = [1] * out.dim()
                    shape[0] = out.shape[0]
                    mask = (torch.rand(shape, device=out.device) < keep).to(out.dtype)
                    out = x + (out - x) * mask / max(keep, 1e-6)
                return out

            module.forward = forward_with_drop_path
            module._laya_drop_path = True
            module.drop_path_rate = rate
            wrapped += 1
        return wrapped


class DynamicPadder:
    """Rounds a collated batch's token length up to ``pad_to_multiple``."""

    @staticmethod
    def pad(torch_module, batch, multiple):
        multiple = int(multiple or 0)
        if multiple <= 1 or batch is None:
            return batch
        length = int(batch["input_ids"].shape[1])
        target = ((length + multiple - 1) // multiple) * multiple
        if target == length:
            return batch
        pad = target - length
        import torch.nn.functional as functional

        def _grow(tensor, value):
            return functional.pad(tensor, (0, pad), value=value)

        result = dict(batch)
        result["input_ids"] = _grow(batch["input_ids"], 0)
        result["attention_mask"] = _grow(batch["attention_mask"], 0)
        for key in ("position_ids", "option_ids"):
            if key in batch:
                result[key] = _grow(batch[key], 0)
        return result


class ProfilerSession:
    """A bounded, fail-soft ``torch.profiler`` wrapper (rank 0 + CUDA only).

    Writes one chrome trace per profiled epoch under ``<output_dir>/<dir>`` and
    feeds the top-ops table (by CUDA time) to an optional ``on_metrics`` sink.
    """

    def __init__(self, torch_module, *, enabled, trace_dir, schedule,
                 on_metrics=None, top_n=15):
        self._torch = torch_module
        self.enabled = bool(enabled)
        self._trace_dir = trace_dir
        self._schedule = schedule
        self._on_metrics = on_metrics
        self._top_n = int(top_n)
        self._profiler = None
        self.epoch = 0

    @staticmethod
    def enabled_for(control, device, rank0):
        if not control.get("profile"):
            return False
        if getattr(device, "type", "cpu") != "cuda":
            return False
        return bool(rank0)

    @classmethod
    def for_training(cls, torch_module, control, device, rank0, output_dir,
                     on_metrics=None):
        enabled = cls.enabled_for(control, device, rank0)
        profile_dir = control.get("profile_dir")
        trace_dir = None
        if enabled and output_dir and profile_dir:
            trace_dir = os.path.join(str(output_dir), str(profile_dir))
        return cls(torch_module, enabled=enabled, trace_dir=trace_dir,
                   schedule=control.get("profile_schedule"),
                   on_metrics=on_metrics)

    def start(self):
        if not self.enabled or self._profiler is not None:
            return False
        schedule = self._schedule or {}
        try:
            torch = self._torch
            activities = [torch.profiler.ProfilerActivity.CPU]
            if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            handle = torch.profiler.schedule(
                wait=int(schedule["wait"]), warmup=int(schedule["warmup"]),
                active=int(schedule["active"]), repeat=int(schedule["repeat"]))
            self._profiler = torch.profiler.profile(
                activities=activities, schedule=handle,
                on_trace_ready=self._handle_trace, profile_memory=True,
                with_stack=False, record_shapes=False)
            self._profiler.__enter__()
            return True
        except Exception as error:
            print("[profiler] disabled (fail-soft): " + type(error).__name__
                  + ": " + str(error)[:160], flush=True)
            self._profiler = None
            return False

    def phase(self, name):
        """A zero-overhead annotation context when the profiler is not active.

        Emitting ``record_function`` unconditionally costs a little on every
        loop iteration even when profiling is off, so the loop annotates
        through this gate: a null context unless a live profiler is running.
        """
        if self._profiler is None:
            import contextlib

            return contextlib.nullcontext()
        return self._torch.profiler.record_function(name)

    def step(self):
        if self._profiler is None:
            return
        try:
            self._profiler.step()
        except Exception as error:
            print("[profiler] step skipped: " + type(error).__name__,
                  flush=True)

    def close(self):
        if self._profiler is None:
            return
        try:
            self._profiler.__exit__(None, None, None)
        except Exception as error:
            print("[profiler] exit skipped: " + type(error).__name__,
                  flush=True)
        finally:
            self._profiler = None

    @staticmethod
    def _device_time_us(event: object) -> float:
        # Older Kaggle images expose the CUDA-specific name.
        duration = getattr(event, "self_device_time_total", None)
        if duration is None:
            duration = getattr(event, "self_cuda_time_total", 0.0)
        return float(duration or 0.0)

    def top_ops(self, profiler: "Any") -> list[tuple[str, float, float, int]]:  # noqa: UP037 -- also injected without future annotations
        rows = [(str(event.key), self._device_time_us(event) / 1000.0,
                 float(event.self_cpu_time_total) / 1000.0, int(event.count))
                for event in profiler.key_averages()]
        return sorted(rows, key=lambda row: row[1], reverse=True)[:self._top_n]

    def _handle_trace(self, profiler):
        try:
            if not self._trace_dir:
                return
            os.makedirs(self._trace_dir, exist_ok=True)
            trace_path = os.path.join(
                self._trace_dir, "epoch_%d.json" % int(self.epoch))
            profiler.export_chrome_trace(trace_path)
            rows = self.top_ops(profiler)
            print("[profiler] chrome trace -> " + trace_path, flush=True)
            print("[profiler] top ops by CUDA time (ms), epoch %d:"
                  % int(self.epoch), flush=True)
            for key, cuda_ms, cpu_ms, count in rows:
                print("  %-38s cuda=%8.3f cpu=%8.3f n=%d"
                      % (key[:38], cuda_ms, cpu_ms, count), flush=True)
            if self._on_metrics is not None:
                self._on_metrics(rows, int(self.epoch), trace_path)
        except Exception as error:
            print("[profiler] trace handler skipped: " + type(error).__name__
                  + ": " + str(error)[:160], flush=True)
