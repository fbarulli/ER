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
    """Patience/min-delta early stopping (direction-aware, pure)."""

    def __init__(self, patience, min_delta=0.0, lower_is_better=False):
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta or 0.0)
        self.lower_is_better = bool(lower_is_better)

    @classmethod
    def from_control(cls, control):
        return cls(
            patience=control.get("early_stop_patience"),
            min_delta=control.get("early_stop_min_delta"),
            lower_is_better=control.get("early_stop_metric") == "dev_loss")

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

    # The landed scheduler used only when a control block carries no ``kind``
    # (an offline unit test); the SSOT default is baked from FinetuneSpec.
    LANDED_KIND = "cosine"

    def __init__(self, optimizer, kind, total_updates, min_lr, warmup,
                 plateau_mode="max", plateau_factor=0.5, plateau_patience=2,
                 onecycle_pct_start=0.3):
        self.optimizer = optimizer
        self.kind = str(kind or self.LANDED_KIND).lower()
        self.total_updates = max(1, int(total_updates))
        self.min_lr = float(min_lr)
        self.warmup = max(0, int(warmup or 0))
        self.plateau_mode = plateau_mode
        # Kept raw: only the plateau/onecycle branches coerce them, so an
        # absent value never breaks the landed cosine path.
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.onecycle_pct_start = onecycle_pct_start
        self.is_plateau = self.kind == "plateau"

    @classmethod
    def from_control(cls, control, optimizer, total_updates, min_lr,
                     lower_is_better=False):
        warmup = cls.effective_warmup_steps(
            control.get("warmup_steps"), control.get("warmup_frac"),
            total_updates)
        return cls(
            optimizer=optimizer,
            kind=control.get("lr_scheduler"),
            total_updates=total_updates,
            min_lr=min_lr,
            warmup=warmup,
            plateau_mode="min" if lower_is_better else "max",
            plateau_factor=control.get("plateau_factor"),
            plateau_patience=control.get("plateau_patience"),
            onecycle_pct_start=control.get("onecycle_pct_start"))

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

    def top_ops(self, profiler):
        events = list(profiler.key_averages())
        try:
            events.sort(key=lambda event: float(
                getattr(event, "self_cuda_time_total", 0.0) or 0.0),
                reverse=True)
        except Exception:
            pass
        rows = []
        for event in events[:self._top_n]:
            rows.append((
                str(event.key),
                float(getattr(event, "self_cuda_time_total", 0.0) or 0.0) / 1000.0,
                float(getattr(event, "self_cpu_time_total", 0.0) or 0.0) / 1000.0,
                int(getattr(event, "count", 0) or 0)))
        return rows

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
