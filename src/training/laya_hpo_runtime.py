"""Pure, GPU/DB-free logic for the laya HPO lane.

The remote Kaggle kernel attaches datasets and installs ``laya`` over pip; it
does NOT clone the repository, so it cannot import this package. ``cli.laya_hpo``
injects these exact function sources into the staged kernel via
``inspect.getsource`` (the ``core.laya_controls`` / ``cli.laya_lane`` precedent),
so this is the ONE implementation and it is unit-tested here with plain Python.

Nothing here imports torch (or optuna/sqlalchemy): the lease/champion stores,
the trial objects and the torch module are dependency-injected, so the objective
wiring (fence before promote, revoke on failure) AND the profiler harness are
pinned without a real PostgreSQL or a GPU.
"""
from __future__ import annotations

import math
from pathlib import Path


def sample_dials(trial, space):
    """Sample every dial with the generic ``trial.suggest_*`` dispatch.

    ``space['dials']`` insertion order defines the suggest-call order (which
    defines the TPE search layout), so a reordered spec is a new search surface
    — deliberately, and visibly.
    """
    sampled = {}
    for name, spec in space.get("dials", {}).items():
        kind = spec["type"]
        if kind == "int":
            sampled[name] = trial.suggest_int(
                name, int(spec["lo"]), int(spec["hi"]))
        elif kind == "float":
            sampled[name] = trial.suggest_float(
                name, float(spec["lo"]), float(spec["hi"]),
                log=bool(spec.get("log", False)))
        elif kind == "categorical":
            sampled[name] = trial.suggest_categorical(
                name, list(spec["choices"]))
        else:
            raise ValueError("unknown laya HPO dial type: " + repr(kind))
    return sampled


def route_dials(base_config, base_control, dials, space):
    """Route sampled dials to the TrainConfig / training-control channels.

    The ``target`` declared per dial is the SSOT; this function knows nothing
    about which knob belongs to which channel.
    """
    config = dict(base_config)
    control = dict(base_control)
    specs = space.get("dials", {})
    for name, value in dials.items():
        target = specs[name]["target"]
        if target == "config":
            config[name] = value
        elif target == "control":
            control[name] = value
        else:
            raise ValueError("unknown laya HPO dial target: " + repr(target))
    return config, control


def finished_trial_count(study, finished_states):
    """Count trials in any terminal state (the resume-safe budget anchor).

    ``finished_states`` are Optuna ``TrialState`` names so this stays
    optuna-free; the kernel passes ``("COMPLETE", "PRUNED", "FAIL")``.
    """
    states = set(finished_states)
    return sum(1 for trial in study.trials if trial.state.name in states)


def per_worker_budget(remaining, workers):
    """Split the remaining trial budget across the worker processes.

    ``workers`` is the actual number of concurrent worker processes (slots) or
    1 (DDP-per-trial serialises). The ceiling avoids leaving trials unrun when
    the budget is not divisible; each worker re-counts finished trials, so a
    resumed session never overshoots by more than the worker count.
    """
    return max(0, math.ceil(max(0, int(remaining)) / max(1, int(workers))))


def objective_value(trial, run_fn, lease_store, champion_store,
                    generation_id, model_key, objective_mode=None):
    """One fenced HPO trial: lease -> run -> assert -> promote -> return.

    Maximizes the value returned by ``run_fn`` (the kernel returns DEV
    accuracy). The held-out/test split is never consulted here. A lease is
    issued for the trial number Optuna assigned, asserted current BEFORE any
    promotion (a zombie worker must never publish), and revoked on ANY failure
    so a replacement worker can re-issue the epoch.

    ``objective_mode`` (optional) shapes a multi-objective return value while
    the champion registry always promotes on the PRIMARY (dev accuracy).
    ``run_fn`` returns ``(accuracy, dev_loss, artifact)`` or, when a secondary
    objective is configured, ``(accuracy, dev_loss, artifact, secondary)``.
    """
    lease = (lease_store.issue(generation_id=generation_id, model_key=model_key,
                               trial_number=int(trial.number))
             if lease_store is not None else None)
    try:
        result = run_fn(trial)
        if len(result) == 4:
            accuracy, dev_loss, artifact, secondary = result
        else:
            accuracy, dev_loss, artifact = result
            secondary = None
        if lease_store is not None:
            lease_store.assert_current(lease)
        trial.set_user_attr("dev_accuracy", float(accuracy))
        if dev_loss is not None:
            trial.set_user_attr("dev_loss", float(dev_loss))
        trial.set_user_attr("checkpoint", str(artifact))
        if champion_store is not None:
            champion_store.promote(
                generation_id=generation_id, model_key=model_key,
                trial_number=int(trial.number), value=float(accuracy),
                artifact_snapshot=str(artifact), lease_epoch=int(lease.epoch))
        value = float(accuracy)
        if objective_mode is not None and objective_mode.multi:
            metrics = {"dev_accuracy": float(accuracy)}
            if secondary is not None:
                metrics[objective_mode.secondary] = float(secondary)
            value = objective_mode.value(metrics)
        return value
    except BaseException:
        if lease_store is not None:
            try:
                lease_store.revoke(lease)
            except Exception:  # noqa: BLE001,S110 - best-effort; the primary error re-raises
                pass
        raise


class FidelityReporter:
    """Report per-epoch dev accuracy to the active Optuna trial and prune.

    Wraps the perf patch's ``_dev_metrics`` (the per-epoch dev evaluation) so
    the ASHA/Hyperband pruner sees the intermediate values it needs. Fail-soft:
    a reporting error never fails training.
    """

    def __init__(self, trial, optuna_module, namespace):
        self.trial = trial
        self.optuna = optuna_module
        self.namespace = namespace
        self._original = None
        self.step = 0

    def install(self):
        original = _namespace_get(self.namespace, "_dev_metrics")
        if original is None or self.trial is None or self.optuna is None:
            return self
        self._original = original
        trial = self.trial
        optuna_module = self.optuna
        state = self

        def reported(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                accuracy = result.get("accuracy")
                if accuracy is not None:
                    trial.report(float(accuracy), state.step)
                    state.step += 1
                    if trial.should_prune():
                        raise optuna_module.TrialPruned(
                            "pruned at fidelity stage " + str(state.step))
            except optuna_module.TrialPruned:
                raise
            except Exception:  # noqa: BLE001,S110 - reporting is best-effort
                pass
            return result

        _namespace_set(self.namespace, "_dev_metrics", reported)
        return self

    def uninstall(self):
        if self._original is not None:
            _namespace_set(self.namespace, "_dev_metrics", self._original)
            self._original = None

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.uninstall()
        return False


# ── per-trial torch.profiler harness (fail-soft, bounded) ──────────────────
# The major phases of a trial, annotated with torch.profiler.record_function.
# The remote kernel monkeypatches the phase callables it can reach (the laya
# train functions, the perf-patch helpers in the kernel namespace, and the
# optimizer instance handed back by `_make_optimizer`) so the annotations land
# WITHOUT editing laya or cli.laya_lane's perf patch.
PHASE_ENCODE = "data.encode"
PHASE_FORWARD = "forward"
PHASE_LOSS = "loss"
PHASE_BACKWARD = "backward"
PHASE_OPTIMIZER_STEP = "optimizer_step"
PHASE_DEV_EVAL = "dev_eval"
PHASE_CHECKPOINT_SAVE = "checkpoint_save"
PHASE_CALIBRATION = "calibration"
PROFILER_PHASES = (
    PHASE_ENCODE, PHASE_FORWARD, PHASE_LOSS, PHASE_BACKWARD,
    PHASE_OPTIMIZER_STEP, PHASE_DEV_EVAL, PHASE_CHECKPOINT_SAVE,
    PHASE_CALIBRATION,
)


def profiler_schedule(torch_module, config):
    """The bounded profile schedule from the config (wait/warmup/active/repeat).

    A bounded schedule samples a SLICE of the trial, never the whole run: the
    wait/warmup cycles are discarded and only `active` cycles per `repeat` are
    captured.
    """
    return torch_module.profiler.schedule(
        wait=int(config["wait"]), warmup=int(config["warmup"]),
        active=int(config["active"]), repeat=int(config["repeat"]))


def top_ops_table(profiler, device_type, limit):
    """The key_averages() top-op table; CUDA time on GPU, CPU time otherwise."""
    averages = profiler.key_averages()
    sort_by = "cuda_time_total" if device_type == "cuda" else "cpu_time_total"
    try:
        return averages.table(sort_by=sort_by, row_limit=int(limit))
    except Exception:  # noqa: BLE001 - an older torch may not take sort_by
        return averages.table(row_limit=int(limit))


def _namespace_get(namespace, attr):
    if isinstance(namespace, dict):
        return namespace.get(attr)
    return getattr(namespace, attr, None)


def _namespace_set(namespace, attr, value):
    if isinstance(namespace, dict):
        namespace[attr] = value
    else:
        setattr(namespace, attr, value)


def _wrapped_phase(fn, record_function, name, on_call=None):
    """Wrap one callable in a record_function annotation (fail-soft caller)."""
    def wrapper(*args, **kwargs):
        with record_function(name):
            result = fn(*args, **kwargs)
        if on_call is not None:
            on_call()
        return result
    wrapper.__name__ = getattr(fn, "__name__", name)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _wrapped_optimizer_factory(fn, record_function, name, on_call=None):
    """Wrap `_make_optimizer` so the RETURNED optimizer's step() is annotated.

    Optimizer subclasses override `step`, so patching a base class would miss
    them; patching the bound instance (AdamW/LAMB/Adafactor all allow it) is the
    only reliable seam.
    """
    def factory(*args, **kwargs):
        optimizer = fn(*args, **kwargs)
        try:
            original_step = optimizer.step

            def profiled_step(*step_args, **step_kwargs):
                with record_function(name):
                    result = original_step(*step_args, **step_kwargs)
                if on_call is not None:
                    on_call()
                return result

            optimizer.step = profiled_step
        except Exception:  # noqa: BLE001 - annotation is optional, never fatal
            return optimizer
        return optimizer
    return factory


def install_phase_hooks(*, torch_module, record_function, laya_train, namespace,
                        on_optimizer_step=None):
    """Annotate the major phases; returns an uninstall callable.

    Every patch is best-effort and restored by the returned callable; a missing
    target (an older laya without `rlcd_loss`, a namespace without the perf
    helpers) is skipped, never fatal.
    """
    restores = []

    def patch(target, attr, name, on_call=None, factory=False):
        original = _namespace_get(target, attr)
        if original is None:
            return
        wrapped = (_wrapped_optimizer_factory(
            original, record_function, name, on_call) if factory else
            _wrapped_phase(original, record_function, name, on_call))
        _namespace_set(target, attr, wrapped)
        restores.append((target, attr, original))

    try:
        patch(laya_train, "encode_item", PHASE_ENCODE)
        patch(laya_train, "soft_ce_loss", PHASE_LOSS)
        patch(laya_train, "rlcd_loss", PHASE_LOSS)
        patch(laya_train, "fit_temperature_map", PHASE_CALIBRATION)
        patch(namespace, "_forward_dtype", PHASE_FORWARD)
        patch(namespace, "_dev_metrics", PHASE_DEV_EVAL)
        patch(namespace, "_save_control_checkpoint", PHASE_CHECKPOINT_SAVE)
        patch(namespace, "_make_optimizer", PHASE_OPTIMIZER_STEP,
              on_optimizer_step, factory=True)
        if torch_module is not None:
            patch(torch_module.Tensor, "backward", PHASE_BACKWARD)
    except BaseException:
        for target, attr, original in reversed(restores):
            _namespace_set(target, attr, original)
        raise

    def uninstall():
        for target, attr, original in reversed(restores):
            _namespace_set(target, attr, original)

    return uninstall


class TrialProfiler:
    """A fail-soft, bounded ``torch.profiler`` wrapper for ONE HPO trial.

    * CPU + CUDA activities, ``profile_memory=True``, ``with_stack=False``;
    * a bounded ``schedule`` (wait/warmup/active/repeat) so only a slice of the
      trial is sampled; the schedule is advanced once per optimizer step;
    * a chrome trace at ``trace_path`` plus a ``key_averages()`` top-op table to
      the logger (stdout) and the optional wandb callback;
    * rank-0 only and CUDA-only (auto-disabled on CPU);
    * a profiler error NEVER fails the trial (``__exit__`` returns False and
      every step is guarded).

    Torch is injected so this is unit-testable without a GPU.
    """

    def __init__(self, *, torch_module, config, trace_path, device_type,
                 laya_train=None, namespace=None, logger=None, wandb_log=None,
                 rank0=True):
        self.torch = torch_module
        self.config = dict(config or {})
        self.trace_path = Path(trace_path)
        self.device_type = str(device_type)
        self.laya_train = laya_train
        self.namespace = namespace
        self.logger = logger
        self.wandb_log = wandb_log
        self.rank0 = bool(rank0)
        self.enabled = False
        self._profiler = None
        self._uninstall = None

    def _emit(self, line):
        if self.logger is None:
            return
        try:
            self.logger(str(line))
        except Exception:  # noqa: BLE001,S110 - logging must never fail a trial
            pass

    def __enter__(self):
        self.enabled = (bool(self.config.get("enabled", True))
                        and self.device_type == "cuda" and self.rank0)
        if not self.enabled:
            self._emit("profiler disabled (config/CPU/non-rank0)")
            return self
        try:
            torch_module = self.torch
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler = torch_module.profiler.profile(
                activities=[torch_module.profiler.ProfilerActivity.CPU,
                            torch_module.profiler.ProfilerActivity.CUDA],
                schedule=profiler_schedule(torch_module, self.config),
                on_trace_ready=self._on_trace_ready,
                profile_memory=True, with_stack=False, record_shapes=False)
            self._uninstall = install_phase_hooks(
                torch_module=torch_module,
                record_function=torch_module.profiler.record_function,
                laya_train=self.laya_train, namespace=self.namespace,
                on_optimizer_step=self.step)
            self._profiler.start()
            self._emit("profiler started -> " + str(self.trace_path))
        except Exception as error:  # noqa: BLE001 - fail soft
            self.enabled = False
            self._emit("profiler setup failed (continued unprofiled): "
                       + str(error)[:200])
        return self

    def step(self):
        """Advance the bounded schedule (called once per optimizer step)."""
        if not self.enabled or self._profiler is None:
            return
        try:
            self._profiler.step()
        except Exception:  # noqa: BLE001,S110
            pass

    def _on_trace_ready(self, profiler):
        try:
            profiler.export_chrome_trace(str(self.trace_path))
            self._emit("profiler trace -> " + str(self.trace_path))
        except Exception as error:  # noqa: BLE001
            self._emit("profiler trace export failed: " + str(error)[:200])

    def _report(self):
        limit = int(self.config.get("top_ops", 15))
        table = top_ops_table(self._profiler, self.device_type, limit)
        self._emit("profiler top ops (device=" + self.device_type
                   + ", top=" + str(limit) + "):\n" + str(table))
        if self.wandb_log is not None:
            try:
                self.wandb_log(table)
            except Exception:  # noqa: BLE001,S110 - tracking is best-effort
                pass

    def __exit__(self, exc_type, exc, tb):
        if self._uninstall is not None:
            try:
                self._uninstall()
            except Exception:  # noqa: BLE001,S110
                pass
            self._uninstall = None
        if self._profiler is not None:
            try:
                self._profiler.stop()
            except Exception:  # noqa: BLE001,S110
                pass
            if self.enabled:
                try:
                    if not self.trace_path.is_file():
                        self._profiler.export_chrome_trace(str(self.trace_path))
                except Exception as error:  # noqa: BLE001
                    self._emit("profiler final export failed: "
                               + str(error)[:200])
                try:
                    self._report()
                except Exception as error:  # noqa: BLE001
                    self._emit("profiler report failed: " + str(error)[:200])
            self._profiler = None
        return False
