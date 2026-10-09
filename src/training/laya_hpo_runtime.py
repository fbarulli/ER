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
import time
import types
from contextlib import nullcontext
from pathlib import Path


def _shared(name: str):
    """Resolve a symbol defined by the earlier injected control-plane module.

    In the staged kernel every injected module shares one namespace, so the
    symbol is a global; on the host the modules are importable packages.
    """
    symbol = globals().get(name)
    if symbol is not None:
        return symbol
    from training import hpo_control_plane

    return getattr(hpo_control_plane, name)


def _sample_one(trial, name, spec):
    """One ``trial.suggest_*`` call, dispatched by the SSOT dial type."""
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(
            name, int(spec["lo"]), int(spec["hi"]),
            log=bool(spec.get("log", False)))
    if kind == "float":
        return trial.suggest_float(
            name, float(spec["lo"]), float(spec["hi"]),
            log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError("unknown laya HPO dial type: " + repr(kind))


def sample_dials(trial, space):
    """Sample every dial with the generic ``trial.suggest_*`` dispatch.

    ``space['dials']`` insertion order defines the suggest-call order (which
    defines the TPE search layout), so a reordered spec is a new search surface
    — deliberately, and visibly.

    A dial may declare a conditional gate ``when: {dial: <gate>, equals: <v>}``:
    it is sampled ONLY when its gate matches. An off gate skips the dependent
    dials entirely, so they keep their SSOT default in the base control block
    (default-OFF, reproducible). A dependent whose gate is declared LATER is
    deferred and resolved once the gate is sampled, so a reordered spec still
    resolves its dependents instead of silently dropping them.
    """
    sampled = {}
    deferred = []
    for name, spec in space.get("dials", {}).items():
        gate = spec.get("when")
        if gate is not None:
            gate_name = gate["dial"]
            if gate_name not in sampled:
                deferred.append((name, spec))
                continue
            if sampled[gate_name] != gate.get("equals", True):
                continue
        sampled[name] = _sample_one(trial, name, spec)
    progress = True
    while deferred and progress:
        progress = False
        pending = []
        for name, spec in deferred:
            gate_name = spec["when"]["dial"]
            if gate_name in sampled:
                if sampled[gate_name] == spec["when"].get("equals", True):
                    sampled[name] = _sample_one(trial, name, spec)
                progress = True
            else:
                pending.append((name, spec))
        deferred = pending
    return sampled


def route_dials(base_config, base_control, dials, space):
    """Route sampled dials to the TrainConfig / training-control channels.

    The ``target`` declared per dial is the SSOT; this function knows nothing
    about which knob belongs to which channel. A dial without a ``target``
    defaults to ``config``.
    """
    config = dict(base_config)
    control = dict(base_control)
    specs = space.get("dials", {})
    for name, value in dials.items():
        spec = specs[name]
        # Respect a conditional gate: a dependent dial is routed ONLY when its
        # gate is present AND matches. This keeps a warm-start seed (which may
        # carry a dependent without its gate) from setting a flag-off value.
        gate = spec.get("when")
        if gate is not None and dials.get(gate["dial"]) != gate.get("equals",
                                                                  True):
            continue
        target = spec.get("target", "config")
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


def trial_primary_value(trial):
    """The comparable PRIMARY scalar of a trial's value.

    Single-objective Optuna trials expose ``.value``; multi-objective trials
    expose ``.values`` (a list) and a ``.value`` of ``None``.  Ranking must use
    the primary element either way, otherwise every multi-objective trial is
    dropped (the old bug).
    """
    values = getattr(trial, "values", None)
    if values:
        return values[0]
    return getattr(trial, "value", None)


def trial_full_value(trial):
    """The full value vector as a list (``[scalar]`` for single-objective)."""
    values = getattr(trial, "values", None)
    if values:
        return list(values)
    value = getattr(trial, "value", None)
    return None if value is None else [value]


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
    """One fenced HPO trial: issue lease -> run -> assert -> record value.

    Maximizes the value returned by ``run_fn`` (the kernel returns DEV
    accuracy). The held-out/test split is never consulted here. A lease is
    issued for the trial number Optuna assigned and a heartbeat renews it while
    ``run_fn`` executes; ``assert_current`` is checked BEFORE the result is
    recorded (an expired/zombie worker is fenced), and the lease is revoked on
    ANY failure so a replacement can re-issue the epoch.

    The champion is deliberately NOT promoted here: promotion may only happen
    AFTER Optuna commits the trial COMPLETE (see ``promote_committed_trial``),
    otherwise a failure between an inline promotion and the commit would record
    the trial FAIL while the registry already pointed at it.

    Control-plane failures (lease issue/assert) are raised as
    ``HpoInfrastructureError`` so ``study.optimize(catch=(Exception,))`` cannot
    swallow them as an ordinary trial failure; ``run_fn`` failures stay normal
    exceptions and are recorded FAIL.

    ``objective_mode`` (optional) shapes a multi-objective return value while
    the champion registry always promotes on the PRIMARY (dev accuracy).
    """
    lease = _issue_lease(lease_store, generation_id, model_key, int(trial.number))
    try:
        with _lease_heartbeat(lease_store, lease):
            result = run_fn(trial)
        if len(result) == 4:
            accuracy, dev_loss, artifact, secondary = result
        else:
            accuracy, dev_loss, artifact = result
            secondary = None
        if lease_store is not None:
            _assert_lease(lease_store, lease)
        trial.set_user_attr("dev_accuracy", float(accuracy))
        if dev_loss is not None:
            trial.set_user_attr("dev_loss", float(dev_loss))
        trial.set_user_attr("checkpoint", str(artifact))
        if lease is not None:
            trial.set_user_attr("hpo_lease_epoch", int(lease.epoch))
        value = float(accuracy)
        if objective_mode is not None and objective_mode.multi:
            metrics = {"dev_accuracy": float(accuracy)}
            if secondary is not None:
                metrics[objective_mode.secondary] = float(secondary)
            value = objective_mode.value(metrics)
        return value
    except BaseException:
        if lease_store is not None and lease is not None:
            try:
                lease_store.revoke(lease)
            except Exception:  # noqa: BLE001,S110 - best-effort; primary error re-raises
                pass
        raise


def _issue_lease(lease_store, generation_id, model_key, trial_number):
    if lease_store is None:
        return None
    try:
        return lease_store.issue(generation_id=generation_id,
                                 model_key=model_key, trial_number=trial_number)
    except BaseException as error:
        if isinstance(error, Exception):
            raise _shared("HpoInfrastructureError")(
                f"lease issue failed for {model_key}/trial {trial_number}: "
                f"{error}") from error
        raise


def _assert_lease(lease_store, lease):
    try:
        lease_store.assert_current(lease)
    except BaseException as error:
        if isinstance(error, Exception):
            raise _shared("HpoInfrastructureError")(str(error)) from error
        raise


def _lease_heartbeat(lease_store, lease):
    if lease_store is None or lease is None:
        return nullcontext()
    factory = getattr(lease_store, "heartbeat", None)
    if factory is None:
        return nullcontext()
    return factory(lease)


def promote_committed_trial(champion_store, trial, *, generation_id, model_key):
    """Promote a trial's champion candidate only AFTER Optuna committed it.

    Returns the promoted ``Champion`` (or the still-best champion) or ``None``
    when there is nothing to promote.  A trial without a fencing lease epoch is
    never published (no unfenced champion); a non-COMPLETE trial is skipped.
    """
    if champion_store is None or trial is None:
        return None
    state = getattr(trial, "state", None)
    state_name = getattr(state, "name", None)
    if state_name is not None and state_name != "COMPLETE":
        return None
    attrs = getattr(trial, "user_attrs", None) or {}
    epoch = attrs.get("hpo_lease_epoch")
    if epoch is None:
        return None
    return champion_store.promote(
        generation_id=generation_id, model_key=model_key,
        trial_number=int(trial.number),
        value=float(attrs["dev_accuracy"]),
        artifact_snapshot=str(attrs["checkpoint"]),
        lease_epoch=int(epoch))


def resolve_champion_artifact(champion_store, *, generation_id, model_key,
                              mode):
    """The champion checkpoint to warm-start from, or ``None``.

    Reads the shared champion registry — the read side that used to be missing,
    which made ``warm_start: champion`` silently fall back to the base model.
    Returns the champion's artifact path, or ``None`` when there is no champion
    yet (so the caller falls back to the base model).  Non-champion modes never
    read the registry.
    """
    if mode != "champion" or champion_store is None:
        return None
    champion = champion_store.read(generation_id=generation_id,
                                   model_key=model_key)
    if champion is None:
        return None
    return str(champion.artifact_snapshot)


class ReservedTrialLoop:
    """Run one atomically-reserved Optuna trial at a time.

    Each iteration: reserve one slot from the shared ``ledger`` (a no-op-return
    when another worker/session holds the budget), run exactly one trial, observe
    it ONCE after commit, then charge it COMPLETE or release it.  A FAIL/PRUNED
    trial releases its slot, so it cannot starve the budget; an
    ``HpoInfrastructureError`` releases the slot and aborts loud.
    """

    def __init__(self, study, objective, ledger, *, optimize=None,
                 optimize_kwargs=None, observer=None, champion_store=None,
                 generation_id=None, model_key=None, timeout_s=0,
                 clock=time.monotonic, log=None):
        self._study = study
        self._objective = objective
        self._ledger = ledger
        self._optimize = optimize
        self._optimize_kwargs = dict(optimize_kwargs or {})
        self._observer = observer
        self._champion_store = champion_store
        self._generation_id = generation_id
        self._model_key = model_key
        self._timeout_s = int(timeout_s or 0)
        self._clock = clock
        self._log = log or (lambda line: None)

    def _run_one(self):
        if self._optimize is not None:
            self._optimize()
            return
        self._study.optimize(self._objective, n_trials=1,
                             **self._optimize_kwargs)

    def _newest(self, before):
        trials = self._study.trials
        return trials[before] if len(trials) > before else None

    @staticmethod
    def _state(trial):
        return getattr(getattr(trial, "state", None), "name", None)

    def _emit_committed(self, trial):
        """Publish one COMMITTED trial to the observer (post-commit, idempotent).

        Named ``_emit_committed`` (not ``_observe``) so the staged kernel never
        carries a pre-commit observe token.
        """
        if self._observer is None:
            return
        try:
            self._observer.observe(trial)
        except Exception as error:  # noqa: BLE001 - observability is best-effort
            self._log("observer skipped: " + str(error)[:160])

    def run(self):
        deadline = (self._clock() + self._timeout_s
                    if self._timeout_s > 0 else None)
        while True:
            if deadline is not None and self._clock() >= deadline:
                self._log("reserved loop: wall-clock timeout reached")
                return "timeout"
            if self._ledger.reserve(1) != 1:
                self._log("reserved loop: shared budget exhausted")
                return "budget_exhausted"
            before = len(self._study.trials)
            try:
                self._run_one()
            except BaseException:
                # Never leave a slot pinned when the worker aborts loud.
                self._ledger.release(1)
                raise
            trial = self._newest(before)
            if trial is None:
                self._ledger.release(1)
                continue
            self._emit_committed(trial)
            if self._state(trial) == "COMPLETE":
                self._ledger.complete(1)
                promote_committed_trial(
                    self._champion_store, trial,
                    generation_id=self._generation_id, model_key=self._model_key)
            else:
                self._ledger.release(1)


class FidelityReporter:
    """Report the per-epoch dev metric to Optuna so ASHA/Hyperband can prune.

    The REAL per-epoch dev evaluation in the staged kernel is the perf patch's
    ``DevEvaluator.metrics`` (a staticmethod returning a ``DevReport`` with an
    ``accuracy`` attribute). This class wraps that seam — never an invented
    ``_dev_metrics`` name — so ``trial.report``/``should_prune`` actually fire.
    The legacy ``_dev_metrics`` callable is still accepted for hosts that keep
    a dict-returning hook.

    Two modes:
      * direct (slots): ``trial``/``optuna_module`` are set; each epoch reports
        and ``should_prune`` can raise ``TrialPruned`` in-process.
      * streaming (DDP rank): ``sink(step, value)`` is called instead; the DDP
        controller owns the trial and reports what the rank streams.
    """

    # In priority order: the real kernel seam first, then the legacy hook.
    DEV_METRICS_CANDIDATES = ("DevEvaluator.metrics", "_dev_metrics")

    def __init__(self, trial, optuna_module, namespace, sink=None):
        self.trial = trial
        self.optuna = optuna_module
        self.namespace = namespace
        self.sink = sink
        self._original = None
        self._parent = None
        self._attr = None
        self.step = 0
        self.installed = False

    @staticmethod
    def _accuracy_of(result):
        """The accuracy on a ``DevReport`` or on a legacy dict hook."""
        if result is None:
            return None
        value = getattr(result, "accuracy", None)
        if value is None and isinstance(result, dict):
            value = result.get("accuracy")
        return value

    def install(self):
        if self.sink is None and (self.trial is None or self.optuna is None):
            return self
        for path in self.DEV_METRICS_CANDIDATES:
            parent, attr = _resolve_path(self.namespace, path)
            if parent is None:
                continue
            original = _namespace_get(parent, attr)
            if original is None:
                continue
            self._parent, self._attr, self._original = parent, attr, original
            break
        if self._original is None:
            return self
        trial = self.trial
        optuna_module = self.optuna
        state = self

        def reported(*args, **kwargs):
            result = self._original(*args, **kwargs)
            try:
                accuracy = state._accuracy_of(result)
                if accuracy is not None:
                    if state.sink is not None:
                        state.sink(state.step, float(accuracy))
                        state.step += 1
                    else:
                        trial.report(float(accuracy), state.step)
                        state.step += 1
                        if trial.should_prune():
                            raise optuna_module.TrialPruned(
                                "pruned at fidelity stage " + str(state.step))
            except Exception as error:
                if optuna_module is not None and isinstance(
                        error, optuna_module.TrialPruned):
                    raise
                # reporting is best-effort; a real training error still escapes
            return result

        _namespace_set(self._parent, self._attr, reported)
        self.installed = True
        return self

    def uninstall(self):
        if self._original is not None:
            _namespace_set(self._parent, self._attr, self._original)
            self._original = None
            self._parent = None
            self._attr = None
            self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.uninstall()
        return False


# ── per-trial torch.profiler harness (fail-soft, bounded) ──────────────────
# The major phases of a trial, annotated with torch.profiler.record_function.
# The remote kernel monkeypatches the phase callables it can reach (the laya
# train functions, the perf-patch's DevEvaluator.metrics, and the optimizer
# instance handed back by TrainingOptimizer.make) so the annotations land
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


def _resolve_path(namespace, path):
    """Resolve a dotted attribute path to ``(parent, attr)`` or ``(None, None)``.

    The first segment is looked up on ``namespace`` (a dict or a module-like
    object); the rest are ordinary attributes. This lets a hook target the REAL
    kernel seam (e.g. ``DevEvaluator.metrics`` on the injected perf patch)
    without the pure runtime importing torch/laya.
    """
    parts = str(path).split(".")
    parent = namespace
    for part in parts[:-1]:
        parent = _namespace_get(parent, part)
        if parent is None:
            return None, None
    attr = parts[-1]
    if _namespace_get(parent, attr) is None:
        return None, None
    return parent, attr


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
    """Wrap `TrainingOptimizer.make` so the RETURNED optimizer's step() is
    annotated (and the bounded schedule advances once per optimizer step).

    Optimizer subclasses override `step`, so patching a base class would miss
    them; patching the bound instance (AdamW/LAMB/Adafactor all allow it) is the
    only reliable seam. The replacement is a BOUND method (``types.MethodType``),
    never a plain function: torch's LR schedulers read ``optimizer.step.__func__``
    while patching the step, so a plain function makes every scheduler
    construction (CosineAnnealingLR, OneCycleLR, ...) raise AttributeError.
    """
    def factory(*args, **kwargs):
        optimizer = fn(*args, **kwargs)
        try:
            original_step = optimizer.step

            def profiled_step(_self, *step_args, **step_kwargs):
                with record_function(name):
                    result = original_step(*step_args, **step_kwargs)
                if on_call is not None:
                    on_call()
                return result

            optimizer.step = types.MethodType(profiled_step, optimizer)
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

    def patch_path(root, candidates, name, on_call=None, factory=False):
        """Patch the first resolvable dotted candidate on ``root``."""
        for path in candidates:
            target, attr = _resolve_path(root, path)
            if target is None:
                continue
            original = _namespace_get(target, attr)
            wrapped = (_wrapped_optimizer_factory(
                original, record_function, name, on_call) if factory else
                _wrapped_phase(original, record_function, name, on_call))
            _namespace_set(target, attr, wrapped)
            restores.append((target, attr, original))
            return True
        return False

    def patch(target, attr, name, on_call=None, factory=False):
        return patch_path(target, (attr,), name, on_call, factory)

    try:
        patch(laya_train, "encode_item", PHASE_ENCODE)
        patch(laya_train, "soft_ce_loss", PHASE_LOSS)
        patch(laya_train, "rlcd_loss", PHASE_LOSS)
        patch(laya_train, "fit_temperature_map", PHASE_CALIBRATION)
        # REAL kernel seams: the perf patch's DevEvaluator.metrics and
        # TrainingOptimizer.make. The old `_forward_dtype`,
        # `_save_control_checkpoint` and `_make_optimizer` names never existed
        # in the injected sources, so those annotations silently vanished and
        # `TrialProfiler.step()` was never driven (empty trace). Prefer the real
        # seams, keep the legacy names as a fallback for older stages.
        patch_path(namespace, ("DevEvaluator.metrics", "_dev_metrics"),
                   PHASE_DEV_EVAL)
        patch_path(namespace, ("TrainingOptimizer.make", "_make_optimizer"),
                   PHASE_OPTIMIZER_STEP, on_optimizer_step, factory=True)
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
