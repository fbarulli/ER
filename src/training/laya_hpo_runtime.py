"""Pure, GPU/DB-free logic for the laya HPO lane.

The remote Kaggle kernel attaches datasets and installs ``laya`` over pip; it
does NOT clone the repository, so it cannot import this package. ``cli.laya_hpo``
injects these exact function sources into the staged kernel via
``inspect.getsource`` (the ``core.laya_controls`` / ``cli.laya_lane`` precedent),
so this is the ONE implementation and it is unit-tested here with plain Python.

Nothing here imports torch, optuna or sqlalchemy: the lease/champion stores and
the trial objects are dependency-injected, so the objective wiring (fence before
promote, revoke on failure) is pinned without a real PostgreSQL or a GPU.
"""
from __future__ import annotations


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


def objective_value(trial, run_fn, lease_store, champion_store,
                    generation_id, model_key):
    """One fenced HPO trial: lease -> run -> assert -> promote -> return.

    Maximizes the value returned by ``run_fn`` (the kernel returns DEV
    accuracy). The held-out/test split is never consulted here. A lease is
    issued for the trial number Optuna assigned, asserted current BEFORE any
    promotion (a zombie worker must never publish), and revoked on ANY failure
    so a replacement worker can re-issue the epoch.
    """
    lease = lease_store.issue(generation_id=generation_id, model_key=model_key,
                              trial_number=int(trial.number))
    try:
        accuracy, dev_loss, artifact = run_fn(trial)
        lease_store.assert_current(lease)
        trial.set_user_attr("dev_accuracy", float(accuracy))
        if dev_loss is not None:
            trial.set_user_attr("dev_loss", float(dev_loss))
        trial.set_user_attr("checkpoint", str(artifact))
        champion_store.promote(
            generation_id=generation_id, model_key=model_key,
            trial_number=int(trial.number), value=float(accuracy),
            artifact_snapshot=str(artifact), lease_epoch=int(lease.epoch))
        return float(accuracy)
    except BaseException:
        try:
            lease_store.revoke(lease)
        except Exception:  # noqa: BLE001,S110 - best-effort; the primary error re-raises
            pass
        raise
