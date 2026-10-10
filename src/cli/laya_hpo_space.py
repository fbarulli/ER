"""The laya HPO search-space SSOT loader (config/laya_hpo_space.yaml).

Split out of ``cli.laya_hpo`` (the lane orchestrator was past the 1k-line
limit): this module owns ONLY the space document — its path binding, load,
boundary validation (pydantic-adjacent, fail-loud) and digest. ``cli.laya_hpo``
re-exports the surface, so every existing consumer
(``cli.laya_hpo.load_space`` and friends) is untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.common import TRAIN_ROOT

#: The search-space document name (beside the other config documents).
SPACE_FILE_NAME = "laya_hpo_space.yaml"
#: The dial routing channels (a dial's ``target`` names one).
TARGETS = ("config", "control")
#: The generic ``trial.suggest_*`` dispatch types.
TYPES = ("int", "float", "categorical")


def space_path(path: str | Path | None = None) -> Path:
    """The search-space config path (SSOT binding; explicit override wins)."""
    if path:
        return Path(path)
    try:
        from core.common import F
        return Path(F["laya_hpo_space"])
    except Exception:  # noqa: BLE001 - fall back to the conventional path
        return TRAIN_ROOT / "config" / SPACE_FILE_NAME


def load_space(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the laya HPO search space (SSOT; fail-loud)."""
    resolved = space_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"laya HPO search space missing: {resolved}")
    space = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise TypeError(f"laya HPO search space must be a mapping: {resolved}")
    validate_space(space)
    return space


def validate_space(space: dict[str, Any]) -> None:
    """Reject a malformed space before a single trial is sampled.

    The dial names are asserted against ``FinetuneSpec`` fields TODAY so a
    rename on the laya training-dials branch fails the loader loudly rather
    than silently sampling a key no trainer reads.
    """
    from core.laya_config import FinetuneSpec

    dials = space.get("dials")
    if not isinstance(dials, dict) or not dials:
        raise ValueError("laya HPO space must declare a non-empty 'dials' mapping")
    known_fields = set(FinetuneSpec.model_fields)
    # The routed channels are the SSOT field tuples (not a second registry):
    # a dial whose name is in neither tuple samples a value no trainer reads.
    from cli import laya_lane

    config_fields = set(laya_lane.FINETUNE_CONFIG_FIELDS)
    control_fields = set(laya_lane.FINETUNE_CONTROL_FIELDS)
    if not space.get("model_key"):
        raise ValueError("laya HPO space must declare a non-empty 'model_key'")
    for name, spec in dials.items():
        if not isinstance(spec, dict):
            raise TypeError(f"laya HPO dial {name!r} must be a mapping")
        if name not in known_fields:
            raise ValueError(
                f"laya HPO dial {name!r} is not a FinetuneSpec field; update "
                "config/laya_hpo_space.yaml when the training dials land")
        target = spec.get("target")
        if target not in TARGETS:
            raise ValueError(
                f"laya HPO dial {name!r} target must be one of {TARGETS}, "
                f"got {target!r}")
        channel = config_fields if target == "config" else control_fields
        if name not in channel:
            raise ValueError(
                f"laya HPO dial {name!r} targets {target!r} but is not in "
                f"FINETUNE_{target.upper()}_FIELDS; it would be a dead/no-op dial")
        if name in config_fields and name in control_fields:
            raise ValueError(
                f"laya HPO dial {name!r} is routed to BOTH channels; split the "
                "SSOT field tuples")
        kind = spec.get("type")
        if kind not in TYPES:
            raise ValueError(
                f"laya HPO dial {name!r} type must be one of {TYPES}, "
                f"got {kind!r}")
        if kind in ("int", "float"):
            if "lo" not in spec or "hi" not in spec:
                raise ValueError(f"laya HPO dial {name!r} needs lo and hi")
            lo, hi = spec["lo"], spec["hi"]
            if lo > hi:
                raise ValueError(
                    f"laya HPO dial {name!r} has an invalid range [{lo}, {hi}]")
            if kind == "int" and not (isinstance(lo, int) and isinstance(hi, int)):
                raise ValueError(f"laya HPO dial {name!r} int bounds must be ints")
        else:  # categorical
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"laya HPO dial {name!r} needs a non-empty choices list")
        gate = spec.get("when")
        if gate is not None:
            if not isinstance(gate, dict) or "dial" not in gate:
                raise ValueError(
                    f"laya HPO dial {name!r} 'when' must declare a gate dial")
            gate_name = gate["dial"]
            if gate_name not in dials:
                raise ValueError(
                    f"laya HPO dial {name!r} gates on unknown dial "
                    f"{gate_name!r}")
            gate_spec = dials[gate_name]
            if gate_spec.get("type") != "categorical":
                raise ValueError(
                    f"laya HPO dial {name!r} gate {gate_name!r} must be "
                    "categorical")
            if "equals" in gate and gate["equals"] not in gate_spec.get(
                    "choices", []):
                raise ValueError(
                    f"laya HPO dial {name!r} gate value {gate['equals']!r} is "
                    f"not a choice of {gate_name!r}")
    if int(space.get("n_trials", 0)) <= 0:
        raise ValueError("laya HPO space n_trials must be a positive integer")
    if int(space.get("n_jobs", 0)) <= 0:
        raise ValueError("laya HPO space n_jobs must be a positive integer")
    objective = space.get("objective") or {}
    if objective.get("direction") not in ("maximize", "minimize"):
        raise ValueError("laya HPO space objective.direction must be maximize|minimize")
    if objective.get("primary") != "dev_accuracy":
        raise ValueError("laya HPO objective.primary must be dev_accuracy")
    if objective.get("secondary") != "dev_loss":
        raise ValueError("laya HPO objective.secondary must be dev_loss")
    if objective.get("forbidden") != "test":
        raise ValueError("laya HPO objective.forbidden must be test")
    profiler = space.get("profiler")
    if profiler is not None:
        if not isinstance(profiler, dict):
            raise TypeError("laya HPO profiler must be a mapping")
        if not isinstance(profiler.get("enabled"), bool):
            raise ValueError("laya HPO profiler.enabled must be a bool")
        for key in ("wait", "warmup", "active", "repeat", "top_ops"):
            value = profiler.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"laya HPO profiler.{key} must be a non-negative int")
        if profiler["active"] < 1:
            raise ValueError("laya HPO profiler.active must be >= 1")
        if profiler["top_ops"] < 1:
            raise ValueError("laya HPO profiler.top_ops must be >= 1")
    plot = space.get("plot")
    if plot is not None:
        if not isinstance(plot, dict):
            raise TypeError("laya HPO plot must be a mapping")
        top_runs = plot.get("top_runs")
        if (not isinstance(top_runs, int) or isinstance(top_runs, bool)
                or top_runs < 1):
            raise ValueError("laya HPO plot.top_runs must be a positive int")
    # The option components own their own validation; assemble once so a bad
    # parallelism/sampler/pruner/fidelity/warm-start/ensemble name fails at
    # load, not mid-sweep.
    from training import laya_hpo_options

    try:
        laya_hpo_options.build_option_set(space)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"laya HPO options invalid: {exc}") from exc


def space_digest(space: dict[str, Any]) -> str:
    """A stable digest of the space (for receipts; never a secret)."""
    payload = json.dumps(space, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
