"""Per-model HPO objective adapters (class-based, one class per model family).

The registry promises every model a real ``(trial, context) -> float``
objective. This module delivers it: each adapter wraps a model runner,
**validates the runner's signature at construction** (so a runner whose calling
convention does not match can never masquerade as an objective), maps the
sampled ``dials`` + ``context`` to the runner's real arguments, and coerces the
result to a float.

The heavy lanes (text/gnn/cascade) are driven through their landed entrypoints
(``model_tracks.run:run`` / ``graph_tracks.train:train`` /
``graph_tracks.report:report_cascade``); everything the adapter cannot know
(an output workdir, how to read the scored metric out of a report) travels in
``context`` and is required LOUDLY, never guessed.
"""
from __future__ import annotations

import importlib
import inspect
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path


def resolve_runner(reference: str):
    """Import ``module:attr``; fail loud on a malformed or missing reference."""
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(f"runner must be 'module:attr', got {reference!r}")
    module_name, _, attr = reference.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise RuntimeError(
            f"objective runner {reference!r} is not defined") from exc


def _returns_none(runner) -> bool:
    """True when a runner's return annotation is ``None`` (a sweep driver)."""
    try:
        annotation = inspect.signature(runner).return_annotation
    except (TypeError, ValueError):
        return False
    return annotation is None or annotation == "None"


class ModelObjective(ABC):
    """Base adapter: ``(trial, dials, context) -> float``.

    ``REQUIRED`` is the exact set of keyword parameters the wrapped runner must
    accept. ``__init__`` binds them against the runner's signature and raises
    ``TypeError`` on a mismatch, so the arity guard is on the REAL runner rather
    than on a 2-arg wrapper.
    """

    runner_ref: str = ""
    REQUIRED: tuple[str, ...] = ()

    def __init__(self, runner=None, *, metric: str):
        self.metric = str(metric)
        self.runner = runner if runner is not None else resolve_runner(self.runner_ref)
        if not callable(self.runner):
            raise TypeError(
                f"{type(self).__name__}: runner must be callable, got "
                f"{type(self.runner).__name__}")
        self._validate_runner(self.runner)

    def _validate_runner(self, runner) -> None:
        try:
            signature = inspect.signature(runner)
            signature.bind(**{name: None for name in self.REQUIRED})
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{type(self).__name__}: runner {self.runner_ref!r} must accept "
                f"{self.REQUIRED}: {exc}") from exc

    def _as_float(self, result):
        """Coerce a runner result (float / metric mapping / objective tuple)."""
        if isinstance(result, Mapping):
            if self.metric not in result:
                raise KeyError(
                    f"{type(self).__name__}: runner result has no metric "
                    f"{self.metric!r}")
            result = result[self.metric]
        if result is None:
            raise TypeError(
                f"{type(self).__name__}: runner {self.runner_ref!r} returned "
                "None; an objective runner must return a float or a metric "
                "mapping")
        if isinstance(result, tuple):
            result = result[0]
        return float(result)

    @abstractmethod
    def score(self, trial, dials, context) -> float:
        """Run the wrapped model and return its scalar metric."""

    def __call__(self, trial, dials, context=None) -> float:
        return self.score(trial, dict(dials or {}), dict(context or {}))


class CallableObjective(ModelObjective):
    """Runner contract: ``runner(dials, context) -> float | Mapping``."""

    REQUIRED = ("dials", "context")

    def score(self, trial, dials, context) -> float:
        return self._as_float(self.runner(dials, context))


class LayaObjective(ModelObjective):
    """Runner contract: the fenced ``objective_value`` (needs the live trial).

    ``objective_value(trial, run_fn, lease_store, champion_store,
    generation_id, model_key, objective_mode=None)``. The driver closures travel
    in ``context``; ``run_fn`` is the per-trial trainer that consumes the
    sampled dials.
    """

    runner_ref = "training.laya_hpo_runtime:objective_value"
    REQUIRED = ("trial", "run_fn", "lease_store", "champion_store",
                "generation_id", "model_key")

    def score(self, trial, dials, context) -> float:
        run_fn = context.get("run_fn") or context.get("train_fn")
        if run_fn is None:
            raise TypeError(
                "LayaObjective needs context['run_fn']: the per-trial trainer "
                "that consumes the sampled dials")
        result = self.runner(
            trial, run_fn, context.get("lease_store"),
            context.get("champion_store"), context.get("generation_id"),
            context.get("model_key"),
            objective_mode=context.get("objective_mode"))
        return self._as_float(result)


class TextObjective(ModelObjective):
    """Runner contract: ``run(config, output, run_tag, *, resume=False) -> Path``."""

    runner_ref = "model_tracks.run:run"
    REQUIRED = ("config", "output", "run_tag")

    def score(self, trial, dials, context) -> float:
        config, output = _materialize_trial(dials, context)
        artifact = self.runner(
            config, output, str(context.get("run_tag", "hpo")),
            resume=bool(context.get("resume", False)))
        return _read_metric(context, artifact, self.metric)


class GnnObjective(ModelObjective):
    """Runner contract: ``train(config_path, *, run_tag, resume=None) -> Path``."""

    runner_ref = "graph_tracks.train:train"
    REQUIRED = ("config_path", "run_tag")

    def score(self, trial, dials, context) -> float:
        config, _output = _materialize_trial(dials, context)
        artifact = self.runner(
            config, run_tag=str(context.get("run_tag", "hpo")),
            resume=context.get("resume"))
        return _read_metric(context, artifact, self.metric)


class CascadeObjective(ModelObjective):
    """Runner contract: ``report_cascade(ranked, relevant, decisions, output, *,
    track='cascade', ...)``."""

    runner_ref = "graph_tracks.report:report_cascade"
    REQUIRED = ("ranked", "relevant", "decisions", "output")

    def score(self, trial, dials, context) -> float:
        missing = [key for key in ("ranked", "relevant", "decisions")
                   if key not in context]
        if missing:
            raise TypeError(
                f"CascadeObjective needs context keys {missing!r} to score the "
                "cascade")
        _config, output = _materialize_trial(dials, context)
        artifact = self.runner(
            context["ranked"], context["relevant"], context["decisions"],
            output, track=str(context.get("track", "cascade")))
        return _read_metric(context, artifact, self.metric)


class BackboneObjective(ModelObjective):
    """The backbone runner ``training.training:run_hpo`` is a SWEEP driver
    (returns ``None``), not a per-trial objective.

    Construction fails loud rather than emitting an objective that returns
    ``None`` (or recursively launches a study). A real backbone objective must
    be a per-trial runner returning a float; tests inject a stub of that shape.
    """

    runner_ref = "training.training:run_hpo"
    REQUIRED = ("args", "data")

    def _validate_runner(self, runner) -> None:
        super()._validate_runner(runner)
        if _returns_none(runner):
            raise TypeError(
                "training.training:run_hpo is a sweep driver returning None; it "
                "cannot be a per-trial objective. Register a per-trial backbone "
                "runner that returns a float.")

    def score(self, trial, dials, context) -> float:  # pragma: no cover
        return self._as_float(self.runner(dials, context))


def _materialize_trial(dials, context):
    """Write the sampled dials to a trial config and return (config, output)."""
    import yaml

    workdir = Path(context.get("workdir") or tempfile.mkdtemp(
        prefix="hpo_trial_"))
    workdir.mkdir(parents=True, exist_ok=True)
    index = context.get("trial_number", context.get("trial", "x"))
    config = workdir / f"config_{index}.yaml"
    config.write_text(yaml.safe_dump(dict(dials), sort_keys=True),
                      encoding="utf-8")
    output = workdir / f"out_{index}"
    return config, output


def _read_metric(context, artifact, metric):
    """Extract a scored metric from a track run via the context's reader."""
    reader = context.get("read_metric")
    if reader is None:
        raise TypeError(
            "track objectives need context['read_metric']: a callable "
            "(artifact, metric) -> float that extracts the scored metric from "
            "the run's report/archive")
    return float(reader(artifact, metric))


__all__ = [
    "BackboneObjective",
    "CallableObjective",
    "CascadeObjective",
    "GnnObjective",
    "LayaObjective",
    "ModelObjective",
    "TextObjective",
    "resolve_runner",
]
