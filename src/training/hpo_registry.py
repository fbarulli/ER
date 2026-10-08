"""Model-agnostic HPO registry: one entry per project model key.

Every model the project trains (``laya``, ``text``, ``gnn``, ``cascade``) has a
real ``search_space()`` (the tunable dials with types/bounds/choices) and a real
``objective()`` descriptor naming the metric, direction and the concrete runner
that evaluates a trial. The registry is deliberately model-agnostic: the lane,
the workers and the control plane consume it by ``model_key`` and never branch
on a model name.

The registry carries no model implementation: ``objective().resolve_runner()``
imports the model's own runner lazily, so importing this module stays light and
there is exactly one place that says which function scores which model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

MODEL_KEYS = ("laya", "text", "gnn", "cascade")


def _laya_space():
    """The laya dials from the search-space SSOT (lazy import: no cycle)."""
    from cli.laya_hpo import load_space
    return load_space()["dials"]


# The non-laya spaces are declarative registry data over each lane's real
# config keys (config/graph_tracks_gnn.yaml, config/text_track.yaml).
_TEXT_SPACE = {
    "hnsw_m": {"type": "int", "lo": 8, "hi": 64},
    "hnsw_ef_construction": {"type": "int", "lo": 64, "hi": 400},
    "hnsw_ef_search": {"type": "int", "lo": 32, "hi": 256},
}
_GNN_SPACE = {
    "hidden_dim": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "epochs": {"type": "int", "lo": 4, "hi": 40},
    "learning_rate": {"type": "float", "lo": 1.0e-4, "hi": 1.0e-2, "log": True},
    "weight_decay": {"type": "float", "lo": 0.0, "hi": 1.0e-2},
    "negative_margin": {"type": "float", "lo": 0.0, "hi": 1.0},
    "max_grad_norm": {"type": "float", "lo": 0.1, "hi": 5.0},
    "early_stopping_patience": {"type": "int", "lo": 1, "hi": 8},
}
_CASCADE_SPACE = {
    "hnsw_m": {"type": "int", "lo": 8, "hi": 64},
    "hnsw_ef_construction": {"type": "int", "lo": 64, "hi": 400},
    "hnsw_ef_search": {"type": "int", "lo": 32, "hi": 256},
}


@dataclass(frozen=True)
class ObjectiveDescriptor:
    """How a trial is scored: metric, direction and the concrete runner."""

    model_key: str
    metric: str
    direction: str
    runner: str
    lower_is_better: bool = False

    def resolve_runner(self):
        """Import and return the model's real runner callable."""
        if ":" not in self.runner:
            raise ValueError(
                f"runner must be 'module:attr', got {self.runner!r}")
        module_name, _, attr = self.runner.partition(":")
        import importlib

        module = importlib.import_module(module_name)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise RuntimeError(
                f"objective runner {self.runner!r} is not defined") from exc

    def as_dict(self):
        return {"model_key": self.model_key, "metric": self.metric,
                "direction": self.direction, "runner": self.runner,
                "lower_is_better": self.lower_is_better}


@dataclass(frozen=True)
class ModelHpoSpec:
    """One model's HPO surface: a search space plus its objective contract."""

    model_key: str
    space: object  # dict, or a zero-arg callable returning a dict
    metric: str
    direction: str
    runner: str
    lower_is_better: bool = False

    def search_space(self):
        space = self.space() if callable(self.space) else self.space
        return copy.deepcopy(space)

    def objective(self):
        return ObjectiveDescriptor(
            model_key=self.model_key, metric=self.metric,
            direction=self.direction, runner=self.runner,
            lower_is_better=self.lower_is_better)

    def as_dict(self):
        return {"model_key": self.model_key,
                "metric": self.metric, "direction": self.direction,
                "runner": self.runner,
                "dials": sorted(self.search_space())}


class HpoModelRegistry:
    """Name -> ``ModelHpoSpec`` registry; the lane's model-agnostic boundary."""

    def __init__(self, specs=()):
        self._specs: dict[str, ModelHpoSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelHpoSpec):
        if not spec.model_key:
            raise ValueError("model spec needs a model_key")
        self._specs[spec.model_key] = spec
        return self

    def get(self, model_key):
        try:
            return self._specs[model_key]
        except KeyError as exc:
            raise KeyError(
                f"unknown model key {model_key!r}; "
                f"registered: {self.keys()}") from exc

    def keys(self):
        return sorted(self._specs)

    def __contains__(self, model_key):
        return model_key in self._specs

    def __len__(self):
        return len(self._specs)

    def search_space(self, model_key):
        return self.get(model_key).search_space()

    def objective(self, model_key):
        return self.get(model_key).objective()

    def describe(self):
        return {key: self.get(key).as_dict() for key in self.keys()}

    def as_dict(self):
        return self.describe()


def default_registry() -> HpoModelRegistry:
    """The four project model keys, each with a real space + objective."""
    return HpoModelRegistry([
        ModelHpoSpec(
            model_key="laya", space=_laya_space, metric="dev_accuracy",
            direction="maximize", runner="training.laya_hpo_runtime:objective_value"),
        ModelHpoSpec(
            model_key="text", space=_TEXT_SPACE, metric="recall_at_k",
            direction="maximize", runner="model_tracks.run:run"),
        ModelHpoSpec(
            model_key="gnn", space=_GNN_SPACE, metric="pair_auc",
            direction="maximize", runner="graph_tracks.train:train"),
        ModelHpoSpec(
            model_key="cascade", space=_CASCADE_SPACE, metric="cascade_recall",
            direction="maximize", runner="graph_tracks.report:report_cascade"),
    ])


__all__ = [
    "MODEL_KEYS",
    "HpoModelRegistry",
    "ModelHpoSpec",
    "ObjectiveDescriptor",
    "default_registry",
]
