"""Model-agnostic HPO registry, sourced from the project SSOTs.

Keys are NOT a parallel list: the canonical model keys come from
``config/paths.yaml models:`` (the project model registry); the track keys and
the lane key come from their own SSOT YAMLs
(``config/text_track.yaml``, ``config/graph_tracks_gnn.yaml``,
``config/graph_tracks_cascade.yaml``, ``config/laya_hpo_space.yaml``).

Each entry exposes a real ``search_space()`` (dial types/bounds DERIVED from the
SSOT values, never re-declared) and an ``objective()`` returning a callable
``TrialObjective`` with the contract ``(trial, context=None) -> float``. The
registry carries no model implementation: the objective's evaluator resolves the
model's own runner lazily.
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from pathlib import Path

# Per-track tunable knobs, named from their SSOT YAML; their bounds are DERIVED
# from the SSOT values at load (see `_dial_from_value`), never hardcoded.
_TRACK_FILES = {
    "text": "config/text_track.yaml",
    "gnn": "config/graph_tracks_gnn.yaml",
    "cascade": "config/graph_tracks_cascade.yaml",
}
_TRACK_TUNABLES = {
    "text": ("hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"),
    "gnn": ("hidden_dim", "epochs", "learning_rate", "weight_decay",
            "negative_margin", "max_grad_norm", "early_stopping_patience"),
    "cascade": ("hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"),
}
_TRACK_OBJECTIVES = {
    "text": ("recall_at_k", "maximize", "model_tracks.run:run",
             "training.hpo_objectives:TextObjective"),
    "gnn": ("pair_auc", "maximize", "graph_tracks.train:train",
            "training.hpo_objectives:GnnObjective"),
    "cascade": ("cascade_recall", "maximize",
                "graph_tracks.report:report_cascade",
                "training.hpo_objectives:CascadeObjective"),
}
# The per-trial objective adapter (``module:attr``) for each family. The adapter
# validates its runner's signature at construction; the backbone runner is a
# sweep driver and fails loud there rather than yielding an objective that
# crashes (or returns None) when invoked.
_ADAPTER_BACKBONE = "training.hpo_objectives:BackboneObjective"
_ADAPTER_LAYA = "training.hpo_objectives:LayaObjective"


def project_root() -> Path:
    from core.project_root import find_project_root
    return find_project_root(Path(__file__).resolve())


def ssot_model_keys() -> list[str]:
    """The canonical model keys from ``config/paths.yaml models:`` (SSOT)."""
    from core.common import load_config
    return list(load_config()["models"])


def training_fixed_params() -> dict:
    """The 0-width ``hpo.tpe_space`` entries: fixed values, never sampled.

    ``epochs: [10, 10]`` is a fixed budget, not a dial; sampling it yields an
    inert zero-width dimension. It is reported here instead so metadata stays
    complete without pretending it is searchable.
    """
    from core.common import hpo_cfg
    return {name: lo for name, (lo, hi) in hpo_cfg()["tpe_space"].items()
            if lo == hi}


def training_hpo_space() -> dict:
    """The main training HPO space from ``config/training.yaml hpo.tpe_space``.

    The SSOT already carries ranges; only the type/log flag is inferred from the
    values (int vs float; log for learning-rate-style knobs). A zero-width
    (``lo == hi``) entry is a fixed param, not a dial, and is skipped.
    """
    from core.common import hpo_cfg
    out = {}
    for name, (lo, hi) in hpo_cfg()["tpe_space"].items():
        if lo == hi:
            continue
        is_int = isinstance(lo, int) and isinstance(hi, int)
        spec = {"type": "int" if is_int else "float",
                "lo": int(lo) if is_int else float(lo),
                "hi": int(hi) if is_int else float(hi),
                "target": "config"}
        if not is_int and "lr" in name:
            spec["log"] = True
        out[name] = spec
    return out


def _dial_from_value(value) -> dict:
    """Derive a dial (type/bounds) from an SSOT scalar — no re-declared bound."""
    if isinstance(value, bool):
        return {"type": "categorical", "choices": [False, True]}
    if isinstance(value, int):
        lo = max(1, value // 2) if value > 0 else 0
        hi = max(value, value * 2, lo + 1)
        return {"type": "int", "lo": int(lo), "hi": int(hi)}
    number = float(value)
    if number > 0.0:
        return {"type": "float", "lo": number * 0.25, "hi": number * 4.0,
                "log": True}
    return {"type": "float", "lo": 0.0, "hi": 1.0}


def _read_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def track_space(track: str, *, root: Path | None = None) -> dict:
    """Read a track's tunable dials from its SSOT YAML (bounds derived)."""
    root = Path(root) if root else project_root()
    raw = _read_yaml(root / _TRACK_FILES[track])
    space = {}
    for knob in _TRACK_TUNABLES[track]:
        if knob not in raw:
            raise KeyError(f"{_TRACK_FILES[track]} has no knob {knob!r}")
        dial = _dial_from_value(raw[knob])
        dial["target"] = "config"
        space[knob] = dial
    return space


def laya_space() -> dict:
    """The laya dials from the search-space SSOT (lazy import: no cycle)."""
    from cli.laya_hpo import load_space
    return load_space()["dials"]


@dataclass(frozen=True)
class TrialObjective:
    """A real ``(trial, context=None) -> float`` objective for one model.

    ``adapter`` is a :class:`training.hpo_objectives.ModelObjective` whose
    constructor validated the wrapped RUNNER's signature (fail loud), so a bare
    runner with the wrong contract can never masquerade as an objective. This
    dataclass owns only sampling + the public ``(trial, context)`` entrypoint.
    """

    model_key: str
    space: dict
    metric: str
    direction: str
    adapter: object
    lower_is_better: bool = False

    def __post_init__(self):
        if not callable(self.adapter):
            raise TypeError(f"{self.model_key}: objective adapter must be callable")
        if not callable(getattr(self.adapter, "score", None)):
            raise TypeError(
                f"{self.model_key}: objective adapter must expose score("
                "trial, dials, context)")

    def sample(self, trial) -> dict:
        from training.laya_hpo_runtime import sample_dials
        return sample_dials(trial, {"dials": self.space})

    def __call__(self, trial, context=None) -> float:
        dials = self.sample(trial)
        return float(self.adapter(trial, dials, dict(context or {})))

    def as_dict(self):
        return {"model_key": self.model_key, "metric": self.metric,
                "direction": self.direction, "lower_is_better": self.lower_is_better,
                "dials": sorted(self.space)}


@dataclass(frozen=True)
class ObjectiveDescriptor:
    """How a trial is scored: metric, direction, runner and the objective."""

    model_key: str
    metric: str
    direction: str
    runner: str
    space: dict = field(default_factory=dict)
    lower_is_better: bool = False
    adapter: str = ""

    def resolve_runner(self):
        """Import and return the model's real runner callable."""
        if ":" not in self.runner:
            raise ValueError(
                f"runner must be 'module:attr', got {self.runner!r}")
        module_name, _, attr = self.runner.partition(":")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise RuntimeError(
                f"objective runner {self.runner!r} is not defined") from exc

    def objective(self) -> TrialObjective:
        """The real objective: an adapter validated against the real runner."""
        from training.hpo_objectives import CallableObjective, resolve_runner
        adapter_cls = (resolve_runner(self.adapter) if self.adapter
                       else CallableObjective)
        runner = resolve_runner(self.runner)
        adapter = adapter_cls(runner, metric=self.metric)
        return TrialObjective(
            model_key=self.model_key, space=copy.deepcopy(self.space),
            metric=self.metric, direction=self.direction,
            adapter=adapter, lower_is_better=self.lower_is_better)

    def as_dict(self):
        return {"model_key": self.model_key, "metric": self.metric,
                "direction": self.direction, "runner": self.runner,
                "lower_is_better": self.lower_is_better,
                "dials": sorted(self.space)}


@dataclass(frozen=True)
class ModelHpoSpec:
    """One model's HPO surface: a search space plus its objective contract."""

    model_key: str
    space: object  # dict, or a zero-arg callable returning a dict
    metric: str
    direction: str
    runner: str
    lower_is_better: bool = False
    adapter: str = ""

    def search_space(self):
        space = self.space() if callable(self.space) else self.space
        return copy.deepcopy(space)

    def objective(self):
        return ObjectiveDescriptor(
            model_key=self.model_key, metric=self.metric,
            direction=self.direction, runner=self.runner,
            space=self.search_space(), lower_is_better=self.lower_is_better,
            adapter=self.adapter)

    def as_dict(self):
        return {"model_key": self.model_key, "metric": self.metric,
                "direction": self.direction, "runner": self.runner,
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
                f"unknown model key {model_key!r}; registered: {self.keys()}") from exc

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


def model_keys(*, root: Path | None = None) -> list[str]:
    """Every registered key: SSOT backbones + track keys + the laya lane key."""
    keys = list(ssot_model_keys())
    keys.extend(_TRACK_FILES)  # text, gnn, cascade
    from cli.laya_hpo import load_space
    keys.append(load_space()["model_key"])
    seen = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return seen


def default_registry(*, root: Path | None = None) -> HpoModelRegistry:
    """The SSOT-sourced registry (no parallel key list, no re-declared bounds)."""
    registry = HpoModelRegistry()
    # Every project model backbone shares the training HPO space/objective. The
    # backbone objective adapter fails loud at construction (run_hpo is a sweep
    # driver, not a per-trial objective).
    for key in ssot_model_keys():
        registry.register(ModelHpoSpec(
            model_key=key, space=training_hpo_space, metric="rand_index_proxy",
            direction="maximize", runner="training.training:run_hpo",
            adapter=_ADAPTER_BACKBONE))
    # The laya lane's own key.
    from cli.laya_hpo import load_space
    laya_key = load_space()["model_key"]
    registry.register(ModelHpoSpec(
        model_key=laya_key, space=laya_space, metric="dev_accuracy",
        direction="maximize",
        runner="training.laya_hpo_runtime:objective_value",
        adapter=_ADAPTER_LAYA))
    # The track keys, spaces read from their SSOT YAMLs.
    for track, (metric, direction, runner, adapter) in _TRACK_OBJECTIVES.items():
        registry.register(ModelHpoSpec(
            model_key=track, space=(lambda t=track: track_space(t, root=root)),
            metric=metric, direction=direction, runner=runner, adapter=adapter))
    return registry


__all__ = [
    "HpoModelRegistry",
    "ModelHpoSpec",
    "ObjectiveDescriptor",
    "TrialObjective",
    "default_registry",
    "laya_space",
    "model_keys",
    "ssot_model_keys",
    "track_space",
    "training_fixed_params",
    "training_hpo_space",
]
