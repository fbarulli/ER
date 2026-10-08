"""Offline public-API tests for the model-agnostic HPO registry.

Every project model key (laya/text/gnn/cascade) must expose a real search space
and a real objective descriptor; the registry itself is model-agnostic.
"""
from __future__ import annotations

import pytest

from cli import laya_hpo
from training import hpo_registry


def test_default_registry_has_every_model_key():
    registry = hpo_registry.default_registry()
    assert registry.keys() == ["cascade", "gnn", "laya", "text"]
    assert set(registry.keys()) == set(hpo_registry.MODEL_KEYS)


@pytest.mark.parametrize("model_key", ["laya", "text", "gnn", "cascade"])
def test_every_model_has_a_real_space_and_objective(model_key):
    registry = hpo_registry.default_registry()
    space = registry.search_space(model_key)
    assert space, f"{model_key} search space is empty"
    for name, spec in space.items():
        assert spec["type"] in ("int", "float", "categorical"), name
    objective = registry.objective(model_key)
    assert objective.model_key == model_key
    assert objective.metric
    assert objective.direction in ("maximize", "minimize")
    assert ":" in objective.runner
    assert isinstance(objective.lower_is_better, bool)


def test_laya_space_is_the_ssot_dials():
    registry = hpo_registry.default_registry()
    assert set(registry.search_space("laya")) == set(
        laya_hpo.load_space()["dials"])


def test_objective_descriptor_resolves_a_real_runner():
    registry = hpo_registry.default_registry()
    # Light, always-importable runners are resolved end to end.
    assert callable(registry.objective("laya").resolve_runner())
    assert callable(registry.objective("gnn").resolve_runner())


def test_registry_describe_is_jsonable():
    import json

    described = hpo_registry.default_registry().describe()
    assert set(described) == {"laya", "text", "gnn", "cascade"}
    json.dumps(described)
    for entry in described.values():
        assert entry["dials"]


def test_unknown_model_key_fails_loud():
    registry = hpo_registry.default_registry()
    with pytest.raises(KeyError, match="unknown model key"):
        registry.get("bogus")


def test_register_custom_spec_and_duplicate_overwrite():
    registry = hpo_registry.HpoModelRegistry()
    spec = hpo_registry.ModelHpoSpec(
        model_key="custom", space={"x": {"type": "int", "lo": 1, "hi": 3}},
        metric="score", direction="maximize", runner="json:loads")
    registry.register(spec)
    assert registry.keys() == ["custom"]
    assert callable(registry.objective("custom").resolve_runner())
    registry.register(spec)
    assert registry.keys() == ["custom"]  # idempotent re-register


def test_objective_descriptor_rejects_bad_runner_reference():
    descriptor = hpo_registry.ObjectiveDescriptor(
        model_key="x", metric="m", direction="maximize", runner="no-colon")
    with pytest.raises(ValueError, match="module:attr"):
        descriptor.resolve_runner()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
