"""Offline public-API tests for the model-agnostic HPO registry.

Every project model key (laya/text/gnn/cascade) must expose a real search space
and a real objective descriptor; the registry itself is model-agnostic.
"""
from __future__ import annotations

import pytest

from cli import laya_hpo
from training import hpo_registry


def test_default_registry_keys_come_from_the_ssot():
    registry = hpo_registry.default_registry()
    keys = set(registry.keys())
    # Keys are SSOT-sourced (paths.yaml models) + track keys + the laya key.
    assert set(hpo_registry.ssot_model_keys()).issubset(keys)
    assert {"laya", "text", "gnn", "cascade"}.issubset(keys)
    assert keys == set(hpo_registry.model_keys())
    assert "minilm_l6" in keys  # a real project model backbone


@pytest.mark.parametrize("model_key", ["laya", "text", "gnn", "cascade",
                                       "minilm_l6"])
def test_every_model_has_a_real_space_and_objective(model_key):
    registry = hpo_registry.default_registry()
    space = registry.search_space(model_key)
    assert space, f"{model_key} search space is empty"
    for name, spec in space.items():
        assert spec["type"] in ("int", "float", "categorical"), name
        assert spec["target"] in ("config", "control"), name
    objective = registry.objective(model_key)
    assert objective.model_key == model_key
    assert objective.metric
    assert objective.direction in ("maximize", "minimize")
    assert ":" in objective.runner
    # The objective exposes the real (trial, context=None) -> float contract.
    obj = objective.objective()
    assert isinstance(obj, hpo_registry.TrialObjective)
    import inspect
    inspect.signature(obj).bind(object())


def test_laya_space_is_the_ssot_dials():
    registry = hpo_registry.default_registry()
    assert set(registry.search_space("laya")) == set(
        laya_hpo.load_space()["dials"])


def test_track_space_bounds_are_derived_from_the_ssot_yaml():
    space = hpo_registry.track_space("gnn")
    # config/graph_tracks_gnn.yaml declares learning_rate: 0.001; the dial's
    # bounds are derived from that value, not re-declared independently.
    dial = space["learning_rate"]
    assert dial["type"] == "float" and dial["log"] is True
    assert dial["lo"] < 0.001 < dial["hi"]
    assert dial["target"] == "config"


def test_objective_descriptor_resolves_a_real_runner():
    registry = hpo_registry.default_registry()
    assert callable(registry.objective("laya").resolve_runner())
    assert callable(registry.objective("gnn").resolve_runner())


def test_registry_describe_is_jsonable():
    import json

    registry = hpo_registry.default_registry()
    described = registry.describe()
    assert set(described) == set(registry.keys())
    json.dumps(described)
    for entry in described.values():
        assert entry["dials"]


def test_trial_objective_calls_the_evaluator_with_sampled_dials():
    from training.laya_hpo_runtime import sample_dials

    class _FakeTrial:
        def suggest_int(self, name, lo, hi):
            return lo

    captured = {}

    def evaluator(dials, context):
        captured.update(dials)
        captured["context"] = context
        return 0.42

    obj = hpo_registry.TrialObjective(
        model_key="m", space={"x": {"type": "int", "lo": 1, "hi": 3}},
        metric="score", direction="maximize", evaluator=evaluator)
    assert obj(_FakeTrial(), {"k": 1}) == 0.42
    assert captured["x"] == 1 and captured["context"] == {"k": 1}
    assert sample_dials(_FakeTrial(), {"dials": obj.space})  # space is samplable


def test_trial_objective_rejects_a_bad_evaluator_signature():
    with pytest.raises(TypeError, match="accept"):
        hpo_registry.TrialObjective(
            model_key="m", space={}, metric="s", direction="maximize",
            evaluator=lambda x, y, z: 0.0)


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
