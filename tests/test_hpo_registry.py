"""Offline public-API tests for the model-agnostic HPO registry.

Every project model key (laya/text/gnn/cascade) must expose a real search space
and a REAL objective that can be invoked; the registry itself is model-agnostic.
The adapters validate the wrapped runner's signature at construction, so a
runner with the wrong calling convention fails loud instead of crashing later.
"""
from __future__ import annotations

import pytest

from cli import laya_hpo
from training import hpo_objectives, hpo_registry


def test_default_registry_keys_come_from_the_ssot():
    registry = hpo_registry.default_registry()
    keys = set(registry.keys())
    # Keys are SSOT-sourced (paths.yaml models) + track keys + the laya key.
    assert set(hpo_registry.ssot_model_keys()).issubset(keys)
    assert {"laya", "text", "gnn", "cascade"}.issubset(keys)
    assert keys == set(hpo_registry.model_keys())
    assert "minilm_l6" in keys  # a real project model backbone


@pytest.mark.parametrize("model_key", ["laya", "text", "gnn", "cascade"])
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
    # Building the objective validates the REAL runner's signature (fail loud).
    obj = objective.objective()
    assert isinstance(obj, hpo_registry.TrialObjective)
    import inspect
    inspect.signature(obj).bind(object())


def test_backbone_objective_fails_loud_at_construction():
    # run_hpo is a sweep driver returning None; it must never masquerade as a
    # per-trial objective. Construction raises instead of crashing on call.
    registry = hpo_registry.default_registry()
    with pytest.raises(TypeError, match="run_hpo"):
        registry.objective("minilm_l6").objective()


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


def test_training_space_skips_zero_width_fixed_params():
    space = hpo_registry.training_hpo_space()
    fixed = hpo_registry.training_fixed_params()
    # config/training.yaml pins epochs: [10, 10]; it is fixed, not a dial.
    assert "epochs" not in space
    assert fixed.get("epochs") == 10
    for name, spec in space.items():
        if spec["type"] in ("int", "float"):
            assert spec["lo"] < spec["hi"], name


def test_trial_objective_calls_the_evaluator_with_sampled_dials():
    class _FakeTrial:
        def suggest_int(self, name, lo, hi, log=False):
            return lo

    captured = {}

    def evaluator(dials, context):
        captured.update(dials)
        captured["context"] = context
        return 0.42

    adapter = hpo_objectives.CallableObjective(evaluator, metric="score")
    obj = hpo_registry.TrialObjective(
        model_key="m", space={"x": {"type": "int", "lo": 1, "hi": 3}},
        metric="score", direction="maximize", adapter=adapter)
    assert obj(_FakeTrial(), {"k": 1}) == 0.42
    assert captured["x"] == 1 and captured["context"] == {"k": 1}


def test_adapter_rejects_a_bad_runner_signature():
    # The arity guard is on the RUNNER, not on a 2-arg wrapper.
    with pytest.raises(TypeError, match="must accept"):
        hpo_objectives.CallableObjective(lambda x, y, z: 0.0, metric="s")
    with pytest.raises(TypeError, match="must accept"):
        hpo_objectives.LayaObjective(lambda: 0.0, metric="s")


# ── every adapter actually RUNS against a stubbed runner (no GPU) ──────────


def _text_stub(seen):
    def run(config, output, run_tag, *, resume=False):
        seen.update(config=str(config), output=str(output), run_tag=run_tag)
        return output
    return run


def _gnn_stub(seen):
    def train(config_path, *, run_tag, resume=None):
        seen.update(config_path=str(config_path), run_tag=run_tag)
        return config_path
    return train


def _cascade_stub(seen):
    def report_cascade(ranked, relevant, decisions, output, *, track="cascade"):
        seen.update(output=str(output), track=track)
        return output
    return report_cascade


def _laya_stub(seen):
    def objective_value(trial, run_fn, lease_store, champion_store,
                        generation_id, model_key, objective_mode=None):
        seen.update(generation_id=generation_id, model_key=model_key)
        return 0.77
    return objective_value


@pytest.mark.parametrize("family", ["callable", "laya", "text", "gnn", "cascade"])
def test_every_adapter_invokes_its_stub_runner(family, tmp_path):
    seen = {}
    trial = object()
    dials = {"some_dial": 1}
    def reader(artifact, metric):
        return 0.61

    context = {"workdir": tmp_path, "read_metric": reader,
               "trial_number": 7}
    if family == "callable":
        adapter = hpo_objectives.CallableObjective(
            lambda dials, context: 0.5, metric="score")
    elif family == "laya":
        adapter = hpo_objectives.LayaObjective(_laya_stub(seen), metric="score")
        context.update(run_fn=lambda t: (0.7, 0.1, "ck"),
                       generation_id="gen", model_key="laya")
    elif family == "text":
        adapter = hpo_objectives.TextObjective(_text_stub(seen), metric="score")
    elif family == "gnn":
        adapter = hpo_objectives.GnnObjective(_gnn_stub(seen), metric="score")
    else:
        adapter = hpo_objectives.CascadeObjective(
            _cascade_stub(seen), metric="score")
        context.update(ranked=["r"], relevant=["v"], decisions=["d"])
    value = adapter(trial, dials, context)
    assert isinstance(value, float)
    if family != "callable":
        assert seen  # the underlying runner really ran


def test_laya_adapter_requires_the_run_fn_context():
    adapter = hpo_objectives.LayaObjective(_laya_stub({}), metric="score")
    with pytest.raises(TypeError, match="run_fn"):
        adapter(object(), {}, {})


def test_track_adapter_requires_a_metric_reader(tmp_path):
    adapter = hpo_objectives.GnnObjective(_gnn_stub({}), metric="score")
    with pytest.raises(TypeError, match="read_metric"):
        adapter(object(), {}, {"workdir": tmp_path})


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
