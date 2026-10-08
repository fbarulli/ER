"""Hardening tests for the laya HPO space/controls (findings #2/#7/#8).

They exercise the public loader/sampler/router APIs, never private internals,
and assert the behaviour the lane relies on:

* every routed dial reaches a channel that consumes it (no tuned-but-ignored);
* a conditional gate is respected on sample AND route, and a dependent declared
  before its gate still resolves (order-independent sampling);
* a ``plateau`` draw can never fail a trial because of warmup.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cli import laya_hpo, laya_lane
from core import laya_controls
from core.laya_config import FinetuneSpec

REPO = Path(__file__).resolve().parents[1]

_CONTROL_FIELDS = tuple(laya_lane.FINETUNE_CONTROL_FIELDS)
_CONFIG_FIELDS = tuple(laya_lane.FINETUNE_CONFIG_FIELDS)


def _defaults():
    spec = FinetuneSpec()
    return ({name: getattr(spec, name) for name in _CONFIG_FIELDS},
            {name: getattr(spec, name) for name in _CONTROL_FIELDS})


def _candidate_values(dial):
    if dial["type"] == "categorical":
        return list(dial["choices"])
    return [dial["lo"], dial["hi"]]


class _FakeTrial:
    """Returns the configured pick for one dial; the OFF default otherwise."""

    def __init__(self, picks):
        self.picks = picks

    def suggest_int(self, name, lo, hi, log=False):
        return self.picks.get(name, lo)

    def suggest_float(self, name, lo, hi, log=False):
        return self.picks.get(name, lo)

    def suggest_categorical(self, name, choices):
        return self.picks.get(name, choices[0])


def _control_source():
    return "\n".join([
        (REPO / "src/core/laya_controls.py").read_text(encoding="utf-8"),
        (REPO / "src/cli/laya_lane.py").read_text(encoding="utf-8"),
    ])


def test_sampled_space_is_a_valid_laya_hpo_space():
    laya_hpo.validate_space(laya_hpo.load_space())


def test_sampling_resolves_a_dependent_declared_before_its_gate():
    # A reordered spec must still resolve dependents (not silently drop them).
    space = copy.deepcopy(laya_hpo.load_space())
    dials = space["dials"]
    alpha = dials.pop("r_drop_alpha")
    reordered = {}
    for name, spec in dials.items():
        if name == "r_drop":
            reordered["r_drop_alpha"] = alpha  # dependent BEFORE its gate
        reordered[name] = spec
    space["dials"] = reordered
    sampled = laya_hpo.sample_dials(_FakeTrial({"r_drop": True}), space)
    assert sampled["r_drop"] is True
    assert "r_drop_alpha" in sampled


def test_route_dials_respects_a_conditional_gate():
    space = laya_hpo.load_space()
    _cfg, control = _defaults()
    # A dependent without its gate must NOT change the SSOT default value.
    _cfg, gated_off = laya_hpo.route_dials(
        {}, control, {"r_drop_alpha": 0.9}, space)
    assert gated_off["r_drop_alpha"] == control["r_drop_alpha"]
    # With the gate on, the dependent routes.
    _cfg, gated_on = laya_hpo.route_dials(
        {}, control, {"r_drop": True, "r_drop_alpha": 0.9}, space)
    assert gated_on["r_drop_alpha"] == 0.9


def test_plateau_draw_is_runnable_even_with_warmup():
    space = laya_hpo.load_space()
    _cfg, control = _defaults()
    _cfg, routed = laya_hpo.route_dials(
        {}, control, {"lr_scheduler": "plateau", "warmup_frac": 0.1}, space)
    controls = laya_controls.TrainingControls.parse(routed, {})
    factory = controls.scheduler(object(), total_updates=100, min_lr=0.0)
    assert factory.is_plateau and factory.warmup == 0


def test_every_dial_samples_routes_and_is_consumed_downstream():
    """The classic HPO bug guard: sample -> route -> consumed.

    For EVERY dial and EVERY categorical choice (or numeric bound), the sampled
    value must land in a channel that reads it (control logic source, or the
    TrainConfig field tuple) — no tuned-but-ignored dial.
    """
    space = laya_hpo.load_space()
    dials = space["dials"]
    base_config, base_control = _defaults()
    control_source = _control_source()

    for name, dial in dials.items():
        gate = dial.get("when")
        for value in _candidate_values(dial):
            picks = {name: value}
            if gate is not None:
                picks[gate["dial"]] = gate.get("equals", True)
            sampled = laya_hpo.sample_dials(_FakeTrial(picks), space)
            assert name in sampled, f"{name} was not sampled"
            config, control = laya_hpo.route_dials(
                base_config, base_control, sampled, space)
            if dial["target"] == "control":
                assert name in control, f"control dial {name!r} was not routed"
                assert (f'control["{name}"]' in control_source
                        or f'control.get("{name}")' in control_source), (
                    f"control dial {name!r} is never read downstream")
            else:
                assert name in config, f"config dial {name!r} was not routed"
                assert name in _CONFIG_FIELDS, (
                    f"config dial {name!r} never reaches TrainConfig")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
