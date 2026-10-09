"""Programmatic SSOT-parity guarantees for the laya HPO lane.

These are the invariants the mission asserts by construction, not by eye: the
search-space dials, the model-key registry, the track bounds and the study
identity each resolve from exactly one SSOT. A violation fails here loudly
rather than surfacing as a silently dead dial or a duplicated registry.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cli import laya_hpo, laya_lane
from core.laya_config import FinetuneSpec
from training import hpo_registry

CONFIG_FIELDS = set(laya_lane.FINETUNE_CONFIG_FIELDS)
CONTROL_FIELDS = set(laya_lane.FINETUNE_CONTROL_FIELDS)
SPEC_FIELDS = set(FinetuneSpec.model_fields)


def test_config_and_control_field_tuples_are_disjoint():
    """A name routed to both channels is an ambiguous SSOT; forbid it."""
    assert not (CONFIG_FIELDS & CONTROL_FIELDS)


@pytest.mark.parametrize("name", list(laya_hpo.load_space()["dials"]))
def test_every_dial_is_a_routed_ssot_field(name):
    dial = laya_hpo.load_space()["dials"][name]
    assert name in SPEC_FIELDS, f"{name} is not a FinetuneSpec field"
    channel = CONTROL_FIELDS if dial["target"] == "control" else CONFIG_FIELDS
    assert name in channel, (
        f"{name} targets {dial['target']!r} but is not in the channel tuple; "
        "it would be a dead/no-op dial")


def test_no_orphan_finetune_field_is_a_routable_dial():
    """Every field the dials may target is in exactly one routed channel.

    The union of the two field tuples must remain the trainer surface; an
    unrouted FinetuneSpec field is allowed (non-HPO knobs), but a name that
    appears in a dial MUST be routed (asserted above).
    """
    assert CONFIG_FIELDS <= SPEC_FIELDS
    assert CONTROL_FIELDS <= SPEC_FIELDS


def test_every_gated_dial_preserves_the_off_default():
    """An OFF gate must leave its dependents at the SSOT default (no-op safe)."""
    space = laya_hpo.load_space()
    gated = {name: spec for name, spec in space["dials"].items()
             if "when" in spec}
    assert gated, "expected gated dials (r_drop/drop_path/batch_size_ramp)"
    default_control = {name: f"SSOT::{name}"
                       for name, spec in space["dials"].items()
                       if spec["target"] == "control"}

    class _OffGateTrial:
        """Every categorical (the gates) resolves to the OFF default."""

        def suggest_int(self, name, lo, hi, log=False):
            return lo

        def suggest_float(self, name, lo, hi, log=False):
            return lo

        def suggest_categorical(self, name, choices):
            return False

    sampled = laya_hpo.sample_dials(_OffGateTrial(), space)
    _, control = laya_hpo.route_dials({}, default_control, sampled, space)
    for name, spec in gated.items():
        assert name not in sampled, f"{name} sampled while its gate is OFF"
        assert control[name] == f"SSOT::{name}", (
            f"OFF gate changed dependent {name}; default not preserved")


def test_model_keys_come_from_one_registry():
    """paths.yaml models + track keys + the laya space key; no second list."""
    from core.common import load_config

    space = laya_hpo.load_space()
    expected = list(load_config()["models"]) + ["text", "gnn", "cascade",
                                                space["model_key"]]
    assert hpo_registry.model_keys() == expected
    assert set(hpo_registry.default_registry().keys()) == set(expected)
    # The laya space key is the model boundary; it is not hardcoded in code.
    assert space["model_key"] == "laya"
    assert hpo_registry.laya_space() == space["dials"]


def test_track_space_bounds_are_derived_from_the_track_yaml():
    """No duplicated bounds: each dial's lo/hi is a function of its SSOT value."""
    root = hpo_registry.project_root()
    for track, rel in hpo_registry._TRACK_FILES.items():
        raw = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        derived = hpo_registry.track_space(track)
        assert set(derived) == set(hpo_registry._TRACK_TUNABLES[track])
        for knob, dial in derived.items():
            assert knob in raw, f"{rel} declares no {knob}"
            value = raw[knob]
            expected = hpo_registry._dial_from_value(value)
            for key, expected_value in expected.items():
                assert dial[key] == expected_value, (track, knob, key)
            assert dial["target"] == "config"


def test_study_identity_resolves_from_one_source(monkeypatch):
    """The shared study name is generation_study_name (the control plane)."""
    from training.hpo_control_plane import generation_study_name

    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-parity")
    generation, key, name = laya_hpo.study_identity()
    assert name == generation_study_name(generation_id=generation, model_key=key)
    assert key == laya_hpo.load_space()["model_key"]
    assert generation == "gen-parity"


def test_space_is_bound_in_the_paths_ssot():
    from core.common import F

    assert "laya_hpo_space" in F
    assert Path(F["laya_hpo_space"]).name == "laya_hpo_space.yaml"
    assert Path(laya_hpo.space_path()).resolve() == \
        Path(F["laya_hpo_space"]).resolve()


def test_dial_space_version_and_count_are_pinned():
    space = laya_hpo.load_space()
    assert space["space_version"] == 2
    assert len(space["dials"]) == 33
