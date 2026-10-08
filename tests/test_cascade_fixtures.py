"""Prepared suite fixtures reflect the text/gnn_only/cascade contract.

The retired ``hybrid`` lane must not survive in the checked-in prepared
fixtures: each regenerated suite dir carries ``cascade.yaml`` (never
``hybrid.yaml``), and its shared projection binds the trained-scorer inputs the
cascade consumes (``gnn_only``) rather than the retired fused ``hybrid`` job.

These assertions are offline and load only the config models + projection JSON
(the CPU smoke's own load path); no GPU, network, or training.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.common import TRAIN_ROOT
from graph_tracks.config import load_config as load_graph_config
from graph_tracks.config import load_text_config
from model_tracks.config import load_config as load_suite_config

#: Smoke children a checked-in fixture may carry (portable, checkout-relative).
SMOKE_FIXTURES = ("data/prepared/smoke_200", "data/prepared/smoke_500")
#: Prepared setup directories regenerated to the current contract.
REGENERATED = (*SMOKE_FIXTURES, "data/track_setup")

#: Lane-config inputs a smoke child re-roots under its own tree.
LANE_INPUT_KEYS = ("listings", "pairs", "input_manifest", "text_index", "gnn_checkpoint")


def _setup(name: str) -> Path:
    return Path(TRAIN_ROOT) / name


@pytest.mark.parametrize("name", REGENERATED)
def test_regenerated_setup_has_cascade_not_hybrid(name):
    setup = _setup(name)
    if not setup.is_dir():
        pytest.skip(f"{name} not present in this checkout")
    assert (setup / "cascade.yaml").is_file(), f"{name} missing cascade.yaml"
    assert not (setup / "hybrid.yaml").exists(), f"{name} still carries hybrid.yaml"


@pytest.mark.parametrize("name", REGENERATED)
def test_cascade_and_companion_configs_load(name):
    setup = _setup(name)
    if not setup.is_dir():
        pytest.skip(f"{name} not present in this checkout")
    cascade = load_graph_config(setup / "cascade.yaml", expected_track="cascade")
    # The cascade composes the trained text ranker + trained gnn_only scorer
    # and fuses no embedding, so it must not declare a text cache. Its two
    # trained inputs are declared together (an explicit sibling results root) or
    # both left null; the null form makes the worker resolve the SAME-RUN
    # artifacts its prerequisite tracks wrote, which is what the committed smoke
    # fixtures need (their results root exists only at run time).
    assert bool(cascade.text_index) == bool(cascade.gnn_checkpoint)
    assert cascade.text_cache is None
    gnn = load_graph_config(setup / "gnn_only.yaml", expected_track="gnn_only")
    assert gnn.track == "gnn_only"
    assert gnn.text_cache is None
    assert load_text_config(setup / "text.yaml").track == "text"


def test_committed_lane_configs_are_portable_and_resolve(monkeypatch):
    """A checked-in smoke names checkout-relative paths that resolve here.

    The retired failure mode was a fixture carrying the machine that built it
    (``/tmp/...``) or a path relative to nothing. Every declared input must
    resolve from the checkout root; a tracked index/checkpoint is an output the
    smoke itself produces, so only its form is asserted.
    """
    checkout = TRAIN_ROOT
    for name in SMOKE_FIXTURES:
        setup = _setup(name)
        if not setup.is_dir():
            pytest.skip(f"{name} not present in this checkout")
        for track in ("gnn_only", "cascade"):
            lane = load_graph_config(setup / f"{track}.yaml", expected_track=track)
            for key in LANE_INPUT_KEYS:
                value = getattr(lane, key, None)
                if not value:
                    continue
                assert not Path(value).is_absolute(), f"{name}/{track}: {key} is machine-local"
                assert Path(value).parts[:2] == Path(name).parts[:2], \
                    f"{name}/{track}: {key} does not name this suite's tree"
            for key in ("listings", "pairs", "input_manifest"):
                path = checkout / getattr(lane, key)
                assert path.is_file(), f"{name}/{track}: {key} does not resolve ({path})"
        suite = load_suite_config(setup / "suite.yaml")
        assert not Path(suite.setup_dir).is_absolute()
        assert not Path(suite.text_bundle).is_absolute()
        assert (checkout / suite.text_bundle).is_file()


def test_smoke_200_cascade_consumes_same_run_artifacts():
    """The S fixture leaves the cascade inputs null, like ``gnn_only.yaml``.

    Pinning them into the prepared setup (``data/prepared/smoke_200/text_index``)
    made the cascade worker raise FileNotFoundError: that path is an artifact of
    a run, not of the setup tree, and a smoke's results root exists only at run
    time. Both fixtures therefore stay null and the worker resolves the run's
    own ``text`` / ``gnn_only`` outputs behind the barrier.
    """
    setup = _setup("data/prepared/smoke_200")
    if not (setup / "cascade.yaml").is_file():
        pytest.skip("smoke_200 not present in this checkout")
    cascade = load_graph_config(setup / "cascade.yaml", expected_track="cascade")
    assert cascade.text_index is None and cascade.gnn_checkpoint is None
    gnn = load_graph_config(setup / "gnn_only.yaml", expected_track="gnn_only")
    assert gnn.text_index is None and gnn.gnn_checkpoint is None


def test_smoke_200_suite_binds_cpu_and_cascade_configs():
    setup = _setup("data/prepared/smoke_200")
    if not (setup / "suite.yaml").is_file():
        pytest.skip("smoke_200 not present in this checkout")
    suite = load_suite_config(setup / "suite.yaml")
    assert suite.device == "cpu"
    assert suite.setup_dir == "data/prepared/smoke_200"
    for track in ("gnn_only", "cascade"):
        assert (setup / f"{track}.yaml").is_file()


def test_shared_projection_binds_the_cascade_inputs_not_hybrid():
    setup = _setup("data/prepared/smoke_200")
    projection = setup / "shared_training_projection.json"
    if not projection.is_file():
        pytest.skip("smoke_200 projection not present in this checkout")
    bindings = json.loads(projection.read_text())["track_bindings"]
    assert "hybrid" not in bindings, "projection still binds the retired hybrid track"
    # The cascade consumes the trained gnn_only scorer; the gnn_only lane is the
    # only shared-graph track the projection declares.
    assert set(bindings) == {"gnn_only"}
