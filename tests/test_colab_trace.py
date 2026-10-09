"""Pin the Colab lane's shared trace propagation (defect audit 2026-10-08).

Two regressions are pinned here:

1. Every spawned training worker's environment must spread
   ``run_trace_env(lane=...)`` so the lane reuses the parent run's ONE
   ``training_trace.csv`` and run id instead of fragmenting per-worker traces.
   ``src/cli/colab.py`` builds worker envs in four remote scripts; this test
   verifies each one spreads the pins and imports the resolver.

2. The default all-track suite config is config-owned
   (``training_cfg().bundle.suite_config``) and must resolve to the same file
   as the former ``TRAIN_ROOT / 'config/model_tracks.yaml'`` literal.

Hermetic: no GPU, no Colab VM, no network. Source inspection plus monkeypatch.
"""
from __future__ import annotations

import re
from pathlib import Path

from cli import colab
from core import common, tracing

ROOT = Path(__file__).resolve().parents[1]
COLAB_PATH = ROOT / "src" / "cli" / "colab.py"


def test_run_trace_env_carries_the_two_trace_pins(monkeypatch):
    """The resolver returns the run's ONE destination and run id, plus the lane."""
    monkeypatch.setattr(
        tracing, "trace_path", lambda: Path("/run/results/logs/training_trace.csv")
    )
    monkeypatch.setattr(tracing, "resolve_run_id", lambda: "run-abc123")

    pins = tracing.run_trace_env(lane="worker_x")

    assert pins[tracing.TRACE_PATH_ENV] == "/run/results/logs/training_trace.csv"
    assert pins[tracing.TRACE_RUN_ENV] == "run-abc123"
    assert pins[tracing.TRACE_LANE_ENV] == "worker_x"


def test_every_spawned_worker_env_spreads_the_trace_pins():
    """All four remote worker-env dicts spread the pins with their lane label."""
    source = COLAB_PATH.read_text(encoding="utf-8")

    spreads = re.findall(r"\*\*run_trace_env\(lane=([a-zA-Z_][a-zA-Z0-9_]*)\)", source)
    # Four sites: multi-worker train (training_name), single-worker train
    # (training_name), HPO (model_key), and mixed (label).
    assert spreads == ["training_name", "training_name", "model_key", "label"], spreads

    # Each of the four remote scripts must import the resolver it spreads.
    assert source.count("from core.tracing import run_trace_env") == 4


def test_suite_config_resolves_to_the_same_file_as_before():
    """The config-owned default resolves byte-identically to the old literal."""
    expected = colab.TRAIN_ROOT / "config/model_tracks.yaml"
    assert colab._suite_config_path() == expected
    assert common.training_cfg().bundle.suite_config == "config/model_tracks.yaml"

    # The former literal is gone from the source: config owns the default now.
    source = COLAB_PATH.read_text(encoding="utf-8")
    assert "TRAIN_ROOT / 'config/model_tracks.yaml'" not in source
    assert "TRAIN_ROOT/'config/model_tracks.yaml'" not in source


def test_suite_config_honors_an_absolute_config_value(monkeypatch, tmp_path):
    """A relative value resolves against TRAIN_ROOT; an absolute one is kept."""
    absolute = tmp_path / "suite.yaml"
    spec = common.training_cfg().bundle.model_copy(
        update={"suite_config": str(absolute)}
    )
    cfg = common.training_cfg().model_copy(update={"bundle": spec})
    monkeypatch.setattr(colab, "training_cfg", lambda: cfg)
    assert colab._suite_config_path() == absolute
