"""The training-bundle stage: one declaration, CPU-lane producer, GPU-lane consumer.

``config colab.prepared_bundles`` is the ONE declaration of the prepared
training bundles. The CPU bundle lane builds every declared entry on its VM as
a stage of its existing prepare process (cli.colab_lane_cpu_bundles) and its
delivery ships them; the GPU lane resolves the launch's own request against the
same declaration. These tests pin the three public behaviours that make the
pair agree — a launch consumes what the CPU lane builds, an undeclared request
fails loud instead of silently preparing locally, and the stage really is part
of the one prepare process (so it inherits the status file, the budget and the
poll).

The regression they pin: before the stage existed, ``--what dual-train`` and
every non-default ``--what train`` (workers, model, masking profile) refused,
because the GPU lane had no CPU-lane bundle for them and could not build one
locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cli.colab_lane import ColabCPULane, ColabGPULane
from cli.colab_lane_cpu_bundles import ColabCPULaneTrainingBundles
from core.common import TRAIN_ROOT, training_cfg


class _Surface:
    """The transport attributes the prepare-launch script reads."""

    SESSION = "lane-test"
    REMOTE_ROOT = "/content/EuromonitoR"
    _BOOTSTRAP = ""


def _request(**overrides) -> argparse.Namespace:
    """One GPU launch's parsed request, defaulting to the standard single-worker train."""
    defaults = {
        "what": "train", "workers": 1, "model": None, "sample": None,
        "masking_profile": None, "dataset_csv": None, "resume_run": None,
        "run_label": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def _declared():
    return training_cfg().colab.prepared_bundles


def test_a_launch_consumes_the_bundles_the_cpu_lane_declares():
    """``--what dual-train`` (and ``--workers 2``) get BOTH declared bundles."""
    lane = ColabGPULane()
    paths = _declared().paths()
    assert len(paths) >= 2, "the declared set must serve a two-worker launch"

    assert lane.prepared_bundles(_request(what="dual-train")) == [
        TRAIN_ROOT / path for path in paths[:2]]
    assert lane.prepared_bundles(_request(workers=2)) == [
        TRAIN_ROOT / path for path in paths[:2]]
    assert lane.prepared_bundles(_request()) == [TRAIN_ROOT / _declared().baseline()]


def test_an_undeclared_request_fails_loud_naming_the_missing_entry():
    """A request the declaration cannot serve refuses; it never prepares one here."""
    lane = ColabGPULane()
    with pytest.raises(ValueError, match="worker=99"):
        lane.prepared_bundles(_request(workers=99))
    with pytest.raises(ValueError, match="colab.prepared_bundles"):
        lane.prepared_bundles(_request(masking_profile="no-such-profile"))
    with pytest.raises(ValueError, match="colab.prepared_bundles"):
        lane.prepared_bundles(_request(model="some_other_model"))


def test_a_foreign_request_still_refuses_before_the_declaration_is_consulted():
    """A resume id, a run label, another CSV: inputs this lane does not own."""
    lane = ColabGPULane()
    for overrides in ({"resume_run": "abc"}, {"run_label": "a,b"},
                      {"dataset_csv": "data/other.csv"}):
        assert lane.prepared_bundles(_request(**overrides)) is None
    assert lane.prepared_bundles(_request(what="smoke")) is None


def test_the_stage_runs_inside_the_one_prepare_process_after_prepare_all():
    """One detached process, one budget, one status file — no second spawn path."""
    lane = ColabCPULane(surface=_Surface())
    script = lane.launch_prepare_script(
        launch_args=', "--run-dir", root + "/results/training_prep/RUN1"',
        cohort_export="dataset.csv")
    compile(script, "<prepare-launch>", "exec")

    stage = " ".join(lane.bundle_stage_command())
    assert lane.bundle_stage_command()[-1] == "cli.colab_lane_cpu_bundles"
    assert stage in script
    # prepare_all runs first: the bundles are built from what it sealed, and the
    # chain (not either stage alone) is what the status file reports on.
    assert script.index('"training.prepare_all"') < script.index(stage)
    assert "timeout --signal=TERM" in script


def test_the_stage_builds_every_declared_bundle_prepare_all_did_not_write():
    """Worker 1 is prepare_all's sealed ``full_bundle``; every other worker is the stage's."""
    declared = _declared()
    entries = declared.stage_entries()
    assert [entry.worker for entry in entries] == [
        entry.worker for entry in declared.entries if entry.worker != 1]

    stage = ColabCPULaneTrainingBundles()
    checkout = Path("/checkout")
    for entry in entries:
        command = stage.bundle_command(checkout, entry)
        assert command[command.index("--model") + 1] == entry.model
        assert command[command.index("--masking-profile") + 1] == entry.masking_profile
        assert command[command.index("--prepare-bundle") + 1] == str(
            checkout / declared.path_of(entry))
        assert command[-2:] == ["--no-mask-effect", "--no-plot"]
