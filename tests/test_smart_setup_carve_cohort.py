"""G3: the carve's ablation cohort describes the carve, not the source.

The defect this locks down (``data/track_setup_5050/ablation_cohort/pairs.csv``
shared the source's inode): ``SmartSetupPartition._link_unchanged`` hardlinked
every member that was not regenerated, so the carve shipped the SOURCE setup's
ablation cohort — a cohort whose rows and difficulty votes never saw the carve.

``publish`` now skips the cohort on the link pass and regenerates it through the
canonical owner ``model_tracks.ablation_cohort.prepare_cohort`` over the carve's
own tree.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import model_tracks.smart_setup as smart_setup
from test_model_tracks_smart_setup import _build_setup


def test_publish_regenerates_the_carve_ablation_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hermetic like the sibling partition fixture: no live labeled census.
    import core.identity_policy
    import training.folds

    monkeypatch.setattr(training.folds, "load_labeled_pairs",
                        lambda _: pd.DataFrame({"gtin1": [], "gtin2": [], "true_label": []}))
    monkeypatch.setattr(core.identity_policy, "review_mask",
                        lambda series: pd.Series(False, index=pd.Series(series).index))
    setup = tmp_path / "setup"
    _build_setup(setup)
    (setup / "setup_manifest.json").write_text("{}")
    (setup / smart_setup.TEXT_BUNDLE_NAME).write_bytes(b"bundle")
    source_cohort = setup / smart_setup.ABLATION_COHORT_DIR / "pairs.csv"
    assert source_cohort.is_file()

    # Stub the heavy collaborators down to the one seam under test: the cohort.
    import graph_tracks.prepare as graph_prepare
    import graph_tracks.setup as graph_setup
    import model_tracks.ablation_cohort as cohort_module
    from training import prepared_bundle as prepared_bundle_module

    monkeypatch.setattr(graph_setup, "write_setup_frames", lambda *a, **k: None)
    monkeypatch.setattr(graph_prepare, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(
        smart_setup.SmartSetupPartition, "_write_lane_configs",
        lambda self, output: None)
    monkeypatch.setattr(
        smart_setup.SmartSetupPartition, "_write_documents",
        lambda self, output, partition: None)
    monkeypatch.setattr(
        prepared_bundle_module.BundleCodec, "read_payload",
        staticmethod(lambda path: {"bundle": str(path)}))
    recorded: dict = {}
    monkeypatch.setattr(
        prepared_bundle_module.FrozenHoldoutPolicy, "adopt",
        staticmethod(lambda path, populations: recorded.update(
            adopted=str(path), populations=populations)))
    monkeypatch.setattr(
        cohort_module, "prepare_cohort",
        lambda out, bundle: recorded.update(setup=str(out), bundle=bundle))

    partition = smart_setup.SmartSetupPartition.from_config(setup)
    output = tmp_path / "carve"
    partition.publish(output, training_tensors=False)

    # The linked bundle adopted the CARVE populations before the cohort read it...
    assert recorded["adopted"] == str(output / smart_setup.TEXT_BUNDLE_NAME)
    assert set(recorded["populations"]) == {"train", "dev", "test"}
    # ...and the cohort was regenerated from the CARVE tree, not the source.
    assert recorded["setup"] == str(output)
    assert recorded["bundle"] == {"bundle": str(output / smart_setup.TEXT_BUNDLE_NAME)}
    assert not (output / smart_setup.ABLATION_COHORT_DIR / "pairs.csv").exists()
