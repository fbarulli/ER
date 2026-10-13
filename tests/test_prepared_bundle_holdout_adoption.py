"""Adopting a published smart-partition carve into a frozen text bundle.

The smart 50/50 partition (``model_tracks.smart_setup``) publishes the carve's
gtin->split populations for the text lane instead of re-bundling text (owner
rule). ``FrozenHoldoutPolicy.adopt`` is the ONE owner that re-freezes a bundle's
holdout from those populations, so the text lane's frozen split becomes the same
carve the graph lane trains on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from training.prepared_bundle import BundleCodec, FrozenHoldoutPolicy


def _write_bundle(path: Path, holdout: dict[str, list[str]]) -> None:
    BundleCodec.write_pickle(path, {
        "row_bc": np.array(["1", "2", "3", "4"], dtype=object),
        "pos": np.array([[0, 1]], dtype=int),
        "holdout_populations": holdout,
    })


def test_adopt_refreezes_the_holdout_and_leaves_hardlinked_siblings(tmp_path):
    source = tmp_path / "text_prepared.pkl.gz"
    _write_bundle(source, {"train": ["1", "2"], "dev": ["3"], "test": ["4"]})
    carve = tmp_path / "carve" / "text_prepared.pkl.gz"
    carve.parent.mkdir()
    os.link(source, carve)

    FrozenHoldoutPolicy.adopt(carve, {"train": ["1", "2"], "dev": ["4"], "test": ["3"]})

    assert BundleCodec.read_payload(carve)["holdout_populations"] == {
        "train": ["1", "2"], "dev": ["4"], "test": ["3"]}
    # The atomic replace breaks the hardlink: the source keeps its own split.
    assert BundleCodec.read_payload(source)["holdout_populations"] == {
        "train": ["1", "2"], "dev": ["3"], "test": ["4"]}
