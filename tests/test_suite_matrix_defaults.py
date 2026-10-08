"""Baked S/M/L suite-matrix defaults: canonical names and the automatic
device flip on the tracked-suite path.

Pinned behaviors:

* the matrix is a SPEC, not a restriction — unknown suite configs keep
  working and the matrix only supplies labels and the device-flip pattern;
* a device-cpu suite under a non-CPU `--gpu` request generates the scratch
  cuda clone under `results/model_tracks/<suite>__gpu/` (only the yamls;
  every data binding stays under `data/`);
* the opt-out env keeps the raw must-agree error for operators.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli import colab
from core.schemas import canonical_suite_matrix
from model_tracks.config import load_config as load_suite


def _setup_dir(tmp_path: Path, *, device: str = "cpu") -> Path:
    setup = tmp_path / "data/prepared/smoke_200"
    (setup / "prepared").mkdir(parents=True)
    texts = {
        "suite.yaml": (
            "setup_dir: data/prepared/smoke_200\n"
            "text_bundle: data/prepared/smoke_200/text_prepared.pkl.gz\n"
            f"device: {device}\n"
        ),
        "gnn_only.yaml": f"track: gnn_only\ndevice: {device}\n",
        "cascade.yaml": f"track: cascade\ndevice: {device}\n",
        "text.yaml": "track: text\n",
        "prepared/input_manifest.json": "{}",
    }
    for name, content in texts.items():
        path = setup / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (setup / "eligible_catalog.csv").write_text("sku_id\na\n", encoding="utf-8")
    (setup / "text_prepared.pkl.gz").write_bytes(b"bundle")
    return setup


def _settings(source: Path):
    return load_suite(source)


def test_the_baked_sml_matrix_is_canonical():
    matrix = canonical_suite_matrix()
    assert [(row.size, row.name, row.device) for row in matrix.suites] == [
        ("S", "smoke_200", "cpu"), ("M", "3k", "cpu"), ("L", "full", "cuda"),
    ]


def test_unknown_suite_configs_still_work():
    assert canonical_suite_matrix().entry("not_a_suite") is None


def test_the_device_flip_pattern_owns_the_defaults():
    flip = canonical_suite_matrix().device_flip
    assert flip.tracks_dir == "model_tracks"
    assert flip.suffix == "__gpu"
    assert flip.opt_out_env == "ER_SUITES_KEEP_DEVICE"


@pytest.fixture()
def suite_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(colab, "RESULTS", tmp_path / "results")
    return tmp_path


def test_a_cpu_suite_flips_automatically(suite_paths, tmp_path):
    _setup_dir(tmp_path)
    source = tmp_path / "data/prepared/smoke_200/suite.yaml"
    flip_path = colab._suite_device_flip(source, _settings(source))
    assert flip_path == tmp_path / "results/model_tracks/smoke_200__gpu/suite.yaml"
    flipped = flip_path.read_text(encoding="utf-8")
    assert "device: cuda" in flipped
    assert "setup_dir: data/prepared/smoke_200" in flipped
    for name in ("gnn_only.yaml", "cascade.yaml", "text.yaml"):
        assert (flip_path.parent / name).is_file()
    assert all(path.suffix == ".yaml" for path in flip_path.parent.iterdir()), \
        "copying ONLY the yamls invites device flips"


def test_default_bundle_is_the_canonical_full_cohort(suite_paths, tmp_path):
    assert colab.default_prepared_input_package() is None
    bundle = tmp_path / "results/kaggle_lane/full/bundle"
    bundle.mkdir(parents=True)
    (bundle / "all_tracks_inputs.tar.zst").write_bytes(b"x")
    assert colab.default_prepared_input_package() == bundle
