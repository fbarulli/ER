"""Pin: the mounted export's cohort identity drives augmentation counts.

History: a local 10k-cohort prep staged dataset_10k.csv as dataset.csv but no
lane set ER_COHORT_TAG, so training.train mined with the full-dataset
vendor quota (300) and the cohort's exhausted cross-vendor pool (287)
killed the run inside AugmentationCoverage. The cohort tag is now derived
from the mounted export bytes; a lane's explicit ER_COHORT_TAG still wins.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from core import common


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'DATA_PATH', tmp_path / 'dataset.csv')
    for name, text in (('dataset_10k.csv', 'ten-k'),
                       ('dataset_50pct.csv', 'fifty-pct')):
        (tmp_path / name).write_text(text)
    return tmp_path


def test_missing_export_is_full(tree):
    assert common.mounted_cohort() == 'full'


def test_byte_identical_cohort_copies_are_those_cohorts(tree):
    tree.joinpath('dataset.csv').write_text('ten-k')
    assert common.mounted_cohort() == '10k'
    tree.joinpath('dataset.csv').write_text('fifty-pct')
    assert common.mounted_cohort() == '50pct'


def test_distinct_export_and_same_size_collisions_stay_full(tree):
    tree.joinpath('dataset.csv').write_text('tenant')
    assert common.mounted_cohort() == 'full'
    (tree / 'dataset_10k.csv').write_text('ten-k-x')
    tree.joinpath('dataset.csv').write_text('ten-k-y')
    assert common.mounted_cohort() == 'full'
