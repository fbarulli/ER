"""Pin: the tracks hair-split fold deal and the shared split declaration.

The smart split is owned ONCE by ``core.smart_split.SmartSplit``. The tracks
holdout deals its positive-graph components through ``SmartSplit.deal_uniform``
(``training.folds.ComponentIndex.folds``), and the shared laya corpus is carved
at the ratios declared in ``config/smart_split.yaml``. These two public
behaviours pin the tracks side of the one owner: the fold deal is byte-stable,
and the declared role map is the contract.
"""
from __future__ import annotations

import numpy as np

from core.smart_split import SmartSplit
from training.folds import component_folds


def test_component_folds_deal_is_the_smart_split_deal() -> None:
    """The deal output is byte-identical to the pre-refactor tracks fold deal."""
    row_bc = np.array([f"bc{i:02d}" for i in range(13)], dtype=object)
    positives = np.array(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype=np.int64
    )
    folds = component_folds(positives, row_bc, 3, 42)
    assert [sorted(fold) for fold in folds] == [
        ["bc00", "bc01", "bc06", "bc07", "bc08", "bc09"],
        ["bc02", "bc03", "bc04", "bc05"],
        ["bc10", "bc11", "bc12"],
    ]


def test_smart_split_roles_are_the_declared_contract() -> None:
    """The owner's role map is the declared train / select / validate contract."""
    owner = SmartSplit.from_config()
    assert owner.roles == {"train": "train", "select": "dev", "validate": "test"}
    assert owner.splits == ("train", "dev", "test")
    assert owner.strata
