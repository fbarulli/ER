"""Focused tests for the FLEX validation size (``split.validation_size``).

The scored validation population (dev + test) used to be pinned at the two
trailing component quarters. ``validation_size`` makes its SIZE configurable
while the null default keeps today's fold-derived cut byte-for-byte.

Two public behaviors are pinned, one test each:

  * the null default reproduces the fold-derived ``holdout_split`` EXACTLY
    (so committed configs and existing artifacts are unchanged);
  * varying ``validation_size`` moves the emitted validation population
    (both the reserved gtin count and the labeled-pair row count).

Deliberately synthetic and cheap: no real artifact is read or rebuilt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.folds import derive_holdout, holdout_split, merged_component_graph

SEED = 7
#: The committed 50/25/25 default (config/training.yaml split:), with the FLEX
#: knob left null exactly as committed.
SPLIT_DEFAULT = {
    "holdout_component_folds": 4,
    "dev_fraction": 0.25,
    "test_fraction": 0.25,
}


def _graph() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """400 gtins in 200 disjoint positive pairs -> 200 components of size 2.

    Every component is whole in one fold, so a labeled pair is entirely
    inside or entirely outside the validation region: the emitted row count
    is the number of pairs whose component landed in dev or test.
    """
    row_bc = np.array([f"bc{i:05d}" for i in range(400)], dtype=object)
    pos = np.array([[i, i + 1] for i in range(0, 400, 2)], dtype=np.int64)
    census = pd.DataFrame(
        {
            "gtin1": [f"bc{i:05d}" for i in range(0, 400, 2)],
            "gtin2": [f"bc{i + 1:05d}" for i in range(0, 400, 2)],
            "true_label": [1] * 200,
        }
    )
    return pos, row_bc, census


def _validation_rows(
    pos: np.ndarray, row_bc: np.ndarray, dev: set[str], test: set[str]
) -> int:
    """Labeled pairs whose BOTH endpoints are in the validation region."""
    valid = dev | test
    return sum(
        1
        for a, b in pos
        if str(row_bc[a]) in valid and str(row_bc[b]) in valid
    )


def test_default_validation_size_reproduces_the_fold_derived_split() -> None:
    pos, row_bc, census = _graph()
    merged, graph_bc, _ = merged_component_graph(pos, row_bc, labeled_pairs=census)
    expected = holdout_split(
        merged, graph_bc, n_folds=4, seed=SEED,
        dev_fraction=0.25, test_fraction=0.25,
    )
    got = derive_holdout(
        pos, row_bc, dict(SPLIT_DEFAULT), seed=SEED, labeled_pairs=census
    )
    assert got == expected, (
        "the null validation_size must reproduce the fold-derived cut exactly; "
        "a divergence means the committed 50/25/25 artifacts would move"
    )


def test_varying_validation_size_moves_the_validation_rows() -> None:
    pos, row_bc, census = _graph()
    outcomes = {}
    for validation_size in (0.25, 0.75, 600):
        train, dev, test = derive_holdout(
            pos, row_bc,
            {**SPLIT_DEFAULT, "validation_size": validation_size},
            seed=SEED, labeled_pairs=census,
        )
        assert train and dev and test, (
            f"validation_size={validation_size} produced an empty role"
        )
        assert not (train & (dev | test)), "train and validation must be disjoint"
        outcomes[validation_size] = _validation_rows(pos, row_bc, dev, test)

    small, large, count = (outcomes[0.25], outcomes[0.75], outcomes[600])
    assert small < large, f"0.25 -> {small} rows, 0.75 -> {large} rows"
    # an entity count target is a size knob too: 600 entities cover more of the
    # 400-gtin graph than the 0.25 fraction, hence more validation rows
    assert count > small, f"600 -> {count} rows, 0.25 -> {small} rows"
