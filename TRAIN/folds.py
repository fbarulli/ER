"""folds.py — connected-component folds over the positive-pair graph
The problem with random splitting
If you just randomly assign each pair to training or testing, the same product (same barcode) might end up in both sets (data leakage).

The problem with splitting by barcode
If you split by barcode, those two barcodes could land in different folds, and the pair gets broken — you lose training/testing examples.

The connected‑component solution
Imagine drawing a line (edge) between two barcodes every time they appear together in a positive pair. Some barcodes link directly, and through a chain of links they
form a cluster (called a connected component). For example:

A is paired with B
B is paired with C
So A, B, C are all in one connected component.

Now, instead of splitting individual pairs or barcodes, we split these clusters. All barcodes in one cluster go into the same fold. That means:
Every positive pair stays entirely inside one fold → no pairs are lost.
No barcode appears in more than one fold → no leakage.
Any barcode that never appears in a positive pair becomes its own little cluster of one.

Why this is good
Fair testing: the model cannot cheat by seeing the same product in training and testing.
No wasted data: every positive pair remains available for training or evaluation.
Leakage prevention: any information that could leak travels along those edges, and we split exactly along those edges, so nothing crosses.
"""

from __future__ import annotations

import numpy as np

from lib.schemas import FoldSets


def component_folds(
    pos: np.ndarray, row_bc: np.ndarray, k: int, seed: int
) -> list[set[str]]:
    """K folds over CONNECTED COMPONENTS of the positive-pair graph.

    Pipeline positives link two DIFFERENT barcodes, so barcode-level folds
    straddle pairs (one endpoint per side) and pairs_in_set silently drops
    them — measured 7,489 positives → ~1,500 straddling per fold boundary,
    test sets shrinking to ~130 pairs. Union-find over the pair edges groups
    transitively-linked barcodes into components; components are shuffled
    (seeded) and dealt to k folds. Every positive pair sits inside ONE
    component → inside ONE fold: no straddle, no silent loss, no leak
    (leakage travels exactly along the edges we split on).

    Unlinked barcodes become singleton components — still fold members so
    their mined negatives split group-aware.

    BOUNDARY CONTRACT (lib.schemas.FoldSets): the returned folds are
    pairwise DISJOINT — a barcode in two folds would put one product in
    train and test at once. Validated on return.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # every barcode in the dataset is a node (singletons included)
    for bc in row_bc:
        if bc and bc not in parent:
            parent[bc] = bc
    # union along positive pairs
    for a, b in pos:
        bca, bcb = str(row_bc[a]), str(row_bc[b])
        if bca and bcb:
            union(bca, bcb)

    # group by root; sort members for deterministic component identity
    comps: dict[str, set[str]] = {}
    for bc in parent:
        comps.setdefault(find(bc), set()).add(bc)
    comp_list = sorted(comps.values(), key=lambda s: sorted(s))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comp_list))
    folds: list[set[str]] = [set() for _ in range(k)]
    for i, comp_idx in enumerate(order):
        folds[i % k] |= comp_list[comp_idx]
    return FoldSets(folds=folds).folds
# (trailing HARDNEG_SIM_THRESHOLD removed — dead constant, no readers; the
# threshold lives in TRAIN/training.yaml pairs.hardneg_sim_threshold)