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

from core.schemas import CalibrationPartition, FoldSets


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


def holdout_split(
    pos: np.ndarray,
    row_bc: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    """The SINGLE derivation of the holdout split: (train, dev, test) barcodes.

    Roles come from ``n_folds``, not from hardcoded indices: ``test =
    quarters[-1]``, ``dev = quarters[-2]``, ``train = the remaining
    quarters``.  ``component_folds`` deals the components round-robin over
    ``n_folds`` groups, so each quarter is ``1.0 / n_folds`` of the graph and
    the realized shares are fixed by the arity.  A configured
    ``dev_fraction``/``test_fraction`` that does not match that share is a
    mis-configured 50/25/25 contract (the schema pins
    ``holdout_component_folds`` at 4 = 0.25/0.25); it raises here instead of
    silently producing 60/20/20 under a "50/25/25" label.
    """
    if n_folds < 2:
        raise ValueError(f"holdout split needs at least 2 component folds, got {n_folds}")
    quarter = 1.0 / n_folds
    if abs(dev_fraction - quarter) > 1e-9 or abs(test_fraction - quarter) > 1e-9:
        raise ValueError(
            f"holdout split contract violated: n_folds={n_folds} deals quarters "
            f"of {quarter:.4f} each, but the split declares dev_fraction="
            f"{dev_fraction} and test_fraction={test_fraction} "
            f"(the 50/25/25 contract requires n_folds=4)"
        )
    quarters = component_folds(pos, row_bc, n_folds, seed)
    return set().union(*quarters[:-2]), quarters[-2], quarters[-1]


def partition_component_pairs(
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
    row_bc: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition pair pools by positive-pair components without leakage.

    The returned tuple is ``(positive_fit, positive_reserved,
    negative_fit, negative_reserved)`` — see ``CalibrationPartition``, which
    validates the populations and the boundary contract on construction.  The
    reservation is made by whole components AND a pair is reserved only when
    BOTH of its endpoints are reserved: a positive sits inside one component
    (so it is unaffected), but a negative links two DIFFERENT components, and
    a left-endpoint-only rule deals the two mirrored orientations of the same
    product pair to opposite sides of the calibration boundary — the fit half
    keeps ``(rep(g1), canon(g2))`` for early stopping while the calibration
    half keeps ``(rep(g2), canon(g1))``.
    """
    if len(positive_pairs) == 0:
        raise ValueError("cannot reserve calibration data without DEV positives")
    if not 0.0 < fraction <= 0.5:
        raise ValueError(
            "calibration fraction must be in (0, 0.5]: the component grid "
            "reserves a whole number of folds and can never reserve a "
            f"majority of DEV, got {fraction}"
        )
    pair_rows = np.unique(positive_pairs.ravel())
    local_index = {int(row): position for position, row in enumerate(pair_rows)}
    local_pairs = np.asarray(
        [
            [local_index[int(left)], local_index[int(right)]]
            for left, right in positive_pairs
        ],
        dtype=int,
    )
    # the identity a barcode is matched by is the STRIPPED one on both the
    # component side and the mask side (a padded barcode used to build a
    # component it could never be reserved by)
    local_barcodes = np.asarray(
        [str(row_bc[int(row)]).strip() for row in pair_rows], dtype=str
    )
    n_folds = max(2, int(np.ceil(1.0 / fraction)))
    component_groups = component_folds(local_pairs, local_barcodes, n_folds, seed)
    n_reserved = min(max(1, int(round(n_folds * fraction))), n_folds - 1)
    selected_barcodes = set().union(*component_groups[:n_reserved])

    def reserved_mask(pairs: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                str(row_bc[int(pair[0])]).strip() in selected_barcodes
                and str(row_bc[int(pair[1])]).strip() in selected_barcodes
                for pair in pairs
            ],
            dtype=bool,
        )

    positive_mask = reserved_mask(positive_pairs)
    negative_mask = reserved_mask(negative_pairs)
    partition = CalibrationPartition(
        positive_fit=positive_pairs[~positive_mask],
        positive_reserved=positive_pairs[positive_mask],
        negative_fit=negative_pairs[~negative_mask],
        negative_reserved=negative_pairs[negative_mask],
        row_bc=row_bc,
        n_positive_pairs=len(positive_pairs),
        n_negative_pairs=len(negative_pairs),
    )
    # checks that can actually fail: the two "population" guards that used to
    # sit here compared pairs[~m] + pairs[m] against len(pairs) for one and the
    # same boolean mask, i.e. they held identically for ANY mask.
    if len(partition.positive_reserved) == 0:
        raise RuntimeError(
            "calibration reservation is empty: no positive-pair component was "
            "reserved — the component identity is broken (check row_bc)"
        )
    if len(negative_pairs) and len(partition.negative_reserved) == 0:
        raise RuntimeError(
            "calibration reservation kept no negative pair: no negative pair "
            "has both endpoints inside a reserved component"
        )
    return partition.pools()
# (trailing HARDNEG_SIM_THRESHOLD removed — dead constant, no readers; the
# threshold lives in config/training.yaml pairs.hardneg_sim_threshold)
