"""_blocking.py: shared blocking evaluation machinery for the euromonitor series.

Single source for the blocking A/B measurement (per-feature utility + greedy
key selection): true-pair construction and the recall/candidates evaluator.
No regexes here — those stay in lib/text.py + data_pipe.py.

Barcodes assert identity only when GS1-checksum VALID (owner ruling) — both
populations (positives here, negatives in build_pairs) exclude checksum-fail
barcodes exactly like missing ones.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from lib.gtin import barcode_validity


def build_pairs(
    df: pd.DataFrame,
    seed: int,
    max_pos_per_group: int,
    n_neg: int,
    barcode_col: str = "barcode",
    retailer_col: str = "retailer",
    title_col: str = "title",
    neg_oversample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth evaluation pairs, 0..n-1 row-indexed.

    positives: title pairs inside the same multi-retailer barcode group, built
    from one row per unique stripped title (capped at max_pos_per_group per
    group) so trivially-identical listings don't inflate agreement. negatives:
    cross-barcode pairs with different title text, sampled in ONE vectorized
    bulk pass (no per-attempt Python loop) — the bulk draw is n_neg *
    neg_oversample candidates, first n_neg valid kept. Returns (pos_pairs,
    neg_pairs) as (N, 2) int arrays.

    Barcode trust (owner ruling): only GS1-checksum-VALID barcodes assert
    identity, on both populations — invalid barcodes are excluded exactly
    like missing ones.
    """
    barcodes = df[barcode_col].fillna("").astype(str)
    titles = df[title_col].fillna("").str.strip()
    rng = np.random.default_rng(seed)

    # Barcode trust (owner ruling): only GS1-checksum-VALID barcodes assert
    # identity. A positive needs the group barcode valid (774 multi-retailer
    # groups were checksum-noise); a negative needs BOTH barcodes valid — a
    # checksum-fail barcode cannot certify "known different" any more than a
    # missing one can. Same population rule as _hard_negatives.py.
    bc_valid = barcode_validity(barcodes).to_numpy()
    known = df[(barcodes.str.len() > 0).to_numpy() & bc_valid]
    multi = known[known.groupby(barcode_col)[retailer_col].transform("nunique") > 1]
    pos_i: list[int] = []
    pos_j: list[int] = []
    for _, g in multi.groupby(barcode_col):
        sub = g.assign(_t=titles.loc[g.index])
        rows = sub[sub["_t"] != ""].drop_duplicates("_t").index.tolist()
        combos = list(combinations(rows, 2))
        if len(combos) > max_pos_per_group:
            chosen = rng.choice(len(combos), max_pos_per_group, replace=False)
            combos = [combos[k] for k in chosen]
        for a, b in combos:
            pos_i.append(a)
            pos_j.append(b)

    # neg_oversample SSOT: TRAIN/training.yaml pairs.neg_oversample when
    # None — the bulk-draw multiplier (was inline 60). Validated by
    # PairsSpec (migrated from EDA/eda.yaml, 2026-09-10).
    if neg_oversample is None:
        from lib.common import training_cfg

        neg_oversample = int(training_cfg().pairs.neg_oversample)
    n = len(df)
    bc = barcodes.to_numpy()
    tt = titles.to_numpy()
    a = rng.integers(0, n, size=n_neg * neg_oversample)
    b = rng.integers(0, n, size=n_neg * neg_oversample)
    # A negative is only a KNOWN non-match when both rows carry a non-empty
    # (and different) barcode AND both pass the GS1 checksum; a GTIN-missing
    # or checksum-fail row has unknown ground truth and must not enter the
    # negative population (matches _hard_negatives.py).
    mask = (
        (a != b)
        & (bc[a] != "")
        & (bc[b] != "")
        & (bc[a] != bc[b])
        & bc_valid[a]
        & bc_valid[b]
        & (tt[a] != tt[b])
    )
    if int(mask.sum()) < n_neg:
        raise RuntimeError(
            f"only {int(mask.sum())} valid negative pairs sampled (need {n_neg})"
        )

    pos_pairs = (
        np.column_stack([pos_i, pos_j]).astype(int)
        if pos_i
        else np.empty((0, 2), dtype=int)
    )
    return pos_pairs, np.column_stack([a[mask][:n_neg], b[mask][:n_neg]]).astype(int)


def build_true_pairs(
    df: pd.DataFrame,
    barcode_col: str = "barcode",
    retailer_col: str = "retailer",
) -> list[tuple[int, int]]:
    """ALL same-barcode pairs inside multi-retailer groups (the ground truth).

    Keeps the ORIGINAL df index (never reset) so pair indices map straight
    into df.loc / feature Series indexed by df.index. Checksum-INVALID
    barcodes assert no identity (owner ruling): their groups are excluded —
    the old monorepo version treated them as ground truth, poisoning the
    recall measurement with export noise.
    """
    barcodes = df[barcode_col].fillna("").astype(str).str.strip()
    bc_valid = barcode_validity(barcodes).to_numpy()
    known = df[(barcodes.str.len() > 0).to_numpy() & bc_valid]
    multi = known[known.groupby(barcode_col)[retailer_col].transform("nunique") > 1]
    pairs: list[tuple[int, int]] = []
    for _, g in multi.groupby(barcode_col):
        pairs.extend(combinations(g.index.tolist(), 2))
    return pairs


def eval_blocking(
    key_series: pd.Series, true_pairs: list[tuple[int, int]]
) -> tuple[float, int, int]:
    """Blocking recall + candidate count for one key series (indexed by df row).

    recall     = true pairs sharing a block / all true pairs.
    candidates = sum of n*(n-1)/2 over blocks (pairwise classifier cost).
    n_blocks   = number of non-empty blocks.
    """
    idx2block = dict(key_series)
    blocks = defaultdict(int)
    for k in key_series:
        blocks[k] += 1
    n_cand = sum(n * (n - 1) // 2 for n in blocks.values())
    retained = sum(
        1 for a, b in true_pairs if idx2block.get(a) == idx2block.get(b)
    )
    recall = retained / len(true_pairs) if true_pairs else float("nan")
    return recall, int(n_cand), len(blocks)
