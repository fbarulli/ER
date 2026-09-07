"""volume_verified_cross_country — extracted from the second06 A/B
experiment into the TRAIN_GPU lib (train_one_config's hard-positive hook;
second06 itself is repo-side history, not part of the standalone lane)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.common import RESULTS, canonical_volume


def volume_verified_cross_country(df: pd.DataFrame) -> np.ndarray:
    """Cross-country GOLD+ pairs (second04 manifest) whose canonical volume agrees.

    These are the verified translation-tax pairs: same barcode, different
    country, same physical volume — hard positives by construction.
    Returns (N, 2) row-index pairs into df.
    """
    # the second04 cross-country manifest is repo-side history; TRAIN_GPU
    # runs without it (the lane's hard-positive signal is the pipeline's
    # proceed-pairs). Missing manifest -> zero pairs, not a crash.
    pairs_csv = RESULTS / "second04_pairs_positive.csv"
    if not pairs_csv.exists():
        return np.empty((0, 2), dtype=int)
    manifest = pd.read_csv(pairs_csv, dtype={"sku_id_a": str, "sku_id_b": str})
    manifest = manifest[manifest["cross_country"]]

    pid_to_idx = {str(pid): i for i, pid in enumerate(df["product_id"].astype(str))}
    a_idx = manifest["sku_id_a"].map(pid_to_idx)
    b_idx = manifest["sku_id_b"].map(pid_to_idx)
    ok = a_idx.notna() & b_idx.notna()
    pairs = np.array(list(zip(a_idx[ok], b_idx[ok])), dtype=int)

    # volume agreement filter (canonical_volume: '330ml' and '0,33 l' collapse)
    vol = canonical_volume(df["title"])["canonical_volume_ml"]
    vol = vol.fillna(-1).to_numpy()
    agrees = (
        (vol[pairs[:, 0]] > 0)
        & (vol[pairs[:, 1]] > 0)
        & (vol[pairs[:, 0]] == vol[pairs[:, 1]])
    )
    return pairs[agrees]
