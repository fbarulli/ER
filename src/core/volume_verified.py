"""volume_verified_cross_country — extracted from the second06 A/B
experiment into the TRAIN_GPU lib (train_one_config's hard-positive hook;
second06 itself is repo-side history, not part of the standalone lane)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.common import F, canonical_volume
from core.manifest import count_drop
from core.schemas import check_cross_country_pair_frame


def volume_verified_cross_country(df: pd.DataFrame) -> np.ndarray:
    """Cross-country GOLD+ pairs (second04 manifest) whose canonical volume agrees.

    These are the verified translation-tax pairs: same barcode, different
    country, same physical volume — hard positives by construction.
    Returns (N, 2) row-index pairs into df.
    """
    # The name reads the SSOT files map (config/paths.yaml
    # files.second04_pairs_positive). Training materializes this manifest
    # immediately before calling this consumer; a missing file is therefore
    # an incomplete data-prep stage, not an empty training population.
    pairs_csv = F["second04_pairs_positive"]
    if not pairs_csv.exists():
        raise FileNotFoundError(
            "volume-verified manifest is not materialized: "
            f"{pairs_csv}. Run python -m training.build_second04_pairs first."
        )
    manifest = pd.read_csv(
        pairs_csv,
        dtype={
            "sku_id_a": "string",
            "sku_id_b": "string",
            "gtin": "string",
            "country_a": "string",
            "country_b": "string",
        },
    )
    check_cross_country_pair_frame(manifest)

    pid_to_idx = {str(pid): i for i, pid in enumerate(df["product_id"].astype(str))}
    a_idx = manifest["sku_id_a"].map(pid_to_idx)
    b_idx = manifest["sku_id_b"].map(pid_to_idx)
    ok = a_idx.notna() & b_idx.notna()
    resolution_drop = count_drop(
        len(manifest), int(ok.sum()), "volume_verified_unresolved_manifest_ids"
    )
    if resolution_drop["dropped"]:
        print(
            f"[volume_verified] {resolution_drop['reason']}: "
            f"{resolution_drop['dropped']:,} removed "
            f"({resolution_drop['before']:,} -> {resolution_drop['after']:,})",
            flush=True,
        )
    # AUDIT 2026-09-09: np.array(list(zip(...))) of an EMPTY selection is
    # shape (0,) not (0, 2) — pairs[:, 0] below then raised IndexError. A
    # manifest with zero resolvable ids (or zero cross-country rows) must
    # return the empty (0, 2) contract, not crash.
    pairs = np.array(list(zip(a_idx[ok], b_idx[ok], strict=True)), dtype=int).reshape(-1, 2)

    # volume agreement filter (canonical_volume: '330ml' and '0,33 l' collapse)
    vol = canonical_volume(df["title"])["canonical_volume_ml"]
    vol = vol.fillna(-1).to_numpy()
    agrees = (
        (vol[pairs[:, 0]] > 0)
        & (vol[pairs[:, 1]] > 0)
        & (vol[pairs[:, 0]] == vol[pairs[:, 1]])
    )
    agreed = pairs[agrees]
    agreement_drop = count_drop(
        len(pairs), len(agreed), "volume_verified_volume_disagreement"
    )
    print(
        f"[volume_verified] {agreement_drop['reason']}: "
        f"{agreement_drop['dropped']:,} removed "
        f"({agreement_drop['before']:,} -> {agreement_drop['after']:,})",
        flush=True,
    )
    return agreed
