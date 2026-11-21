"""The stratum fixture's declared source-GTIN override must reach the scorer.

``training.generate_rand_stratum_sweep`` writes ``source_gtin`` per row as an
"evaluation override: callers must apply it to the source SKU gtin before
scoring". The loaders merged the labels but never applied it, so the GTIN
ablation exam was scored on the untouched dataset GTIN and every row reported
``both_equal`` - the ``different`` / ``one_missing`` strata silently vanished.
"""

from __future__ import annotations

import pandas as pd

from training.rand_matching import _apply_source_gtin


def test_declared_source_gtin_overrides_the_dataset_gtin():
    declared = pd.DataFrame(
        {
            "SKU_ID": ["sku-a", "sku-b", "sku-c"],
            "gtin": ["4006381333931", "4006381333931", "4006381333931"],
            "source_gtin": ["9999999999999", "", None],
        }
    )
    untouched = pd.DataFrame({"SKU_ID": ["sku-a"], "gtin": ["4006381333931"]})

    applied = _apply_source_gtin(declared.copy())

    assert applied["gtin"].tolist() == ["9999999999999", "", ""]
    assert _apply_source_gtin(untouched.copy()).equals(untouched)
