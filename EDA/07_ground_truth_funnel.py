"""01g: How much ground truth is actually usable — funnel + geography.

The barcode label ("same barcode => same product") is our only supervised
signal. This step quantifies its usable size: total SKUs -> has barcode ->
multi-retailer (>=2 retailers, the only cross-retailer "same product" signal)
-> clean (exclude conflicting/mislabeled barcodes) -> usable positive pairs,
and shows where the usable labels live geographically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lib.blocking import build_pairs
from lib.common import RESULTS, SEED, has_barcode, load_dataset

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    has_bc = has_barcode(df)

    n_total = len(df)
    n_barcoded = int(has_bc.sum())

    # multi-retailer barcode groups: the ONLY cross-retailer "same product" signal
    barcoded = df[has_bc].copy()
    barcoded["n_retailers"] = barcoded.groupby("barcode")["retailer"].transform(
        "nunique"
    )
    multi = barcoded[barcoded["n_retailers"] > 1]
    n_multi_rows = len(multi)

    # conflicting-barcode groups (same title+brand, >1 unique barcode) = mislabeled -> exclude
    conf = multi.groupby(["title", "brand"])["barcode"].nunique()
    conflicting = conf[conf > 1]
    conflict_mask = multi.set_index(["title", "brand"]).index.isin(conflicting.index)
    n_conf_rows = int(conflict_mask.sum())
    n_clean_rows = n_multi_rows - n_conf_rows

    # usable positive pairs (title-deduped within each clean multi-retailer group, cap 4).
    # The SAME conflict exclusion as the "clean" stage is applied BEFORE pair
    # building so the funnel is monotonic (usable pairs <= clean rows). Seed comes
    # from _common (config SSOT), not a hardcoded 42.
    clean_multi = multi.loc[~conflict_mask].reset_index(drop=True)
    pos, _ = build_pairs(clean_multi, SEED, 4, 10_000)
    n_pos = len(pos)

    print(f"total SKUs:              {n_total:,}")
    print(f"has barcode:             {n_barcoded:,}  ({n_barcoded / n_total:.1%})")
    print(f"multi-retailer rows:     {n_multi_rows:,}  ({n_multi_rows / n_total:.1%})")
    print(f"clean (excl conflicting):{n_clean_rows:,}  ({n_clean_rows / n_total:.1%})")
    print(f"usable positive pairs:   {n_pos:,}")
    print(f"conflicting-barcode groups (excluded as mislabeled): {len(conflicting):,}")

    # ---- plot 1: the funnel ----
    stages = [
        "total SKUs",
        "has barcode",
        "multi-retailer",
        "clean",
        "usable pos pairs",
    ]
    values = [n_total, n_barcoded, n_multi_rows, n_clean_rows, n_pos]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.barh(
        stages[::-1],
        values[::-1],
        color=["#4C72B0", "#4C72B0", "#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_xscale("log")
    for i, v in enumerate(values[::-1]):
        ax.text(v, i, f"  {v:,}", va="center")
    ax.set_xlabel("count (log scale)")
    ax.set_title("Usable ground-truth funnel (barcode → cross-retailer → clean)")
    fig.savefig(EDA_OUT / "01g_ground_truth_funnel.png", dpi=150)
    plt.close(fig)

    # ---- plot 2: usable labels by country ----
    # assign each multi-retailer barcode group to the countries its rows span
    grp_countries = multi.groupby("barcode")["country"].nunique().rename("n_countries")
    country_rows = (
        multi.groupby("country")["barcode"].nunique().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    country_rows.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("multi-retailer barcodes (usable-label groups)")
    ax.set_title("Where the usable ground truth lives, by country")
    fig.savefig(EDA_OUT / "01g_usable_by_country.png", dpi=150)
    plt.close(fig)

    # cross-country rate: how much of the label is single-country vs cross-country
    single = int((grp_countries == 1).sum())
    print(f"\nmulti-retailer barcode groups: {len(grp_countries):,}")
    print(f"  single-country groups: {single:,} ({single / len(grp_countries):.1%})")
    print(
        f"  cross-country groups:  {len(grp_countries) - single:,} ({(len(grp_countries) - single) / len(grp_countries):.1%})"
    )
    print("\nwrote 01g_ground_truth_funnel.png + 01g_usable_by_country.png")


if __name__ == "__main__":
    main()
