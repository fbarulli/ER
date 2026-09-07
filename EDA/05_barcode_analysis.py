"""01e: Barcode distribution and its relationship to other features.

Barcode (GTIN) is the ground-truth hard link, but it only covers ~42% of SKUs.
This step asks WHERE that coverage lives and WHETHER it is biased: barcode
length/anomalies, coverage by retailer/country/category, and barcode presence
vs price/volume. If coverage is systematically non-random, the barcode-based
evaluation is itself biased.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.common import RESULTS, canonical_volume, has_barcode, load_dataset
from lib.text import MACRO_MAP

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    bc = df["barcode"].fillna("").astype(str)
    df["has_bc"] = has_barcode(df)
    df["bc_len"] = bc.str.len()
    df["macro"] = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
    df["vol"] = canonical_volume(df["title"])["canonical_volume_ml"]
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")

    print(f"barcode coverage: {df['has_bc'].mean():.1%}")
    print(f"unique barcodes: {bc[df['has_bc']].nunique():,}")
    print("barcode length distribution:")
    print(bc[df["has_bc"]].str.len().value_counts().sort_index().to_string())

    # ---- coverage by segment ----
    for col in ["retailer", "country", "macro"]:
        cov = df.groupby(col)["has_bc"].mean().sort_values(ascending=False)
        print(f"\ncoverage by {col} (top/bottom):")
        print(pd.concat([cov.head(5), cov.tail(5)]).round(3).to_string())

    # ---- barcode vs price / volume ----
    print("\nmedian price: has_bc vs no_bc:")
    print(df.groupby("has_bc")["price_num"].median().round(2).to_string())
    print("\nvolume coverage: has_bc vs no_bc:")
    print(
        df.groupby("has_bc")["vol"].agg(lambda s: s.notna().mean()).round(3).to_string()
    )

    # ---- cardinality: SKUs per barcode ----
    multi = bc[df["has_bc"]].value_counts()
    print(f"\nbarcodes with >1 SKU: {(multi > 1).sum():,} / {len(multi):,}")
    print(f"max SKUs per barcode: {multi.max():,}")
    n_retailers = df[df["has_bc"]].groupby("barcode")["retailer"].nunique()
    print(f"multi-retailer barcodes (>=2 retailers): {(n_retailers > 1).sum():,}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    # length distribution
    df[df["has_bc"]]["bc_len"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0, 0], color="#4C72B0"
    )
    axes[0, 0].set_title("Barcode length distribution (digits)")
    axes[0, 0].set_xlabel("length")
    axes[0, 0].set_ylabel("SKUs")

    # coverage by country
    cov = df.groupby("country")["has_bc"].mean().sort_values()
    cov.plot(kind="barh", ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Barcode coverage by country")
    axes[0, 1].set_xlabel("coverage")

    # coverage by macro category
    cov2 = df.groupby("macro")["has_bc"].mean().sort_values()
    cov2.plot(kind="barh", ax=axes[1, 0], color="#C44E52")
    axes[1, 0].set_title("Barcode coverage by macro category")
    axes[1, 0].set_xlabel("coverage")

    # price: has_bc vs not (box/log)
    for key, sub in [
        ("has barcode", df[df["has_bc"]]),
        ("no barcode", df[~df["has_bc"]]),
    ]:
        axes[1, 1].hist(
            np.log1p(sub["price_num"].dropna()), bins=40, alpha=0.6, label=key
        )
    axes[1, 1].set_title("Price distribution (log1p): barcoded vs not")
    axes[1, 1].set_xlabel("log1p(price)")
    axes[1, 1].legend()

    fig.savefig(EDA_OUT / "01e_barcode_analysis.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {EDA_OUT / '01e_barcode_analysis.png'}")


if __name__ == "__main__":
    main()
