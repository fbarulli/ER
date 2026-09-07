"""01d: Description missingness — how much, where, and is it systematic?

description is the field with the biggest gap after barcode (12,063 rows,
~17%). This step maps the missingness by retailer / country / category /
brand / barcode-coverage to answer: is it random noise or a systematic
pattern (e.g. certain retailers never ship descriptions)? It also checks
what ELSE is missing in the description-less rows (are they the poorest
rows in the catalog?).

Outputs (RESULTS = project/experiments/results/euromonitor/):
  01d_description_missing.csv   one display table: dimension, key, rows,
                                missing, missing_rate
  01d_missing_by_category.png   missing rate by top categories
  01d_missing_by_retailer.png   missing rate by top retailers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from lib.common import RESULTS, has_barcode, load_dataset

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)

CSV_MISSING = EDA_OUT / "01d_description_missing.csv"
PNG_CATEGORY = EDA_OUT / "01d_missing_by_category.png"
PNG_RETAILER = EDA_OUT / "01d_missing_by_retailer.png"

TOP_PER_DIM = 15


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    missing = df["description"].isna()
    n_missing = int(missing.sum())
    n_total = len(df)
    print(
        f"description: {n_missing:,} missing / {n_total:,} rows "
        f"({n_missing / n_total:.1%})"
    )

    rows: list[dict] = []

    def add(dimension: str, key: str, n: int, miss: int) -> None:
        rows.append(
            {
                "dimension": dimension,
                "key": key,
                "rows": n,
                "missing": miss,
                "missing_rate": round(miss / n, 4) if n else None,
            }
        )

    # ---- per-dimension breakdown ---------------------------------------------
    for dim in ["retailer", "country", "category", "brand"]:
        agg = df.groupby(dim, dropna=False)["description"].agg(
            rows="size", miss=lambda s: int(s.isna().sum())
        )
        agg = agg.sort_values("miss", ascending=False).head(TOP_PER_DIM)
        for key, r in agg.iterrows():
            add(dim, str(key), int(r["rows"]), int(r["miss"]))

    # ---- barcode correlation: is the gap shared with the other hole? ---------
    known_bc = has_barcode(df)
    add("barcode", "has_barcode", int(known_bc.sum()), int(missing[known_bc].sum()))
    add("barcode", "no_barcode", int((~known_bc).sum()), int(missing[~known_bc].sum()))
    double_missing = int((missing & ~known_bc).sum())
    print(
        f"  rows missing BOTH description and barcode: {double_missing:,} "
        f"({double_missing / n_total:.1%})"
    )

    # ---- what else is complete in the description-less rows? ------------------
    poor = df[missing]
    for col in ["title", "brand", "category", "price", "barcode"]:
        present = int(poor[col].fillna("").astype(str).str.len().gt(0).sum())
        add("missing_rows_profile", col, len(poor), len(poor) - present)

    frame = pd.DataFrame(rows)
    frame.to_csv(CSV_MISSING, index=False)
    print(f"wrote {CSV_MISSING} (display table, {len(frame)} rows)")

    # ---- figures ---------------------------------------------------------------
    def _rate_plot(dim: str, out_path: Path, title: str) -> None:
        sub = frame[frame["dimension"] == dim].sort_values("missing_rate")
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        bars = ax.barh(
            sub["key"].astype(str), sub["missing_rate"] * 100, color="#C44E52"
        )
        ax.bar_label(
            bars, labels=[f"{m:,}" for m in sub["missing"]], fontsize=7, padding=2
        )
        ax.set_xlim(0, max(sub["missing_rate"].max() * 100 * 1.15, 10))  # data-derived
        ax.set_xlabel("rows missing description (%)  [bar label = n missing]")
        ax.set_title(title)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    _rate_plot(
        "category", PNG_CATEGORY, "Description missing rate by category (top 15)"
    )
    print(f"wrote {PNG_CATEGORY}")
    _rate_plot(
        "retailer", PNG_RETAILER, "Description missing rate by retailer (top 15)"
    )
    print(f"wrote {PNG_RETAILER}")

    # ---- printed report -----------------------------------------------------------
    print("\nby barcode:")
    for _, r in frame[frame["dimension"] == "barcode"].iterrows():
        print(
            f"  {r['key']:<12} {r['rows']:>7,} rows  {r['missing']:>6,} missing "
            f"({r['missing_rate']:.1%})"
        )
    print("\nmissing-row profile (what IS present in description-less rows):")
    for _, r in frame[frame["dimension"] == "missing_rows_profile"].iterrows():
        print(
            f"  {r['key']:<10} present in {r['rows'] - r['missing']:>6,} / "
            f"{r['rows']:,} ({1 - r['missing'] / r['rows']:.1%})"
        )
    cat = frame[frame["dimension"] == "category"].sort_values(
        "missing_rate", ascending=False
    )
    print("\ntop categories by missing rate:")
    for _, r in cat.head(5).iterrows():
        print(
            f"  {r['key'][:42]:<42} {r['missing']:>5,} / {r['rows']:>6,} "
            f"({r['missing_rate']:.0%})"
        )
    ret = frame[frame["dimension"] == "retailer"]
    all_missing = ret[ret["missing"] == ret["rows"]]
    print(
        f"\nretailers with 100% missing description: {len(all_missing)} "
        f"({all_missing['missing'].sum():,} rows affected)"
    )


if __name__ == "__main__":
    main()
