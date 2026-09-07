"""01: Euromonitor dataset EDA — coverage, completeness, noise, attributes.

Follows the repo experiment convention: writes `01_eda_describe.csv` and
`01_eda.png` under RESULTS (project/experiments/results/euromonitor/), plus
the BARCODE-coverage breakdown CSV, the attributes-key table CSV, and the
per-column dtype table CSV. Every figure is saved as PNG, headless (Agg),
dpi 150.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.common import RESULTS, column_profile, has_barcode, load_dataset
from lib.text import attributes_keys

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)

CSV_DESCRIBE = EDA_OUT / "01_eda_describe.csv"
CSV_BARCODE = EDA_OUT / "01_barcode_coverage.csv"
CSV_ATTR_KEYS = EDA_OUT / "01_attributes_keys.csv"
CSV_DTYPES = EDA_OUT / "01_dtypes.csv"
PNG_BARCODE = EDA_OUT / "01_barcode_coverage.png"
PNG_RETAILER = EDA_OUT / "01_retailer_country.png"
PNG_ATTR = EDA_OUT / "01_attributes_keys.png"
PNG_DTYPES = EDA_OUT / "01_dtypes.png"

# Columns with long free text (excluded from describe table).
TEXT_COLS = [
    "title",
    "description",
    "category_path",
    "url",
    "image_url",
    "attributes",
]


def plot_barcode_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """SKU-per-BARCODE distribution split by retailer multiplicity (log-y).

    Shows which BARCODEs give real cross-retailer matching signal (multi-
    retailer) vs single-retailer ones that only help validation.
    """
    known = df.loc[has_barcode(df)]
    per_retailer_count = known.groupby("barcode")["retailer"].nunique()
    sku_count = known.groupby("barcode").size()
    multi = per_retailer_count > 1

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(
        sku_count[~multi],
        bins=range(1, 46),
        alpha=0.7,
        label=f"single-retailer ({(~multi).sum():,} BARCODEs)",
        color="#999",
    )
    ax.hist(
        sku_count[multi],
        bins=range(1, 46),
        alpha=0.7,
        label=f"multi-retailer ({multi.sum():,} BARCODEs)",
        color="#4C72B0",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 45)
    ax.set_ylim(bottom=0.5)  # log axis floor; top auto but stable via data
    ax.set_xlabel("SKUs per BARCODE")
    ax.set_ylabel("BARCODE count (log)")
    ax.set_title(f"BARCODE multiplicity — {multi.mean():.1%} cross-retailer")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_retailer_country(df: pd.DataFrame, out_path: Path) -> None:
    """Top retailers by SKU count — where the catalog lives."""
    top = df["retailer"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_xlim(0, int(top.values.max()) * 1.05)
    ax.set_xlabel("SKU count")
    ax.set_title("Top 15 retailers by SKU count")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attributes_keys(keys, out_path: Path) -> None:
    """Most frequent attributes key names across retailers."""
    top = pd.Series(dict(keys.most_common(15)))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="crest")
    ax.set_xlim(0, int(top.values.max()) * 1.05)
    ax.set_xlabel("occurrences (first 5k rows)")
    ax.set_title("Top attributes keys")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dtypes(dtype_frame: pd.DataFrame, out_path: Path) -> None:
    """Per-column completeness (non-null) with the numeric/text content signal.

    Lets a reviewer SEE the dataset shape: one bar per column, sorted by
    completeness; color = numeric_like (blue = all-numeric content, gray =
    free text); the numeric_like % is annotated per bar. Data-derived x-axis
    (max*1.1 headroom) per repo convention.
    """
    frame = dtype_frame.sort_values("non_null", ascending=True)
    colors = ["#4C72B0" if ok >= 0.99 else "#BBBBBB" for ok in frame["numeric_like"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    bars = ax.barh(frame["column"], frame["non_null"], color=colors)
    labels = [
        f"{v:,}  ·  {nl:.0%} numeric" if nl >= 0.99 else f"{v:,}"
        for v, nl in zip(frame["non_null"], frame["numeric_like"])
    ]
    ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
    ax.set_xlim(
        0, int(frame["non_null"].max()) * 1.4
    )  # data-derived headroom for labels
    ax.set_xlabel("non-null rows (of 71,623)")
    ax.set_title("Dataset shape: completeness + numeric/text content per column")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print(
        f"retailers: {df['retailer'].nunique():,}  countries: {df['country'].nunique()}"
    )

    # ---- describe CSV (numeric-ish columns; raw strings so counts are honest)
    numeric_cols = [c for c in df.columns if c not in TEXT_COLS]
    desc = df[numeric_cols].describe(include="all").T
    desc.to_csv(CSV_DESCRIBE)
    print(f"wrote {CSV_DESCRIBE}")

    # ---- BARCODE coverage CSV
    known = has_barcode(df)
    per = df.loc[known, "barcode"].value_counts()
    multi_by_barcode = df.loc[known].groupby("barcode")["retailer"].nunique()
    # DISPLAY TABLE ONLY: top-25 BARCODEs + aggregate summary, never the full 15K list.
    barcode_frame = pd.DataFrame(
        {"barcode": per.index, "skus": per.values}
    ).sort_values("skus", ascending=False)
    top = barcode_frame.head(25).copy()
    top.loc[len(top)] = ["TOTAL_BARCODES", len(barcode_frame)]
    top.loc[len(top)] = ["MULTI_RETAILER_BARCODES", int((multi_by_barcode > 1).sum())]
    top.loc[len(top)] = ["MEAN_SKUS_PER_BARCODE", round(per.mean(), 2)]
    top.to_csv(CSV_BARCODE, index=False)
    print(f"wrote {CSV_BARCODE} (display table, {len(top)} rows)")

    # ---- attributes keys CSV
    keys = attributes_keys(df["attributes"])
    pd.DataFrame(keys.most_common(50), columns=["key", "occurrences"]).to_csv(
        CSV_ATTR_KEYS, index=False
    )
    print(f"wrote {CSV_ATTR_KEYS}")

    # ---- dtypes CSV (display table). Loader reads dtype=str, so the stored
    # dtype is "object" everywhere — the numeric_like hint shows the real
    # content signal (fraction of non-null values that parse as a number).
    # Single source: the shared column_profile helper (also used by 01b).
    dtype_frame = column_profile(df)
    dtype_frame[["column", "stored_dtype", "non_null", "numeric_like"]].to_csv(
        CSV_DTYPES, index=False
    )
    print(f"wrote {CSV_DTYPES} (display table, {len(dtype_frame)} rows)")
    for _, r in dtype_frame.iterrows():
        print(
            f"  {r['column']:<22} dtype={r['stored_dtype']:<8} "
            f"non_null={r['non_null']:>6}  numeric_like={r['numeric_like']:.0%}"
        )
    plot_dtypes(dtype_frame, PNG_DTYPES)
    print(f"wrote {PNG_DTYPES}")

    # ---- figures
    plot_barcode_coverage(df, PNG_BARCODE)
    print(f"wrote {PNG_BARCODE}")
    plot_retailer_country(df, PNG_RETAILER)
    print(f"wrote {PNG_RETAILER}")
    plot_attributes_keys(keys, PNG_ATTR)
    print(f"wrote {PNG_ATTR}")


if __name__ == "__main__":
    main()
