"""01c: Sparsity, variability & noise profile — the dataset's noise budget.

Implements the mixed-data noise checklist (variability vs noise, per column
type): numeric stats + 1.5xIQR outliers, BARCODE price/brand conflicts,
categorical cardinality + entropy + long-tail sparsity + case variants,
text length/vocab + HTML/placeholder noise, and semantic noise (weight units
in a beverage catalog). Every number below is computed once and written to
one DISPLAY TABLE; plots carry data-derived limits (repo convention).

Outputs (RESULTS = project/experiments/results/euromonitor/):
  01c_sparsity_noise.csv   one display table: group | metric | value
  01c_price_outliers.png   price distribution + 1.5xIQR noise bounds
  01c_brand_long_tail.png  Pareto curve of brand concentration
  01c_cardinality.png      cardinality + entropy per categorical column
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

from lib.common import RESULTS, load_dataset
from lib.text import NOISE_HTML_RE, NOISE_PLACEHOLDER_RE, PACK_RE, WEIGHT_UNIT_RE

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)

CSV_PROFILE = EDA_OUT / "01c_sparsity_noise.csv"
PNG_PRICE = EDA_OUT / "01c_price_outliers.png"
PNG_BRAND = EDA_OUT / "01c_brand_long_tail.png"
PNG_CARD = EDA_OUT / "01c_cardinality.png"

RARE_CUTOFF = 5  # values appearing <=5x are long-tail noise candidates


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    rows: list[dict] = []

    def add(group: str, metric: str, value) -> None:
        rows.append({"group": group, "metric": metric, "value": value})

    # ---- numeric: price -----------------------------------------------------
    price = pd.to_numeric(df["price"], errors="coerce")
    p = price.dropna()
    q1, q3 = p.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_pct = float(((p < lo) | (p > hi)).mean())
    add("numeric", "price_n", len(p))
    add("numeric", "price_missing", int(price.isna().sum()))
    add("numeric", "price_min", round(float(p.min()), 4))
    add("numeric", "price_max", round(float(p.max()), 4))
    add("numeric", "price_mean", round(float(p.mean()), 4))
    add("numeric", "price_std", round(float(p.std()), 4))
    add("numeric", "price_cv", round(float(p.std() / p.mean()), 4))
    add("numeric", "price_iqr", round(float(iqr), 4))
    add("numeric", "price_outlier_pct_1_5iqr", round(out_pct, 4))

    # ---- numeric-like identifiers: barcode, product_id ------------------------------
    barcode = df["barcode"].fillna("").str.strip()
    known = barcode[barcode.str.len() > 0]
    add("identifier", "barcode_cardinality", int(known.nunique()))
    add("identifier", "barcode_coverage_pct", round(len(known) / len(df), 4))
    add("identifier", "barcode_non_digit_rows", int(known.str.contains(r"\D").sum()))
    add(
        "identifier",
        "barcode_anomalous_length_values",
        int(known[~known.str.len().isin([8, 12, 13, 14])].nunique()),
    )
    add(
        "identifier",
        "product_id_unique_pct",
        round(df["product_id"].dropna().nunique() / len(df), 4),
    )

    # ---- categorical: cardinality, entropy, sparsity, case variants -----------
    for col in ["retailer", "country", "brand", "category"]:
        s = df[col].dropna().astype(str).str.strip()
        vc = s.value_counts()
        card = len(vc)
        ent = float(entropy(vc.values / vc.sum(), base=2))
        rare = vc[vc <= RARE_CUTOFF]
        add("categorical", f"{col}_cardinality", card)
        add("categorical", f"{col}_entropy_bits", round(ent, 2))
        add("categorical", f"{col}_sparsity_pct_rare", round(len(rare) / card, 4))
        add("categorical", f"{col}_longtail_row_share", round(rare.sum() / len(s), 4))
        low = s.str.lower()
        orig_per_low = s.groupby(low).nunique()
        variants = orig_per_low[orig_per_low > 1]
        add("categorical", f"{col}_case_variant_groups", len(variants))
        add(
            "categorical",
            f"{col}_case_variant_rows",
            int(low.isin(variants.index).sum()),
        )

    # ---- text: length / vocab / html / placeholder ----------------------------
    for col in ["title", "description"]:
        s = df[col].dropna().astype(str)
        add("text", f"{col}_avg_length", round(float(s.str.len().mean()), 1))
        add("text", f"{col}_avg_words", round(float(s.str.split().str.len().mean()), 1))
        add(
            "text",
            f"{col}_vocab_size",
            int(s.str.lower().str.split().explode().nunique()),
        )
        add(
            "text",
            f"{col}_html_noise_pct",
            round(float(s.str.contains(NOISE_HTML_RE, na=False).mean()), 4),
        )
        add(
            "text",
            f"{col}_placeholder_pct",
            round(float(s.str.contains(NOISE_PLACEHOLDER_RE, na=False).mean()), 4),
        )
        add("text", f"{col}_empty_pct", round(float(s.str.strip().eq("").mean()), 4))

    # ---- semantic / relational noise -------------------------------------------
    # Same BARCODE, different prices or brands: the BARCODE group is internally
    # contradictory, so at least one retailer row is wrong.
    known_rows = df[barcode.str.len() > 0].copy()
    known_rows["price"] = pd.to_numeric(known_rows["price"], errors="coerce")
    with_price = known_rows.dropna(subset=["price"])
    barcode_prices = with_price.groupby("barcode")["price"].nunique()
    add(
        "semantic",
        "barcode_price_conflict_pct_all",
        round(float((barcode_prices > 1).mean()), 4),
    )
    multi = known_rows[
        known_rows.groupby("barcode")["retailer"].transform("nunique") > 1
    ]
    multi_prices = multi.dropna(subset=["price"]).groupby("barcode")["price"].nunique()
    add(
        "semantic",
        "barcode_price_conflict_pct_multi",
        round(float((multi_prices > 1).mean()), 4),
    )
    multi_brands = multi.groupby("barcode")["brand"].nunique()
    add(
        "semantic",
        "barcode_brand_conflict_pct_multi",
        round(float((multi_brands > 1).mean()), 4),
    )
    weight_rows = df["title"].fillna("").str.contains(WEIGHT_UNIT_RE, na=False)
    add("semantic", "weight_unit_rows", int(weight_rows.sum()))
    add("semantic", "weight_unit_share_pct", round(float(weight_rows.mean()), 4))
    pack_rows = (
        df["title"]
        .fillna("")
        .map(lambda s: bool(PACK_RE.search(s)) if pd.notna(s) else False)
    )
    add("semantic", "pack_structure_rows", int(pack_rows.sum()))
    add("semantic", "pack_structure_share_pct", round(float(pack_rows.mean()), 4))

    # ---- write display table ---------------------------------------------------
    profile = pd.DataFrame(rows)
    profile.to_csv(CSV_PROFILE, index=False)
    print(f"wrote {CSV_PROFILE} (display table, {len(profile)} rows)")

    # ---- figures ----------------------------------------------------------------
    logp = np.log10(p[p > 0])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    counts, _, _ = ax.hist(logp, bins=80, color="#4C72B0")
    for bound, label in [(lo, "Q1−1.5·IQR"), (hi, "Q3+1.5·IQR")]:
        if 0 < bound < p.max() * 1.5:
            ax.axvline(
                np.log10(bound),
                color="#C44E52",
                linestyle="--",
                linewidth=1,
                label=label,
            )
    ax.set_xlim(logp.min(), logp.max())  # data-derived
    ax.set_ylim(0, counts.max() * 1.1)  # data-derived
    ax.set_xticks(np.log10([0.01, 0.1, 1, 10, 100, 1000, 10000]))
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1k", "10k"])
    ax.set_xlabel("price (log10)")
    ax.set_ylabel("SKU count")
    ax.set_title(f"Price distribution — {out_pct:.1%} of values beyond 1.5×IQR")
    ax.legend(fontsize=8)
    fig.savefig(PNG_PRICE, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_PRICE}")

    vc = df["brand"].dropna().value_counts()
    cum = vc.cumsum() / vc.sum()
    top80 = int(np.argmax(cum.values >= 0.8)) + 1
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(np.arange(1, len(vc) + 1), cum.values * 100, color="#4C72B0", lw=1.5)
    ax.axhline(80, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(1, len(vc))  # data-derived
    ax.set_ylim(0, 100.5)
    ax.set_xlabel("brand rank (log)")
    ax.set_ylabel("cumulative % of rows")
    ax.set_title(
        f"Brand concentration: {len(vc):,} brands, top-80% of rows at rank {top80:,}"
    )
    fig.savefig(PNG_BRAND, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_BRAND}")

    cat_cols = ["retailer", "country", "brand", "category"]
    card_vals = [
        int(df[c].dropna().astype(str).str.strip().nunique()) for c in cat_cols
    ]
    ent_vals = []
    for c in cat_cols:
        s = df[c].dropna().astype(str).str.strip()
        ent_vals.append(
            round(float(entropy(s.value_counts(normalize=True), base=2)), 2)
        )
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(len(cat_cols))
    bars = ax.bar(x, card_vals, color=sns.color_palette("muted", len(cat_cols)))
    ax.bar_label(bars, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(card_vals) * 10)  # log floor + data-derived headroom
    ax.set_xticks(x, cat_cols)
    ax.set_ylabel("unique values (log)")
    for i, ent in enumerate(ent_vals):
        ax.text(
            i, card_vals[i] * 2.2, f"H={ent}", ha="center", fontsize=8, color="#C44E52"
        )
    ax.set_title("Categorical cardinality (entropy H in bits)")
    fig.savefig(PNG_CARD, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_CARD}")

    # ---- printed report -----------------------------------------------------------
    print(
        f"\nprice: n={len(p):,}  cv={p.std() / p.mean():.2f}  "
        f"outliers(1.5xIQR)={out_pct:.1%}"
    )
    non_digit = int(known.str.contains(r"\D").sum())
    print(
        f"barcode: {known.nunique():,} values, {len(known) / len(df):.1%} coverage, "
        f"{non_digit:,} rows with non-digit chars"
    )
    print(
        f"brand conflicts within multi-retailer BARCODE groups: "
        f"{float((multi_brands > 1).mean()):.1%} of {len(multi_brands):,} groups"
    )
    print(
        f"price conflicts: all-BARCODE {float((barcode_prices > 1).mean()):.1%}  "
        f"multi-retailer {float((multi_prices > 1).mean()):.1%}"
    )
    print(
        f"weight-unit rows (semantic noise): {int(weight_rows.sum()):,} "
        f"({weight_rows.mean():.2%})  |  pack-structure rows: "
        f"{int(pack_rows.sum()):,} ({pack_rows.mean():.2%})"
    )


if __name__ == "__main__":
    main()
