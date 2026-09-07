"""01b: See the dataset — three visual views of all 71,623 product rows.

The dtype table (01) is aggregate; these plots put the data itself in front
of a reviewer:

  01b_column_scatter.png   columns as bubbles: completeness (x) vs log
                          cardinality (y); color = numeric content, bubble
                          size = cardinality — where the gaps and the
                          diversity live.
  01b_product_space.png   every dot = ONE product: TF-IDF (name+brand+
                          category) -> TruncatedSVD 2-D, colored by category.
                          The literal "scatter plot of the dataset".
  01b_price_strip.png     price across the top retailers, log-y,
                          colored by country (sampled deterministically for
                          render speed; a real "price vs text" view).

No new dependencies: sklearn (already in venv) does TF-IDF + SVD. Every axis
limit is data-derived per the repo plot convention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

from lib.common import RESULTS, SEED, column_profile, load_dataset
from lib.matching import build_vectorizer

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)

PNG_COLUMN = EDA_OUT / "01b_column_scatter.png"
PNG_SPACE = EDA_OUT / "01b_product_space.png"
PNG_PRICE = EDA_OUT / "01b_price_strip.png"
CSV_SPACE = EDA_OUT / "01b_product_space.csv"
CSV_PLOT_DATA = EDA_OUT / "01b_product_space_plot_data.csv"


def plot_column_scatter(
    dtype_info: pd.DataFrame, out_path: Path, total_rows: int
) -> None:
    """Columns as bubbles: completeness (x) vs value diversity (y, log).

    A different cut than the plain completeness x numeric-content plot: the
    y-axis becomes log10(cardinality) so MISSINGNESS and DIVERSITY are shown
    together; color = numeric content (colorbar: red text -> blue numeric),
    bubble size scales with cardinality. One glance isolates the problem
    columns: barcode sits alone at 42% complete / 15k values / fully numeric.
    """
    frame = dtype_info.sort_values("non_null").reset_index(drop=True)
    x = frame["non_null"] / total_rows * 100  # completeness = % of total rows
    y = np.log10(frame["cardinality"].clip(lower=1))
    sizes = 40 + 80 * (y / y.max())
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=frame["numeric_like"],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    for i, r in frame.iterrows():
        ax.annotate(
            r["column"],
            (x.iloc[i], y.iloc[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(50, color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.15, y.max() * 1.12)  # data-derived headroom
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(np.log10([1, 10, 100, 1000, 10000, 100000]))
    ax.set_yticklabels(["1", "10", "100", "1k", "10k", "100k"])
    ax.set_xlabel("completeness (% of rows populated)")
    ax.set_ylabel("unique values (log)")
    ax.set_title("Column view: completeness × diversity × content type")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("numeric content (0=text, 1=numbers)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_product_space(df: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF (title+brand+category) -> SVD 2-D embedding, one row per product.

    The EXACT data fed into 01b_product_space.png: svd_1/svd_2 are the dot
    coordinates, color_group is the category-derived label that determines
    each dot's color (top-8 categories by row count, everything else 'other').
    The plot and the CSVs share this single computation (same vectorizer,
    same seed, same axes) — no second source of truth.
    """
    text = (
        df["title"].fillna("")
        + " "
        + df["brand"].fillna("")
        + " "
        + df["category"].fillna("")
    )
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(text)
    embedding = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(X)
    top_cats = df["category"].value_counts().nlargest(8).index
    embed = pd.DataFrame(
        {
            "product_id": df["product_id"].values,
            "category": df["category"].fillna("").values,
            "svd_1": embedding[:, 0],
            "svd_2": embedding[:, 1],
        }
    )
    embed["color_group"] = embed["category"].where(
        embed["category"].isin(top_cats), "other"
    )
    return embed


def plot_product_space(embed: pd.DataFrame, out_path: Path) -> None:
    """Every dot = one product from the shared SVD embedding (compute_product_space).

    Products whose text is similar land near each other; color = category so
    a reviewer can see whether the product space is structured by category.
    """
    top_cats = embed["category"].value_counts().nlargest(8).index
    colors = {c: col for c, col in zip(top_cats, sns.color_palette("tab10", 8))}
    colors["other"] = "#BBBBBB"
    labels = embed["color_group"]  # shared with the plot-data CSV
    coords = embed[["svd_1", "svd_2"]].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for cat in list(top_cats) + ["other"]:
        mask = labels == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=1.2,
            alpha=0.35,
            color=colors[cat],
            label=f"{cat} ({mask.sum():,})",
        )
    lo = np.percentile(coords, 0.1, axis=0)
    hi = np.percentile(coords, 99.9, axis=0)
    span = (hi - lo) * 0.05  # 5% headroom beyond the 0.1-99.9% data span
    ax.set_xlim(lo[0] - span[0], hi[0] + span[0])
    ax.set_ylim(lo[1] - span[1], hi[1] + span[1])
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title(
        "Product space: TF-IDF(title+brand+category) -> SVD, "
        f"{len(embed):,} products, colored by category"
    )
    ax.legend(fontsize=7, markerscale=6, loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_price_strip(df: pd.DataFrame, out_path: Path) -> None:
    """Price across top retailers (log-y), colored by country, sampled.

    Deterministic sample (SEED) capped per retailer so the strip stays
    renderable; the sample is representative, not a data dump.
    """
    price = pd.to_numeric(df["price"], errors="coerce")
    data = df.assign(price=price).dropna(subset=["price"])
    data = data[data["price"] > 0]
    top = data["retailer"].value_counts().nlargest(10).index
    subset = data[data["retailer"].isin(top)]
    rng = np.random.default_rng(SEED)
    pieces = []
    for _, g in subset.groupby("retailer"):
        pieces.append(g.sample(n=min(len(g), 2000), random_state=rng))
    sampled = pd.concat(pieces).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.stripplot(
        data=sampled,
        x="retailer",
        y="price",
        hue="country",
        palette="tab20",
        alpha=0.4,
        size=2.2,
        jitter=0.3,
        legend=False,
        ax=ax,
    )
    ax.set_yscale("log")
    q = np.quantile(sampled["price"], [0.005, 0.995])
    ax.set_ylim(q[0], q[1])  # data-derived (0.5-99.5 percentile span)
    ax.set_xlim(-0.5, len(top) - 0.5)
    ax.set_xlabel("retailer (top 10 by SKU count)")
    ax.set_ylabel("price (log scale)")
    ax.set_title(
        f"Price per retailer — {len(sampled):,} sampled SKUs "
        f"(≤2,000/retailer, seed {SEED}), colored by country"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # column scatter reuses the shared dtype computation (single source in _common)
    profile = column_profile(df)
    plot_column_scatter(profile, PNG_COLUMN, total_rows=len(df))
    print(f"wrote {PNG_COLUMN}")

    embed = compute_product_space(df)
    # The end result fed into the graph: coordinates + the color label per dot.
    embed[["svd_1", "svd_2"]].to_csv(CSV_SPACE, index=False)
    print(f"wrote {CSV_SPACE} (plotted coordinates only, {len(embed)} rows)")
    embed[["svd_1", "svd_2", "color_group"]].to_csv(CSV_PLOT_DATA, index=False)
    print(
        f"wrote {CSV_PLOT_DATA} (plot input: coords + color label, {len(embed)} rows)"
    )
    plot_product_space(embed, PNG_SPACE)
    print(f"wrote {PNG_SPACE}")

    plot_price_strip(df, PNG_PRICE)
    print(f"wrote {PNG_PRICE}")


if __name__ == "__main__":
    main()
