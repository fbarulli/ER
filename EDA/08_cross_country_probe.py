"""01h: Cross-country probe — does the model handle cross-border matching?

Barcode ground truth is ~95% single-country, so cross-country matching is
unlabeled. This step builds a SILVER-label cross-country proxy (same
brand+category, different country, different barcode) and scores it with the
bi-encoder against three reference populations: in-country positives
(same barcode), random negatives, and hard negatives (same category,
different brand). Also writes a small spot-check CSV of the highest-scoring
cross-country pairs for manual labeling.
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

from lib.blocking import build_pairs
from lib.common import RESULTS, SEED, canonical_volume, load_dataset_deduped
from lib.text import MACRO_MAP

EDA_OUT = RESULTS / "eda"
EDA_OUT.mkdir(parents=True, exist_ok=True)
from lib.nlp import encode_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
from lib.common import DATA_DIR

CACHE = str(DATA_DIR / "embeddings_cache")


def main() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset_deduped().reset_index(drop=True)
    df["macro"] = df["category"].fillna("").map(lambda c: MACRO_MAP.get(c, "OTHER"))
    df["vol"] = canonical_volume(df["title"])["canonical_volume_ml"]

    payload = (
        df["title"].fillna("")
        + " | "
        + df["brand"].fillna("")
        + " | "
        + df["category"].fillna("")
    ).tolist()
    emb, _ = encode_corpus(
        MODEL, payload, batch_size=256, max_seq_length=128, cache_dir=CACHE
    )

    def cos(pairs):
        pairs = np.asarray(pairs)
        return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)

    # reference populations
    pos, neg = build_pairs(df, SEED, 4, 10_000)
    pos_s = cos(pos)
    neg_s = cos(neg)

    # hard negatives: same category, different brand (already-mined signature)
    brand = df["brand"].fillna("").astype(str).to_numpy()
    cat = df["category"].fillna("").astype(str).to_numpy()
    hard_neg = [(a, b) for a, b in neg if cat[a] == cat[b] and brand[a] != brand[b]]
    hard_neg = np.array(hard_neg[:3000])
    hard_s = cos(hard_neg)

    # ---- silver cross-country proxy: same brand+category, different country ----
    cross_pairs = []
    for (b, c), g in df[df["brand"].fillna("") != ""].groupby(
        ["brand", "category"], sort=False
    ):
        by_country = {ct: grp.index[0] for ct, grp in g.groupby("country")}
        if len(by_country) < 2:
            continue
        cts = list(by_country)
        a = by_country[cts[0]]
        bb = by_country[cts[1]]
        cross_pairs.append((a, bb))
        if len(cross_pairs) >= 3000:
            break
    cross_pairs = np.array(cross_pairs)
    cross_s = cos(cross_pairs)
    print(f"cross-country proxy pairs: {len(cross_pairs):,}")

    # volume agreement within the proxy (a sanity read on label quality)
    va, vb = (
        df["vol"].to_numpy()[cross_pairs[:, 0]],
        df["vol"].to_numpy()[cross_pairs[:, 1]],
    )
    both = ~pd.isna(va) & ~pd.isna(vb)
    vol_agree = (both & (va == vb)).sum() / max(both.sum(), 1)
    print(f"cross-country proxy volume agreement (when both present): {vol_agree:.1%}")

    groups = {
        "in-country\npositives (barcode)": pos_s,
        "cross-country\nproxy (brand+cat)": cross_s,
        "hard negatives\n(same cat, diff brand)": hard_s,
        "random negatives": neg_s,
    }
    print("\nscore distributions:")
    for name, s in groups.items():
        print(
            f"  {name:<38} n={len(s):>6}  median={np.median(s):.3f}  p90={np.quantile(s, 0.9):.3f}  >=0.55: {(s >= 0.55).mean():.1%}"
        )

    # ---- plot: ECDFs (reads threshold-clearance %/median/p90 directly off the curve,
    # unlike overlapping histograms which are hard to compare across very different n's) ----
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#BBBBBB"]
    for (name, s), color in zip(groups.items(), colors):
        xs = np.sort(s)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(
            xs,
            ys,
            color=color,
            label=f"{name.replace(chr(10), ' ')} (n={len(s):,})",
            lw=1.8,
        )
    ax.axvline(0.55, color="k", ls="--", lw=1, label="operating threshold 0.55")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("cumulative fraction of pairs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Cross-country probe: does the model treat cross-border pairs as matches?"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(EDA_OUT / "01h_cross_country_probe.png", dpi=150)
    plt.close(fig)

    # ---- spot-check CSV: highest-scoring cross-country pairs ----
    order = np.argsort(-cross_s)[:60]
    title = df["title"].fillna("").astype(str).to_numpy()
    cntry = df["country"].fillna("").astype(str).to_numpy()
    bnd = df["brand"].fillna("").astype(str).to_numpy()
    spot = pd.DataFrame(
        {
            "title_a": title[cross_pairs[order, 0]],
            "title_b": title[cross_pairs[order, 1]],
            "brand": bnd[cross_pairs[order, 0]],
            "country_a": cntry[cross_pairs[order, 0]],
            "country_b": cntry[cross_pairs[order, 1]],
            "vol_a": va[order],
            "vol_b": vb[order],
            "cosine": np.round(cross_s[order], 4),
        }
    )
    spot.to_csv(EDA_OUT / "01h_cross_country_spotcheck.csv", index=False)
    print(
        f"\nwrote 01h_cross_country_probe.png + 01h_cross_country_spotcheck.csv ({len(spot)} pairs)"
    )
    print("\nsample spot-check pairs:")
    print(spot.head(8).to_string(index=False, max_colwidth=26))


if __name__ == "__main__":
    main()
