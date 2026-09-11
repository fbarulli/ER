"""blocking_audit.py — ported from the monorepo's 05_blocking_feature_selection.py
(the file that answered "which blocking key is better", measured — not argued).

The question: our lane blocks candidate pairs by BRAND (data_pipe pairs all
GTINs sharing a brand → 135,769 gate pairs). Is that the right key? This
audit measures EVERY candidate key the same way:

  blocking recall = share of true (same-product) pairs sharing a block
  candidates      = pairwise classifier cost, sum of n*(n-1)/2 per block

Ground truth = build_true_pairs: same VALID barcode (GS1 checksum — owner
ruling), multi-retailer groups, ALL pairs (uncapped). Cross-retailer by
construction, so RETAILER-as-key is the sanity control (must score ~0).

Three sections (verbatim from the monorepo, adapted to this lane's src/core/):
  A. singles            — each feature alone as the whole key, plus
                          agree_when_both: P(feature equal | both present) —
                          the SCORING-side utility that made volume a scoring
                          feature despite being a broken blocking key.
  B. composites         — curated combinations + the volume cliff.
  C. greedy forward     — recall-first selection under a candidate budget;
                          each step adds the feature that keeps recall
                          highest (ties broken by lower cost).

Sanity gates (hard failures, the lane doctrine):
  - retailer recall must be < 0.25 AND below brand (cross-retailer truth).
  - greedy path must be monotone: recall and candidates never increase.

Outputs (artifacts/results/, SSOT names):
  blocking_feature_audit.csv   singles + composites + greedy path
  blocking_feature_audit.png   recall-vs-candidates scatter (log-x)
"""

from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.blocking import build_true_pairs, eval_blocking
from core.common import (
    RESULTS,
    category_macros,
    load_dataset_deduped,
    plot_dpi,
    training_cfg,
)
from core.text import extract_pack_counts, extract_volume_ml

# MACRO_MAP moved to config (SSOT): config/paths.yaml category_macros —
# read via lib.common.category_macros(), never a module-level copy.
MACRO_MAP = category_macros()

CSV_AUDIT = RESULTS / "blocking_feature_audit.csv"
PNG_AUDIT = RESULTS / "blocking_feature_audit.png"

# AUDIT FIX (round 2 F18, round 3): the audit's policy knobs read the SSOT
# (config/training.yaml audit: block) via training_cfg — were inline
# literals (BUDGET = 5_000_000 / MIN_RECALL = 0.95), a second declaration
# the config could not steer. Same values, config-owned.
BUDGET = int(training_cfg().audit.blocking_budget)
MIN_RECALL = float(training_cfg().audit.blocking_min_recall)


def agree_when_both(raw: pd.Series, pairs) -> float | None:
    """Scoring-side utility: P(feature equal | both members present)."""
    va = raw.loc[[a for a, _ in pairs]].to_numpy()
    vb = raw.loc[[b for _, b in pairs]].to_numpy()
    both = pd.notna(va) & pd.notna(vb)
    if not both.any():
        return None
    return float((va[both] == vb[both]).mean())


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset_deduped()
    pairs = build_true_pairs(df)
    print(f"true pairs (valid-barcode, multi-retailer): {len(pairs):,}")

    # ---- feature columns ----------------------------------------------------
    vol_ml = df["title"].fillna("").map(lambda s: extract_volume_ml(s)[0])
    price = pd.to_numeric(df["price"], errors="coerce")
    # rank(method="min"): ties must land in the SAME bucket — method="first"
    # breaks ties by row order (row-order noise, not a price difference).
    price_bucket = pd.qcut(price.rank(method="min"), 20, labels=False).astype(float)

    features = {
        "brand": df["brand"].fillna("").str.strip().replace("", "NO_BRAND"),
        "strict_category": df["category"].fillna("").str.strip(),
        "macro_category": df["category"].fillna("").map(MACRO_MAP).fillna("UNKNOWN"),
        "volume_ml": vol_ml.fillna("NO_VOL").astype(str),
        "price_bucket": price_bucket.fillna("NO_PRICE").astype(str),
        "country": df["country"].fillna("NO_COUNTRY"),
        "pack_count": df["title"].fillna("").map(
            lambda s: str(sorted(extract_pack_counts(s))) if extract_pack_counts(s) else "NO_PACK"
        ),
        "retailer": df["retailer"].fillna("NO_RETAILER"),  # control
    }
    raw_for_agree = {
        "brand": df["brand"].fillna("").str.strip().replace("", np.nan),
        "strict_category": df["category"].fillna("").str.strip(),
        "macro_category": df["category"].fillna("").map(MACRO_MAP),
        "volume_ml": vol_ml,
        "price_bucket": price_bucket,
        "country": df["country"].fillna(""),
        "pack_count": df["title"].fillna("").map(
            lambda s: str(sorted(extract_pack_counts(s))) if extract_pack_counts(s) else np.nan
        ),
        "retailer": df["retailer"].fillna(""),
    }

    total_pairs = len(df) * (len(df) - 1) // 2
    rows: list[dict] = []

    # ---- A. singles ---------------------------------------------------------
    print("\nA. single features as the whole blocking key:")
    for name, key in features.items():
        rec, cand, n_blocks = eval_blocking(key, pairs)
        agree = agree_when_both(raw_for_agree[name], pairs)
        rows.append(
            {
                "type": "single",
                "key": name,
                "recall": round(rec, 4),
                "candidates": cand,
                "blocks": n_blocks,
                "agree_when_both": round(agree, 4) if agree is not None else None,
            }
        )
        print(
            f"  {name:<16} recall={rec:.3f}  cand={cand:>10,}  "
            f"agree_both={agree:.3f}"
            if agree is not None
            else f"  {name:<16} recall={rec:.3f}  cand={cand:>10,}  agree_both=n/a"
        )

    # ---- control: retailer must be far below brand --------------------------
    rec_retailer = next(r["recall"] for r in rows if r["key"] == "retailer")
    rec_brand = next(r["recall"] for r in rows if r["key"] == "brand")
    if rec_retailer > 0.25 or rec_retailer > rec_brand:
        raise AssertionError(
            f"sanity FAILED: retailer recall {rec_retailer:.3f} "
            "must be far below brand and < 0.25"
        )

    # ---- B. composite frontier ---------------------------------------------
    composites = {
        "brand|macro": features["brand"] + "|" + features["macro_category"],
        "brand|strict": features["brand"] + "|" + features["strict_category"],
        "brand|macro|country": features["brand"]
        + "|"
        + features["macro_category"]
        + "|"
        + features["country"],
        "brand|macro|volume": features["brand"]
        + "|"
        + features["macro_category"]
        + "|"
        + features["volume_ml"],
        "brand|strict|volume": features["brand"]
        + "|"
        + features["strict_category"]
        + "|"
        + features["volume_ml"],
        "brand|macro|price": features["brand"]
        + "|"
        + features["macro_category"]
        + "|"
        + features["price_bucket"],
        "brand|volume": features["brand"] + "|" + features["volume_ml"],
    }
    print("\nB. composite frontier:")
    for name, key in composites.items():
        rec, cand, n_blocks = eval_blocking(key, pairs)
        rows.append(
            {
                "type": "composite",
                "key": name,
                "recall": round(rec, 4),
                "candidates": cand,
                "blocks": n_blocks,
                "agree_when_both": None,
            }
        )
        print(f"  {name:<22} recall={rec:.3f}  cand={cand:>10,}")

    # ---- C. greedy forward selection ---------------------------------------
    chosen: list[str] = []
    key = pd.Series("ALL", index=df.index)
    print(
        f"\nC. greedy forward selection (budget {BUDGET:,} candidates, "
        f"min recall {MIN_RECALL}):"
    )
    print(f"  start: recall=1.000 cand={total_pairs:,}")
    while True:
        best = None
        for name, f in features.items():
            if name in chosen:
                continue
            r, c, _ = eval_blocking(key.astype(str) + "|" + f, pairs)
            if (
                c <= BUDGET
                and r >= MIN_RECALL
                and (best is None or r > best[1] or (r == best[1] and c < best[2]))
            ):
                best = (name, r, c)
        if best is None:
            break
        name, r, c = best
        chosen.append(name)
        key = key.astype(str) + "|" + features[name]
        rows.append(
            {
                "type": "greedy_step",
                "key": " -> ".join(chosen),
                "recall": round(r, 4),
                "candidates": c,
                "blocks": None,
                "agree_when_both": None,
            }
        )
        print(f"  + {name:<16} recall={r:.4f} cand={c:>10,}")

    rec_final, cand_final, _ = eval_blocking(key, pairs)
    print(
        f"\n  final key: {' x '.join(chosen)}  recall={rec_final:.4f} "
        f"candidates={cand_final:,}"
    )
    rows.append(
        {
            "type": "greedy_final",
            "key": " x ".join(chosen),
            "recall": round(rec_final, 4),
            "candidates": cand_final,
            "blocks": None,
            "agree_when_both": None,
        }
    )

    # sanity: greedy path is monotone (recall and candidates never increase)
    greedy = [r for r in rows if r["type"] == "greedy_step"]
    rec_seq = [r["recall"] for r in greedy] + [round(rec_final, 4)]
    cand_seq = [r["candidates"] for r in greedy] + [cand_final]
    if any(a < b - 1e-9 for a, b in pairwise(rec_seq)) or any(
        a < b - 1e-9 for a, b in pairwise(cand_seq)
    ):
        raise AssertionError("sanity FAILED: greedy path not monotone")
    print("  [PASS] greedy path monotone (recall/candidates never increase)")

    # ---- outputs ------------------------------------------------------------
    frame = pd.DataFrame(rows)
    frame.to_csv(CSV_AUDIT, index=False)
    print(f"\nwrote {CSV_AUDIT} (display table, {len(frame)} rows)")

    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    for t, color in [
        ("single", "#BBBBBB"),
        ("composite", "#4C72B0"),
        ("greedy_step", "#C44E52"),
        ("greedy_final", "#2E7D32"),
    ]:
        sub = frame[frame["type"] == t]
        if len(sub):
            ax.scatter(
                sub["candidates"], sub["recall"], s=75, color=color,
                label=t.replace("_", " "), zorder=3,
            )
            for _, r in sub.iterrows():
                ax.annotate(
                    str(r["key"]).split("|")[0][:16],
                    (r["candidates"], r["recall"]),
                    fontsize=9, xytext=(4, 3), textcoords="offset points",
                )
    ax.set_xscale("log")
    ax.set_xlim(1, total_pairs * 1.5)
    ax.set_ylim(0, 1.02)
    ax.axhline(MIN_RECALL, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("candidate pairs (log)")
    ax.set_ylabel("blocking recall")
    ax.set_title(f"Blocking feature audit — recall vs cost (budget {BUDGET:,})")
    ax.legend(fontsize=10)
    fig.savefig(PNG_AUDIT, dpi=plot_dpi())
    plt.close(fig)
    print(f"wrote {PNG_AUDIT}")


if __name__ == "__main__":
    main()
