"""Owner's model-evaluation script (bundle-adapted paths).

Needs embedding_similarities.csv (zero_shot_similarities.py) and
labeled_pairs.csv (true_label per pair — generate from the gate's
auto_duplicate rule or a manual review round).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from lib.common import RESULTS, SEED, F, plot_dpi

LABELED_PAIRS_CSV = RESULTS / F["labeled_pairs"]
EMBED_SIM_CSV = RESULTS / F["embedding_similarities"]
CANON_CSV = RESULTS / F["canonical_records"]
from lib.common import load_config

MODEL_COLUMNS = dict(load_config()["sim_columns"])

labeled = pd.read_csv(LABELED_PAIRS_CSV, dtype={"gtin1": str, "gtin2": str})
if not EMBED_SIM_CSV.exists():
    raise SystemExit(
        f"{EMBED_SIM_CSV.name} missing — run TRAIN/zero_shot_sims.py first"
    )
emb_sim = pd.read_csv(EMBED_SIM_CSV, dtype={"gtin1": str, "gtin2": str})
# gtin as str: a UPC-12 canonical (leading zero) read as int64 NaNs out the
# canon map join — latent dtype bug (0 rows affected TODAY, but any
# leading-zero GTIN would silently lose its canonical)
canon = pd.read_csv(CANON_CSV, dtype={"gtin": str})
# AUDIT FIX (round 2 F14, round 3): inner-merge drop accounting — nothing
# may drop silently (transparency contract). Measured on the real CSVs:
# 19,916 -> 19,916, zero rows lost TODAY; this print keeps any drift loud,
# and an empty merge is a hard stop (upstream sims are missing entirely).
_n_before = len(labeled)
df = labeled.merge(emb_sim, on=["gtin1", "gtin2"], how="inner")
_n_after = len(df)
print(
    f"[merge] labeled x embedding_similarities (inner on gtin1/gtin2): "
    f"{_n_before:,} -> {_n_after:,} rows ({_n_before - _n_after:,} dropped)"
)
if _n_after == 0:
    raise SystemExit(
        "[merge] inner join lost ALL rows — embedding_similarities.csv "
        "does not cover labeled_pairs (rerun TRAIN/zero_shot_sims.py)"
    )
gtin_to_canon = dict(zip(canon["gtin"].astype(str), canon["canonical"].astype(str)))
df["canon1"] = df["gtin1"].map(gtin_to_canon)
df["canon2"] = df["gtin2"].map(gtin_to_canon)


def evaluate_model(
    df: pd.DataFrame,
    sim_col: str,
    true_col: str = "true_label",
    threshold: float | None = None,
) -> dict:
    y_true = df[true_col].values
    y_scores = df[sim_col].values
    roc_auc = roc_auc_score(y_true, y_scores)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        threshold = thresholds[np.argmax(j_scores)]
        print(f"  Optimal threshold (Youden): {threshold:.4f}")
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "roc_auc": roc_auc,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


summary_rows = []
# models whose sim column is present in the sweep output (a partial sweep
# is evaluated — never crash on a missing column, say it loudly instead)
_missing = [c for c in MODEL_COLUMNS.values() if c not in df.columns]
if _missing:
    print(f"[warn] no similarity column yet (sweep incomplete): {_missing}")
for model_name, sim_col in MODEL_COLUMNS.items():
    if sim_col not in df.columns:
        print(f"\nMODEL: {model_name} — skipped (no {sim_col} in sweep csv)")
        continue
    print(f"\n{'=' * 70}\nMODEL: {model_name}\n{'=' * 70}")
    metrics = evaluate_model(df, sim_col)
    summary_rows.append({"model": model_name, **metrics})
    print(f"  ROC AUC      : {metrics['roc_auc']:.4f}")
    print(f"  Threshold    : {metrics['threshold']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(
        f"  TP: {metrics['tp']}  TN: {metrics['tn']}  FP: {metrics['fp']}  FN: {metrics['fn']}"
    )
    df_model = df.copy()
    df_model["pred"] = (df_model[sim_col] >= metrics["threshold"]).astype(int)
    for kind, m in (
        ("False Positives", (df_model.true_label == 0) & (df_model.pred == 1)),
        ("False Negatives", (df_model.true_label == 1) & (df_model.pred == 0)),
    ):
        sub = df_model[m]
        print(
            f"\n  {kind} (predicted duplicate but not): {len(sub)}"
            if kind.startswith("False P")
            else f"\n  {kind} (actual duplicate but missed): {len(sub)}"
        )
        for _, row in (
            sub.sample(min(3, len(sub)), random_state=SEED).iterrows()
            if len(sub)
            else []
        ):
            print(f"    sim={row[sim_col]:.3f}")
            print(f"      GTIN1 {row['gtin1']}: {str(row['canon1'])[:80]}")
            print(f"      GTIN2 {row['gtin2']}: {str(row['canon2'])[:80]}")

summary_df = pd.DataFrame(summary_rows)
print("\nSUMMARY TABLE:")
print(summary_df.to_string(index=False))
summary_df.to_csv(RESULTS / F["model_evaluation_summary"], index=False)
print(f"\nSaved summary to {RESULTS / F['model_evaluation_summary']}")

# ── per-model result plots (zero-shot embedding similarity) ──────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = [c for c in MODEL_COLUMNS.values() if c in df.columns]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# (a) ROC curves — all models overlaid
for m in models:
    y = df["true_label"].values
    s = df[m].values
    fpr, tpr, _ = roc_curve(y, s)
    axes[0].plot(fpr, tpr, lw=1.8, label=f"{m} (AUC {roc_auc_score(y, s):.3f})")
axes[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
axes[0].set_xlabel("false-positive rate")
axes[0].set_ylabel("true-positive rate")
_n_pos = int((df["true_label"] == 1).sum())
_n_neg = int((df["true_label"] == 0).sum())
axes[0].set_title(f"ROC — zero-shot gate pairs (n: pos={_n_pos:,} / neg={_n_neg:,})")
axes[0].legend(fontsize=9)
# (b) per-model metric bars
x = np.arange(len(summary_df))
w = 0.27
for k, (metric, color) in enumerate(
    (("precision", "#4c72b0"), ("recall", "#55a868"), ("f1", "#c44e52"))
):
    axes[1].bar(x + (k - 1) * w, summary_df[metric], w, color=color, label=metric)
# actual confusion counts per model above each metric triplet
for _i, _r in summary_df.iterrows():
    axes[1].text(
        x[_i] + w,
        1.02,
        f"TP {_r['tp']:,}\nFP {_r['fp']:,}\nFN {_r['fn']:,}",
        ha="center",
        fontsize=7,
    )
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary_df["model"], fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].set_title("precision / recall / F1 at Youden threshold")
axes[1].legend(fontsize=9)
fig.tight_layout()
out1 = RESULTS / "model_comparison_roc.png"
fig.savefig(out1, dpi=plot_dpi())  # SSOT (audit round 2 F03)
plt.close(fig)
print(f"[plot] {out1}")

# score distributions by class — one panel per model
fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4), squeeze=False)
for i, m in enumerate(models):
    ax = axes[0][i]
    pos_s = df.loc[df.true_label == 1, m]
    neg_s = df.loc[df.true_label == 0, m]
    ax.hist(pos_s, bins=50, alpha=0.6, color="#55a868", label=f"pos (n={len(pos_s):,})")
    ax.hist(
        neg_s, bins=50, alpha=0.6, color="#c44e52", label=f"hard-neg (n={len(neg_s):,})"
    )
    thr = summary_df.loc[summary_df.model == m, "threshold"]
    if len(thr):
        ax.axvline(
            float(thr.iloc[0]), color="black", ls="--", lw=1.2, label="Youden thr"
        )
    ax.set_title(m, fontsize=10)
    ax.set_xlabel("cosine similarity")
    ax.legend(fontsize=8)
fig.suptitle("zero-shot similarity by class (canonical texts)", fontsize=11)
fig.tight_layout()
out2 = RESULTS / "model_score_distributions.png"
fig.savefig(out2, dpi=plot_dpi())  # SSOT (audit round 2 F03)
plt.close(fig)
print(f"[plot] {out2}")
