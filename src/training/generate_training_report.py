"""Generate post-run holdout metrics and plots from downloaded artifacts.

The training lane already writes the metric histories and a scored test-pair
CSV.  This report keeps those values together and adds the plots that are
useful for review: loss curves, ranking metrics, operating metrics, ROC/PR,
score distributions, and confusion matrices at both the DEV-fit Youden
threshold and the configured fixed threshold.

Example::

    python -m training.generate_training_report \
      --metrics training_results/<run>/worker_1/*fold_metrics.csv \
      --pairs training_results/<run>/worker_1/*fold0_pairs.csv \
      --out-dir training_results/<run>/worker_1/report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core.common import plot_dpi


def _json_list(value) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    try:
        parsed = json.loads(str(value))
        return [float(x) for x in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _metric_column(columns: list[str], prefix: str) -> str | None:
    matches = sorted(c for c in columns if c.startswith(prefix))
    return matches[0] if matches else None


def _confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y, scores >= threshold, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tn + tp) / len(y)) if len(y) else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
    }


def _score_overlap(negative: np.ndarray, positive: np.ndarray) -> float:
    """Histogram overlap coefficient for the two score populations."""
    if not len(negative) or not len(positive):
        return float("nan")
    bins = np.linspace(-1.0, 1.0, 101)
    neg_hist, _ = np.histogram(negative, bins=bins, density=True)
    pos_hist, _ = np.histogram(positive, bins=bins, density=True)
    return float(np.minimum(neg_hist, pos_hist).sum() * (bins[1] - bins[0]))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=plot_dpi(), bbox_inches="tight")
    plt.close(fig)
    print(f"[report] {path}", flush=True)


def generate_report(
    metrics_path: str | Path,
    pair_paths: list[str | Path],
    out_dir: str | Path,
    train_score_paths: list[str | Path] | None = None,
    random_score_paths: list[str | Path] | None = None,
) -> dict:
    metrics_path = Path(metrics_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    ok = metrics[metrics["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError(f"no successful folds in {metrics_path}")

    pair_frames = []
    for path in pair_paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            pair_frames.append(frame)
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    train_score_frames = []
    for path in train_score_paths or []:
        frame = pd.read_csv(path)
        if not frame.empty:
            train_score_frames.append(frame)
    train_scores = (
        pd.concat(train_score_frames, ignore_index=True)
        if train_score_frames
        else pd.DataFrame()
    )
    random_score_frames = []
    for path in random_score_paths or []:
        frame = pd.read_csv(path)
        if not frame.empty:
            random_score_frames.append(frame)
    random_scores = (
        pd.concat(random_score_frames, ignore_index=True)
        if random_score_frames
        else pd.DataFrame()
    )

    f1_col = _metric_column(list(ok.columns), "f1_at_")
    precision_col = _metric_column(list(ok.columns), "precision_at_")
    recall_col = _metric_column(list(ok.columns), "recall_at_")
    summary_cols = [
        "fold", "auc", "pr_auc", "average_precision", "acc_at_thr",
        "hits_at_1", "precision_at_1", "recall_at_1", "precision_at_5",
        "recall_at_5", "precision_at_10", "recall_at_10", "best_dev_ap",
        "final_train_loss", "train_s", "fold_s",
    ]
    for column in (f1_col, precision_col, recall_col):
        if column and column not in summary_cols:
            summary_cols.append(column)
    summary = ok[[c for c in summary_cols if c in ok.columns]].copy()
    summary.to_csv(out / "metrics_summary.csv", index=False)

    aggregate = {
        c: {"mean": float(ok[c].mean()), "std": float(ok[c].std(ddof=0))}
        for c in summary.columns
        if c != "fold" and pd.api.types.is_numeric_dtype(ok[c])
    }
    (out / "metrics_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Training and DEV validation loss histories.
    fig, axes = plt.subplots(
        1, len(ok), figsize=(4.5 * len(ok), 3.8), squeeze=False
    )
    for i, (_, row) in enumerate(ok.iterrows()):
        ax = axes[0, i]
        train_loss = _json_list(row.get("train_loss_hist"))
        dev_loss = _json_list(row.get("dev_loss_hist"))
        dev_ap = _json_list(row.get("dev_ap_hist"))
        if train_loss:
            ax.plot(train_loss, label="train loss", color="#c44e52")
        if dev_loss:
            ax.plot(dev_loss, label="dev loss", color="#4c72b0", ls="--")
        if dev_ap:
            ax.axvline(
                int(np.argmax(dev_ap)), color="#55a868", ls=":",
                label=f"best dev AP {max(dev_ap):.3f}",
            )
        ax.set_title(f"fold {int(row['fold'])}")
        ax.set_xlabel("logged step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Training vs DEV validation loss")
    fig.tight_layout()
    _save(fig, out / "training_vs_dev_loss.png")

    # Same curves on an epoch axis. The trainer records the fractional epoch
    # for every loss/evaluation event; fall back to event index for older CSVs
    # that predate the explicit epoch histories.
    fig, axes = plt.subplots(
        1, len(ok), figsize=(4.5 * len(ok), 3.8), squeeze=False
    )
    for i, (_, row) in enumerate(ok.iterrows()):
        ax = axes[0, i]
        train_loss = _json_list(row.get("train_loss_hist"))
        dev_loss = _json_list(row.get("dev_loss_hist"))
        train_epoch = _json_list(row.get("train_epoch_hist"))
        dev_epoch = _json_list(row.get("dev_epoch_hist"))
        if len(train_epoch) != len(train_loss):
            train_epoch = list(range(1, len(train_loss) + 1))
        if len(dev_epoch) != len(dev_loss):
            dev_epoch = list(range(1, len(dev_loss) + 1))
        if train_loss:
            ax.plot(train_epoch, train_loss, marker="o", label="train loss")
        if dev_loss:
            ax.plot(dev_epoch, dev_loss, marker="o", label="dev loss")
        ax.set_title(f"fold {int(row['fold'])}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Training vs DEV validation loss by epoch")
    fig.tight_layout()
    _save(fig, out / "training_vs_dev_loss_by_epoch.png")

    # Ranking and threshold metrics from the holdout CSV.
    rank_names = [
        "hits_at_1", "precision_at_1", "recall_at_1", "precision_at_5",
        "recall_at_5", "precision_at_10", "recall_at_10",
    ]
    rank_present = [c for c in rank_names if c in ok.columns]
    if rank_present:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(rank_present))
        width = 0.8 / max(len(ok), 1)
        for i, (_, row) in enumerate(ok.iterrows()):
            ax.bar(x + i * width, row[rank_present].astype(float), width, label=f"fold {int(row['fold'])}")
        ax.set_xticks(x + width * (len(ok) - 1) / 2, [c.replace("_", " ") for c in rank_present], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Holdout ranking metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "holdout_ranking_metrics.png")

    operating = ["auc", "pr_auc", "acc_at_thr"]
    operating += [c for c in (f1_col, precision_col, recall_col) if c]
    operating = [c for c in operating if c in ok.columns]
    if operating:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(operating))
        width = 0.8 / max(len(ok), 1)
        for i, (_, row) in enumerate(ok.iterrows()):
            ax.bar(x + i * width, row[operating].astype(float), width, label=f"fold {int(row['fold'])}")
        ax.set_xticks(x + width * (len(ok) - 1) / 2, [c.replace("_", " ") for c in operating], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Holdout AUC / AP / accuracy / F1 / precision / recall")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "holdout_operating_metrics.png")

    confusion_rows = []
    ranking_rows = []
    if not pairs.empty:
        pairs["fold"] = pairs["fold"].astype(int)
        fig, axes = plt.subplots(1, len(ok), figsize=(4.5 * len(ok), 4), squeeze=False)
        for i, (_, row) in enumerate(ok.iterrows()):
            fold = int(row["fold"])
            part = pairs[pairs["fold"].eq(fold)]
            if part.empty:
                continue
            y = part["label"].to_numpy(dtype=int)
            scores = part["score"].to_numpy(dtype=float)
            order = np.argsort(-scores, kind="stable")
            ranked = y[order]
            for k in (1, 5, 10):
                top = ranked[: min(k, len(ranked))]
                ranking_rows.append(
                    {
                        "fold": fold,
                        "k": k,
                        "hits_at_k": int(top.sum()),
                        "available_k": int(len(top)),
                        "precision_at_k": float(top.mean()) if len(top) else 0.0,
                        "recall_at_k": float(top.sum() / max(int(y.sum()), 1)),
                    }
                )
            thresholds = {
                "dev_youden": float(row["youden_thr"]),
                "fixed": float(row.get("threshold_at_90pct_recall", 0.55)),
            }
            # The configured fixed operating threshold is encoded in the
            # f1/precision/recall column suffix, not the 90%-recall threshold.
            if f1_col and "_at_" in f1_col:
                thresholds["fixed"] = float(f1_col.rsplit("_at_", 1)[1])
            mats = []
            for name, threshold in thresholds.items():
                result = _confusion(y, scores, threshold)
                confusion_rows.append({"fold": fold, "operating_point": name, **result})
                mats.append((name, result))
            ax = axes[0, i]
            # Show the ship threshold (DEV-fit Youden); fixed threshold stays
            # in confusion_matrices.csv for an explicit operating comparison.
            cm = confusion_matrix(y, scores >= thresholds["dev_youden"], labels=[0, 1])
            ax.imshow(cm, cmap="Blues")
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, int(cm[r, c]), ha="center", va="center")
            ax.set_xticks([0, 1], ["pred 0", "pred 1"])
            ax.set_yticks([0, 1], ["true 0", "true 1"])
            ax.set_title(f"fold {fold}\nDEV Youden={thresholds['dev_youden']:.3f}")
        fig.suptitle("Holdout confusion matrix (DEV-fit threshold applied to TEST)")
        fig.tight_layout()
        _save(fig, out / "holdout_confusion_matrix.png")

        confusion_df = pd.DataFrame(confusion_rows)
        confusion_df.to_csv(out / "confusion_matrices.csv", index=False)
        pd.DataFrame(ranking_rows).to_csv(out / "ranking_hits_at_k.csv", index=False)

        # ROC, PR and class score distributions use only the scored TEST pairs.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for fold, part in pairs.groupby("fold", sort=True):
            y = part["label"].to_numpy(dtype=int)
            scores = part["score"].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, scores)
            prec, rec, _ = precision_recall_curve(y, scores)
            axes[0].plot(fpr, tpr, label=f"fold {fold} AUC={roc_auc_score(y, scores):.3f}")
            axes[1].plot(rec, prec, label=f"fold {fold} AP={average_precision_score(y, scores):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="false-positive rate", ylabel="true-positive rate", title="TEST ROC")
        axes[1].set(xlabel="recall", ylabel="precision", title="TEST precision-recall")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, out / "holdout_roc_pr_curves.png")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, color in ((1, "#4c72b0"), (0, "#c44e52")):
            values = pairs.loc[pairs["label"].eq(label), "score"]
            ax.hist(values, bins=30, alpha=0.55, color=color, label=f"label {label} (n={len(values):,})")
        ax.set(
            xlabel="raw cosine score",
            ylabel="pairs",
            title="TEST raw cosine score distributions",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "holdout_score_distributions.png")

        # Explicit train-vs-holdout class-overlap view. A widening gap on
        # train with persistent overlap on holdout is the visual overfitting
        # signature; similar overlap in both panels points to underfitting or
        # noisy/insufficient features.
        distributions = [("train", train_scores), ("holdout", pairs)]
        overlap_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, (split_name, frame) in zip(axes, distributions):
            if frame.empty:
                ax.set_title(f"{split_name.title()} scores unavailable")
                ax.axis("off")
                continue
            negative = frame.loc[frame["label"].eq(0), "score"].to_numpy(dtype=float)
            positive = frame.loc[frame["label"].eq(1), "score"].to_numpy(dtype=float)
            overlap = _score_overlap(negative, positive)
            overlap_rows.append(
                {
                    "split": split_name,
                    "overlap_coefficient": overlap,
                    "label_0_mean": float(np.mean(negative)) if len(negative) else float("nan"),
                    "label_1_mean": float(np.mean(positive)) if len(positive) else float("nan"),
                    "label_0_n": int(len(negative)),
                    "label_1_n": int(len(positive)),
                }
            )
            bins = np.linspace(-1, 1, 51)
            ax.hist(negative, bins=bins, density=True, alpha=0.45, label="label 0")
            ax.hist(positive, bins=bins, density=True, alpha=0.45, label="label 1")
            ax.set_title(f"{split_name.title()} (overlap={overlap:.3f})")
            ax.set_xlabel("raw cosine score")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("density")
        fig.suptitle("Raw cosine score distributions: train vs holdout")
        fig.tight_layout()
        _save(fig, out / "train_holdout_score_distributions.png")
        pd.DataFrame(overlap_rows).to_csv(out / "score_distribution_overlap.csv", index=False)

    random_easy_plot = out / "random_easy_score_distributions.png"
    random_easy_csv = out / "random_easy_metrics.csv"
    if not random_scores.empty:
        rows = []
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        hard_positive = pairs.loc[pairs["label"].eq(1), "score"].to_numpy(dtype=float)
        hard_negative = pairs.loc[pairs["label"].eq(0), "score"].to_numpy(dtype=float)
        random_positive = random_scores.loc[
            random_scores["population"].eq("holdout_pos"), "score"
        ].to_numpy(dtype=float)
        random_negative = random_scores.loc[
            random_scores["population"].eq("random_neg"), "score"
        ].to_numpy(dtype=float)
        panels = [
            ("hard negative", hard_positive, hard_negative),
            ("random/easy negative", random_positive, random_negative),
        ]
        for ax, (name, positive, negative) in zip(axes, panels):
            if not len(positive) or not len(negative):
                ax.set_title(f"{name}: unavailable")
                ax.axis("off")
                continue
            y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
            s = np.r_[positive, negative]
            rows.extend(
                [
                    {
                        "population": name,
                        "label": 1,
                        "n": len(positive),
                        "median": float(np.median(positive)),
                        "mean": float(np.mean(positive)),
                    },
                    {
                        "population": name,
                        "label": 0,
                        "n": len(negative),
                        "median": float(np.median(negative)),
                        "mean": float(np.mean(negative)),
                    },
                    {
                        "population": name,
                        "label": "auc",
                        "n": len(s),
                        "median": float(roc_auc_score(y, s)),
                        "mean": float(average_precision_score(y, s)),
                    },
                ]
            )
            bins = np.linspace(-1, 1, 51)
            ax.hist(negative, bins=bins, density=True, alpha=0.45, label="label 0")
            ax.hist(positive, bins=bins, density=True, alpha=0.45, label="label 1")
            ax.set_title(
                f"{name}\nlabel 0 median={np.median(negative):.3f}"
            )
            ax.set_xlabel("raw cosine score")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("density")
        fig.suptitle("Hard vs random/easy negative score separation")
        fig.tight_layout()
        _save(fig, random_easy_plot)
        pd.DataFrame(rows).to_csv(random_easy_csv, index=False)

    report = {
        "metrics": str(metrics_path),
        "pairs": [str(Path(p)) for p in pair_paths],
        "folds": int(len(ok)),
        "score_basis": "normalized_embedding_dot_product_raw_cosine",
        "rule_based_reconciliation_applied": False,
        "summary_csv": str(out / "metrics_summary.csv"),
        "loss_by_epoch_plot": str(out / "training_vs_dev_loss_by_epoch.png"),
        "train_holdout_score_plot": str(out / "train_holdout_score_distributions.png"),
        "score_overlap_csv": str(out / "score_distribution_overlap.csv"),
        "random_easy_score_plot": str(random_easy_plot) if random_scores is not None and not random_scores.empty else None,
        "random_easy_metrics_csv": str(random_easy_csv) if random_scores is not None and not random_scores.empty else None,
        "confusion_csv": str(out / "confusion_matrices.csv") if confusion_rows else None,
        "ranking_csv": str(out / "ranking_hits_at_k.csv") if ranking_rows else None,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True, help="fold_metrics.csv")
    ap.add_argument("--pairs", nargs="*", default=None, help="fold*_pairs.csv files")
    ap.add_argument("--train-scores", nargs="*", default=None, help="fold*_train_scores.csv files")
    ap.add_argument("--random-scores", nargs="*", default=None, help="fold*_random_easy_scores.csv files")
    ap.add_argument("--out-dir", required=True, help="report output directory")
    args = ap.parse_args()
    metrics = Path(args.metrics)
    pairs = [Path(p) for p in args.pairs] if args.pairs else sorted(metrics.parent.glob("*fold*_pairs.csv"))
    train_scores = (
        [Path(p) for p in args.train_scores]
        if args.train_scores is not None
        else sorted(metrics.parent.glob("*fold*_train_scores.csv"))
    )
    random_scores = (
        [Path(p) for p in args.random_scores]
        if args.random_scores is not None
        else sorted(metrics.parent.glob("*fold*_random_easy_scores.csv"))
    )
    generate_report(metrics, pairs, args.out_dir, train_scores, random_scores)


if __name__ == "__main__":
    main()
