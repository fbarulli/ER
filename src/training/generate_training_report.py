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
import ast
import json
from pathlib import Path

# Keep local report generation visible and workspace-local; never fall back to
# the read-only home cache (or an implicit /tmp matplotlib cache).
from core.common import TRAIN_ROOT

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

from core.common import load_config, plot_dpi, rand_matching_cfg, recall_column_suffix

# 05-03/06-3: the recall-tied fold-metric column follows the config SSOT
# (rand_matching.target_recall) with the producer's own helper — a hardcoded
# recall suffix would silently miss the column of a retuned lane.
_RECALL_KEY = recall_column_suffix(float(rand_matching_cfg()["target_recall"]))
_RECALL_THRESHOLD_COL = f"threshold_at_{_RECALL_KEY}_recall"
REPORT_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


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


def _threshold_sweep(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute fixed-threshold confusion metrics for every scored fold."""
    columns = [
        "fold", "threshold", "tp", "fp", "fn", "tn",
        "precision", "recall", "fpr",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, int | float]] = []
    for fold, part in pairs.groupby("fold", sort=True):
        y = part["label"].to_numpy(dtype=int)
        scores = part["score"].to_numpy(dtype=float)
        for threshold in REPORT_THRESHOLDS:
            metrics = _confusion(y, scores, threshold)
            positives = int((y == 1).sum())
            negatives = int((y == 0).sum())
            rows.append(
                {
                    "fold": int(fold),
                    "threshold": float(threshold),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "fpr": float(metrics["fp"] / negatives) if negatives else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _score_overlap(negative: np.ndarray, positive: np.ndarray) -> float:
    """Histogram overlap coefficient for the two score populations."""
    if not len(negative) or not len(positive):
        return float("nan")
    bins = np.linspace(-1.0, 1.0, 101)
    neg_hist, _ = np.histogram(negative, bins=bins, density=True)
    pos_hist, _ = np.histogram(positive, bins=bins, density=True)
    return float(np.minimum(neg_hist, pos_hist).sum() * (bins[1] - bins[0]))


def _add_attribute_conflicts(
    pairs: pd.DataFrame,
    data_path: str | Path,
    canonical_path: str | Path,
) -> pd.DataFrame:
    """Attach volume/pack/flavor conflict labels to legacy pair dumps."""
    if "attribute_conflict_type" in pairs.columns:
        return pairs
    data = pd.read_csv(data_path, dtype=str).fillna("")
    canon = pd.read_csv(canonical_path, dtype=str).fillna("")
    sku_lookup = data.set_index("product_id").to_dict("index")
    canon_lookup = canon.set_index("gtin").to_dict("index")
    from core.attribute_conflicts import (
        canonical_attribute_info,
        conflict_columns,
        sku_attribute_info,
    )

    canonical_gtins = sorted(str(gtin) for gtin in canon["gtin"])
    cache: dict[str, dict[str, object]] = {}

    def endpoint_ref(row: pd.Series, side: str) -> str:
        """Resolve current IDs and legacy payload-index pair columns."""
        id_col = f"sku_id_{side}"
        if id_col in row.index:
            return str(row[id_col])
        for col in (f"payload_idx_{side}", side):
            if col not in row.index:
                continue
            value = row[col]
            try:
                index = int(float(value))
            except (TypeError, ValueError):
                return str(value)
            if index < len(data):
                return str(data.iloc[index]["product_id"])
            canon_index = index - len(data)
            if 0 <= canon_index < len(canonical_gtins):
                return f"canon#{canonical_gtins[canon_index]}"
            barcode_col = f"barcode_{side}"
            if barcode_col in row.index and str(row[barcode_col]).strip():
                return f"canon#{str(row[barcode_col]).strip()}"
            raise ValueError(
                f"legacy pair endpoint index {index} is outside the payload "
                f"and has no {barcode_col} fallback"
            )
        raise ValueError(
            f"pair dump lacks an endpoint column for side {side!r}; "
            f"columns={list(row.index)}"
        )

    def attrs(ref: object) -> dict[str, object]:
        key = str(ref)
        if key in cache:
            return cache[key]
        if key.startswith("canon#"):
            gtin = key.removeprefix("canon#")
            record = canon_lookup.get(gtin)
            if record is None:
                raise KeyError(f"pair endpoint references unknown canonical {gtin!r}")
            info = canonical_attribute_info(record)
        else:
            record = sku_lookup.get(key)
            if record is None:
                raise KeyError(f"pair endpoint references unknown SKU {key!r}")
            info = sku_attribute_info(record.get("title", ""), record.get("attributes", ""))
        cache[key] = info
        return info

    def conflict(row: pd.Series) -> dict[str, object]:
        left = attrs(endpoint_ref(row, "a"))
        right = attrs(endpoint_ref(row, "b"))
        return conflict_columns(left, right)

    labels = pd.DataFrame([conflict(row) for _, row in pairs.iterrows()])
    return pd.concat([pairs.reset_index(drop=True), labels], axis=1)


def _add_pair_metadata(
    pairs: pd.DataFrame,
    data_path: str | Path,
    canonical_path: str | Path,
) -> pd.DataFrame:
    """Attach endpoint GTIN/brand/category fields for robust slicing."""
    data = pd.read_csv(data_path, dtype=str, keep_default_na=False).fillna("")
    canon = pd.read_csv(canonical_path, dtype=str, keep_default_na=False).fillna("")
    sku_lookup = data.set_index("product_id").to_dict("index")
    canon_lookup = canon.set_index("gtin").to_dict("index")
    canonical_gtins = sorted(str(gtin) for gtin in canon["gtin"])

    def endpoint_ref(row: pd.Series, side: str) -> str:
        id_col = f"sku_id_{side}"
        if id_col in row.index and _text(row[id_col]):
            return _text(row[id_col])
        for col in (f"payload_idx_{side}", side):
            if col not in row.index:
                continue
            value = row[col]
            try:
                index = int(float(value))
            except (TypeError, ValueError):
                return _text(value)
            if index < len(data):
                return str(data.iloc[index]["product_id"])
            canon_index = index - len(data)
            if 0 <= canon_index < len(canonical_gtins):
                return f"canon#{canonical_gtins[canon_index]}"
            barcode_col = f"barcode_{side}"
            if barcode_col in row.index and _text(row[barcode_col]):
                return f"canon#{_text(row[barcode_col])}"
        raise ValueError(
            f"pair dump lacks a resolvable endpoint for side {side!r}; "
            f"columns={list(row.index)}"
        )

    def metadata(ref: str) -> dict[str, str]:
        if ref.startswith("masked#"):
            # Older pair dumps exposed appended masked-copy payload entries as
            # ``masked#<payload-index>`` but did not persist their source SKU
            # or masking audit. Preserve the report and identify the endpoint
            # honestly; do not guess a title/GTIN from the payload index.
            return {
                "gtin": "",
                "brand": "",
                "category": "",
                "title": f"[{ref}: source metadata unavailable]",
                "attributes": "",
                "retailer": "",
                "country": "",
                "canonical": "",
            }
        if ref.startswith("canon#"):
            gtin = ref.removeprefix("canon#")
            record = canon_lookup.get(gtin)
            if record is None:
                raise KeyError(f"unknown canonical endpoint {gtin!r}")
            return {
                "gtin": gtin,
                "brand": _text(record.get("mode_brand", record.get("brand", ""))),
                "category": _text(record.get("mode_type", record.get("category", ""))),
                "title": _text(record.get("canonical", "")),
                "attributes": json.dumps(
                    {
                        key: _text(record.get(key, ""))
                        for key in (
                            "mode_flavor", "volume_set", "pack_set",
                            "package_type_set", "package_material_set",
                        )
                        if _text(record.get(key, ""))
                    },
                    sort_keys=True,
                ),
                "retailer": "",
                "country": "",
                "canonical": _text(record.get("canonical", "")),
            }
        record = sku_lookup.get(ref)
        if record is None:
            raise KeyError(f"unknown SKU endpoint {ref!r}")
        return {
            "gtin": _text(record.get("barcode", "")),
            "brand": _text(record.get("brand", "")),
            "category": _text(record.get("category", record.get("category_path", ""))),
            "title": _text(record.get("title", "")),
            "attributes": _text(record.get("attributes", "")),
            "retailer": _text(record.get("retailer", "")),
            "country": _text(record.get("country", "")),
            "canonical": "",
        }

    additions: list[dict[str, str]] = []
    for _, row in pairs.iterrows():
        additions.append(
            {
                **{f"{key}_a": value for key, value in metadata(endpoint_ref(row, "a")).items()},
                **{f"{key}_b": value for key, value in metadata(endpoint_ref(row, "b")).items()},
            }
        )
    enriched = pd.DataFrame(additions, index=pairs.index)
    result = pairs.copy()
    for column in enriched.columns:
        if column not in result.columns:
            result[column] = enriched[column]
        else:
            current = result[column].map(str).replace({"nan": "", "None": ""})
            result[column] = current.where(current.str.strip().ne(""), enriched[column])
    return result


def _run_robust_validation(
    pairs: pd.DataFrame,
    out: Path,
) -> dict[str, object] | None:
    if pairs.empty:
        return None
    cfg = load_config()["evaluation"]["robust_validation"]
    if not bool(cfg["enabled"]):
        return None
    from training.robust_validation import run_robust_validation

    return run_robust_validation(
        pairs,
        out,
        n_folds=int(cfg["n_folds"]),
        repeats=int(cfg["repeats"]),
        seed=int(cfg["seed"]),
        min_slice_size=int(cfg["min_slice_size"]),
        min_test_negatives=int(cfg["min_test_negatives"]),
        max_split_attempts=int(cfg["max_split_attempts"]),
        dimensions=tuple(str(value) for value in cfg["dimensions"]),
        operating_thresholds={
            str(key): float(value)
            for key, value in cfg["operating_thresholds"].items()
        },
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=plot_dpi(), bbox_inches="tight")
    plt.close(fig)
    print(f"[report] {path}", flush=True)


def _finite_number(value: object) -> int | float | None:
    """Return a JSON-safe numeric scalar, omitting NaN/inf values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _numeric_row(row: pd.Series, *, exclude: set[str] = frozenset()) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in row.items():
        if key in exclude:
            continue
        number = _finite_number(value)
        if number is not None:
            result[str(key)] = number
    return result


def _numeric_tree(value):
    if isinstance(value, dict):
        return {
            str(key): child
            for key, raw in value.items()
            if (child := _numeric_tree(raw)) is not None
        }
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value) if np.isfinite(value) else None
    return None


def _report_safe_tree(value):
    """Convert metric payloads to JSON-safe scalars without losing numbers."""
    if isinstance(value, dict):
        return {
            str(key): _report_safe_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_report_safe_tree(child) for child in value]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _finite_number(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _parse_metric_payload(value: object) -> object:
    """Parse JSON/Python-literal metric payloads emitted by fold CSVs."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return value


def _rand_metric_tree(row: pd.Series) -> dict[str, object]:
    """Retain every Rand/calibration number, including nested fold tables."""
    prefixes = (
        "calibration_",
        "collapse_",
        "diagnostic_",
        "attribute_conflict_",
    )
    exact = {"calibrated_threshold", "plausible_group_count"}
    return {
        str(key): _report_safe_tree(_parse_metric_payload(value))
        for key, value in row.items()
        if str(key).startswith(prefixes) or str(key) in exact
    }


def _rand_metric_aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, int | float]]:
    """Aggregate scalar Rand fields while preserving the per-fold values."""
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value is None or not np.isfinite(float(value)):
                continue
            numeric.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": _finite_number(np.mean(values)),
            "std": _finite_number(np.std(values)),
            "n": len(values),
        }
        for key, values in sorted(numeric.items())
    }


def generate_report(
    metrics_path: str | Path,
    pair_paths: list[str | Path],
    out_dir: str | Path,
    train_score_paths: list[str | Path] | None = None,
    random_score_paths: list[str | Path] | None = None,
    data_path: str | Path | None = None,
    canonical_path: str | Path | None = None,
    uniformity_summary: dict | None = None,
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
    if (
        not pairs.empty
        and "attribute_conflict_type" not in pairs.columns
        and data_path is not None
        and canonical_path is not None
    ):
        pairs = _add_attribute_conflicts(pairs, data_path, canonical_path)
    if not pairs.empty and data_path is not None and canonical_path is not None:
        pairs = _add_pair_metadata(pairs, data_path, canonical_path)
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
    robust_validation = _run_robust_validation(pairs, out)

    # Fixed operating-point sweep requested for model review.  This uses the
    # scored holdout rows only; thresholds are never selected from this table.
    threshold_sweep = _threshold_sweep(pairs)
    threshold_sweep.to_csv(out / "threshold_sweep.csv", index=False)
    if not threshold_sweep.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for metric, color in (
            ("precision", "#4c72b0"),
            ("recall", "#55a868"),
            ("fpr", "#c44e52"),
        ):
            grouped = threshold_sweep.groupby("threshold", sort=True)[metric]
            ax.plot(
                grouped.mean().index,
                grouped.mean().values,
                marker="o",
                label=metric,
                color=color,
            )
        ax.set(
            xlabel="decision threshold",
            ylabel="rate",
            title="Holdout precision / recall / false-positive rate sweep",
            ylim=(0, 1.05),
        )
        ax.set_xticks(list(REPORT_THRESHOLDS))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "threshold_sweep.png")

        # AUC is threshold-independent, so show it as the discrimination
        # baseline while plotting the threshold-dependent operating points.
        # This avoids the misleading implication that lowering a threshold
        # can improve ROC AUC; it only trades precision/recall/FPR.
        auc_values = []
        for _, part in pairs.groupby("fold", sort=True):
            if part["label"].nunique() > 1:
                auc_values.append(
                    roc_auc_score(part["label"].to_numpy(), part["score"].to_numpy())
                )
        mean_auc = float(np.mean(auc_values)) if auc_values else float("nan")
        tuning = threshold_sweep.groupby("threshold", sort=True).mean(numeric_only=True)
        tuning["f1"] = (
            2 * tuning["precision"] * tuning["recall"]
            / (tuning["precision"] + tuning["recall"]).replace(0, np.nan)
        ).fillna(0.0)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for metric, color in (
            ("precision", "#4c72b0"),
            ("recall", "#55a868"),
            ("f1", "#8172b2"),
            ("fpr", "#c44e52"),
        ):
            ax.plot(tuning.index, tuning[metric], marker="o", label=metric, color=color)
        ax.set(
            xlabel="decision threshold",
            ylabel="rate",
            title=f"ANN threshold tuning (mean ROC AUC={mean_auc:.3f})",
            ylim=(0, 1.05),
        )
        ax.set_xticks(list(REPORT_THRESHOLDS))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "auc_threshold_tuning.png")

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

    # Ranking quality over training. AP is the evaluator's PR/ranking signal;
    # AUC, precision and recall are plotted when the evaluator emitted them.
    fig, axes = plt.subplots(
        1, len(ok), figsize=(4.5 * len(ok), 3.8), squeeze=False
    )
    for i, (_, row) in enumerate(ok.iterrows()):
        ax = axes[0, i]
        epochs = _json_list(row.get("dev_metric_epoch_hist"))
        ap = _json_list(row.get("dev_ap_hist"))
        auc = _json_list(row.get("dev_auc_hist"))
        precision = _json_list(row.get("dev_precision_hist"))
        recall = _json_list(row.get("dev_recall_hist"))
        if not epochs:
            epochs = list(range(1, max(len(ap), len(auc), len(precision), len(recall)) + 1))
        for values, label, color in (
            (ap, "dev AP / PR", "#4c72b0"),
            (auc, "dev ROC AUC", "#55a868"),
            (precision, "dev precision", "#c44e52"),
            (recall, "dev recall", "#8172b2"),
        ):
            if values:
                x = epochs[: len(values)] if len(epochs) >= len(values) else list(range(1, len(values) + 1))
                ax.plot(x, values, marker="o", label=label, color=color)
        ax.set_title(f"fold {int(row['fold'])}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Ranking quality over training")
    fig.tight_layout()
    _save(fig, out / "ranking_quality_by_epoch.png")

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
            # Retrieval ranking is per source SKU/query.  Ranking the whole
            # pair table globally makes top-k recall meaningless when there
            # are thousands of positives and only k rows are inspected.
            source_column = "sku_id_a" if "sku_id_a" in part.columns else "source_sku_id"
            if source_column not in part.columns:
                raise ValueError(
                    "pair dump lacks a source SKU column required for retrieval metrics"
                )
            query_groups = [
                group.sort_values("score", ascending=False, kind="stable")
                for _, group in part.groupby(source_column, sort=False)
                if bool((group["label"].astype(int) == 1).any())
            ]
            for k in (1, 5, 10):
                query_hits = []
                query_precisions = []
                for group in query_groups:
                    top = group.head(k)["label"].to_numpy(dtype=int)
                    query_hits.append(int(top.sum() > 0))
                    query_precisions.append(float(top.mean()) if len(top) else 0.0)
                n_queries = len(query_groups)
                ranking_rows.append(
                    {
                        "fold": fold,
                        "k": k,
                        "hits_at_k": int(sum(query_hits)),
                        "available_k": int(n_queries),
                        "queries": int(n_queries),
                        "precision_at_k": float(np.mean(query_precisions)) if query_precisions else 0.0,
                        "recall_at_k": float(sum(query_hits) / max(n_queries, 1)),
                    }
                )
            thresholds = {"dev_youden": float(row["youden_thr"])}
            # The configured fixed operating threshold is encoded in the
            # f1/precision/recall column suffix, not the recall-target
            # threshold.
            if f1_col and "_at_" in f1_col:
                thresholds["fixed"] = float(f1_col.rsplit("_at_", 1)[1])
            elif _RECALL_THRESHOLD_COL in row and pd.notna(
                row[_RECALL_THRESHOLD_COL]
            ):
                thresholds["fixed"] = float(row[_RECALL_THRESHOLD_COL])
            else:
                raise ValueError(
                    f"fold {fold}: no fixed operating threshold is present; "
                    "report generation refuses an implicit 0.55 fallback"
                )
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
        for ax, (split_name, frame) in zip(axes, distributions, strict=True):
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

        # Attribute-conflict error breakdown. Pair exports carry the parser's
        # volume/pack/flavor conflict type, so this is computed on exactly the
        # same raw-cosine population and threshold as the holdout metrics.
        if "attribute_conflict_type" in pairs.columns:
            attr = pairs.copy()
            threshold_by_fold = ok.set_index("fold")["youden_thr"].astype(float)
            attr["threshold"] = attr["fold"].map(threshold_by_fold)
            attr["error"] = (
                ((attr["label"] == 1) & (attr["score"] < attr["threshold"]))
                | ((attr["label"] == 0) & (attr["score"] >= attr["threshold"]))
            )

            def _attr_bucket(value: object) -> str:
                parts = sorted({part for part in str(value).split("+") if part and part != "none"})
                return "multiple" if len(parts) > 1 else (parts[0] if parts else "none")

            attr["attribute_bucket"] = attr["attribute_conflict_type"].map(_attr_bucket)
            breakdown = (
                attr.groupby(["attribute_bucket", "label"], dropna=False)
                .agg(n=("error", "size"), errors=("error", "sum"), mean_score=("score", "mean"))
                .reset_index()
            )
            breakdown["error_rate"] = breakdown["errors"] / breakdown["n"]
            breakdown.to_csv(out / "attribute_error_breakdown.csv", index=False)
            plot_data = breakdown.pivot(index="attribute_bucket", columns="label", values="error_rate").fillna(0)
            plot_data = plot_data.rename(columns={0: "label 0 error rate", 1: "label 1 error rate"})
            fig, ax = plt.subplots(figsize=(8, 4.8))
            plot_data.plot(kind="bar", ax=ax, color=["#c44e52", "#4c72b0"])
            ax.set_xlabel("attribute conflict type")
            ax.set_ylabel("error rate at DEV-fit Youden threshold")
            ax.set_title("Holdout error breakdown by volume / pack / flavor conflict")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(title="population")
            fig.tight_layout()
            _save(fig, out / "attribute_error_breakdown.png")

            # Concrete examples make the aggregate slice actionable. Keep a
            # deterministic, bounded sample of the most confident mistakes
            # in each fold/attribute/error direction; this avoids hiding the
            # examples behind a random sample or an oversized report.
            attr["error_kind"] = np.select(
                [
                    attr["error"] & attr["label"].eq(0),
                    attr["error"] & attr["label"].eq(1),
                ],
                ["false_positive", "false_negative"],
                default="correct",
            )
            attr["error_margin"] = (attr["score"] - attr["threshold"]).abs()
            examples = attr[attr["error"]].copy()
            if not examples.empty:
                sort_columns = [
                    column
                    for column in (
                        "fold", "attribute_bucket", "error_kind", "error_margin",
                        "score", "sku_id_a", "sku_id_b",
                    )
                    if column in examples.columns
                ]
                examples = examples.sort_values(
                    sort_columns,
                    ascending=[
                        column not in {"error_margin", "score"}
                        for column in sort_columns
                    ],
                    kind="mergesort",
                )
                examples = (
                    examples.groupby(
                        ["fold", "attribute_bucket", "error_kind"],
                        sort=True,
                        group_keys=False,
                    )
                    .head(50)
                    .reset_index(drop=True)
                )
            example_columns = [
                "fold", "attribute_bucket", "attribute_conflict_type",
                "error_kind", "label", "score", "threshold", "error_margin",
                "sku_id_a", "sku_id_b", "gtin_a", "gtin_b",
                "brand_a", "brand_b", "category_a", "category_b",
                "retailer_a", "retailer_b", "country_a", "country_b",
                "title_a", "title_b", "attributes_a", "attributes_b",
            ]
            examples[[column for column in example_columns if column in examples]].to_csv(
                out / "attribute_error_examples.csv", index=False
            )

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
        for ax, (name, positive, negative) in zip(axes, panels, strict=True):
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

    # Fine-tuning efficiency view: one point per fold, with labels retained
    # even when only one fold exists. encode_s is the post-training embedding
    # cost used by the reported holdout evaluation.
    if {"auc", "encode_s"}.issubset(ok.columns):
        pareto = ok[["fold", "auc", "encode_s"]].dropna()
        if not pareto.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.scatter(pareto["encode_s"], pareto["auc"], s=70, color="#4c72b0")
            for _, row in pareto.iterrows():
                ax.annotate(f"fold {int(row['fold'])}", (row["encode_s"], row["auc"]), xytext=(5, 5), textcoords="offset points")
            ax.set_xlabel("holdout encode time (s)")
            ax.set_ylabel("holdout ROC AUC")
            ax.set_ylim(0, 1.05)
            ax.set_title("Accuracy / latency trade-off")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            _save(fig, out / "auc_vs_encode_time.png")

    # report.json is a metric summary. Artifact locations belong to the
    # filesystem/W&B manifest, not the metric contract consumed downstream.
    numeric_metrics: dict[str, dict] = {}
    rand_metric_folds: dict[str, dict[str, object]] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    history_columns = {
        "train_loss_hist": "train_loss",
        "train_epoch_hist": "train_epoch",
        "dev_ap_hist": "dev_ap",
        "dev_auc_hist": "dev_auc",
        "dev_precision_hist": "dev_precision",
        "dev_recall_hist": "dev_recall",
        "dev_loss_hist": "dev_loss",
        "dev_epoch_hist": "dev_epoch",
        "dev_metric_epoch_hist": "dev_metric_epoch",
    }
    for _, metric_row in ok.iterrows():
        fold_key = f"fold_{int(metric_row['fold'])}"
        numeric_metrics[fold_key] = _numeric_row(
            metric_row,
            exclude={"status", "model", "payload", *history_columns},
        )
        rand_metric_folds[fold_key] = _rand_metric_tree(metric_row)
        histories[fold_key] = {
            target: [
                number
                for value in _json_list(metric_row.get(source))
                if (number := _finite_number(value)) is not None
            ]
            for source, target in history_columns.items()
        }

    def _numeric_csv(path: Path, key_columns: tuple[str, ...]) -> dict[str, dict]:
        if not path.is_file():
            return {}
        frame = pd.read_csv(path)
        output: dict[str, dict] = {}
        for _, row in frame.iterrows():
            key = "/".join(str(row[column]) for column in key_columns if column in row)
            if not key:
                key = str(len(output))
            output[key] = _numeric_row(row, exclude=set(key_columns))
        return output

    report = {
        "report_version": 3,
        "folds": int(len(ok)),
        "metrics": numeric_metrics,
        "histories": histories,
        "aggregate": {
            str(key): {
                str(stat): _finite_number(value)
                for stat, value in values.items()
                if _finite_number(value) is not None
            }
            for key, values in aggregate.items()
        },
        "score_overlap": _numeric_csv(out / "score_distribution_overlap.csv", ("split",)),
        "confusion": _numeric_csv(out / "confusion_matrices.csv", ("fold", "operating_point")),
        "threshold_sweep": _numeric_csv(out / "threshold_sweep.csv", ("fold", "threshold")),
        "ranking": _numeric_csv(out / "ranking_hits_at_k.csv", ("fold", "k")),
        "random_easy": _numeric_csv(out / "random_easy_metrics.csv", ("population", "label")),
        "attribute_errors": _numeric_csv(out / "attribute_error_breakdown.csv", ("attribute_bucket", "label")),
        "attribute_error_examples": {
            "rows": int(
                len(pd.read_csv(out / "attribute_error_examples.csv"))
            )
            if (out / "attribute_error_examples.csv").is_file()
            else 0,
            "artifact": "attribute_error_examples.csv",
        },
        "robust_validation": robust_validation or {},
        "uniformity": _numeric_tree(uniformity_summary or {}),
        "rand_matching": {
            "folds": rand_metric_folds,
            "aggregate": _rand_metric_aggregate(
                [
                    {
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                    for metrics in rand_metric_folds.values()
                ]
            ),
        },
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
    ap.add_argument("--data", default=None, help="dataset_deduped.csv for legacy pair attribute enrichment")
    ap.add_argument("--canonicals", default=None, help="canonical_records.csv for legacy pair attribute enrichment")
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
    generate_report(
        metrics,
        pairs,
        args.out_dir,
        train_scores,
        random_scores,
        args.data,
        args.canonicals,
    )


if __name__ == "__main__":
    main()
