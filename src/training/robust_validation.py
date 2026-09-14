"""Leakage-aware repeated validation and error slicing for scored pair dumps.

The training lane writes one component-aware holdout pair dump.  This module
reuses that untouched scored half to estimate split variance without fitting a
threshold on the rows it scores:

* repeated seeded folds are dealt over positive-pair components;
* SKU and GTIN endpoint identifiers stay together in every fold;
* negative pairs that cross an assigned boundary are counted and excluded;
* brand, category, and attribute slices are reported on every test fold.

The result is deliberately report-oriented: CSVs and plots are written by
``run_robust_validation`` and the returned dictionaries are JSON/W&B/MLflow
safe summaries.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_DIMENSIONS = ("brand", "category", "attribute")
METRIC_COLUMNS = (
    "repeat",
    "seed",
    "fold",
    "status",
    "n_train",
    "n_test",
    "n_positive",
    "n_negative",
    "n_straddling",
    "n_unassigned",
    "threshold",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "errors",
)
SLICE_COLUMNS = (
    "repeat",
    "seed",
    "fold",
    "dimension",
    "slice",
    "status",
    "n",
    "n_positive",
    "n_negative",
    "threshold",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "errors",
    "error_rate",
)
OPERATING_COLUMNS = (
    "repeat",
    "seed",
    "fold",
    "operating_point",
    "status",
    "n_test",
    "n_positive",
    "n_negative",
    "threshold",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "errors",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _key(value: object) -> str:
    return re.sub(r"\s+", " ", _text(value).casefold())


def _stable_seed(seed: int, repeat: int) -> int:
    digest = hashlib.sha256(f"{seed}:{repeat}".encode()).hexdigest()
    return int(digest[:8], 16)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            self.parent[value], value = root, self.parent[value]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _endpoint_keys(row: pd.Series, side: str) -> tuple[str, ...]:
    keys: list[str] = []
    for field in ("sku_id", "gtin"):
        value = _key(row.get(f"{field}_{side}", ""))
        if value and value not in {"nan", "none", "null"}:
            keys.append(f"{field}:{value}")
    # Existing pair dumps identify canonical endpoints as canon#GTIN.  This
    # fallback keeps the split GTIN-aware even when a legacy dump lacks gtin_*.
    sku = _key(row.get(f"sku_id_{side}", ""))
    if sku.startswith("canon#"):
        gtin = sku.removeprefix("canon#")
        if gtin:
            keys.append(f"gtin:{gtin}")
    return tuple(dict.fromkeys(keys))


def _fold_assignment(
    pairs: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, int], dict[int, set[str]]]:
    """Assign identifier components to folds, using positive edges only."""
    union_find = _UnionFind()
    row_keys: list[tuple[str, ...]] = []
    for _, row in pairs.iterrows():
        keys = _endpoint_keys(row, "a") + _endpoint_keys(row, "b")
        keys = tuple(dict.fromkeys(keys))
        row_keys.append(keys)
        for key in keys:
            union_find.add(key)
        if int(row["label"]) == 1 and len(keys) > 1:
            for key in keys[1:]:
                union_find.union(keys[0], key)

    components: dict[str, set[str]] = {}
    for key in union_find.parent:
        components.setdefault(union_find.find(key), set()).add(key)
    ordered = sorted(components.values(), key=lambda group: sorted(group))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    key_to_fold: dict[str, int] = {}
    fold_groups: dict[int, set[str]] = {fold: set() for fold in range(n_folds)}
    for position, component_index in enumerate(order):
        fold = position % n_folds
        for key in ordered[int(component_index)]:
            key_to_fold[key] = fold
        fold_groups[fold].update(ordered[int(component_index)])
    return key_to_fold, fold_groups


def _youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    positives = max(int(labels.sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    j = np.cumsum(ordered_labels) / positives - np.cumsum(1 - ordered_labels) / negatives
    return float(scores[order[int(np.argmax(j))]])


def _metrics(part: pd.DataFrame, threshold: float | None) -> dict[str, object]:
    if part.empty:
        return {key: None for key in METRIC_COLUMNS if key not in {"status", "n_train", "n_test"}}
    labels = part["label"].to_numpy(dtype=int)
    scores = part["score"].to_numpy(dtype=float)
    result: dict[str, object] = {
        "n": int(len(part)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "threshold": threshold,
    }
    if threshold is None or len(np.unique(labels)) < 2:
        result.update(
            status="insufficient_class_support",
            roc_auc=None,
            pr_auc=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            errors=None,
        )
        return result
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result.update(
        status="ok",
        roc_auc=float(roc_auc_score(labels, scores)),
        pr_auc=float(average_precision_score(labels, scores)),
        accuracy=float((tp + tn) / len(labels)),
        precision=float(precision),
        recall=float(recall),
        f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        errors=int(fp + fn),
    )
    return result


def _slice_values(row: pd.Series, dimension: str) -> tuple[str, ...]:
    if dimension in {"brand", "category"}:
        values = {_key(row.get(f"{dimension}_{side}", "")) for side in ("a", "b")}
        values.discard("")
        return tuple(sorted(values)) or ("<missing>",)
    value = _key(row.get("attribute_conflict_type", ""))
    if value:
        return (value,)
    conflicts = []
    for field in ("volume_conflict", "pack_conflict", "flavor_conflict"):
        raw = _key(row.get(field, ""))
        if raw in {"1", "true", "yes"}:
            conflicts.append(field.removesuffix("_conflict"))
    return ("+".join(conflicts) if conflicts else "none",)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "missing"


def _aggregate(frame: pd.DataFrame, keys: tuple[str, ...] = ()) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    numeric = [
        column for column in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "error_rate", "errors")
        if column in frame.columns
    ]
    output: dict[str, dict[str, object]] = {}
    groups = frame.groupby(list(keys), dropna=False) if keys else [("all", frame)]
    for group_key, part in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        name = "::".join(str(value) for value in group_key)
        values: dict[str, object] = {"n": int(len(part))}
        for column in numeric:
            series = pd.to_numeric(part[column], errors="coerce").dropna()
            if len(series):
                values[f"{column}_mean"] = float(series.mean())
                values[f"{column}_std"] = float(series.std(ddof=0))
        output[name] = values
    return output


def run_robust_validation(
    pairs: pd.DataFrame,
    out_dir: str | Path,
    *,
    n_folds: int = 5,
    repeats: int = 3,
    seed: int = 42,
    min_slice_size: int = 25,
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    operating_thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Run repeated component-aware validation over a scored pair frame."""
    required = {"label", "score", "sku_id_a", "sku_id_b"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"robust validation pair dump missing columns: {sorted(missing)}")
    if n_folds < 3 or repeats < 2:
        raise ValueError("robust validation requires at least 3 folds and 2 repeats")
    frame = pairs.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    if not set(frame["label"].unique()) <= {0, 1}:
        raise ValueError("robust validation labels must be binary 0/1")

    fold_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        repeat_seed = _stable_seed(seed, repeat)
        key_to_fold, _ = _fold_assignment(frame, n_folds, repeat_seed)
        assignments: list[int | None] = []
        for _, row in frame.iterrows():
            keys = _endpoint_keys(row, "a") + _endpoint_keys(row, "b")
            folds = {key_to_fold[key] for key in keys if key in key_to_fold}
            assignments.append(next(iter(folds)) if len(folds) == 1 else None)
        assignment = np.asarray(assignments, dtype=object)
        for fold in range(n_folds):
            test_mask = assignment == fold
            train_mask = np.asarray([value is not None and value != fold for value in assignment], dtype=bool)
            straddling = int(np.sum([value is None for value in assignment]))
            train = frame.loc[train_mask]
            test = frame.loc[test_mask]
            threshold = (
                _youden_threshold(train["score"].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
                if len(train) and len(np.unique(train["label"])) == 2
                else None
            )
            result = _metrics(test, threshold)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "fold": fold,
                    "status": result.get("status", "empty"),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_positive": result.get("n_positive"),
                    "n_negative": result.get("n_negative"),
                    "n_straddling": straddling,
                    "n_unassigned": straddling,
                    "threshold": result.get("threshold"),
                    "roc_auc": result.get("roc_auc"),
                    "pr_auc": result.get("pr_auc"),
                    "accuracy": result.get("accuracy"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1": result.get("f1"),
                    "errors": result.get("errors"),
                }
            )
            for operating_point, operating_threshold in (operating_thresholds or {}).items():
                operating = _metrics(test, float(operating_threshold))
                operating_rows.append(
                    {
                        "repeat": repeat,
                        "seed": repeat_seed,
                        "fold": fold,
                        "operating_point": str(operating_point),
                        "status": operating.get("status", "empty"),
                        "n_test": int(len(test)),
                        "n_positive": operating.get("n_positive"),
                        "n_negative": operating.get("n_negative"),
                        "threshold": float(operating_threshold),
                        "roc_auc": operating.get("roc_auc"),
                        "pr_auc": operating.get("pr_auc"),
                        "accuracy": operating.get("accuracy"),
                        "precision": operating.get("precision"),
                        "recall": operating.get("recall"),
                        "f1": operating.get("f1"),
                        "errors": operating.get("errors"),
                    }
                )
            if test.empty or threshold is None:
                continue
            for dimension in dimensions:
                if dimension not in DEFAULT_DIMENSIONS:
                    raise ValueError(f"unsupported robust-validation slice: {dimension}")
                memberships: dict[str, list[int]] = {}
                for index, row in test.iterrows():
                    for value in _slice_values(row, dimension):
                        memberships.setdefault(value, []).append(index)
                for slice_name, indices in sorted(memberships.items()):
                    part = test.loc[indices]
                    if len(part) < min_slice_size:
                        continue
                    result = _metrics(part, threshold)
                    slice_rows.append(
                        {
                            "repeat": repeat,
                            "seed": repeat_seed,
                            "fold": fold,
                            "dimension": dimension,
                            "slice": slice_name,
                            "status": result.get("status", "empty"),
                            "n": int(len(part)),
                            "n_positive": result.get("n_positive"),
                            "n_negative": result.get("n_negative"),
                            "threshold": result.get("threshold"),
                            "roc_auc": result.get("roc_auc"),
                            "pr_auc": result.get("pr_auc"),
                            "accuracy": result.get("accuracy"),
                            "precision": result.get("precision"),
                            "recall": result.get("recall"),
                            "f1": result.get("f1"),
                            "errors": result.get("errors"),
                            "error_rate": (
                                float(result["errors"] / len(part))
                                if result.get("errors") is not None
                                else None
                            ),
                        }
                    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows, columns=METRIC_COLUMNS)
    slice_frame = pd.DataFrame(slice_rows, columns=SLICE_COLUMNS)
    operating_frame = pd.DataFrame(operating_rows, columns=OPERATING_COLUMNS)
    fold_frame.to_csv(out / "robust_validation_fold_metrics.csv", index=False)
    slice_frame.to_csv(out / "robust_validation_slices.csv", index=False)
    operating_frame.to_csv(out / "robust_validation_operating_points.csv", index=False)

    # Small, reviewable plots: distribution of repeated-fold scores and the
    # worst sufficiently-supported slices by dimension.
    ok_folds = fold_frame[fold_frame["status"].eq("ok")]
    if not ok_folds.empty:
        metrics = [column for column in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1") if column in ok_folds]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(
            [ok_folds[column].astype(float) for column in metrics],
            tick_labels=[m.replace("_", " ") for m in metrics],
            showmeans=True,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title(f"Repeated {n_folds}-fold SKU/GTIN-aware validation ({repeats} repeats)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "robust_validation_fold_metrics.png", dpi=150)
        plt.close(fig)

    ok_operating = operating_frame[operating_frame["status"].eq("ok")]
    if not ok_operating.empty:
        metrics = [column for column in ("accuracy", "precision", "recall", "f1") if column in ok_operating]
        means = ok_operating.groupby("operating_point", sort=False)[metrics].mean()
        ax = means.plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05), color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        ax.set_xlabel("operating point")
        ax.set_ylabel("mean test score")
        ax.set_title("Repeated validation at configured operating thresholds")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower left")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out / "robust_validation_operating_points.png", dpi=150)
        plt.close(fig)

    if not slice_frame.empty:
        aggregate_slices = (
            slice_frame.groupby(["dimension", "slice"], as_index=False)
            .agg(n=("n", "sum"), error_rate_mean=("error_rate", "mean"), error_rate_std=("error_rate", "std"))
        )
        top = aggregate_slices.sort_values(["dimension", "error_rate_mean", "n"], ascending=[True, False, False]).groupby("dimension", sort=False).head(15)
        fig, axes = plt.subplots(1, len(top["dimension"].unique()) or 1, figsize=(6.5 * max(len(top["dimension"].unique()), 1), 5), squeeze=False)
        for axis, (dimension, part) in zip(axes[0], top.groupby("dimension", sort=True), strict=False):
            part = part.sort_values("error_rate_mean")
            axis.barh(part["slice"], part["error_rate_mean"], color="#c44e52")
            axis.set_xlim(0, 1)
            axis.set_xlabel("mean error rate")
            axis.set_title(f"Worst supported {dimension} slices")
            axis.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "robust_validation_slice_errors.png", dpi=150)
        plt.close(fig)
    else:
        aggregate_slices = pd.DataFrame(columns=["dimension", "slice", "n", "error_rate_mean", "error_rate_std"])

    fold_aggregate = _aggregate(ok_folds)
    fold_aggregate = fold_aggregate.get("all", {})
    operating_aggregate = _aggregate(operating_frame, ("operating_point",))
    slice_aggregate = _aggregate(slice_frame, ("dimension", "slice"))
    return {
        "config": {
            "n_folds": n_folds,
            "repeats": repeats,
            "seed": seed,
            "min_slice_size": min_slice_size,
            "dimensions": list(dimensions),
            "operating_thresholds": {
                str(key): float(value)
                for key, value in (operating_thresholds or {}).items()
            },
            "grouping": "positive-pair SKU/GTIN connected components; crossing negatives excluded and counted",
        },
        "fold_metrics": fold_frame.to_dict("records"),
        "slices": slice_frame.to_dict("records"),
        "operating_points": operating_frame.to_dict("records"),
        "aggregate": fold_aggregate,
        "operating_aggregate": operating_aggregate,
        "slice_aggregate": slice_aggregate,
        "files": [
            "robust_validation_fold_metrics.csv",
            "robust_validation_slices.csv",
            "robust_validation_operating_points.csv",
            "robust_validation_fold_metrics.png",
            "robust_validation_operating_points.png",
            "robust_validation_slice_errors.png",
        ],
    }
