"""Cheap, calibration-aware trial diagnostics for the Optuna objective.

The trial lane deliberately calls the same direct-assignment and clustering
metric functions as the final matcher.  It does not use the holdout quarter,
connected components for submission, or pairwise AUC as its objective.
"""

from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from core.graph_diagnostics import candidate_graph_diagnostics
from core.attribute_conflicts import canonical_attribute_info, conflict_columns, sku_attribute_info
from core.common import F, RESULTS, metadata_text, row_metadata_text
from core.gtin import is_valid_gtin_checksum
from core.schemas import check_canonical_records_frame
from core.structured_features import fuse_numpy
from training.rand_matching import (
    GTIN_STATUSES,
    choose_assignments,
    gtin_metrics,
    prediction_metrics,
    _threshold_at_recall,
    _youden_threshold,
)
from training.uniformity import select_unrelated_pairs


class CalibrationMetricRow(BaseModel):
    """Pydantic contract for the shared train/calibration metric payload."""

    model_config = ConfigDict(extra="forbid")

    calibration_proxy_source: str
    calibration_status: str
    calibration_reason: str | None = None
    calibration_positive_pairs: int
    calibration_negative_pairs: int
    calibration_sku_count: int
    calibration_candidate_duplicate_rows_removed: int
    calibrated_threshold: float

    calibration_rand_index: float
    calibration_adjusted_rand: float
    calibration_group_precision: float
    calibration_group_recall: float
    calibration_pairwise_precision: float
    calibration_pairwise_recall: float
    calibration_pairwise_f1: float
    calibration_over_merge_rate: float
    calibration_under_merge_rate: float
    calibration_predicted_group_count: int
    calibration_expected_group_count: int
    calibration_plausible_group_count: int
    calibration_unmatched_skus: int
    calibration_threshold_fold_median: float
    calibration_threshold_fold_min: float
    calibration_threshold_fold_max: float
    calibration_threshold_plateau_points: int
    calibration_threshold_stable: int
    calibration_youden_threshold: float
    calibration_precision_at_target_recall_threshold: float
    calibration_precision_at_threshold: float
    calibration_recall_at_threshold: float
    calibration_gtin_strata: int
    calibration_sensitivity_table: str
    diagnostic_component_size_distribution: str
    diagnostic_edge_count: int
    plausible_group_count: int
    diagnostic_component_count: int
    diagnostic_max_component_size: int
    diagnostic_score_diameter: float
    diagnostic_bridge_edge_count: int
    diagnostic_weakest_bridge_score: float
    attribute_conflict_error_rate: float
    attribute_conflict_status: str
    calibration_both_equal_rand_index: float
    calibration_both_equal_precision: float
    calibration_both_equal_recall: float
    calibration_different_rand_index: float
    calibration_different_precision: float
    calibration_different_recall: float
    calibration_one_missing_rand_index: float
    calibration_one_missing_precision: float
    calibration_one_missing_recall: float
    calibration_both_missing_rand_index: float
    calibration_both_missing_precision: float
    calibration_both_missing_recall: float
    calibration_ALL_rand_index: float
    calibration_ALL_precision: float
    calibration_ALL_recall: float
    collapse_guardrail_enabled: int | None = None
    collapse_status: str | None = None
    collapse_requested_pairs: int | None = None
    collapse_unrelated_pairs: int | None = None
    collapse_median_cosine: float | None = None
    collapse_p90_cosine: float | None = None
    collapse_cosine_std: float | None = None
    collapse_embedding_norm_mean: float | None = None
    collapse_embedding_norm_std: float | None = None
    collapse_median_flag: int | None = None
    collapse_p90_flag: int | None = None
    collapse_penalty: float | None = None


# One reporting contract is shared by ordinary train, fixed-grid HPO, and
# Optuna.  Numeric aggregate fields are derived from the Pydantic contract so
# adding a metric cannot silently omit it from fold aggregation. Optional
# collapse fields are intentionally excluded when the guardrail is disabled.
CALIBRATION_AGGREGATE_FIELDS = tuple(
    name
    for name, field in CalibrationMetricRow.model_fields.items()
    if field.annotation in (int, float)
)


def numeric_calibration_metrics(row: dict) -> dict[str, float | int]:
    """Return finite calibration diagnostics suitable for tracking APIs."""
    metrics: dict[str, float | int] = {}
    for key, value in row.items():
        if not key.startswith(
            ("calibration_", "collapse_", "diagnostic_", "attribute_conflict_")
        ):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            continue
        if np.isfinite(value):
            metrics[key] = value
    return metrics


def unavailable_calibration_metrics(
    *,
    reason: str,
    positive_pairs: int,
    negative_pairs: int,
) -> dict[str, str | int]:
    """Record an unavailable calibration surface without dropping the fold."""
    return {
        "calibration_status": "unavailable",
        "calibration_reason": reason,
        "calibration_positive_pairs": int(positive_pairs),
        "calibration_negative_pairs": int(negative_pairs),
    }


@lru_cache(maxsize=1)
def _canonical_record_map() -> dict[str, dict[str, object]]:
    records = pd.read_csv(
        RESULTS / F["canonical_records"],
        dtype={"gtin": str},
        keep_default_na=False,
    )
    check_canonical_records_frame(records)
    return {
        str(row["gtin"]): row.to_dict()
        for _, row in records.iterrows()
    }


def _trusted_gtin(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and is_valid_gtin_checksum(text) else ""


def _status(left: object, right: object) -> str:
    left_gtin = _trusted_gtin(left)
    right_gtin = _trusted_gtin(right)
    if not left_gtin and not right_gtin:
        return "both_missing"
    if not left_gtin or not right_gtin:
        return "one_missing"
    return "both_equal" if left_gtin == right_gtin else "different"


def _source_metadata(row: pd.Series) -> dict[str, str | int]:
    """Retain every source field used by matching and its presence state."""
    fields = (
        "title",
        "attributes",
        "brand",
        "country",
        "category",
        "category_path",
        "retailer",
    )
    values = {
        field: row_metadata_text(row, field)
        for field in fields
    }
    values["record_json"] = json.dumps(
        {str(key): metadata_text(value) for key, value in row.to_dict().items()},
        sort_keys=True,
    )
    return {
        **{f"sku_{field}": value for field, value in values.items()},
        **{
            f"sku_{field}_present": int(bool(value.strip()))
            for field, value in values.items()
            if field != "record_json"
        },
    }


def _canonical_metadata(record: dict[str, object]) -> dict[str, str | int]:
    """Retain the complete canonical record alongside parsed gate fields."""
    return {
        "candidate_text": metadata_text(record.get("canonical")),
        "candidate_record_json": json.dumps(
            {str(key): metadata_text(value) for key, value in record.items()},
            sort_keys=True,
        ),
    }


def _thresholds(cfg: dict) -> np.ndarray:
    matcher = cfg["rand_matching"]
    values = np.arange(
        float(matcher["threshold_min"]),
        float(matcher["threshold_max"]) + float(matcher["threshold_step"]) / 2,
        float(matcher["threshold_step"]),
    )
    return np.round(values, 10)


def _candidate_frame(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    df: pd.DataFrame,
    row_bc: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n_pos = len(pos_pairs)
    pairs = np.vstack([pos_pairs, neg_pairs])
    if len(scores) != len(pairs):
        raise ValueError("trial proxy scores are not aligned with pair rows")
    records: list[dict[str, object]] = []
    truth: dict[str, str] = {}
    truth_sources: dict[str, int] = {}
    canonical_records = _canonical_record_map()
    for pair_index, (pair, score) in enumerate(zip(pairs, scores, strict=True)):
        source, target = (int(pair[0]), int(pair[1]))
        if source < 0 or source >= len(df):
            raise ValueError(f"proxy SKU row {source} is outside dataset rows")
        if target < 0 or target >= len(row_bc):
            raise ValueError(f"proxy canonical row {target} is outside row_bc")
        sku_row = df.iloc[source]
        sku_id = row_metadata_text(sku_row, "product_id", "SKU_ID")
        candidate_gtin = str(row_bc[target]).strip()
        if not sku_id:
            raise ValueError(f"empty SKU identity in proxy source row {source}")
        if not candidate_gtin:
            raise ValueError(f"empty canonical GTIN in proxy payload row {target}")
        candidate_record = canonical_records.get(candidate_gtin)
        if candidate_record is None:
            raise ValueError(
                f"canonical GTIN {candidate_gtin!r} missing from canonical_records.csv"
            )
        sku_info = sku_attribute_info(
            row_metadata_text(sku_row, "title"),
            row_metadata_text(sku_row, "attributes", "attr"),
        )
        source_metadata = _source_metadata(sku_row)
        candidate_metadata = _canonical_metadata(candidate_record)
        candidate_info = canonical_attribute_info(candidate_record)
        rules = conflict_columns(sku_info, candidate_info)
        sku_gtin = row_metadata_text(sku_row, "barcode", "gtin")
        gtin_status = _status(sku_gtin, candidate_gtin)
        if pair_index < n_pos:
            existing = truth.get(sku_id)
            if existing is not None and existing != candidate_gtin:
                raise ValueError(f"SKU {sku_id} has conflicting proxy truth GTINs")
            truth[sku_id] = candidate_gtin
            truth_sources[sku_id] = source
        records.append(
            {
                "SKU_ID": sku_id,
                "sku_gtin": sku_gtin,
                "candidate_gtin": candidate_gtin,
                "score": float(score),
                "exact_gtin": int(gtin_status == "both_equal"),
                "gtin_status": gtin_status,
                "rule_ok": int(rules["attribute_conflict_type"] == "none"),
                "attribute_conflict_type": str(rules["attribute_conflict_type"]),
                "attribute_matches": int(
                    sum(
                        not rules[key]
                        for key in (
                            "volume_conflict",
                            "pack_conflict",
                            "flavor_conflict",
                        )
                    )
                ),
                **source_metadata,
                "sku_brand": row_metadata_text(sku_row, "brand"),
                "sku_volume": str(sorted(sku_info["volume"])),
                "sku_pack": str(sorted(sku_info["pack"])),
                "sku_flavor": str(sku_info["flavor"]),
                "sku_volume_present": int(bool(sku_info["volume"])),
                "sku_pack_present": int(bool(sku_info["pack"])),
                "sku_flavor_present": int(bool(sku_info["flavor"])),
                **candidate_metadata,
                "candidate_brand": metadata_text(candidate_record.get("mode_brand")),
                "candidate_volume": str(sorted(candidate_info["volume"])),
                "candidate_pack": str(sorted(candidate_info["pack"])),
                "candidate_flavor": str(candidate_info["flavor"]),
                "candidate_brand_present": int(
                    bool(metadata_text(candidate_record.get("mode_brand")).strip())
                ),
                "candidate_volume_present": int(bool(candidate_info["volume"])),
                "candidate_pack_present": int(bool(candidate_info["pack"])),
                "candidate_flavor_present": int(bool(candidate_info["flavor"])),
                "source_row_index": str(source),
            }
        )
    candidates = pd.DataFrame(records)
    duplicate_keys = ["SKU_ID", "candidate_gtin"]
    duplicate_mask = candidates.duplicated(duplicate_keys, keep=False)
    duplicate_count = int(candidates.duplicated(duplicate_keys, keep="first").sum())
    if duplicate_count:
        varying = (
            candidates.loc[duplicate_mask]
            .groupby(duplicate_keys, sort=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if varying.any():
            raise ValueError(
                "duplicate proxy candidates disagree on score or metadata: "
                f"{list(varying[varying].index)[:5]}"
            )
        candidates = candidates.drop_duplicates(duplicate_keys, keep="first")
    expected = set(truth)
    observed = set(candidates["SKU_ID"])
    if expected != observed:
        raise ValueError(
            "proxy candidate construction changed SKU population: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"unexpected={sorted(observed - expected)[:5]}"
        )
    truth_frame = pd.DataFrame(
        [{"SKU_ID": sku_id, "true_item_id": item_id} for sku_id, item_id in truth.items()]
    )
    statuses = [
        _status(
            row_metadata_text(df.iloc[truth_sources[sku_id]], "barcode", "gtin"),
            item_id,
        )
        for sku_id, item_id in truth.items()
    ]
    truth_frame["gtin_status"] = statuses
    return candidates, truth_frame, duplicate_count


def _score_pairs(
    model,
    payload: list[str],
    structured_features: np.ndarray,
    pairs: np.ndarray,
    structured_weight: float,
    batch_size: int,
) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty(0, dtype=float)
    rows = np.unique(pairs.ravel())
    row_to_position = {int(row): position for position, row in enumerate(rows)}
    indices = np.asarray(
        [[row_to_position[int(left)], row_to_position[int(right)]] for left, right in pairs],
        dtype=int,
    )
    embeddings = model.encode(
        [payload[int(row)] for row in rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = fuse_numpy(
        embeddings,
        np.asarray(structured_features)[rows],
        structured_weight,
    )
    return np.sum(embeddings[indices[:, 0]] * embeddings[indices[:, 1]], axis=1)


def _assignment_metrics(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    threshold: float,
    *,
    include_graph_diagnostics: bool = False,
) -> dict[str, float | int | str]:
    predicted = choose_assignments(candidates, threshold)
    return prediction_metrics(
        predicted,
        truth[["SKU_ID", "true_item_id"]],
        candidates=candidates,
        threshold=threshold,
        include_graph_diagnostics=include_graph_diagnostics,
    )


def _fit_threshold(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[float, dict[str, float | int | str]]:
    rows = [
        (float(threshold), _assignment_metrics(candidates, truth, float(threshold)))
        for threshold in thresholds
    ]
    threshold, metrics = max(
        rows,
        key=lambda item: (
            float(item[1]["rand_index"]),
            float(item[1]["adjusted_rand"]),
            -item[0],
        ),
    )
    return threshold, metrics


def _fold_ids(truth: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    items = np.asarray(sorted(truth["true_item_id"].astype(str).unique()))
    order = np.random.default_rng(seed).permutation(len(items))
    return {str(items[position]): int(index % n_folds) for index, position in enumerate(order)}


def _collapse_stats(
    model,
    df: pd.DataFrame,
    payload: list[str],
    cfg: dict,
    batch_size: int,
    *,
    requested: bool,
) -> dict[str, float | int | str]:
    guardrail = cfg["hpo"]["collapse_guardrail"]
    if not requested or not bool(guardrail["enabled"]):
        return {
            "collapse_guardrail_enabled": 0,
            "collapse_status": "not_requested" if not requested else "disabled",
            "collapse_requested_pairs": int(guardrail["unrelated_pairs"]),
            "collapse_unrelated_pairs": 0,
        }
    requested_pairs = int(guardrail["unrelated_pairs"])
    pairs = select_unrelated_pairs(
        df,
        payload,
        n_pairs=requested_pairs,
        seed=int(guardrail["seed"]),
    )
    if not pairs:
        raise RuntimeError(
            "collapse guardrail could not construct its configured unrelated-pair "
            f"sample: required={requested_pairs} selected=0"
        )
    sample_status = "ok" if len(pairs) == requested_pairs else "insufficient_pairs"
    if sample_status == "insufficient_pairs":
        print(
            "[collapse-guardrail] insufficient unrelated pairs; "
            f"requested={requested_pairs} selected={len(pairs)}",
            flush=True,
        )
    pair_array = np.asarray(pairs, dtype=int)
    embeddings = model.encode(
        [payload[int(row)] for row in pair_array.ravel()],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(float).eps)
    scores = np.sum(normalized[0::2] * normalized[1::2], axis=1)
    median = float(np.median(scores))
    p90 = float(np.quantile(scores, 0.90))
    std = float(np.std(scores))
    return {
        "collapse_guardrail_enabled": 1,
        "collapse_status": sample_status,
        "collapse_requested_pairs": requested_pairs,
        "collapse_unrelated_pairs": int(len(scores)),
        "collapse_median_cosine": median,
        "collapse_p90_cosine": p90,
        "collapse_cosine_std": std,
        "collapse_embedding_norm_mean": float(np.mean(norms)),
        "collapse_embedding_norm_std": float(np.std(norms)),
        "collapse_median_flag": int(median > float(guardrail["median_penalty_start"])),
        "collapse_p90_flag": int(p90 > float(guardrail["p90_penalty_start"])),
    }


def _collapse_penalty(stats: dict, cfg: dict) -> float:
    guardrail = cfg["hpo"]["collapse_guardrail"]
    if not bool(guardrail["enabled"]):
        return 0.0
    if stats["collapse_status"] == "not_requested":
        return 0.0
    if stats["collapse_status"] == "insufficient_pairs":
        return float(guardrail["penalty_weight"])
    median_excess = max(
        0.0,
        float(stats["collapse_median_cosine"]) - float(guardrail["median_penalty_start"]),
    )
    p90_excess = max(
        0.0,
        float(stats["collapse_p90_cosine"]) - float(guardrail["p90_penalty_start"]),
    )
    variance_excess = max(
        0.0,
        float(guardrail["cosine_std_floor"]) - float(stats["collapse_cosine_std"]),
    )
    return float(guardrail["penalty_weight"]) * (
        median_excess + p90_excess + variance_excess
    )


def evaluate_calibration_trial(
    *,
    model,
    df: pd.DataFrame,
    payload: list[str],
    structured_features: np.ndarray,
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    row_bc: np.ndarray,
    structured_weight: float,
    batch_size: int,
    config: dict,
    include_collapse_guardrail: bool,
) -> dict[str, float | int | str]:
    """Evaluate one trained model on a held-out direct-assignment calibration split."""
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        raise ValueError(
            "calibration metrics require positive and negative calibration pairs"
        )
    candidate_pairs = np.vstack([pos_pairs, neg_pairs])
    scores = _score_pairs(
        model,
        payload,
        structured_features,
        candidate_pairs,
        structured_weight,
        batch_size,
    )
    candidates, truth, duplicate_count = _candidate_frame(
        pos_pairs, neg_pairs, df, row_bc, scores
    )
    n_folds = int(config["hpo"]["calibration_folds"])
    fold_map = _fold_ids(truth, n_folds, int(config["hpo"]["collapse_guardrail"]["seed"]))
    thresholds = _thresholds(config)
    fold_rows: list[dict[str, float | int]] = []
    validation_candidates: list[pd.DataFrame] = []
    validation_truth: list[pd.DataFrame] = []
    for fold in range(n_folds):
        check_items = {item for item, value in fold_map.items() if value == fold}
        fit_items = set(fold_map) - check_items
        fit_truth = truth[truth["true_item_id"].isin(fit_items)]
        check_truth = truth[truth["true_item_id"].isin(check_items)]
        fit_candidates = candidates[candidates["SKU_ID"].isin(fit_truth["SKU_ID"])]
        check_candidates = candidates[candidates["SKU_ID"].isin(check_truth["SKU_ID"])]
        if fit_truth.empty or check_truth.empty:
            raise ValueError(f"HPO calibration fold {fold} has an empty fit/check side")
        threshold, fit_metrics = _fit_threshold(fit_candidates, fit_truth, thresholds)
        check_metrics = _assignment_metrics(check_candidates, check_truth, threshold)
        validation_candidates.append(check_candidates)
        validation_truth.append(check_truth)
        fold_rows.append(
            {
                "calibration_fold": fold,
                "calibrated_threshold": threshold,
                "fit_rand_index": float(fit_metrics["rand_index"]),
                "check_rand_index": float(check_metrics["rand_index"]),
                "check_adjusted_rand": float(check_metrics["adjusted_rand"]),
                "check_group_precision": float(check_metrics["group_precision"]),
                "check_group_recall": float(check_metrics["group_recall"]),
                "check_over_merge_rate": float(check_metrics["over_merge_rate"]),
                "check_under_merge_rate": float(check_metrics["under_merge_rate"]),
            }
        )
    final_threshold = float(np.median([row["calibrated_threshold"] for row in fold_rows]))
    validation_candidate_frame = pd.concat(validation_candidates, ignore_index=True)
    validation_truth_frame = pd.concat(validation_truth, ignore_index=True)
    overall = _assignment_metrics(
        validation_candidate_frame,
        validation_truth_frame,
        final_threshold,
        include_graph_diagnostics=True,
    )
    sensitivity = [
        (
            float(threshold),
            _assignment_metrics(
                validation_candidate_frame,
                validation_truth_frame,
                float(threshold),
            ),
        )
        for threshold in thresholds
    ]
    best_rand = max(float(row[1]["rand_index"]) for row in sensitivity)
    plateau_count = sum(
        float(row[1]["rand_index"]) >= best_rand - float(config["rand_matching"]["plateau_tolerance"])
        for row in sensitivity
    )
    collapse = _collapse_stats(
        model,
        df,
        payload,
        config,
        batch_size,
        requested=include_collapse_guardrail,
    )
    result: dict[str, float | int | str] = {
        "calibration_proxy_source": "dev_component_safe_split",
        "calibration_status": "available",
        "calibration_positive_pairs": int(len(pos_pairs)),
        "calibration_negative_pairs": int(len(neg_pairs)),
        "calibration_sku_count": int(truth["SKU_ID"].nunique()),
        "calibration_candidate_duplicate_rows_removed": duplicate_count,
        "calibrated_threshold": final_threshold,
        "calibration_threshold_fold_median": final_threshold,
        "calibration_threshold_fold_min": float(min(row["calibrated_threshold"] for row in fold_rows)),
        "calibration_threshold_fold_max": float(max(row["calibrated_threshold"] for row in fold_rows)),
        "calibration_threshold_plateau_points": int(plateau_count),
        "calibration_threshold_stable": int(
            plateau_count >= int(config["rand_matching"]["plateau_min_points"])
        ),
        "calibration_rand_index": float(overall["rand_index"]),
        "calibration_adjusted_rand": float(overall["adjusted_rand"]),
        "calibration_group_precision": float(overall["group_precision"]),
        "calibration_group_recall": float(overall["group_recall"]),
        "calibration_pairwise_precision": float(overall["pairwise_precision"]),
        "calibration_pairwise_recall": float(overall["pairwise_recall"]),
        "calibration_pairwise_f1": float(overall["pairwise_f1"]),
        "calibration_over_merge_rate": float(overall["over_merge_rate"]),
        "calibration_under_merge_rate": float(overall["under_merge_rate"]),
        "calibration_predicted_group_count": int(overall["predicted_group_count"]),
        "calibration_expected_group_count": int(overall["expected_group_count"]),
        "calibration_plausible_group_count": int(overall["plausible_group_count"]),
        "calibration_unmatched_skus": int(overall["unmatched_skus"]),
        "calibration_youden_threshold": float(
            _youden_threshold(
                candidates["score"].to_numpy(float),
                (candidates["candidate_gtin"] == candidates["SKU_ID"].map(
                    truth.set_index("SKU_ID")["true_item_id"]
                )).to_numpy(int),
            )
        ),
        "calibration_precision_at_target_recall_threshold": float(
            _threshold_at_recall(
                candidates["score"].to_numpy(float),
                (candidates["candidate_gtin"] == candidates["SKU_ID"].map(
                    truth.set_index("SKU_ID")["true_item_id"]
                )).to_numpy(int),
                float(config["rand_matching"]["target_recall"]),
            )
        ),
        "calibration_sensitivity_table": json.dumps(
            [
                {
                    "threshold": threshold,
                    "rand_index": float(metrics["rand_index"]),
                    "adjusted_rand": float(metrics["adjusted_rand"]),
                    "precision": float(metrics["pairwise_precision"]),
                    "recall": float(metrics["pairwise_recall"]),
                    "over_merge_rate": float(metrics["over_merge_rate"]),
                    "under_merge_rate": float(metrics["under_merge_rate"]),
                    "predicted_group_count": int(metrics["predicted_group_count"]),
                }
                for threshold, metrics in sensitivity
            ],
            sort_keys=True,
        ),
    }
    calibrated_metrics = overall
    result["calibration_precision_at_threshold"] = float(
        calibrated_metrics["pairwise_precision"]
    )
    result["calibration_recall_at_threshold"] = float(
        calibrated_metrics["pairwise_recall"]
    )
    result.update(collapse)
    result["collapse_penalty"] = _collapse_penalty(result, config)
    result.update(candidate_graph_diagnostics(candidates, final_threshold))
    strata_rows = gtin_metrics(
        choose_assignments(candidates, final_threshold),
        truth,
        "trial",
        final_threshold,
        candidates,
    )
    result["calibration_gtin_strata"] = len(strata_rows)
    for row in strata_rows:
        status = str(row["gtin_status"])
        result[f"calibration_{status}_rand_index"] = float(row["rand_index"])
        result[f"calibration_{status}_precision"] = float(row["pairwise_precision"])
        result[f"calibration_{status}_recall"] = float(row["pairwise_recall"])
    truth_by_sku = truth.set_index("SKU_ID")["true_item_id"]
    true_candidates = candidates.loc[
        candidates["candidate_gtin"].eq(candidates["SKU_ID"].map(truth_by_sku))
    ]
    if true_candidates.empty:
        raise RuntimeError("calibration proxy lost every true canonical candidate")
    result["attribute_conflict_error_rate"] = float(
        true_candidates["attribute_conflict_type"].ne("none").mean()
    )
    result["attribute_conflict_status"] = "true_candidate_gate_check"
    CalibrationMetricRow.model_validate(result)
    return result
