"""Cheap, calibration-aware trial diagnostics for the Optuna objective.

The trial lane deliberately calls the same direct-assignment and clustering
metric functions as the final matcher.  It does not use the holdout quarter,
connected components for submission, or pairwise AUC as its objective.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.attribute_conflicts import sku_attribute_info
from core.common import canonical_records_frame, row_metadata_text
from core.graph_diagnostics import candidate_graph_diagnostics
from core.structured_features import fuse_numpy
from training.rand_matching import (
    GTIN_STATUSES,
    METRIC_COLUMNS,
    candidate_gate_fields,
    choose_assignments,
    gtin_status,
    gtin_metrics,
    _fit_fold_threshold,
    _candidate_labels,
    prediction_metrics,
    _threshold_at_recall,
    _youden_threshold,
)
from training.uniformity import collapse_diagnostics


class CalibrationUnavailableReasonCode(str, Enum):
    """Stable machine-readable reasons for an unavailable calibration surface."""

    EMPTY_SPLIT = "empty_calibration_split"
    EVALUATOR_FAILED = "calibration_evaluator_failed"


CALIBRATION_UNAVAILABLE_REASON_CODES: dict[CalibrationUnavailableReasonCode, int] = {
    CalibrationUnavailableReasonCode.EMPTY_SPLIT: 1,
    CalibrationUnavailableReasonCode.EVALUATOR_FAILED: 2,
}
CALIBRATION_UNAVAILABLE_REASON_UNCLASSIFIED = 0

# Compatibility aliases for callers that use the owned reason constants.  The
# enum above is the single source of the string values.
CALIBRATION_REASON_EMPTY_SPLIT = CalibrationUnavailableReasonCode.EMPTY_SPLIT.value
CALIBRATION_REASON_EVALUATOR_FAILED = (
    CalibrationUnavailableReasonCode.EVALUATOR_FAILED.value
)


class CalibrationMetricRow(BaseModel):
    """Pydantic contract for the shared train/calibration metric payload."""

    model_config = ConfigDict(extra="forbid")

    calibration_proxy_source: str
    calibration_status: str
    calibration_reason: str | None = None
    calibration_reason_code: CalibrationUnavailableReasonCode | None = None
    calibration_positive_pairs: int
    calibration_negative_pairs: int
    calibration_sku_count: int
    calibration_candidate_duplicate_rows_removed: int = Field(ge=0, default=0)
    calibration_candidate_duplicate_rows: int = Field(ge=0, default=0)
    calibration_candidate_duplicate_groups: int = Field(ge=0, default=0)
    calibration_candidate_duplicate_reason: str = Field(
        min_length=1,
        default="none",
    )
    calibration_candidate_duplicate_policy: str = Field(
        min_length=1,
        default="preserve_all_candidate_rows; assignment_selects_best_per_sku",
    )
    calibrated_threshold: float
    calibration_threshold_tie_break: list[str] | None = None
    calibration_reconciliation_scope: str
    calibration_threshold_fold_count: int
    calibration_threshold_support_count: int
    calibration_threshold_support_minimum: int
    calibration_threshold_support_sufficient: int

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
    calibration_threshold_fold_min: float
    calibration_threshold_fold_max: float
    calibration_threshold_plateau_points: int
    calibration_threshold_stable: int
    calibration_youden_threshold: float
    calibration_precision_at_target_recall_threshold: float
    calibration_precision_at_threshold: float
    calibration_recall_at_threshold: float
    calibration_gtin_strata: int
    calibration_sensitivity_table: list["CalibrationSensitivityRow"]
    calibration_sensitivity_by_gtin_status: list["CalibrationSensitivityRow"]
    calibration_fold_collapse: list["CalibrationFoldMetricRow"]
    diagnostic_component_size_distribution: dict[str, int]
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
    calibration_non_finite_count: int
    calibration_non_finite_fields: str
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
    collapse_operating_threshold: float | None = None
    collapse_crossing_rate: float | None = None
    collapse_crossing_rate_ceiling: float | None = None
    collapse_crossing_rate_flag: int | None = None
    collapse_diagnostics_available: int | None = None
    collapse_healthy: int | None = None
    collapse_penalty: float | None = None

    @field_validator("calibration_candidate_duplicate_reason")
    @classmethod
    def _duplicate_reason_is_explicit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("duplicate candidate reason must be explicit")
        return value

class CalibrationFoldMetricRow(BaseModel):
    """Strict contract for one fit/check calibration fold."""

    model_config = ConfigDict(extra="forbid")

    calibration_fold: int
    calibrated_threshold: float
    reconciliation_scope: str
    fit_rand_index: float
    check_rand_index: float
    check_adjusted_rand: float
    check_group_precision: float
    check_group_recall: float
    check_over_merge_rate: float
    check_under_merge_rate: float
    collapse_status: Literal[
        "not_requested", "disabled", "unavailable", "ok", "insufficient_pairs"
    ]
    collapse_diagnostics_available: int = Field(ge=0, le=1)
    collapse_healthy: int = Field(ge=0, le=1)
    collapse_median_cosine: float | None = None
    collapse_p90_cosine: float | None = None
    collapse_cosine_std: float | None = None
    collapse_operating_threshold: float | None = None
    collapse_crossing_rate: float | None = None
    collapse_crossing_rate_ceiling: float | None = None
    collapse_crossing_rate_flag: int | None = None
    collapse_penalty: float = Field(ge=0.0)
    fit_rand_index_minus_collapse_penalty: float


class CalibrationSensitivityRow(BaseModel):
    """Strict contract for one held-out threshold/status sensitivity point."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    gtin_status: str = Field(min_length=1)
    reconciliation_scope: str
    rand_index: float
    adjusted_rand: float
    precision: float
    recall: float
    over_merge_rate: float
    under_merge_rate: float
    predicted_group_count: int

    @field_validator("gtin_status")
    @classmethod
    def _known_gtin_status(cls, value: str) -> str:
        if value != "ALL" and value not in GTIN_STATUSES:
            raise ValueError(
                f"unknown GTIN sensitivity status {value!r}; "
                f"expected ALL or {GTIN_STATUSES}"
            )
        return value


CalibrationMetricRow.model_rebuild()


# One reporting contract is shared by ordinary train, fixed-grid HPO, and
# Optuna.  Numeric aggregate fields are derived from the Pydantic contract so
# adding a metric cannot silently omit it from fold aggregation. Optional
# collapse fields are intentionally excluded when the guardrail is disabled.
CALIBRATION_AGGREGATE_FIELDS = tuple(
    name
    for name, field in CalibrationMetricRow.model_fields.items()
    if field.annotation in (int, float)
)


# The metric keys a calibration row owns. The artifact row (written to the
# fold-metrics CSV) and the tracking lane (MLflow/W&B) must agree on which
# keys they scan for non-finite values: they previously spelled this set twice
# with different prefixes under the single name
# `calibration_non_finite_count`, so one artifact could call a row dirty while
# the other called it clean.
CALIBRATION_METRIC_PREFIXES = (
    "calibration_",
    "collapse_",
    "diagnostic_",
    "attribute_conflict_",
)


def calibration_non_finite_fields(row: dict) -> list[str]:
    """Return the sorted numeric calibration metric keys that are not finite.

    Single definition of the non-finite rule, shared by the artifact row and
    the tracking lane (see CALIBRATION_METRIC_PREFIXES).
    """
    return sorted(
        key
        for key, value in row.items()
        if key.startswith(CALIBRATION_METRIC_PREFIXES)
        and not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and not np.isfinite(value)
    )


# 06-4: the artifact row names WHICH calibration metrics were not finite, but
# that list is a `str`, so the numeric filter drops it and the tracked count
# has no cause beside it.  Each named field is re-emitted as a numeric flag so
# count and cause travel together through MLflow/W&B.  A row that names no
# field adds no key.
NON_FINITE_FIELD_METRIC_PREFIX = "calibration_non_finite/"

# 03-3: a fold whose calibration never ran carries no MEASURED calibration
# metric — no calibration_non_finite_count and no aggregate diagnostic — so
# "never measured" was indistinguishable from "measured clean" in the tracking
# lane.  (Its payload does carry the two SIZE counts,
# calibration_positive_pairs/calibration_negative_pairs, which report how big
# the calibration split was, not what it scored.)  These numeric codes are how
# such a row says so; the producers pass one of the stable reason codes below.
# The human-readable reason remains available for diagnosis, but no numeric
# tracking field depends on parsing its wording.
def _unavailable_reason_code(row: dict) -> int:
    """Map a typed reason code to its stable numeric tracking code."""
    reason_code = row.get("calibration_reason_code")
    try:
        return CALIBRATION_UNAVAILABLE_REASON_CODES.get(
            CalibrationUnavailableReasonCode(reason_code),
            CALIBRATION_UNAVAILABLE_REASON_UNCLASSIFIED,
        )
    except (TypeError, ValueError):
        pass
    return CALIBRATION_UNAVAILABLE_REASON_UNCLASSIFIED


def numeric_calibration_metrics(row: dict) -> dict[str, float | int]:
    """Return finite calibration diagnostics suitable for tracking APIs.

    The non-finite count is NOT re-derived here.  The artifact row already
    carries the value measured by ``calibration_non_finite_fields`` over the
    one shared prefix set, and the loop below forwards it, so the tracked
    number and the fold-metrics CSV number are the same measurement.  A row
    without that measurement — a fold whose calibration never ran — therefore
    reports no count at all instead of a clean-looking 0.

    Two things travel with the count so the tracking lane can tell the three
    states apart without the CSV (06-4, 03-3):

    * ``calibration_non_finite/<field>`` = 1 for every field the row named, so
      a non-zero count can be resolved to the metrics that caused it;
    * ``calibration_available`` = 0 plus ``calibration_unavailable_reason_code``
      for a row that carries no measured count at all, i.e. a fold whose
      calibration surface does not exist.  Available rows are unchanged: they
      are identified by carrying the count, so no key is added for them.
    """
    metrics: dict[str, float | int] = {}
    for key, value in row.items():
        if not key.startswith(CALIBRATION_METRIC_PREFIXES):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            continue
        if np.isfinite(value):
            metrics[key] = value
    recorded = row.get("calibration_non_finite_fields")
    for field in recorded.split(",") if isinstance(recorded, str) else ():
        if field:
            metrics[f"{NON_FINITE_FIELD_METRIC_PREFIX}{field}"] = 1
    if "calibration_non_finite_count" not in metrics:
        metrics["calibration_available"] = 0
        metrics["calibration_unavailable_reason_code"] = _unavailable_reason_code(row)
    return metrics


def unavailable_calibration_metrics(
    *,
    reason_code: CalibrationUnavailableReasonCode | str,
    reason: str,
    positive_pairs: int,
    negative_pairs: int,
) -> dict[str, str | int]:
    """Record an unavailable calibration surface without dropping the fold."""
    try:
        normalized_code = CalibrationUnavailableReasonCode(reason_code)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown calibration unavailable reason code: {reason_code!r}") from exc
    if normalized_code not in CALIBRATION_UNAVAILABLE_REASON_CODES:
        raise ValueError(f"unknown calibration unavailable reason code: {reason_code!r}")
    return {
        "calibration_status": "unavailable",
        "calibration_reason_code": normalized_code,
        "calibration_reason": reason,
        "calibration_positive_pairs": int(positive_pairs),
        "calibration_negative_pairs": int(negative_pairs),
    }


def _canonical_record_map() -> dict[str, dict[str, object]]:
    records = canonical_records_frame()
    return {
        str(row["gtin"]): row.to_dict()
        for _, row in records.iterrows()
    }


def _json_safe(value: object) -> object:
    """Convert source/canonical metadata to lossless JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _metadata_json(values: dict[object, object]) -> str:
    """Serialize complete row metadata without coercing every value to text."""
    return json.dumps(
        _json_safe(values),
        sort_keys=True,
        allow_nan=False,
    )


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
        if pair_index < n_pos:
            existing = truth.get(sku_id)
            if existing is not None and existing != candidate_gtin:
                raise ValueError(f"SKU {sku_id} has conflicting proxy truth GTINs")
            existing_source = truth_sources.get(sku_id)
            if existing_source is not None and existing_source != source:
                raise ValueError(
                    f"SKU {sku_id} maps to multiple source rows in proxy pairs: "
                    f"{existing_source} and {source}"
                )
            truth[sku_id] = candidate_gtin
            truth_sources[sku_id] = source
        record = candidate_gate_fields(
            sku_row,
            sku_info,
            candidate_gtin,
            candidate_record,
            float(score),
            sku_id=sku_id,
            source_row_index=str(source),
            retrieval_source="calibration_pair",
        )
        record["sku_record_json"] = _metadata_json(sku_row.to_dict())
        record["candidate_record_json"] = _metadata_json(candidate_record)
        records.append(record)
    candidates = pd.DataFrame(records)
    duplicate_keys = ["SKU_ID", "candidate_gtin"]
    duplicate_mask = candidates.duplicated(duplicate_keys, keep=False)
    duplicate_count = int(candidates.duplicated(duplicate_keys, keep="first").sum())
    duplicate_groups = int(
        candidates.loc[duplicate_mask, duplicate_keys]
        .drop_duplicates()
        .shape[0]
    )
    varying = pd.Series(dtype=bool)
    if duplicate_count:
        varying = (
            candidates.loc[duplicate_mask]
            .groupby(duplicate_keys, sort=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
    if duplicate_count and varying.any():
        duplicate_reason = "repeated_sku_canonical_candidate_with_metadata_conflict"
    elif duplicate_count:
        duplicate_reason = "repeated_sku_canonical_candidate"
    else:
        duplicate_reason = "none"
    # Preserve every scored pair.  The final assignment helper deliberately
    # selects one accepted canonical per SKU, while the trace retains every
    # candidate row and its complete source/canonical metadata.
    candidates.attrs["duplicate_rows"] = duplicate_count
    candidates.attrs["duplicate_groups"] = duplicate_groups
    candidates.attrs["duplicate_reason"] = duplicate_reason
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
        gtin_status(
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


def _stratified_sensitivity_rows(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    threshold: float,
    reconciliation_scope: str,
    prediction: pd.DataFrame,
) -> list[CalibrationSensitivityRow]:
    """Measure one threshold through the same assignment/gate path per GTIN stratum."""
    rows: list[CalibrationSensitivityRow] = []
    for metrics in gtin_metrics(
        prediction,
        truth,
        "calibration_sensitivity",
        threshold,
        candidates,
    ):
        rows.append(
            CalibrationSensitivityRow.model_validate(
                {
                    "threshold": float(threshold),
                    "gtin_status": str(metrics["gtin_status"]),
                    "reconciliation_scope": reconciliation_scope,
                    "rand_index": float(metrics["rand_index"]),
                    "adjusted_rand": float(metrics["adjusted_rand"]),
                    "precision": float(metrics["pairwise_precision"]),
                    "recall": float(metrics["pairwise_recall"]),
                    "over_merge_rate": float(metrics["over_merge_rate"]),
                    "under_merge_rate": float(metrics["under_merge_rate"]),
                    "predicted_group_count": int(metrics["predicted_group_count"]),
                }
            )
        )
    return rows


def _fit_threshold(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[float, dict[str, float | int | str]]:
    threshold, _, _ = _fit_fold_threshold(
        candidates,
        truth,
        thresholds,
    )
    return threshold, _assignment_metrics(candidates, truth, threshold)


def _fold_ids(truth: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    items = np.asarray(sorted(truth["true_item_id"].astype(str).unique()))
    order = np.random.default_rng(seed).permutation(len(items))
    return {str(items[position]): int(index % n_folds) for index, position in enumerate(order)}


def _fold_collapse_stats(
    *,
    model,
    df: pd.DataFrame,
    payload: list[str],
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    cfg: dict,
    batch_size: int,
    requested: bool,
) -> dict[str, float | int | str]:
    """Run the shared collapse diagnostic on one calibration check fold."""
    fold_candidates = candidates[candidates["SKU_ID"].isin(truth["SKU_ID"])]
    source_rows = pd.to_numeric(
        fold_candidates[["SKU_ID", "source_row_index"]]
        .drop_duplicates("SKU_ID")["source_row_index"],
        errors="raise",
    ).astype(int)
    if len(source_rows) != truth["SKU_ID"].nunique():
        raise ValueError("calibration fold lost SKU source-row metadata")
    fold_df = df.iloc[source_rows.to_numpy()].reset_index(drop=True)
    fold_payload = [payload[int(row)] for row in source_rows]
    if not requested:
        return collapse_diagnostics(
            model=model,
            df=fold_df,
            payload=fold_payload,
            config={
                **cfg,
                "collapse_guardrail": {
                    **cfg["collapse_guardrail"],
                    "enabled": False,
                },
            },
            batch_size=batch_size,
        )
    return collapse_diagnostics(
        model=model,
        df=fold_df,
        payload=fold_payload,
        config=cfg,
        batch_size=batch_size,
    )


def _collapse_penalty(stats: dict, cfg: dict) -> float:
    guardrail = cfg["collapse_guardrail"]
    if not bool(guardrail["enabled"]):
        return 0.0
    if stats["collapse_status"] in {"not_requested", "disabled"}:
        return 0.0
    if stats["collapse_status"] in {"insufficient_pairs", "unavailable"}:
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
    crossing_excess = max(
        0.0,
        float(stats["collapse_crossing_rate"])
        - float(guardrail["crossing_rate_ceiling"]),
    )
    return float(guardrail["penalty_weight"]) * (
        median_excess + p90_excess + variance_excess + crossing_excess
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
    duplicate_groups = int(candidates.attrs.get("duplicate_groups", 0))
    duplicate_reason = str(candidates.attrs.get("duplicate_reason", "none"))
    n_folds = int(config["hpo"]["calibration_folds"])
    minimum_support = int(config["rand_matching"]["threshold_min_fold_support"])
    if n_folds < minimum_support:
        raise ValueError(
            "HPO calibration has insufficient fold support for a meaningful median: "
            f"configured={n_folds}, required={minimum_support}"
        )
    fold_map = _fold_ids(truth, n_folds, int(config["collapse_guardrail"]["seed"]))
    thresholds = _thresholds(config)
    fold_rows: list[CalibrationFoldMetricRow] = []
    validation_candidates: list[pd.DataFrame] = []
    validation_truth: list[pd.DataFrame] = []
    reconciliation_scope = str(
        config["rand_matching"]["threshold_reconciliation_scope"]
    )
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
        fold_collapse = _fold_collapse_stats(
            model=model,
            df=df,
            payload=payload,
            candidates=candidates,
            truth=check_truth,
            cfg=config,
            batch_size=batch_size,
            requested=include_collapse_guardrail,
        )
        validation_candidates.append(check_candidates)
        validation_truth.append(check_truth)
        fold_penalty = _collapse_penalty(fold_collapse, config)
        fold_rows.append(
            CalibrationFoldMetricRow.model_validate(
                {
                "calibration_fold": fold,
                "calibrated_threshold": threshold,
                "reconciliation_scope": reconciliation_scope,
                "fit_rand_index": float(fit_metrics["rand_index"]),
                "check_rand_index": float(check_metrics["rand_index"]),
                "check_adjusted_rand": float(check_metrics["adjusted_rand"]),
                "check_group_precision": float(check_metrics["group_precision"]),
                "check_group_recall": float(check_metrics["group_recall"]),
                "check_over_merge_rate": float(check_metrics["over_merge_rate"]),
                "check_under_merge_rate": float(check_metrics["under_merge_rate"]),
                "collapse_status": str(fold_collapse["collapse_status"]),
                "collapse_diagnostics_available": int(
                    fold_collapse["collapse_diagnostics_available"]
                ),
                "collapse_healthy": int(fold_collapse["collapse_healthy"]),
                "collapse_median_cosine": fold_collapse.get(
                    "collapse_median_cosine"
                ),
                "collapse_p90_cosine": fold_collapse.get("collapse_p90_cosine"),
                "collapse_cosine_std": fold_collapse.get("collapse_cosine_std"),
                "collapse_operating_threshold": fold_collapse.get(
                    "collapse_operating_threshold"
                ),
                "collapse_crossing_rate": fold_collapse.get(
                    "collapse_crossing_rate"
                ),
                "collapse_crossing_rate_ceiling": fold_collapse.get(
                    "collapse_crossing_rate_ceiling"
                ),
                "collapse_crossing_rate_flag": fold_collapse.get(
                    "collapse_crossing_rate_flag"
                ),
                "collapse_penalty": fold_penalty,
                "fit_rand_index_minus_collapse_penalty": float(
                    fit_metrics["rand_index"]
                )
                - fold_penalty,
                }
            )
        )
    final_threshold = float(np.median([row.calibrated_threshold for row in fold_rows]))
    validation_candidate_frame = pd.concat(validation_candidates, ignore_index=True)
    validation_truth_frame = pd.concat(validation_truth, ignore_index=True)
    sensitivity: list[CalibrationSensitivityRow] = []
    stratified_sensitivity: list[CalibrationSensitivityRow] = []
    sensitivity_metrics: dict[float, dict[str, float | int | str]] = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        prediction = choose_assignments(
            validation_candidate_frame,
            threshold_value,
        )
        metrics = prediction_metrics(
            prediction,
            validation_truth_frame[["SKU_ID", "true_item_id"]],
            candidates=validation_candidate_frame,
            threshold=threshold_value,
            include_graph_diagnostics=False,
        )
        sensitivity_metrics[threshold_value] = metrics
        sensitivity.append(
            CalibrationSensitivityRow.model_validate(
                {
                    "threshold": threshold_value,
                    "gtin_status": "ALL",
                    "reconciliation_scope": reconciliation_scope,
                    "rand_index": float(metrics["rand_index"]),
                    "adjusted_rand": float(metrics["adjusted_rand"]),
                    "precision": float(metrics["pairwise_precision"]),
                    "recall": float(metrics["pairwise_recall"]),
                    "over_merge_rate": float(metrics["over_merge_rate"]),
                    "under_merge_rate": float(metrics["under_merge_rate"]),
                    "predicted_group_count": int(metrics["predicted_group_count"]),
                }
            )
        )
        stratified_sensitivity.extend(
            _stratified_sensitivity_rows(
                validation_candidate_frame,
                validation_truth_frame,
                threshold_value,
                reconciliation_scope,
                prediction,
            )
        )
    overall = dict(
        sensitivity_metrics.get(final_threshold)
        or _assignment_metrics(
            validation_candidate_frame,
            validation_truth_frame,
            final_threshold,
            include_graph_diagnostics=False,
        )
    )
    overall.update(
        candidate_graph_diagnostics(validation_candidate_frame, final_threshold)
    )
    best_rand = max(row.rand_index for row in sensitivity)
    plateau_count = sum(
        row.rand_index >= best_rand - float(config["rand_matching"]["plateau_tolerance"])
        for row in sensitivity
    )
    collapse_config = config
    if not include_collapse_guardrail:
        collapse_config = {
            **config,
            "collapse_guardrail": {
                **config["collapse_guardrail"],
                "enabled": False,
            },
        }
    collapse = collapse_diagnostics(
        model=model,
        df=df,
        payload=payload,
        config=collapse_config,
        batch_size=batch_size,
        requested=include_collapse_guardrail,
    )
    # Alternative thresholds are reported on the same untouched check-fold
    # population as the primary calibration result.  The fold fit data was
    # used only to choose each fold threshold; the full calibration population
    # must never be used to recompute the reported objective.
    calibration_scores, calibration_labels = _candidate_labels(
        validation_candidate_frame,
        validation_truth_frame,
    )
    result: dict[str, float | int | str] = {
        "calibration_proxy_source": str(
            config["rand_matching"]["calibration_proxy_source"]
        ),
        "calibration_status": "available",
        "calibration_positive_pairs": int(len(pos_pairs)),
        "calibration_negative_pairs": int(len(neg_pairs)),
        "calibration_sku_count": int(truth["SKU_ID"].nunique()),
        "calibration_candidate_duplicate_rows_removed": 0,
        "calibration_candidate_duplicate_rows": duplicate_count,
        "calibration_candidate_duplicate_groups": duplicate_groups,
        "calibration_candidate_duplicate_reason": duplicate_reason,
        "calibration_candidate_duplicate_policy": (
            "preserve_all_candidate_rows; assignment_selects_best_per_sku"
        ),
        "calibrated_threshold": final_threshold,
        "calibration_threshold_tie_break": list(
            config["rand_matching"]["threshold_tie_break"]
        ),
        "calibration_reconciliation_scope": reconciliation_scope,
        "calibration_threshold_fold_count": n_folds,
        "calibration_threshold_support_count": len(fold_rows),
        "calibration_threshold_support_minimum": minimum_support,
        "calibration_threshold_support_sufficient": int(
            len(fold_rows) >= minimum_support
        ),
        "calibration_threshold_fold_min": float(min(row.calibrated_threshold for row in fold_rows)),
        "calibration_threshold_fold_max": float(max(row.calibrated_threshold for row in fold_rows)),
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
            _youden_threshold(calibration_scores, calibration_labels)
        ),
        "calibration_precision_at_target_recall_threshold": float(
            _threshold_at_recall(
                calibration_scores,
                calibration_labels,
                float(config["rand_matching"]["target_recall"]),
            )
        ),
        "calibration_sensitivity_table": [row.model_dump() for row in sensitivity],
        "calibration_sensitivity_by_gtin_status": [
            row.model_dump() for row in stratified_sensitivity
        ],
        "calibration_fold_collapse": [row.model_dump() for row in fold_rows],
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
    result.update(
        {
            key: overall[key]
            for key in METRIC_COLUMNS
            if key.startswith("diagnostic_") or key == "plausible_group_count"
        }
    )
    result["diagnostic_component_size_distribution"] = json.loads(
        str(result["diagnostic_component_size_distribution"])
    )
    validation_predictions = choose_assignments(
        validation_candidate_frame,
        final_threshold,
    )
    strata_rows = gtin_metrics(
        validation_predictions,
        validation_truth_frame,
        "trial",
        final_threshold,
        validation_candidate_frame,
    )
    result["calibration_gtin_strata"] = len(strata_rows)
    for row in strata_rows:
        status = str(row["gtin_status"])
        result[f"calibration_{status}_rand_index"] = float(row["rand_index"])
        result[f"calibration_{status}_precision"] = float(row["pairwise_precision"])
        result[f"calibration_{status}_recall"] = float(row["pairwise_recall"])
    truth_by_sku = validation_truth_frame.set_index("SKU_ID")["true_item_id"]
    true_candidates = validation_candidate_frame.loc[
        validation_candidate_frame["candidate_gtin"].eq(
            validation_candidate_frame["SKU_ID"].map(truth_by_sku)
        )
    ]
    if true_candidates.empty:
        raise RuntimeError("calibration proxy lost every true canonical candidate")
    result["attribute_conflict_error_rate"] = float(
        true_candidates["attribute_conflict_type"].ne("none").mean()
    )
    result["attribute_conflict_status"] = "true_candidate_gate_check"
    non_finite_fields = calibration_non_finite_fields(result)
    result["calibration_non_finite_count"] = len(non_finite_fields)
    result["calibration_non_finite_fields"] = ",".join(non_finite_fields)
    CalibrationMetricRow.model_validate(result)
    return result
