"""Final Rand Index-calibrated SKU-to-canonical matching.

This is the standalone version of ``notebooks/final_submission.ipynb``.
It calibrates a cosine threshold on canonical-disjoint folds, reports GTIN
sensitivity, and writes the final ``SKU_ID,ITEM_ID`` submission.

Required inputs may be supplied as CLI arguments or environment variables:

    FINETUNED_CHECKPOINT
    CALIBRATION_INPUT
    HOLDOUT_INPUT

Example::

    er-rand-match \
      --checkpoint /path/to/checkpoint \
      --calibration-input /path/to/calibration.csv \
      --holdout-input /path/to/holdout.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
)

from core.attribute_conflicts import (
    canonical_attribute_info,
    conflict_columns,
    sku_attribute_info,
)
from core.common import (
    F,
    RESULTS,
    TRAIN_ROOT,
    load_config,
    load_dataset_deduped,
    metadata_text,
    rand_matching_cfg,
    row_metadata_text,
)
from core.gtin import is_valid_gtin_checksum
from core.structured_features import (
    append_text as append_structured_text,
    canonical_info as canonical_structured_info,
    fuse_numpy,
    vector as structured_vector,
)
import matplotlib.pyplot as plt
from pipeline import (
    canonical_model_text,
    clean_sku_text,
    load_canonical_map,
    strip_schema_words,
)


GTIN_STATUSES = ("both_equal", "different", "one_missing", "both_missing")
ASSIGNMENT_COLUMNS = ("SKU_ID", "ITEM_ID", "score", "gtin_status")
METRIC_COLUMNS = (
    "n",
    "rand_index",
    "adjusted_rand",
    "group_precision",
    "group_recall",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "pairwise_accuracy",
    "over_merge_rate",
    "under_merge_rate",
    "predicted_group_count",
    "true_group_count",
    "unmatched_skus",
    "tp",
    "tn",
    "fp",
    "fn",
    "pair_count",
)
GATE_COLUMNS = (
    "gtin_gate",
    "attribute_gate",
    "threshold_gate",
    "assignment_gate",
)


def _unmatched_prefix() -> str:
    return str(rand_matching_cfg()["unmatched_prefix"])


def _fit_recall_column(target_recall: float) -> str:
    return f"fit_threshold_at_{target_recall:.0%}_recall"


# ---------------------------------------------------------------------------
# Pydantic output-contract models (column-set validation at write time)
# ---------------------------------------------------------------------------

class _SubmissionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_deduped: str
    checkpoint: str
    final_threshold: float
    unmatched_prefix: str
    rows: int
    unique_items: int
    unmatched: int
    lineage: str


class _SubmissionColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: tuple[str, ...] = Field(min_length=1)

    def validate_frame(self, frame: pd.DataFrame, label: str) -> None:
        actual = tuple(frame.columns)
        if actual != self.columns:
            raise ValueError(
                f"{label} column contract violated: "
                f"expected={self.columns}, actual={actual}"
            )


class _DiagnosticsColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_columns: frozenset[str] = Field(min_length=1)

    def validate_frame(self, frame: pd.DataFrame, label: str) -> None:
        actual = frozenset(frame.columns)
        if actual != self.expected_columns:
            missing = sorted(self.expected_columns - actual)
            unexpected = sorted(actual - self.expected_columns)
            raise ValueError(
                f"{label} column contract violated: "
                f"missing={missing}, unexpected={unexpected}"
            )


_SUBMISSION_COLUMNS_SPEC = _SubmissionColumnSpec(columns=("SKU_ID", "ITEM_ID"))
_DIAGNOSTICS_COLUMNS_SPEC = _DiagnosticsColumnSpec(
    expected_columns=frozenset(
        {
            # candidate trace (_candidate_row)
            "SKU_ID",
            "sku_gtin",
            "candidate_gtin",
            "score",
            "exact_gtin",
            "gtin_status",
            "gate_reason",
            "sku_title",
            "sku_attributes",
            "sku_brand",
            "sku_country",
            "sku_category",
            "sku_category_path",
            "sku_retailer",
            "sku_volume",
            "sku_pack",
            "sku_flavor",
            "source_row_index",
            "true_item_id",
            "calibration_fold",
            "candidate_text",
            "candidate_brand",
            "candidate_volume",
            "candidate_pack",
            "candidate_flavor",
            "rule_ok",
            "attribute_conflict_type",
            "attribute_matches",
            # annotation (_annotate_candidates)
            "gtin_compatible",
            "score_pass",
            "accepted",
            *GATE_COLUMNS,
            "rejection_reason",
            # selection (_assignments_with_trace)
            "selected",
            # merged provenance columns
            "ITEM_ID",
            "n_candidates",
            "source_file",
        }
    )
)


class RandMatcher:
    """Encode canonical items and score SKU candidates against them."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        batch_size: int,
        top_k: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.top_k = top_k
        self.config = load_config()
        self.structured_config = self.config["training"]["structured_features"]
        self.canonical = load_canonical_map()
        self.item_ids = [str(value) for value in self.canonical]
        if not self.item_ids:
            raise RuntimeError("canonical map is empty; candidate retrieval cannot run")
        self.item_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }

        records = pd.read_csv(
            RESULTS / F["canonical_records"],
            dtype=str,
            keep_default_na=False,
        )
        if records["gtin"].duplicated().any():
            raise RuntimeError("canonical_records.csv contains duplicate GTIN rows")
        self.record_map = {
            str(row["gtin"]): row.to_dict()
            for _, row in records.iterrows()
        }
        missing_records = sorted(set(self.item_ids) - set(self.record_map))
        if missing_records:
            raise RuntimeError(
                "canonical metadata is missing for canonical GTINs: "
                f"{missing_records[:10]}"
                + (" ..." if len(missing_records) > 10 else "")
            )

        self.model = SentenceTransformer(str(checkpoint))
        self.structured_enabled = bool(self.structured_config["enabled"])
        self.structured_text = (
            self.structured_enabled
            and bool(self.structured_config["append_to_text"])
        )
        self.structured_weight = (
            float(self.structured_config["embedding_weight"])
            if self.structured_enabled
            and bool(self.structured_config["feed_to_loss"])
            else 0.0
        )

        item_infos = [
            canonical_structured_info(self.record_map[item_id])
            if self.structured_enabled
            else {"volume": set(), "pack": set()}
            for item_id in self.item_ids
        ]
        item_texts = [
            append_structured_text(
                strip_schema_words(canonical_model_text(self.canonical[item_id])),
                info,
                enabled=self.structured_text,
            )
            for item_id, info in zip(self.item_ids, item_infos, strict=True)
        ]
        item_embeddings = self.model.encode(
            item_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        item_features = np.asarray(
            [self._structured_vector(info) for info in item_infos],
            dtype=np.float32,
        )
        self.item_embeddings = fuse_numpy(
            item_embeddings,
            item_features,
            self.structured_weight,
        )
        self.item_embeddings_t = torch.as_tensor(self.item_embeddings)

        print(
            f"loaded {len(self.item_ids):,} canonical items from "
            f"{self.checkpoint}"
        )

    def _structured_vector(self, info: dict) -> np.ndarray:
        return structured_vector(
            info,
            volume_scale_ml=float(self.structured_config["volume_scale_ml"]),
            pack_scale=float(self.structured_config["pack_scale"]),
            max_set_size=int(self.structured_config["max_set_size"]),
        )

    @staticmethod
    def _gtin(value: object) -> str:
        text = metadata_text(value).strip()
        return "" if text.lower() in {"nan", "none", "null"} else text

    @classmethod
    def _trusted_gtin(cls, value: object) -> str:
        """Return a GTIN only when it is a valid identity signal."""
        text = cls._gtin(value)
        return text if text and is_valid_gtin_checksum(text) else ""

    def _text_and_info(self, frame: pd.DataFrame) -> tuple[list[str], list[dict]]:
        infos = [
            sku_attribute_info(
                row_metadata_text(row, "title"),
                row_metadata_text(row, "attributes", "attr"),
            )
            for _, row in frame.iterrows()
        ]
        texts = [
            append_structured_text(
                strip_schema_words(
                    clean_sku_text(
                        row_metadata_text(row, "title"),
                        row_metadata_text(row, "attributes", "attr"),
                        row_metadata_text(row, "brand"),
                    )
                ),
                info,
                enabled=self.structured_text,
            )
            for (_, row), info in zip(frame.iterrows(), infos, strict=True)
        ]
        return texts, infos

    _INVALID_SKU_SENTINELS = frozenset({"", "nan", "none", "null"})

    @staticmethod
    def _normalise_skus(skus: pd.DataFrame) -> pd.DataFrame:
        frame = skus.copy()
        if "SKU_ID" not in frame and "product_id" in frame:
            frame = frame.rename(columns={"product_id": "SKU_ID"})
        if "SKU_ID" not in frame:
            raise ValueError("input must contain SKU_ID or product_id")
        frame["SKU_ID"] = frame["SKU_ID"].astype(str).str.strip()
        bad = frame.loc[
            frame["SKU_ID"].str.lower().isin(RandMatcher._INVALID_SKU_SENTINELS),
            "SKU_ID",
        ]
        if not bad.empty:
            raise ValueError(
                f"input contains invalid SKU_ID values "
                f"(blank/whitespace/nan/none/null): {bad.tolist()[:10]}"
            )
        if frame["SKU_ID"].duplicated().any():
            raise ValueError("matching input contains duplicate SKU_ID values")
        return frame

    def _encode_skus(
        self,
        texts: list[str],
        sku_infos: list[dict],
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        features = np.asarray(
            [self._structured_vector(info) for info in sku_infos],
            dtype=np.float32,
        )
        return fuse_numpy(embeddings, features, self.structured_weight)

    def _candidate_indexes(
        self,
        hits: list[dict],
        sku_gtin: str,
    ) -> set[int]:
        indexes = {int(hit["corpus_id"]) for hit in hits}
        trusted_gtin = self._trusted_gtin(sku_gtin)
        if trusted_gtin in self.item_index:
            indexes.add(self.item_index[trusted_gtin])
        return indexes

    def _candidate_row(
        self,
        row: pd.Series,
        sku_info: dict,
        embedding: np.ndarray,
        candidate_index: int,
    ) -> dict[str, object]:
        sku_gtin = self._gtin(row_metadata_text(row, "barcode", "gtin"))
        candidate_gtin = self.item_ids[candidate_index]
        candidate_record = self.record_map[candidate_gtin]
        candidate_info = canonical_attribute_info(candidate_record)
        rules = conflict_columns(sku_info, candidate_info)
        status = self.gtin_status(sku_gtin, candidate_gtin)
        exact = int(status == "both_equal")
        if status == "different":
            gate_reason = "gtin_conflict"
        elif exact:
            gate_reason = "exact_gtin"
        elif rules["attribute_conflict_type"] != "none":
            gate_reason = "attribute_conflict"
        else:
            gate_reason = "cosine_candidate"
        return {
            "SKU_ID": str(row["SKU_ID"]),
            "sku_gtin": sku_gtin,
            "candidate_gtin": candidate_gtin,
            "score": float(np.dot(embedding, self.item_embeddings[candidate_index])),
            "exact_gtin": exact,
            "gtin_status": status,
            "gate_reason": gate_reason,
            "sku_title": row_metadata_text(row, "title"),
            "sku_attributes": row_metadata_text(row, "attributes", "attr"),
            "sku_brand": row_metadata_text(row, "brand"),
            "sku_country": row_metadata_text(row, "country"),
            "sku_category": row_metadata_text(row, "category"),
            "sku_category_path": row_metadata_text(row, "category_path"),
            "sku_retailer": row_metadata_text(row, "retailer"),
            "sku_volume": json.dumps(sorted(sku_info["volume"])),
            "sku_pack": json.dumps(sorted(sku_info["pack"])),
            "sku_flavor": str(sku_info["flavor"]),
            "source_row_index": str(row.name),
            "true_item_id": row_metadata_text(row, "true_item_id"),
            "calibration_fold": row_metadata_text(row, "calibration_fold"),
            "candidate_text": str(self.canonical[candidate_gtin]),
            "candidate_brand": metadata_text(candidate_record["mode_brand"]),
            "candidate_volume": json.dumps(sorted(candidate_info["volume"])),
            "candidate_pack": json.dumps(sorted(candidate_info["pack"])),
            "candidate_flavor": str(candidate_info["flavor"]),
            "rule_ok": int(rules["attribute_conflict_type"] == "none"),
            "attribute_conflict_type": str(rules["attribute_conflict_type"]),
            "attribute_matches": int(
                sum(
                    not rules[key]
                    for key in ("volume_conflict", "pack_conflict", "flavor_conflict")
                )
            ),
        }

    def gtin_status(self, sku_gtin: object, candidate_gtin: object) -> str:
        left = self._trusted_gtin(sku_gtin)
        right = self._trusted_gtin(candidate_gtin)
        if not left and not right:
            return "both_missing"
        if not left or not right:
            return "one_missing"
        return "both_equal" if left == right else "different"

    def score_candidates(
        self,
        skus: pd.DataFrame,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        top_k = self.top_k if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be positive")
        frame = self._normalise_skus(skus)
        texts, sku_infos = self._text_and_info(frame)
        embeddings = self._encode_skus(texts, sku_infos)
        hits = util.semantic_search(
            torch.as_tensor(embeddings),
            self.item_embeddings_t,
            top_k=min(top_k, len(self.item_ids)),
        )

        rows: list[dict[str, object]] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            sku_gtin = self._gtin(row_metadata_text(row, "barcode", "gtin"))
            for index in sorted(self._candidate_indexes(hits[position], sku_gtin)):
                rows.append(
                    self._candidate_row(
                        row,
                        sku_infos[position],
                        embeddings[position],
                        index,
                    )
                )
        candidates = pd.DataFrame(rows)
        candidate_ids = set(candidates["SKU_ID"]) if not candidates.empty else set()
        missing_ids = sorted(set(frame["SKU_ID"]) - candidate_ids)
        if missing_ids:
            raise RuntimeError(
                "candidate retrieval dropped SKU_ID values: "
                f"{missing_ids[:10]}" + (" ..." if len(missing_ids) > 10 else "")
            )
        return candidates


def _annotate_candidates(candidates: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = candidates.copy()
    frame["gtin_compatible"] = frame["gtin_status"].ne("different")
    frame["score_pass"] = frame["score"] >= float(threshold)
    frame["accepted"] = frame["gtin_compatible"] & (
        frame["exact_gtin"].astype(bool)
        | (frame["rule_ok"].astype(bool) & frame["score_pass"])
    )
    frame["gtin_gate"] = np.select(
        [
            frame["gtin_status"].eq("different"),
            frame["exact_gtin"].astype(bool),
        ],
        ["veto", "lock"],
        default="allow_unknown",
    )
    frame["attribute_gate"] = np.select(
        [
            frame["exact_gtin"].astype(bool) & ~frame["rule_ok"].astype(bool),
            frame["exact_gtin"].astype(bool),
            frame["rule_ok"].astype(bool),
        ],
        [
            "override_exact_gtin",
            "exact_gtin_checked",
            "allow_agree_or_unknown",
        ],
        default="veto_known_conflict",
    )
    frame["threshold_gate"] = np.select(
        [
            frame["exact_gtin"].astype(bool),
            frame["score_pass"],
        ],
        ["bypass_exact_gtin", "pass"],
        default="fail",
    )
    frame["assignment_gate"] = np.where(
        frame["accepted"],
        "accepted_candidate",
        "rejected_candidate",
    )
    frame["rejection_reason"] = np.select(
        [
            ~frame["gtin_compatible"],
            frame["exact_gtin"].astype(bool),
            ~frame["rule_ok"].astype(bool),
            ~frame["score_pass"],
        ],
        [
            "gtin_conflict",
            "exact_gtin_lock",
            "attribute_conflict",
            "below_threshold",
        ],
        default="accepted_candidate",
    )
    expected_gate = tuple(GATE_COLUMNS)
    actual_gate = [c for c in frame.columns if c in expected_gate]
    if tuple(actual_gate) != expected_gate:
        raise RuntimeError(
            f"GATE_COLUMNS contract violated: expected={expected_gate}, "
            f"actual={tuple(actual_gate)}"
        )
    return frame


def _assignments_with_trace(
    candidates: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return (
            pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            candidates.copy(),
        )
    frame = _annotate_candidates(candidates, threshold)
    accepted = frame[frame["accepted"]].sort_values(
        ["SKU_ID", "exact_gtin", "score", "attribute_matches", "candidate_gtin"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    selected = accepted.drop_duplicates("SKU_ID", keep="first").copy()
    selected_keys = selected[["SKU_ID", "candidate_gtin"]].assign(selected=1)
    trace = frame.copy()
    trace_keys = pd.MultiIndex.from_frame(
        trace[["SKU_ID", "candidate_gtin"]]
    )
    selected_key_index = pd.MultiIndex.from_frame(
        selected_keys[["SKU_ID", "candidate_gtin"]]
    )
    missing_selected = selected_key_index.difference(trace_keys)
    if len(missing_selected):
        raise RuntimeError(
            "selected assignment key is absent from candidate trace: "
            f"{list(missing_selected)}"
        )
    trace["selected"] = trace_keys.isin(selected_key_index).astype("int8")
    trace["assignment_gate"] = np.where(
        trace["selected"].astype(bool),
        "selected_best_candidate",
        trace["assignment_gate"],
    )
    best = selected[
        ["SKU_ID", "candidate_gtin", "score", "gtin_status"]
    ].rename(columns={"candidate_gtin": "ITEM_ID"})
    best = best.loc[:, list(ASSIGNMENT_COLUMNS)]
    all_skus = candidates[["SKU_ID"]].drop_duplicates()
    output = all_skus.merge(best, on="SKU_ID", how="left")
    prefix = _unmatched_prefix()
    output["ITEM_ID"] = output["ITEM_ID"].fillna(
        prefix + output["SKU_ID"].astype(str)
    )
    return output, trace


def choose_assignments(
    candidates: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Choose one direct canonical assignment per SKU.

    This intentionally does not build connected components or perform
    transitive similarity chaining.
    """
    output, _ = _assignments_with_trace(candidates, threshold)
    return output


def _combination_count(n: int) -> int:
    return n * (n - 1) // 2


def _pairwise_counts(true_labels: pd.Series, predicted_labels: pd.Series) -> dict[str, int]:
    """Compute pairwise clustering confusion counts without an O(n²) matrix."""
    frame = pd.DataFrame(
        {"true": true_labels.astype(str), "predicted": predicted_labels.astype(str)}
    )
    total = _combination_count(len(frame))
    true_same = sum(
        _combination_count(int(count))
        for count in frame["true"].value_counts().tolist()
    )
    predicted_same = sum(
        _combination_count(int(count))
        for count in frame["predicted"].value_counts().tolist()
    )
    true_pred_same = sum(
        _combination_count(int(count))
        for count in frame.groupby(["true", "predicted"], sort=False).size().tolist()
    )
    false_positive = predicted_same - true_pred_same
    false_negative = true_same - true_pred_same
    true_negative = total - true_pred_same - false_positive - false_negative
    return {
        "tp": int(true_pred_same),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "pair_count": int(total),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prediction_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    """Return clustering metrics for one assignment population.

    ``group_precision`` and ``group_recall`` are B-cubed, item-weighted
    measures.  The pairwise measures operate on all unordered SKU pairs;
    they are the direct group-equivalence interpretation of Rand Index.
    """
    truth_frame = truth[["SKU_ID", "true_item_id"]].copy()
    pred_frame = pred[["SKU_ID", "ITEM_ID"]].copy()
    truth_frame["SKU_ID"] = truth_frame["SKU_ID"].astype(str)
    pred_frame["SKU_ID"] = pred_frame["SKU_ID"].astype(str)
    truth_ids = set(truth_frame["SKU_ID"])
    pred_ids = set(pred_frame["SKU_ID"])
    missing_predictions = sorted(truth_ids - pred_ids)
    unexpected_predictions = sorted(pred_ids - truth_ids)
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "metric population mismatch: "
            f"missing_predictions={missing_predictions[:10]}, "
            f"unexpected_predictions={unexpected_predictions[:10]}"
        )
    if truth_frame["SKU_ID"].duplicated().any() or pred_frame["SKU_ID"].duplicated().any():
        raise ValueError("metric inputs must contain one row per SKU_ID")
    merged = truth_frame.merge(
        pred_frame,
        on="SKU_ID",
        how="left",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("no calibration rows matched predictions")

    counts = _pairwise_counts(merged["true_item_id"], merged["ITEM_ID"])
    tp, tn, fp, fn = (
        counts["tp"],
        counts["tn"],
        counts["fp"],
        counts["fn"],
    )
    intersections = merged.groupby(
        ["true_item_id", "ITEM_ID"], sort=False
    ).size()
    predicted_sizes = merged.groupby("ITEM_ID").size()
    true_sizes = merged.groupby("true_item_id").size()
    group_precision = []
    group_recall = []
    for (true_id, predicted_id), intersection in intersections.items():
        group_precision.extend(
            [
                float(intersection / predicted_sizes[predicted_id])
            ] * int(intersection)
        )
        group_recall.extend(
            [float(intersection / true_sizes[true_id])] * int(intersection)
        )

    metrics = {
        "n": int(len(merged)),
        "rand_index": _safe_ratio(tp + tn, counts["pair_count"]),
        "adjusted_rand": float(
            adjusted_rand_score(merged["true_item_id"], merged["ITEM_ID"])
        ),
        "group_precision": float(np.mean(group_precision)),
        "group_recall": float(np.mean(group_recall)),
        "pairwise_precision": _safe_ratio(tp, tp + fp),
        "pairwise_recall": _safe_ratio(tp, tp + fn),
        "pairwise_f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "pairwise_accuracy": _safe_ratio(tp + tn, counts["pair_count"]),
        "over_merge_rate": _safe_ratio(fp, tp + fp),
        "under_merge_rate": _safe_ratio(fn, tp + fn),
        "predicted_group_count": int(merged["ITEM_ID"].nunique()),
        "true_group_count": int(merged["true_item_id"].nunique()),
        "unmatched_skus": int(
            merged["ITEM_ID"].astype(str).str.startswith(_unmatched_prefix()).sum()
        ),
        **counts,
    }
    if tuple(metrics) != METRIC_COLUMNS:
        raise RuntimeError(
            "prediction metric contract drifted: "
            f"expected={METRIC_COLUMNS}, actual={tuple(metrics)}"
        )
    return metrics


def gtin_metrics(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    fold: object,
    threshold: float,
) -> list[dict[str, float | int | str]]:
    truth_frame = truth[["SKU_ID", "true_item_id", "gtin_status"]].copy()
    pred_frame = pred[["SKU_ID", "ITEM_ID"]].copy()
    truth_frame["SKU_ID"] = truth_frame["SKU_ID"].astype(str)
    pred_frame["SKU_ID"] = pred_frame["SKU_ID"].astype(str)
    merged = truth_frame.merge(
        pred_frame,
        on="SKU_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(truth_frame):
        raise ValueError("GTIN-stratified metric population changed during merge")
    groups = [
        (status, merged[merged["gtin_status"].eq(status)])
        for status in GTIN_STATUSES
    ]
    groups.append(("ALL", merged))
    rows = []
    for status, group in groups:
        if group.empty:
            row = {key: np.nan for key in METRIC_COLUMNS}
            row.update(
                {
                    "check_fold": fold,
                    "threshold": float(threshold),
                    "gtin_status": status,
                    "selection_method": "sensitivity",
                    "n": 0,
                    "predicted_group_count": 0,
                    "true_group_count": 0,
                    "unmatched_skus": 0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "pair_count": 0,
                }
            )
            rows.append(row)
            continue
        rows.append(
            {
                "check_fold": fold,
                "threshold": float(threshold),
                "gtin_status": status,
                "selection_method": "sensitivity",
                **prediction_metrics(
                    group[["SKU_ID", "ITEM_ID"]],
                    group[["SKU_ID", "true_item_id"]],
                ),
            }
        )
    return rows


def _youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return a Youden-J threshold fitted on one labeled population."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    tpr = np.cumsum(ordered_labels) / max(int(labels.sum()), 1)
    fpr = np.cumsum(1 - ordered_labels) / max(int((labels == 0).sum()), 1)
    best = np.flatnonzero((tpr - fpr) == np.max(tpr - fpr))
    return float(scores[order[best[-1]]])


def _threshold_at_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> float:
    """Return the highest observed score retaining the target recall."""
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    thresholds = np.unique(scores)[::-1]
    valid = [
        threshold
        for threshold in thresholds
        if float(np.sum((scores >= threshold) & (labels == 1))) / positives
        >= target_recall
    ]
    return float(max(valid)) if valid else float("nan")


def _recall_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """Return recall achieved at ``threshold`` (NaN when there are no positives)."""
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    return float(np.sum((scores >= threshold) & (labels == 1)) / positives)


def _candidate_labels(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    scored = candidates[["SKU_ID", "candidate_gtin", "score"]].merge(
        truth[["SKU_ID", "true_item_id"]],
        on="SKU_ID",
        how="inner",
    )
    return (
        scored["score"].to_numpy(dtype=float),
        scored["candidate_gtin"].astype(str).eq(
            scored["true_item_id"].astype(str)
        ).to_numpy(dtype=int),
    )


def _retrieval_diagnostics(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, int]:
    """Measure how many truth SKUs have at least one retrieved candidate.

    ``without_candidate_gtin_trusted`` counts the unretrieved SKUs that carry a
    GTIN signal (``gtin_status`` other than ``both_missing``), i.e. SKUs that an
    exact-GTIN rescue in ``_candidate_indexes`` could in principle have
    retrieved — so their absence is a hard retrieval failure.
    """
    truth_ids = set(truth["SKU_ID"].astype(str))
    candidate_ids = (
        set(candidates["SKU_ID"].astype(str)) if not candidates.empty else set()
    )
    with_any = truth_ids & candidate_ids
    without = truth_ids - candidate_ids
    gtin_trusted = 0
    if "gtin_status" in truth.columns:
        by_sku = truth.drop_duplicates("SKU_ID")
        gtin_trusted = int(
            by_sku.loc[
                by_sku["SKU_ID"].astype(str).isin(without)
                & by_sku["gtin_status"].ne("both_missing"),
                "SKU_ID",
            ].nunique()
        )
    return {
        "truth_population": len(truth_ids),
        "with_candidate": len(with_any),
        "without_candidate": len(without),
        "without_candidate_gtin_trusted": gtin_trusted,
    }


def _load_calibration_frame(
    matcher: RandMatcher,
    calibration_input: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(calibration_input, dtype=str, keep_default_na=False)
    required = {"SKU_ID", "true_item_id", "calibration_fold"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"calibration input missing columns: {sorted(missing)}")
    labels["SKU_ID"] = labels["SKU_ID"].astype(str)
    labels["true_item_id"] = labels["true_item_id"].astype(str)
    if labels["SKU_ID"].duplicated().any():
        raise ValueError("calibration input contains duplicate SKU_ID values")
    if labels[["SKU_ID", "true_item_id", "calibration_fold"]].eq("").any().any():
        raise ValueError("calibration input contains blank identity or fold values")
    base = load_dataset_deduped().rename(columns={"product_id": "SKU_ID"})
    base["SKU_ID"] = base["SKU_ID"].astype(str)
    unknown_ids = sorted(set(labels["SKU_ID"]) - set(base["SKU_ID"]))
    if unknown_ids:
        raise ValueError(
            "calibration input contains SKU_ID values absent from the dataset: "
            f"{unknown_ids[:10]}" + (" ..." if len(unknown_ids) > 10 else "")
        )
    calibration = base.merge(
        labels[["SKU_ID", "true_item_id", "calibration_fold"]],
        on="SKU_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(calibration) != len(labels):
        raise RuntimeError("calibration merge changed the labeled population")
    if calibration.empty:
        raise ValueError("calibration input does not match the deduped dataset")
    calibration["SKU_ID"] = calibration["SKU_ID"].astype(str)
    calibration["true_item_id"] = calibration["true_item_id"].astype(str)
    folds_per_item = calibration.groupby("true_item_id")[
        "calibration_fold"
    ].nunique()
    if (folds_per_item > 1).any():
        raise ValueError(
            "calibration is not canonical-disjoint: an item appears in multiple folds"
        )
    if calibration["calibration_fold"].nunique() < 2:
        raise ValueError("calibration requires at least two canonical-disjoint folds")
    calibration["gtin_status"] = [
        matcher.gtin_status(row_metadata_text(row, "barcode", "gtin"), row["true_item_id"])
        for _, row in calibration.iterrows()
    ]
    return calibration


def _fold_partitions(
    calibration: pd.DataFrame,
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    fold: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_skus = set(
        calibration.loc[calibration["calibration_fold"] != fold, "SKU_ID"]
    )
    check_skus = set(
        calibration.loc[calibration["calibration_fold"] == fold, "SKU_ID"]
    )
    return (
        truth[truth.SKU_ID.isin(fit_skus)],
        truth[truth.SKU_ID.isin(check_skus)],
        candidates[candidates.SKU_ID.isin(fit_skus)],
        candidates[candidates.SKU_ID.isin(check_skus)],
    )


def _sweep_assignments(
    candidates: pd.DataFrame,
    thresholds: np.ndarray,
) -> dict[float, pd.DataFrame]:
    """Compute assignments for many thresholds with one annotation + sort.

    Threshold-independent candidate columns are annotated once; the
    accepted-subset is pre-sorted once with the exact tie-break order used by
    ``_assignments_with_trace``; per threshold only ``score_pass`` /
    ``accepted`` are recomputed before the SKU dedup and the UNMATCHED fill.
    """
    if candidates.empty:
        return {float(t): pd.DataFrame(columns=ASSIGNMENT_COLUMNS) for t in thresholds}
    all_skus = candidates[["SKU_ID"]].drop_duplicates()
    base = (
        candidates[
            [
                "SKU_ID",
                "candidate_gtin",
                "score",
                "exact_gtin",
                "gtin_status",
                "rule_ok",
                "attribute_matches",
            ]
        ]
        .copy()
    )
    base["gtin_compatible"] = base["gtin_status"].ne("different")
    base = base.sort_values(
        ["SKU_ID", "exact_gtin", "score", "attribute_matches", "candidate_gtin"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    exact = base["exact_gtin"].values.astype(bool)
    rule_ok = base["rule_ok"].values.astype(bool)
    compatible = base["gtin_compatible"].values
    scores = base["score"].values
    result: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        t = float(threshold)
        accepted = compatible & (exact | (rule_ok & (scores >= t)))
        selected = base[accepted].drop_duplicates("SKU_ID", keep="first")
        best = selected[
            ["SKU_ID", "candidate_gtin", "score", "gtin_status"]
        ].rename(columns={"candidate_gtin": "ITEM_ID"})
        best = best.loc[:, list(ASSIGNMENT_COLUMNS)]
        output = all_skus.merge(best, on="SKU_ID", how="left")
        prefix = _unmatched_prefix()
        output["ITEM_ID"] = output["ITEM_ID"].fillna(
            prefix + output["SKU_ID"].astype(str)
        )
        result[t] = output
    return result


def _fit_fold_threshold(
    fit_candidates: pd.DataFrame,
    fit_truth: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[float, list[dict[str, float]], float]:
    sweep = _sweep_assignments(fit_candidates, thresholds)
    fit_rows: list[dict[str, float]] = []
    for threshold in thresholds:
        t = float(threshold)
        metrics = prediction_metrics(sweep[t], fit_truth)
        fit_rows.append(
            {
                "threshold": t,
                "rand_index": metrics["rand_index"],
                "unmatched_skus": metrics["unmatched_skus"],
                "n": metrics["n"],
            }
        )
    selected_row = max(fit_rows, key=lambda r: (r["rand_index"], r["threshold"]))
    fit_unmatched_fraction = float(selected_row["unmatched_skus"]) / max(
        int(selected_row["n"]), 1
    )
    return selected_row["threshold"], fit_rows, fit_unmatched_fraction


def _alternative_thresholds(
    fit_candidates: pd.DataFrame,
    fit_truth: pd.DataFrame,
    rand_threshold: float,
    target_recall: float,
) -> dict[str, dict[str, float | str]]:
    scores, labels = _candidate_labels(fit_candidates, fit_truth)
    positives = int(labels.sum())
    youden_val = _youden_threshold(scores, labels)
    precision_val = _threshold_at_recall(scores, labels, target_recall)
    if positives == 0:
        youden_reason = "no_positives"
        precision_reason = "no_positives"
    else:
        youden_reason = "fitted" if not np.isnan(youden_val) else "no_positives"
        precision_reason = (
            "fitted" if not np.isnan(precision_val) else "target_recall_unreachable"
        )
    if np.isnan(precision_val):
        achieved_recall = _recall_at_threshold(
            scores, labels, float(np.min(scores))
        )
    else:
        achieved_recall = _recall_at_threshold(scores, labels, float(precision_val))
    return {
        "rand_index": {
            "threshold": float(rand_threshold),
            "reason": "fitted",
            "achieved_recall": float("nan"),
        },
        "youden": {
            "threshold": float(youden_val),
            "reason": youden_reason,
            "achieved_recall": float("nan"),
        },
        f"precision_at_{target_recall:.0%}_recall": {
            "threshold": float(precision_val),
            "reason": precision_reason,
            "achieved_recall": achieved_recall,
        },
    }


def _fold_sensitivity(
    check_candidates: pd.DataFrame,
    check_truth: pd.DataFrame,
    fold: object,
    thresholds: np.ndarray,
    alternatives: dict[str, dict[str, float | str]],
) -> list[dict[str, float | int | str]]:
    sweep = _sweep_assignments(
        check_candidates,
        np.unique(
            np.concatenate(
                [
                    np.asarray(thresholds, dtype=float),
                    np.array(
                        [
                            float(info["threshold"])
                            for info in alternatives.values()
                            if not np.isnan(float(info["threshold"]))
                        ],
                        dtype=float,
                    ),
                ]
            )
        ),
    )
    rows: list[dict[str, float | int | str]] = []
    for method, info in alternatives.items():
        threshold = float(info["threshold"])
        reason = str(info["reason"])
        if np.isnan(threshold):
            row = {key: np.nan for key in METRIC_COLUMNS}
            row.update(
                {
                    "check_fold": fold,
                    "threshold": float("nan"),
                    "gtin_status": "ALL",
                    "selection_method": method,
                    "sensitivity_reason": reason,
                    "n": 0,
                    "predicted_group_count": 0,
                    "true_group_count": 0,
                    "unmatched_skus": 0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "pair_count": 0,
                }
            )
            rows.append(row)
            continue
        prediction = sweep[float(threshold)]
        row_data: dict[str, float | int | str] = {
            "check_fold": fold,
            "threshold": float(threshold),
            "gtin_status": "ALL",
            "selection_method": method,
            "sensitivity_reason": "fitted",
            **prediction_metrics(prediction, check_truth),
        }
        rows.append(row_data)
    for threshold in thresholds:
        prediction = sweep[float(threshold)]
        for row_data in gtin_metrics(
            prediction, check_truth, fold, float(threshold)
        ):
            row_data.setdefault("sensitivity_reason", "fitted")
            rows.append(row_data)
    return rows


def _plateau_diagnostic(
    sensitivity: pd.DataFrame,
    plateau_tolerance: float,
    plateau_min_points: int,
    selected_df: pd.DataFrame | None = None,
) -> dict:
    all_rows = sensitivity[
        sensitivity["gtin_status"].eq("ALL")
        & sensitivity["selection_method"].eq("sensitivity")
    ]
    by_threshold = all_rows.groupby("threshold", as_index=False).rand_index.mean()
    best_rand = float(by_threshold.rand_index.max())
    near_best = by_threshold[
        by_threshold.rand_index >= best_rand - plateau_tolerance
    ]
    result: dict = {
        "criterion": (
            f"mean Rand Index within {plateau_tolerance:g} of best across "
            f"at least {plateau_min_points} grid points"
        ),
        "best_mean_rand_index": best_rand,
        "near_best_thresholds": near_best.threshold.astype(float).tolist(),
        "width": float(near_best.threshold.max() - near_best.threshold.min()),
        "stable": bool(len(near_best) >= plateau_min_points),
        "narrow_peak_warning": bool(len(near_best) < plateau_min_points),
    }
    if selected_df is not None and "fit_unmatched_fraction" in selected_df.columns:
        max_uf = float(selected_df["fit_unmatched_fraction"].max())
        result["max_fit_unmatched_fraction"] = max_uf
        result["degenerate_unmatched_plateau"] = bool(max_uf > 0.5)
    return result


def calibrate_threshold(
    matcher: RandMatcher,
    calibration_input: Path,
    thresholds: np.ndarray,
    *,
    target_recall: float,
    plateau_tolerance: float,
    plateau_min_points: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    calibration = _load_calibration_frame(matcher, calibration_input)
    candidates = matcher.score_candidates(calibration)
    truth = calibration[
        ["SKU_ID", "true_item_id", "calibration_fold", "gtin_status"]
    ].drop_duplicates("SKU_ID")
    folds = sorted(calibration["calibration_fold"].unique())
    selected = []
    sensitivity: list[dict] = []

    for fold in folds:
        fit_truth, check_truth, fit_candidates, check_candidates = _fold_partitions(
            calibration, truth, candidates, fold
        )
        best_threshold, fit_rows, fit_unmatched_fraction = _fit_fold_threshold(
            fit_candidates, fit_truth, thresholds
        )
        alternatives = _alternative_thresholds(
            fit_candidates, fit_truth, best_threshold, target_recall
        )
        fit_by_threshold = {r["threshold"]: r for r in fit_rows}
        youden_info = alternatives["youden"]
        precision_key = f"precision_at_{target_recall:.0%}_recall"
        precision_info = alternatives[precision_key]
        retrieval = _retrieval_diagnostics(fit_candidates, fit_truth)
        fit_col = _fit_recall_column(target_recall)
        selected.append(
            {
                "check_fold": fold,
                "selected_threshold": float(best_threshold),
                "fit_rand_index": float(
                    fit_by_threshold[best_threshold]["rand_index"]
                ),
                "fit_youden_threshold": float(youden_info["threshold"]),
                "fit_youden_reason": youden_info["reason"],
                fit_col: float(precision_info["threshold"]),
                "fit_recall_reason": precision_info["reason"],
                "fit_recall_achieved": float(precision_info["achieved_recall"]),
                "fit_unmatched_fraction": fit_unmatched_fraction,
                "truth_population": retrieval["truth_population"],
                "with_candidate": retrieval["with_candidate"],
                "without_candidate": retrieval["without_candidate"],
                "without_candidate_gtin_trusted": retrieval[
                    "without_candidate_gtin_trusted"
                ],
            }
        )
        sensitivity.extend(
            _fold_sensitivity(
                check_candidates,
                check_truth,
                fold,
                thresholds,
                alternatives,
            )
        )

    selected_df = pd.DataFrame(selected)
    sensitivity_df = pd.DataFrame(sensitivity)
    final_threshold = float(selected_df.selected_threshold.median())
    plateau = _plateau_diagnostic(
        sensitivity_df,
        plateau_tolerance,
        plateau_min_points,
        selected_df=selected_df,
    )
    alternatives = sensitivity_df[
        sensitivity_df["gtin_status"].eq("ALL")
        & sensitivity_df["selection_method"].ne("sensitivity")
    ].copy()
    return final_threshold, selected_df, sensitivity_df, alternatives, plateau


def _write_calibration_outputs(
    matcher: RandMatcher,
    output_dir: Path,
    output_names: dict[str, str],
    selected_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    alternatives_df: pd.DataFrame,
    plateau: dict,
    final_threshold: float,
    target_recall: float,
) -> None:
    selected_df.to_csv(output_dir / output_names["threshold_selection_by_fold"], index=False)
    sensitivity_df.to_csv(
        output_dir / output_names["threshold_sensitivity_by_gtin_status"],
        index=False,
    )
    alternatives_df.to_csv(output_dir / output_names["threshold_comparison"], index=False)
    plateau.update(
        {
            "final_threshold": final_threshold,
            "selection_method": "median of fold-selected Rand Index thresholds",
            "target_recall": target_recall,
            "tie_break": ["exact_gtin", "score", "attribute_matches", "candidate_gtin"],
            "unmatched_item_id": _unmatched_prefix() + "<SKU_ID>",
            "no_transitive_chaining": True,
        }
    )
    (output_dir / output_names["plateau_diagnostic"]).write_text(
        json.dumps(plateau, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_sensitivity_plot(
        sensitivity_df,
        final_threshold,
        output_dir / output_names["threshold_sensitivity_plot"],
        int(matcher.config["plots"]["dpi"]),
    )


def _write_sensitivity_plot(
    sensitivity: pd.DataFrame,
    final_threshold: float,
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for status, group in sensitivity.groupby("gtin_status"):
        if status == "ALL" or group["selection_method"].ne("sensitivity").all():
            continue
        curve = group.groupby("threshold", as_index=False).rand_index.mean()
        axis.plot(curve.threshold, curve.rand_index, marker="o", label=status)
    axis.axvline(final_threshold, color="black", linestyle="--", label=f"final={final_threshold:.2f}")
    axis.set(
        xlabel="Cosine threshold",
        ylabel="Rand Index",
        title="Threshold sensitivity by GTIN availability",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _load_labeled_input(path: Path, name: str) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"SKU_ID", "true_item_id"} - set(labels.columns)
    if missing:
        raise ValueError(f"{name} input missing columns: {sorted(missing)}")
    labels["SKU_ID"] = labels["SKU_ID"].astype(str)
    labels["true_item_id"] = labels["true_item_id"].astype(str)
    if labels[["SKU_ID", "true_item_id"]].eq("").any().any():
        raise ValueError(f"{name} input contains blank identity values")
    if labels["SKU_ID"].duplicated().any():
        raise ValueError(f"{name} input contains duplicate SKU_ID values")
    return labels


def _check_calibration_holdout_disjoint(
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
) -> None:
    if set(calibration["SKU_ID"]) & set(holdout["SKU_ID"]):
        raise ValueError("calibration and holdout inputs share SKU_ID values")
    if set(calibration["true_item_id"]) & set(holdout["true_item_id"]):
        raise ValueError("calibration and holdout inputs share canonical identities")


def _evaluate_holdout(
    matcher: RandMatcher,
    holdout_labels: pd.DataFrame,
    final_threshold: float,
) -> list[dict]:
    base = load_dataset_deduped().rename(columns={"product_id": "SKU_ID"})
    base["SKU_ID"] = base["SKU_ID"].astype(str)
    unknown_ids = sorted(
        set(holdout_labels["SKU_ID"]) - set(base["SKU_ID"])
    )
    if unknown_ids:
        raise ValueError(
            "holdout input contains SKU_ID values absent from the dataset: "
            f"{unknown_ids[:10]}" + (" ..." if len(unknown_ids) > 10 else "")
        )
    holdout = base.merge(
        holdout_labels[["SKU_ID", "true_item_id"]],
        on="SKU_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(holdout) != len(holdout_labels):
        raise RuntimeError("holdout merge changed the labeled population")
    holdout["SKU_ID"] = holdout["SKU_ID"].astype(str)
    holdout["true_item_id"] = holdout["true_item_id"].astype(str)
    holdout["gtin_status"] = [
        matcher.gtin_status(row_metadata_text(row, "barcode", "gtin"), row["true_item_id"])
        for _, row in holdout.iterrows()
    ]
    candidates = matcher.score_candidates(holdout)
    truth = holdout[["SKU_ID", "true_item_id", "gtin_status"]].drop_duplicates("SKU_ID")
    predictions = choose_assignments(candidates, final_threshold)
    return gtin_metrics(predictions, truth, "holdout", final_threshold)


def _write_final_submission(
    matcher: RandMatcher,
    output_dir: Path,
    output_names: dict[str, str],
    final_threshold: float,
) -> pd.DataFrame:
    skus = load_dataset_deduped()
    candidates = matcher.score_candidates(skus)
    predictions, candidate_trace = _assignments_with_trace(
        candidates,
        final_threshold,
    )
    submission = predictions[["SKU_ID", "ITEM_ID"]].copy()
    _SUBMISSION_COLUMNS_SPEC.validate_frame(submission, "submission")
    if submission.SKU_ID.duplicated().any():
        raise AssertionError("submission has duplicate SKU_ID values")
    expected_ids = set(skus["product_id"].astype(str))
    actual_ids = set(submission["SKU_ID"].astype(str))
    if actual_ids != expected_ids:
        raise AssertionError(
            "submission changed the SKU population: "
            f"missing={sorted(expected_ids - actual_ids)[:10]}, "
            f"unexpected={sorted(actual_ids - expected_ids)[:10]}"
        )
    submission.to_csv(output_dir / output_names["submission"], index=False)
    diagnostics = candidate_trace.merge(
        predictions[["SKU_ID", "ITEM_ID"]],
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    ).merge(
        candidates.groupby("SKU_ID", as_index=False).agg(
            n_candidates=("candidate_gtin", "nunique")
        ),
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    diagnostics["source_file"] = str(F["dataset_deduped"])
    _DIAGNOSTICS_COLUMNS_SPEC.validate_frame(diagnostics, "diagnostics")
    diagnostics.to_csv(output_dir / output_names["diagnostics"], index=False)
    prefix = _unmatched_prefix()
    provenance = _SubmissionProvenance(
        dataset_deduped=str(F["dataset_deduped"]),
        checkpoint=str(matcher.checkpoint),
        final_threshold=final_threshold,
        unmatched_prefix=prefix,
        rows=len(submission),
        unique_items=int(submission["ITEM_ID"].nunique()),
        unmatched=int(submission["ITEM_ID"].astype(str).str.startswith(prefix).sum()),
        lineage=(
            f"submission+diagnostics derive from {F['dataset_deduped']} "
            f"(row identity = source_row_index); diagnostics row for SKU X "
            f"links to ITEM_ID accepted in submission; canonical item text "
            f"via results/{F['canonical_records']}"
        ),
    )
    provenance_path = output_dir / "submission_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"provenance: {provenance_path}")
    return submission


def write_outputs(
    matcher: RandMatcher,
    calibration_input: Path,
    holdout_input: Path,
    thresholds: np.ndarray,
    *,
    output_dir: Path,
    output_names: dict[str, str],
    target_recall: float,
    plateau_tolerance: float,
    plateau_min_points: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_labels = _load_labeled_input(calibration_input, "calibration")
    holdout_labels = _load_labeled_input(holdout_input, "holdout")
    _check_calibration_holdout_disjoint(calibration_labels, holdout_labels)
    result = calibrate_threshold(
        matcher,
        calibration_input,
        thresholds,
        target_recall=target_recall,
        plateau_tolerance=plateau_tolerance,
        plateau_min_points=plateau_min_points,
    )
    final_threshold, selected, sensitivity, alternatives, plateau = result
    _write_calibration_outputs(
        matcher,
        output_dir,
        output_names,
        selected,
        sensitivity,
        alternatives,
        plateau,
        final_threshold,
        target_recall,
    )
    pd.DataFrame(_evaluate_holdout(matcher, holdout_labels, final_threshold)).to_csv(
        output_dir / output_names["holdout_metrics"], index=False
    )
    submission = _write_final_submission(
        matcher,
        output_dir,
        output_names,
        final_threshold,
    )
    print(
        {
            "folds": selected.check_fold.tolist(),
            "selected_thresholds": selected.selected_threshold.tolist(),
            "final_threshold": final_threshold,
            "rows": len(submission),
            "unique_items": submission.ITEM_ID.nunique(),
            "unmatched": int(submission.ITEM_ID.str.startswith(_unmatched_prefix()).sum()),
            "path": str(output_dir / output_names["submission"]),
        }
    )


def _path_argument(
    value: str | None,
    env_name: str,
    *,
    required: bool = True,
) -> Path | None:
    raw = value or os.environ.get(env_name)
    if not raw:
        if required:
            raise RuntimeError(
                f"{env_name} is required; pass the corresponding CLI argument"
            )
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = TRAIN_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and run final Rand Index SKU matching"
    )
    parser.add_argument(
        "--checkpoint",
        help="fine-tuned SentenceTransformer directory "
        "(or FINETUNED_CHECKPOINT)",
    )
    parser.add_argument(
        "--calibration-input",
        help="canonical-disjoint calibration CSV "
        "(or CALIBRATION_INPUT)",
    )
    parser.add_argument(
        "--holdout-input",
        help="frozen, untouched holdout CSV (or HOLDOUT_INPUT)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = _path_argument(args.checkpoint, "FINETUNED_CHECKPOINT")
    calibration_input = _path_argument(
        args.calibration_input,
        "CALIBRATION_INPUT",
    )
    holdout_input = _path_argument(
        args.holdout_input,
        "HOLDOUT_INPUT",
    )

    cfg = rand_matching_cfg()
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = TRAIN_ROOT / output_dir
    output_dir = output_dir.resolve()
    thresholds = np.round(
        np.arange(
            float(cfg["threshold_min"]),
            float(cfg["threshold_max"])
            + float(cfg["threshold_step"]) / 2,
            float(cfg["threshold_step"]),
        ),
        2,
    )

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            "FINETUNED_CHECKPOINT must point to a checkpoint directory: "
            f"{checkpoint}"
        )
    if not calibration_input.is_file():
        raise FileNotFoundError(
            "CALIBRATION_INPUT must point to a calibration CSV: "
            f"{calibration_input}"
        )
    if not holdout_input.is_file():
        raise FileNotFoundError(
            "HOLDOUT_INPUT must point to a frozen holdout CSV: "
            f"{holdout_input}"
        )
    matcher = RandMatcher(
        checkpoint,
        batch_size=int(cfg["batch_size"]),
        top_k=int(cfg["top_k"]),
    )
    output_names = {str(key): str(value) for key, value in cfg["outputs"].items()}
    write_outputs(
        matcher,
        calibration_input,
        holdout_input,
        thresholds,
        output_dir=output_dir,
        output_names=output_names,
        target_recall=float(cfg["target_recall"]),
        plateau_tolerance=float(cfg["plateau_tolerance"]),
        plateau_min_points=int(cfg["plateau_min_points"]),
    )


if __name__ == "__main__":
    main()
