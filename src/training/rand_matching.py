"""Final Rand Index-calibrated SKU-to-canonical matching.

This is the standalone version of ``notebooks/final_submission.ipynb``.
It calibrates a cosine threshold on canonical-disjoint folds, reports GTIN
sensitivity, and writes the final ``SKU_ID,ITEM_ID`` submission.

Architecture note: this is a SKU-to-canonical retrieval and *direct
assignment* lane. It embeds SKU and canonical records, retrieves canonical
top-K candidates, applies gates, and selects one canonical ID per SKU. Its
candidate bipartite graph is diagnostic-only; it does not create SKU-to-SKU
edges or use connected components as the prediction mechanism. Rand/ARI are
therefore evaluated over equivalence induced by the selected canonical IDs.

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
import hashlib
import json
import os
import unicodedata
from decimal import Decimal
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
)

from core.ann_config import load_ann_config
from core.attribute_conflicts import (
    canonical_attribute_info,
    conflict_columns,
    critical_attribute_evaluation,
    flavor_overlap_metrics,
    normalized_flavor_tokens,
    sku_attribute_info,
)
from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS
from core.common import (
    CONFIG_PATH,
    F,
    RESULTS,
    TRAIN_ROOT,
    TRAINING_CONFIG_PATH,
    canonical_records_frame,
    load_local_sentence_transformer,
    load_config,
    load_dataset_deduped,
    metadata_text,
    rand_matching_cfg,
    row_metadata_text,
)
from core.graph_diagnostics import (
    CANDIDATE_GATE_COLUMNS,
    CANDIDATE_GRAPH_DIAGNOSTIC_COLUMNS,
    PLAUSIBLE_GROUP_COUNT_COLUMN,
    candidate_graph_diagnostics,
    empty_candidate_graph_diagnostics,
)
from core.gtin import is_valid_gtin_checksum
from core.manifest import sha256_file
from core.schemas import GTIN_STATUSES, THRESHOLD_TIE_BREAK_CRITERIA
from core.structured_features import (
    append_text as append_structured_text,
    canonical_info as canonical_structured_info,
    fuse_numpy,
    sku_info as sku_structured_info,
    vector as structured_vector,
)
from core.unit_canonicalization import UNIT_CANONICALIZATION_VERSION
from pipeline import (
    canonical_evidence_text,
    canonical_model_text,
    clean_sku_text,
    load_canonical_map,
    strip_schema_words,
)
from training.hnsw_index import PersistentHnswIndex, normalize_embeddings


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
    "expected_group_count",
    PLAUSIBLE_GROUP_COUNT_COLUMN,
    "unmatched_skus",
    *CANDIDATE_GRAPH_DIAGNOSTIC_COLUMNS,
    "tp",
    "tn",
    "fp",
    "fn",
    "pair_count",
)
GATE_COLUMNS = (
    "gtin_gate",
    "attribute_gate",
    "brand_gate",
    "threshold_gate",
    "assignment_gate",
)
ASSIGNMENT_SORT_COLUMNS = (
    "SKU_ID",
    "exact_gtin",
    "score",
    "attribute_matches",
    "candidate_gtin",
)
ASSIGNMENT_SORT_ASCENDING = (True, False, False, False, True)
SOURCE_ROW_INDEX_COLUMN = "source_row_index"
INVALID_ID_SENTINELS = frozenset({"", "nan", "none", "null"})

# ---------------------------------------------------------------------------
# Deferral scope of ``targeted_veto_gate``.
#
# The gate has exactly three outcomes: hard reject on explicit conflict,
# deferral to the human-review route, or auto_merge. The deferral covers ONE
# case: the *decisive* identity evidence -- the pack count and the volume that
# the dial ``missing_pack_or_volume_route`` and its config comment both name --
# is unknown on a side the decision needs. It is not an evidence-completeness
# requirement.
#
# The other five critical dimensions (package_type, flavor, carbonation,
# sweetener, pulp) are veto-only: an explicit conflict rejects, and their
# ABSENCE is already handled by a dedicated, bounded lever
# (``confidence_penalty_mask``, "A field contributes only when it is absent on
# both endpoints. This avoids treating a one-sided parser miss as a conflict").
# Requiring all seven to be explicit instead made ``auto_merge`` unreachable:
# on the live 1,592-pair gate-positive population it left 1 pair (0.06%)
# auto-mergeable, because ``pulp_set`` is populated on only ~2% of canonical
# records.
#
# ``targeted_missing_attributes``/``..._count`` still report EVERY dimension,
# so the audit census that the diagnostics rely on is unchanged.
DEFERRAL_DIMENSIONS: tuple[str, ...] = ("pack", "volume")

# One row per *unordered* holdout SKU pair for which the truth and the
# predicted assignment disagree about whether the SKUs are the same item.
# These are deliberately pair-level, rather than candidate-level, records:
# Rand errors are errors in this equivalence relation.
PAIR_DISAGREEMENT_COLUMNS = (
    "sku_id_a",
    "sku_id_b",
    "disagreement_type",
    "true_group_id",
    "predicted_group_id",
    "pair_score",
    "edge_exists",
    "candidate_generated",
    "candidate_generation_status",
    "candidate_generation_source",
    "failure_stage",
    "error_classification",
    "attribute_gate_result",
    "component_size_true",
    "component_size_pred",
    "number_of_pairwise_errors_caused",
)


def _unmatched_prefix() -> str:
    return str(rand_matching_cfg()["unmatched_prefix"])


def _reconciliation_scope() -> str:
    return str(rand_matching_cfg()["threshold_reconciliation_scope"])


def _final_threshold_by_gtin_status() -> dict[str, float]:
    """Return the config-owned final threshold for every GTIN stratum."""
    configured = rand_matching_cfg()["threshold_by_gtin_status"]
    return {str(status): float(value) for status, value in configured.items()}


def _normalize_brand(value: object) -> str:
    """Normalize a brand for equality without treating missing as a value."""
    normalized = unicodedata.normalize("NFKC", metadata_text(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _brand_conflict(left: object, right: object) -> bool:
    """Return true only when both brands are present and disagree."""
    left_normalized = _normalize_brand(left)
    right_normalized = _normalize_brand(right)
    return bool(
        left_normalized and right_normalized and left_normalized != right_normalized
    )


def _sets_overlap_with_volume_tolerance(
    left: set[object],
    right: set[object],
    *,
    relative_tolerance: float,
    absolute_tolerance_ml: float,
) -> bool:
    """Return whether any canonical volume pair agrees within tolerance."""
    for left_value in left:
        for right_value in right:
            left_ml = float(left_value)
            right_ml = float(right_value)
            allowed = max(
                float(absolute_tolerance_ml),
                float(relative_tolerance) * max(abs(left_ml), abs(right_ml)),
            )
            if abs(left_ml - right_ml) <= allowed:
                return True
    return False


def targeted_veto_gate(
    sku_info: dict[str, object],
    candidate_info: dict[str, object],
    *,
    sku_brand: object,
    candidate_brand: object,
    exact_gtin: bool,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify a candidate using the shared critical-attribute contract.

    Any explicit conflict in any critical dimension hard-blocks a non-exact
    match. Unknown pack or volume evidence -- the decisive identity evidence
    the ``missing_pack_or_volume_route`` dial names -- routes to review, so it
    can never silently become an automatic graph edge. Absence of the
    veto-only categorical dimensions does not defer: it is reported in
    ``targeted_missing_attributes`` for audit and is bounded instead by
    ``confidence_penalty_mask`` (see ``DEFERRAL_DIMENSIONS``).
    ``targeted_pack_gate_pass`` is retained as the public boolean audit field
    and mirrors the full critical-attribute outcome computed below (all
    dimensions explicit and agreeing).
    """
    settings = config or rand_matching_cfg()["targeted_veto_gates"]
    left_pack = set(sku_info.get("pack") or set())
    right_pack = set(candidate_info.get("pack") or set())
    left_volume = set(sku_info.get("volume") or set())
    right_volume = set(candidate_info.get("volume") or set())
    left_package_type = set(sku_info.get("package_type") or set())
    right_package_type = set(candidate_info.get("package_type") or set())
    left_brand = _normalize_brand(sku_brand)
    right_brand = _normalize_brand(candidate_brand)
    relative_tolerance = float(settings["volume_relative_tolerance"])
    absolute_tolerance_ml = float(settings["volume_absolute_tolerance_ml"])

    critical = critical_attribute_evaluation(
        sku_info,
        candidate_info,
        volume_relative_tolerance=relative_tolerance,
        volume_absolute_tolerance_ml=absolute_tolerance_ml,
    )
    # SSOT (audit 2026-09-15): this audit column used to call a second
    # ``pack_gate`` that lived in core.attribute_conflicts and answered the
    # opposite way from the training-label gate for the same input. It is now
    # derived from the SAME evaluation object that already drives the veto and
    # defer routing below, so the reported boolean can no longer disagree with
    # the decision it claims to audit.
    pack_gate_pass = not critical["conflicts"] and not critical["unknown"]

    pack_conflict = bool(left_pack and right_pack and not (left_pack & right_pack))
    volume_conflict = bool(
        left_volume
        and right_volume
        and not _sets_overlap_with_volume_tolerance(
            left_volume,
            right_volume,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_ml=absolute_tolerance_ml,
        )
    )
    brand_conflict = bool(left_brand and right_brand and left_brand != right_brand)
    package_type_conflict = bool(
        left_package_type
        and right_package_type
        and not (left_package_type & right_package_type)
    )
    missing: list[str] = []
    deferral_missing: list[str] = []
    for dimension in CRITICAL_ATTRIBUTE_DIMENSIONS:
        key = "flavor_set" if dimension == "flavor" else dimension
        decisive = dimension in DEFERRAL_DIMENSIONS
        for side, info in (("a", sku_info), ("b", candidate_info)):
            if info.get(key):
                continue
            missing.append(f"{dimension}_{side}")
            if decisive:
                deferral_missing.append(f"{dimension}_{side}")
    common = {
        "targeted_pack_conflict": int(pack_conflict),
        "targeted_volume_conflict": int(volume_conflict),
        "targeted_brand_conflict": int(brand_conflict),
        "targeted_package_type_conflict": int(package_type_conflict),
        "targeted_missing_attributes": ",".join(missing),
        "targeted_missing_attribute_count": len(missing),
        "targeted_pack_a": json.dumps(sorted(left_pack)),
        "targeted_pack_b": json.dumps(sorted(right_pack)),
        "targeted_volume_ml_a": json.dumps(sorted(left_volume)),
        "targeted_volume_ml_b": json.dumps(sorted(right_volume)),
        "targeted_package_type_a": json.dumps(sorted(left_package_type)),
        "targeted_package_type_b": json.dumps(sorted(right_package_type)),
        "targeted_pack_gate_pass": int(pack_gate_pass),
        "targeted_critical_conflicts": ",".join(critical["conflicts"]),
        "targeted_critical_agreements": ",".join(critical["agreements"]),
        "targeted_brand_a": left_brand,
        "targeted_brand_b": right_brand,
        "targeted_volume_relative_tolerance": relative_tolerance,
        "targeted_volume_absolute_tolerance_ml": absolute_tolerance_ml,
    }
    if exact_gtin and bool(settings["preserve_exact_gtin"]):
        return common | {
            "targeted_gate_decision": "exact_gtin_lock",
            "targeted_gate_reason": "exact_gtin_preserved",
            "targeted_gate_route": "auto_merge",
        }
    if not bool(settings["enabled"]):
        return common | {
            "targeted_gate_decision": "allow",
            "targeted_gate_reason": "disabled",
            "targeted_gate_route": "auto_merge",
        }

    veto_reasons: list[str] = [
        f"{dimension}_mismatch" for dimension in critical["conflicts"]
    ]
    if brand_conflict and bool(settings["brand_mismatch_veto"]):
        veto_reasons.append("brand_mismatch")
    if veto_reasons:
        return common | {
            "targeted_gate_decision": "veto",
            "targeted_gate_reason": "+".join(veto_reasons),
            "targeted_gate_route": "reject",
        }
    if deferral_missing:
        return common | {
            "targeted_gate_decision": "defer",
            "targeted_gate_reason": "missing_pack_or_volume:"
            + ",".join(deferral_missing),
            "targeted_gate_route": str(settings["missing_pack_or_volume_route"]),
        }
    return common | {
        "targeted_gate_decision": "allow",
        "targeted_gate_reason": "attributes_compatible",
        "targeted_gate_route": "auto_merge",
    }


def confidence_penalty_mask(
    sku_info: dict[str, object],
    candidate_info: dict[str, object],
    *,
    exact_gtin: bool,
    config: dict[str, object] | None = None,
) -> tuple[float, str]:
    """Return a monotonic score penalty for jointly missing evidence.

    A field contributes only when it is absent on *both* endpoints. This
    avoids treating a one-sided parser miss as a conflict. Exact identities
    remain untouched by default, and the mask never raises a score.
    """
    settings = config or rand_matching_cfg()["confidence_penalty_mask"]
    if not bool(settings["enabled"]):
        return 0.0, "disabled"
    if exact_gtin and bool(settings["preserve_exact_gtin"]):
        return 0.0, "exact_gtin_preserved"
    attributes = [str(value) for value in settings["critical_attributes"]]
    jointly_missing = _jointly_missing_attributes(sku_info, candidate_info, attributes)
    if len(jointly_missing) < int(settings["minimum_joint_missing"]):
        return 0.0, "sufficient_attribute_evidence"
    penalty = min(
        float(settings["max_penalty"]),
        len(jointly_missing) * float(settings["penalty_per_joint_missing"]),
    )
    return penalty, "jointly_missing:" + ",".join(jointly_missing)


def _jointly_missing_attributes(
    sku_info: dict[str, object],
    candidate_info: dict[str, object],
    attributes: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Name shared missing evidence without filtering the candidate pair."""
    fields = attributes or ("volume", "pack", "flavor")
    return [
        attribute
        for attribute in fields
        if not sku_info.get(attribute) and not candidate_info.get(attribute)
    ]


def flavor_overlap_penalty(
    sku_info: dict[str, object],
    candidate_info: dict[str, object],
    *,
    exact_gtin: bool,
    config: dict[str, object] | None = None,
) -> tuple[float, float, float, str]:
    """Return Jaccard, overlap, bounded penalty, and an audit reason.

    Missing flavor is handled by :func:`confidence_penalty_mask`, not treated
    as a mismatch here.  The overlap coefficient lets a specific multi-token
    flavor (``apple lemon``) match its shared canonical flavor (``apple``),
    while Jaccard is retained in the trace for diagnosis.
    """
    settings = config or rand_matching_cfg()["flavor_overlap_penalty"]
    left = sku_info.get("flavor")
    right = candidate_info.get("flavor")
    jaccard, overlap = flavor_overlap_metrics(left, right)
    if not bool(settings["enabled"]):
        return jaccard, overlap, 0.0, "disabled"
    if exact_gtin and bool(settings["preserve_exact_gtin"]):
        return jaccard, overlap, 0.0, "exact_gtin_preserved"
    if not normalized_flavor_tokens(left) or not normalized_flavor_tokens(right):
        return jaccard, overlap, 0.0, "insufficient_flavor_evidence"
    minimum = float(settings["minimum_overlap"])
    if overlap >= minimum:
        return jaccard, overlap, 0.0, "sufficient_flavor_overlap"
    severity = (minimum - overlap) / minimum
    max_penalty = float(settings["max_penalty"])
    penalty = min(max_penalty, severity * max_penalty)
    return jaccard, overlap, penalty, "low_flavor_overlap"


def _threshold_selection_key(row: dict[str, float | int]) -> tuple[float, ...]:
    """Encode the configured threshold tie-break policy for ``max``."""
    values = {
        "rand_index": float(row["rand_index"]),
        "fewest_unmatched_skus": -float(row["unmatched_skus"]),
        "lowest_threshold": -float(row["threshold"]),
    }
    policy = rand_matching_cfg()["threshold_tie_break"]
    if tuple(sorted(policy)) != tuple(sorted(THRESHOLD_TIE_BREAK_CRITERIA)):
        raise ValueError(
            "rand_matching.threshold_tie_break contains an unsupported criterion"
        )
    return tuple(values[name] for name in policy)


def fit_threshold_key(row: dict[str, float | int]) -> tuple[float, ...]:
    """Public notebook/shared-lane name for the config-driven fit rule."""
    return _threshold_selection_key(row)


def _fit_recall_column(target_recall: float) -> str:
    return f"fit_threshold_at_{target_recall:.0%}_recall"


def _threshold_grid(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> np.ndarray:
    """Build the configured threshold grid without imposing display precision."""
    minimum = Decimal(str(threshold_min))
    maximum = Decimal(str(threshold_max))
    step = Decimal(str(threshold_step))
    if step <= 0 or minimum > maximum:
        raise ValueError("threshold grid requires min <= max and step > 0")
    count = int((maximum - minimum) // step)
    return np.asarray(
        [float(minimum + step * index) for index in range(count + 1)],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Pydantic output-contract models (column-set validation at write time)
# ---------------------------------------------------------------------------


class _FileProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    rows: int | None = Field(default=None, ge=0)


class _SubmissionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_deduped: _FileProvenance
    canonical_records: _FileProvenance
    calibration_input: _FileProvenance
    holdout_input: _FileProvenance
    paths_config: _FileProvenance
    training_config: _FileProvenance
    checkpoint: _FileProvenance
    final_threshold: float
    threshold_by_gtin_status: dict[str, float]
    brand_conflict_veto: bool
    confidence_penalty_mask: dict[str, object]
    flavor_overlap_penalty: dict[str, object]
    targeted_veto_gates: dict[str, object]
    unmatched_prefix: str
    rows: int
    unique_items: int
    unmatched: int
    calibration_folds: tuple[str, ...] = Field(min_length=2)
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


class _MetricColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_columns: frozenset[str] = Field(min_length=1)
    optional_columns: frozenset[str] = frozenset()

    def validate_frame(self, frame: pd.DataFrame, label: str) -> None:
        actual = set(frame.columns)
        missing = sorted(self.required_columns - actual)
        if missing:
            raise ValueError(f"{label} metric contract violated: missing={missing}")
        unexpected = sorted(actual - self.required_columns - self.optional_columns)
        if unexpected:
            raise ValueError(
                f"{label} metric contract violated: unexpected={unexpected}"
            )


_SUBMISSION_COLUMNS_SPEC = _SubmissionColumnSpec(columns=("SKU_ID", "ITEM_ID"))
_DIAGNOSTICS_COLUMNS_SPEC = _DiagnosticsColumnSpec(
    expected_columns=frozenset(
        {
            # candidate trace (_candidate_row); gate columns are owned by the
            # shared graph-diagnostic contract below.
            *CANDIDATE_GATE_COLUMNS,
            "sku_gtin",
            "sku_gtin_present",
            "sku_gtin_valid",
            "candidate_rank",
            "retrieval_source",
            "raw_score",
            "confidence_penalty",
            "confidence_penalty_reason",
            "jointly_missing_attributes",
            "jointly_missing_attribute_count",
            "flavor_jaccard",
            "flavor_overlap",
            "flavor_penalty",
            "flavor_penalty_reason",
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
            "sku_package_type",
            "sku_flavor",
            "sku_carbonation",
            "sku_sweetener",
            "sku_pulp",
            "sku_title_present",
            "sku_attributes_present",
            "sku_brand_present",
            "sku_country_present",
            "sku_category_present",
            "sku_category_path_present",
            "sku_retailer_present",
            "sku_volume_present",
            "sku_pack_present",
            "sku_package_type_present",
            "sku_flavor_present",
            "sku_carbonation_present",
            "sku_sweetener_present",
            "sku_pulp_present",
            "source_row_index",
            "true_item_id",
            "calibration_fold",
            "candidate_text",
            "candidate_brand",
            "candidate_volume",
            "candidate_pack",
            "candidate_package_type",
            "candidate_flavor",
            "candidate_carbonation",
            "candidate_sweetener",
            "candidate_pulp",
            "candidate_brand_present",
            "candidate_volume_present",
            "candidate_pack_present",
            "candidate_package_type_present",
            "candidate_flavor_present",
            "candidate_carbonation_present",
            "candidate_sweetener_present",
            "candidate_pulp_present",
            "brand_conflict",
            "attribute_conflict_type",
            "attribute_matches",
            "targeted_pack_conflict",
            "targeted_volume_conflict",
            "targeted_brand_conflict",
            "targeted_package_type_conflict",
            "targeted_pack_gate_pass",
            "targeted_critical_conflicts",
            "targeted_critical_agreements",
            "targeted_missing_attributes",
            "targeted_missing_attribute_count",
            "targeted_pack_a",
            "targeted_pack_b",
            "targeted_volume_ml_a",
            "targeted_volume_ml_b",
            "targeted_package_type_a",
            "targeted_package_type_b",
            "targeted_brand_a",
            "targeted_brand_b",
            "targeted_volume_relative_tolerance",
            "targeted_volume_absolute_tolerance_ml",
            "targeted_gate_decision",
            "targeted_gate_reason",
            "targeted_gate_route",
            # annotation (_annotate_candidates)
            "effective_threshold",
            "brand_compatible",
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
            # evaluation audit context
            "evaluation_partition",
            "evaluation_fold",
            "evaluation_threshold",
            "predicted_ITEM_ID",
            "true_candidate_retrieved",
            "true_candidate_accepted",
            "prediction_correct",
            "error_type",
        }
    )
)
_METRIC_COLUMNS_SPEC = _MetricColumnSpec(
    required_columns=frozenset(METRIC_COLUMNS),
    optional_columns=frozenset(
        {
            "check_fold",
            "threshold",
            "gtin_status",
            "selection_method",
            "sensitivity_reason",
            "reconciliation_scope",
        }
    ),
)


def _field_present(row: pd.Series, primary: str, alias: str | None = None) -> int:
    """Return a source-field presence flag without changing source values."""
    if primary in row.index:
        value = row[primary]
    elif alias is not None and alias in row.index:
        value = row[alias]
    else:
        return 0
    return int(bool(metadata_text(value).strip()))


def _value_present(value: object) -> int:
    return int(bool(metadata_text(value).strip()))


def _ensure_source_row_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the source dataset row identity across dataframe operations."""
    result = frame.copy()
    if SOURCE_ROW_INDEX_COLUMN in result.columns:
        raw_identity = result[SOURCE_ROW_INDEX_COLUMN]
        if raw_identity.isna().any():
            raise ValueError("source_row_index contains missing values")
        identity = raw_identity.astype(str).str.strip()
    else:
        identity = pd.Series(
            result.index.astype(str),
            index=result.index,
            name=SOURCE_ROW_INDEX_COLUMN,
        )
    invalid = identity.eq("") | identity.str.lower().isin(INVALID_ID_SENTINELS)
    if invalid.any():
        raise ValueError("source_row_index contains blank or sentinel values")
    if identity.duplicated().any():
        raise ValueError("source_row_index must identify one source row uniquely")
    result[SOURCE_ROW_INDEX_COLUMN] = identity
    return result


def trusted_gtin(value: object) -> str:
    """Return a GTIN only when it is a valid identity signal."""
    text = metadata_text(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text if text and is_valid_gtin_checksum(text) else ""


def gtin_status(sku_gtin: object, candidate_gtin: object) -> str:
    """Apply the shared four-state GTIN gate taxonomy."""
    left = trusted_gtin(sku_gtin)
    right = trusted_gtin(candidate_gtin)
    if not left and not right:
        return "both_missing"
    if not left or not right:
        return "one_missing"
    return "both_equal" if left == right else "different"


def candidate_gate_fields(
    row: pd.Series,
    sku_info: dict,
    candidate_gtin: str,
    candidate_record: dict[str, object],
    score: float,
    *,
    sku_id: str,
    source_row_index: str,
    candidate_rank: int | None = None,
    retrieval_source: str = "unknown",
) -> dict[str, object]:
    """Build the shared candidate gate record used by all matching lanes."""
    candidate_info = canonical_attribute_info(candidate_record)
    sku_gtin = metadata_text(row_metadata_text(row, "barcode", "gtin")).strip()
    status = gtin_status(sku_gtin, candidate_gtin)
    exact = int(status == "both_equal")
    targeted_gate = targeted_veto_gate(
        sku_info,
        candidate_info,
        sku_brand=row_metadata_text(row, "brand"),
        candidate_brand=candidate_record.get("mode_brand"),
        exact_gtin=bool(exact),
    )
    rules = conflict_columns(sku_info, candidate_info)
    # The ANN assignment lane owns a configurable volume tolerance.  Replace
    # the generic exact-set volume result with the targeted gate result while
    # retaining the shared flavor classification and exact pack semantics.
    targeted_settings = rand_matching_cfg()["targeted_veto_gates"]
    if bool(targeted_settings["enabled"]):
        rules["volume_conflict"] = int(
            bool(targeted_settings["volume_mismatch_veto"])
            and bool(targeted_gate["targeted_volume_conflict"])
        )
        rules["pack_conflict"] = int(
            bool(targeted_settings["pack_mismatch_veto"])
            and bool(targeted_gate["targeted_pack_conflict"])
        )
        rules["package_type_conflict"] = int(
            bool(targeted_settings["package_type_mismatch_veto"])
            and bool(targeted_gate["targeted_package_type_conflict"])
        )
    conflict_names = [
        name
        for name in CRITICAL_ATTRIBUTE_DIMENSIONS
        if bool(rules[f"{name}_conflict"])
    ]
    rules["attribute_conflict_type"] = (
        "+".join(conflict_names) if conflict_names else "none"
    )
    confidence_penalty, confidence_penalty_reason = confidence_penalty_mask(
        sku_info,
        candidate_info,
        exact_gtin=bool(exact),
    )
    flavor_jaccard, flavor_overlap, flavor_penalty, flavor_penalty_reason = (
        flavor_overlap_penalty(
            sku_info,
            candidate_info,
            exact_gtin=bool(exact),
        )
    )
    adjusted_score = max(
        -1.0,
        float(score) - confidence_penalty - flavor_penalty,
    )
    jointly_missing = _jointly_missing_attributes(sku_info, candidate_info)
    brand_conflict = int(
        bool(rand_matching_cfg()["brand_conflict_veto"])
        and bool(targeted_gate["targeted_brand_conflict"])
        and (
            not bool(targeted_settings["enabled"])
            or bool(targeted_settings["brand_mismatch_veto"])
        )
    )
    gate_reason = (
        str(targeted_gate["targeted_gate_reason"])
        if targeted_gate["targeted_gate_route"] != "auto_merge"
        else "different_gtin_thresholded"
        if status == "different"
        else "exact_gtin"
        if exact
        else "brand_conflict"
        if brand_conflict
        else "attribute_conflict"
        if rules["attribute_conflict_type"] != "none"
        else "cosine_candidate"
    )
    return {
        "SKU_ID": sku_id,
        "sku_gtin": sku_gtin,
        "sku_gtin_present": _field_present(row, "barcode", "gtin"),
        "sku_gtin_valid": int(bool(trusted_gtin(sku_gtin))),
        "candidate_gtin": candidate_gtin,
        "candidate_rank": candidate_rank,
        "retrieval_source": retrieval_source,
        "raw_score": float(score),
        "confidence_penalty": confidence_penalty,
        "confidence_penalty_reason": confidence_penalty_reason,
        "jointly_missing_attributes": ",".join(jointly_missing),
        "jointly_missing_attribute_count": len(jointly_missing),
        "flavor_jaccard": flavor_jaccard,
        "flavor_overlap": flavor_overlap,
        "flavor_penalty": flavor_penalty,
        "flavor_penalty_reason": flavor_penalty_reason,
        "score": adjusted_score,
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
        "sku_package_type": json.dumps(
            sorted(sku_info.get("package_type") or set())
        ),
        "sku_flavor": str(sku_info["flavor"]),
        "sku_carbonation": json.dumps(sorted(sku_info.get("carbonation") or set())),
        "sku_sweetener": json.dumps(sorted(sku_info.get("sweetener") or set())),
        "sku_pulp": json.dumps(sorted(sku_info.get("pulp") or set())),
        "sku_title_present": _field_present(row, "title"),
        "sku_attributes_present": _field_present(row, "attributes", "attr"),
        "sku_brand_present": _field_present(row, "brand"),
        "sku_country_present": _field_present(row, "country"),
        "sku_category_present": _field_present(row, "category"),
        "sku_category_path_present": _field_present(row, "category_path"),
        "sku_retailer_present": _field_present(row, "retailer"),
        "sku_volume_present": int(bool(sku_info["volume"])),
        "sku_pack_present": int(bool(sku_info["pack"])),
        "sku_package_type_present": int(bool(sku_info.get("package_type"))),
        "sku_flavor_present": int(bool(sku_info["flavor"])),
        "sku_carbonation_present": int(bool(sku_info.get("carbonation"))),
        "sku_sweetener_present": int(bool(sku_info.get("sweetener"))),
        "sku_pulp_present": int(bool(sku_info.get("pulp"))),
        "source_row_index": source_row_index,
        "candidate_text": metadata_text(candidate_record.get("canonical")),
        "candidate_brand": metadata_text(candidate_record.get("mode_brand")),
        "brand_conflict": brand_conflict,
        "candidate_volume": json.dumps(sorted(candidate_info["volume"])),
        "candidate_pack": json.dumps(sorted(candidate_info["pack"])),
        "candidate_package_type": json.dumps(sorted(candidate_info["package_type"])),
        "candidate_flavor": str(candidate_info["flavor"]),
        "candidate_carbonation": json.dumps(sorted(candidate_info.get("carbonation") or set())),
        "candidate_sweetener": json.dumps(sorted(candidate_info.get("sweetener") or set())),
        "candidate_pulp": json.dumps(sorted(candidate_info.get("pulp") or set())),
        "candidate_brand_present": _value_present(candidate_record.get("mode_brand")),
        "candidate_volume_present": int(bool(candidate_info["volume"])),
        "candidate_pack_present": int(bool(candidate_info["pack"])),
        "candidate_package_type_present": int(bool(candidate_info["package_type"])),
        "candidate_flavor_present": int(bool(candidate_info["flavor"])),
        "candidate_carbonation_present": int(bool(candidate_info.get("carbonation"))),
        "candidate_sweetener_present": int(bool(candidate_info.get("sweetener"))),
        "candidate_pulp_present": int(bool(candidate_info.get("pulp"))),
        "rule_ok": int(rules["attribute_conflict_type"] == "none"),
        "attribute_conflict_type": str(rules["attribute_conflict_type"]),
        "attribute_matches": int(
            sum(
                not rules[key]
                for key in (
                    "volume_conflict",
                    "pack_conflict",
                    "package_type_conflict",
                    "flavor_conflict",
                    "carbonation_conflict",
                    "sweetener_conflict",
                    "pulp_conflict",
                )
            )
        ),
        **targeted_gate,
    }


class RandMatcher:
    """Encode canonical items and score SKU candidates against them."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        batch_size: int,
        top_k: int,
        ann_index_dir: Path | None = None,
        rebuild_ann_index: bool = False,
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

        records = canonical_records_frame()
        if records["gtin"].duplicated().any():
            raise RuntimeError("canonical_records.csv contains duplicate GTIN rows")
        self.record_map = {
            str(row["gtin"]): row.to_dict() for _, row in records.iterrows()
        }
        missing_records = sorted(set(self.item_ids) - set(self.record_map))
        if missing_records:
            raise RuntimeError(
                "canonical metadata is missing for canonical GTINs: "
                f"{missing_records[:10]}"
                + (" ..." if len(missing_records) > 10 else "")
            )

        self.model = load_local_sentence_transformer(
            str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.structured_enabled = bool(self.structured_config["enabled"])
        self.structured_text = self.structured_enabled and bool(
            self.structured_config["append_to_text"]
        )
        self.structured_weight = (
            float(self.structured_config["embedding_weight"])
            if self.structured_enabled and bool(self.structured_config["feed_to_loss"])
            else 0.0
        )
        self.preprocessing_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "structured_features": self.structured_config,
                    "unit_canonicalization": UNIT_CANONICALIZATION_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        item_infos = [
            canonical_structured_info(self.record_map[item_id])
            if self.structured_enabled
            else {"volume": set(), "pack": set()}
            for item_id in self.item_ids
        ]
        item_texts = [
            append_structured_text(
                strip_schema_words(canonical_model_text(" ".join((
                    self.canonical[item_id],
                    str(self.record_map[item_id].get("mode_brand", "")),
                    str(self.record_map[item_id].get("mode_type", "")),
                    canonical_evidence_text(
                        self.record_map[item_id].get("description_evidence", "")
                    ),
                    canonical_evidence_text(
                        self.record_map[item_id].get("breadcrumb_evidence", "")
                    ),
                )))),
                info,
                enabled=self.structured_text,
            )
            for item_id, info in zip(self.item_ids, item_infos, strict=True)
        ]
        item_features = np.asarray(
            [self._structured_vector(info) for info in item_infos],
            dtype=np.float32,
        )
        ann_settings = load_ann_config()
        ann_cfg = ann_settings.index
        configured_output = Path(ann_cfg.output_dir)
        if not configured_output.is_absolute():
            configured_output = TRAIN_ROOT / configured_output
        self.ann_index = PersistentHnswIndex(
            ann_index_dir or configured_output,
            ef_construction=ann_cfg.ef_construction,
            M=ann_cfg.M,
            ef_search=ann_cfg.ef_search,
            space=ann_cfg.space,
        )
        model_name = ann_settings.embedding.model
        if not rebuild_ann_index:
            try:
                self.ann_index.load(
                    ids=self.item_ids,
                    checkpoint=self.checkpoint,
                    model_name=model_name,
                    preprocessing_fingerprint=self.preprocessing_fingerprint,
                )
                self.item_embeddings = self.ann_index.embeddings
                if self.item_embeddings is None:
                    raise ValueError("persisted HNSW index loaded without embeddings")
                print(f"loaded persisted HNSW index from {self.ann_index.output_dir}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"persisted HNSW index needs rebuild: {exc}")
                rebuild_ann_index = True
        if rebuild_ann_index:
            item_embeddings = self.model.encode(
                item_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.item_embeddings = normalize_embeddings(
                fuse_numpy(
                    item_embeddings,
                    item_features,
                    self.structured_weight,
                )
            )
            metadata = self.ann_index.build(
                self.item_embeddings,
                self.item_ids,
                checkpoint=self.checkpoint,
                model_name=model_name,
                preprocessing_fingerprint=self.preprocessing_fingerprint,
            )
            print(
                f"built HNSW index count={metadata['count']:,} "
                f"dim={metadata['dim']} at {self.ann_index.output_dir}"
            )

        print(f"loaded {len(self.item_ids):,} canonical items from {self.checkpoint}")

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
        return trusted_gtin(value)

    def _text_and_info(
        self, frame: pd.DataFrame
    ) -> tuple[list[str], list[dict], list[dict]]:
        # Keep gate attributes confidence-aware: an absent pack count remains
        # unknown to the gate.  The model-side structured channel has a
        # separate source representation whose missing pack count defaults to
        # the canonical singleton token pack_qty_1.
        gate_infos = [
            sku_attribute_info(
                row_metadata_text(row, "title"),
                row_metadata_text(row, "attributes", "attr"),
            )
            for _, row in frame.iterrows()
        ]
        model_infos = [
            sku_structured_info(
                row_metadata_text(row, "title"),
                row_metadata_text(row, "attributes", "attr"),
            )
            if self.structured_enabled
            else {"volume": set(), "pack": set()}
            for _, row in frame.iterrows()
        ]
        texts = [
            append_structured_text(
                strip_schema_words(
                    clean_sku_text(
                        row_metadata_text(row, "title"),
                        row_metadata_text(row, "attributes", "attr"),
                        row_metadata_text(row, "brand"),
                        row_metadata_text(row, "description", "description_short_eng"),
                        row_metadata_text(row, "category", "category_path"),
                        row_metadata_text(row, "category_path", "breadcrumbs_eng"),
                    )
                ),
                info,
                enabled=self.structured_text,
            )
            for (_, row), info in zip(frame.iterrows(), model_infos, strict=True)
        ]
        return texts, gate_infos, model_infos

    _INVALID_SKU_SENTINELS = INVALID_ID_SENTINELS

    @staticmethod
    def _normalise_skus(skus: pd.DataFrame) -> pd.DataFrame:
        frame = skus.copy()
        if "SKU_ID" not in frame and "product_id" in frame:
            frame = frame.rename(columns={"product_id": "SKU_ID"})
        if "SKU_ID" not in frame:
            raise ValueError("input must contain SKU_ID or product_id")
        frame = _ensure_source_row_identity(frame)
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
    ) -> dict[int, tuple[int | None, str]]:
        indexes = {
            int(candidate_index): (rank, "hnsw_semantic_top_k")
            for rank, candidate_index in enumerate(hits, start=1)
        }
        trusted_gtin = self._trusted_gtin(sku_gtin)
        if trusted_gtin in self.item_index:
            index = self.item_index[trusted_gtin]
            if index in indexes:
                rank, _ = indexes[index]
                indexes[index] = (rank, "hnsw_semantic_top_k+exact_gtin")
            else:
                indexes[index] = (None, "exact_gtin_rescue")
        return indexes

    def _candidate_row(
        self,
        row: pd.Series,
        sku_info: dict,
        embedding: np.ndarray,
        candidate_index: int,
        candidate_rank: int | None,
        retrieval_source: str,
    ) -> dict[str, object]:
        sku_gtin = self._gtin(row_metadata_text(row, "barcode", "gtin"))
        candidate_gtin = self.item_ids[candidate_index]
        candidate_record = self.record_map[candidate_gtin]
        base = candidate_gate_fields(
            row,
            sku_info,
            candidate_gtin,
            candidate_record,
            float(np.dot(embedding, self.item_embeddings[candidate_index])),
            sku_id=str(row["SKU_ID"]),
            source_row_index=str(row[SOURCE_ROW_INDEX_COLUMN]),
            candidate_rank=candidate_rank,
            retrieval_source=retrieval_source,
        )
        base.update(
            {
                "true_item_id": row_metadata_text(row, "true_item_id"),
                "calibration_fold": row_metadata_text(row, "calibration_fold"),
            }
        )
        return base

    def gtin_status(self, sku_gtin: object, candidate_gtin: object) -> str:
        return gtin_status(sku_gtin, candidate_gtin)

    def score_candidates(
        self,
        skus: pd.DataFrame,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        top_k = self.top_k if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be positive")
        frame = self._normalise_skus(skus)
        texts, gate_infos, model_infos = self._text_and_info(frame)
        embeddings = self._encode_skus(texts, model_infos)
        hit_labels, _ = self.ann_index.query(embeddings, top_k=top_k)

        rows: list[dict[str, object]] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            sku_gtin = self._gtin(row_metadata_text(row, "barcode", "gtin"))
            candidate_indexes = self._candidate_indexes(
                hit_labels[position].tolist(), sku_gtin
            )
            for index in sorted(candidate_indexes):
                candidate_rank, retrieval_source = candidate_indexes[index]
                rows.append(
                    self._candidate_row(
                        row,
                        gate_infos[position],
                        embeddings[position],
                        index,
                        candidate_rank,
                        retrieval_source,
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


def _annotate_candidates(
    candidates: pd.DataFrame,
    threshold: float,
    *,
    threshold_by_gtin_status: dict[str, float] | None = None,
) -> pd.DataFrame:
    frame = candidates.copy()
    if "brand_conflict" not in frame.columns:
        raise ValueError("candidate trace is missing required brand_conflict")
    # A different valid GTIN is an explicit threshold stratum, not a hard
    # veto. It still has to pass score, brand, and attribute gates. Exact GTIN
    # remains locked; missing-GTIN strata remain thresholded as before.
    frame["gtin_compatible"] = True
    if threshold_by_gtin_status is None:
        effective_threshold = pd.Series(
            float(threshold), index=frame.index, dtype=float
        )
    else:
        effective_threshold = frame["gtin_status"].map(threshold_by_gtin_status)
        if effective_threshold.isna().any():
            missing_statuses = sorted(
                frame.loc[effective_threshold.isna(), "gtin_status"].unique()
            )
            raise ValueError(
                "threshold_by_gtin_status is missing GTIN status values: "
                f"{missing_statuses}"
            )
        effective_threshold = effective_threshold.astype(float)
    frame["effective_threshold"] = effective_threshold
    frame["score_pass"] = frame["score"] >= frame["effective_threshold"]
    frame["brand_compatible"] = frame["brand_conflict"].eq(0) | frame[
        "exact_gtin"
    ].astype(bool)
    targeted_route = (
        frame["targeted_gate_route"].astype(str)
        if "targeted_gate_route" in frame.columns
        else pd.Series("auto_merge", index=frame.index, dtype=str)
    )
    targeted_auto_merge = targeted_route.eq("auto_merge")
    frame["accepted"] = frame["gtin_compatible"] & (
        frame["exact_gtin"].astype(bool)
        | (
            targeted_auto_merge
            & frame["rule_ok"].astype(bool)
            & frame["brand_compatible"]
            & frame["score_pass"]
        )
    )
    frame["gtin_gate"] = np.select(
        [
            frame["gtin_status"].eq("different"),
            frame["exact_gtin"].astype(bool),
        ],
        ["threshold", "lock"],
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
    frame["brand_gate"] = np.select(
        [
            frame["exact_gtin"].astype(bool),
            frame["brand_conflict"].astype(bool),
        ],
        ["exact_gtin_lock", "veto"],
        default="allow_equal_or_unknown",
    )
    frame["threshold_gate"] = np.select(
        [
            frame["exact_gtin"].astype(bool),
            frame["score_pass"],
        ],
        ["bypass_exact_gtin", "pass"],
        default="fail",
    )
    frame["assignment_gate"] = np.select(
        [frame["accepted"], targeted_route.eq("human_review")],
        ["accepted_candidate", "human_review_candidate"],
        default="rejected_candidate",
    )
    frame["rejection_reason"] = np.select(
        [
            ~frame["gtin_compatible"],
            frame["exact_gtin"].astype(bool),
            targeted_route.eq("reject"),
            targeted_route.eq("human_review"),
            frame["brand_conflict"].astype(bool),
            ~frame["rule_ok"].astype(bool),
            ~frame["score_pass"],
        ],
        [
            "gtin_conflict",
            "exact_gtin_lock",
            "targeted_attribute_veto",
            "human_review_missing_pack_or_volume",
            "brand_conflict",
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
    *,
    threshold_by_gtin_status: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return (
            pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            candidates.copy(),
        )
    frame = _annotate_candidates(
        candidates,
        threshold,
        threshold_by_gtin_status=threshold_by_gtin_status,
    )
    accepted = frame[frame["accepted"]].sort_values(
        list(ASSIGNMENT_SORT_COLUMNS),
        ascending=list(ASSIGNMENT_SORT_ASCENDING),
        kind="mergesort",
    )
    selected = accepted.drop_duplicates("SKU_ID", keep="first").copy()
    selected_keys = selected[["SKU_ID", "candidate_gtin"]].assign(selected=1)
    trace = frame.copy()
    trace_keys = pd.MultiIndex.from_frame(trace[["SKU_ID", "candidate_gtin"]])
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
    best = selected[["SKU_ID", "candidate_gtin", "score", "gtin_status"]].rename(
        columns={"candidate_gtin": "ITEM_ID"}
    )
    best = best.loc[:, list(ASSIGNMENT_COLUMNS)]
    all_skus = candidates[["SKU_ID"]].drop_duplicates()
    output = all_skus.merge(best, on="SKU_ID", how="left")
    prefix = _unmatched_prefix()
    output["ITEM_ID"] = output["ITEM_ID"].fillna(prefix + output["SKU_ID"].astype(str))
    return output, trace


def _merge_audit_context(
    candidates: pd.DataFrame,
    trace: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = trace.merge(
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
    if len(diagnostics) != len(trace):
        raise RuntimeError("audit trace merge changed the candidate population")
    candidate_ids = set(candidates["SKU_ID"].astype(str))
    prediction_ids = set(predictions["SKU_ID"].astype(str))
    if candidate_ids != prediction_ids:
        raise RuntimeError(
            "audit candidates and predictions disagree on SKU population: "
            f"missing={sorted(candidate_ids - prediction_ids)[:10]}, "
            f"unexpected={sorted(prediction_ids - candidate_ids)[:10]}"
        )
    return diagnostics


def _truth_audit_context(
    candidates: pd.DataFrame,
    trace: pd.DataFrame,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Build one labeled audit summary per SKU for trace enrichment."""
    truth_frame = truth[["SKU_ID", "true_item_id"]].copy()
    if truth_frame["SKU_ID"].duplicated().any():
        raise RuntimeError("audit truth contains duplicate SKU_ID values")
    expected_ids = set(truth_frame["SKU_ID"].astype(str))
    actual_ids = set(predictions["SKU_ID"].astype(str))
    if expected_ids != actual_ids:
        raise RuntimeError(
            "audit truth and predictions disagree on SKU population: "
            f"missing={sorted(expected_ids - actual_ids)[:10]}, "
            f"unexpected={sorted(actual_ids - expected_ids)[:10]}"
        )

    true_item_by_sku = truth_frame.set_index("SKU_ID")["true_item_id"]
    candidate_truth = candidates.assign(
        true_item_id=candidates["SKU_ID"].map(true_item_by_sku)
    )
    retrieved_by_sku = (
        candidate_truth.assign(
            is_true_candidate=lambda frame: (
                frame["candidate_gtin"]
                .astype(str)
                .eq(frame["true_item_id"].astype(str))
            )
        )
        .groupby("SKU_ID")["is_true_candidate"]
        .any()
    )
    trace_truth = trace.assign(true_item_id=trace["SKU_ID"].map(true_item_by_sku))
    accepted_by_sku = (
        trace_truth.assign(
            is_true_accepted=lambda frame: (
                frame["candidate_gtin"]
                .astype(str)
                .eq(frame["true_item_id"].astype(str))
                & frame["accepted"].astype(bool)
            )
        )
        .groupby("SKU_ID")["is_true_accepted"]
        .any()
    )

    summary = predictions[["SKU_ID", "ITEM_ID"]].copy()
    summary["true_item_id"] = summary["SKU_ID"].map(true_item_by_sku)
    summary["true_candidate_retrieved"] = (
        summary["SKU_ID"].map(retrieved_by_sku).fillna(False).astype("int8")
    )
    summary["true_candidate_accepted"] = (
        summary["SKU_ID"].map(accepted_by_sku).fillna(False).astype("int8")
    )
    predicted = summary["ITEM_ID"].astype(str)
    actual = summary["true_item_id"].astype(str)
    correct = predicted.eq(actual)
    summary["prediction_correct"] = correct.astype("int8")
    summary["error_type"] = np.select(
        [
            correct,
            predicted.str.startswith(_unmatched_prefix()),
            summary["true_candidate_retrieved"].eq(0),
            summary["true_candidate_accepted"].eq(0),
        ],
        ["correct", "unmatched", "retrieval_miss", "true_candidate_rejected"],
        default="wrong_assignment",
    )
    return summary.drop(columns=["ITEM_ID"])


def _audit_trace(
    candidates: pd.DataFrame,
    trace: pd.DataFrame,
    predictions: pd.DataFrame,
    truth: pd.DataFrame | None,
    *,
    partition: str,
    fold: object,
    threshold: float,
) -> pd.DataFrame:
    """Attach prediction/error context to every retained candidate row."""
    diagnostics = _merge_audit_context(candidates, trace, predictions)

    diagnostics["source_file"] = str(F["dataset_deduped"])
    diagnostics["evaluation_partition"] = partition
    diagnostics["evaluation_fold"] = "" if fold is None else str(fold)
    diagnostics["evaluation_threshold"] = float(threshold)
    diagnostics["predicted_ITEM_ID"] = diagnostics["ITEM_ID"]

    if truth is None:
        diagnostics["true_candidate_retrieved"] = pd.NA
        diagnostics["true_candidate_accepted"] = pd.NA
        diagnostics["prediction_correct"] = pd.NA
        diagnostics["error_type"] = "unlabeled"
        return diagnostics
    summary = _truth_audit_context(candidates, trace, predictions, truth)
    diagnostics = diagnostics.drop(columns=["true_item_id"])
    diagnostics = diagnostics.merge(
        summary,
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    return diagnostics


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


def _pairwise_counts(
    true_labels: pd.Series, predicted_labels: pd.Series
) -> dict[str, int]:
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


def _pair_group_value(left: str, right: str) -> str:
    """Encode a pair's group labels without losing either endpoint value."""
    return json.dumps([left, right], ensure_ascii=False, separators=(",", ":"))


def pair_disagreements(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    trace: pd.DataFrame,
) -> pd.DataFrame:
    """Return every unordered pair on which truth and assignment disagree.

    ``edge_exists`` is one only for a predicted-same relation.  This matcher
    assigns each SKU directly to a canonical record (it does not build a SKU
    graph), so it is the meaningful equivalent of a predicted graph edge.
    ``pair_score`` is the weaker selected canonical-assignment score of the
    two endpoints. ``candidate_generated`` says whether both endpoints' true
    canonical records were retrieved. ``failure_stage`` prioritizes ANN
    retrieval, then score threshold, then gate rejection, before assigning a
    remaining error to ranking/assignment. These evidence fields intentionally
    describe the decision that created the predicted grouping, not a
    separately computed SKU-to-SKU similarity that this lane never used.
    """
    truth_frame = truth[["SKU_ID", "true_item_id"]].copy()
    pred_frame = pred[["SKU_ID", "ITEM_ID"]].copy()
    for frame in (truth_frame, pred_frame):
        frame["SKU_ID"] = frame["SKU_ID"].astype(str)
    if (
        truth_frame["SKU_ID"].duplicated().any()
        or pred_frame["SKU_ID"].duplicated().any()
    ):
        raise ValueError("pair disagreement inputs must contain one row per SKU_ID")
    merged = truth_frame.merge(
        pred_frame, on="SKU_ID", how="inner", validate="one_to_one"
    )
    if len(merged) != len(truth_frame) or len(merged) != len(pred_frame):
        raise ValueError("pair disagreement population mismatch")
    merged["true_item_id"] = merged["true_item_id"].astype(str)
    merged["ITEM_ID"] = merged["ITEM_ID"].astype(str)

    # At most one selected trace record exists per assigned SKU. An unmatched
    # SKU has no selected candidate and therefore deliberately gets missing
    # score/gate evidence rather than invented evidence.
    selected = trace.loc[trace["selected"].astype(bool)].copy()
    if selected.duplicated("SKU_ID").any():
        raise RuntimeError("candidate trace has multiple selected rows for a SKU")
    evidence = selected[["SKU_ID", "score", "attribute_gate"]].copy()
    evidence["SKU_ID"] = evidence["SKU_ID"].astype(str)
    evidence = evidence.rename(
        columns={"score": "selected_score", "attribute_gate": "selected_attribute_gate"}
    )
    merged = merged.merge(evidence, on="SKU_ID", how="left", validate="one_to_one")
    trace_candidates = trace[["SKU_ID", "candidate_gtin"]].copy()
    trace_candidates["SKU_ID"] = trace_candidates["SKU_ID"].astype(str)
    trace_candidates["candidate_gtin"] = trace_candidates["candidate_gtin"].astype(str)
    for column, default in (
        ("retrieval_source", "unknown"),
        ("score_pass", False),
        ("accepted", False),
    ):
        trace_candidates[column] = trace[column] if column in trace else default
    trace_candidates["retrieval_source"] = trace_candidates["retrieval_source"].astype(
        str
    )
    true_candidate_rows = trace_candidates.merge(
        truth_frame,
        on="SKU_ID",
        how="inner",
        validate="many_to_one",
    )
    true_candidate_rows = true_candidate_rows.loc[
        true_candidate_rows["candidate_gtin"].eq(true_candidate_rows["true_item_id"])
    ]
    true_candidate_evidence = {
        str(sku): {
            "source": "+".join(sorted(set(rows["retrieval_source"].astype(str)))),
            "score_pass": bool(rows["score_pass"].astype(bool).any()),
            "accepted": bool(rows["accepted"].astype(bool).any()),
        }
        for sku, rows in true_candidate_rows.groupby("SKU_ID", sort=False)
    }
    by_sku = merged.set_index("SKU_ID").to_dict("index")

    def error_count(frame: pd.DataFrame, other_column: str) -> int:
        return int(
            _combination_count(len(frame))
            - sum(
                _combination_count(int(n)) for n in frame.groupby(other_column).size()
            )
        )

    def pair_row(
        sku_a: str,
        sku_b: str,
        disagreement_type: str,
        caused: int,
    ) -> dict[str, object]:
        left, right = by_sku[sku_a], by_sku[sku_b]
        scores = [left["selected_score"], right["selected_score"]]
        finite_scores = [float(score) for score in scores if pd.notna(score)]
        endpoint_evidence = [
            true_candidate_evidence.get(sku_a),
            true_candidate_evidence.get(sku_b),
        ]
        generated = all(item is not None for item in endpoint_evidence)
        sources = [
            item["source"] if item is not None else "not_generated"
            for item in endpoint_evidence
        ]
        if not generated:
            failure_stage = "retrieval_candidate_universe_failure"
        elif not all("semantic_top_k" in source for source in sources):
            failure_stage = "retrieval_ann_failure"
        elif not all(item["score_pass"] for item in endpoint_evidence if item):
            failure_stage = "scoring_low_score"
        elif not all(item["accepted"] for item in endpoint_evidence if item):
            failure_stage = "attribute_gate_rejection"
        else:
            failure_stage = "ranking_or_assignment"
        generation_status = (
            "generated" if generated else "unretrieved_candidate_generation_failure"
        )
        return {
            "sku_id_a": sku_a,
            "sku_id_b": sku_b,
            "disagreement_type": disagreement_type,
            # A false merge has two true labels; a false split has two predicted
            # labels. JSON keeps the requested singular columns lossless.
            "true_group_id": _pair_group_value(
                left["true_item_id"], right["true_item_id"]
            ),
            "predicted_group_id": _pair_group_value(left["ITEM_ID"], right["ITEM_ID"]),
            "pair_score": min(finite_scores) if len(finite_scores) == 2 else np.nan,
            "edge_exists": int(left["ITEM_ID"] == right["ITEM_ID"]),
            "candidate_generated": generated,
            "candidate_generation_status": generation_status,
            "candidate_generation_source": f"{sources[0]}|{sources[1]}",
            "failure_stage": failure_stage,
            "error_classification": f"{disagreement_type}_{failure_stage}",
            "attribute_gate_result": (
                f"{left['selected_attribute_gate']}|{right['selected_attribute_gate']}"
                if pd.notna(left["selected_attribute_gate"])
                and pd.notna(right["selected_attribute_gate"])
                else "not_selected_candidate"
            ),
            "component_size_true": int(
                (merged["true_item_id"] == left["true_item_id"]).sum()
            ),
            "component_size_pred": int((merged["ITEM_ID"] == left["ITEM_ID"]).sum()),
            "number_of_pairwise_errors_caused": caused,
        }

    rows: list[dict[str, object]] = []
    # False merge: same predicted assignment, different true canonical item.
    for _, predicted_group in merged.groupby("ITEM_ID", sort=False):
        caused = error_count(predicted_group, "true_item_id")
        if not caused:
            continue
        true_groups = [
            sorted(group["SKU_ID"].tolist())
            for _, group in predicted_group.groupby("true_item_id", sort=False)
        ]
        for left_group, right_group in combinations(true_groups, 2):
            for sku_a, sku_b in product(left_group, right_group):
                rows.append(pair_row(sku_a, sku_b, "false_merge", caused))

    # False split: same true canonical item, different predicted assignment.
    for _, true_group in merged.groupby("true_item_id", sort=False):
        caused = error_count(true_group, "ITEM_ID")
        if not caused:
            continue
        predicted_groups = [
            sorted(group["SKU_ID"].tolist())
            for _, group in true_group.groupby("ITEM_ID", sort=False)
        ]
        for left_group, right_group in combinations(predicted_groups, 2):
            for sku_a, sku_b in product(left_group, right_group):
                rows.append(pair_row(sku_a, sku_b, "false_split", caused))

    return pd.DataFrame(rows, columns=PAIR_DISAGREEMENT_COLUMNS)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prediction_metrics(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    threshold: float,
    include_graph_diagnostics: bool = True,
) -> dict[str, float | int | str]:
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
    if (
        truth_frame["SKU_ID"].duplicated().any()
        or pred_frame["SKU_ID"].duplicated().any()
    ):
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
    intersections = merged.groupby(["true_item_id", "ITEM_ID"], sort=False).size()
    predicted_sizes = merged.groupby("ITEM_ID").size()
    true_sizes = merged.groupby("true_item_id").size()
    group_precision = []
    group_recall = []
    for (true_id, predicted_id), intersection in intersections.items():
        group_precision.extend(
            [float(intersection / predicted_sizes[predicted_id])] * int(intersection)
        )
        group_recall.extend(
            [float(intersection / true_sizes[true_id])] * int(intersection)
        )

    graph = (
        candidate_graph_diagnostics(candidates, threshold)
        if include_graph_diagnostics
        else empty_candidate_graph_diagnostics()
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
        "expected_group_count": int(merged["true_item_id"].nunique()),
        "plausible_group_count": int(graph["plausible_group_count"]),
        "unmatched_skus": int(
            merged["ITEM_ID"].astype(str).str.startswith(_unmatched_prefix()).sum()
        ),
        **{key: graph[key] for key in METRIC_COLUMNS if key.startswith("diagnostic_")},
        **counts,
    }
    if tuple(metrics) != METRIC_COLUMNS:
        raise RuntimeError(
            "prediction metric contract drifted: "
            f"expected={METRIC_COLUMNS}, actual={tuple(metrics)}"
        )
    _METRIC_COLUMNS_SPEC.validate_frame(
        pd.DataFrame([metrics]),
        "prediction metrics",
    )
    return metrics


def gtin_metrics(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    fold: object,
    threshold: float,
    candidates: pd.DataFrame,
    *,
    include_graph_diagnostics: bool = True,
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
        (status, merged[merged["gtin_status"].eq(status)]) for status in GTIN_STATUSES
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
                    "expected_group_count": 0,
                    "unmatched_skus": 0,
                    **empty_candidate_graph_diagnostics(),
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
                    candidates=candidates[candidates["SKU_ID"].isin(group["SKU_ID"])],
                    threshold=threshold,
                    include_graph_diagnostics=include_graph_diagnostics,
                ),
            }
        )
    _METRIC_COLUMNS_SPEC.validate_frame(pd.DataFrame(rows), "GTIN metrics")
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
    scores = scored["score"].to_numpy(dtype=float)
    labels = (
        scored["candidate_gtin"]
        .astype(str)
        .eq(scored["true_item_id"].astype(str))
        .to_numpy(dtype=int)
    )
    retrieved = set(scored["SKU_ID"].astype(str))
    missing = sorted(set(truth["SKU_ID"].astype(str)) - retrieved)
    if missing:
        floor = float(np.nextafter(scores.min(), -np.inf)) if len(scores) else -np.inf
        scores = np.concatenate([scores, np.full(len(missing), floor)])
        labels = np.concatenate([labels, np.ones(len(missing), dtype=int)])
    return scores, labels


def _retrieval_diagnostics(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, int | float]:
    """Measure candidate-population health and true-candidate retrieval.

    ``with_candidate`` only proves that the retrieval pipeline emitted some
    candidate for an SKU. The true-candidate and ANN-only fields are the
    decision-useful retrieval measures: a truth absent from their candidate
    universe is an unretrieved case, never a threshold/model error.
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
    true_by_sku = truth.drop_duplicates("SKU_ID").set_index("SKU_ID")["true_item_id"]
    candidate_truth = candidates[["SKU_ID", "candidate_gtin"]].copy()
    candidate_truth["SKU_ID"] = candidate_truth["SKU_ID"].astype(str)
    candidate_truth["candidate_gtin"] = candidate_truth["candidate_gtin"].astype(str)
    true_candidate = candidate_truth.loc[
        candidate_truth["candidate_gtin"].eq(
            candidate_truth["SKU_ID"].map(true_by_sku).astype(str)
        )
    ]
    true_candidate_skus = set(true_candidate["SKU_ID"])
    ann_true_candidate_skus: set[str] = set()
    if "retrieval_source" in candidates:
        ann_sources = candidates.loc[
            candidates["retrieval_source"]
            .astype(str)
            .str.contains("semantic_top_k", regex=False),
            ["SKU_ID", "candidate_gtin"],
        ].copy()
        ann_sources["SKU_ID"] = ann_sources["SKU_ID"].astype(str)
        ann_sources["candidate_gtin"] = ann_sources["candidate_gtin"].astype(str)
        ann_true_candidate_skus = set(
            ann_sources.loc[
                ann_sources["candidate_gtin"].eq(
                    ann_sources["SKU_ID"].map(true_by_sku).astype(str)
                ),
                "SKU_ID",
            ]
        )
    return {
        "truth_population": len(truth_ids),
        "with_candidate": len(with_any),
        "without_candidate": len(without),
        "without_candidate_gtin_trusted": gtin_trusted,
        "with_true_candidate": len(true_candidate_skus),
        "without_true_candidate": len(truth_ids - true_candidate_skus),
        "true_candidate_recall": _safe_ratio(len(true_candidate_skus), len(truth_ids)),
        "ann_with_true_candidate": len(ann_true_candidate_skus),
        "ann_without_true_candidate": len(truth_ids - ann_true_candidate_skus),
        "ann_true_candidate_recall": _safe_ratio(
            len(ann_true_candidate_skus), len(truth_ids)
        ),
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
    base = _ensure_source_row_identity(
        load_dataset_deduped().rename(columns={"product_id": "SKU_ID"})
    )
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
    if calibration[SOURCE_ROW_INDEX_COLUMN].duplicated().any():
        raise RuntimeError("calibration merge duplicated source row identities")
    if calibration.empty:
        raise ValueError("calibration input does not match the deduped dataset")
    calibration["SKU_ID"] = calibration["SKU_ID"].astype(str)
    calibration["true_item_id"] = calibration["true_item_id"].astype(str)
    folds_per_item = calibration.groupby("true_item_id")["calibration_fold"].nunique()
    if (folds_per_item > 1).any():
        raise ValueError(
            "calibration is not canonical-disjoint: an item appears in multiple folds"
        )
    fold_count = int(calibration["calibration_fold"].nunique())
    minimum_support = int(matcher.config["rand_matching"]["threshold_min_fold_support"])
    if fold_count < minimum_support:
        raise ValueError(
            "calibration has insufficient fold support for a meaningful median: "
            f"observed={fold_count}, required={minimum_support}"
        )
    calibration["gtin_status"] = [
        matcher.gtin_status(
            row_metadata_text(row, "barcode", "gtin"), row["true_item_id"]
        )
        for _, row in calibration.iterrows()
    ]
    return calibration


def _fold_partitions(
    calibration: pd.DataFrame,
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    fold: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_skus = set(calibration.loc[calibration["calibration_fold"] != fold, "SKU_ID"])
    check_skus = set(calibration.loc[calibration["calibration_fold"] == fold, "SKU_ID"])
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
    """Compute threshold assignments through the final gate/reconciliation path."""
    if candidates.empty:
        return {float(t): pd.DataFrame(columns=ASSIGNMENT_COLUMNS) for t in thresholds}
    result: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        result[float(threshold)] = _assignments_with_trace(
            candidates, float(threshold)
        )[0]
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
        metrics = prediction_metrics(
            sweep[t],
            fit_truth,
            candidates=fit_candidates,
            threshold=t,
            include_graph_diagnostics=False,
        )
        fit_rows.append(
            {
                "threshold": t,
                "rand_index": metrics["rand_index"],
                "unmatched_skus": metrics["unmatched_skus"],
                "n": metrics["n"],
            }
        )
    selected_row = max(fit_rows, key=_threshold_selection_key)
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
    if np.isnan(precision_val) and len(scores):
        achieved_recall = _recall_at_threshold(scores, labels, float(np.min(scores)))
    elif np.isnan(precision_val):
        achieved_recall = float("nan")
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
            **prediction_metrics(
                prediction,
                check_truth,
                candidates=check_candidates,
                threshold=threshold,
                include_graph_diagnostics=False,
            ),
        }
        rows.append(row_data)
    for threshold in thresholds:
        prediction = sweep[float(threshold)]
        for row_data in gtin_metrics(
            prediction,
            check_truth,
            fold,
            float(threshold),
            check_candidates,
            include_graph_diagnostics=False,
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
    near_best = by_threshold[by_threshold.rand_index >= best_rand - plateau_tolerance]
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
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    calibration = _load_calibration_frame(matcher, calibration_input)
    candidates = matcher.score_candidates(calibration)
    truth = calibration[
        ["SKU_ID", "true_item_id", "calibration_fold", "gtin_status"]
    ].drop_duplicates("SKU_ID")
    folds = sorted(calibration["calibration_fold"].unique())
    selected = []
    sensitivity: list[dict] = []
    audit_traces: list[pd.DataFrame] = []

    for fold in folds:
        fit_truth, check_truth, fit_candidates, check_candidates = _fold_partitions(
            calibration, truth, candidates, fold
        )
        best_threshold, fit_rows, fit_unmatched_fraction = _fit_fold_threshold(
            fit_candidates, fit_truth, thresholds
        )
        fit_predictions, fit_trace = _assignments_with_trace(
            fit_candidates,
            best_threshold,
        )
        check_predictions, check_trace = _assignments_with_trace(
            check_candidates,
            best_threshold,
        )
        audit_traces.extend(
            [
                _audit_trace(
                    fit_candidates,
                    fit_trace,
                    fit_predictions,
                    fit_truth,
                    partition="fit",
                    fold=fold,
                    threshold=best_threshold,
                ),
                _audit_trace(
                    check_candidates,
                    check_trace,
                    check_predictions,
                    check_truth,
                    partition="check",
                    fold=fold,
                    threshold=best_threshold,
                ),
            ]
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
                "fit_rand_index": float(fit_by_threshold[best_threshold]["rand_index"]),
                "fit_youden_threshold": float(youden_info["threshold"]),
                "fit_youden_reason": youden_info["reason"],
                fit_col: float(precision_info["threshold"]),
                "fit_recall_reason": precision_info["reason"],
                "fit_recall_achieved": float(precision_info["achieved_recall"]),
                "fit_unmatched_fraction": fit_unmatched_fraction,
                # The ANN implementation currently exhaustively searches the
                # encoded canonical matrix; top_k is therefore its sole
                # recall-control parameter and must travel with its metrics.
                "ann_top_k": int(matcher.top_k),
                "truth_population": retrieval["truth_population"],
                "with_candidate": retrieval["with_candidate"],
                "without_candidate": retrieval["without_candidate"],
                "without_candidate_gtin_trusted": retrieval[
                    "without_candidate_gtin_trusted"
                ],
                "with_true_candidate": retrieval["with_true_candidate"],
                "without_true_candidate": retrieval["without_true_candidate"],
                "true_candidate_recall": retrieval["true_candidate_recall"],
                "ann_with_true_candidate": retrieval["ann_with_true_candidate"],
                "ann_without_true_candidate": retrieval["ann_without_true_candidate"],
                "ann_true_candidate_recall": retrieval["ann_true_candidate_recall"],
                "reconciliation_scope": _reconciliation_scope(),
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
    calibration_trace = pd.concat(audit_traces, ignore_index=True)
    return (
        final_threshold,
        selected_df,
        sensitivity_df,
        alternatives,
        plateau,
        calibration_trace,
    )


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
    calibration_trace: pd.DataFrame,
) -> None:
    _METRIC_COLUMNS_SPEC.validate_frame(
        sensitivity_df,
        "threshold sensitivity metrics",
    )
    _METRIC_COLUMNS_SPEC.validate_frame(
        alternatives_df,
        "threshold comparison metrics",
    )
    selected_df.to_csv(
        output_dir / output_names["threshold_selection_by_fold"], index=False
    )
    sensitivity_df.to_csv(
        output_dir / output_names["threshold_sensitivity_by_gtin_status"],
        index=False,
    )
    alternatives_df.to_csv(
        output_dir / output_names["threshold_comparison"], index=False
    )
    _DIAGNOSTICS_COLUMNS_SPEC.validate_frame(
        calibration_trace,
        "calibration diagnostics",
    )
    calibration_trace.to_csv(
        output_dir / output_names["calibration_diagnostics"],
        index=False,
    )
    plateau.update(
        {
            "final_threshold": final_threshold,
            "selection_method": "median of fold-selected Rand Index thresholds",
            "calibration_fold_count": int(selected_df["check_fold"].nunique()),
            "threshold_support_count": int(
                selected_df["selected_threshold"].notna().sum()
            ),
            "threshold_support_minimum": int(
                rand_matching_cfg()["threshold_min_fold_support"]
            ),
            "threshold_support_sufficient": bool(
                len(selected_df)
                >= int(rand_matching_cfg()["threshold_min_fold_support"])
            ),
            "target_recall": target_recall,
            "tie_break": list(rand_matching_cfg()["threshold_tie_break"]),
            "reconciliation_scope": _reconciliation_scope(),
            "assignment_tie_break": [
                f"{column} {'asc' if ascending else 'desc'}"
                for column, ascending in zip(
                    ASSIGNMENT_SORT_COLUMNS,
                    ASSIGNMENT_SORT_ASCENDING,
                    strict=True,
                )
            ],
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
    axis.axvline(
        final_threshold,
        color="black",
        linestyle="--",
        label=f"final={final_threshold:.2f}",
    )
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


_ANN_MISS_COLUMNS = (
    "SKU_ID",
    "true_item_id",
    "sku_gtin",
    "sku_title",
    "sku_attributes",
    "sku_brand",
    "sku_category",
    "sku_volume",
    "sku_pack",
    "sku_flavor",
    "true_candidate_retrieval_source",
    "true_candidate_score",
    "true_candidate_score_pass",
    "true_candidate_accepted",
    "true_candidate_attribute_gate",
    "ann_top_candidate_gtin",
    "ann_top_candidate_score",
    "ann_top_candidate_rank",
    "union_selected_item_id",
    "union_selected_score",
)


def _ann_missed_true_matches(
    trace: pd.DataFrame,
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Export true candidates absent from ANN, with decision evidence.

    Exact-GTIN rescue may make a row recoverable in the union lane, but it is
    still an ANN retrieval miss. This artifact keeps those cases separate from
    score and gate failures.
    """
    true = truth[["SKU_ID", "true_item_id"]].copy()
    true["SKU_ID"] = true["SKU_ID"].astype(str)
    rows = trace.drop(
        columns=["true_item_id", "true_candidate_retrieved", "true_candidate_accepted"],
        errors="ignore",
    ).merge(true, on="SKU_ID", how="inner", validate="many_to_one")
    is_true = rows["candidate_gtin"].astype(str).eq(rows["true_item_id"].astype(str))
    is_ann = (
        rows["retrieval_source"].astype(str).str.contains("semantic_top_k", regex=False)
    )
    true_rows = rows.loc[is_true].copy()
    ann_true_ids = set(true_rows.loc[is_ann.loc[true_rows.index], "SKU_ID"].astype(str))
    missed_ids = set(true["SKU_ID"]) - ann_true_ids
    missed_true = true_rows.loc[true_rows["SKU_ID"].astype(str).isin(missed_ids)].copy()
    ann_rows = rows.loc[
        rows["retrieval_source"].astype(str).str.contains("semantic_top_k", regex=False)
    ].copy()
    ann_best = (
        ann_rows.sort_values(
            ["SKU_ID", "candidate_rank", "score"], ascending=[True, True, False]
        )
        .drop_duplicates("SKU_ID")[
            ["SKU_ID", "candidate_gtin", "score", "candidate_rank"]
        ]
        .rename(
            columns={
                "candidate_gtin": "ann_top_candidate_gtin",
                "score": "ann_top_candidate_score",
                "candidate_rank": "ann_top_candidate_rank",
            }
        )
    )
    selected = trace.loc[
        trace["selected"].astype(bool), ["SKU_ID", "candidate_gtin", "score"]
    ].rename(
        columns={
            "candidate_gtin": "union_selected_item_id",
            "score": "union_selected_score",
        }
    )
    result = missed_true.merge(
        ann_best, on="SKU_ID", how="left", validate="one_to_one"
    ).merge(selected, on="SKU_ID", how="left", validate="one_to_one")
    result = result.rename(
        columns={
            "retrieval_source": "true_candidate_retrieval_source",
            "score": "true_candidate_score",
            "score_pass": "true_candidate_score_pass",
            "accepted": "true_candidate_accepted",
            "attribute_gate": "true_candidate_attribute_gate",
        }
    )
    return (
        result.reindex(columns=_ANN_MISS_COLUMNS)
        .sort_values("SKU_ID")
        .reset_index(drop=True)
    )


def _retrieval_ablation_metrics(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Compare union retrieval to ANN-only retrieval at one fixed threshold."""
    modes = {
        "union_ann_plus_rescue": candidates,
        "ann_only": candidates.loc[
            candidates["retrieval_source"]
            .astype(str)
            .str.contains("semantic_top_k", regex=False)
        ].copy(),
    }
    rows: list[dict[str, object]] = []
    for mode, population in modes.items():
        predictions, _ = _assignments_with_trace(
            population,
            threshold,
            threshold_by_gtin_status=_final_threshold_by_gtin_status(),
        )
        metrics = gtin_metrics(
            predictions,
            truth,
            mode,
            threshold,
            population,
            include_graph_diagnostics=True,
        )
        rows.extend(
            {
                "retrieval_mode": mode,
                "ann_top_k": int(rand_matching_cfg()["top_k"]),
                **row,
            }
            for row in metrics
        )
    return pd.DataFrame(rows)


def _evaluate_holdout(
    matcher: RandMatcher,
    holdout_labels: pd.DataFrame,
    final_threshold: float,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _ensure_source_row_identity(
        load_dataset_deduped().rename(columns={"product_id": "SKU_ID"})
    )
    base["SKU_ID"] = base["SKU_ID"].astype(str)
    unknown_ids = sorted(set(holdout_labels["SKU_ID"]) - set(base["SKU_ID"]))
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
    if holdout[SOURCE_ROW_INDEX_COLUMN].duplicated().any():
        raise RuntimeError("holdout merge duplicated source row identities")
    holdout["SKU_ID"] = holdout["SKU_ID"].astype(str)
    holdout["true_item_id"] = holdout["true_item_id"].astype(str)
    holdout["gtin_status"] = [
        matcher.gtin_status(
            row_metadata_text(row, "barcode", "gtin"), row["true_item_id"]
        )
        for _, row in holdout.iterrows()
    ]
    candidates = matcher.score_candidates(holdout)
    truth = holdout[["SKU_ID", "true_item_id", "gtin_status"]].drop_duplicates("SKU_ID")
    predictions, trace = _assignments_with_trace(
        candidates,
        final_threshold,
        threshold_by_gtin_status=_final_threshold_by_gtin_status(),
    )
    metrics = gtin_metrics(
        predictions,
        truth,
        "holdout",
        final_threshold,
        candidates,
        # Holdout is the frozen final evaluation.  Preserve the complete
        # metric contract, including component/bridge diagnostics, for every
        # GTIN stratum and the ALL row; do not rely on gtin_metrics' default.
        include_graph_diagnostics=True,
    )
    diagnostics = _audit_trace(
        candidates,
        trace,
        predictions,
        truth,
        partition="holdout",
        fold="holdout",
        threshold=final_threshold,
    )
    disagreements = pair_disagreements(predictions, truth, trace)
    ann_misses = _ann_missed_true_matches(trace, truth, predictions)
    ablation = _retrieval_ablation_metrics(candidates, truth, final_threshold)
    return metrics, diagnostics, disagreements, ann_misses, ablation


def _sha256_path(path: Path) -> tuple[str, int]:
    """Fingerprint one file or a checkpoint directory deterministically."""
    if path.is_symlink():
        raise ValueError(f"provenance path must not be a symlink: {path}")
    if path.is_file():
        return sha256_file(path), 1
    if not path.is_dir():
        raise FileNotFoundError(f"provenance path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(
        child for child in path.rglob("*") if child.is_file() and not child.is_symlink()
    )
    for child in files:
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest(), len(files)


def _file_provenance(path: Path, rows: int | None = None) -> _FileProvenance:
    digest, _ = _sha256_path(path)
    return _FileProvenance(path=str(path), sha256=digest, rows=rows)


def _write_provenance(
    output_dir: Path,
    output_name: str,
    matcher: RandMatcher,
    submission: pd.DataFrame,
    calibration_input: Path,
    holdout_input: Path,
    calibration_labels: pd.DataFrame,
    holdout_labels: pd.DataFrame,
    final_threshold: float,
) -> None:
    checkpoint_hash, checkpoint_files = _sha256_path(matcher.checkpoint)
    provenance = _SubmissionProvenance(
        dataset_deduped=_file_provenance(
            F["dataset_deduped"],
            rows=len(submission),
        ),
        canonical_records=_file_provenance(
            RESULTS / F["canonical_records"],
            rows=len(matcher.record_map),
        ),
        calibration_input=_file_provenance(
            calibration_input,
            rows=len(calibration_labels),
        ),
        holdout_input=_file_provenance(
            holdout_input,
            rows=len(holdout_labels),
        ),
        paths_config=_file_provenance(CONFIG_PATH),
        training_config=_file_provenance(TRAINING_CONFIG_PATH),
        checkpoint=_FileProvenance(
            path=str(matcher.checkpoint),
            sha256=checkpoint_hash,
            rows=checkpoint_files,
        ),
        final_threshold=final_threshold,
        threshold_by_gtin_status=_final_threshold_by_gtin_status(),
        brand_conflict_veto=bool(rand_matching_cfg()["brand_conflict_veto"]),
        confidence_penalty_mask=dict(rand_matching_cfg()["confidence_penalty_mask"]),
        flavor_overlap_penalty=dict(rand_matching_cfg()["flavor_overlap_penalty"]),
        targeted_veto_gates=dict(rand_matching_cfg()["targeted_veto_gates"]),
        unmatched_prefix=_unmatched_prefix(),
        rows=len(submission),
        unique_items=int(submission["ITEM_ID"].nunique()),
        unmatched=int(
            submission["ITEM_ID"].astype(str).str.startswith(_unmatched_prefix()).sum()
        ),
        calibration_folds=tuple(
            sorted(calibration_labels["calibration_fold"].astype(str).unique())
        ),
        lineage=(
            "submission and audit traces derive from dataset_deduped; "
            "source_row_index links every candidate to its source row; "
            "candidate_gtin links every decision to canonical_records"
        ),
    )
    provenance_path = output_dir / output_name
    provenance_path.write_text(
        json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"provenance: {provenance_path}")


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
        threshold_by_gtin_status=_final_threshold_by_gtin_status(),
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
    diagnostics = _audit_trace(
        candidates,
        candidate_trace,
        predictions,
        None,
        partition="final",
        fold=None,
        threshold=final_threshold,
    )
    _DIAGNOSTICS_COLUMNS_SPEC.validate_frame(diagnostics, "diagnostics")
    diagnostics.to_csv(output_dir / output_names["diagnostics"], index=False)
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
    (
        final_threshold,
        selected,
        sensitivity,
        alternatives,
        plateau,
        calibration_diagnostics,
    ) = result
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
        calibration_diagnostics,
    )
    (
        holdout_metrics,
        holdout_diagnostics,
        holdout_pair_disagreements,
        holdout_ann_missed_true_matches,
        holdout_retrieval_ablation_metrics,
    ) = _evaluate_holdout(
        matcher,
        holdout_labels,
        final_threshold,
    )
    holdout_metrics_frame = pd.DataFrame(holdout_metrics)
    _METRIC_COLUMNS_SPEC.validate_frame(
        holdout_metrics_frame,
        "holdout metrics",
    )
    holdout_metrics_frame.to_csv(
        output_dir / output_names["holdout_metrics"], index=False
    )
    _DIAGNOSTICS_COLUMNS_SPEC.validate_frame(
        holdout_diagnostics,
        "holdout diagnostics",
    )
    holdout_diagnostics.to_csv(
        output_dir / output_names["holdout_diagnostics"],
        index=False,
    )
    holdout_pair_disagreements.to_csv(
        output_dir / output_names["holdout_pair_disagreements"],
        index=False,
    )
    holdout_ann_missed_true_matches.to_csv(
        output_dir / output_names["holdout_ann_missed_true_matches"],
        index=False,
    )
    holdout_retrieval_ablation_metrics.to_csv(
        output_dir / output_names["holdout_retrieval_ablation_metrics"],
        index=False,
    )
    submission = _write_final_submission(
        matcher,
        output_dir,
        output_names,
        final_threshold,
    )
    _write_provenance(
        output_dir,
        output_names["provenance"],
        matcher,
        submission,
        calibration_input,
        holdout_input,
        calibration_labels,
        holdout_labels,
        final_threshold,
    )
    print(
        {
            "folds": selected.check_fold.tolist(),
            "selected_thresholds": selected.selected_threshold.tolist(),
            "final_threshold": final_threshold,
            "rows": len(submission),
            "unique_items": submission.ITEM_ID.nunique(),
            "unmatched": int(
                submission.ITEM_ID.str.startswith(_unmatched_prefix()).sum()
            ),
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
        help="fine-tuned SentenceTransformer directory (or FINETUNED_CHECKPOINT)",
    )
    parser.add_argument(
        "--calibration-input",
        help="canonical-disjoint calibration CSV (or CALIBRATION_INPUT)",
    )
    parser.add_argument(
        "--holdout-input",
        help="frozen, untouched holdout CSV (or HOLDOUT_INPUT)",
    )
    parser.add_argument(
        "--ann-index-dir",
        type=Path,
        help="persisted HNSW artifact directory (default: training_ANN.yaml)",
    )
    parser.add_argument(
        "--rebuild-ann-index",
        action="store_true",
        help="rebuild the complete catalog HNSW artifact",
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
    thresholds = _threshold_grid(
        float(cfg["threshold_min"]),
        float(cfg["threshold_max"]),
        float(cfg["threshold_step"]),
    )

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"FINETUNED_CHECKPOINT must point to a checkpoint directory: {checkpoint}"
        )
    if not calibration_input.is_file():
        raise FileNotFoundError(
            f"CALIBRATION_INPUT must point to a calibration CSV: {calibration_input}"
        )
    if not holdout_input.is_file():
        raise FileNotFoundError(
            f"HOLDOUT_INPUT must point to a frozen holdout CSV: {holdout_input}"
        )
    ann_cfg = load_ann_config()
    matcher = RandMatcher(
        checkpoint,
        batch_size=int(ann_cfg.embedding.encode_batch_size),
        top_k=int(ann_cfg.index.top_k),
        ann_index_dir=args.ann_index_dir,
        rebuild_ann_index=args.rebuild_ann_index,
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
