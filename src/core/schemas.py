"""src/core/schemas.py — pydantic contracts + shape assertions for the training
transformation pipeline (owner directive: pydantic + shape assertions for
better control).

TWO jobs:

  1. CONFIG CONTRACTS — one pydantic model per split config file, validated
     at load by lib.common (fail-loudly at import, never mid-run):
       DataConfig       config/paths.yaml     — paths/files/column_mapping/seed/models
       TrainingConfig   config/training.yaml — the training lane's knobs
     (EDA/eda.yaml was deleted with the EDA dir 2026-09-10; its five
     TRAIN-consumed keys — plots.dpi, pairs.max_pos_per_group/n_neg/
     neg_oversample, strip_audit_sample — migrated into config/training.yaml.)

  2. BOUNDARY CONTRACTS — validated containers at every transform boundary
     in the pipeline (small objects, never per-row hot loops):
       ExtractedAttributes   pipeline.extract_all
       GateResult            pipeline.three_way_gate
       CanonicalRecord       pipeline.generate_canonical
       PairArrays            index-pair containers (pos/neg/hp)
       TrainingData          pipeline.build_training_data (the bundle)
       MaskingResult         TRAIN.masking.augment_positives
       DataTuple             the (df, payload, structured_features, row_bc,
                             country, pos, hp_pairs, emb0) 8-tuple crossing into
                             train_one_config
       TrainConfig           the per-config dict train_one_config receives
       FoldSets              TRAIN.folds.component_folds output
       EvalSummaryRow        one model_evaluation_summary.csv row (the
                             zero-shot lane's report boundary)
       TraceRow              one core.tracing.TRACE_COLUMNS row (the
                             consolidated pipeline trace)

     Frame checks (DataFrame column/domain contracts) live in
     check_canonical_records_frame / check_gate_results_frame /
     check_labeled_pairs_frame / check_eval_summary_frame /
     check_zero_shot_similarity_frame / check_cross_country_pair_frame /
     check_trace_frame — used at CSV write/read boundaries. Every one of
     them is registered by name in FRAME_CHECKERS (the discovery surface).

HOW core.tracing WIRES THE TRACE CONTRACT (interface for the tracing owner —
this module does NOT import core.tracing's writers and core/tracing.py does
not need a new column tuple: TRACE_FRAME_COLUMNS *IS* core.tracing's own
TRACE_COLUMNS, imported, never re-declared):

    from core.schemas import TraceRow, check_trace_frame

    # 1. row boundary — ``record()`` builds one row; validate it there so a
    #    hand-written row (a scope typo, a dropped_count the arithmetic
    #    disagrees with) dies at the producer, not in the CSV:
        TraceRow.model_validate(row)          # returns the validated row

    # 2. frame boundary — ``assert_trace_frame`` and ``TraceRun.write``
    #    call the frame checker; it takes and returns a DataFrame and raises
    #    ValueError (ValidationError is a ValueError), exactly like the
    #    other check_*_frame contracts:
        def assert_trace_frame(frame, *, path=""):
            try:
                check_trace_frame(frame)
            except ValueError as exc:
                raise ValueError(f"trace frame {path}: {exc}") from exc

    # 3. WRITE boundary — call it on the concatenated frame BEFORE
    #    ``core.manifest.atomic_write_csv`` in TraceRun.write, so a
    #    corrupted stage can never land in results/logs/trace.csv:
        frame = pd.concat([existing, self.rows()], ignore_index=True)
        check_trace_frame(frame)
        atomic_write_csv(frame, target, index=False)

    The checker accepts BOTH shapes the trace actually takes: the in-memory
    frame (None / NaN counts) and the CSV read-back frame
    (``read_trace`` uses dtype=str + keep_default_na=False, so an absent
    count arrives as ""). A non-empty, non-numeric count cell still fails.

Doctrine (owner Q27): NO FALLBACKS. Optional-with-default means
"config may omit it" ONLY where the model declares a default and the
accessor raises when missing from the merged view. Nothing silently
defaults to an inline literal at the call site.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from core.tracing import TRACE_COLUMNS as TRACE_FRAME_COLUMNS


THRESHOLD_TIE_BREAK_CRITERIA = (
    "rand_index",
    "fewest_unmatched_skus",
    "lowest_threshold",
)
GTIN_STATUSES = ("both_equal", "different", "one_missing", "both_missing")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════


class DataFilesSpec(BaseModel):
    """The file-name contract — every CSV/JSON the pipeline touches.

    Keys are load-bearing: lib.common.F and every consumer index them.
    Stopword lists and category macros are centralized in
    ``config/vocabulary.json`` and validated by ``core.common``.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    dataset_deduped: str
    dataset_deduped_sample_3000: str
    dataset_deduped_train_minus_3000: str
    sku_to_rep: str
    canonical_records: str
    gate_results: str
    labeled_pairs: str
    balanced_pairs_sample_3000: str
    embedding_similarities: str
    model_evaluation_summary: str
    attribute_separation_summary: str
    attribute_separation_values: str
    fold_metrics: str
    hpo_grid_csv: str
    # hpo_tpe_best REMOVED (audit round 2 F02, finished round 3): dead
    # field — F["hpo_tpe_best"] had zero readers anywhere.
    dedupe_summary: str
    ambiguous_offer_groups: str
    removals: str
    four_pop_scores: str
    field_ablation: str
    data_scaling: str
    data_quality_summary: str
    data_quality_columns: str
    data_quality_gtin_groups: str
    title_attribute_evidence: str
    title_attribute_summary: str
    title_removed_tokens: str
    attribute_agreement_summary: str
    attribute_agreement_conflicts: str
    package_gate_impact_summary: str
    package_gate_impact_pairs: str
    # cache_dir REMOVED (audit 2026-09-09): dead knob, zero readers
    number_reference: str
    # second04_pairs_positive (audit round 2 F13): repo-side cross-country
    # gold-pair manifest (src/core/volume_verified); declared SSOT-side.
    second04_pairs_positive: str
    results_pointer: str
    colab_live_log: str
    colab_training_log: str


class DataPathsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: str
    data_dir: str
    results_dir: str
    training_results_dir: str
    models_dir: str
    embeddings_dir: str
    mlruns_dir: str
    logs_dir: str


class LayoutSpec(BaseModel):
    """One owned layout template for a GENERATED artifact (training-hpo).

    root is a binding root token: repo | data | results | results_training |
    results_hpo.  template uses {field} placeholders; fields declares the
    exact placeholder set (a caller-supplied extra drops in a crash).
    """

    model_config = ConfigDict(extra="forbid")

    root: str
    owner: str
    template: str
    fields: dict[str, str] = Field(default_factory=dict)

    @field_validator("root")
    @classmethod
    def _known_root(cls, v: str) -> str:
        if v not in {"repo", "data", "results", "results_training", "results_hpo"}:
            raise ValueError(
                f"unknown layout root {v!r}; must be one of "
                "repo|data|results|results_training|results_hpo"
            )
        return v

    @field_validator("fields")
    @classmethod
    def _known_field_types(cls, v: dict[str, str]) -> dict[str, str]:
        for typ in v.values():
            if typ not in {"str", "int", "float"}:
                raise ValueError(f"unsupported layout field type {typ!r}")
        return v


class DvcPublicationPointer(BaseModel):
    """One tracked pointer exposed for clean-checkout DVC recovery."""

    model_config = ConfigDict(extra="forbid")

    pointer: str = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)


class DvcPublicationManifest(BaseModel):
    """Pydantic contract for the tracked DVC publication index."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    run_id: str = Field(min_length=1)
    worker: int = Field(ge=1)
    remote: str = Field(min_length=1)
    verified_download: bool
    pointers: list[DvcPublicationPointer] = Field(min_length=1)


class ResultBundleFile(BaseModel):
    """One file in the verified remote result archive."""

    model_config = ConfigDict(extra="forbid", strict=True)

    worker: int = Field(ge=1)
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"[0-9a-f]{64}")


class ResultBundleExcludedFile(BaseModel):
    """One explicitly excluded remote file recorded for transparency."""

    model_config = ConfigDict(extra="forbid", strict=True)

    worker: int = Field(ge=1)
    path: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ResultBundleManifest(BaseModel):
    """Manifest for one atomic Colab result transfer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1"]
    run_id: str = Field(min_length=1)
    workers: int = Field(ge=1)
    included: list[ResultBundleFile] = Field(min_length=1)
    excluded: list[ResultBundleExcludedFile]


class DataConfig(BaseModel):
    """config/paths.yaml — the SHARED data contract (paths, file names, column
    mapping, seed, category-macro taxonomy, model registry, owned layouts).
    Domain knobs live in their own dir: config/training.yaml."""

    model_config = ConfigDict(extra="forbid")

    paths: DataPathsSpec
    files: DataFilesSpec
    layouts: dict[str, LayoutSpec] = Field(default_factory=dict)
    column_mapping: dict[str, str] = Field(min_length=1)
    seed: int
    models: dict[str, str] = Field(min_length=1)
    embedding_model_keys: list[str] = Field(min_length=1)

    model_validator(mode="after")

    @classmethod
    def _files_roots_resolve(cls, v: "DataConfig") -> "DataConfig":
        """Every files./layouts. binding must carry a known root token +
        non-empty name — the resolver needs both; an unknown root would
        otherwise fail mid-run instead of at import."""
        known = {"repo", "data", "results", "results_training", "results_hpo"}
        for key, value in v.files.model_dump().items():
            root, _, name = str(value).partition(":")
            if root not in known or not name:
                raise ValueError(
                    f"files.{key}={value!r}: must be 'root:name' with root in "
                    f"{sorted(known)}"
                )
        return v

    @field_validator("models")
    @classmethod
    def _registry_is_local(cls, v: dict[str, str]) -> dict[str, str]:
        for key, reference in v.items():
            path = Path(reference)
            if not reference.strip():
                raise ValueError(f"models.{key} must be a non-empty local path")
            if path.is_absolute() or "://" in reference or ".." in path.parts:
                raise ValueError(
                    f"models.{key} must be a project-owned relative model path, "
                    f"got {reference!r}"
                )
        if "multilingual_l12" not in v or not v["multilingual_l12"]:
            raise ValueError(
                "models.multilingual_l12 missing — the trainer base resolves "
                "from it (train.py --model default; no-fallback doctrine)"
            )
        return v

    @model_validator(mode="after")
    def _embedding_models_are_registered(self) -> "DataConfig":
        unknown = sorted(set(self.embedding_model_keys) - set(self.models))
        if unknown:
            raise ValueError(
                "embedding_model_keys contains unknown model registry key(s): "
                + ", ".join(unknown)
            )
        if len(set(self.embedding_model_keys)) != len(self.embedding_model_keys):
            raise ValueError("embedding_model_keys must not contain duplicates")
        return self


class SplitSpec(BaseModel):
    """Component-aware split contract (config/training.yaml split:).

    train/dev/test are COMPONENT shares over the positive-pair graph —
    asserted to sum to 1.0 (no silent re-normalization: a mis-configured
    split must CRASH, not quietly produce 60/20/20). Calibration is a
    canonical-disjoint subset of DEV, controlled by calibration_dev_fraction.
    In mode "holdout" the realized split comes from holdout_component_folds
    through the single derivation folds.holdout_split: component_folds deals
    whole COMPONENTS round-robin over that many groups, test is the last
    group, dev the one before it and train the rest. Each group therefore
    holds APPROXIMATELY — never exactly — 1.0 / holdout_component_folds of the
    graph: the deal is per component and components differ in size (measured
    on real data, the four quarter shares are 0.2500/0.2500/0.2499/0.2499 of
    barcodes and 0.2488/0.2513/0.2498/0.2502 of positive pairs). What this
    model pins exactly is the DECLARED quarter, and only when mode is
    "holdout": dev_fraction and test_fraction must each be within 1e-9 of
    1.0 / holdout_component_folds (4 -> 0.25/0.25), so a knob the lane cannot
    honour is a load error and not a mid-lane crash. In mode "cv" the split
    envelope must be 1.0/0.0/0.0: the lane deals cv_folds component folds
    across the full data and carves DEV out of each fold's TRAIN side with
    training.dev_fraction. The same disagreement re-raises inside
    folds.holdout_split for holdout runs, which remains the runtime backstop.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["holdout", "cv"]
    train_fraction: float = Field(ge=0.0, le=1.0)
    dev_fraction: float = Field(ge=0.0, le=1.0)
    calibration_dev_fraction: float = Field(gt=0.0, le=0.5)
    calibration_seed_offset: int = Field(ge=0)
    holdout_component_folds: Literal[4]
    test_fraction: float = Field(ge=0.0, le=1.0)
    fixed_threshold: float = Field(gt=0.0, lt=1.0)
    cv_folds: int = Field(ge=2)

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> SplitSpec:
        s = self.train_fraction + self.dev_fraction + self.test_fraction
        if abs(s - 1.0) > 1e-9:
            raise ValueError(
                f"split fractions must sum to 1.0, got {s} "
                f"({self.train_fraction}+{self.dev_fraction}+"
                f"{self.test_fraction})"
            )
        return self

    @model_validator(mode="after")
    def _mode_matches_fraction_contract(self) -> SplitSpec:
        """Reject fraction declarations the selected builder cannot honor.

        Holdout derives train/dev/test from ``holdout_component_folds``, so
        DEV and TEST must each name one dealt quarter. CV has a different
        builder: ``component_folds(cv_folds)`` partitions the full data and
        the training-level ``training.dev_fraction`` carves early-stopping
        DEV from each fold's training side. Therefore the split envelope in
        CV is 100% of the data entering cross-validation and no holdout DEV or
        TEST share: 1.0/0.0/0.0. Requiring that sentinel prevents 0.25/0.25
        from being accepted as if it described a CV lane it does not build.

        ``folds.holdout_split`` keeps the same checks as the runtime backstop;
        this validator is the earlier load-time contract.
        """
        if self.mode == "cv":
            if (
                abs(self.train_fraction - 1.0) > 1e-9
                or abs(self.dev_fraction) > 1e-9
                or abs(self.test_fraction) > 1e-9
            ):
                raise ValueError(
                    "cv split contract violated: component_folds(cv_folds) "
                    "partitions the full data and training.dev_fraction "
                    "carves per-fold DEV; split must declare "
                    "train_fraction=1.0, dev_fraction=0.0, "
                    "test_fraction=0.0"
                )
            return self

        # Literal[4] is the only arity the 50/25/25 contract supports, but the
        # check is written for any n_folds (the runtime helper accepts them).
        n_folds = int(self.holdout_component_folds)
        quarter = 1.0 / n_folds
        if (
            abs(self.dev_fraction - quarter) > 1e-9
            or abs(self.test_fraction - quarter) > 1e-9
        ):
            raise ValueError(
                f"holdout split contract violated: mode=holdout with "
                f"holdout_component_folds={n_folds} deals quarters of "
                f"{quarter:.4f} each, but the split declares dev_fraction="
                f"{self.dev_fraction} and test_fraction={self.test_fraction} "
                f"(the 50/25/25 contract requires dev_fraction == "
                f"test_fraction == 1.0/holdout_component_folds = {quarter})"
            )
        return self


class UniformitySpec(BaseModel):
    """Unrelated-pair embedding-space collapse diagnostic."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    sample_pairs: int = Field(ge=1)
    seed: int
    checkpoint_scope: Literal["all", "final"]


class RobustValidationSpec(BaseModel):
    """Repeated leakage-aware validation and error-slice reporting."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    n_folds: int = Field(ge=3)
    repeats: int = Field(ge=2)
    seed: int
    min_slice_size: int = Field(ge=1)
    min_test_negatives: int = Field(ge=5)
    max_split_attempts: int = Field(ge=1)
    dimensions: list[Literal["brand", "category", "attribute"]] = Field(
        min_length=3, max_length=3
    )
    operating_thresholds: dict[str, float] = Field(min_length=2)

    @model_validator(mode="after")
    def _dimensions_are_complete(self) -> RobustValidationSpec:
        if set(self.dimensions) != {"brand", "category", "attribute"}:
            raise ValueError(
                "evaluation.robust_validation.dimensions must contain exactly "
                "brand, category, and attribute"
            )
        required = {"balanced_review", "high_precision"}
        if set(self.operating_thresholds) != required:
            raise ValueError(
                "evaluation.robust_validation.operating_thresholds must contain "
                "exactly balanced_review and high_precision"
            )
        if any(
            not 0.0 < float(value) < 1.0 for value in self.operating_thresholds.values()
        ):
            raise ValueError(
                "evaluation.robust_validation.operating_thresholds must be in (0, 1)"
            )
        return self


# ── encoder token budget (guards the silent truncation of field groups) ────


class TokenBudgetReport(BaseModel):
    """What the encoder window actually kept, per payload.

    The structured tail is appended LAST, so at ``max_seq_length`` it is the
    first thing truncated — measured at 11.9 % of target texts losing the whole
    tail, silently. This record makes that countable instead of invisible:
    a dropped field group is a named, numbered event, not a shorter string.
    """

    model_config = ConfigDict(extra="forbid")

    max_seq_length: int = Field(ge=1)
    n_records: int = Field(ge=0)
    n_over_budget: int = Field(ge=0)
    n_field_groups_dropped: int = Field(ge=0)
    dropped_groups: dict[str, int]

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> TokenBudgetReport:
        if self.n_over_budget > self.n_records:
            raise ValueError("n_over_budget cannot exceed n_records")
        if self.n_field_groups_dropped != sum(self.dropped_groups.values()):
            raise ValueError(
                "n_field_groups_dropped must equal the sum of dropped_groups"
            )
        return self


# ── attribute separation metrics (results/training/attribute_separation_*.csv) ──
# How well each product attribute separates TRUE pairs from FALSE ones, at the
# attribute level and per attribute value.  Computed from labelled pairs plus
# canonical attributes only: it needs NO model and NO training run.
SEPARATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "attribute",
    "n_positive",
    "n_negative",
    "n_unobservable",
    "p_agree_positive",
    "p_agree_negative",
    "separation",
    "reportable",
    "flagged_weak",
    "negative_class_saturated",
)
SEPARATION_VALUE_COLUMNS: tuple[str, ...] = (
    "attribute",
    "value",
    "n_positive",
    "n_negative",
    "p_match_positive",
    "p_match_negative",
    "separation",
    "reportable",
    "flagged_weak",
)


class AttributeSeparationSpec(BaseModel):
    """Support and flagging rules for the attribute separation metrics.

    STATISTICAL HONESTY: a separation score is only meaningful with support on
    BOTH classes.  A value observed in two pairs is reported with its counts
    but is never flagged as a defect — ``reportable`` is False and
    ``flagged_weak`` stays False.  Thresholds are config, not literals.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    min_pairs_per_class: int = Field(ge=1)
    min_value_support: int = Field(ge=1)
    flag_below: float = Field(ge=-1.0, le=1.0)


class SeparationSummaryRow(BaseModel):
    """One attribute_separation_summary.csv row: one product attribute."""

    model_config = ConfigDict(extra="forbid")

    attribute: str = Field(min_length=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    n_unobservable: int = Field(ge=0)
    p_agree_positive: float = Field(ge=0.0, le=1.0)
    p_agree_negative: float = Field(ge=0.0, le=1.0)
    separation: float = Field(ge=-1.0, le=1.0)
    reportable: bool
    flagged_weak: bool
    # The attribute agrees on EVERY negative pair, so it cannot separate this
    # population by construction and its 0.0 is a property of the pair
    # sampling, not a finding about the attribute. Read every score with it.
    negative_class_saturated: bool

    @model_validator(mode="after")
    def _flag_requires_support(self) -> SeparationSummaryRow:
        if self.flagged_weak and not self.reportable:
            raise ValueError(
                f"{self.attribute}: flagged_weak=True with reportable=False — "
                "an under-supported score must never be reported as a defect"
            )
        if self.negative_class_saturated and self.separation > 0:
            raise ValueError(
                f"{self.attribute}: negative_class_saturated with a positive "
                "separation is impossible — the rule or the counts are wrong"
            )
        return self


class SeparationValueRow(BaseModel):
    """One attribute_separation_values.csv row: one value of one attribute."""

    model_config = ConfigDict(extra="forbid")

    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    p_match_positive: float = Field(ge=0.0, le=1.0)
    p_match_negative: float = Field(ge=0.0, le=1.0)
    separation: float = Field(ge=-1.0, le=1.0)
    reportable: bool
    flagged_weak: bool

    @model_validator(mode="after")
    def _flag_requires_support(self) -> SeparationValueRow:
        if self.flagged_weak and not self.reportable:
            raise ValueError(
                f"{self.attribute}={self.value!r}: flagged_weak=True with "
                "reportable=False — an under-supported value must never be "
                "reported as a defect"
            )
        return self


class EvaluationSpec(BaseModel):
    """Zero-shot evaluation protocol (config/training.yaml evaluation:) —
    the component dev/test split behind evaluate_models.

    HOLDOUT DISCIPLINE (self-fit leak closed 2026-09-14): the Youden
    threshold is fit on the DEV fold and applied verbatim to TEST — the
    old lane fitted it on the very set it scored. These knobs only steer
    WHICH component folds play which role; they can not re-couple the
    threshold to the scored half."""

    model_config = ConfigDict(extra="forbid")

    component_split_k: int = Field(ge=2)
    dev_fold: int = Field(ge=0)
    test_fold: int = Field(ge=0)
    retrieval_ks: list[int] = Field(min_length=1)
    robust_validation: RobustValidationSpec
    uniformity: UniformitySpec
    attribute_separation: AttributeSeparationSpec

    @model_validator(mode="after")
    def _folds_distinct_and_in_range(self) -> EvaluationSpec:
        if self.dev_fold >= self.component_split_k:
            raise ValueError(
                f"evaluation.dev_fold {self.dev_fold} >= component_split_k "
                f"{self.component_split_k}"
            )
        if self.test_fold >= self.component_split_k:
            raise ValueError(
                f"evaluation.test_fold {self.test_fold} >= component_split_k "
                f"{self.component_split_k}"
            )
        if self.dev_fold == self.test_fold:
            raise ValueError(
                f"evaluation.dev_fold == test_fold ({self.dev_fold}) — the "
                f"threshold-fit half and the scored half must be DISJOINT"
            )
        if self.retrieval_ks != sorted(set(self.retrieval_ks)):
            raise ValueError("evaluation.retrieval_ks must be unique and ascending")
        if 1 not in self.retrieval_ks:
            raise ValueError("evaluation.retrieval_ks must include 1 for Hits@1")
        return self


class RandMatchingOutputsSpec(BaseModel):
    """Output filenames for the final Rand Index matching lane."""

    model_config = ConfigDict(extra="forbid")

    submission: str = Field(min_length=1)
    diagnostics: str = Field(min_length=1)
    threshold_selection_by_fold: str = Field(min_length=1)
    threshold_sensitivity_by_gtin_status: str = Field(min_length=1)
    threshold_sensitivity_plot: str = Field(min_length=1)
    threshold_comparison: str = Field(min_length=1)
    plateau_diagnostic: str = Field(min_length=1)
    holdout_metrics: str = Field(min_length=1)
    calibration_diagnostics: str = Field(min_length=1)
    holdout_diagnostics: str = Field(min_length=1)
    holdout_pair_disagreements: str = Field(min_length=1)
    holdout_ann_missed_true_matches: str = Field(min_length=1)
    holdout_retrieval_ablation_metrics: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


class RandTruthSplitsSpec(BaseModel):
    """Deterministic, disjoint truth inputs for the final matcher."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)
    calibration_output: str = Field(min_length=1)
    holdout_output: str = Field(min_length=1)
    sample_size: int = Field(ge=6)
    calibration_size: int = Field(ge=3)
    calibration_folds: int = Field(ge=3)
    seed: int

    @model_validator(mode="after")
    def _sizes_are_valid(self) -> RandTruthSplitsSpec:
        if self.calibration_size >= self.sample_size:
            raise ValueError(
                "rand_matching.truth_splits.calibration_size must be smaller "
                "than sample_size"
            )
        if self.sample_size - self.calibration_size < 3:
            raise ValueError(
                "rand_matching.truth_splits must reserve at least three holdout rows"
            )
        if self.calibration_size < self.calibration_folds:
            raise ValueError(
                "rand_matching.truth_splits.calibration_size must provide "
                "at least one truth per calibration fold"
            )
        return self


class RandStratumSweepSpec(BaseModel):
    """Deterministic non-degenerate GTIN-ablation evaluation fixture."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)
    output: str = Field(min_length=1)
    identities_per_status: int = Field(ge=3)
    skus_per_identity: int = Field(ge=2)
    calibration_folds: int = Field(ge=3)
    seed: int

    @model_validator(mode="after")
    def _has_fold_support(self) -> RandStratumSweepSpec:
        if self.identities_per_status < self.calibration_folds:
            raise ValueError(
                "rand_matching.stratum_sweep.identities_per_status must provide "
                "at least one identity per calibration fold"
            )
        return self


class RandMatchingSpec(BaseModel):
    """Final direct SKU-to-canonical Rand Index matching contract."""

    model_config = ConfigDict(extra="forbid")

    class ConfidencePenaltyMaskSpec(BaseModel):
        """Conservative penalty for non-exact pairs with shared missing evidence."""

        model_config = ConfigDict(extra="forbid")

        enabled: bool
        critical_attributes: list[Literal["volume", "pack", "flavor"]] = Field(
            min_length=1
        )
        minimum_joint_missing: int = Field(ge=1)
        penalty_per_joint_missing: float = Field(ge=0.0, le=1.0)
        max_penalty: float = Field(ge=0.0, le=1.0)
        preserve_exact_gtin: bool

        @model_validator(mode="after")
        def _mask_is_valid(self) -> RandMatchingSpec.ConfidencePenaltyMaskSpec:
            if len(set(self.critical_attributes)) != len(self.critical_attributes):
                raise ValueError("critical_attributes must not contain duplicates")
            if self.minimum_joint_missing > len(self.critical_attributes):
                raise ValueError(
                    "minimum_joint_missing cannot exceed critical_attributes count"
                )
            return self

    class FlavorOverlapPenaltySpec(BaseModel):
        """Bounded penalty for non-exact pairs with weak flavor overlap."""

        model_config = ConfigDict(extra="forbid")

        enabled: bool
        minimum_overlap: float = Field(gt=0.0, le=1.0)
        max_penalty: float = Field(ge=0.0, le=1.0)
        preserve_exact_gtin: bool

    class TargetedVetoGatesSpec(BaseModel):
        """Hard attribute guards for non-exact automatic assignments."""

        model_config = ConfigDict(extra="forbid")

        enabled: bool
        pack_mismatch_veto: bool
        volume_mismatch_veto: bool
        package_type_mismatch_veto: bool
        brand_mismatch_veto: bool
        missing_pack_or_volume_route: Literal["human_review"]
        volume_relative_tolerance: float = Field(ge=0.0, le=1.0)
        volume_absolute_tolerance_ml: float = Field(ge=0.0)
        preserve_exact_gtin: bool

    output_dir: str = Field(min_length=1)
    truth_splits: RandTruthSplitsSpec
    stratum_sweep: RandStratumSweepSpec
    outputs: RandMatchingOutputsSpec
    top_k: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    threshold_min: float = Field(ge=-1.0, le=1.0)
    threshold_max: float = Field(ge=-1.0, le=1.0)
    threshold_step: float = Field(gt=0.0)
    threshold_by_gtin_status: dict[str, float]
    brand_conflict_veto: bool
    targeted_veto_gates: TargetedVetoGatesSpec
    confidence_penalty_mask: ConfidencePenaltyMaskSpec
    flavor_overlap_penalty: FlavorOverlapPenaltySpec
    target_recall: float = Field(gt=0.0, le=1.0)
    threshold_tie_break: list[
        Literal["rand_index", "fewest_unmatched_skus", "lowest_threshold"]
    ] = Field(min_length=3, max_length=3)
    threshold_reconciliation_scope: Literal["final_assignment_gtin_and_attribute_gates"]
    threshold_min_fold_support: int = Field(ge=2)
    calibration_proxy_source: str = Field(min_length=1)
    calibration_different_gtin_selection: Literal[
        "lowest_source_row_lexicographic_target"
    ]
    plateau_tolerance: float = Field(gt=0.0)
    plateau_min_points: int = Field(ge=2)
    # SSOT for the unmatched-SKU ITEM_ID prefix. rand_matching.py currently
    # hardcodes "UNMATCHED_" in 4 sites; its consuming agent will read this
    # key once the schema demands it here (extra="forbid" makes the schema
    # field itself the contract — a missing yaml key crashes EVERY lane at
    # import, which is exactly the enforce-your-SSOT blast radius wanted).
    unmatched_prefix: str = Field(min_length=1)

    @model_validator(mode="after")
    def _threshold_range_is_valid(self) -> RandMatchingSpec:
        if self.threshold_min > self.threshold_max:
            raise ValueError(
                "rand_matching.threshold_min must not exceed threshold_max"
            )
        expected_statuses = set(GTIN_STATUSES)
        configured_statuses = set(self.threshold_by_gtin_status)
        if configured_statuses != expected_statuses:
            raise ValueError(
                "rand_matching.threshold_by_gtin_status must cover exactly "
                f"{GTIN_STATUSES}, got {sorted(configured_statuses)}"
            )
        invalid_thresholds = {
            status: threshold
            for status, threshold in self.threshold_by_gtin_status.items()
            if not -1.0 <= float(threshold) <= 1.0
        }
        if invalid_thresholds:
            raise ValueError(
                "rand_matching.threshold_by_gtin_status values must be in "
                f"[-1, 1], got {invalid_thresholds}"
            )
        expected = set(THRESHOLD_TIE_BREAK_CRITERIA)
        configured = set(self.threshold_tie_break)
        if configured != expected or len(self.threshold_tie_break) != len(configured):
            raise ValueError(
                "rand_matching.threshold_tie_break must contain exactly one of "
                f"each criterion {sorted(expected)}; order controls priority"
            )
        expected_statuses = set(GTIN_STATUSES)
        if set(self.threshold_by_gtin_status) != expected_statuses:
            raise ValueError(
                "rand_matching.threshold_by_gtin_status must contain exactly "
                f"{sorted(expected_statuses)}"
            )
        if any(
            not -1.0 <= float(value) <= 1.0
            for value in self.threshold_by_gtin_status.values()
        ):
            raise ValueError(
                "rand_matching.threshold_by_gtin_status values must be in [-1, 1]"
            )
        return self


class MaskingSpec(BaseModel):
    """Masking augmentation contract (config/training.yaml masking:)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    profile: str = Field(min_length=1)
    frac: float = Field(ge=0.0, le=1.0)
    mask_hard_negatives: bool
    hard_negative_frac: float = Field(ge=0.0, le=1.0)
    mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_lo: float = Field(ge=0.0, lt=1.0)
    mask_hi: float = Field(gt=0.0, le=1.0)
    hard_negative_mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_negative_mask_lo: float = Field(ge=0.0, lt=1.0)
    hard_negative_mask_hi: float = Field(gt=0.0, le=1.0)
    track_visibility: bool
    track_per_epoch: bool

    @model_validator(mode="after")
    def _band_ordered(self) -> MaskingSpec:
        if self.mask_lo >= self.mask_hi:
            raise ValueError(
                f"masking.mask_lo must be < mask_hi, got "
                f"{self.mask_lo} >= {self.mask_hi}"
            )
        if self.hard_negative_mask_lo >= self.hard_negative_mask_hi:
            raise ValueError(
                "masking.hard_negative_mask_lo must be < "
                f"hard_negative_mask_hi, got {self.hard_negative_mask_lo} >= "
                f"{self.hard_negative_mask_hi}"
            )
        return self


class MaskingProfileSpec(BaseModel):
    """Config-owned overrides for a named masking experiment."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    frac: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_hard_negatives: bool | None = None
    hard_negative_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_lo: float | None = Field(default=None, ge=0.0, lt=1.0)
    mask_hi: float | None = Field(default=None, gt=0.0, le=1.0)
    hard_negative_mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_negative_mask_lo: float | None = Field(default=None, ge=0.0, lt=1.0)
    hard_negative_mask_hi: float | None = Field(default=None, gt=0.0, le=1.0)
    track_visibility: bool | None = None
    track_per_epoch: bool | None = None


class UniformityRegularizationSpec(BaseModel):
    """Batch-level anti-collapse regularization for contrastive training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    weight: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)
    min_batch_size: int = Field(ge=2)


class TrainingSpec(BaseModel):
    """Training runtime knobs (config/training.yaml training:).

    All keys are REQUIRED (no-fallback doctrine): a missing knob crashes at
    config load, not at the call site. The training lane reads them
    exclusively through lib.common.runtime().
    """

    model_config = ConfigDict(extra="forbid")

    class StructuredFeaturesSpec(BaseModel):
        model_config = ConfigDict(extra="forbid")

        enabled: bool
        append_to_text: bool
        feed_to_loss: bool
        embedding_weight: float = Field(ge=0.0)
        volume_scale_ml: float = Field(gt=0.0)
        pack_scale: float = Field(gt=0.0)
        max_set_size: int = Field(ge=1)
        # Universal symmetry: an attribute that was NOT observed must be
        # represented the same way on both sides of a pair. An unobserved pack
        # count is an implicit 1.0 (one unit) on BOTH sides, in the text
        # channel and in the numeric vector alike.
        implicit_pack_qty: float = Field(gt=0.0)

    class ModelInputSpec(BaseModel):
        """Model-input text composition (config/training.yaml training.model_input:).

        ONE profile switch plus the single granular decision the profiles
        disagree about — whether the description/breadcrumb evidence channel
        participates in the model text.  ``cleaned`` is defined as excluding
        that channel, so the contradictory combination is rejected here
        rather than being silently ignored at the call site.
        """

        model_config = ConfigDict(extra="forbid")

        profile: Literal["legacy", "cleaned"]
        include_evidence: bool

        @model_validator(mode="after")
        def _cleaned_excludes_evidence(self) -> TrainingSpec.ModelInputSpec:
            if self.profile == "cleaned" and self.include_evidence:
                raise ValueError(
                    "training.model_input.profile 'cleaned' excludes the "
                    "description/breadcrumb evidence channel by definition; "
                    "set include_evidence: false, or profile: legacy to keep it"
                )
            return self

    class ModelInputComposition(BaseModel):
        """The ACTIVE encoder-text contract, as recorded on artifacts.

        ``profile``/``include_evidence`` say WHICH composition was selected;
        ``fingerprint`` is a stable digest of those two, so an artifact can
        name its input contract and two artifacts built from different
        compositions are distinguishable without diffing the text itself.
        Written to the run trace, the checkpoint manifest, the prepared-bundle
        manifest and the ANN reuse fingerprint — one record, not four shapes.
        """

        model_config = ConfigDict(extra="forbid")

        profile: Literal["legacy", "cleaned"]
        include_evidence: bool
        fingerprint: str = Field(min_length=64, max_length=64)

        @classmethod
        def from_spec(
            cls, spec: TrainingSpec.ModelInputSpec
        ) -> TrainingSpec.ModelInputComposition:
            payload = {
                "profile": spec.profile,
                "include_evidence": spec.include_evidence,
            }
            return cls(
                **payload,
                fingerprint=hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )

    class LateEpochLrDecaySpec(BaseModel):
        """Config-owned LR reduction for the later training epochs."""

        model_config = ConfigDict(extra="forbid")

        enabled: bool
        start_epoch_fraction: float = Field(ge=0.0, le=1.0)
        multiplier: float = Field(gt=0.0, le=1.0)

    class RandomEasyNegativesSpec(BaseModel):
        """Split-safe easy-negative mixing for contrastive training."""

        model_config = ConfigDict(extra="forbid")

        enabled: bool
        ratio_to_hard: float = Field(ge=0.0)
        candidate_pool_size: int = Field(ge=1)

    base_model: str = Field(min_length=1)
    uniformity_regularization: UniformityRegularizationSpec
    late_epoch_lr_decay: LateEpochLrDecaySpec
    random_easy_negatives: RandomEasyNegativesSpec
    structured_features: StructuredFeaturesSpec
    model_input: ModelInputSpec

    # The default SentenceTransformer path is a tied-weight two-tower
    # encoder: SKU and canonical text are encoded independently, then
    # compared by cosine.  The optional cross-encoder remains a second-stage
    # reranker, never a replacement for retrieval.
    architecture: Literal["two_tower"]
    loss: Literal["contrastive", "mnrl", "triplet"]
    contrastive_margin: float = Field(gt=0.0, le=2.0)
    hard_positives: bool
    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)
    projection_dropout: float = Field(ge=0.0, lt=1.0)
    label_smoothing: float = Field(ge=0.0, lt=0.5)
    lr_scheduler: str
    max_grad_norm: float = Field(gt=0.0)
    es_patience: int = Field(ge=1)
    es_threshold: float = Field(ge=0.0)
    layer_decay: float = Field(gt=0.0, le=1.0)
    save_total_limit: int = Field(ge=1)
    rerank_max_length: int = Field(ge=8)
    batch_size_cpu: int = Field(ge=1)
    batch_size_cuda: int = Field(ge=1)
    batch_size_embed: int = Field(ge=1)
    batch_size_eval: int = Field(ge=1)
    max_seq_length: int = Field(ge=8)
    eval_steps_per_epoch: int = Field(ge=1)
    dev_fraction: float = Field(gt=0.0, lt=1.0)
    max_triples: int = Field(ge=1)
    track_datapoint_usage: bool


class PairsSpec(BaseModel):
    """Pair-label thresholds + eval-pair caps (config/training.yaml pairs:) —
    cosine gates over gate_results similarity plus the build_pairs caps
    (migrated from EDA/eda.yaml when the EDA dir was deleted, 2026-09-10)."""

    model_config = ConfigDict(extra="forbid")

    proceed_sim_threshold: float = Field(ge=0.0, le=1.0)
    hardneg_sim_threshold: float = Field(ge=0.0, le=1.0)
    max_pos_per_group: int = Field(ge=1)
    n_neg: int = Field(ge=0)
    neg_oversample: int = Field(ge=1)
    balance_train_classes: bool


class TrainingPlotsSpec(BaseModel):
    """Plot rendering (config/training.yaml plots:) — the one DPI every
    fig.savefig in the tree renders at (was EDA/eda.yaml plots.dpi)."""

    model_config = ConfigDict(extra="forbid")

    dpi: int = Field(ge=50, le=600)


class AuditSpec(BaseModel):
    """Audit-lane knobs (config/training.yaml audit:) — strip-audit sample
    size (was EDA/eda.yaml strip_audit_sample) + the blocking-feature
    audit's policy knobs (were inline BUDGET/MIN_RECALL literals in
    src/training/blocking_audit.py; audit round 2 F18, moved round 3) + the
    strip-audit similarity ladder's Jaccard band edges
    strip_ladder_bands (were an inline literal list in
    src/training/strip_audit.py; SSOT move — contiguous, ascending, each
    lo < hi) + the silent-drop guardrail's manifest knobs (SILENT_DROPS
    task 2): manifest_dir — where per-stage manifests live (a manifest
    written LAST is the stage's completion marker; consumers resolve it
    relative to the repo root via lib.common._path);
    source_export_expected_rows — the approved raw-export census
    (dataset.csv row count) the source-drift gate checks every loaded
    export against; source_drift_threshold_pct — relative row-count
    drift allowed on that census before the gate fails (0.0 = exact
    match required); manifest_stages — the registry of stages that MUST
    produce a manifest (an orchestrator lane may append its own). All four
    are optional-with-default; the defaults are mirrored explicitly in
    the yaml so the SSOT stays self-documenting."""

    model_config = ConfigDict(extra="forbid")

    strip_audit_sample: int = Field(ge=1)
    strip_ladder_bands: list[BandSpec] = Field(min_length=1)
    blocking_budget: int = Field(ge=1)
    blocking_min_recall: float = Field(gt=0.0, le=1.0)
    # ── silent-drop guardrail knobs (SILENT_DROPS task 2; tasks 3+ consume) ──
    # manifest_dir: per-stage manifests live here, one <stage>.json per
    #   stage, written LAST — a manifest's presence with status "complete"
    #   IS the stage's completion marker. Consumers resolve it relative
    #   to the repo root via lib.common._path.
    manifest_dir: str = Field(default="results/manifests", min_length=1)
    # source_export_expected_rows: the approved raw-export census — the
    #   current dataset.csv row count. The source-drift gate (task 9)
    #   compares every loaded export against this number so a changed
    #   source export is loud, never silent.
    source_export_expected_rows: int = Field(default=71_623, ge=1)
    # source_drift_threshold_pct: relative row-count drift allowed on the
    #   source export before that gate fails. 0.0 = exact match required
    #   (any row-count change trips the gate).
    source_drift_threshold_pct: float = Field(default=0.0, ge=0.0)
    # source_export_expected_sha256: byte-level fingerprint of the approved
    # raw export.  A row census alone cannot detect a substituted export
    # whose row count happens to match.
    source_export_expected_sha256: str = Field(
        default="539c247292de41d065a7e1b472a845cc122f95cee9087cf567099cf312fab88c",
        pattern=r"^[0-9a-f]{64}$",
    )
    # manifest_stages: stages that MUST produce a manifest, in pipeline
    #   order. An orchestrator lane may append its own later; this list is
    #   the required-minimum registry the verify pass walks.
    manifest_stages: list[str] = Field(
        default=[
            "dedupe",
            "data_prep",
            "labeled_pairs",
            "evaluate_models",
            "zero_shot_sims",
        ],
        min_length=0,
    )

    @model_validator(mode="after")
    def _contiguous_ascending(self) -> AuditSpec:
        """The ladder must be a contiguous ascending cover of [0, 1+eps]:
        band i's hi == band i+1's lo, and the first lo is 0.0. That is
        the shape strip_audit's `for lo, hi in bands` loop depends on —
        every SKU lands in exactly one bucket."""
        bs = self.strip_ladder_bands
        if float(bs[0].lo) != 0.0:
            raise ValueError(f"strip_ladder_bands must start at lo=0.0, got {bs[0].lo}")
        for a, b in itertools.pairwise(bs):
            if float(a.hi) != float(b.lo):
                raise ValueError(
                    "strip_ladder_bands must be contiguous (band hi == "
                    f"next lo), got {a.hi} then {b.lo}"
                )
        return self


class GateSpec(BaseModel):
    """Three-way-gate decision thresholds (config/training.yaml gate:) — the
    gate's decision table. Formerly signature defaults in
    pipeline.three_way_gate (audit round 2, F01): vol_tolerance (relative
    volume-overlap cut), raw_conf_threshold (min parse confidence to gate),
    consistency_fallback_threshold (below -> fallback, not hard_no)."""

    model_config = ConfigDict(extra="forbid")

    vol_tolerance: float = Field(gt=0.0, lt=1.0)
    raw_conf_threshold: float = Field(gt=0.0, le=1.0)
    consistency_fallback_threshold: float = Field(gt=0.0, le=1.0)


class BandSpec(BaseModel):
    """One cosine band — lo < hi enforced. Accepts the YAML list form
    [lo, hi] (the established config style) or an explicit {lo, hi}."""

    model_config = ConfigDict(extra="forbid")

    lo: float
    hi: float

    @model_validator(mode="before")
    @classmethod
    def _coerce_pair(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return {"lo": v[0], "hi": v[1]}
        return v

    @model_validator(mode="after")
    def _lo_lt_hi(self) -> BandSpec:
        if not self.lo < self.hi:
            raise ValueError(f"band must satisfy lo < hi, got [{self.lo}, {self.hi}]")
        return self


class BandsSpec(BaseModel):
    """Cosine bands (config/training.yaml bands:) — eval/rerank.

    mining_band was REMOVED (audit round 2, F21): band("mining_band") had
    zero callers — the live mining band is mining.band "0.45-0.80"."""

    model_config = ConfigDict(extra="forbid")

    eval_mining: BandSpec
    rerank_band: BandSpec


class AnnMiningSpec(BaseModel):
    """Generic cosine-ANN miner and fine-tuned refresh knobs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    target: int = Field(ge=0)
    band: str
    band_mode: str
    k: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    exclude_conflicting: bool
    refresh_enabled: bool
    refresh_every_epochs: int = Field(ge=1)
    candidate_multiplier: int = Field(ge=1)
    score_quantiles: str
    max_per_canonical: int = Field(ge=1)
    max_per_brand: int = Field(ge=1)

    @field_validator("band_mode")
    @classmethod
    def _band_mode_allowed(cls, v: str) -> str:
        allowed = {"fixed", "adaptive_quantile", "intersection"}
        if v not in allowed:
            raise ValueError(
                "mining.ann.band_mode must be one of "
                f"{', '.join(sorted(allowed))}, got {v!r}"
            )
        return v

    @field_validator("band")
    @classmethod
    def _band_parses(cls, v: str) -> str:
        try:
            lo, hi = (float(x) for x in v.split("-"))
        except ValueError as e:
            raise ValueError(
                f'mining.ann.band must be "lo-hi" floats, got {v!r}'
            ) from e
        if not lo < hi:
            raise ValueError(f"mining.ann.band must satisfy lo < hi, got {v!r}")
        return v

    @field_validator("score_quantiles")
    @classmethod
    def _quantiles_parse(cls, v: str) -> str:
        try:
            lo, hi = (float(x) for x in v.split("-"))
        except ValueError as e:
            raise ValueError('mining.ann.score_quantiles must be "lo-hi" floats') from e
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(
                "mining.ann.score_quantiles must satisfy 0 <= lo < hi <= 1"
            )
        return v


class AttributeConflictMiningSpec(BaseModel):
    """Same-brand/name critical-attribute conflict miner knobs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    target: int = Field(ge=0)
    band: str
    min_similarity: float = Field(gt=0.0, lt=1.0)
    same_product_name: bool

    @field_validator("band")
    @classmethod
    def _band_parses(cls, v: str) -> str:
        try:
            lo, hi = (float(x) for x in v.split("-"))
        except ValueError as e:
            raise ValueError(
                f'mining.attribute_conflict.band must be "lo-hi" floats, got {v!r}'
            ) from e
        if not lo < hi:
            raise ValueError(
                f"mining.attribute_conflict.band must satisfy lo < hi, got {v!r}"
            )
        return v


class MiningSpec(BaseModel):
    """Per-method hard-negative mining configuration."""

    model_config = ConfigDict(extra="forbid")

    ann: AnnMiningSpec
    attribute_conflict: AttributeConflictMiningSpec


class MiningProfileSpec(BaseModel):
    """Named experiment profile selecting which configured miners are active."""

    model_config = ConfigDict(extra="forbid")

    ann_enabled: bool
    attribute_conflict_enabled: bool


class HpoGridRowSpec(BaseModel):
    """One fixed-grid config (config/training.yaml hpo.grid[] / hpo.quick[]):
    epochs x lr x warmup-percent — second07's axes, verbatim."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup: int = Field(ge=0, le=100)


class HpoSpaceSpec(BaseModel):
    """TPE search ranges (config/training.yaml hpo.tpe_space:) — each value is
    the [lo, hi] suggest range for its knob. lr is sampled log-uniform."""

    model_config = ConfigDict(extra="forbid")

    epochs: tuple[int, int]
    lr: tuple[float, float]
    warmup_ratio: tuple[float, float]
    weight_decay: tuple[float, float]
    negative_mask_frac: tuple[float, float]
    uniformity_weight: tuple[float, float]

    @model_validator(mode="after")
    def _ranges_ordered(self) -> HpoSpaceSpec:
        for name in (
            "epochs",
            "lr",
            "warmup_ratio",
            "weight_decay",
            "negative_mask_frac",
            "uniformity_weight",
        ):
            lo, hi = getattr(self, name)
            valid = lo <= hi if name == "epochs" else lo < hi
            if not valid:
                raise ValueError(
                    f"hpo.tpe_space.{name} has an invalid range [{lo}, {hi}]"
                )
        return self


class ObjectiveSpec(BaseModel):
    """HPO selection signal per split mode (hpo.objective — test-leak fix
    2026-09-12). Both split modes select on the component-safe calibration
    Rand proxy; holdout calibration is dev-side and CV calibration is
    validation-side. A config typo changing either key dies at load."""

    model_config = ConfigDict(extra="forbid")

    holdout: Literal["rand_index_proxy"]
    cv: Literal["rand_index_proxy"]


class CollapseGuardrailSpec(BaseModel):
    """Shared embedding-collapse monitoring and penalty contract."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    profile: str = Field(min_length=1)
    unrelated_pairs: int = Field(ge=1)
    seed: int
    median_penalty_start: float = Field(ge=-1.0, le=1.0)
    p90_penalty_start: float = Field(ge=-1.0, le=1.0)
    cosine_std_floor: float = Field(gt=0.0)
    penalty_weight: float = Field(ge=0.0)
    reject_median: float = Field(ge=-1.0, le=1.0)
    max_token_frequency: float = Field(gt=0.0, le=1.0)
    operating_threshold: float = Field(ge=-1.0, le=1.0)
    crossing_rate_ceiling: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> CollapseGuardrailSpec:
        if self.median_penalty_start > self.reject_median:
            raise ValueError(
                "collapse_guardrail.median_penalty_start must not exceed reject_median"
            )
        if self.p90_penalty_start < self.median_penalty_start:
            raise ValueError(
                "collapse_guardrail.p90_penalty_start must be >= median_penalty_start"
            )
        return self


class CollapseGuardrailProfileSpec(BaseModel):
    """Config-owned operating-point override for a guardrail trial."""

    model_config = ConfigDict(extra="forbid")

    operating_threshold: float = Field(ge=-1.0, le=1.0)


class HpoSpec(BaseModel):
    """HPO lane knobs (config/training.yaml hpo:) — the fixed grid, the
    --quick smoke subset, the TPE space, and trial budgets. Formerly
    module-level literals in src/training/hpo.py / src/training/training.py."""

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(min_length=1)
    grid: list[HpoGridRowSpec] = Field(min_length=1)
    quick: list[HpoGridRowSpec] = Field(min_length=1)
    tpe_space: HpoSpaceSpec
    calibration_folds: int = Field(ge=2)
    n_trials: int = Field(ge=1)
    n_jobs: int = Field(ge=1)
    persistence: Literal["dvc", "local", "none"]
    # selection protocol (test-leak fix, 2026-09-12): WHICH signal a sweep
    # ranks configs on, pinned PER SPLIT MODE so a config typo can never
    # re-couple the holdout objective to the test quarter. Both modes stay
    # on rand_index_proxy; calibration remains separate from final holdout.
    objective: ObjectiveSpec
    # selection-mode folds skip the test-side eval + pair dump (recorded as
    # test_eval=skipped_selection_mode). MUST stay true: computing a
    # per-config test metric re-opens the leak this closed.
    selection_skip_test_eval: bool = Field()

    @field_validator("models")
    @classmethod
    def _models_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(not model.strip() for model in value):
            raise ValueError("hpo.models must contain unique non-empty registry keys")
        return value


class RerankSpec(BaseModel):
    """07e A/B decision rule (config/training.yaml rerank:) — the quantitative
    "must clearly improve": the hybrid ships only when ΔPR-AUC or ΔF1 beats
    the bi-encoder by at least these margins. Formerly inline 0.005s in
    src/training/rerank.py."""

    model_config = ConfigDict(extra="forbid")

    min_delta_pr_auc: float = Field(ge=0.0, le=1.0)
    min_delta_f1: float = Field(ge=0.0, le=1.0)


class SweepSpec(BaseModel):
    """Ablation-sweep axes (config/training.yaml sweep:) — the 07-series
    payload variants, train-frac curve, smoke/sweep sample sizes, and the
    07e rerank cross-encoder id. src/cli/colab.py derives its smoke-sample /
    train-frac / rerank defaults from this block."""

    model_config = ConfigDict(extra="forbid")

    payload_variants: list[str] = Field(min_length=1)
    train_fracs: list[float] = Field(min_length=1)
    smoke_sample: int = Field(ge=1)
    sweep_sample: int = Field(ge=1)
    rerank_model: str = Field(min_length=1)

    @field_validator("train_fracs")
    @classmethod
    def _fracs_in_unit(cls, v: list[float]) -> list[float]:
        bad = [f for f in v if not 0.0 < f < 1.0]
        if bad:
            raise ValueError(f"sweep.train_fracs must be in (0,1), got {bad}")
        return v


class WandbTrackingSpec(BaseModel):
    """W&B configuration; its API key is environment-only."""

    model_config = ConfigDict(extra="forbid")

    project: str = Field(min_length=1)
    mode: Literal["online", "offline", "disabled"]


class TrackingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb: WandbTrackingSpec


class ValidationInferenceSpec(BaseModel):
    """Post-training inference performed before a Colab worker is published."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    source_csv: str = Field(min_length=1)
    input_csv: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    device: Literal["cpu", "cuda"]
    error_threshold: float = Field(ge=0.0, le=1.0)
    thresholds: list[float] = Field(min_length=1)

    @field_validator("thresholds")
    @classmethod
    def _valid_thresholds(cls, values: list[float]) -> list[float]:
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("validation inference thresholds must be in [0, 1]")
        if values != sorted(set(values)):
            raise ValueError(
                "validation inference thresholds must be unique and sorted"
            )
        return values


class RuntimePackagesSpec(BaseModel):
    """Colab VM distributions the launcher asserts, one list per lane.

    The Colab image already ships most of the training stack, so these are the
    distributions the launcher hands to the VM installer — not an environment
    lock.  ``prepared`` is the local-bundle training lane (the VM only trains
    and scores); ``full`` adds the HPO/zero-shot extras.
    """

    model_config = ConfigDict(extra="forbid")

    prepared: list[str] = Field(min_length=1)
    full: list[str] = Field(min_length=1)

    @field_validator("prepared", "full")
    @classmethod
    def _clean_entries(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("runtime package entries must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("runtime package entries must be unique")
        return values


class ColabSpec(BaseModel):
    """Remote checkout/runtime settings for the Colab training lane."""

    model_config = ConfigDict(extra="forbid")

    repository: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    git_remote_name: str = Field(min_length=1)
    remote_root: str = Field(min_length=1)
    session: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    remote_data_prep: Literal[False] = False
    training_dataset_csv: str = Field(min_length=1)
    runtime_packages: RuntimePackagesSpec
    prefer_uv_install: bool
    hpo_mode: Literal["sequential", "parallel_same_vm"]
    hpo_workers: int = Field(ge=1, le=3)
    train_workers: int = Field(ge=1, le=12)
    smoke_workers: int = Field(ge=1, le=3)
    mixed_train_workers: Literal[1]
    mixed_sims_workers: Literal[1]
    mixed_mining_profile: str = Field(min_length=1)
    sims_model: str = Field(min_length=1)
    log_poll_seconds: int = Field(ge=1, le=30)
    probe_timeout_seconds: int = Field(ge=60, le=1800)
    probe_retries: int = Field(ge=1, le=10)
    probe_retry_backoff_seconds: int = Field(ge=1, le=120)
    artifact_repo_id: str = Field(min_length=1)
    artifact_repo_private: bool
    mask_effect_after_train: bool
    smoke_epochs: int = Field(ge=1)
    dvc_remote_url: str = Field(min_length=1)
    dagshub_repo: str = Field(min_length=1)
    worker_timeout_seconds: int = Field(ge=60)
    result_download_timeout_seconds: int = Field(ge=60)
    result_download_heartbeat_seconds: int = Field(ge=1, le=300)
    result_archive_name: str = Field(min_length=1)
    result_manifest_name: str = Field(min_length=1)
    result_download_excluded_dirs: list[str] = Field(min_length=1)
    worker_monitor_seconds: int = Field(ge=1, le=300)
    dvc_enabled: bool = True
    dvc_workers: int = Field(ge=1, le=3)
    dvc_jobs: int = Field(ge=1, le=32)
    result_events_file: str = Field(min_length=1)
    dvc_events_file: str = Field(min_length=1)
    dvc_push_retries: int = Field(ge=1, le=10)
    dvc_push_backoff_seconds: int = Field(ge=1, le=120)
    validation_inference: ValidationInferenceSpec


class TrainingConfig(BaseModel):
    """config/training.yaml — the training lane's OWN config (in its dir).

    Validated as a whole at load; lib.common exposes it through
    training_cfg()/runtime()/band()/SSOT_LOSS/SSOT_CONTRASTIVE_MARGIN.
    Cross-config checks (sim_columns vs the root models registry) happen in
    lib.common at merge time — this model can't see the root config by
    design (one file, one contract).
    """

    model_config = ConfigDict(extra="forbid")

    sim_columns: dict[str, str] = Field(min_length=1)
    masking: MaskingSpec
    masking_profiles: dict[str, MaskingProfileSpec] = Field(min_length=1)
    split: SplitSpec
    evaluation: EvaluationSpec
    collapse_guardrail: CollapseGuardrailSpec
    collapse_guardrail_profiles: dict[str, CollapseGuardrailProfileSpec] = Field(
        min_length=1
    )
    gate: GateSpec
    training: TrainingSpec
    pairs: PairsSpec
    plots: TrainingPlotsSpec
    audit: AuditSpec
    bands: BandsSpec
    mining: MiningSpec
    mining_profiles: dict[str, MiningProfileSpec] = Field(default_factory=dict)
    hpo: HpoSpec
    rerank: RerankSpec
    sweep: SweepSpec
    tracking: TrackingSpec
    colab: ColabSpec
    rand_matching: RandMatchingSpec
    # The NER lane's legacy settings live under the training SSOT too. Their
    # shape is intentionally open while the older standalone scripts are
    # retired; core.common owns parsing/path expansion for every consumer.
    ner: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _fixed_epoch_space_matches_training(self) -> TrainingConfig:
        """Allow a fixed epoch budget only when it matches the SSOT budget."""
        lo, hi = self.hpo.tpe_space.epochs
        if lo == hi and lo != self.training.epochs:
            raise ValueError(
                "hpo.tpe_space.epochs may be fixed only to "
                f"training.epochs={self.training.epochs}, got [{lo}, {hi}]"
            )
        return self

    @model_validator(mode="after")
    def _masking_profile_is_registered(self) -> TrainingConfig:
        if self.masking.profile not in self.masking_profiles:
            raise ValueError(
                "masking.profile must name a configured masking_profiles entry: "
                f"{self.masking.profile!r} not in {sorted(self.masking_profiles)}"
            )
        for name, profile in self.masking_profiles.items():
            values = profile.model_dump(exclude_none=True)
            lo = values.get("mask_lo", self.masking.mask_lo)
            hi = values.get("mask_hi", self.masking.mask_hi)
            neg_lo = values.get(
                "hard_negative_mask_lo", self.masking.hard_negative_mask_lo
            )
            neg_hi = values.get(
                "hard_negative_mask_hi", self.masking.hard_negative_mask_hi
            )
            if lo >= hi or neg_lo >= neg_hi:
                raise ValueError(
                    f"masking_profiles.{name} has an invalid mask extent band"
                )
        return self

    @model_validator(mode="after")
    def _collapse_profile_is_registered(self) -> TrainingConfig:
        if self.collapse_guardrail.profile not in self.collapse_guardrail_profiles:
            raise ValueError(
                "collapse_guardrail.profile must name a configured "
                "collapse_guardrail_profiles entry: "
                f"{self.collapse_guardrail.profile!r} not in "
                f"{sorted(self.collapse_guardrail_profiles)}"
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY CONTRACTS — pipeline transforms
# ═══════════════════════════════════════════════════════════════════════════

GateDecision = Literal["hard_no", "fallback", "proceed"]
GATE_DECISIONS: frozenset[str] = frozenset({"hard_no", "fallback", "proceed"})

# digit-verdict vocabulary (data_pipe token_verdict + the reference CSV)
TokenVerdict = Literal["strip", "keep_brand", "keep_nutrient", "keep_name"]
TOKEN_VERDICTS: frozenset[str] = frozenset(
    {"strip", "keep_brand", "keep_nutrient", "keep_name"}
)


class ExtractedAttributes(BaseModel):
    """extract_all() output — per-row extracted attributes feeding BOTH the
    canonical build and the gate. Confidence bounds asserted (a confidence
    of 1.7 would silently pass every >= 0.85 gate check)."""

    model_config = ConfigDict(extra="forbid")

    flavor: str
    type: str
    volume_ml: float = Field(ge=0.0)
    volume_confidence: float = Field(ge=0.0, le=1.0)
    volume_raw: str
    volume_status: str
    pack_qty: int = Field(ge=1)
    pack_confidence: float = Field(ge=0.0, le=1.0)
    package_types: list[str] = Field(default_factory=list)
    package_materials: list[str] = Field(default_factory=list)
    flavor_set: set[str] = Field(default_factory=set)
    carbonation_set: set[str] = Field(default_factory=set)
    sweetener_set: set[str] = Field(default_factory=set)
    pulp_set: set[str] = Field(default_factory=set)


class GateResult(BaseModel):
    """three_way_gate() output — decision domain + non-empty reason."""

    model_config = ConfigDict(extra="forbid")

    decision: GateDecision
    reason: str = Field(min_length=1)


class CanonicalRecord(BaseModel):
    """generate_canonical() output — one canonical per (valid-checksum) GTIN.
    Set fields stay sets for the gate's intersection logic; the CSV write
    sorts them for byte-determinism (run_within_brand_pipeline)."""

    model_config = ConfigDict(extra="forbid")

    gtin: str = Field(min_length=1)
    canonical: str
    mode_brand: str
    mode_flavor: str
    mode_type: str
    salient_ngrams: list[str]
    dropped_redundant_ngrams: list[str]
    volume_set: set[float]
    pack_set: set[int]
    package_type_set: set[str] = Field(default_factory=set)
    package_material_set: set[str] = Field(default_factory=set)
    flavor_set: set[str] = Field(default_factory=set)
    carbonation_set: set[str] = Field(default_factory=set)
    sweetener_set: set[str] = Field(default_factory=set)
    pulp_set: set[str] = Field(default_factory=set)
    volume_confidence: float = Field(ge=0.0, le=1.0)
    pack_confidence: float = Field(ge=0.0, le=1.0)
    volume_consistency: float = Field(ge=0.0, le=1.0)
    pack_consistency: float = Field(ge=0.0, le=1.0)
    n_titles: int = Field(ge=1)
    # Source evidence retained for feature ablations.  These are deliberately
    # not folded into `canonical`: that text is the frozen gate/model input.
    description_evidence: list[str] = Field(default_factory=list)
    breadcrumb_evidence: list[str] = Field(default_factory=list)

    @field_validator("volume_set")
    @classmethod
    def _volumes_positive(cls, v: set[float]) -> set[float]:
        bad = [x for x in v if x <= 0 or math.isnan(x)]
        if bad:
            raise ValueError(f"volume_set must hold positive volumes, got {bad}")
        return v

    @field_validator("pack_set")
    @classmethod
    def _packs_positive(cls, v: set[int]) -> set[int]:
        bad = [x for x in v if x < 1]
        if bad:
            raise ValueError(f"pack_set must hold pack counts >= 1, got {bad}")
        return v


# ── index-pair containers (numpy shape contracts) ─────────────────────────


class PairArrays(BaseModel):
    """(N, 2) int index pairs with the SHAPE assertion: indices in range of
    n_payload. Both pos and neg containers satisfy this contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pos: np.ndarray
    neg: np.ndarray
    n_payload: int = Field(ge=0)

    @field_validator("pos", "neg")
    @classmethod
    def _pair_matrix(cls, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v)
        if arr.size == 0:
            if arr.shape != (0, 2):
                raise ValueError(
                    f"empty pair array must be shape (0,2), got {arr.shape}"
                )
            return arr.astype(int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"pair array must be (N,2), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"pair indices must be integer, got dtype {arr.dtype}")
        if (arr < 0).any():
            raise ValueError("pair indices must be >= 0")
        return arr

    @model_validator(mode="after")
    def _indices_in_range(self) -> PairArrays:
        for name in ("pos", "neg"):
            arr = getattr(self, name)
            if arr.size and int(arr.max()) >= self.n_payload:
                raise ValueError(
                    f"{name} index {int(arr.max())} out of range "
                    f"(n_payload={self.n_payload})"
                )
        return self


class TrainingStats(BaseModel):
    """build_training_data stats — every count the lane prints; nothing
    drops silently (transparency contract)."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int = Field(ge=0)
    n_sku_with_canonical: int = Field(ge=0)
    n_pos_empty_dropped: int = Field(ge=0)
    n_empty_sku_texts: int = Field(ge=0)
    n_empty_canon_texts: int = Field(ge=0)
    n_canonicals: int = Field(ge=0)
    n_pos_gate_rows: int = Field(ge=0)
    n_neg_same_canonical_dropped: int = Field(ge=0)
    n_neg_gate_rows: int = Field(ge=0)
    n_neg_resolved: int = Field(ge=0)
    n_neg_hard_no_band: int = Field(ge=0)
    n_neg_forward_resolved: int = Field(ge=0)
    n_neg_reverse_resolved: int = Field(ge=0)
    n_neg_forward_source_unresolved: int = Field(ge=0)
    n_neg_forward_target_unresolved: int = Field(ge=0)
    n_neg_reverse_source_unresolved: int = Field(ge=0)
    n_neg_reverse_target_unresolved: int = Field(ge=0)
    n_neg_resolution_dropped: int = Field(ge=0)
    n_neg_dropped: int = Field(ge=0)
    n_targeted_attribute_candidates: int = Field(default=0, ge=0)
    n_targeted_attribute_resolved: int = Field(default=0, ge=0)


class TrainingData(BaseModel):
    """build_training_data output — the OFFICIAL pair bundle. Cross-shape
    assertions: len(row_bc) == len(payload); every pos/neg index in range;
    gtin_to_row targets < len(payload)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: list[str]
    structured_features: list[list[float]]
    row_bc: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    targeted_attribute_neg: np.ndarray = Field(
        default_factory=lambda: np.empty((0, 2), dtype=int)
    )
    gtin_to_row: dict[str, int]
    stats: TrainingStats

    @field_validator("row_bc")
    @classmethod
    def _row_bc_vector(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError(f"row_bc must be 1-D, got shape {arr.shape}")
        return arr

    @field_validator("pos", "neg", "targeted_attribute_neg")
    @classmethod
    def _pair_matrix(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.size == 0:
            if arr.shape != (0, 2):
                raise ValueError(
                    f"empty pair array must be shape (0,2), got {arr.shape}"
                )
            return arr.astype(int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"pair array must be (N,2), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"pair indices must be integer, got {arr.dtype}")
        if (arr < 0).any():
            raise ValueError("pair indices must be >= 0")
        return arr

    @model_validator(mode="after")
    def _shapes_agree(self) -> TrainingData:
        n = len(self.payload)
        if len(self.row_bc) != n:
            raise ValueError(f"row_bc length {len(self.row_bc)} != payload length {n}")
        if len(self.structured_features) != n:
            raise ValueError(
                "structured_features length "
                f"{len(self.structured_features)} != payload length {n}"
            )
        # Rectangularity (audit gap closed): consumers call np.asarray() on
        # this list, so a RAGGED one dies several frames away with numpy's
        # "inhomogeneous shape" instead of with the field named.
        # core.structured_features.vector() is fixed-width by construction;
        # this assertion is what keeps it that way.
        if self.structured_features:
            widths = {len(row) for row in self.structured_features}
            if len(widths) > 1:
                raise ValueError(
                    "structured_features must be a rectangular matrix, got "
                    f"row widths {sorted(widths)}"
                )
        for name in ("pos", "neg", "targeted_attribute_neg"):
            arr = getattr(self, name)
            if arr.size and int(arr.max()) >= n:
                raise ValueError(
                    f"{name} index {int(arr.max())} out of range (payload={n})"
                )
        bad_rows = {g: r for g, r in self.gtin_to_row.items() if r >= n or r < 0}
        if bad_rows:
            raise ValueError(f"gtin_to_row targets out of range: {bad_rows}")
        return self


class MaskAuditEntry(BaseModel):
    """One masked-copy audit row — realized extent in [0,1], indices in
    range of the ORIGINAL payload length (anchor/pair sides)."""

    model_config = ConfigDict(extra="forbid")

    anchor_payload_idx: int = Field(ge=0)
    copy_payload_idx: int = Field(ge=0)
    pair_payload_idx: int = Field(ge=0)
    barcode: str
    realized_extent: float = Field(ge=0.0, le=1.0)
    configured_mask_lo: float | None = Field(default=None, ge=0.0, le=1.0)
    configured_mask_hi: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    anchor_text: str
    masked_text: str
    population: str = "positive"


class MaskingResult(BaseModel):
    """augment_positives output — pos' extends the payload space; the SHAPE
    assertion is that payload' and row_bc' stay length-locked and every new
    pos' index is in range of the EXTENDED payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pos: np.ndarray
    payload: list[str]
    row_bc: np.ndarray
    n_added: int = Field(ge=0)
    audit: list[MaskAuditEntry]

    @field_validator("pos")
    @classmethod
    def _pair_matrix(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.size == 0:
            if arr.shape != (0, 2):
                raise ValueError(
                    f"empty pair array must be shape (0,2), got {arr.shape}"
                )
            return arr.astype(int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"pair array must be (N,2), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"pair indices must be integer, got {arr.dtype}")
        if (arr < 0).any():
            raise ValueError("pair indices must be >= 0")
        return arr

    @model_validator(mode="after")
    def _shapes_agree(self) -> MaskingResult:
        n = len(self.payload)
        if len(self.row_bc) != n:
            raise ValueError(f"row_bc length {len(self.row_bc)} != payload length {n}")
        if self.pos.size and int(self.pos.max()) >= n:
            raise ValueError(
                f"pos index {int(self.pos.max())} out of range (payload={n})"
            )
        if self.n_added != len(self.audit):
            raise ValueError(f"n_added {self.n_added} != audit rows {len(self.audit)}")
        return self

    def audit_dicts(self) -> list[dict]:
        """The audit rows as plain dicts (the historical return surface —
        train.py dumps them straight into a DataFrame)."""
        return [a.model_dump() for a in self.audit]


class DataTuple(BaseModel):
    """The 8-tuple crossing into train_one_config:
    (df, payload, structured_features, row_bc, country, pos, hp_pairs, emb0).

    Cross-shape assertions (the exact class of bug the audits found —
    country shorter than payload after canonical/masked extension):
      - len(row_bc) == len(payload) >= len(df)
      - len(country) >= len(payload)   (padded for canonicals/masked copies)
      - pos / hp_pairs indices < len(payload)
      - emb0.shape == (len(payload), d)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_df: int = Field(ge=0)
    payload: list[str]
    structured_features: np.ndarray
    row_bc: np.ndarray
    country: np.ndarray
    pos: np.ndarray
    hp_pairs: np.ndarray
    emb0: np.ndarray

    @field_validator("structured_features")
    @classmethod
    def _structured_matrix(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 2:
            raise ValueError(f"structured_features must be 2-D, got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"structured_features must be float, got {arr.dtype}")
        return arr

    @field_validator("row_bc", "country")
    @classmethod
    def _vector_1d(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError(f"side array must be 1-D, got shape {arr.shape}")
        return arr

    @field_validator("pos", "hp_pairs")
    @classmethod
    def _pair_matrix(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.size == 0:
            if arr.shape != (0, 2):
                raise ValueError(
                    f"empty pair array must be shape (0,2), got {arr.shape}"
                )
            return arr.astype(int)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"pair array must be (N,2), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"pair indices must be integer, got {arr.dtype}")
        if (arr < 0).any():
            raise ValueError("pair indices must be >= 0")
        return arr

    @field_validator("emb0")
    @classmethod
    def _emb_matrix(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.size == 0:
            if arr.ndim != 2:
                raise ValueError(f"empty emb must be 2-D, got shape {arr.shape}")
            return arr
        if arr.ndim != 2:
            raise ValueError(f"emb0 must be (N, d), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"emb0 must be float, got {arr.dtype}")
        return arr

    @model_validator(mode="after")
    def _shapes_agree(self) -> DataTuple:
        n = len(self.payload)
        if len(self.structured_features) != n:
            raise ValueError(
                f"structured_features length {len(self.structured_features)} != payload length {n}"
            )
        if len(self.row_bc) != n:
            raise ValueError(f"row_bc length {len(self.row_bc)} != payload length {n}")
        if n < self.n_df:
            raise ValueError(f"payload length {n} < df rows {self.n_df}")
        if len(self.country) < n:
            raise ValueError(
                f"country length {len(self.country)} < payload length {n} — "
                "pad BEFORE building the tuple (train.py country-pad block)"
            )
        for name in ("pos", "hp_pairs"):
            arr = getattr(self, name)
            if arr.size and int(arr.max()) >= n:
                raise ValueError(
                    f"{name} index {int(arr.max())} out of range (payload={n})"
                )
        if self.emb0.size and self.emb0.shape[0] != n:
            raise ValueError(f"emb0 rows {self.emb0.shape[0]} != payload length {n}")
        return self


class TrainConfig(BaseModel):
    """The per-config dict train_one_config receives (DEFAULT_CFG/HPO/grid
    rows). Constraints mirror the HF Trainer's own validations — but fire
    at the boundary with a named field instead of an anonymous traceback."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["two_tower"]
    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)
    projection_dropout: float = Field(ge=0.0, lt=1.0)
    label_smoothing: float = Field(ge=0.0, lt=0.5)
    random_easy_enabled: bool
    random_easy_ratio_to_hard: float = Field(ge=0.0)
    random_easy_candidate_pool_size: int = Field(ge=1)
    lr_scheduler: str
    max_grad_norm: float = Field(gt=0.0)
    patience: int = Field(ge=1)
    es_threshold: float = Field(ge=0.0)
    uniformity_weight: float = Field(ge=0.0)
    late_epoch_decay_enabled: bool
    late_epoch_decay_start_fraction: float = Field(ge=0.0, le=1.0)
    late_epoch_decay_multiplier: float = Field(gt=0.0, le=1.0)


class FoldSets(BaseModel):
    """component_folds output — k disjoint barcode sets covering the
    pair-graph components. Disjointness is the anti-leak guarantee; a
    straddling barcode would put one product in two folds."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    folds: list[set[str]]

    @model_validator(mode="after")
    def _folds_disjoint(self) -> FoldSets:
        seen: set[str] = set()
        for i, fold in enumerate(self.folds):
            overlap = seen & fold
            if overlap:
                raise ValueError(
                    f"folds[{i}] leaks {len(overlap)} barcodes seen in earlier "
                    f"folds (e.g. {sorted(overlap)[:3]})"
                )
            seen |= fold
        return self


class CalibrationPartition(BaseModel):
    """folds.partition_component_pairs output — the DEV split into a fit half
    and a component-safe calibration half.

    All four pools are same-shaped ``(n, 2)`` int arrays and their ORDER is
    the producer's positional contract, so a wrong-order unpack at the call
    site is invisible without this model; ``pools()`` is the single accessor.
    Validated on construction (the sibling FoldSets boundary is validated the
    same way): the populations are conserved, and no identity crosses the
    calibration boundary. The boundary is stated two ways because the two
    pools are not the same shape of graph — a positive pair sits inside ONE
    component (both endpoints share its barcode), so its barcodes must be
    disjoint across the halves, while a negative pair BY CONSTRUCTION links
    two different components, so the same guarantee can only hold at the
    granularity of the unordered identity pair: both endpoints of a negative
    must be reserved together, otherwise the mirrored orientation of the same
    product pair stays behind in the fit half.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    positive_fit: np.ndarray
    positive_reserved: np.ndarray
    negative_fit: np.ndarray
    negative_reserved: np.ndarray
    # the barcode vector the boundary contract is stated over (not payload)
    row_bc: np.ndarray = Field(exclude=True, repr=False)
    n_positive_pairs: int = Field(ge=0)
    n_negative_pairs: int = Field(ge=0)
    n_negative_pairs_excluded: int = Field(ge=0, default=0)

    def pools(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The positional contract, in declaration order."""
        return (
            self.positive_fit,
            self.positive_reserved,
            self.negative_fit,
            self.negative_reserved,
        )

    def _identity(self, row: object) -> str:
        return str(self.row_bc[int(row)]).strip()

    def identities(self, pool: np.ndarray) -> set[str]:
        """Barcode identities touched by one pool."""
        return {self._identity(row) for row in pool.ravel()}

    def identity_pairs(self, pool: np.ndarray) -> set[frozenset[str]]:
        """Unordered identity pair of every pair in one pool."""
        return {frozenset((self._identity(a), self._identity(b))) for a, b in pool}

    @model_validator(mode="after")
    def _populations_and_disjoint_identities(self) -> CalibrationPartition:
        for name, pool in zip(
            ("positive_fit", "positive_reserved", "negative_fit", "negative_reserved"),
            self.pools(),
            strict=True,
        ):
            if pool.ndim != 2 or pool.shape[1] != 2:
                raise ValueError(
                    f"{name} must be an (n, 2) pair array, got shape {pool.shape}"
                )
        if (
            len(self.positive_fit) + len(self.positive_reserved)
            != self.n_positive_pairs
        ):
            raise ValueError("positive calibration partition changed its population")
        if (
            len(self.negative_fit)
            + len(self.negative_reserved)
            + self.n_negative_pairs_excluded
            != self.n_negative_pairs
        ):
            raise ValueError("negative calibration partition changed its population")
        shared = self.identities(self.positive_fit) & self.identities(
            self.positive_reserved
        )
        if shared:
            raise ValueError(
                f"{len(shared)} positive identities cross the calibration "
                f"boundary (e.g. {sorted(shared)[:3]})"
            )
        reserved_positive_identities = self.identities(self.positive_reserved)
        fit_positive_identities = self.identities(self.positive_fit)
        fit_negative_leak = (
            self.identities(self.negative_fit) & reserved_positive_identities
        )
        if fit_negative_leak:
            raise ValueError(
                f"{len(fit_negative_leak)} reserved positive identities occur in "
                f"fit negatives (e.g. {sorted(fit_negative_leak)[:3]})"
            )
        reserved_negative_leak = (
            self.identities(self.negative_reserved) & fit_positive_identities
        )
        if reserved_negative_leak:
            raise ValueError(
                f"{len(reserved_negative_leak)} fit positive identities occur in "
                f"reserved negatives (e.g. {sorted(reserved_negative_leak)[:3]})"
            )
        crossed = self.identity_pairs(self.negative_fit) & self.identity_pairs(
            self.negative_reserved
        )
        if crossed:
            raise ValueError(
                f"{len(crossed)} negative identity pairs cross the calibration "
                f"boundary (e.g. {sorted(sorted(pair) for pair in crossed)[:2]})"
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# FRAME CONTRACTS — DataFrame column/domain checks at CSV boundaries
# ═══════════════════════════════════════════════════════════════════════════

CANONICAL_RECORDS_COLUMNS: tuple[str, ...] = (
    "gtin",
    "canonical",
    "mode_brand",
    "mode_flavor",
    "mode_type",
    "salient_ngrams",
    "dropped_redundant_ngrams",
    "volume_set",
    "pack_set",
    "package_type_set",
    "package_material_set",
    "flavor_set",
    "carbonation_set",
    "sweetener_set",
    "pulp_set",
    "volume_confidence",
    "pack_confidence",
    "volume_consistency",
    "pack_consistency",
    "n_titles",
    "description_evidence",
    "breadcrumb_evidence",
)

GATE_RESULTS_COLUMNS: tuple[str, ...] = (
    "gtin1",
    "gtin2",
    "canon1",
    "canon2",
    "gate_decision",
    "gate_reason",
    "similarity",
)

LABELED_PAIRS_COLUMNS: tuple[str, ...] = ("gtin1", "gtin2", "true_label")

ZERO_SHOT_TRACE_COLUMNS: tuple[str, ...] = (
    "gtin1",
    "gtin2",
    "gate_decision",
    "gate_reason",
    "canonical1",
    "canonical2",
    "canonical_model_text1",
    "canonical_model_text2",
    "model_input_text1",
    "model_input_text2",
    "source_row_ids1",
    "source_row_ids2",
    "source_sku_ids1",
    "source_sku_ids2",
    "source_metadata1",
    "source_metadata2",
    "canonical_metadata1",
    "canonical_metadata2",
    "mask_status1",
    "mask_status2",
    "mask_applied1",
    "mask_applied2",
    "mask_realized_extent1",
    "mask_realized_extent2",
    "mask_config_fingerprint",
    "model_keys",
    "lineage_id",
)


class ZeroShotTraceRow(BaseModel):
    """One zero-shot pair row with model-input and source lineage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gtin1: StrictStr = Field(min_length=1)
    gtin2: StrictStr = Field(min_length=1)
    gate_decision: StrictStr = Field(min_length=1)
    gate_reason: StrictStr = Field(min_length=1)
    canonical1: StrictStr = Field(min_length=1)
    canonical2: StrictStr = Field(min_length=1)
    canonical_model_text1: StrictStr = Field(min_length=1)
    canonical_model_text2: StrictStr = Field(min_length=1)
    model_input_text1: StrictStr = Field(min_length=1)
    model_input_text2: StrictStr = Field(min_length=1)
    source_row_ids1: StrictStr = Field(min_length=2)
    source_row_ids2: StrictStr = Field(min_length=2)
    source_sku_ids1: StrictStr = Field(min_length=2)
    source_sku_ids2: StrictStr = Field(min_length=2)
    source_metadata1: StrictStr = Field(min_length=2)
    source_metadata2: StrictStr = Field(min_length=2)
    canonical_metadata1: StrictStr = Field(min_length=2)
    canonical_metadata2: StrictStr = Field(min_length=2)
    mask_status1: StrictStr = Field(min_length=1)
    mask_status2: StrictStr = Field(min_length=1)
    mask_applied1: bool
    mask_applied2: bool
    mask_realized_extent1: float = Field(ge=0.0, le=1.0)
    mask_realized_extent2: float = Field(ge=0.0, le=1.0)
    mask_config_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    model_keys: StrictStr = Field(min_length=3)
    lineage_id: StrictStr = Field(min_length=16, max_length=64)


def check_zero_shot_similarity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the traceable zero-shot output without dropping rows."""
    columns = tuple(df.columns)
    missing = [c for c in ZERO_SHOT_TRACE_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"zero-shot output missing trace columns: {missing}")
    sim_columns = [c for c in columns if c.startswith("sim_")]
    if not sim_columns:
        raise ValueError("zero-shot output has no similarity columns")
    if df[list(ZERO_SHOT_TRACE_COLUMNS)].isna().any().any():
        raise ValueError("zero-shot trace columns contain missing values")
    for column in sim_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"zero-shot similarity column {column!r} has non-numeric values"
            )
    for row in df[list(ZERO_SHOT_TRACE_COLUMNS)].to_dict("records"):
        ZeroShotTraceRow.model_validate(row)
    return df


CROSS_COUNTRY_PAIR_COLUMNS: tuple[str, ...] = (
    "sku_id_a",
    "sku_id_b",
    "cross_country",
    "gtin",
    "country_a",
    "country_b",
)


class CrossCountryPairRow(BaseModel):
    """One generated second04 cross-country hard-positive manifest row."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sku_id_a: StrictStr = Field(min_length=1)
    sku_id_b: StrictStr = Field(min_length=1)
    cross_country: StrictBool
    gtin: StrictStr = Field(min_length=1)
    country_a: StrictStr = Field(min_length=1)
    country_b: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_cross_country_pair(self) -> "CrossCountryPairRow":
        from core.gtin import is_valid_gtin_checksum

        if self.sku_id_a == self.sku_id_b:
            raise ValueError("cross-country manifest cannot contain self-pairs")
        if not self.cross_country or self.country_a == self.country_b:
            raise ValueError("manifest rows must represent different countries")
        if not is_valid_gtin_checksum(self.gtin):
            raise ValueError(f"manifest GTIN is invalid: {self.gtin!r}")
        return self


def check_cross_country_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete second04 CSV frame without dropping rows."""
    columns = tuple(df.columns)
    if columns != CROSS_COUNTRY_PAIR_COLUMNS:
        raise ValueError(
            f"cross-country pair frame columns {columns} != contract "
            f"{CROSS_COUNTRY_PAIR_COLUMNS}"
        )
    rows = [CrossCountryPairRow.model_validate(row) for row in df.to_dict("records")]
    return df


def check_canonical_records_frame(df: pd.DataFrame) -> pd.DataFrame:
    """canonical_records.csv contract: exact columns, unique GTINs,
    non-empty canonical strings, confidences/consistencies in [0,1],
    n_titles >= 1. Returns df unchanged (assert-only boundary)."""
    cols = tuple(df.columns)
    if cols != CANONICAL_RECORDS_COLUMNS:
        raise ValueError(
            f"canonical_records frame columns {cols} != contract "
            f"{CANONICAL_RECORDS_COLUMNS}"
        )
    if df["gtin"].duplicated().any():
        dups = df.loc[df["gtin"].duplicated(), "gtin"].unique()[:3]
        raise ValueError(f"canonical_records has duplicate GTINs: {list(dups)}")
    empty = (df["canonical"].fillna("").astype(str).str.strip() == "").sum()
    if empty:
        raise ValueError(f"{empty} empty canonical strings")
    for col in (
        "volume_confidence",
        "pack_confidence",
        "volume_consistency",
        "pack_consistency",
    ):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any() or ((s < 0) | (s > 1)).any():
            bad = int(((s < 0) | (s > 1)).sum())
            raise ValueError(
                f"canonical_records.{col} outside [0,1]: {bad} rows "
                f"(min {s.min()}, max {s.max()})"
            )
    n = pd.to_numeric(df["n_titles"], errors="coerce")
    if n.isna().any() or (n < 1).any():
        raise ValueError(f"canonical_records.n_titles < 1 on {(n < 1).sum()} rows")
    return df


def check_gate_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """gate_results.csv contract: exact columns, decision domain, similarity
    in [0,1] (Jaccard), non-empty GTIN pairs, gtin1 != gtin2."""
    cols = tuple(df.columns)
    if cols != GATE_RESULTS_COLUMNS:
        raise ValueError(
            f"gate_results frame columns {cols} != contract {GATE_RESULTS_COLUMNS}"
        )
    bad = set(df["gate_decision"].unique()) - set(GATE_DECISIONS)
    if bad:
        raise ValueError(f"gate_decision outside domain: {sorted(bad)}")
    s = pd.to_numeric(df["similarity"], errors="coerce")
    if s.isna().any() or ((s < 0) | (s > 1)).any():
        raise ValueError(
            f"gate_results.similarity outside [0,1]: min {s.min()}, max {s.max()}"
        )
    gtin_empty = df["gtin1"].fillna("").astype(str).str.strip().eq("") | df[
        "gtin2"
    ].fillna("").astype(str).str.strip().eq("")
    if gtin_empty.any():
        raise ValueError(f"{int(gtin_empty.sum())} rows with empty gtin endpoint")
    self_pairs = (df["gtin1"] == df["gtin2"]).sum()
    if self_pairs:
        raise ValueError(f"{int(self_pairs)} self-pairs (gtin1 == gtin2)")
    empty_reason = (df["gate_reason"].fillna("").astype(str).str.strip() == "").sum()
    if empty_reason:
        raise ValueError(f"{int(empty_reason)} rows with empty gate_reason")
    return df


def check_labeled_pairs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """labeled_pairs.csv contract: exact columns, true_label in {0,1},
    non-empty GTIN endpoints, no duplicate (gtin1, gtin2) rows."""
    cols = tuple(df.columns)
    if cols != LABELED_PAIRS_COLUMNS:
        raise ValueError(
            f"labeled_pairs frame columns {cols} != contract {LABELED_PAIRS_COLUMNS}"
        )
    bad = set(pd.to_numeric(df["true_label"], errors="coerce").dropna().unique()) - {
        0,
        1,
    }
    if bad:
        raise ValueError(f"true_label outside {{0,1}}: {sorted(bad)}")
    if pd.to_numeric(df["true_label"], errors="coerce").isna().any():
        raise ValueError("true_label has non-numeric rows")
    gtin_empty = df["gtin1"].fillna("").astype(str).str.strip().eq("") | df[
        "gtin2"
    ].fillna("").astype(str).str.strip().eq("")
    if gtin_empty.any():
        raise ValueError(f"{int(gtin_empty.sum())} rows with empty gtin endpoint")
    dups = df.duplicated(subset=["gtin1", "gtin2"]).sum()
    if dups:
        raise ValueError(f"{int(dups)} duplicate (gtin1, gtin2) rows")
    return df


# ── consolidated trace (core.tracing, results/logs/trace.csv) ──────────────
#
# TRACE_FRAME_COLUMNS is core.tracing.TRACE_COLUMNS, IMPORTED (see the module
# docstring): the row contract is declared exactly once, in the module that
# writes it. A copy here would be the "second declaration" this tree removed.

TRACE_SCOPES: tuple[str, ...] = ("run", "entity", "group")
TraceScope = Literal["run", "entity", "group"]


class TraceRow(BaseModel):
    """One row of the consolidated pipeline trace (core.tracing.TRACE_COLUMNS).

    The trace is the tree's narrative artifact — read top to bottom it IS the
    data flow — so its row contract carries the same teeth as every other
    boundary contract:

      scope         run | entity | group, literally; a typo'd scope would
                    silently create a fourth population nobody counts.
      stage / step  both non-empty (whitespace-only counts as empty: the
                    reader greps these, so a blank one is a lost row).
      at            a real timezone-aware ISO-8601 instant, not free text.
      detail        JSON-encoded TEXT by contract (core.tracing._detail_text
                    serializes it), so a dict here is a producer bug.
      counts        in/out/dropped non-negative ints or None, with
                    dropped_count DERIVED: the arithmetic (in - out) lives
                    here ONCE and a disagreement is a ValidationError. A row
                    may not state a drop it cannot derive, and may not lose
                    the derivation while stating both ends.

    ``extra="forbid"`` in the same style as the other row specs: an
    undeclared trace column is a producer that believes in a contract this
    module does not have.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    step: str = Field(min_length=1)
    scope: TraceScope
    key: str
    in_count: int | None = Field(ge=0)
    out_count: int | None = Field(ge=0)
    dropped_count: int | None = Field(ge=0)
    reason: str
    detail: str
    source: str
    producer: str
    at: str

    @field_validator("run_id", "stage", "step")
    @classmethod
    def _non_blank_identity(cls, v: str, info) -> str:
        """Whitespace-only is empty. ``run_id`` in particular must be present:
        a row belonging to no run cannot be reasoned about, and the trace's
        whole duplicate-row policy is built on that axis."""
        if not str(v).strip():
            raise ValueError(f"trace {info.field_name} must be non-empty")
        return v

    @field_validator("in_count", "out_count", "dropped_count", mode="before")
    @classmethod
    def _count_cell(cls, v: object) -> object:
        """Absent counts arrive three ways; only those three mean None.

        core.tracing.record emits Python None; the in-memory frame turns it
        into NaN; core.tracing.read_trace reads dtype=str with
        keep_default_na=False, so it comes back as "". Everything else —
        including a boolean, which is an int subclass — is handed to the int
        validator so non-numeric text still FAILS instead of reading as
        "no count".
        """
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"trace count must be an integer, got bool {v!r}")
        if isinstance(v, str):
            text = v.strip()
            return None if not text else text
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    @field_validator("stage", "step")
    @classmethod
    def _not_blank(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"trace {info.field_name} must be non-empty")
        return v

    @field_validator("at")
    @classmethod
    def _iso_instant(cls, v: str) -> str:
        try:
            parsed = datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(
                f"trace at={v!r} is not an ISO-8601 instant "
                f"(core.tracing.record emits datetime.now(timezone.utc).isoformat())"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError(
                f"trace at={v!r} has no UTC offset; the trace is stamped with "
                f"datetime.now(timezone.utc).isoformat()"
            )
        return v

    @model_validator(mode="after")
    def _dropped_is_derived(self) -> TraceRow:
        """dropped_count == in_count - out_count, or the row is a lie.

        The arithmetic is written ONCE, here, and never re-implemented in a
        producer: core.tracing.record derives it, this validator proves it.
        """
        if self.in_count is None or self.out_count is None:
            if self.dropped_count is not None:
                raise ValueError(
                    "trace dropped_count is derived from in_count/out_count, "
                    f"so a row cannot state dropped_count={self.dropped_count} "
                    "without both ends"
                )
            return self
        derived = self.in_count - self.out_count
        if derived < 0:
            raise ValueError(
                f"trace out_count {self.out_count} exceeds in_count "
                f"{self.in_count} — a step cannot emit more than it received"
            )
        if self.dropped_count is None:
            raise ValueError(
                f"trace row carries in_count={self.in_count} and "
                f"out_count={self.out_count} but no dropped_count "
                f"(the derived value is {derived})"
            )
        if self.dropped_count != derived:
            raise ValueError(
                f"trace dropped_count {self.dropped_count} contradicts "
                f"in_count - out_count ({self.in_count} - {self.out_count} = "
                f"{derived})"
            )
        return self


def check_trace_frame(df: pd.DataFrame) -> pd.DataFrame:
    """trace.csv contract: exact columns (order included), every row a valid
    TraceRow. Returns df unchanged (assert-only boundary, like the others).

    Row errors are re-raised as ValueError carrying the row index — a
    5000-row trace otherwise reports a violation with no way to find it.
    """
    cols = tuple(df.columns)
    if cols != TRACE_FRAME_COLUMNS:
        raise ValueError(
            f"trace frame columns {cols} != contract {TRACE_FRAME_COLUMNS}"
        )
    for index, row in enumerate(df.to_dict("records")):
        try:
            TraceRow.model_validate(row)
        except ValidationError as exc:
            raise ValueError(
                f"trace frame row {index} violates TraceRow: {exc}"
            ) from exc
    return df


# ── zero-shot evaluation summary (model_evaluation_summary.csv) ────────────

# cosine similarity lives in [-1, 1] but float32 dot products overshoot
# by rounding noise (measured max 1.0000004 on the real sweep CSV); the
# Youden thresholds are picked FROM those scores, so their bounds admit
# exactly this epsilon and nothing more.
_COSINE_EPS = 1e-6

EVAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "eval_half",
    "threshold_source",
    "youden_thr_dev",
    "pr_auc",
    "precision_at_1",
    "recall_at_1",
    "precision_at_5",
    "recall_at_5",
    "precision_at_10",
    "recall_at_10",
    "hits_at_1",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "n_dev",
    "n_test",
    "youden_thr_test_descriptive",
)


class EvalSummaryRow(BaseModel):
    """One model_evaluation_summary.csv row — written by
    src/training/evaluate_models.py, validated BEFORE the CSV write.

    HOLDOUT DISCIPLINE: youden_thr_dev is fit on the DEV component fold
    and applied verbatim to TEST; every metric is computed on TEST only.
    eval_half pins the scored half; threshold_source records where the
    threshold came from, so a row can not silently claim a different
    provenance. youden_thr_test_descriptive is the leak diagnostic
    (threshold argmax ON test — report-only, never applied)."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    eval_half: Literal["test"] = Field()
    threshold_source: Literal["dev_youden"]
    youden_thr_dev: float = Field(ge=0.0, le=1.0 + _COSINE_EPS)
    pr_auc: float = Field(ge=0.0, le=1.0)
    precision_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_1: float = Field(ge=0.0, le=1.0)
    precision_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    precision_at_10: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    hits_at_1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    tp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    n_dev: int = Field(ge=1)
    n_test: int = Field(ge=1)
    youden_thr_test_descriptive: float = Field(ge=0.0, le=1.0 + _COSINE_EPS)

    @model_validator(mode="after")
    def _counts_match_totals(self) -> EvalSummaryRow:
        total = self.tp + self.tn + self.fp + self.fn
        if total != self.n_test:
            raise ValueError(
                f"confusion counts (tp+tn+fp+fn={total}) != n_test "
                f"({self.n_test}) — the row mixes halves or halves disagree"
            )
        return self


def check_eval_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """model_evaluation_summary.csv contract: exact columns (order
    included — the CSV is a documented report artifact), every row a
    valid EvalSummaryRow, no NaN/inf anywhere in the frame."""
    cols = tuple(df.columns)
    if cols != EVAL_SUMMARY_COLUMNS:
        raise ValueError(
            f"eval summary frame columns {cols} != contract {EVAL_SUMMARY_COLUMNS}"
        )
    rows = [EvalSummaryRow.model_validate(r) for r in df.to_dict("records")]
    if len(rows) != len(df):
        raise ValueError(
            f"validated {len(rows)} rows but frame has {len(df)} — "
            f"rows were dropped or duplicated in validation"
        )
    num = df.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        bad = ~np.isfinite(num.to_numpy(dtype=float))
        raise ValueError(
            f"eval summary has non-finite values: {int(bad.sum())} cells "
            f"(columns {list(num.columns)})"
        )
    return df


# ── frame-checker registry (the discovery surface) ─────────────────────────
#
# Named lookup for every frame contract in this module, so a consumer that
# only knows the ARTIFACT (or the tracing owner wiring assert_trace_frame)
# can reach the checker without importing each name by hand:
#
#     from core.schemas import FRAME_CHECKERS
#     FRAME_CHECKERS["trace"](frame)
#
# The values are the checker functions themselves — this is a directory, not
# a second declaration: it defines no columns and no domains.
FRAME_CHECKERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "canonical_records": check_canonical_records_frame,
    "gate_results": check_gate_results_frame,
    "labeled_pairs": check_labeled_pairs_frame,
    "cross_country_pairs": check_cross_country_pair_frame,
    "zero_shot_similarity": check_zero_shot_similarity_frame,
    "eval_summary": check_eval_summary_frame,
    "trace": check_trace_frame,
}


# ── verdict-map adapter (the number-token reference CSV) ──────────────────

_VerdictMap = TypeAdapter(dict[str, TokenVerdict])


def check_verdict_map(m: dict[str, str]) -> dict[str, str]:
    """token -> verdict map from the reference CSV: every verdict in the
    strip/keep_* vocabulary (a typo'd verdict would silently never match
    the startswith('keep') branch in strip_number_tokens)."""
    return _VerdictMap.validate_python(m)


# ── per-stage manifest (SILENT_DROPS task 3 — the runtime artifact
#    contract written by src/core/manifest.finish_manifest) ──────────────────


class ManifestFile(BaseModel):
    """One file row inside a StageManifest — inputs and outputs share the
    shape; `expected` is outputs-only (True = listed in expected_outputs,
    False = unexpected extra; None on inputs)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)  # repo-root-relative POSIX path
    sha256: str = Field(min_length=64, max_length=64)  # lowercase hex
    rows: int | None = None  # CSV row count when known
    cols: int | None = None
    expected: bool | None = None


class StageManifest(BaseModel):
    """results/manifests/<stage>.json — the silent-drop guardrail's per-
    stage record (SILENT_DROPS design sketch). Written LAST via atomic
    rename (src/core/manifest.finish_manifest), so its presence with
    status "complete" IS the stage's completion marker; a crashed stage
    leaves at most the previous run's manifest plus .tmp-* residue.

    Closure invariant (enforced by finish_manifest and verify_manifest):
    row_accounting.input_rows == output_rows + sum(dropped.values()).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    stage: str = Field(min_length=1)
    started: str  # ISO-8601 UTC
    finished: str | None = None
    status: Literal["running", "complete", "failed"]
    inputs: list[ManifestFile]
    outputs: list[ManifestFile]
    # {input_rows, output_rows, dropped: {reason: count}} — kept a flat
    # dict (not a nested model) so stages can add reasons without a
    # schema bump; the closure invariant above is what's contractual.
    row_accounting: dict[str, Any]
    environment: dict[str, str]  # git_sha, config_sha256, seed, host
    expected_outputs: list[str]
