"""lib/schemas.py — pydantic contracts + shape assertions for the training
transformation pipeline (owner directive: pydantic + shape assertions for
better control).

TWO jobs:

  1. CONFIG CONTRACTS — one pydantic model per split config file, validated
     at load by lib.common (fail-loudly at import, never mid-run):
       DataConfig       00_config.yaml     — paths/files/column_mapping/seed/models
       TrainingConfig   TRAIN/training.yaml — the training lane's knobs
     (EDA/eda.yaml was deleted with the EDA dir 2026-09-10; its five
     TRAIN-consumed keys — plots.dpi, pairs.max_pos_per_group/n_neg/
     neg_oversample, strip_audit_sample — migrated into TRAIN/training.yaml.)

  2. BOUNDARY CONTRACTS — validated containers at every transform boundary
     in the pipeline (small objects, never per-row hot loops):
       ExtractedAttributes   data_pipe.extract_all
       GateResult            data_pipe.three_way_gate
       CanonicalRecord       data_pipe.generate_canonical
       PairArrays            index-pair containers (pos/neg/hp)
       TrainingData          data_pipe.build_training_data (the bundle)
       MaskingResult         TRAIN.masking.augment_positives
       DataTuple             the (df, payload, row_bc, country, pos,
                             hp_pairs, emb0) 7-tuple crossing into
                             train_one_config
       TrainConfig           the per-config dict train_one_config receives
       FoldSets              TRAIN.folds.component_folds output

     Frame checks (DataFrame column/domain contracts) live in
     check_canonical_records_frame / check_gate_results_frame /
     check_labeled_pairs_frame — used at CSV write/read boundaries.

Doctrine (owner Q27): NO FALLBACKS. Optional-with-default means
"config may omit it" ONLY where the model declares a default and the
accessor raises when missing from the merged view. Nothing silently
defaults to an inline literal at the call site.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════


class DataFilesSpec(BaseModel):
    """The file-name contract — every CSV/JSON the pipeline touches.

    Keys are load-bearing: lib.common.F and every consumer index them.
    `stopwords` points at lib/pipe_stopwords.json (the data_pipe word-list
    SSOT — moved there 2026-09-08; matching.py's sklearn set is a separate
    file `sklearn_stopwords` in lib/).
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    dataset_deduped: str
    sku_to_rep: str
    canonical_records: str
    gate_results: str
    labeled_pairs: str
    embedding_similarities: str
    model_evaluation_summary: str
    fold_metrics: str
    hpo_grid_csv: str
    # hpo_tpe_best REMOVED (audit round 2 F02, finished round 3): dead
    # field — F["hpo_tpe_best"] had zero readers anywhere.
    dedupe_summary: str
    ambiguous_offer_groups: str
    four_pop_scores: str
    field_ablation: str
    data_scaling: str
    # cache_dir REMOVED (audit 2026-09-09): dead knob, zero readers
    stopwords: str
    sklearn_stopwords: str
    number_reference: str
    # second04_pairs_positive (audit round 2 F13): repo-side cross-country
    # gold-pair manifest (lib/volume_verified); declared SSOT-side.
    second04_pairs_positive: str
    results_pointer: str


class DataPathsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: str
    data_dir: str
    results_dir: str
    models_dir: str
    models_dir_sibling: str
    embeddings_dir: str
    mlruns_dir: str
    logs_dir: str


class DataConfig(BaseModel):
    """00_config.yaml — the SHARED data contract (paths, file names, column
    mapping, seed, model registry). Domain knobs live in their own dir:
    TRAIN/training.yaml."""

    model_config = ConfigDict(extra="forbid")

    paths: DataPathsSpec
    files: DataFilesSpec
    column_mapping: dict[str, str] = Field(min_length=1)
    seed: int
    models: dict[str, str] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _registry_has_trainer_base(cls, v: dict[str, str]) -> dict[str, str]:
        if "multilingual_l12" not in v or not v["multilingual_l12"]:
            raise ValueError(
                "models.multilingual_l12 missing — the trainer base resolves "
                "from it (train.py --model default; no-fallback doctrine)"
            )
        return v


class SplitSpec(BaseModel):
    """Component-aware split contract (TRAIN/training.yaml split:).

    train/dev/test are COMPONENT shares over the positive-pair graph —
    asserted to sum to 1.0 (no silent re-normalization: a mis-configured
    split must CRASH, not quietly produce 60/20/20).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["holdout", "cv"]
    train_fraction: float = Field(ge=0.0, le=1.0)
    dev_fraction: float = Field(ge=0.0, le=1.0)
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


class MaskingSpec(BaseModel):
    """Masking augmentation contract (TRAIN/training.yaml masking:)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    frac: float = Field(ge=0.0, le=1.0)
    mask_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_lo: float = Field(ge=0.0, lt=1.0)
    mask_hi: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _band_ordered(self) -> MaskingSpec:
        if self.mask_lo >= self.mask_hi:
            raise ValueError(
                f"masking.mask_lo must be < mask_hi, got "
                f"{self.mask_lo} >= {self.mask_hi}"
            )
        return self


class TrainingSpec(BaseModel):
    """Training runtime knobs (TRAIN/training.yaml training:).

    All keys are REQUIRED (no-fallback doctrine): a missing knob crashes at
    config load, not at the call site. The training lane reads them
    exclusively through lib.common.runtime().
    """

    model_config = ConfigDict(extra="forbid")

    loss: Literal["contrastive", "mnrl", "triplet"]
    contrastive_margin: float = Field(gt=0.0, le=2.0)
    hard_positives: bool
    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)
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
    n_target_mining: int = Field(ge=1)
    hardneg_k: int = Field(ge=1)


class PairsSpec(BaseModel):
    """Pair-label thresholds + eval-pair caps (TRAIN/training.yaml pairs:) —
    cosine gates over gate_results similarity plus the build_pairs caps
    (migrated from EDA/eda.yaml when the EDA dir was deleted, 2026-09-10)."""

    model_config = ConfigDict(extra="forbid")

    proceed_sim_threshold: float = Field(ge=0.0, le=1.0)
    hardneg_sim_threshold: float = Field(ge=0.0, le=1.0)
    max_pos_per_group: int = Field(ge=1)
    n_neg: int = Field(ge=0)
    neg_oversample: int = Field(ge=1)


class TrainingPlotsSpec(BaseModel):
    """Plot rendering (TRAIN/training.yaml plots:) — the one DPI every
    fig.savefig in the tree renders at (was EDA/eda.yaml plots.dpi)."""

    model_config = ConfigDict(extra="forbid")

    dpi: int = Field(ge=50, le=600)


class AuditSpec(BaseModel):
    """Audit-lane knobs (TRAIN/training.yaml audit:) — strip-audit sample
    size (was EDA/eda.yaml strip_audit_sample) + the blocking-feature
    audit's policy knobs (were inline BUDGET/MIN_RECALL literals in
    TRAIN/blocking_audit.py; audit round 2 F18, moved round 3)."""

    model_config = ConfigDict(extra="forbid")

    strip_audit_sample: int = Field(ge=1)
    blocking_budget: int = Field(ge=1)
    blocking_min_recall: float = Field(gt=0.0, le=1.0)


class GateSpec(BaseModel):
    """Three-way-gate decision thresholds (TRAIN/training.yaml gate:) — the
    gate's decision table. Formerly signature defaults in
    data_pipe.three_way_gate (audit round 2, F01): vol_tolerance (relative
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
    """Cosine bands (TRAIN/training.yaml bands:) — eval/rerank.

    mining_band was REMOVED (audit round 2, F21): band("mining_band") had
    zero callers — the live mining band is mining.band "0.45-0.80"."""

    model_config = ConfigDict(extra="forbid")

    eval_mining: BandSpec
    rerank_band: BandSpec


class MiningSpec(BaseModel):
    """Hard-negative mining knobs (TRAIN/training.yaml mining:) — `band` is
    the in-batch mining band string "lo-hi"; `k` is the ANN block size."""

    model_config = ConfigDict(extra="forbid")

    band: str
    k: int = Field(ge=1)

    @field_validator("band")
    @classmethod
    def _band_parses(cls, v: str) -> str:
        try:
            lo, hi = (float(x) for x in v.split("-"))
        except ValueError as e:
            raise ValueError(
                f'mining.band must be "lo-hi" floats, got {v!r}'
            ) from e
        if not lo < hi:
            raise ValueError(f"mining.band must satisfy lo < hi, got {v!r}")
        return v


class HpoGridRowSpec(BaseModel):
    """One fixed-grid config (TRAIN/training.yaml hpo.grid[] / hpo.quick[]):
    epochs x lr x warmup-percent — second07's axes, verbatim."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup: int = Field(ge=0, le=100)


class HpoSpaceSpec(BaseModel):
    """TPE search ranges (TRAIN/training.yaml hpo.tpe_space:) — each value is
    the [lo, hi] suggest range for its knob. lr is sampled log-uniform."""

    model_config = ConfigDict(extra="forbid")

    epochs: tuple[int, int]
    lr: tuple[float, float]
    warmup_ratio: tuple[float, float]
    weight_decay: tuple[float, float]

    @model_validator(mode="after")
    def _ranges_ordered(self) -> HpoSpaceSpec:
        for name in ("epochs", "lr", "warmup_ratio", "weight_decay"):
            lo, hi = getattr(self, name)
            if not lo < hi:
                raise ValueError(
                    f"hpo.tpe_space.{name} must satisfy lo < hi, got [{lo}, {hi}]"
                )
        return self


class HpoSpec(BaseModel):
    """HPO lane knobs (TRAIN/training.yaml hpo:) — the fixed grid, the
    --quick smoke subset, the TPE space, and trial budgets. Formerly
    module-level literals in TRAIN/hpo.py / TRAIN/training.py."""

    model_config = ConfigDict(extra="forbid")

    grid: list[HpoGridRowSpec] = Field(min_length=1)
    quick: list[HpoGridRowSpec] = Field(min_length=1)
    tpe_space: HpoSpaceSpec
    n_trials: int = Field(ge=1)
    n_jobs: int = Field(ge=1)


class RerankSpec(BaseModel):
    """07e A/B decision rule (TRAIN/training.yaml rerank:) — the quantitative
    "must clearly improve": the hybrid ships only when ΔPR-AUC or ΔF1 beats
    the bi-encoder by at least these margins. Formerly inline 0.005s in
    TRAIN/rerank.py."""

    model_config = ConfigDict(extra="forbid")

    min_delta_pr_auc: float = Field(ge=0.0, le=1.0)
    min_delta_f1: float = Field(ge=0.0, le=1.0)


class SweepSpec(BaseModel):
    """Ablation-sweep axes (TRAIN/training.yaml sweep:) — run_all step 4's
    payload variants, train-frac curve, smoke/sweep sample sizes, and the
    07e rerank cross-encoder id. Formerly inline literals in run_all.py /
    colab_backend.py."""

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


class TrainingConfig(BaseModel):
    """TRAIN/training.yaml — the training lane's OWN config (in its dir).

    Validated as a whole at load; lib.common exposes it through
    training_cfg()/runtime()/band()/SSOT_LOSS/SSOT_CONTRASTIVE_MARGIN.
    Cross-config checks (sim_columns vs the root models registry) happen in
    lib.common at merge time — this model can't see the root config by
    design (one file, one contract).
    """

    model_config = ConfigDict(extra="forbid")

    sim_columns: dict[str, str] = Field(min_length=1)
    masking: MaskingSpec
    split: SplitSpec
    gate: GateSpec
    training: TrainingSpec
    pairs: PairsSpec
    plots: TrainingPlotsSpec
    audit: AuditSpec
    bands: BandsSpec
    mining: MiningSpec
    hpo: HpoSpec
    rerank: RerankSpec
    sweep: SweepSpec



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
    volume_confidence: float = Field(ge=0.0, le=1.0)
    pack_confidence: float = Field(ge=0.0, le=1.0)
    volume_consistency: float = Field(ge=0.0, le=1.0)
    pack_consistency: float = Field(ge=0.0, le=1.0)
    n_titles: int = Field(ge=1)

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
                raise ValueError(f"empty pair array must be shape (0,2), got {arr.shape}")
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
    n_neg_gate_rows: int = Field(ge=0)
    n_neg_resolved: int = Field(ge=0)
    n_neg_dropped: int = Field(ge=0)


class TrainingData(BaseModel):
    """build_training_data output — the OFFICIAL pair bundle. Cross-shape
    assertions: len(row_bc) == len(payload); every pos/neg index in range;
    gtin_to_row targets < len(payload)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: list[str]
    row_bc: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    gtin_to_row: dict[str, int]
    stats: TrainingStats

    @field_validator("row_bc")
    @classmethod
    def _row_bc_vector(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError(f"row_bc must be 1-D, got shape {arr.shape}")
        return arr

    @field_validator("pos", "neg")
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
            raise ValueError(
                f"row_bc length {len(self.row_bc)} != payload length {n}"
            )
        for name in ("pos", "neg"):
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
    anchor_text: str
    masked_text: str


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
            raise ValueError(
                f"row_bc length {len(self.row_bc)} != payload length {n}"
            )
        if self.pos.size and int(self.pos.max()) >= n:
            raise ValueError(
                f"pos index {int(self.pos.max())} out of range (payload={n})"
            )
        if self.n_added != len(self.audit):
            raise ValueError(
                f"n_added {self.n_added} != audit rows {len(self.audit)}"
            )
        return self

    def audit_dicts(self) -> list[dict]:
        """The audit rows as plain dicts (the historical return surface —
        train.py dumps them straight into a DataFrame)."""
        return [a.model_dump() for a in self.audit]


class DataTuple(BaseModel):
    """The 7-tuple crossing into train_one_config:
    (df, payload, row_bc, country, pos, hp_pairs, emb0).

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
    row_bc: np.ndarray
    country: np.ndarray
    pos: np.ndarray
    hp_pairs: np.ndarray
    emb0: np.ndarray

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
        if len(self.row_bc) != n:
            raise ValueError(
                f"row_bc length {len(self.row_bc)} != payload length {n}"
            )
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
            raise ValueError(
                f"emb0 rows {self.emb0.shape[0]} != payload length {n}"
            )
        return self


class TrainConfig(BaseModel):
    """The per-config dict train_one_config receives (DEFAULT_CFG/HPO/grid
    rows). Constraints mirror the HF Trainer's own validations — but fire
    at the boundary with a named field instead of an anonymous traceback."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0)
    lr_scheduler: str
    max_grad_norm: float = Field(gt=0.0)
    patience: int = Field(ge=1)
    es_threshold: float = Field(ge=0.0)


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
    "volume_confidence",
    "pack_confidence",
    "volume_consistency",
    "pack_consistency",
    "n_titles",
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
            f"gate_results frame columns {cols} != contract "
            f"{GATE_RESULTS_COLUMNS}"
        )
    bad = set(df["gate_decision"].unique()) - set(GATE_DECISIONS)
    if bad:
        raise ValueError(f"gate_decision outside domain: {sorted(bad)}")
    s = pd.to_numeric(df["similarity"], errors="coerce")
    if s.isna().any() or ((s < 0) | (s > 1)).any():
        raise ValueError(
            f"gate_results.similarity outside [0,1]: "
            f"min {s.min()}, max {s.max()}"
        )
    gtin_empty = (
        df["gtin1"].fillna("").astype(str).str.strip().eq("")
        | df["gtin2"].fillna("").astype(str).str.strip().eq("")
    )
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
            f"labeled_pairs frame columns {cols} != contract "
            f"{LABELED_PAIRS_COLUMNS}"
        )
    bad = set(pd.to_numeric(df["true_label"], errors="coerce").dropna().unique()) - {
        0,
        1,
    }
    if bad:
        raise ValueError(f"true_label outside {{0,1}}: {sorted(bad)}")
    if pd.to_numeric(df["true_label"], errors="coerce").isna().any():
        raise ValueError("true_label has non-numeric rows")
    gtin_empty = (
        df["gtin1"].fillna("").astype(str).str.strip().eq("")
        | df["gtin2"].fillna("").astype(str).str.strip().eq("")
    )
    if gtin_empty.any():
        raise ValueError(f"{int(gtin_empty.sum())} rows with empty gtin endpoint")
    dups = df.duplicated(subset=["gtin1", "gtin2"]).sum()
    if dups:
        raise ValueError(f"{int(dups)} duplicate (gtin1, gtin2) rows")
    return df


# ── verdict-map adapter (the number-token reference CSV) ──────────────────

_VerdictMap = TypeAdapter(dict[str, TokenVerdict])


def check_verdict_map(m: dict[str, str]) -> dict[str, str]:
    """token -> verdict map from the reference CSV: every verdict in the
    strip/keep_* vocabulary (a typo'd verdict would silently never match
    the startswith('keep') branch in strip_number_tokens)."""
    return _VerdictMap.validate_python(m)
