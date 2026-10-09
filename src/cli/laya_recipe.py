"""Laya lane contract registry: decision bindings, trainer fields, literals.

SSOT for the laya lane's staged payload contract: the per-kind decision
bindings, the ``TrainConfig`` / training-control / eval-calibration field
tuples, and the file/slug literals every staged surface names. The
``LayaRecipeFactory`` reads the recipe surface off ONE ``LayaSpec`` so no
caller re-derives a default; the laya lane entry point injects the spec.
"""
from __future__ import annotations

from typing import Any

from core.laya_config import LayaSpec

KINDS = ("kaggle", "colab")
GPU_KINDS = ("attribute", "identity", "laya-cli-eval", "finetune",
             "finetune-smoke", "finetune-eval", "holdout-eval")

DECISION_KERNEL_CODE_FILE = "laya_decision.py"
EVAL_KERNEL_CODE_FILE = "laya_evals.py"
COLAB_NOTEBOOK_NAME = "laya_decision_colab.py"
QUESTION_SCHEMA_FILE = "laya.question.json"
# The kernel's inputs do NOT travel with `kaggle kernels push`: they are
# the er-laya-requests dataset (spec.dataset_slug), staged as its own
# payload dir (moved on --execute, mirroring the kernels-push path).
DATASET_PAYLOAD_DIR = "dataset_payload"
DATASET_METADATA_FILE = "dataset-metadata.json"
DATASET_CSV_NAME = "dataset.csv"  # DECISION_CSV resolves THIS name
# The base-model archive travels as a sealed `inputs` Bundle: this is the
# surface-owned role manifest name (the graph bundlers own theirs the same way
# via graph_tracks.artifacts.name); `Bundle.seal_archive` writes it as the one
# container manifest and verifies the archive as it seals.
BASE_MODEL_MANIFEST_FILE = "base_model_manifest.json"

# ── fine-tune kind (owner: "lets run laya") ────────────────────────────────
# The JSONL corpus built by scripts/laya_build_dataset.py (data/laya/
# {train,dev,test}.jsonl + receipt.json) travels as its OWN kaggle dataset
# (spec.finetune_dataset_slug), distinct from the decision payloads'
# er-laya-requests dataset: a corpus version must never drop the decision
# inputs (and vice versa). The kernel wraps the REAL `laya-train` CLI on a
# single T4 and tars the checkpoint back.
FINETUNE_DECISION = "finetune"
# The CPU end-to-end smoke kind: the SAME kernel template + staging +
# push surface, but pinned to CPU, a tiny subset corpus and dedicated
# slugs (laya.finetune_smoke), so a smoke never trains on the production
# corpus nor overwrites the production kernel.
FINETUNE_SMOKE_DECISION = "finetune-smoke"
FINETUNE_CODE_FILE = "laya_finetune.py"
FINETUNE_CORPUS_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")
FINETUNE_CORPUS_RECEIPT = "receipt.json"

# ── fine-tune EVAL-only kind (held-out score, no retrain) ──────────────────
# A dedicated `--decision finetune-eval` kernel loads a fine-tuned checkpoint
# and scores the corpus HELD-OUT split (default test.jsonl) with
# `laya.train.load_checkpoint` -> `calibration_records` -> `evaluate_records`.
# It attaches the SAME corpus dataset (er-laya-train) + the fine-tuned
# checkpoint dataset, installs laya, and writes eval_report.json + a receipt
# into /kaggle/working for fetch-back. NO training, NO Hub.
FINETUNE_EVAL_DECISION = "finetune-eval"
FINETUNE_EVAL_CODE_FILE = "laya_finetune_eval.py"
FINETUNE_EVAL_REPORT_FILE = "eval_report.json"
FINETUNE_EVAL_RECEIPT_FILE = "laya_finetune-eval.receipt.json"

# ── holdout-eval kind: component-disjoint verification, run on Kaggle ───────
# Scores a fine-tuned checkpoint on the staged holdout (real pairs + P0 + gate
# strata) in-session and writes the clustered, gate-stratified report, so the
# honest verification never runs on the operator box. The holdout travels as a
# staged JSONL dataset (one composed identity state + label + stratum per row);
# the checkpoint rides the finetune_ckpt_dataset.
HOLDOUT_EVAL_DECISION = "holdout-eval"
HOLDOUT_EVAL_CODE_FILE = "laya_holdout_eval.py"
HOLDOUT_EVAL_REPORT_FILE = "holdout_report.json"
HOLDOUT_EVAL_RECEIPT_FILE = "laya_holdout-eval.receipt.json"
HOLDOUT_JSONL = "holdout.jsonl"
HOLDOUT_CATALOG_FILE = "holdout_catalog.csv"
# The corpus split name is a config literal; map it to the JSONL file the
# corpus dataset carries. Never duplicated: the tuple above is the SSOT.
FINETUNE_EVAL_SPLIT_FILES = {
    "train": FINETUNE_CORPUS_FILES[0],
    "dev": FINETUNE_CORPUS_FILES[1],
    "test": FINETUNE_CORPUS_FILES[2],
}
# `pip install laya`; pin laya>=0.3.29 (the version the flags were verified
# against: /tmp/opc/laya_pkg329/bin/laya-train --help).
# Deprecated module alias (the ``laya_package`` template value the legacy
# tests/test_lane_fixes.py renderer passes). The SSOT is
# ``laya.finetune_package``; every staged payload resolves it from the spec.
FINETUNE_LAYA_PACKAGE = LayaSpec().finetune_package
# The FULL `laya.train.TrainConfig` field surface the finetune kernel
# builds from `laya.finetune` (SSOT): every trainer knob is YAML-driven,
# never a code literal. `eval_data` is supplied by the kernel (the attached
# dev split); `device` is a runtime resolver input, not a TrainConfig field.
FINETUNE_CONFIG_FIELDS = (
    "epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr",
    "min_lr", "weight_decay", "grad_clip",
    "loss", "label_smoothing", "rl_samples", "sigma_start", "sigma_end",
    "w_sph", "w_rps",
    "shuffle_options", "option_layout", "max_len", "head_max_len",
    "text_column", "label_column", "question_id", "instructions",
    "freeze_encoder",
    "calib_max", "calib_frac", "calib_seed", "target_error", "min_abstain_n",
    "seed", "amp", "gradient_checkpointing", "log_every",
)

# The training-CONTROL surface the staged perf patch reads (SSOT). These knobs
# are deliberately NOT in FINETUNE_CONFIG_FIELDS: laya's `TrainConfig` rejects
# unknown kwargs, so they travel as a SEPARATE `FINETUNE_CONTROL` JSON/repr
# block the perf patch reads from a module global. One tuple, so the shadowing
# lint proves every declared knob reaches the baked literal.
FINETUNE_CONTROL_FIELDS = (
    # Phase 1 (default-ON)
    "eval_dev", "early_stop", "early_stop_patience", "early_stop_min_delta",
    "early_stop_metric", "keep_best", "save_each_epoch", "resume",
    "lr_scheduler", "warmup_frac", "warmup_steps", "plateau_patience",
    "plateau_factor", "onecycle_pct_start", "abstain_confidence",
    # Phase 2 (default-OFF)
    "unfreeze_after_epoch", "layer_decay", "ema", "ema_decay", "swa", "swa_lr",
    "swa_start_frac", "optimizer",
    # Phase 3 (default-OFF)
    "class_weight", "balanced_sample", "hard_example_frac", "curriculum",
    "adv_eps", "adv_kind", "temperature_scale",
    # Phase 4 (default-OFF except log_grad_norm)
    "compile_model", "amp_dtype", "tf32", "log_grad_norm",
    "write_error_artifacts", "deterministic",
    # torch.profiler (default-OFF; opt in per run)
    "profile", "profile_dir", "profile_schedule",
    # Extended knobs (all default-OFF; HPO-searchable)
    "no_decay_bias_norm", "optim_state_dtype", "lr_scaling", "base_batch",
    "r_drop", "r_drop_alpha", "drop_path", "drop_path_rate",
    "drop_path_schedule", "dynamic_padding", "pad_to_multiple",
    "batch_size_ramp", "batch_ramp_start_frac", "batch_ramp_epochs",
    "loss_schedule", "w_sph_end", "w_rps_end", "contrastive_margin",
    "contrastive_margin_end",
)

# The `EvalCalibrationSpec` surface the held-out eval kernels consume: the
# per-type temperature fit + the optional abstention (`min_confidence`) fit.
# One tuple, so the shadowing lint (`finetune_config` precedent) proves every
# declared knob reaches the baked literal.
EVAL_CALIBRATION_FIELDS = (
    "temperature", "abstention", "target_error", "min_abstain_n",
    "min_confidence",
)

# Per decision kind: the required header columns, the state column the
# decision state is built from, and what the run decides. THE CSV BINDING
# ITSELF lives in the SSOT (LayaSpec.decision_csv_bindings, resolved
# through core.common.F here) — never duplicated in a second registry.
DECISION_BINDINGS: dict[str, dict[str, Any]] = {
    "attribute": {
        "wanted_columns": ("sku_id", "sku_name_eng", "attribute"),
        "state_column": "attribute",
        # The columns that address one row (the kernel's ``_row`` tags).
        "record_columns": ("sku_id", "sku_name_eng"),
        "description": ("attribute-channel typed decision over the export's "
                        "attribute text. NOT a replacement for the frozen "
                        "SKU_ITEM/GTIN attribution path — it asks whether "
                        "laya's calibrated route agrees on the same "
                        "attributes the task layout already fixed"),
    },
    "identity": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "record_columns": ("gtin1", "gtin2"),
        "description": ("identity typed decision over the frozen P0 "
                        "validation population. NOT a replacement for the "
                        "candidate-generation + gate + ann/rerank path — "
                        "its questions ask whether the candidates available "
                        "from the routes agree with the same identity "
                        "evidence the task layout already fixed"),
    },
    "laya-cli-eval": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
        "record_columns": ("gtin1", "gtin2"),
        "description": ("the laya-evals harness score over the same "
                        "identity decision samples, verifying the shared "
                        "transport + recall identity"),
    },
    "finetune": {
        # Not a per-row decision CSV: the corpus row keys are the contract
        # (the JSONL the laya trainer consumes). Kept in the same registry so
        # `--decision finetune` rides the lane's existing binding surface.
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        # Not a per-row decision CSV: the corpus row is the unit, so no
        # decision record columns exist (the corpus adapter owns this grain).
        "record_columns": None,
        "description": ("fine-tune the convaiinnovations/laya checkpoint on "
                        "the verified-label JSONL corpus (state + identity "
                        "cases) via the real laya-train CLI on a single T4"),
    },
    "finetune-smoke": {
        # The tiny CPU end-to-end validation of the SAME finetune kernel:
        # corpus grain like `finetune`, but staged with the laya.finetune_smoke
        # dials/slugs and enable_gpu=False.
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        "record_columns": None,
        "description": ("CPU end-to-end smoke of the finetune kernel: 1 epoch, "
                        "micro-batch 1 on a tiny subset corpus, dedicated "
                        "dataset/kernel slugs"),
    },
    "finetune-eval": {
        # Not a per-row decision CSV either: the eval-only kernel scores an
        # attached fine-tuned checkpoint against the corpus split, so the
        # corpus row keys are the contract (mirrors the finetune entry).
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        "record_columns": None,
        "description": ("HELD-OUT eval-only score of an attached fine-tuned "
                        "laya checkpoint on the corpus test split: loads the "
                        "checkpoint, runs calibration_records + "
                        "evaluate_records, writes eval_report.json. No "
                        "training, no Hub fetch."),
    },
    "holdout-eval": {
        # Corpus-grain like the two finetune kinds: the component-disjoint
        # holdout rows (state + expected + stratum) are the contract, built by
        # stage_holdout_dataset_payload; `LayaLane.stage()` routes the kind to
        # stage_holdout_eval_kernel.
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        "record_columns": None,
        "description": ("component-disjoint holdout verification of an "
                        "attached fine-tuned laya checkpoint: writes the "
                        "clustered, gate-stratified holdout report. No "
                        "training, no Hub fetch."),
    },
}


class LayaRecipeFactory:
    """The trainer-recipe surface for ONE resolved ``LayaSpec`` (SSOT reader).

    Every method reads a declared field tuple off the injected spec, so the
    lane never re-spells a default and a caller cannot supply a second recipe.
    """

    def __init__(self, spec: LayaSpec):
        self._spec = spec

    def finetune_config(self) -> dict[str, Any]:
        """The full `laya.train.TrainConfig` kwargs from `laya.finetune` (SSOT).

        `shuffle_options` is normalised to a tuple (the TrainConfig annotation)
        while staying JSON/repr-bakeable. Never a second recipe registry: the
        field list above names exactly the `FinetuneSpec` surface.
        """
        ft = self._spec.finetune
        config = {name: getattr(ft, name) for name in FINETUNE_CONFIG_FIELDS}
        config["shuffle_options"] = tuple(config["shuffle_options"])
        return config

    def finetune_control(self) -> dict[str, Any]:
        """The training-control kwargs the perf patch bakes (SSOT).

        Distinct from `finetune_config`: these NEVER reach `TrainConfig`.
        Defaults are the `FinetuneSpec` defaults (Phase 1 on, Phases 2-4 off).
        """
        ft = self._spec.finetune
        return {name: getattr(ft, name) for name in FINETUNE_CONTROL_FIELDS}

    def eval_calibration_config(self) -> dict[str, Any]:
        """The `laya.eval_calibration` selection the eval kernels bake (SSOT).

        Never a second registry: the field list above names exactly the
        `EvalCalibrationSpec` surface. The defaults reproduce the landed eval
        exactly (temperature fit on, abstention fit off, no pinned scalar).
        """
        ec = self._spec.eval_calibration
        return {name: getattr(ec, name) for name in EVAL_CALIBRATION_FIELDS}

    def decision_binding(self, decision_kind: str) -> str:
        """SSOT F-binding for a decision kind (fail-loud on an unknown kind)."""
        if decision_kind not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                             f"expected {list(DECISION_BINDINGS)}")
        bindings = self._spec.decision_csv_bindings
        binding = bindings.get(decision_kind)
        if not binding:
            raise RuntimeError(
                f"config laya.decision_csv_bindings carries no entry for "
                f"{decision_kind!r}; name the config/paths.yaml files: binding "
                "before staging")
        return binding
