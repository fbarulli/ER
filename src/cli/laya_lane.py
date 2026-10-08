"""Laya decision lane (branch laya-lane).

Typed-decision surface for the `laya` package (PyPI `laya`: non-
autoregressive decision engine, Python >= 3.10, torch/transformers wheel
stack): stage the laya.question schema + ER decision inputs on THIS box,
then run the typed decision questions on a REMOTE GPU session:
  * kind="kaggle" — a Kaggle GPU kernel payload (metadata + script +
    receipt under results/laya_lane/kaggle/<decision>/); `--execute`
    drives `kaggle kernels push`, everything else is offline staging.
  * kind="colab"  — a Colab notebook payload + receipt under
    results/laya_lane/colab/<decision>/, delivered in the kaggle-lane
    receipts style. The notebook is the delivery CONTRACT: this lane
    never imports or edits cli.colab / cli.colab_lane and never opens a
    session.

Owner rulings honored (docs/laya-lane.md):
* 2xT4 -> SINGLE T4 per owner ruling: no double accelerator (the session
  pins one CUDA device; the staged payload never requests a second GPU);
* laya installs over pip (`pip install laya`), never vendored;
* exports return via /kaggle/working tar + a hashed receipt.

Contract + evidence: tests/test_laya_lane.py (offline, no network).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.common import TRAIN_ROOT, training_cfg
from core.manifest import atomic_write_json, sha256_file

# One roof (kaggle_lane precedent: TRAIN_ROOT/logs/<lane>/).
KINDS = ("kaggle", "colab")
GPU_KINDS = ("attribute", "identity", "laya-cli-eval", "finetune",
             "finetune-eval")
LANE_LOG_NAME = "lane.log"
# One fresh lane.log per run: first write of this process truncates, later
# writes append (owner order 2026-10-07: overwrite, never append-sprawl).
_LANE_LOG_STARTED = False
DEFAULT_GPU = "T4"  # single T4; NEVER "2xT4" (no double accelerator)
STAGE_ROOT = "results/laya_lane"

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

# ── fine-tune kind (owner: "lets run laya") ────────────────────────────────
# The JSONL corpus built by scripts/laya_build_dataset.py (data/laya/
# {train,dev,test}.jsonl + receipt.json) travels as its OWN kaggle dataset
# (spec.finetune_dataset_slug), distinct from the decision payloads'
# er-laya-requests dataset: a corpus version must never drop the decision
# inputs (and vice versa). The kernel wraps the REAL `laya-train` CLI on a
# single T4 and tars the checkpoint back.
FINETUNE_DECISION = "finetune"
FINETUNE_CODE_FILE = "laya_finetune.py"
FINETUNE_CORPUS_DIR = "data/laya"
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
# The corpus split name is a config literal; map it to the JSONL file the
# corpus dataset carries. Never duplicated: the tuple above is the SSOT.
FINETUNE_EVAL_SPLIT_FILES = {
    "train": FINETUNE_CORPUS_FILES[0],
    "dev": FINETUNE_CORPUS_FILES[1],
    "test": FINETUNE_CORPUS_FILES[2],
}
# `pip install laya`; pin laya>=0.3.29 (the version the flags were verified
# against: /tmp/opc/laya_pkg329/bin/laya-train --help).
FINETUNE_LAYA_PACKAGE = "laya>=0.3.29"
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


def finetune_config(spec: Any | None = None) -> dict[str, Any]:
    """The full `laya.train.TrainConfig` kwargs from `laya.finetune` (SSOT).

    `shuffle_options` is normalised to a tuple (the TrainConfig annotation)
    while staying JSON/repr-bakeable. Never a second recipe registry: the
    field list above names exactly the `FinetuneSpec` surface.
    """
    ft = (spec or _spec()).finetune
    config = {name: getattr(ft, name) for name in FINETUNE_CONFIG_FIELDS}
    config["shuffle_options"] = tuple(config["shuffle_options"])
    return config

PUBLISHED_RUNTIME_FILES = (
    'artifacts/evidence/attribute_universe_census.json', 'artifacts/evidence/semantics/family_registry.json', 'artifacts/evidence/semantics/tau_sweep.json', 'artifacts/evidence/semantics/value_universe.json',
    'artifacts/models/all-MiniLM-L6-v2/1_Pooling/config.json', 'artifacts/models/all-MiniLM-L6-v2/README.md', 'artifacts/models/all-MiniLM-L6-v2/config.json', 'artifacts/models/all-MiniLM-L6-v2/config_sentence_transformers.json',
    'artifacts/models/all-MiniLM-L6-v2/data_config.json', 'artifacts/models/all-MiniLM-L6-v2/model.safetensors', 'artifacts/models/all-MiniLM-L6-v2/modules.json', 'artifacts/models/all-MiniLM-L6-v2/sentence_bert_config.json',
    'artifacts/models/all-MiniLM-L6-v2/special_tokens_map.json', 'artifacts/models/all-MiniLM-L6-v2/tokenizer.json', 'artifacts/models/all-MiniLM-L6-v2/tokenizer_config.json', 'artifacts/models/all-MiniLM-L6-v2/train_script.py',
    'artifacts/models/all-MiniLM-L6-v2/vocab.txt', 'artifacts/wheels/hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl', 'colab_backend.py', 'config/attribute_ablation.yaml',
    'config/graph_tracks_gnn.yaml', 'config/graph_tracks_hybrid.yaml', 'config/identity_dimensions.yaml', 'config/identity_reviews.json',
    'config/laya.question.json', 'config/model_tracks.yaml', 'config/paths.yaml', 'config/text_track.yaml',
    'config/training.yaml', 'config/training_ANN.yaml', 'config/vocabulary.json', 'data/canonical_records.csv',
    'data/dataset_deduped.csv', 'data/final_validation.csv', 'data/gate_results.csv', 'data/labeled_pairs.csv',
    'data/number_tokens_reference.csv', 'data/prepared/full/worker_1_baseline.pkl.gz', 'data/prepared/full/worker_1_baseline.pkl.gz.json', 'data/prepared/smoke_200/gnn_only.yaml',
    'data/prepared/smoke_200/hybrid.yaml', 'data/prepared/smoke_200/prepared/graph_plan.json', 'data/prepared/smoke_200/prepared/input_manifest.json', 'data/prepared/smoke_200/prepared/listings.json',
    'data/prepared/smoke_200/prepared/report_attributes.json', 'data/prepared/smoke_200/setup_manifest.json', 'data/prepared/smoke_200/shared_training_data.json', 'data/prepared/smoke_200/suite.yaml',
    'data/prepared/smoke_200/text.yaml', 'data/prepared/smoke_200/text_export_request.json', 'data/prepared/smoke_200/text_prepared.pkl.gz', 'data/prepared/smoke_200/text_prepared.pkl.gz.json',
    'data/prepared/smoke_200/text_training_binding.json', 'data/prepared/smoke_200__clean_shared_inputs/listings.json', 'data/prepared/smoke_200__clean_shared_inputs/report_attributes.json', 'data/prepared/smoke_500/gnn_only.yaml',
    'data/prepared/smoke_500/hybrid.yaml', 'data/prepared/smoke_500/prepared/input_manifest.json', 'data/prepared/smoke_500/prepared/listings.json', 'data/prepared/smoke_500/prepared/report_attributes.json',
    'data/prepared/smoke_500/setup_manifest.json', 'data/prepared/smoke_500/suite.yaml', 'data/prepared/smoke_500/text_prepared.pkl.gz', 'data/prepared/smoke_500/text_prepared.pkl.gz.json',
    'data/sku_to_rep.csv', 'dataset.csv', 'pyproject.toml', 'requirements.txt',
    'requirements/graph_tracks.txt', 'scripts/__init__.py', 'scripts/analyze_brand_matching.py', 'scripts/analyze_human_review_features.py',
    'scripts/analyze_incorrect_predictions.py', 'scripts/analyze_model_input.py', 'scripts/apply_bundle_scope_holds.py', 'scripts/apply_identity_review_exclusions.py',
    'scripts/apply_reading_verdicts.py', 'scripts/attribute_capture_audit.py', 'scripts/attribute_probes.py', 'scripts/attribute_universe_census.py',
    'scripts/audit_added_sugar.py', 'scripts/audit_attribute_readings.py', 'scripts/audit_feature_capture.py', 'scripts/audit_gtin_discovery_followup.py',
    'scripts/audit_identity_context.py', 'scripts/audit_identity_dimensions.py', 'scripts/audit_local_identity_evidence.py', 'scripts/audit_resume_state.py',
    'scripts/augment_catalog.py', 'scripts/benchmarks/graph_pooling_cpu.py', 'scripts/brand_differentiation_audit.py', 'scripts/build_attribute_semantics.py',
    'scripts/build_field_slice.py', 'scripts/build_gtin_less_linkage.py', 'scripts/build_stratified_holdout.py', 'scripts/build_validation_slice_sample.py',
    'scripts/census_bundle_scope.py', 'scripts/check_proceed_precision.py', 'scripts/colab_tailscale_userspace.sh', 'scripts/collapse_probe.py',
    'scripts/compare_item_pair_sets.py', 'scripts/compute_strata.py', 'scripts/count_evidence.py', 'scripts/dedupe_invalid_gtin_groups.py',
    'scripts/dedupe_predicate_scorecard.py', 'scripts/diet_manifest.py', 'scripts/encode_prepared_embeddings.py', 'scripts/evaluate_gate_logic.py',
    'scripts/evaluate_jev_identity_fixes.py', 'scripts/export_atlas_embeddings.py', 'scripts/fallback_adjudication.py', 'scripts/feed_reliability.py',
    'scripts/finalize_full_evidence_rebuild.py', 'scripts/flip_validity_audit.py', 'scripts/format_submission.py', 'scripts/fresh_feature_gate_report.py',
    'scripts/install_hpo_connectivity_deps.sh', 'scripts/investigate_identity_residuals.py', 'scripts/kaggle_auth_sync.py', 'scripts/mask_sensitivity_probe.py',
    'scripts/material_carbonation_verdicts.py', 'scripts/measure_gate_regex_fixes.py', 'scripts/measure_pair_difficulty.py', 'scripts/minimal_flip_slice.py',
    'scripts/negative_local_checks.py', 'scripts/negative_missing_probe.py', 'scripts/negative_supply_discriminator.py', 'scripts/permutation_census.py',
    'scripts/profile_colab_setup.py', 'scripts/pseudo_gtin_census.py', 'scripts/raw_tcp_listener.sh', 'scripts/rebuild_balanced_augmentation.py',
    'scripts/rebuild_training_handoff.py', 'scripts/regex_capture_review.py', 'scripts/regex_miss_evidence.py', 'scripts/regex_miss_review.py',
    'scripts/regex_residual_audit.py', 'scripts/render_gtin_repair_results.py', 'scripts/render_identity_fixes.py', 'scripts/repair_augmented_features.py',
    'scripts/repair_reviewed_catalog.py', 'scripts/replay_identity_residuals.py', 'scripts/report_jev_rebuild.py', 'scripts/review_source_consistency.py',
    'scripts/run_colab_ablation.py', 'scripts/run_colab_bundle.sh', 'scripts/run_colab_embeddings.py', 'scripts/run_colab_smoke.sh',
    'scripts/run_full_training.sh', 'scripts/run_raw_tcp_bridge_probe.py', 'scripts/sample_dataset_10k.py', 'scripts/seed_brand_aliases.py',
    'scripts/show_model_input_comparison.py', 'scripts/sid_graph_eval.py', 'scripts/sid_hybrid_eval.py', 'scripts/sid_phase0_report.py',
    'scripts/slice_scale_ladder.py', 'scripts/smoke_graph_tracks.py', 'scripts/triage_remaining_gtin_flavors.py', 'scripts/untrusted_resid_remeasure.py',
    'scripts/validate_postgres_optuna_bridge.py', 'scripts/verdict_biggest_merges.py', 'scripts/verify_suite_archive.py', 'src/__init__.py',
    'src/cli/__init__.py', 'src/cli/colab.py', 'src/cli/colab_bundle.py', 'src/cli/colab_cli_entry.py',
    'src/cli/colab_data_bundle_prep.py', 'src/cli/colab_lane.py', 'src/cli/colab_retention.py', 'src/cli/colab_self_watch.py',
    'src/cli/kaggle_chain.py', 'src/cli/kaggle_cli.py', 'src/cli/kaggle_datasets.py', 'src/cli/kaggle_kernel_templates.py',
    'src/cli/kaggle_kernels.py', 'src/cli/kaggle_lane.py', 'src/cli/kaggle_lifecycle.py', 'src/cli/kaggle_monitor.py',
    'src/cli/kaggle_outputs.py', 'src/cli/kaggle_runtime.py', 'src/cli/laya_lane.py', 'src/cli/log_capture.py',
    'src/core/__init__.py', 'src/core/ann_config.py', 'src/core/archive_reader.py', 'src/core/attribute_conflicts.py',
    'src/core/attribute_decision.py', 'src/core/attribute_universe.py', 'src/core/attribute_vocabulary.py', 'src/core/audit_guard.py',
    'src/core/audit_json.py', 'src/core/blocking.py', 'src/core/bootstrap_ci.py', 'src/core/columns.py',
    'src/core/common.py', 'src/core/coverage_contracts.py', 'src/core/critical_attributes.py', 'src/core/date_evidence.py',
    'src/core/declared_identity.py', 'src/core/deduplication.py', 'src/core/disjoint_sets.py', 'src/core/encoding_inputs.py',
    'src/core/execution_policy.py', 'src/core/gpu_execution.py', 'src/core/graph_diagnostics.py', 'src/core/gtin.py',
    'src/core/hard_negatives.py', 'src/core/identity_policy.py', 'src/core/manifest.py', 'src/core/model_input.py',
    'src/core/nlp.py', 'src/core/pair_policy.py', 'src/core/performance.py', 'src/core/portable_archive.py',
    'src/core/product_context.py', 'src/core/product_dimensions.py', 'src/core/product_selection.py', 'src/core/progress.py',
    'src/core/project_root.py', 'src/core/ranking_metrics.py', 'src/core/record_linkage.py', 'src/core/runtime_inputs.py',
    'src/core/schemas.py', 'src/core/sku_identity.py', 'src/core/step_trace.py', 'src/core/structured_features.py',
    'src/core/sweetener_values.py', 'src/core/text.py', 'src/core/timing.py', 'src/core/tracing.py',
    'src/core/training_profiler.py', 'src/core/unit_canonicalization.py', 'src/core/url_evidence.py', 'src/core/volume_verified.py',
    'src/core/wandb_ctx.py', 'src/core/worker_telemetry.py', 'src/graph_tracks/README.md', 'src/graph_tracks/__init__.py',
    'src/graph_tracks/artifacts.py', 'src/graph_tracks/benchmark.py', 'src/graph_tracks/bundle.py', 'src/graph_tracks/config.py',
    'src/graph_tracks/data.py', 'src/graph_tracks/dvc.py', 'src/graph_tracks/infer.py', 'src/graph_tracks/model.py',
    'src/graph_tracks/pooling.py', 'src/graph_tracks/preflight.py', 'src/graph_tracks/prepare.py', 'src/graph_tracks/prepared_inputs.py',
    'src/graph_tracks/report.py', 'src/graph_tracks/report_attributes.py', 'src/graph_tracks/report_manifest.py', 'src/graph_tracks/report_slices.py',
    'src/graph_tracks/setup.py', 'src/graph_tracks/text_cache.py', 'src/graph_tracks/tracking.py', 'src/graph_tracks/train.py',
    'src/graph_tracks/worker_package.py', 'src/model_tracks/__init__.py', 'src/model_tracks/ablation.py', 'src/model_tracks/ablation_cohort.py',
    'src/model_tracks/ablation_inputs.py', 'src/model_tracks/ablation_retrieval.py', 'src/model_tracks/archive_verification.py', 'src/model_tracks/baseline_ablation.py',
    'src/model_tracks/baseline_export.py', 'src/model_tracks/colab.py', 'src/model_tracks/config.py', 'src/model_tracks/data_gate.py',
    'src/model_tracks/embedding_forward.py', 'src/model_tracks/embedding_staging.py', 'src/model_tracks/incremental.py', 'src/model_tracks/live_logs.py',
    'src/model_tracks/local_complete.py', 'src/model_tracks/package.py', 'src/model_tracks/parallel.py', 'src/model_tracks/portable_layout.py',
    'src/model_tracks/post_training_ablation.py', 'src/model_tracks/preflight.py', 'src/model_tracks/publish.py', 'src/model_tracks/resource_profile.py',
    'src/model_tracks/resume.py', 'src/model_tracks/run.py', 'src/model_tracks/run_history.py', 'src/model_tracks/run_retention.py',
    'src/model_tracks/shared_graph_data.py', 'src/model_tracks/smoke_inputs.py', 'src/model_tracks/snapshot_completion.py', 'src/model_tracks/staged_ablation.py',
    'src/model_tracks/telemetry.py', 'src/model_tracks/text_export.py', 'src/model_tracks/text_report.py', 'src/model_tracks/training_data.py',
    'src/model_tracks/worker.py', 'src/ner/__init__.py', 'src/ner/colab_ner.py', 'src/ner/config_loader.py',
    'src/ner/ner.py', 'src/ner/ner_product_attributes.py', 'src/pipeline.py', 'src/predict_items.py',
    'src/training/__init__.py', 'src/training/ann_refresh.py', 'src/training/artifact_store.py', 'src/training/attestation.py',
    'src/training/attribute_agreement_audit.py', 'src/training/attribute_separation.py', 'src/training/attrition.py', 'src/training/audit_identity_retention.py',
    'src/training/balanced_augmentation.py', 'src/training/base_data.py', 'src/training/blocking_audit.py', 'src/training/build_ann_index.py',
    'src/training/build_final_validation.py', 'src/training/build_reference.py', 'src/training/build_second04_pairs.py', 'src/training/build_title_attribute_evidence.py',
    'src/training/cluster_quality_plot.py', 'src/training/complete_colab_worker.py', 'src/training/composition_plot.py', 'src/training/data_prep.py',
    'src/training/data_quality_audit.py', 'src/training/dedupe.py', 'src/training/difficulty.py', 'src/training/dvc_store.py',
    'src/training/evaluate_models.py', 'src/training/folds.py', 'src/training/gate_replay.py', 'src/training/generate_rand_stratum_sweep.py',
    'src/training/generate_rand_truth.py', 'src/training/generate_training_report.py', 'src/training/handoff.py', 'src/training/hnsw_index.py',
    'src/training/hpo.py', 'src/training/hpo_champions.py', 'src/training/hpo_control_plane.py', 'src/training/hpo_fencing.py',
    'src/training/hpo_metrics.py', 'src/training/hpo_persistence.py', 'src/training/labeled_pairs.py', 'src/training/losses.py',
    'src/training/masking.py', 'src/training/negative_supply.py', 'src/training/package_gate_impact_audit.py', 'src/training/plots.py',
    'src/training/preparation_run.py', 'src/training/prepare_all.py', 'src/training/prepare_all_trace.py', 'src/training/prepare_embeddings.py',
    'src/training/prepare_tokens.py', 'src/training/prepared_bundle.py', 'src/training/prioritize_false_merge_components.py', 'src/training/rand_matching.py',
    'src/training/report_plots.py', 'src/training/report_rows.py', 'src/training/rerank.py', 'src/training/robust_validation.py',
    'src/training/run_plan.py', 'src/training/sample_balanced_pairs.py', 'src/training/sampler.py', 'src/training/selftest.py',
    'src/training/semantic_ids.py', 'src/training/sid_graph.py', 'src/training/sid_hybrid.py', 'src/training/strip_audit.py',
    'src/training/token_inputs.py', 'src/training/train.py', 'src/training/train_prepared.py', 'src/training/training.py',
    'src/training/uniformity.py', 'src/training/validation_inference.py', 'src/training/zero_shot_sims.py',
)

# Per decision kind: the required header columns, the state column the
# decision state is built from, and what the run decides. THE CSV BINDING
# ITSELF lives in the SSOT (LayaSpec.decision_csv_bindings, resolved
# through core.common.F here) — never duplicated in a second registry.
DECISION_BINDINGS: dict[str, dict[str, Any]] = {
    "attribute": {
        "wanted_columns": ("sku_id", "sku_name_eng", "attribute"),
        "state_column": "attribute",
        "description": ("attribute-channel typed decision over the export's "
                        "attribute text. NOT a replacement for the frozen "
                        "SKU_ITEM/GTIN attribution path — it asks whether "
                        "laya's calibrated route agrees on the same "
                        "attributes the task layout already fixed"),
    },
    "identity": {
        "wanted_columns": ("gtin1", "gtin2", "true_label"),
        "state_column": "attribute_pairs",
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
        "description": ("fine-tune the convaiinnovations/laya checkpoint on "
                        "the verified-label JSONL corpus (state + identity "
                        "cases) via the real laya-train CLI on a single T4"),
    },
    "finetune-eval": {
        # Not a per-row decision CSV either: the eval-only kernel scores an
        # attached fine-tuned checkpoint against the corpus split, so the
        # corpus row keys are the contract (mirrors the finetune entry).
        "wanted_columns": ("state", "questions", "expected"),
        "state_column": "state",
        "description": ("HELD-OUT eval-only score of an attached fine-tuned "
                        "laya checkpoint on the corpus test split: loads the "
                        "checkpoint, runs calibration_records + "
                        "evaluate_records, writes eval_report.json. No "
                        "training, no Hub fetch."),
    },
}


def _spec():
    return training_cfg().laya


def decision_binding(decision_kind: str) -> str:
    """SSOT F-binding for a decision kind (fail-loud on an unknown kind)."""
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    bindings = _spec().decision_csv_bindings
    binding = bindings.get(decision_kind)
    if not binding:
        raise RuntimeError(
            f"config laya.decision_csv_bindings carries no entry for "
            f"{decision_kind!r}; name the config/paths.yaml files: binding "
            "before staging")
    return binding


def staging_dir() -> Path:
    """The lane staging root (TRAIN_ROOT-relative; SSOT laya.staging_dir)."""
    return (TRAIN_ROOT / _spec().staging_dir).resolve()


def lane_logs_dir() -> Path:
    """The lane transcript dir (one canonical roof: TRAIN_ROOT/logs)."""
    return TRAIN_ROOT / "logs" / "laya"


def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix.

    Mirrors kaggle_lane._stamp: the CET convention landed there (owner
    order 2026-10-07) and this lane follows it; the "[laya-lane UTC-stamp]"
    phrasing in the relaunch brief predates that convention.
    """
    return (f"[laya-lane "
            f"{datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}]")


def _log_lane(line: str) -> None:
    """Timestamped lane logging: console plus one fresh lane log per run.

    The file is truncated on the first write of this process and appended
    afterwards, so a new run writes over the previous run's transcript
    (owner order 2026-10-07: fresh file per run, never append-sprawl).
    Best-effort on the file side — a log-write failure is printed and
    never allowed to mask the operation's own outcome.
    """
    global _LANE_LOG_STARTED
    stamp = f"{datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}"
    print(f"[laya-lane {stamp}] {line}", flush=True)
    try:
        log_dir = lane_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if _LANE_LOG_STARTED else "w"
        with (log_dir / LANE_LOG_NAME).open(mode, encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
        _LANE_LOG_STARTED = True
    except OSError as error:
        print(_stamp(), f"[laya-lane] lane.log write failed ({error}); "
              "continuing", flush=True)


# ── decision-input staging (dry-safe; fail-loud on a missing contract) ─────
def _measure_csv(path: Path, wanted_columns: tuple[str, ...]) -> dict[str, Any]:
    """Stdlib CSV census: header check + row count + sha256 + bytes.

    Deliberately NOT pandas: staging must run anywhere (including a box
    without the frame stack), and the contract is only the columns.
    """
    import csv as _csv

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"decision input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = _csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"decision input has no header row: {path}")
        missing = [column for column in wanted_columns if column not in header]
        if missing:
            raise ValueError(
                f"decision input {path.name} is missing columns {missing} "
                f"(header: {header})")
        rows = sum(1 for _ in reader)
    if rows == 0:
        raise ValueError(f"decision input has no data rows: {path}")
    return {"rows": rows, "columns": list(header),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


# The accuracy/F1 metric contract a harvest agent needs when a decision
# CSV carries ground-truth labels (owner order 2026-10-07: "add accuracy
# + f1 ... give it all the pairs we know are the same"): the EXPECTED
# row count + label distribution computed from the csv itself + the gold
# columns the harvest reads — so the harvest computes accuracy/F1 against
# these WITHOUT re-deriving the expectation.
_METRIC_EXPECTATION_KEYS = ("expected_rows", "expected_label_distribution",
                            "metric_expectation")


def _metric_expectation(path: Path, columns: list[str]) -> dict[str, Any]:
    """Expected-metric contract fields for a labeled decision CSV.

    Computes `expected_rows` + `expected_label_distribution` from the
    csv's own `true_label` column (stdlib read; a csv without the column
    returns {} — the contract only ever attaches to labeled decisions).
    Fail-loud: a `true_label` column carrying values outside {0, 1}
    raises before any receipt lands.
    """
    if "true_label" not in columns:
        return {}
    import csv as _csv
    from collections import Counter

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = _csv.DictReader(handle)
        labels = Counter(row["true_label"] for row in rows)
    unknown = sorted(set(labels) - {"0", "1"})
    if unknown:
        raise ValueError(
            f"decision input {path.name} carries true_label values "
            f"outside {{0, 1}}: {unknown}")
    return {
        "expected_rows": sum(labels.values()),
        "expected_label_distribution": {
            label: labels[label] for label in sorted(labels)},
        "metric_expectation": {
            "accuracy_gold": "label",
            "f1_gold": "identity_claim-vs-true_label",
        },
    }


def stage_decision_input(kind: str, *, decision_kind: str,
                         override: Path | None = None) -> dict[str, Any]:
    """Stage ONE decision CSV under results/laya_lane/<kind>/<decision>/.

    The staged copy is receipted with the measured census (rows, columns,
    sha256, bytes) — the transport-identity contract the kaggle-lane
    package receipts carry. `override` names an alternate source CSV on
    this box (e.g. dataset_50pct.csv for the half cohort).
    """
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}")
    from core.common import F

    entry = DECISION_BINDINGS[decision_kind]
    source = Path(override) if override else F[decision_binding(decision_kind)]
    stage = staging_dir() / kind / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    census = _measure_csv(source, entry["wanted_columns"])
    destination = stage / source.name
    shutil.copy2(source, destination)
    receipt = {
        "kind": kind, "decision_kind": decision_kind,
        "binding": decision_binding(decision_kind),
        "source": str(source), "staged": str(destination),
        "rows": census["rows"], "columns": census["columns"],
        "sha256": census["sha256"], "bytes": census["bytes"],
        "description": entry["description"],
        **_metric_expectation(source, census["columns"]),
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged decision input [{kind}/{decision_kind}] "
              f"{source.name} rows={census['rows']} "
              f"sha256={census['sha256'][:12]} -> {destination}")
    return receipt


def stage_question_schema(kind: str, *,
                          override: Path | None = None) -> dict[str, Any]:
    """Stage the laya.question schema under results/laya_lane/<kind>/.

    Dry-safe + fail-loud: a missing schema file raises FileNotFoundError
    and a schema without a 'questions' dict raises ValueError (no silent
    empty schema placeholder is ever staged).
    """
    spec = _spec()
    source = Path(override) if override else TRAIN_ROOT / spec.question_schema
    stage = staging_dir() / kind / "question"
    stage.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"laya.question schema not found: {source}")
    schema = json.loads(source.read_text(encoding="utf-8"))
    questions = schema.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            f"laya.question schema at {source} carries no 'questions' dict")
    destination = stage / QUESTION_SCHEMA_FILE
    shutil.copy2(source, destination)
    receipt = {
        "question_schema": spec.question_schema,
        "staged": str(destination),
        "questions": sorted(questions),
        "sha256": sha256_file(source),
    }
    atomic_write_json(receipt, stage / "question.receipt.json")
    _log_lane(f"staged question schema [{kind}] {source.name} "
              f"({len(questions)} questions) -> {destination}")
    return receipt


# The laya payloads ride the ATTACHED er-laya-requests dataset, never a
# repo clone: the shipped core.runtime_inputs.checkout_preflight_script
# emits `_runtime_root = Path(root)` for clone lanes and a laya payload
# defines no root (NameError at line 36 killed the first remote boot).
# This dedicated template instead verifies THE ATTACHED INPUTS: files
# land under /kaggle/input/<slug>/ and resolve_input rglob's by name.
# REPOSITORY/BRANCH/REVISION/_runtime_files stay assigned at top level
# so the laya push gate still literal-evals them (_staged_laya_push_
# preflight — the push gate lives in this lane; the shared clone-lane
# staged_kernel_preflight checked the git tree for the attached-inputs
# inventory, which never matches a dataset-carried payload).
LAYA_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@DECISION_CSV@", @QUESTION_SCHEMA_FILE@)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED dataset inputs (the er-laya-requests dataset
    mounts under /kaggle/input/<slug>/ and resolve_input searches INPUTS
    recursively by name); fail loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The fine-tune corpus travels as its own attached dataset: this preflight
# verifies THE ATTACHED INPUTS (train/dev/test JSONL land under
# /kaggle/input/<slug>/ and rglob finds them by name). It bakes the same
# REPOSITORY/BRANCH/REVISION/_runtime_files inventory the lane push gate
# literal-evals.
FINETUNE_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@TRAIN_JSONL@", "@DEV_JSONL@", "@TEST_JSONL@")
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus inputs (the finetune dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The eval-only kernel attaches the SAME corpus dataset and verifies the ONE
# held-out split it scores (rglob finds the JSONL under /kaggle/input/<slug>/).
# The CHECKPOINT is a separate attached dataset, resolved in-kernel by the
# `rl_agent_config.json` rglob (never vendored here): the push gate's
# `_runtime_files` inventory can only verify files that live in the staged
# dataset_payload, so the checkpoint inventory stays out of it and fails loud
# in `resolve_checkpoint()` instead.
FINETUNE_EVAL_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@EVAL_JSONL@",)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus split (the corpus dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# ── kernel / notebook payload composition ──────────────────────────────────
def _kernel_script_gate(script: str) -> None:
    """Staging-time AST gate (kaggle_lane._kernel_script_gate mirror):
    never stage an unparseable payload or one that references an undeclared
    UPPER_CASE template constant (the v4 NameError error class)."""
    import ast

    parsed = ast.parse(script)
    defined = {node.id for stmt in ast.walk(parsed)
               if isinstance(stmt, ast.Assign)
               for node in stmt.targets if isinstance(node, ast.Name)}
    undeclared = {expr.id for expr in ast.walk(parsed)
                  if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load)
                  and expr.id.isupper() and expr.id not in defined}
    if undeclared:
        raise ValueError(f"staged kernel uses undeclared constants: "
                         f"{sorted(undeclared)}; regenerate the template")


def _module_scope_gate(script: str) -> None:
    """Post-substitution module-scope AST scan (regression pin for the
    BUG-1 NameError class: a template substitution emitting an undefined
    TOP-LEVEL load — e.g. `_runtime_root = Path(root)` — can never stage
    again). Every name Loaded at module scope (compound statements
    recurse; function/class bodies are their own scopes and skipped) must
    be a builtin, an import binding, a def/class name, or a bound target.
    Raise loud (never a silent payload) BEFORE the atomic writes.
    """
    import ast
    import builtins

    compile(script, "<laya-payload>", "exec")
    tree = ast.parse(script)
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    bound.add(alias.asname or alias.name.split(".")[0])
                elif alias.name != "*":
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    loaded = {node.id
              for stmt in tree.body
              if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))
              for node in ast.walk(stmt)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
              and not (node.id.startswith("__") and node.id.endswith("__"))}
    undeclared = sorted(loaded - bound)
    if undeclared:
        raise ValueError(f"staged kernel loads undefined top-level names: "
                         f"{undeclared}; a NameError-class payload must "
                         "never stage again")


def _template(script: str, values: dict[str, str]) -> str:
    for token, replacement in values.items():
        script = script.replace(f"@{token}@", replacement)
    return script


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {TRAIN_ROOT}: {result.stderr.strip()}")
    return result.stdout.strip()


def decision_tag() -> str:
    """UTC-stamped run tag (the laya-lane stamp SURFACE stays UTC inside
    the payload because the remote session may not share this box's zone;
    the console/lane.log stamp itself stays Europe/Paris per the landed
    kaggle_lane convention)."""
    return datetime.now(ZoneInfo("UTC")).strftime("%m%dT%H%M%SZ")


DECISION_KERNEL_SCRIPT = '''\
"""ER typed decisions via laya on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling (2xT4 -> 1xT4; never requests the double
accelerator): pins one CUDA device, installs laya over pip, reads the
question schema + decision CSV attached as the kaggle dataset inputs,
runs the router's typed questions per row, and writes the decision
results + receipt into /kaggle/working for hash-verified fetch-back.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
CHECKPOINT_HUB = "@CHECKPOINT_HUB@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"
BATCH_SIZE = @BATCH_SIZE@
MIN_CONFIDENCE = @MIN_CONFIDENCE@
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_upgrade_laya():
    """laya installs over pip; torch must already be 2.14 cu13x."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya_lane payload dataset)")


def predict_batch(agent, states, questions):
    """One batched call per chunk; min_confidence only when gated."""
    if MIN_CONFIDENCE > 0:
        return agent.predict_batch(states, questions,
                                   min_confidence=MIN_CONFIDENCE)
    return agent.predict_batch(states, questions)


def main():
    pip_upgrade_laya()
    device = pick_device()
    import laya
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    agent = laya.load(CHECKPOINT_HUB, device=device)
    rows = []
    with decision_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    log("loaded " + str(len(rows)) + " " + DECISION_KIND
        + " state rows from " + decision_csv.name)
    results = []
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start:start + BATCH_SIZE]
        states = [row.get(STATE_COLUMN, "") for row in chunk]
        answers = predict_batch(agent, states, questions)
        for row, answer in zip(chunk, answers):
            answer["_row"] = {key: value for key, value in row.items()
                              if key != STATE_COLUMN}
            results.append(answer)
        log("decided " + str(min(start + BATCH_SIZE, len(rows))) + "/"
            + str(len(rows)) + " rows")
    WORKING.mkdir(parents=True, exist_ok=True)
    out = WORKING / (DECISION_KIND + ".decisions.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item) + "\\n")
    log("wrote " + str(out) + " (" + str(len(results)) + " decisions)")
    receipt = {
        "gpu_kind": DECISION_KIND,
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "checkpoint_hub": CHECKPOINT_HUB,
        "batch_size": BATCH_SIZE,
        "min_confidence": MIN_CONFIDENCE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "decision_csv_sha256": hashlib.sha256(
            decision_csv.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_decision.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_decision.tar.gz", "w:gz") as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_decision.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged laya_decision.tar.gz + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''

EVAL_KERNEL_SCRIPT = '''\
"""laya-evals harness score on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling; installs laya over pip, derives the laya-evals
JSONL (state, questions, expected) from the staged identity decision CSV +
question schema, runs `laya-evals run`, and stages the report.json +
report.md + receipt into /kaggle/working.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
QUESTION_SCHEMA_FILE = @QUESTION_SCHEMA_FILE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name
        + " (expected the staged laya_lane payload dataset)")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", LAYA_PACKAGE], check=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    questions_path = resolve_input(QUESTION_SCHEMA_FILE)
    questions = json.loads(questions_path.read_text())["questions"]
    decision_csv = resolve_input(DECISION_CSV)
    WORKING.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = WORKING / "identity_evals.jsonl"
    with decision_csv.open(newline="") as handle, dataset_jsonl.open(
            "w", encoding="utf-8") as out:
        for row in csv.DictReader(handle):
            expected = int(row["true_label"])
            out.write(json.dumps({
                "state": row.get("attribute_pairs", ""),
                "questions": questions,
                "expected": {"identity_claim": expected},
            }) + "\\n")
    log("staged " + str(dataset_jsonl) + " from " + decision_csv.name)
    command = [sys.executable, "-m", "laya.evals", "run",
               str(dataset_jsonl), "--json", str(WORKING / "report.json")]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    receipt = {
        "gpu_kind": "laya-cli-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "question_schema_sha256": hashlib.sha256(
            questions_path.read_bytes()).hexdigest(),
        "evals_dataset_sha256": hashlib.sha256(
            dataset_jsonl.read_bytes()).hexdigest(),
    }
    (WORKING / "laya_evals.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    log("staged laya-evals report + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''

NOTEBOOK_SCRIPT = '''\
"""Laya typed-decision Colab payload (cli.laya_lane; delivery contract).

BOOK-END CONTRACT ONLY — no cli.colab import, NO session call. Operator
instruction set for the notebook wrapper:
  1. pip install laya (torch stack per laya's docs; python >= 3.10);
  2. upload the staged laya.question.json + the staged decision CSV as
     the notebook's own payload mounts;
  3. run the decision loop on a SINGLE GPU (never the double accelerator);
  4. write the receipt json + the export tar under the notebook's own
     export mount, then read it back into results/laya_lane/colab/<op>/.
"""

from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
DECISION_KIND = "@DECISION_KIND@"
RUN_TAG = "@RUN_TAG@"
DECISION_CSV = "@DECISION_CSV@"
STATE_COLUMN = "@STATE_COLUMN@"

WORKING = Path("/content/laya_out")


def main() -> None:
    print("[laya-lane] colab payload staged; delivery contract only",
          flush=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    print("[laya-lane] decision kind: " + DECISION_KIND
          + " batch: " + str(len(list(WORKING.iterdir()))), flush=True)


if __name__ == "__main__":
    main()
'''


# Runtime device patch for the finetune kernel (laya<=0.4.0). `finetune()`
# evaluates the base checkpoint via `calibration_records()` BEFORE
# `train_model()` calls `model.to(device)`, so `load_checkpoint()`'s CPU
# model meets cuda `input_ids` and `index_select` raises "index is on
# cuda:0, different from other tensors on cpu" on the T4. Injected into the
# kernel below at the `@DEVICE_PATCH@` marker; it leaves the recipe, flags
# and the single-T4 rule untouched.
FINETUNE_DEVICE_PATCH_SOURCE = '''\
def force_model_to_device(model, device):
    # Move every module, and every registered buffer (non-persistent ones
    # included), onto `device` before any forward pass.
    import torch
    device = torch.device(device)
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
        module.to(device)
    return model.to(device)


def apply_device_patch():
    # Wrap the two forward entrypoints so the model is on the training
    # device before any forward pass. `calibration_records` is the crash:
    # it runs the base checkpoint on device inputs while the model is
    # still CPU. `train_model` is wrapped for the same invariant.
    from laya import train as laya_train

    original_calibration_records = laya_train.calibration_records

    def calibration_records(model, tok, items, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_calibration_records(
            model, tok, items, device, *args, **kwargs)

    laya_train.calibration_records = calibration_records

    original_train_model = laya_train.train_model

    def train_model(model, tok, items, config, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_train_model(
            model, tok, items, config, device, *args, **kwargs)

    laya_train.train_model = train_model
'''


# Movement/redundancy PERF_PATCH for the finetune kernel (laya>=0.3.29).
# Replaces laya.train.train_model with a faithful copy carrying exactly three
# recipe-neutral changes:
#   (1) the running loss is accumulated as a 0-dim CUDA tensor and synced with
#       ONE .item() per epoch (the stock loop synced twice per micro-step,
#       train.py:696 and :698);
#   (2) the batch tensors the loop consumes are moved to the device ONCE (the
#       stock loop re-moved marker_mask and qtype after _forward had already
#       moved them, train.py:608 vs :671);
#   (3) encode_item is memoized per (item id, option order) so steady-state
#       epochs skip re-tokenizing (train.py:665). draw_option_order is still
#       called in the same per-step order, so the RNG stream is identical.
# The recipe flags, grad-accum window, clipping, scheduler and seed paths are
# byte-for-byte the stock loop. Opt out (patch AND sampler) with
# ER_LAYA_PERF_PATCH=0. Injected at the `@PERF_PATCH@` marker.
FINETUNE_PERF_PATCH_SOURCE = '''\
PERF_PATCH_ENV = "ER_LAYA_PERF_PATCH"


def perf_patch_enabled():
    # one env flag disables BOTH the movement patch and the GPU sampler.
    return os.environ.get(PERF_PATCH_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _perf_train_model(model, tok, items, config, device, max_len, head_max_len,
                      on_epoch_end=None, parallel=False):
    # Faithful copy of laya.train.train_model (0.3.29) with the three
    # recipe-neutral changes described in the source header.
    import torch
    from laya import train as laya_train

    config.validate()
    if not items:
        raise ValueError("no training items")
    amp = (device.type == "cuda") if config.amp is None else bool(config.amp)
    checkpointing = (amp if config.gradient_checkpointing is None
                     else bool(config.gradient_checkpointing))
    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif checkpointing and hasattr(model.encoder,
                                   "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    if config.freeze_encoder:
        model.encoder.eval()

    groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad],
               "lr": config.head_lr}]
    if not config.freeze_encoder:
        groups.insert(0, {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("encoder.") and p.requires_grad],
            "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(items) / config.micro_batch)
    updates = max(1, math.ceil(steps_per_epoch / config.grad_accum)
                  * config.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=config.min_lr)
    scaler = (torch.amp.GradScaler("cuda")
              if amp and device.type == "cuda" else None)

    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g["params"]]
    history = []
    cache = {}
    hits = lookups = 0
    for epoch in range(config.epochs):
        epoch_items = list(items)
        random.Random(config.seed + epoch).shuffle(epoch_items)
        sigma = laya_train.sigma_at(epoch, config.epochs, config.sigma_start,
                                    config.sigma_end)
        total, n_steps = None, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = []
            for it in epoch_items[start:start + config.micro_batch]:
                order = laya_train.draw_option_order(
                    it, order_rng, config.shuffle_options)
                key = (id(it), tuple(order) if order is not None else None,
                       max_len, head_max_len, parallel)
                encoded = cache.get(key)
                if encoded is None:
                    encoded = laya_train.encode_item(
                        tok, it, max_len, head_max_len, order, parallel)
                    cache[key] = encoded
                else:
                    hits += 1
                lookups += 1
                chunk.append(encoded)
            batch = laya_train.collate_items([chunk], tok.pad_token_id)
            # (2) one device move for what the loop consumes; _forward's own
            # .to(device) on the same device is then a no-op.
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)
            logits = laya_train._forward(model, batch, device, amp,
                                         config.freeze_encoder)
            if config.loss == "rlcd":
                loss = laya_train.rlcd_loss(logits, target, mask, qtype, sigma,
                                            config.rl_samples, config.w_sph,
                                            config.w_rps)
            else:
                loss = laya_train.soft_ce_loss(logits, target, mask)
            window_start = (n_steps // config.grad_accum) * config.grad_accum
            window_size = min(config.grad_accum,
                              steps_per_epoch - window_start)
            scaled = loss / window_size
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            n_steps += 1
            if (n_steps % config.grad_accum == 0
                    or start + config.micro_batch >= len(epoch_items)):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # (1) stay on-GPU: accumulate the loss, sync once per epoch.
            detached = loss.detach()
            total = detached if total is None else total + detached
            if config.log_every and n_steps % config.log_every == 0:
                print("epoch %d/%d step %d" % (epoch + 1, config.epochs,
                                               n_steps), flush=True)
        mean = (float(total.item() / max(1, n_steps))
                if total is not None else 0.0)
        history.append(mean)
        print("epoch %d/%d mean loss %.4f (encode memo hits %d/%d)"
              % (epoch + 1, config.epochs, mean, hits, lookups), flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean)
    model.eval()
    return history


def apply_perf_patch():
    # Apply BEFORE the device patch so the device wrapper closes over (and
    # preserves) this loop; opt out with ER_LAYA_PERF_PATCH=0.
    if not perf_patch_enabled():
        print("[perf-patch] disabled via " + PERF_PATCH_ENV, flush=True)
        return False
    from laya import train as laya_train
    laya_train.train_model = _perf_train_model
    print("[perf-patch] laya.train.train_model patched: on-GPU loss (1 sync/"
          "epoch), single device move, encode memoization", flush=True)
    return True


def start_gpu_sampler():
    # 1 Hz nvidia-smi sampler -> /kaggle/working/gpu_usage.log (rides the
    # fetch-back tar). No-op when the flag is off, nvidia-smi is absent, or
    # the box is CPU-only.
    if not perf_patch_enabled():
        return None
    if shutil.which("nvidia-smi") is None:
        log("gpu sampler: nvidia-smi absent; skipping")
        return None
    path = WORKING / "gpu_usage.log"
    stop = threading.Event()
    query = ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"]

    def _loop():
        while not stop.is_set():
            try:
                proc = subprocess.run(query, capture_output=True, text=True,
                                      timeout=5)
                if proc.returncode == 0 and proc.stdout.strip():
                    line = proc.stdout.strip().splitlines()[0]
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\\n")
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(1.0)

    thread = threading.Thread(target=_loop, name="gpu-sampler", daemon=True)
    thread.start()
    log("gpu sampler: 1 Hz -> " + str(path))
    return stop, thread


def stop_gpu_sampler(handle):
    if not handle:
        return
    stop, thread = handle
    stop.set()
    thread.join(timeout=5)


def summarize_gpu_usage(path):
    if not path.is_file():
        return None
    utils, mems, total_mb = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util = float(parts[0].rstrip("%").strip())
            used = float(parts[1].split()[0])
            total = float(parts[2].split()[0])
        except (ValueError, IndexError):
            continue
        utils.append(util)
        mems.append(used)
        total_mb = total
    if not utils:
        return None
    return {
        "samples": len(utils),
        "util_min_pct": min(utils),
        "util_max_pct": max(utils),
        "util_mean_pct": sum(utils) / len(utils),
        "mem_used_peak_mb": max(mems),
        "mem_total_mb": total_mb,
    }
'''


FINETUNE_KERNEL_SCRIPT = '''\
"""ER laya fine-tune on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling (2xT4 -> 1xT4; never requests the double
accelerator): pins one CUDA device, installs laya over pip (pinned
`laya>=0.3.29`), reads the attached JSONL corpus (train/dev/test +
receipt, the er-laya-train dataset), extracts the attached base-model
archive (the er-laya-base dataset; the shipped convaiinnovations/laya
checkpoint) to a local dir, builds the FULL `laya.train.TrainConfig` from
the YAML-driven `FINETUNE_CONFIG` (every trainer knob is config SSOT), and
calls `laya.train.finetune(...)` directly with the extracted DIRECTORY as
the base -- so `resolve_checkpoint_dir` takes the isdir branch and NEVER
calls the Hub. It writes the checkpoint + a receipt into /kaggle/working
for hash-verified fetch-back.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
TRAIN_JSONL = "@TRAIN_JSONL@"
DEV_JSONL = "@DEV_JSONL@"
TEST_JSONL = "@TEST_JSONL@"
BASE_MODEL_ARCHIVE = "@BASE_MODEL_ARCHIVE@"
BASE_MODEL_DIR = "@BASE_MODEL_DIR@"
FINETUNE_DEVICE = "@FINETUNE_DEVICE@"
FINETUNE_CONFIG = @FINETUNE_CONFIG@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

@DEVICE_PATCH@

@PERF_PATCH@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4).

    `auto`/`cuda` require a live cuda session and resolve to device 0; an
    explicit other device (e.g. `cpu`) is passed through while cuda stays
    pinned to device 0, so a second accelerator is never visible."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if FINETUNE_DEVICE not in ("auto", "cuda"):
        log("device configured: " + FINETUNE_DEVICE
            + " (cuda pinned to device 0)")
        return FINETUNE_DEVICE
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya finetune dataset)")


def open_zstd(path):
    """Open a `.tar.zst` stream with whichever zstd binding the session has.

    The base checkpoint ships as a zstd tar (the project transport); never
    falls back to the network for the checkpoint itself. Python 3.14 exposes
    `compression.zstd`; the Kaggle 3.13 image needs `zstandard` (installed
    on demand only when neither binding is importable)."""
    try:
        from compression import zstd
        return zstd.open(path, "rb")
    except ImportError:
        pass
    try:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", "zstandard"], check=True)
    import zstandard
    return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))


def extract_base_model(archive):
    """Extract the attached base-model tar.zst and return the directory that
    carries rl_agent_config.json.

    `--base` then points at a LOCAL dir, so laya's resolve_checkpoint_dir
    takes the isdir branch and NEVER calls snapshot_download (the HF
    dependency is gone from this path)."""
    destination = WORKING / "base_model"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stream = open_zstd(str(archive))
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            try:
                tar.extractall(destination, filter="data")
            except TypeError:
                tar.extractall(destination)
    finally:
        stream.close()
    candidate = destination / BASE_MODEL_DIR
    if (candidate / "rl_agent_config.json").is_file():
        return candidate
    for found in sorted(destination.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError(
        "base-model archive carried no rl_agent_config.json")


def run_laya_finetune(train_path, dev_path, base_model, out_dir, device):
    """Apply the PERF patch then the device patch, build the FULL
    `TrainConfig` from FINETUNE_CONFIG, and call `laya.train.finetune`
    directly.

    The `laya-train` CLI only exposes a subset of the trainer surface, so
    the non-CLI knobs are set by constructing the config here and calling
    `finetune` in-process (the monkeypatches reach the same
    `train_model`/`calibration_records` entrypoints finetune calls). PERF
    first so the device wrapper closes over the patched train_model (see
    FINETUNE_PERF_PATCH_SOURCE)."""
    apply_perf_patch()
    apply_device_patch()
    from laya import train as laya_train
    config = laya_train.TrainConfig(**FINETUNE_CONFIG,
                                    eval_data=str(dev_path))
    config.validate()
    log("TrainConfig: " + json.dumps(FINETUNE_CONFIG, sort_keys=True))
    return laya_train.finetune(
        data=str(train_path), model_dir=str(base_model),
        output_dir=str(out_dir), config=config, device=device)


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pip_install_laya()
    device = pick_device()
    train = resolve_input(TRAIN_JSONL)
    dev = resolve_input(DEV_JSONL)
    test = resolve_input(TEST_JSONL)
    log("corpus: " + train.name + " + " + dev.name + " (+ " + test.name + ")")
    WORKING.mkdir(parents=True, exist_ok=True)
    archive = resolve_input(BASE_MODEL_ARCHIVE)
    log("base-model archive: " + str(archive))
    base_model = extract_base_model(archive)
    log("base model: " + str(base_model))
    out_dir = WORKING / "checkpoint"
    gpu_handle = start_gpu_sampler()
    try:
        summary = run_laya_finetune(train, dev, base_model, out_dir, device)
    finally:
        stop_gpu_sampler(gpu_handle)
    receipt = {
        "gpu_kind": "finetune",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "base_model": str(base_model),
        "base_model_archive": str(archive),
        "device": device,
        "perf_patch_enabled": perf_patch_enabled(),
        "gpu_usage": summarize_gpu_usage(WORKING / "gpu_usage.log"),
        "recipe": FINETUNE_CONFIG,
        "output_dir": str(out_dir),
        "corpus_sha256": {TRAIN_JSONL: sha256_of(train),
                          DEV_JSONL: sha256_of(dev),
                          TEST_JSONL: sha256_of(test)},
    }
    if isinstance(summary, dict):
        for key in ("train_items", "calibration_items", "eval_items",
                    "temperature", "epoch_loss"):
            if key in summary:
                receipt[key] = summary[key]
    report = out_dir / "train_report.json"
    if report.is_file():
        receipt["train_report"] = json.loads(report.read_text())
    (WORKING / "laya_finetune.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_finetune.tar.gz", "w:gz") as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_finetune.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged laya_finetune.tar.gz + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''


FINETUNE_EVAL_KERNEL_SCRIPT = '''\
"""ER laya fine-tune EVAL-ONLY on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling: pins one CUDA device, installs laya over pip
(pinned), reads the attached corpus HELD-OUT split + the attached fine-tuned
checkpoint dataset, loads the checkpoint (`laya.train.load_checkpoint`), runs
`calibration_records` + `evaluate_records` on the held-out split, and writes
eval_report.json (before vs after temperature calibration;
eval_mode=held_out, is_held_out=true) + a receipt into /kaggle/working for
hash-verified fetch-back. NO training, NO Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
EVAL_JSONL = "@EVAL_JSONL@"
EVAL_SPLIT = "@EVAL_SPLIT@"
CKPT_DIR_HINT = "@CKPT_DIR@"
CHECKPOINT_PATH = "@CHECKPOINT_PATH@"
BATCH_SIZE = @BATCH_SIZE@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya eval corpus dataset)")


def resolve_checkpoint():
    """The fine-tuned checkpoint dir: an explicit CHECKPOINT_PATH when baked,
    else the CKPT_DIR_HINT match, else the rl_agent_config.json rglob under
    /kaggle/input. Never a Hub fetch."""
    if CHECKPOINT_PATH:
        candidate = Path(CHECKPOINT_PATH)
        if candidate.is_dir() and (candidate / "rl_agent_config.json").is_file():
            return candidate
        raise FileNotFoundError(
            "CHECKPOINT_PATH carries no rl_agent_config.json: "
            + CHECKPOINT_PATH)
    if CKPT_DIR_HINT:
        for found in sorted(INPUTS.rglob(CKPT_DIR_HINT)):
            if (found.is_dir()
                    and (found / "rl_agent_config.json").is_file()):
                return found
    for found in sorted(INPUTS.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError(
        "attached inputs carried no fine-tuned checkpoint "
        "(rl_agent_config.json); attach the checkpoint dataset")


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pip_install_laya()
    device = pick_device()
    import torch
    from laya import train as laya_train
    eval_path = resolve_input(EVAL_JSONL)
    checkpoint = resolve_checkpoint()
    log("checkpoint: " + str(checkpoint))
    log("held-out split: " + str(eval_path) + " (" + EVAL_SPLIT + ")")
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(eval_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise SystemExit(
            "eval split " + eval_path.name + " produced no usable items "
            "(skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=BATCH_SIZE, parallel=parallel)
    before = laya_train.evaluate_records(records)
    fitted = laya_train.fit_temperature_map(records)
    after = laya_train.evaluate_records(
        records, fitted.get("temperature"),
        fitted.get("temperature_by_options"))
    comparison = {
        "delta_accuracy": round(
            after["accuracy"] - before["accuracy"], 4),
        "delta_ece": (round(after["ece"] - before["ece"], 4)
                      if after["ece"] is not None
                      and before["ece"] is not None else None),
        "delta_brier": (round(after["brier"] - before["brier"], 4)
                        if after["brier"] is not None
                        and before["brier"] is not None else None),
        "delta_mean_confidence": round(
            after["mean_confidence"] - before["mean_confidence"], 4),
    }
    report = {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": eval_path.name,
        "eval_split": EVAL_SPLIT,
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "run_tag": RUN_TAG,
        "before": before,
        "after": after,
        "comparison": comparison,
        "temperature": fitted.get("temperature"),
        "temperature_by_options": fitted.get("temperature_by_options"),
    }
    WORKING.mkdir(parents=True, exist_ok=True)
    report_path = WORKING / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\\n",
                           encoding="utf-8")
    log("wrote " + str(report_path) + " (accuracy before/after "
        + str(before["accuracy"]) + "/" + str(after["accuracy"]) + ")")
    receipt = {
        "gpu_kind": "finetune-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "eval_split": EVAL_SPLIT,
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_jsonl_sha256": sha256_of(eval_path),
        "checkpoint": str(checkpoint),
        "report_sha256": sha256_of(report_path),
    }
    (WORKING / "laya_finetune-eval.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_finetune_eval.tar.gz", "w:gz") as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_finetune_eval.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged eval_report.json + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''


def stage_dataset_payload(decision_kind: str, *, dataset_slug: str,
                          question_source: Path,
                          decision_source: Path) -> dict[str, Any]:
    """Stage the DATASET payload for spec.dataset_slug (dry-safe).

    Builds results/laya_lane/kaggle/<decision>/dataset_payload/: the
    kaggle `dataset-metadata.json` (title/id/licenses per the kaggle-lane
    payload shape) + copies of the staged laya.question.json and the
    staged decision CSV RENAMED to dataset.csv (the DECISION_CSV name the
    kernel resolves via INPUTS.rglob once the dataset attaches).
    """
    if not dataset_slug:
        raise RuntimeError(
            "config laya.dataset_slug is unset; name the input dataset "
            "(owner/slug) before staging")
    stage = staging_dir() / "kaggle" / decision_kind / DATASET_PAYLOAD_DIR
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {"title": "er laya requests", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    shutil.copy2(question_source, stage / QUESTION_SCHEMA_FILE)
    shutil.copy2(decision_source, stage / DATASET_CSV_NAME)
    payload_files = (QUESTION_SCHEMA_FILE, DATASET_CSV_NAME)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "metadata": metadata,
        "files": {name: sha256_file(stage / name) for name in payload_files},
    }
    atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
    _log_lane(f"staged dataset payload [{decision_kind}] {dataset_slug} "
              f"files={list(payload_files)} -> {stage}")
    return receipt


def package_base_model(*, source_dir: Path, dataset_slug: str,
                       archive_name: str, member_name: str,
                       output_dir: Path | None = None) -> dict[str, Any]:
    """Package the local base checkpoint tree as a `.tar.zst` dataset payload.

    The fine-tune base checkpoint is 647 MB (plain git caps at 100 MB), so
    it ships the way the project ships large payloads: a zstd tar attached
    as a kaggle dataset. Streams `source_dir`'s whole tree under ONE
    top-level member named `member_name`, so extraction yields a dir
    carrying `rl_agent_config.json` (exactly what laya's
    resolve_checkpoint_dir needs to take the local-dir branch). Uses the
    project's zstd tar writer (core.archive_reader.tar_archive) and lands
    the kaggle `dataset-metadata.json` beside the archive. Results live
    under results/laya_lane/base_model (never committed).
    """
    from core.archive_reader import tar_archive

    source_dir = Path(source_dir)
    if not (source_dir / "rl_agent_config.json").is_file():
        raise FileNotFoundError(
            f"base-model source {source_dir} carries no rl_agent_config.json")
    if not dataset_slug:
        raise RuntimeError(
            "config laya.base_model_dataset is unset; name the base-model "
            "dataset (owner/slug) before packaging")
    stage = Path(output_dir) if output_dir else staging_dir() / "base_model"
    stage.mkdir(parents=True, exist_ok=True)
    archive_path = stage / archive_name
    if archive_path.exists():
        archive_path.unlink()
    with tar_archive(archive_path, "w") as archive:
        archive.add(str(source_dir), arcname=member_name, recursive=True)
    metadata = {"title": "er laya base", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "archive": archive_name,
        "member": member_name,
        "source": str(source_dir),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "metadata": metadata,
    }
    atomic_write_json(receipt, stage / "base_model.receipt.json")
    _log_lane(f"packaged base model {dataset_slug} member={member_name} "
              f"archive={archive_name} bytes={receipt['bytes']} -> {stage}")
    return receipt


def publish_laya_dataset(decision_kind: str, *, run_tag: str,
                         execute: bool) -> dict[str, Any]:
    """`--execute`-gated create-or-version of the laya inputs dataset.

    The staged play_500.csv + laya.question.json do NOT travel with
    `kaggle kernels push`: the kernel attaches spec.dataset_slug
    (fbarulli/er-laya-requests), so the dataset must exist remotely
    BEFORE the push. Dry run: returns the plan, never spawns a kaggle
    subprocess. Executed: datasets create when the dataset does not
    exist remotely, else datasets version (-r --dir-mode zip -m
    "laya inputs <tag>"); the helpers are IMPORTED from the kaggle lane
    (cli.kaggle_datasets / cli.kaggle_lane), never copied. The dataset
    version is recorded in the decision receipt.
    """
    payload = staging_dir() / "kaggle" / decision_kind / DATASET_PAYLOAD_DIR
    metadata_file = payload / DATASET_METADATA_FILE
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "payload": str(payload)}
    if not execute:
        plan["note"] = ("the dataset attach rides --execute only "
                        "(mirroring the kernels-push gate)")
        _log_lane(f"dry-run: dataset payload for {decision_kind} would "
                  f"publish to the remote surface at {payload}")
        return plan
    if not metadata_file.is_file():
        raise RuntimeError(
            "--activate gate: no staged dataset payload at "
            f"{payload} ({DATASET_METADATA_FILE} is missing); stage first")
    corpus_kind = decision_kind in (FINETUNE_DECISION, FINETUNE_EVAL_DECISION)
    slug = (_spec().finetune_dataset_slug if corpus_kind
            else _spec().dataset_slug)
    plan["slug"] = slug
    if not slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; name the corpus "
            "dataset (owner/slug) before an executed attach"
            if corpus_kind else
            "config laya.dataset_slug is unset; name the input dataset "
            "(owner/slug) before an executed attach")
    from cli import kaggle_lane as lane
    from cli.kaggle_datasets import KaggleDatasets

    executable = lane._require_kaggle_executable(
        lane._spec().kaggle_executable)
    current = KaggleDatasets._dataset_current_version(slug)
    version = current.get("dataset_version")
    if version:
        plan["action"] = "version"
        # `-r` and `--dir-mode` are one argparse option: `-r --dir-mode
        # zip` fails with "argument -r/--dir-mode: expected one
        # argument" (fail-loud met live on the version path).
        command = [executable, "datasets", "version", "-r", "zip",
                   "-m", f"laya inputs {run_tag}",
                   "-p", str(payload)]
    else:
        plan["action"] = "create"
        command = [executable, "datasets", "create", "-p", str(payload)]
    plan["command"] = command
    _, _ = lane._run_kaggle(command)
    plan["returncode"] = 0
    refreshed = KaggleDatasets._dataset_current_version(slug)
    plan["dataset_version"] = refreshed.get("dataset_version") or version
    plan["published"] = True
    receipt_path = staging_dir() / "kaggle" / decision_kind \
        / f"{decision_kind}.receipt.json"
    if receipt_path.is_file():
        body = json.loads(receipt_path.read_text(encoding="utf-8"))
        body["dataset"].update({"action": plan["action"],
                                "version": plan["dataset_version"]})
        atomic_write_json(body, receipt_path)
    _log_lane(f"published dataset {slug} | action={plan['action']} "
              f"version={plan['dataset_version']} rc=0")
    return plan


def stage_decision_kernel(*, decision_kind: str, revision: str | None = None,
                          run_tag: str | None = None,
                          input_override: Path | None = None,
                          checkpoint_path: Path | None = None
                          ) -> dict[str, Any]:
    """Stage the kaggle decision kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/<decision_kind>/:
      kernel-metadata.json + <code_file>.py + <decision_kind>.receipt.json
      (+ the staged question schema + decision input receipts).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.export_dataset_slug set (the target kernel owner/slug);
      * the question schema + decision CSV stage from their SSOT bindings.
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if decision_kind == FINETUNE_DECISION:
        # The fine-tune kind is corpus-driven (JSONL dataset), not a
        # per-row decision CSV: it has its own staging surface, reached
        # through the same `--decision` dispatch.
        return stage_finetune_kernel(revision=revision, run_tag=run_tag)
    if decision_kind == FINETUNE_EVAL_DECISION:
        # The eval-only kind is corpus- + checkpoint-driven: it has its
        # own staging surface, reached through the same `--decision`
        # dispatch. No training, no Hub.
        return stage_finetune_eval_kernel(revision=revision, run_tag=run_tag,
                                          checkpoint_path=checkpoint_path)
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.export_dataset_slug
    if not slug:
        raise RuntimeError(
            "config laya.export_dataset_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    dataset_slug = spec.dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.dataset_slug is unset; the kernel inputs travel "
            "as that dataset (owner/slug) — the staged play_500.csv + "
            "laya.question.json do NOT ride `kaggle kernels push`; "
            "name it before staging")
    # ── the published-tip invariant ('origin/<branch> == HEAD'): the pin
    # resolves BEFORE any payload write; a pin that misses the fetched
    # branch tip never stages (the 84ce2d0-vs-02dec14 staged-race class).
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    question = stage_question_schema("kaggle")
    input_receipt = stage_decision_input("kaggle",
                                         decision_kind=decision_kind,
                                         override=input_override)
    dataset_receipt = stage_dataset_payload(
        decision_kind, dataset_slug=dataset_slug,
        question_source=Path(question["staged"]),
        decision_source=Path(input_receipt["staged"]))
    stage = staging_dir() / "kaggle" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    code_file = (DECISION_KERNEL_CODE_FILE if decision_kind != "laya-cli-eval"
                 else EVAL_KERNEL_CODE_FILE)
    template = (DECISION_KERNEL_SCRIPT if decision_kind != "laya-cli-eval"
                else EVAL_KERNEL_SCRIPT)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator;
        # the script itself pins CUDA_VISIBLE_DEVICES=0.
        "enable_internet": True,
        # THE INPUTS TRAVEL AS THE DATASET: kernels push does NOT ship
        # the co-located csv/schema files, so resolve_input would
        # FileNotFoundError once boot passes — attach the dataset slug.
        "dataset_sources": [dataset_slug],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    values = {
        "LAYA_PACKAGE": spec.laya_package,
        "CHECKPOINT_HUB": spec.checkpoint_hub,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": DATASET_CSV_NAME,
        "STATE_COLUMN": entry["state_column"],
        "BATCH_SIZE": str(spec.laya_decision_batch_size),
        "MIN_CONFIDENCE": repr(spec.min_router_confidence),
        "QUESTION_SCHEMA_FILE": repr(QUESTION_SCHEMA_FILE),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
    }
    # two-pass substitution (a nested value's @tokens@ are never
    # re-scanned once it is inserted): the preflight bakes its own
    # literal tuple FIRST, then drops into the script — the push gate
    # (_staged_laya_push_preflight) literal-evals `_runtime_files`.
    preflight = _template(LAYA_RUNTIME_PREFLIGHT, values)
    script = _template(template, {**values,
                                  "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / code_file).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": code_file,
        "question_schema": question["staged"],
        "question_sha256": question["sha256"],
        "decision_input": staged_csv,
        "decision_sha256": input_receipt["sha256"],
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "checkpoint_hub": spec.checkpoint_hub,
        "batch_size": spec.laya_decision_batch_size,
        "min_confidence": spec.min_router_confidence,
        "epochs": spec.laya_decision_epochs,
        "state_column": entry["state_column"],
        "evals_enabled": bool(spec.laya_evals_enabled),
        "calibration": bool(spec.calibration),
        "onnx": bool(spec.onnx),
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    # the labeled decision csv's metric contract rides the staged receipt
    # (expected rows + label distribution + the gold columns) so the
    # harvest computes accuracy/F1 without re-deriving the expectation
    receipt.update({key: input_receipt[key]
                    for key in _METRIC_EXPECTATION_KEYS
                    if key in input_receipt})
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    # The decision csv is already co-located in the payload dir (the
    # stage-decision-input destination IS staging/<kind>/<decision>/);
    # the question schema lands beside it for the payload dataset bind.
    shutil.copy2(question["staged"], stage / QUESTION_SCHEMA_FILE)
    _log_lane(f"staged kaggle kernel [{decision_kind}] ({spec.gpu}) "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_finetune_dataset_payload(*, dataset_slug: str,
                                   corpus_dir: Path,
                                   kind: str = FINETUNE_DECISION
                                   ) -> dict[str, Any]:
    """Stage the fine-tune CORPUS as a kaggle dataset payload (dry-safe).

    Builds results/laya_lane/kaggle/<kind>/dataset_payload/: the kaggle
    `dataset-metadata.json` + the three split JSONL + the builder receipt.
    Distinct from the decision datasets (spec.dataset_slug) so a corpus
    version never drops the decision inputs (and vice versa). `kind` names
    the staging surface (`finetune` or the eval-only `finetune-eval`), so
    each kernel's push gate finds its own `dataset_payload` beside it.
    """
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; name the corpus "
            "dataset (owner/slug) before staging")
    corpus_dir = Path(corpus_dir)
    stage = staging_dir() / "kaggle" / kind / DATASET_PAYLOAD_DIR
    stage.mkdir(parents=True, exist_ok=True)
    metadata = {"title": "er laya train", "id": dataset_slug,
                "licenses": [{"name": "other"}]}
    atomic_write_json(metadata, stage / DATASET_METADATA_FILE)
    files = list(FINETUNE_CORPUS_FILES) + [FINETUNE_CORPUS_RECEIPT]
    for name in files:
        source = corpus_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                f"fine-tune corpus file not found: {source} (build it with "
                "scripts/laya_build_dataset.py)")
        shutil.copy2(source, stage / name)
    receipt = {
        "dataset": dataset_slug,
        "payload": str(stage),
        "metadata": metadata,
        "files": {name: sha256_file(stage / name) for name in files},
    }
    atomic_write_json(receipt, stage / "dataset_payload.receipt.json")
    _log_lane(f"staged finetune dataset payload {dataset_slug} "
              f"files={files} -> {stage}")
    return receipt


def stage_finetune_kernel(*, revision: str | None = None,
                          run_tag: str | None = None) -> dict[str, Any]:
    """Stage the kaggle fine-tune kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/finetune/:
      kernel-metadata.json + laya_finetune.py + finetune.receipt.json
      (+ the staged corpus dataset payload).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.finetune_kernel_slug + spec.finetune_dataset_slug set;
      * the corpus JSONL + receipt stage from data/laya (spec constant).
    """
    spec = _spec()
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the laya lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.finetune_kernel_slug
    if not slug:
        raise RuntimeError(
            "config laya.finetune_kernel_slug is unset; name the target "
            "kernel (owner/slug) before staging")
    dataset_slug = spec.finetune_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; the corpus travels "
            "as that dataset (owner/slug) — name it before staging")
    base_dataset = spec.base_model_dataset
    if not base_dataset:
        raise RuntimeError(
            "config laya.base_model_dataset is unset; the base checkpoint "
            "travels as that dataset (owner/slug) — the finetune kernel "
            "must extract a LOCAL dir, never fetch from the Hub")
    # The published-tip invariant ('origin/<branch> == HEAD') resolves
    # BEFORE any payload write, exactly like stage_decision_kernel.
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    dataset_receipt = stage_finetune_dataset_payload(
        dataset_slug=dataset_slug,
        corpus_dir=TRAIN_ROOT / FINETUNE_CORPUS_DIR)
    stage = staging_dir() / "kaggle" / FINETUNE_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": FINETUNE_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator;
        # the script itself pins CUDA_VISIBLE_DEVICES=0.
        "enable_internet": True,
        # THE CORPUS + THE BASE CHECKPOINT TRAVEL AS DATASETS: kernels push
        # does NOT ship the co-located JSONL files, and the 647 MB base
        # checkpoint cannot ride git — attach the corpus slug AND the
        # base-model archive dataset (er-laya-base).
        "dataset_sources": [dataset_slug, base_dataset],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    recipe = finetune_config(spec)
    values = {
        "LAYA_PACKAGE": FINETUNE_LAYA_PACKAGE,
        "BASE_MODEL_ARCHIVE": spec.base_model_archive,
        "BASE_MODEL_DIR": spec.base_model_dir,
        "RUN_TAG": tag,
        "TRAIN_JSONL": FINETUNE_CORPUS_FILES[0],
        "DEV_JSONL": FINETUNE_CORPUS_FILES[1],
        "TEST_JSONL": FINETUNE_CORPUS_FILES[2],
        # The FULL TrainConfig surface rides one repr-baked Python literal:
        # the kernel constructs `TrainConfig(**FINETUNE_CONFIG)` directly.
        "FINETUNE_CONFIG": repr(recipe),
        "FINETUNE_DEVICE": spec.finetune.device,
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
        "DEVICE_PATCH": FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": FINETUNE_PERF_PATCH_SOURCE,
    }
    # two-pass substitution (a nested value's @tokens@ are never re-scanned
    # once it is inserted): the preflight bakes its own literal tuple FIRST,
    # then drops into the script — the push gate
    # (_staged_laya_push_preflight) literal-evals `_runtime_files`.
    preflight = _template(FINETUNE_RUNTIME_PREFLIGHT, values)
    script = _template(FINETUNE_KERNEL_SCRIPT, {**values,
                                                "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / FINETUNE_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": FINETUNE_DECISION,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": FINETUNE_CODE_FILE,
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "laya_package": FINETUNE_LAYA_PACKAGE,
        # The base checkpoint is the attached er-laya-base dataset archive,
        # extracted in-kernel; the Hub id is NOT passed as --base anymore.
        "base_model": {"dataset": base_dataset,
                       "archive": spec.base_model_archive,
                       "dir": spec.base_model_dir},
        "recipe": recipe,
        "device": spec.finetune.device,
        "corpus_dir": str(TRAIN_ROOT / FINETUNE_CORPUS_DIR),
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    atomic_write_json(receipt, stage / f"{FINETUNE_DECISION}.receipt.json")
    _log_lane(f"staged kaggle finetune kernel ({spec.gpu}) run_tag={tag} "
              f"-> {stage}")
    return receipt


def stage_finetune_eval_kernel(*, revision: str | None = None,
                               run_tag: str | None = None,
                               checkpoint_path: Path | None = None
                               ) -> dict[str, Any]:
    """Stage the kaggle fine-tune EVAL-ONLY kernel payload (dry-safe).

    Writes under results/laya_lane/kaggle/finetune-eval/:
      kernel-metadata.json + laya_finetune_eval.py +
      finetune-eval.receipt.json (+ the SAME staged corpus dataset payload
      as the finetune kind).
    Fail-loud preconditions (no silent skip):
      * spec.laya_decision_epochs > 0 (0 = disabled, nothing may stage);
      * spec.finetune_eval_kernel_slug + spec.finetune_dataset_slug set;
      * a checkpoint source: spec.finetune_ckpt_dataset OR checkpoint_path;
      * spec.finetune_eval_split names a corpus split.

    The kernel loads the attached checkpoint and scores the attached split:
    no training, no Hub. The checkpoint dataset is attached as a second
    `dataset_sources` entry; an explicit `checkpoint_path` is baked as
    CHECKPOINT_PATH and takes precedence in-kernel.
    """
    spec = _spec()
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the laya lane is "
            "disabled (no payload may stage a GPU session)")
    slug = spec.finetune_eval_kernel_slug
    if not slug:
        raise RuntimeError(
            "config laya.finetune_eval_kernel_slug is unset; name the target "
            "eval kernel (owner/slug) before staging")
    dataset_slug = spec.finetune_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; the corpus travels "
            "as that dataset (owner/slug) — name it before staging")
    ckpt_dataset = spec.finetune_ckpt_dataset
    if not ckpt_dataset and not checkpoint_path:
        raise RuntimeError(
            "config laya.finetune_ckpt_dataset is unset and no checkpoint "
            "path was given; the eval-only kernel needs a fine-tuned "
            "checkpoint dataset (owner/slug) or an explicit path")
    split = spec.finetune_eval_split
    if split not in FINETUNE_EVAL_SPLIT_FILES:
        raise ValueError(
            f"config laya.finetune_eval_split {split!r} is not one of "
            f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
    # The published-tip invariant resolves BEFORE any payload write.
    repository = training_cfg().kaggle.repository
    branch = training_cfg().kaggle.branch
    revision = revision or _git_revision()
    from core import runtime_inputs
    tip = runtime_inputs.require_published_tip_match(
        revision, repository, branch)
    dataset_receipt = stage_finetune_dataset_payload(
        dataset_slug=dataset_slug,
        corpus_dir=TRAIN_ROOT / FINETUNE_CORPUS_DIR,
        kind=FINETUNE_EVAL_DECISION)
    stage = staging_dir() / "kaggle" / FINETUNE_EVAL_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    dataset_sources = [dataset_slug]
    if ckpt_dataset and not checkpoint_path:
        dataset_sources.append(ckpt_dataset)
    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": FINETUNE_EVAL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        # single T4: the payload never requests the double accelerator.
        "enable_internet": True,
        # THE HELD-OUT SPLIT + THE CHECKPOINT TRAVEL AS DATASETS: the corpus
        # dataset carries the JSONL split, the checkpoint dataset carries
        # the fine-tuned checkpoint dir (rl_agent_config.json).
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    eval_jsonl = FINETUNE_EVAL_SPLIT_FILES[split]
    values = {
        "LAYA_PACKAGE": FINETUNE_LAYA_PACKAGE,
        "RUN_TAG": tag,
        "EVAL_JSONL": eval_jsonl,
        "EVAL_SPLIT": split,
        "CKPT_DIR": spec.finetune_ckpt_dir,
        "CHECKPOINT_PATH": str(checkpoint_path) if checkpoint_path else "",
        "BATCH_SIZE": str(spec.finetune_eval_batch_size),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
    }
    # two-pass substitution (the preflight bakes its own literal tuple
    # first; the push gate literal-evals `_runtime_files`).
    preflight = _template(FINETUNE_EVAL_RUNTIME_PREFLIGHT, values)
    script = _template(FINETUNE_EVAL_KERNEL_SCRIPT,
                       {**values, "RUNTIME_PREFLIGHT": preflight})
    _kernel_script_gate(script)
    _module_scope_gate(script)
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / FINETUNE_EVAL_CODE_FILE).write_text(script, encoding="utf-8")
    receipt = {
        "kernel": slug,
        "kind": FINETUNE_EVAL_DECISION,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": FINETUNE_EVAL_CODE_FILE,
        "dataset": {"slug": dataset_slug,
                    "payload": dataset_receipt["payload"],
                    "files": dataset_receipt["files"]},
        "checkpoint_dataset": ckpt_dataset,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_dir_hint": spec.finetune_ckpt_dir,
        "eval_split": split,
        "eval_jsonl": eval_jsonl,
        "laya_package": FINETUNE_LAYA_PACKAGE,
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    atomic_write_json(receipt,
                      stage / f"{FINETUNE_EVAL_DECISION}.receipt.json")
    _log_lane(f"staged kaggle finetune-eval kernel ({spec.gpu}) "
              f"split={split} ckpt={ckpt_dataset or checkpoint_path} "
              f"run_tag={tag} -> {stage}")
    return receipt


def stage_colab_notebook(*, decision_kind: str,
                         run_tag: str | None = None) -> dict[str, Any]:
    """Colab notebook payload in the receipts style (dry-safe).

    Returns a receipt dict; the payload script lands at
    results/laya_lane/colab/<decision_kind>/laya_decision_colab.py.
    Fail-loud preconditions match stage_decision_kernel (epochs, slug).
    """
    spec = _spec()
    if decision_kind not in DECISION_BINDINGS:
        raise ValueError(f"unknown decision kind: {decision_kind!r}; "
                         f"expected {list(DECISION_BINDINGS)}")
    if spec.laya_decision_epochs <= 0:
        raise RuntimeError(
            "config laya.laya_decision_epochs <= 0: the decision lane is "
            "disabled (no payload may stage a session)")
    if not spec.export_dataset_slug and not spec.dataset_slug:
        raise RuntimeError(
            "config laya.export_dataset_slug / dataset_slug both unset; "
            "name the target surface (owner/slug) before staging")
    question = stage_question_schema("colab")
    input_receipt = stage_decision_input("colab",
                                         decision_kind=decision_kind)
    stage = staging_dir() / "colab" / decision_kind
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or spec.run_tag_prefix + decision_tag()
    entry = DECISION_BINDINGS[decision_kind]
    staged_csv = input_receipt["staged"]
    script = _template(NOTEBOOK_SCRIPT, {
        "LAYA_PACKAGE": spec.laya_package,
        "DECISION_KIND": decision_kind,
        "RUN_TAG": tag,
        "DECISION_CSV": Path(staged_csv).name,
        "STATE_COLUMN": entry["state_column"],
    })
    _kernel_script_gate(script)
    _module_scope_gate(script)
    notebook = stage / COLAB_NOTEBOOK_NAME
    notebook.write_text(script, encoding="utf-8")
    receipt = {
        "kernel": COLAB_NOTEBOOK_NAME,
        "kind": decision_kind,
        "gpu": "T4 (single)",
        "run_tag": tag,
        "staged": str(stage),
        "notebook": str(notebook),
        "receipt_style": "results/laya_lane/<kind>/<op>/...",
        "question_schema": question["staged"],
        "decision_input": staged_csv,
        "epochs": spec.laya_decision_epochs,
        "note": "the colab CLI surface is unchanged; delivery contract only",
    }
    atomic_write_json(receipt, stage / f"{decision_kind}.receipt.json")
    _log_lane(f"staged colab notebook payload [{decision_kind}] "
              f"run_tag={tag} -> {notebook}")
    return receipt


def _staged_laya_push_preflight(stage_dir: Path) -> None:
    """Laya push gate (the clone-lane staged_kernel_preflight shape, with
    the laya payload semantics): the ATTACHED-inputs inventory
    (`_runtime_files`) rides the er-laya-requests DATASET, never the git
    checkout — so the inventory is verified against the STAGED
    dataset_payload, and remote_revision_preflight (a core helper, never
    edited here) checks only the publish pin with an empty inventory."""
    import ast
    import json

    from core.runtime_inputs import remote_revision_preflight

    metadata = json.loads((stage_dir / "kernel-metadata.json").read_text())
    script = (stage_dir / metadata['code_file']).read_text()
    values = {}
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                        'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'}:
                    values[target.id] = ast.literal_eval(node.value)
    required = {'REPOSITORY', 'BRANCH', 'REVISION', '_runtime_files'}
    if required - values.keys():
        raise ValueError(
            'Staged kernel lacks runtime preflight inventory; regenerate it')
    payload = stage_dir / "dataset_payload"
    missing = [name for name in values['_runtime_files']
               if not (payload / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Staged dataset payload {payload} is missing attached inputs: "
            + ", ".join(missing) + "; stage the payload first")
    remote_revision_preflight(values['REPOSITORY'], values['BRANCH'], (),
                              revision=values['REVISION'])


# ── the executed ops (fail-loud, --execute gated) ─────────────────────────
def push_kaggle_kernel(stage_dir: Path, *, execute: bool,
                       activate: bool = True) -> dict[str, Any]:
    """`kaggle kernels push` a staged payload, `--execute`-gated.

    Dry run: returns the plan + argv, never spawns the kaggle subprocess.
    Executed: requires the staged metadata file (--activate gate), runs
    the laya push preflight (_staged_laya_push_preflight), pushes, and
    embeds the CLI's own output in the raised RuntimeError on a failing
    returncode.
    """
    argv = [sys.executable, "-m", "kaggle", "kernels", "push",
            "-p", str(stage_dir)]
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "argv": argv, "stage": str(stage_dir)}
    if not execute:
        _log_lane(f"dry-run: would run {' '.join(argv)}")
        return plan
    metadata_file = Path(stage_dir) / "kernel-metadata.json"
    if not metadata_file.is_file():
        raise RuntimeError(
            "--activate gate: no staged kernel at "
            f"{stage_dir} (kernel-metadata.json is missing); stage first "
            "(--what stage-kernel)")
    _staged_laya_push_preflight(Path(stage_dir))
    result = subprocess.run(argv, cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    output = result.stdout or ""
    rc = result.returncode
    plan["returncode"] = rc
    if rc != 0:
        tail = output.strip()[-4000:] or "(kaggle produced no output)"
        raise RuntimeError(
            f"kaggle command failed (rc={rc}): {' '.join(argv)}\n"
            f"--- kaggle output ---\n{tail}")
    plan["pushed"] = True
    _log_lane(f"pushed kernel payload: {' '.join(argv)} rc=0")
    return plan


def collect_kaggle_result(decision_kind: str, slug: str, *,
                          execute: bool = False) -> dict[str, Any]:
    """`kaggle kernels output` for a staged/decided kernel.

    Dry run: returns the plan only. Executed: pulls the payload and
    verifies the receipt (laya_<decision_kind>.receipt.json inside the
    archive) against the locally staged receipt — fail-loud on a sha256
    mismatch. Results install under results/laya_lane/fetch/<decision>/.
    """
    plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                            "decision_kind": decision_kind, "slug": slug}
    if not execute:
        _log_lane(f"dry-run: would fetch kernel output for {slug}")
        return plan
    stage = staging_dir() / "fetch" / decision_kind
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", slug,
         "-p", str(stage)], cwd=TRAIN_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output failed (rc={result.returncode}) for "
            f"{slug}: {result.stdout.strip()[-4000:]}")
    archives = sorted(stage.glob("*.tar.gz")) or sorted(stage.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"kaggle kernels output staged no archive "
                           f"under {stage} (slug {slug})")
    receipt_name = f"laya_{decision_kind}.receipt.json"
    reports: dict[str, Any] = {}
    with tarfile.open(archives[0], "r:*") as tar:
        members = tar.getnames()
        if receipt_name not in members:
            raise RuntimeError(
                f"fetched archive {archives[0].name} carries no "
                f"{receipt_name}; the kernel receipt contract failed")
        payload = json.loads(tar.extractfile(receipt_name)
                             .read().decode())
        # Extract the JSON payloads the kernel wrote (eval_report.json and
        # siblings) into the fetch dir so the report path is reproducible
        # offline: the returned plan carries them keyed by member name.
        for member in members:
            name = Path(member).name
            if not name.endswith(".json") or name == receipt_name:
                continue
            body = tar.extractfile(member).read()
            (stage / name).write_bytes(body)
            try:
                reports[name] = json.loads(body.decode())
            except (ValueError, UnicodeDecodeError):
                continue
    plan.update({"archive": str(archives[0]), "members": members,
                 "receipt": payload, "reports": reports})
    _log_lane(f"fetched kernel output for {slug}: "
              f"archive={archives[0].name} members={len(members)} "
              f"reports={sorted(reports)}")
    return plan


def local_eval_checkpoint(checkpoint_dir: Path, *,
                          eval_data: Path | None = None,
                          out_dir: Path | None = None,
                          split: str | None = None,
                          batch_size: int | None = None,
                          limit: int | None = None) -> dict[str, Any]:
    """Local (CPU) held-out eval of a fetched fine-tuned checkpoint.

    The offline twin of the `finetune-eval` kernel: loads the checkpoint with
    `laya.train.load_checkpoint` on CPU and runs `calibration_records` +
    `evaluate_records` on the corpus split (default `data/laya/test.jsonl`),
    writing the same `eval_report.json` (+ receipt) under `out_dir` (default
    results/laya_lane/local_eval). Requires `laya` + torch installed locally;
    fails loud before any work when they are missing. Never touches the
    network and never trains.
    """
    spec = _spec()
    split = split or spec.finetune_eval_split
    if split not in FINETUNE_EVAL_SPLIT_FILES:
        raise ValueError(
            f"eval split {split!r} is not one of "
            f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "rl_agent_config.json").is_file():
        raise FileNotFoundError(
            f"checkpoint {checkpoint_dir} carries no rl_agent_config.json")
    if eval_data is None:
        eval_data = (TRAIN_ROOT / FINETUNE_CORPUS_DIR
                     / FINETUNE_EVAL_SPLIT_FILES[split])
    eval_data = Path(eval_data)
    if not eval_data.is_file():
        raise FileNotFoundError(f"eval data not found: {eval_data}")
    if out_dir is None:
        out_dir = staging_dir() / "local_eval"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or spec.finetune_eval_batch_size
    try:
        import torch
        from laya import train as laya_train
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "local eval needs the laya package + torch installed on this "
            f"box (pip install {spec.laya_package}): {error}") from error
    device = torch.device("cpu")
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint_dir))
    model = model.to(device).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(eval_data))
    if limit:
        rows = rows[:limit]
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise RuntimeError(
            f"eval data {eval_data} produced no usable items "
            f"(skipped: {skipped!r})")
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=batch_size, parallel=parallel)
    before = laya_train.evaluate_records(records)
    fitted = laya_train.fit_temperature_map(records)
    after = laya_train.evaluate_records(
        records, fitted.get("temperature"),
        fitted.get("temperature_by_options"))
    report = {
        "eval_mode": "held_out",
        "is_held_out": True,
        "device": "cpu",
        "eval_source": eval_data.name,
        "eval_split": split,
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint_dir),
        "before": before,
        "after": after,
        "temperature": fitted.get("temperature"),
        "temperature_by_options": fitted.get("temperature_by_options"),
    }
    atomic_write_json(report, out_dir / FINETUNE_EVAL_REPORT_FILE)
    receipt = {
        "gpu_kind": FINETUNE_EVAL_DECISION,
        "device": "cpu",
        "eval_split": split,
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_data": str(eval_data),
        "eval_data_sha256": sha256_file(eval_data),
        "checkpoint": str(checkpoint_dir),
        "report": str(out_dir / FINETUNE_EVAL_REPORT_FILE),
    }
    atomic_write_json(receipt, out_dir / FINETUNE_EVAL_RECEIPT_FILE)
    _log_lane(f"local cpu eval [{split}] items={len(items)} "
              f"accuracy={after['accuracy']} -> {out_dir}")
    return report


class LayaLane:
    """The lane surface: results/laya_lane/<kind>/<decision>/ receipts.

    ONE class, TWO kinds: kind names the remote surface ("kaggle",
    "colab"); anything else fails loud.
    """

    kind: str

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown laya lane kind: {kind!r}; "
                             f"expected {list(KINDS)}")
        self.kind = kind
        self._spec = _spec()

    def stage(self, decision_kind: str, *,
              input_override: Path | None = None,
              checkpoint_path: Path | None = None) -> dict[str, Any]:
        """Stage the payload (offline, dry-safe)."""
        if self.kind == "kaggle":
            return stage_decision_kernel(
                decision_kind=decision_kind, input_override=input_override,
                checkpoint_path=checkpoint_path)
        return stage_colab_notebook(decision_kind=decision_kind)

    def push(self, stage_dir: Path, *, execute: bool = False,
             activate: bool = True) -> dict[str, Any]:
        """`--execute` gated push (kaggle kind only)."""
        if self.kind != "kaggle":
            raise RuntimeError("push is a kaggle-lane operation")
        return push_kaggle_kernel(stage_dir, execute=execute,
                                  activate=activate)

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Dispatch the parsed main() args through this lane."""
        if args.decision not in DECISION_BINDINGS:
            raise ValueError(f"unknown decision kind: {args.decision!r}")
        if args.kind != self.kind:
            raise ValueError(f"--kind {args.kind!r} does not match the "
                             f"lane kind {self.kind!r}")
        return self.stage(args.decision, input_override=args.decision_input)


# ── main ───────────────────────────────────────────────────────────────────
def _spawn_stream_follower(slug: str) -> None:
    """Follow a pushed kernel's live session log into the laya lane transcript.

    The laya lane otherwise has no visibility into the remote session (it never
    opens a stream), so the training tqdm never reaches ``logs/laya/lane.log``.
    This spawns the kaggle lane's SSE follower against the pushed slug so the
    live output lands there. Detached (setsid) so a wrapper/shell death cannot
    orphan or kill the follower.
    """
    from core.common import TRAIN_ROOT

    log = TRAIN_ROOT / "logs/laya/lane.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from pathlib import Path\n"
        "from cli.kaggle_lane import stream_kernel_logs\n"
        f"stream_kernel_logs({slug!r}, log_path=Path({str(log)!r}))\n"
    )
    with log.open("ab") as handle:
        subprocess.Popen(
            [sys.executable, "-c", code], cwd=TRAIN_ROOT,
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONPATH": str(TRAIN_ROOT / "src")},
            start_new_session=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, default="kaggle")
    parser.add_argument("--decision", choices=GPU_KINDS, default="attribute",
                        help="which typed decision run to stage "
                             "(default: attribute)")
    parser.add_argument("--execute", action="store_true",
                        help="make the remote call (kaggle kernels push); "
                             "the default is an offline dry-run")
    parser.add_argument("--decision-input", type=Path, default=None,
                        help="alternate decision source CSV on this box "
                             "(forwarded as the staging override; "
                             "kaggle staging path)")
    parser.add_argument("--fetch", action="store_true",
                        help="fetch a pushed kernel's output (kaggle "
                             "kernels output) instead of staging; combine "
                             "with --decision + --slug and --execute")
    parser.add_argument("--slug", default=None,
                        help="the pushed kernel slug (owner/slug) for "
                             "--fetch")
    parser.add_argument("--local-eval", action="store_true",
                        help="run the CPU held-out eval of a fetched "
                             "fine-tuned checkpoint instead of staging")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="the fine-tuned checkpoint dir for "
                             "--local-eval (must carry rl_agent_config.json)")
    parser.add_argument("--eval-data", type=Path, default=None,
                        help="the eval JSONL for --local-eval (default: the "
                             "config split under data/laya)")
    parser.add_argument("--eval-out", type=Path, default=None,
                        help="output dir for --local-eval "
                             "(default: results/laya_lane/local_eval)")
    parser.add_argument("--eval-split", choices=tuple(FINETUNE_EVAL_SPLIT_FILES),
                        default=None,
                        help="the corpus split for --local-eval "
                             "(default: config laya.finetune_eval_split)")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="cap the number of eval rows for --local-eval")
    args = parser.parse_args()

    if args.local_eval:
        # Local CPU eval path: no staging, no network, no kernel.
        if args.checkpoint is None:
            parser.error("--local-eval requires --checkpoint PATH")
        report = local_eval_checkpoint(
            args.checkpoint, eval_data=args.eval_data, out_dir=args.eval_out,
            split=args.eval_split, limit=args.eval_limit)
        print(json.dumps(report, indent=2), flush=True)
        return

    if args.fetch:
        # Fetch path: kaggle kernels output for a pushed kernel; the eval
        # report JSONs land under results/laya_lane/fetch/<decision>/.
        if not args.slug:
            parser.error("--fetch requires --slug owner/slug")
        plan = collect_kaggle_result(args.decision, args.slug,
                                     execute=args.execute)
        print(json.dumps(plan, indent=2), flush=True)
        return

    lane = LayaLane(args.kind)
    receipt = lane.stage(args.decision, input_override=args.decision_input,
                         checkpoint_path=args.checkpoint)
    print(_stamp(), f"[laya-lane] staged {args.kind}/{args.decision} payload: "
          f"{json.dumps(receipt, indent=2)}", flush=True)
    if args.execute and args.kind == "kaggle":
        stage_dir = Path(receipt["staged"])
        # the inputs travel as the dataset BEFORE the push (the kernel
        # metadata attaches spec.dataset_slug; a missing/drifting dataset
        # would FileNotFoundError resolve_input once boot passes)
        dataset_plan = publish_laya_dataset(args.decision,
                                            run_tag=receipt["run_tag"],
                                            execute=True)
        print(json.dumps(dataset_plan, indent=2), flush=True)
        push_plan = lane.push(stage_dir, execute=True)
        print(json.dumps(push_plan, indent=2), flush=True)
        # Follow the pushed kernel's live session log into logs/laya/lane.log
        # (the lane otherwise has no remote visibility and never shows a tqdm).
        _spawn_stream_follower(
            json.loads((stage_dir / "kernel-metadata.json").read_text())["id"])
    elif args.execute:
        _log_lane("colab payloads are a delivery contract only; nothing "
                  "to --execute")
    else:
        _log_lane("dry-run only; pass --execute to touch the remote surface")


if __name__ == "__main__":
    main()
