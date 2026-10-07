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
GPU_KINDS = ("attribute", "identity", "laya-cli-eval")
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
    name = torch.cuda.get_device_name(0)
    log("device pinned: " + name + " (single GPU, never a second one)")
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
    slug = _spec().dataset_slug
    plan["slug"] = slug
    if not slug:
        raise RuntimeError(
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
                          input_override: Path | None = None) -> dict[str, Any]:
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
    with tarfile.open(archives[0], "r:*") as tar:
        members = tar.getnames()
        if receipt_name not in members:
            raise RuntimeError(
                f"fetched archive {archives[0].name} carries no "
                f"{receipt_name}; the kernel receipt contract failed")
        payload = json.loads(tar.extractfile(receipt_name)
                             .read().decode())
    plan.update({"archive": str(archives[0]), "members": members,
                 "receipt": payload})
    _log_lane(f"fetched kernel output for {slug}: "
              f"archive={archives[0].name} members={len(members)}")
    return plan


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
              input_override: Path | None = None) -> dict[str, Any]:
        """Stage the payload (offline, dry-safe)."""
        if self.kind == "kaggle":
            return stage_decision_kernel(decision_kind=decision_kind,
                                         input_override=input_override)
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
    args = parser.parse_args()
    lane = LayaLane(args.kind)
    receipt = lane.stage(args.decision, input_override=args.decision_input)
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
    elif args.execute:
        _log_lane("colab payloads are a delivery contract only; nothing "
                  "to --execute")
    else:
        _log_lane("dry-run only; pass --execute to touch the remote surface")


if __name__ == "__main__":
    main()
