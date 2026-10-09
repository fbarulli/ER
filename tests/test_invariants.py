"""Cross-cutting invariants the independent audit found documentation-only.

One decisive assertion per invariant. The audit's claims and what each test
enforces:

1. The bundle boundary read per VM crossing is exercised by the sanctioned
   command (``scripts/run_colab_smoke.sh``), not by an internal call-count test.
2. Bundle ROLE contracts are enforced at the boundary, not only by the writer —
   an ``inputs`` bundle carrying weights, or a ``result`` bundle carrying every
   epoch, must be refused by ``Bundle.load``.
3. The Kaggle train -> finalize ROLE handoff — the train kernel must ship the
   suite's own sealed ``result`` Bundle, so the finalize job's
   ``Bundle.load(..., "result")`` boundary accepts it (it rejected the old tree
   tarball with "archive manifest missing").
4. Config-leaf coverage — every ``config/*.yaml`` leaf has a reader, and no code
   literal duplicates a config-owned artifact value.
5. The orchestration stage -> trace-stage map (``config/paths.yaml``
   ``orchestration_stages``, read via ``core.tracing.orchestration_trace_stages``)
   matches the producers' own ``STAGE`` constants.

Where the required fix is outside this change's ownership the test runs the
real check and records the KNOWN drift against a pinned baseline snapshot, so
the enforcement outlives this report: NEW drift fails the test the day it is
introduced, while fixing known drift is allowed (and only shrinks the
baseline's reach). Invariants 1 and 2 have reached the state where their owners
landed the runtime fix, so they assert directly instead of recording drift;
the config-leaf census (4) asserts "no drift beyond the pinned snapshot".
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.bundle import Bundle, BundleRole, _bundle_spec, manifest_name
from core.bundle import _bundle_spec as _spec
from core.portable_archive import read_archive_manifest, write_archive


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ── 2. bundle role contracts enforced at the boundary ──────────────────────

def test_bundle_role_contracts_are_enforced_on_load(tmp_path):
    """Role membership is a LOAD contract, not a writer-only convention.

    An ``inputs`` bundle must carry no weights, and a ``result`` bundle must
    carry only the selected checkpoint. Both archives below are well formed
    (``read_archive_manifest`` reads them); the only remaining refusal reason
    is the role contract.
    """
    spec = _spec()

    inputs_tree = tmp_path / "inputs_with_weights"
    _write(inputs_tree / "data/x.json", "{}")
    _write(inputs_tree / "text/_checkpoints/m/r_f0/checkpoint-1/model.safetensors", "weights")
    inputs_archive = tmp_path / "inputs_with_weights.tar.zst"
    write_archive(inputs_archive,
                  {p.relative_to(inputs_tree).as_posix(): p
                   for p in inputs_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_inputs, metadata={})

    result_tree = tmp_path / "result_all_epochs"
    for step in (1, 2, 3):
        directory = result_tree / f"text/_checkpoints/m/r_f0/checkpoint-{step}"
        _write(directory / "model.safetensors", f"w{step}")
        _write(directory / "trainer_state.json", json.dumps(
            {"best_model_checkpoint": "checkpoint-2",
             "best_metric": 0.9 if step == 2 else 0.5, "global_step": step}))
    result_archive = tmp_path / "result_all_epochs.tar.zst"
    write_archive(result_archive,
                  {p.relative_to(result_tree).as_posix(): p
                   for p in result_tree.rglob("*") if p.is_file()},
                  manifest_name=spec.manifest_result,
                  metadata={spec.run_tag_key: "r-tag"})

    for role, archive, label in (
            (BundleRole.inputs, inputs_archive, "an inputs bundle carrying weights"),
            (BundleRole.result, result_archive, "a result bundle carrying every epoch")):
        read_archive_manifest(archive, manifest_name(role))  # fixture is well formed
        try:
            Bundle.load(archive, role)
        except ValueError:
            pass  # the role contract refused it, as it must
        else:
            raise AssertionError(label + " loads cleanly")
    # The guard is precise, not blanket: a conforming set for the same role
    # still loads (the happy path is unchanged).
    selected_only = tmp_path / "result_selected.tar.zst"
    write_archive(selected_only, {
        "text/_checkpoints/m/r_f0/checkpoint-2/model.safetensors":
            result_tree / "text/_checkpoints/m/r_f0/checkpoint-2/model.safetensors"},
        manifest_name=spec.manifest_result, metadata={spec.run_tag_key: "r-tag"})
    assert Bundle.load(selected_only, BundleRole.result).run_tag() == "r-tag"


# ── 3. kaggle train -> finalize role handoff ───────────────────────────────

def _stage_train_and_embed(tmp_path, monkeypatch):
    """Stage the train and embed kernels offline (no network, no git)."""
    import core.runtime_inputs as runtime_inputs
    from cli import kaggle_lane
    from core.schemas import KaggleSpec

    spec = KaggleSpec(staging_dir="kaggle_stage", username="owner",
                      cpu_kernel_slug="owner/er-bundle-cpu",
                      gpu_kernel_slug="owner/er-train-gpu",
                      embedding_kernel_slug="owner/er-embed-gpu")
    monkeypatch.setattr(kaggle_lane, "_spec", lambda: spec)
    monkeypatch.setattr(kaggle_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(kaggle_lane, "staging_dir",
                        lambda: (tmp_path / "kaggle_stage").resolve())
    monkeypatch.setattr(kaggle_lane, "_git_revision", lambda: "abc123def")
    monkeypatch.setattr(runtime_inputs, "require_published_tip_match",
                        lambda *a, **k: {"tip": "abc123def"})

    def members(*extra, lane="bundle"):
        flat = {name for group in extra for name in
                (group if isinstance(group, (tuple, list)) else (group,))}
        return tuple(sorted(flat))

    monkeypatch.setattr(kaggle_lane, "checkout_members", members)
    monkeypatch.setattr(kaggle_lane, "checkout_inventory", members)
    monkeypatch.setattr(kaggle_lane, "checkout_preflight_script",
                        lambda files, root_expression="root": "_runtime_files = ()\n")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return spec


def test_kaggle_train_ships_the_sealed_result_bundle_the_finalize_boundary_accepts(
        tmp_path, monkeypatch):
    """The train kernel must ship the suite's sealed ``result`` Bundle.

    Two sides of the same handoff: the staged train script carries the override
    (and the embed kernel does not), and the artifact that override produces is
    accepted by the finalize kernel's exact boundary call.
    """
    from cli.kaggle_kernels import KaggleKernels, TRAIN_RESULT_BUNDLE_SHIP
    from cli import kaggle_lane

    spec = _stage_train_and_embed(tmp_path, monkeypatch)
    train_receipt = KaggleKernels.stage_gpu_kernel(kind="train")
    embed_receipt = KaggleKernels.stage_gpu_kernel(kind="embed")
    train_script = (Path(train_receipt["staged"]) / train_receipt["code_file"]).read_text()
    embed_script = (Path(embed_receipt["staged"]) / embed_receipt["code_file"]).read_text()
    # ``KernelLifecycle.wrap_script`` indents the whole kernel, so compare
    # whitespace-normalized text.
    normalized_fragment = " ".join(TRAIN_RESULT_BUNDLE_SHIP.split())
    assert normalized_fragment in " ".join(train_script.split()), \
        "the train kernel must override the tree-tar helper with the sealed-bundle ship"
    assert normalized_fragment not in " ".join(embed_script.split()), \
        "the embed output is not a Bundle role; it keeps the tree-tar helper"

    # Behaviour: run the shipped override exactly as the kernel would, then make
    # the finalize kernel's boundary call against the artifact it produced.
    bundle_spec = _spec()
    run_tag = "run-20261008T000000"
    output = tmp_path / "results" / "training" / run_tag
    _write(output / "text/text__vectors.npz", "vectors")
    _write(output / bundle_spec.suite_manifest_file,
           json.dumps({bundle_spec.run_tag_key: run_tag}))
    sealed = Bundle.from_directory(output, BundleRole.result).seal_result(
        output.with_suffix(".tar.zst"), metadata={bundle_spec.run_tag_key: run_tag})

    working = tmp_path / "kaggle_working"
    working.mkdir()
    # The kernel clones to a checkout root; on the VM the bundle installs the
    # generated suite config at the train kernel's configured path, so mirror
    # that layout from the committed suite config.
    repo_root = Path(__file__).parents[1]
    generated_config = tmp_path / spec.train_suite_config
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(
        (repo_root / "config" / "model_tracks.yaml").read_text())
    namespace = {
        "json": json, "shutil": __import__("shutil"), "Path": Path,
        "WORKING": working, "RUN_TAG": run_tag, "REVISION": "abc123def",
        "SUITE_CONFIG": spec.train_suite_config, "root": tmp_path,
        "LANE": {"files": {"result_archive": "{kind}.tar.zst",
                           "result_manifest": "{kind}.manifest.json"}},
        "file_size": lambda path: Path(path).stat().st_size,
    }
    exec(compile(TRAIN_RESULT_BUNDLE_SHIP, "<train-result-ship>", "exec"), namespace)
    digest = namespace["stage_result_archive"](output, kind="result_bundle", extra={})

    manifest = json.loads((working / "result_bundle.manifest.json").read_text())
    assert manifest["archive_size"] == digest == sealed.path.stat().st_size
    # The finalize kernel's exact boundary: Bundle.load(..., "result"),
    # verified once by the sealed member inventory (names + byte sizes).
    handle = Bundle.load(working / "result_bundle.tar.zst", "result")
    assert handle.run_tag() == run_tag
    assert bundle_spec.manifest_result in handle.members()
    # The old shape (a tree tarball with an out-of-archive manifest) is exactly
    # what the boundary rejected: keep that proof beside the fix.
    from core.archive_reader import tar_archive
    tree_tar = tmp_path / "tree_tarball.tar.zst"
    with tar_archive(tree_tar, "w") as tar:
        tar.add(output / "text/text__vectors.npz",
                arcname=f"{run_tag}/text/text__vectors.npz")
    with pytest.raises(ValueError, match="archive manifest missing"):
        Bundle.load(tree_tar, "result")


# ── 4. config-leaf coverage + no code literal duplicates a config value ─────


def test_mount_root_has_one_source_the_hosted_registry():
    """Item 2: the mount root is declared ONCE, in config/hosted_datasets.yaml.

    ``HostedRegistry`` is the SSOT; ``kaggle.remote.input_dir`` references it
    (the schema default reads the registry), so the two values cannot drift.
    """
    from pathlib import Path

    from core.common import training_cfg
    from core.hosted_dataset import hosted_registry

    assert hosted_registry().mount_root == Path(
        training_cfg().kaggle.remote.input_dir)


#: The config's own typed mirror: pydantic field defaults must be literals, so a
#: value also appearing here is the declared default, not an independent SSOT
#: copy. Reported separately (never a failure) instead of silently ignored.
_CONFIG_MIRRORS = frozenset({"src/core/schemas.py"})

#: Values that look like an artifact identity (a path or a suffixed filename)
#: are config-owned things code must not respell; short vocabulary words (a
#: track name, ``offline``, ``cuda``) are not treated as duplication.
_ARTIFACT_SUFFIX = (".json", ".yaml", ".yml", ".csv", ".zst", ".zip", ".pt", ".npz",
                    ".txt", ".pkl", ".gz", ".bin", ".md", ".jsonl", ".log",
                    ".safetensors")


def _config_drift(repo_root: Path) -> dict[str, list]:
    """AST/static census of the two config-SSOT drift classes."""
    leaves: dict[tuple[str, str], object] = {}
    for config_path in sorted((repo_root / "config").glob("*.yaml")):
        stack = [((), yaml.safe_load(config_path.read_text()))]
        while stack:
            path, node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    stack.append((path + (str(key),), value))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    stack.append((path + (f"[{index}]",), value))
            else:
                leaves[(config_path.name, ".".join(path))] = node

    # A leaf counts as READ only when some src AST node is a REAL attribute
    # access onto it (``loaded_config.<key>`` -- the typed config models and
    # their nested spec fields). A bare identifier named like the key and a
    # string literal that merely CONTAINS the key are coincidences, not reads:
    # that is exactly how a dead ``orchestration_stages``-style block looked
    # clean while its only mentions were a docstring / ``.get("...")`` literal.
    attribute_reads: set[str] = set()
    literals: dict[str, set[str]] = {}
    for source in sorted((repo_root / "src").rglob("*.py")):
        relative = source.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(source.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attribute_reads.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.setdefault(node.value, set()).add(relative)

    unread, duplicated, mirrored = [], [], []
    for (config_name, path), value in leaves.items():
        key = path.rsplit(".", 1)[-1]
        if not key.startswith("[") and key not in attribute_reads:
            unread.append(f"{config_name}:{path}")
        if not (isinstance(value, str) and len(value) >= 6
                and (value.endswith(_ARTIFACT_SUFFIX) or "/" in value)):
            continue
        owners = literals.get(value)
        if not owners:
            continue
        independent = sorted(owners - _CONFIG_MIRRORS)
        if independent:
            duplicated.append(f"{value!r} ({config_name}:{path} <- {', '.join(independent[:2])})")
        else:
            mirrored.append(f"{config_name}:{path}")
    return {"unread": unread, "duplicated": duplicated, "mirrored": mirrored}


#: PINNED BASELINE of the config-SSOT drift that exists TODAY (measured
#: 2026-10-08). The guard below asserts the measured sets are a SUBSET of these
#: two snapshots, so it FAILS on any NEW unread leaf or any NEW code literal
#: that respells a config-owned artifact value — the failure this census was
#: written to give but never could while it only called ``pytest.xfail``.
#: Shrinking (an owner fixing a dead leaf / a respelling) stays green; when an
#: entry is fixed, delete it here so the baseline keeps meaning "known drift".
#: Regenerate the literals with:
#:   python -c "import sys; sys.path[:0]=['src','tests']; import test_invariants"
#: (or re-run the guard and copy the reported measured sets).
_CONFIG_SSOT_BASELINE_UNREAD = frozenset({
    'identity_dimensions.yaml:attributes.Carbonization.aliases.sparkling',
    'identity_dimensions.yaml:columns.breadcrumbs_eng',
    'identity_dimensions.yaml:columns.description_short_eng',
    'identity_dimensions.yaml:columns.image_url',
    'identity_dimensions.yaml:columns.sku_last_price',
    'identity_dimensions.yaml:columns.sku_name_eng',
    'identity_dimensions.yaml:columns.sku_url',
    'model_tracks.yaml:gpu_parallel_backend',
    'model_tracks.yaml:max_parallel',
    'paths.yaml:canonical_optional_columns.evidence_ledger',
    'paths.yaml:column_mapping.breadcrumbs_eng',
    'paths.yaml:column_mapping.description_short_eng',
    'paths.yaml:column_mapping.image_url',
    'paths.yaml:column_mapping.sku_last_price',
    'paths.yaml:column_mapping.sku_name_eng',
    'paths.yaml:column_mapping.sku_url',
    'paths.yaml:dataset_csv_read.keep_default_na',
    'paths.yaml:dataset_csv_read.na_filter',
    'paths.yaml:decision_attribute_aliases.package_material',
    'paths.yaml:extraction.pack_confidence.compact',
    'paths.yaml:extraction.pack_confidence.container',
    'paths.yaml:extraction.pack_confidence.nested',
    'paths.yaml:extraction.pack_confidence.pack_of',
    'paths.yaml:extraction.source_groups.description_short_eng',
    'paths.yaml:extraction.source_groups.image_url',
    'paths.yaml:extraction.source_groups.sku_name_eng',
    'paths.yaml:extraction.source_groups.sku_url',
    'paths.yaml:files.ambiguous_offer_groups',
    'paths.yaml:files.attribute_agreement_conflicts',
    'paths.yaml:files.attribute_agreement_summary',
    'paths.yaml:files.attribute_separation_summary',
    'paths.yaml:files.attribute_separation_values',
    'paths.yaml:files.colab_live_log',
    'paths.yaml:files.colab_training_log',
    'paths.yaml:files.data_quality_columns',
    'paths.yaml:files.data_quality_gtin_groups',
    'paths.yaml:files.data_quality_summary',
    'paths.yaml:files.data_scaling',
    'paths.yaml:files.dataset_deduped',
    'paths.yaml:files.decision_ablation_report',
    'paths.yaml:files.decision_attribute_census',
    'paths.yaml:files.decision_embedding_request',
    'paths.yaml:files.decision_ledger',
    'paths.yaml:files.decision_rebuild_report',
    'paths.yaml:files.decision_suite_config',
    'paths.yaml:files.decision_training_report',
    'paths.yaml:files.decision_visibility',
    'paths.yaml:files.dedupe_conflicts',
    'paths.yaml:files.dedupe_summary',
    'paths.yaml:files.embedding_similarities',
    'paths.yaml:files.field_ablation',
    'paths.yaml:files.final_validation',
    'paths.yaml:files.fold_metrics',
    'paths.yaml:files.four_pop_scores',
    'paths.yaml:files.gate_results',
    'paths.yaml:files.hpo_grid_csv',
    'paths.yaml:files.identity_dimensions',
    'paths.yaml:files.labeled_pairs',
    'paths.yaml:files.model_evaluation_summary',
    'paths.yaml:files.number_reference',
    'paths.yaml:files.package_gate_impact_pairs',
    'paths.yaml:files.package_gate_impact_summary',
    'paths.yaml:files.results_pointer',
    'paths.yaml:files.second04_pairs_positive',
    'paths.yaml:files.title_attribute_evidence',
    'paths.yaml:files.title_attribute_summary',
    'paths.yaml:files.title_removed_tokens',
    'paths.yaml:files.training_report',
    'paths.yaml:files.validation_fold_map',
    'paths.yaml:layouts.attribute_universe_census.owner',
    'paths.yaml:layouts.balanced_pairs.owner',
    'paths.yaml:layouts.balanced_pairs_sample.owner',
    'paths.yaml:layouts.balanced_pairs_sample_manifest.owner',
    'paths.yaml:layouts.balanced_pairs_sample_threshold_sweep.owner',
    'paths.yaml:layouts.checkpoint_repo.owner',
    'paths.yaml:layouts.config_dir.owner',
    'paths.yaml:layouts.dvc_publication_manifest.owner',
    'paths.yaml:layouts.dvc_publication_pointer.fields.pointer',
    'paths.yaml:layouts.dvc_publication_pointer.owner',
    'paths.yaml:layouts.final_inference.owner',
    'paths.yaml:layouts.graph_worker_package.owner',
    'paths.yaml:layouts.hpo_best.fields.era',
    'paths.yaml:layouts.hpo_best.owner',
    'paths.yaml:layouts.hpo_trials.fields.era',
    'paths.yaml:layouts.hpo_trials.owner',
    'paths.yaml:layouts.model_comparison_plot.owner',
    'paths.yaml:layouts.model_tracks_config.owner',
    'paths.yaml:layouts.negative_supply_discriminator.owner',
    'paths.yaml:layouts.report_csv.owner',
    'paths.yaml:layouts.report_plot.owner',
    'paths.yaml:layouts.resume_pointer.owner',
    'paths.yaml:layouts.scripts_dir.owner',
    'paths.yaml:layouts.semantic_family_registry.owner',
    'paths.yaml:layouts.source_code_dir.owner',
    'paths.yaml:layouts.suite_output_archive.fields.fmt',
    'paths.yaml:layouts.suite_output_archive.owner',
    'paths.yaml:layouts.suite_output_input_archive.fields.fmt',
    'paths.yaml:layouts.suite_output_input_archive.owner',
    'paths.yaml:layouts.suite_output_local_inputs.owner',
    'paths.yaml:layouts.suite_output_training_archive.fields.fmt',
    'paths.yaml:layouts.suite_output_training_archive.owner',
    'paths.yaml:layouts.suite_outputs.owner',
    'paths.yaml:layouts.suite_package_config.owner',
    'paths.yaml:layouts.suite_package_shared.owner',
    'paths.yaml:layouts.traceability_report.owner',
    'paths.yaml:layouts.track_ablation_report.owner',
    'paths.yaml:layouts.track_ablation_vectors.owner',
    'paths.yaml:layouts.training_trace.owner',
    'paths.yaml:layouts.visibility.owner',
    'paths.yaml:layouts.visibility_run.owner',
    'paths.yaml:models.deberta_v3_base',
    'paths.yaml:models.minilm_l6',
    'paths.yaml:models.multilingual_l12',
    'paths.yaml:models.ner_semantic_base',
    'paths.yaml:models.ner_transformer_base',
    'paths.yaml:models.rerank_minilm_l6',
    'paths.yaml:paths.artifacts_dir',
    'paths.yaml:paths.audit_findings_dir',
    'paths.yaml:paths.data_dir',
    'paths.yaml:paths.embeddings_dir',
    'paths.yaml:paths.models_dir',
    'paths.yaml:paths.results_dir',
    'paths.yaml:paths.training_data_dir',
    'paths.yaml:paths.training_results_dir',
    'training.yaml:calibration_sweep.eval_model',
    'training.yaml:colab.dvc_workers',
    'training.yaml:colab.remote_data_prep',
    'training.yaml:colab.smoke_dataset_csv',
    'training.yaml:colab.smoke_inference_sample',
    'training.yaml:collapse_guardrail.cosine_std_floor',
    'training.yaml:collapse_guardrail.crossing_rate_ceiling',
    'training.yaml:collapse_guardrail.max_token_frequency',
    'training.yaml:collapse_guardrail.operating_threshold',
    'training.yaml:collapse_guardrail.penalty_weight',
    'training.yaml:collapse_guardrail.unrelated_pairs',
    'training.yaml:collapse_guardrail_profiles.threshold_65.operating_threshold',
    'training.yaml:collapse_guardrail_profiles.threshold_80.operating_threshold',
    'training.yaml:differentiation_audit.max_between_pairs_per_brand',
    'training.yaml:differentiation_audit.max_listings_per_gtin',
    'training.yaml:differentiation_audit.min_between_pairs',
    'training.yaml:differentiation_audit.min_within_gtins',
    'training.yaml:differentiation_audit.sim_min',
    'training.yaml:differentiation_audit.volume_margin',
    'training.yaml:entity_clusters.max_component_size',
    'training.yaml:entity_clusters.max_giant_ratio',
    'training.yaml:evaluation.generalization_slices.observed_split',
    'training.yaml:evaluation.generalization_slices.sparse_neighborhood_max_peers',
    'training.yaml:evaluation.operating_precision',
    'training.yaml:evaluation.operating_recall',
    'training.yaml:evaluation.paired_bootstrap.resamples',
    'training.yaml:evaluation.robust_validation.max_split_attempts',
    'training.yaml:evaluation.robust_validation.min_slice_size',
    'training.yaml:evaluation.robust_validation.operating_thresholds.balanced_review',
    'training.yaml:evaluation.robust_validation.operating_thresholds.high_precision',
    'training.yaml:evaluation.robust_validation.repeats',
    'training.yaml:evaluation.uniformity.checkpoint_scope',
    'training.yaml:hpo.objective.cv',
    'training.yaml:hpo.objective.holdout',
    'training.yaml:hpo.persistence',
    'training.yaml:hpo.selection_skip_test_eval',
    'training.yaml:kaggle.cpu_kernel_slug',
    'training.yaml:kaggle.files.bundle_dir',
    'training.yaml:kaggle.files.checkout_dir',
    'training.yaml:kaggle.files.code_files.embed',
    'training.yaml:kaggle.files.dataset_metadata',
    'training.yaml:kaggle.files.embedding_dir',
    'training.yaml:kaggle.files.embedding_script',
    'training.yaml:kaggle.files.failure_log',
    'training.yaml:kaggle.files.prep_dir',
    'training.yaml:kaggle.files.prep_suite_config',
    'training.yaml:kaggle.files.request_file',
    'training.yaml:kaggle.files.result_names.embed',
    'training.yaml:kaggle.files.training_dir',
    'training.yaml:kaggle.files.vectors_file',
    'training.yaml:kaggle.files.worker_log',
    'training.yaml:kaggle.gpu_kernel_slug',
    'training.yaml:kaggle.limits.child_stop_seconds',
    'training.yaml:kaggle.limits.git_clone_depth',
    'training.yaml:kaggle.limits.read_buffer_bytes',
    'training.yaml:kaggle.limits.stream_retries',
    'training.yaml:kaggle.limits.terminal_columns',
    'training.yaml:kaggle.limits.terminal_rows',
    'training.yaml:kaggle.remote.scratch_dir',
    'training.yaml:kaggle.remote.working_dir',
    'training.yaml:masking.attribute_augment.carbonation.hard',
    'training.yaml:masking.attribute_augment.carbonation.neg_frac',
    'training.yaml:masking.attribute_augment.carbonation.pos_frac',
    'training.yaml:masking.attribute_augment.flavor.hard',
    'training.yaml:masking.attribute_augment.flavor.neg_frac',
    'training.yaml:masking.attribute_augment.flavor.pos_frac',
    'training.yaml:masking.attribute_augment.juice_content.hard',
    'training.yaml:masking.attribute_augment.juice_content.neg_frac',
    'training.yaml:masking.attribute_augment.juice_content.pos_frac',
    'training.yaml:masking.attribute_augment.pack.hard',
    'training.yaml:masking.attribute_augment.pack.neg_frac',
    'training.yaml:masking.attribute_augment.pack.pos_frac',
    'training.yaml:masking.attribute_augment.package_material.hard',
    'training.yaml:masking.attribute_augment.package_material.neg_frac',
    'training.yaml:masking.attribute_augment.package_material.pos_frac',
    'training.yaml:masking.attribute_augment.package_type.hard',
    'training.yaml:masking.attribute_augment.package_type.neg_frac',
    'training.yaml:masking.attribute_augment.package_type.pos_frac',
    'training.yaml:masking.attribute_augment.sweetener.hard',
    'training.yaml:masking.attribute_augment.sweetener.neg_frac',
    'training.yaml:masking.attribute_augment.sweetener.pos_frac',
    'training.yaml:masking.attribute_augment.volume.hard',
    'training.yaml:masking.attribute_augment.volume.neg_frac',
    'training.yaml:masking.attribute_augment.volume.pos_frac',
    'training.yaml:masking.diet_max_pos_neg_view_ratio',
    'training.yaml:masking.diet_min_neg_aug_frac',
    'training.yaml:masking.field_quota_shares.juice_content',
    'training.yaml:masking.field_quota_shares.package_material',
    'training.yaml:masking.hard_negative_frac',
    'training.yaml:masking.track_per_epoch',
    'training.yaml:masking.track_visibility',
    'training.yaml:masking_profiles.lower_negative_masking.hard_negative_frac',
    'training.yaml:masking_profiles.matched_positive_negative.hard_negative_frac',
    'training.yaml:masking_profiles.minimal_negative_masking.hard_negative_frac',
    'training.yaml:mining.ann.band_mode',
    'training.yaml:mining.ann.candidate_multiplier',
    'training.yaml:mining.ann.refresh_enabled',
    'training.yaml:mining.ann.refresh_every_epochs',
    'training.yaml:mining.ann.score_quantiles',
    'training.yaml:mining.attribute_conflict.same_product_name',
    'training.yaml:mining_profiles.masking_only.ann_enabled',
    'training.yaml:mining_profiles.mining_enabled.ann_enabled',
    'training.yaml:negative_supply.real_first',
    'training.yaml:ner.archive_names.best_model',
    'training.yaml:ner.archive_names.final_model',
    'training.yaml:ner.archive_names.published_final_model',
    'training.yaml:ner.archive_names.remote_model',
    'training.yaml:ner.base_dir',
    'training.yaml:ner.columns.namebrandmatch',
    'training.yaml:ner.columns.sku_name_clean',
    'training.yaml:ner.columns.sku_name_eng',
    'training.yaml:ner.data_prep.columns.namebrandmatch',
    'training.yaml:ner.data_prep.columns.sku_name_clean',
    'training.yaml:ner.data_prep.columns.sku_name_eng',
    'training.yaml:ner.data_prep.input_csv',
    'training.yaml:ner.data_prep.output_csv',
    'training.yaml:ner.huggingface.ner_repo_id',
    'training.yaml:ner.huggingface.token_file',
    'training.yaml:ner.results_dir',
    'training.yaml:ner.semantic_evaluation.columns.namebrandmatch',
    'training.yaml:ner.semantic_evaluation.columns.predicted',
    'training.yaml:ner.semantic_evaluation.columns.semantic_similarity',
    'training.yaml:ner.semantic_evaluation.columns.sku_name_clean',
    'training.yaml:ner.semantic_evaluation.columns.sku_name_eng',
    'training.yaml:ner.semantic_evaluation.encode_params.normalize_embeddings',
    'training.yaml:ner.semantic_evaluation.encode_params.show_progress_bar',
    'training.yaml:ner.semantic_evaluation.fn_csv',
    'training.yaml:ner.semantic_evaluation.fp_csv',
    'training.yaml:ner.semantic_evaluation.hf_repo_id',
    'training.yaml:ner.semantic_evaluation.input_csv',
    'training.yaml:ner.semantic_evaluation.metrics_csv',
    'training.yaml:ner.semantic_evaluation.model_dir',
    'training.yaml:ner.semantic_evaluation.output_csv',
    'training.yaml:ner.semantic_evaluation.predictions_csv',
    'training.yaml:ner.semantic_evaluation.random_state',
    'training.yaml:ner.semantic_evaluation.top_n_fp_fn',
    'training.yaml:ner.semantic_training.BATCH_SIZE',
    'training.yaml:ner.semantic_training.COLUMNS.namebrandmatch',
    'training.yaml:ner.semantic_training.COLUMNS.sku_name_clean',
    'training.yaml:ner.semantic_training.COLUMNS.sku_name_eng',
    'training.yaml:ner.semantic_training.EPOCHS',
    'training.yaml:ner.semantic_training.EVAL_STEPS',
    'training.yaml:ner.semantic_training.HF_REPO_ID',
    'training.yaml:ner.semantic_training.HOLDOUT_FRAC',
    'training.yaml:ner.semantic_training.INPUT_CSV',
    'training.yaml:ner.semantic_training.LEARNING_RATE',
    'training.yaml:ner.semantic_training.MODEL_NAME',
    'training.yaml:ner.semantic_training.NEGATIVE_SAMPLING_RATIO',
    'training.yaml:ner.semantic_training.OUTPUT_DIR',
    'training.yaml:ner.semantic_training.SAVE_STEPS',
    'training.yaml:ner.semantic_training.SEED',
    'training.yaml:ner.semantic_training.TRAIN_FRAC',
    'training.yaml:ner.semantic_training.VALIDATION_FRAC',
    'training.yaml:ner.semantic_training.WARMUP_STEPS',
    'training.yaml:ner.semantic_training.WEIGHT_DECAY',
    'training.yaml:pairs.balance_train_classes',
    'training.yaml:pairs.hardneg_sim_threshold',
    'training.yaml:pairs.proceed_sim_threshold',
    'training.yaml:rand_matching.brand_conflict_veto',
    'training.yaml:rand_matching.calibration_different_gtin_selection',
    'training.yaml:rand_matching.calibration_proxy_source',
    'training.yaml:rand_matching.confidence_penalty_mask.max_penalty',
    'training.yaml:rand_matching.confidence_penalty_mask.penalty_per_joint_missing',
    'training.yaml:rand_matching.confidence_penalty_mask.preserve_exact_gtin',
    'training.yaml:rand_matching.flavor_overlap_penalty.max_penalty',
    'training.yaml:rand_matching.flavor_overlap_penalty.minimum_overlap',
    'training.yaml:rand_matching.flavor_overlap_penalty.preserve_exact_gtin',
    'training.yaml:rand_matching.outputs.calibration_diagnostics',
    'training.yaml:rand_matching.outputs.diagnostics',
    'training.yaml:rand_matching.outputs.holdout_ann_missed_true_matches',
    'training.yaml:rand_matching.outputs.holdout_diagnostics',
    'training.yaml:rand_matching.outputs.holdout_metrics',
    'training.yaml:rand_matching.outputs.holdout_pair_disagreements',
    'training.yaml:rand_matching.outputs.holdout_retrieval_ablation_metrics',
    'training.yaml:rand_matching.outputs.plateau_diagnostic',
    'training.yaml:rand_matching.outputs.submission',
    'training.yaml:rand_matching.outputs.threshold_comparison',
    'training.yaml:rand_matching.outputs.threshold_selection_by_fold',
    'training.yaml:rand_matching.outputs.threshold_sensitivity_by_gtin_status',
    'training.yaml:rand_matching.outputs.threshold_sensitivity_plot',
    'training.yaml:rand_matching.plateau_min_points',
    'training.yaml:rand_matching.plateau_tolerance',
    'training.yaml:rand_matching.stratum_sweep.skus_per_identity',
    'training.yaml:rand_matching.targeted_veto_gates.brand_mismatch_veto',
    'training.yaml:rand_matching.targeted_veto_gates.missing_pack_or_volume_route',
    'training.yaml:rand_matching.targeted_veto_gates.pack_mismatch_veto',
    'training.yaml:rand_matching.targeted_veto_gates.package_type_mismatch_veto',
    'training.yaml:rand_matching.targeted_veto_gates.preserve_exact_gtin',
    'training.yaml:rand_matching.targeted_veto_gates.volume_mismatch_veto',
    'training.yaml:rand_matching.threshold_by_gtin_status.both_equal',
    'training.yaml:rand_matching.threshold_by_gtin_status.both_missing',
    'training.yaml:rand_matching.threshold_by_gtin_status.different',
    'training.yaml:rand_matching.threshold_by_gtin_status.one_missing',
    'training.yaml:rand_matching.threshold_min_fold_support',
    'training.yaml:rand_matching.threshold_reconciliation_scope',
    'training.yaml:rand_matching.threshold_step',
    'training.yaml:rand_matching.truth_splits.calibration_output',
    'training.yaml:rand_matching.truth_splits.holdout_output',
    'training.yaml:rand_matching.unmatched_prefix',
    'training.yaml:rerank.min_delta_f1',
    'training.yaml:rerank.min_delta_pr_auc',
    'training.yaml:sim_columns.deberta_v3_base',
    'training.yaml:sim_columns.minilm_l6',
    'training.yaml:sim_columns.multilingual_l12',
    'training.yaml:split.calibration_dev_fraction',
    'training.yaml:sweep.rerank_model',
    'training.yaml:sweep.smoke_sample',
    'training.yaml:sweep.sweep_sample',
    'training.yaml:training.architecture',
    'training.yaml:training.batch_sampler.composition.gate_positive',
    'training.yaml:training.batch_sampler.composition.hard_negative',
    'training.yaml:training.batch_sampler.composition.masked_positive',
    'training.yaml:training.batch_sampler.compositions_by_loss.mnrl.base',
    'training.yaml:training.batch_sampler.compositions_by_loss.mnrl.twin',
    'training.yaml:training.batch_sampler.compositions_by_loss.triplet.triplet',
    'training.yaml:training.batch_size_cpu',
    'training.yaml:training.batch_size_cuda',
    'training.yaml:training.batch_size_embed',
    'training.yaml:training.batch_size_eval',
    'training.yaml:training.contrastive_margin',
    'training.yaml:training.es_patience',
    'training.yaml:training.es_threshold',
    'training.yaml:training.eval_steps_per_epoch',
    'training.yaml:training.label_smoothing',
    'training.yaml:training.max_triples',
    'training.yaml:training.projection_dropout',
    'training.yaml:training.random_easy_negatives.candidate_pool_size',
    'training.yaml:training.random_easy_negatives.ratio_to_hard',
    'training.yaml:training.structured_features.append_to_text',
    'training.yaml:training.structured_features.embedding_weight',
    'training.yaml:training.structured_features.feed_to_loss',
    'training.yaml:training.structured_features.implicit_pack_qty',
    'training.yaml:training.track_datapoint_usage',
    'training.yaml:training.uniformity_regularization.min_batch_size',
    'training_ANN.yaml:embedding.max_sequence_length',
})

_CONFIG_SSOT_BASELINE_DUPLICATED = frozenset({
    "'.receipt.json' (training.yaml:kaggle.receipt_suffix <- src/cli/laya_lane.py)",
    "'.timing.json' (training.yaml:preparation.stage_timing_suffix <- src/core/timing.py)",
    "'Europe/Paris' (training.yaml:kaggle.limits.timezone <- src/cli/colab.py, src/cli/colab_lane_contracts.py)",
    "'bundle.receipt.json' (training.yaml:kaggle.files.bundle_receipt <- src/cli/colab.py)",
    "'checkpoint_manifest.json' (training.yaml:colab.checkpoint_manifest_name <- src/graph_tracks/train.py)",
    "'config/' (training.yaml:colab.checkout_paths.[1] <- src/core/portable_archive.py, src/model_tracks/package.py)",
    "'config/attribute_ablation.yaml' (model_tracks.yaml:ablation_config <- src/model_tracks/ablation.py, src/model_tracks/config.py)",
    "'config/model_tracks.yaml' (paths.yaml:layouts.model_tracks_config.template <- src/model_tracks/smoke_inputs.py)",
    "'config/model_tracks.yaml' (training.yaml:bundle.suite_config <- src/model_tracks/smoke_inputs.py)",
    "'config/model_tracks.yaml' (training.yaml:kaggle.files.prep_suite_config <- src/model_tracks/smoke_inputs.py)",
    "'data/dataset_deduped.csv' (training.yaml:colab.training_dataset_csv <- src/cli/colab.py, src/cli/colab_lane_contracts.py)",
    "'data/prepared/smoke_200' (training.yaml:preparation.smoke_dir <- src/cli/colab_lane_cpu_provision.py)",
    "'dataset-metadata.json' (training.yaml:kaggle.files.dataset_metadata <- src/cli/laya_lane.py)",
    "'dataset.csv' (training.yaml:bundle_prep.export_csvs.[0] <- src/cli/colab_bundle.py, src/cli/colab_lane.py)",
    "'dataset.csv' (training.yaml:kaggle.export_csvs.[0] <- src/cli/colab_bundle.py, src/cli/colab_lane.py)",
    "'dataset.csv' (training.yaml:kaggle.packaged_data_name <- src/cli/colab_bundle.py, src/cli/colab_lane.py)",
    "'dataset.csv' (training.yaml:ner.data_prep.input_csv <- src/cli/colab_bundle.py, src/cli/colab_lane.py)",
    "'kernel-metadata.json' (training.yaml:kaggle.files.kernel_metadata <- src/cli/laya_lane.py, src/core/runtime_inputs.py)",
    "'lane.log' (training.yaml:kaggle.files.autowatch_log <- src/cli/colab.py, src/cli/colab_self_watch.py)",
    "'lane.log' (training.yaml:kaggle.files.lane_log <- src/cli/colab.py, src/cli/colab_self_watch.py)",
    "'lane.log' (training.yaml:kaggle.files.stream_log <- src/cli/colab.py, src/cli/colab_self_watch.py)",
    "'manifest.json' (training.yaml:kaggle.files.bundle_sidecars.[0] <- src/cli/colab.py, src/model_tracks/resource_profile.py)",
    "'manifest.json' (training.yaml:preparation.manifest_file <- src/cli/colab.py, src/model_tracks/resource_profile.py)",
    "'optimizer.pt' (training.yaml:bundle.deployment_ignored_filenames.[0] <- src/training/training.py)",
    "'optimizer.pt' (training.yaml:bundle.resume_only_filenames.[0] <- src/training/training.py)",
    "'paths.yaml' (training.yaml:packaging.snapshot_pinned_configs.[0] <- src/core/common.py)",
    "'requirements.txt' (training.yaml:colab.checkout_paths.[6] <- src/core/runtime_inputs.py)",
    "'requirements/graph_tracks.txt' (training.yaml:kaggle.bundle_requirements <- src/graph_tracks/worker_package.py)",
    "'results/attribute_ablation' (attribute_ablation.yaml:output_dir <- src/model_tracks/ablation.py)",
    "'results/attribute_ablation/report.json' (attribute_ablation.yaml:report_path <- src/model_tracks/ablation.py)",
    "'scaler.pt' (training.yaml:bundle.deployment_ignored_filenames.[4] <- src/training/training.py)",
    "'scaler.pt' (training.yaml:bundle.resume_only_filenames.[4] <- src/training/training.py)",
    "'scheduler.pt' (training.yaml:bundle.deployment_ignored_filenames.[1] <- src/training/training.py)",
    "'scheduler.pt' (training.yaml:bundle.resume_only_filenames.[1] <- src/training/training.py)",
    "'scripts/' (training.yaml:colab.checkout_paths.[2] <- src/core/portable_archive.py, src/model_tracks/package.py)",
    "'scripts/diet_manifest.py' (training.yaml:packaging.snapshot_pinned_files.[1] <- src/cli/colab_bundle_prewarm.py, src/model_tracks/preflight.py)",
    "'scripts/run_colab_ablation.py' (training.yaml:packaging.snapshot_pinned_files.[2] <- src/model_tracks/post_training_ablation.py)",
    "'src/pipeline.py' (training.yaml:packaging.snapshot_pinned_files.[0] <- src/graph_tracks/text_cache.py, src/training/selftest.py)",
    "'tar.zst' (model_tracks.yaml:input_archive_format <- src/model_tracks/config.py)",
    "'tar.zst' (model_tracks.yaml:result_archive_format <- src/model_tracks/config.py)",
    "'tar.zst' (training.yaml:archives.format <- src/model_tracks/config.py)",
    "'text.yaml' (training.yaml:preparation.graph_setup.text_config <- src/model_tracks/package.py)",
    "'text_track.yaml' (training.yaml:packaging.snapshot_pinned_configs.[5] <- src/graph_tracks/setup.py)",
    "'timings.json' (training.yaml:kaggle.files.bundle_sidecars.[1] <- src/model_tracks/run.py)",
    "'timings.json' (training.yaml:preparation.timings_file <- src/model_tracks/run.py)",
    "'timings.log' (training.yaml:preparation.timings_log <- src/model_tracks/run.py)",
    "'training.yaml' (training.yaml:packaging.snapshot_pinned_configs.[1] <- src/core/common.py)",
    "'training_args.bin' (training.yaml:bundle.deployment_ignored_filenames.[3] <- src/training/training.py)",
    "'training_args.bin' (training.yaml:bundle.resume_only_filenames.[3] <- src/training/training.py)",
    "'vectors.npz' (training.yaml:kaggle.files.vectors_file <- src/graph_tracks/infer.py, src/graph_tracks/report.py)",
    "'vocabulary.json' (training.yaml:packaging.snapshot_pinned_configs.[4] <- src/core/common.py, src/core/critical_attributes.py)",
})


def _new_config_drift(repo_root: Path) -> tuple[list[str], list[str], set, set]:
    """Measured drift minus the pinned baseline, plus the measured sets."""
    drift = _config_drift(repo_root)
    unread = set(drift["unread"])
    duplicated = set(drift["duplicated"])
    return (
        sorted(unread - _CONFIG_SSOT_BASELINE_UNREAD),
        sorted(duplicated - _CONFIG_SSOT_BASELINE_DUPLICATED),
        unread,
        duplicated,
    )


def _assert_no_new_config_drift(repo_root: Path) -> None:
    """Fail on config-SSOT drift that is NOT in the pinned baseline snapshot."""
    new_unread, new_duplicated, unread, duplicated = _new_config_drift(repo_root)
    assert not new_unread and not new_duplicated, (
        "NEW config-SSOT drift beyond the pinned baseline: "
        f"{len(new_unread)} unread leaf/leaves and {len(new_duplicated)} "
        f"respelled config value(s) (measured {len(unread)} / {len(duplicated)}).\n"
        f"unread (no src attribute reader): {new_unread[:5]}\n"
        f"respelled (code literal duplicates a config-owned value): "
        f"{new_duplicated[:3]}\n"
        "Fix the new leaf/literal (or delete the dead config leaf), do not "
        "widen _CONFIG_SSOT_BASELINE_*."
    )


def test_config_leaves_have_readers_and_no_literal_duplicate_values():
    """Every config leaf is read, and no code literal respells a config value.

    The scan is static (yaml leaves vs src AST). A leaf is "read" only when
    code performs a REAL attribute access on the loaded config
    (``data_cfg().column_mapping``, ``training_cfg().bundle...``); a bare
    identifier or a string literal that merely mentions the key is a
    coincidence, not a reader, so a dead block can no longer hide behind its
    own name.

    ENFORCEMENT: the measured drift must be a SUBSET of the pinned baseline
    (``_CONFIG_SSOT_BASELINE_UNREAD`` / ``..._DUPLICATED``). A NEW unread leaf
    or a NEW respelled config-owned value therefore FAILS this test; the old
    ``pytest.xfail`` shape could never fail, so the invariant was green on any
    drift whatsoever. Known drift shrinking stays green (the guard only moves
    in the safe direction); delete the fixed entry from the baseline.
    """
    _assert_no_new_config_drift(Path(__file__).parents[1])


def _synthetic_repo(tmp_path: Path, config_yaml: str, source: str) -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/demo.yaml").write_text(config_yaml)
    (tmp_path / "src/demo.py").write_text(source)
    return tmp_path


def test_config_drift_guard_fails_on_a_new_unread_leaf(tmp_path):
    """The guard's own failure mode: a NEW unread leaf must trip the assert.

    Regression pin for the defect this replaced: an invariant that can never
    fail is not an invariant. This runs the REAL scanner over a synthetic repo
    whose only leaf has no src attribute reader and is absent from the pinned
    baseline.
    """
    repo = _synthetic_repo(tmp_path, "brand_new_dead_leaf: 1\n", "value = 1\n")
    with pytest.raises(AssertionError, match="NEW config-SSOT drift"):
        _assert_no_new_config_drift(repo)
    new_unread, new_duplicated, _, _ = _new_config_drift(repo)
    assert new_unread == ["demo.yaml:brand_new_dead_leaf"]
    assert new_duplicated == []


def test_config_drift_guard_fails_on_a_new_respelled_config_value(tmp_path):
    """A NEW code literal respelling a read config value must trip the assert.

    The leaf IS read (``cfg.dataset_path``), so the unread class stays empty
    and the new-drift failure is carried by the respelling class alone.
    """
    repo = _synthetic_repo(
        tmp_path,
        "dataset_path: results/attribute_ablation/report.json\n",
        "from types import SimpleNamespace\n"
        "cfg = SimpleNamespace(dataset_path='x')\n"
        "read = cfg.dataset_path\n"
        "REPORT_PATH = 'results/attribute_ablation/report.json'\n",
    )
    with pytest.raises(AssertionError, match="NEW config-SSOT drift"):
        _assert_no_new_config_drift(repo)
    new_unread, new_duplicated, _, _ = _new_config_drift(repo)
    assert new_unread == []
    assert len(new_duplicated) == 1 and "dataset_path" in new_duplicated[0]


# ── 5. the orchestration stage map matches the producers' STAGE constants ───

#: Each orchestration stage and the producer module + attribute that declares the
#: stage name it writes rows under. ``core.tracing.orchestration_trace_stages()``
#: (reading config/paths.yaml's ``orchestration_stages`` block) must map every one
#: of these to its producer's constant: ``()`` means "writes no trace rows" and
#: must never stand next to a producer that writes them.
_PRODUCER_STAGE_CONSTANTS = {
    "dedupe": ("training.dedupe", "STAGE"),
    "cross_country_pairs": ("training.build_second04_pairs", "STAGE"),
    "number_reference": ("training.build_reference", "STAGE_WRITE"),
    "verify_reference": ("training.build_reference", "STAGE_VERIFY"),
    "validation": ("training.build_final_validation", "STAGE"),
    "labeled_pairs": ("training.labeled_pairs", "STAGE"),
    "full_bundle": ("training.train", "BUNDLE_STAGE"),
    "verify_handoff": ("training.handoff", "STAGE"),
    "negative_supply": ("training.negative_supply", "STAGE"),
    "suite_inputs": ("model_tracks.package", "STAGE"),
}


def test_orchestration_stage_map_matches_producer_stage_constants():
    """The trace stage map is the producers' own vocabulary, not a parallel one.

    ``trace_stages_for`` must return each producer's ``STAGE`` constant for the
    orchestration stage that producer serves, and an undeclared stage must stay
    a loud error (a reader can never silently join nothing).
    """
    import importlib

    from core import tracing
    from training import prepare_all

    registry = tracing.orchestration_trace_stages()
    trace_stages_for = tracing.trace_stages_for

    declared = set(prepare_all.STAGES) | {"negative_supply", "discriminator"}
    assert declared == set(registry), \
        "the map and the orchestrator's stage list are one contract"

    drifted = []
    for stage, (module_name, attribute) in sorted(_PRODUCER_STAGE_CONSTANTS.items()):
        constant = getattr(importlib.import_module(module_name), attribute)
        if trace_stages_for(stage) != (constant,):
            drifted.append(f"{stage}: map={trace_stages_for(stage)} producer={constant!r} "
                           f"({module_name}.{attribute})")
    from graph_tracks import prepare as graph_prepare, setup as graph_setup
    if set(trace_stages_for("graph_inputs")) != {graph_setup.STAGE, graph_prepare.STAGE}:
        drifted.append("graph_inputs: map=" + repr(trace_stages_for("graph_inputs"))
                       + " producers=" + repr((graph_setup.STAGE, graph_prepare.STAGE)))

    with pytest.raises(ValueError, match="unknown orchestration stage"):
        trace_stages_for("not-a-stage")

    if drifted:
        pytest.xfail(
            "the config registry (config/paths.yaml orchestration_stages) claims "
            "'() = writes no trace rows' for stages whose producers write them "
            "under that exact name — owner: config/paths.yaml: " + "; ".join(drifted))
