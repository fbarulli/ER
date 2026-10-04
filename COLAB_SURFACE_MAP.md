# Colab surface ownership

This maps entry points and source ownership. It does not certify a live GPU
run. Prepared inputs must be regenerated after source/config changes; artifact
generation is currently deferred.

| Boundary | Shared owner / contract |
|---|---|
| Full preparation | `training.prepare_all.PreparationState`, per-stage `StageManifest`, source/config/raw/checkpoint provenance |
| Native text bundle | `PreparedBundleManifest`, `PreparedTokenInputs`, frozen training plan, config-owned drift policy |
| Suite configuration | `model_tracks.config.SuiteConfig` from the explicit suite YAML |
| Graph/text lane configuration | `graph_tracks.config.GraphConfig`, `TextConfig`, `RetrievalConfig` from staged lane YAMLs |
| Source overlay | `model_tracks.package.runtime_snapshot_files` and `core.portable_archive.RuntimeSnapshot`; includes every Python source under `src/` |
| Suite and standalone graph packaging | Shared runtime snapshot and `core.portable_archive.write_archive` / `verify_archive` |
| GPU forwarding | `PreparedEmbeddingForward`, native-token identity, checkpoint and request hashes |
| Worker completion | `TrackCompletion`, `TrackInventory`; the same validators govern current files, downloaded archives and suite recovery |
| Training-to-report binding | `TrainingInputBinding`, `RuntimeBinding`, suite config, preflight identity and exact source inventory |
| Pair/retrieval reports | Shared graph report functions, `RetrievalReportContext`, `TrackReportManifest`, config-owned operating targets |
| Measurements | `SectionTiming`, `RefreshTiming`, `PerformanceSummary`, `ProfilerMetadata` |
| Saved ablation | `ablation.Settings`, staged input contracts and saved dev-threshold/checkpoint binding |
| Publication | Existing suite/graph/native publishers consume verified immutable generations; their storage formats remain explicit |

`colab_backend.py` delegates to `cli.colab.main`. The default `--what tracks`
selects the three-track suite. Native `--what smoke` without a suite config is a
separate text-training entry point. An all-track smoke uses an explicit smoke
suite config and prepared projection. Embedding and ablation scripts use the
same runtime snapshot, archive validation and config owners.

The native checkpoint marker names are configurable under `colab`; graph
checkpoint names are namespaced artifact-format identifiers owned by
`graph_tracks.artifacts`. They are separate formats, not competing settings.

## Complete direct module inventory

Every Python module in `src/cli`, `src/model_tracks`, and `src/graph_tracks` is
assigned below. Imported training/core/NER modules are included recursively in
the runtime snapshot. Preparation entry points also include
`training.prepare_all`, `training.data_prep`, `training.labeled_pairs`,
`training.base_data`, `training.prepared_bundle`, `training.prepare_tokens`,
`training.prepare_embeddings`, `training.train_prepared`, and
`training.complete_colab_worker`. Script entry points are
`run_colab_smoke.sh`, `run_full_training.sh`, `run_colab_embeddings.py`,
`run_colab_ablation.py`, and `rebuild_training_handoff.py` under `scripts/`.

**CLI entry and runtime transport**

[src/cli/colab.py](src/cli/colab.py), [src/cli/colab_cli_entry.py](src/cli/colab_cli_entry.py).

**Suite configuration and admission**

[src/model_tracks/config.py](src/model_tracks/config.py), [src/model_tracks/preflight.py](src/model_tracks/preflight.py), [src/model_tracks/smoke_inputs.py](src/model_tracks/smoke_inputs.py).

**Frozen inputs and recovery**

[src/model_tracks/package.py](src/model_tracks/package.py), [src/model_tracks/resume.py](src/model_tracks/resume.py), [src/model_tracks/snapshot_completion.py](src/model_tracks/snapshot_completion.py).

**Launch and worker lifetime**

[src/model_tracks/colab.py](src/model_tracks/colab.py), [src/model_tracks/parallel.py](src/model_tracks/parallel.py), [src/model_tracks/run.py](src/model_tracks/run.py), [src/model_tracks/worker.py](src/model_tracks/worker.py).

**GPU exports**

[src/model_tracks/baseline_export.py](src/model_tracks/baseline_export.py), [src/model_tracks/embedding_forward.py](src/model_tracks/embedding_forward.py), [src/model_tracks/text_export.py](src/model_tracks/text_export.py).

**CPU completion and publication**

[src/model_tracks/incremental.py](src/model_tracks/incremental.py), [src/model_tracks/local_complete.py](src/model_tracks/local_complete.py), [src/model_tracks/publish.py](src/model_tracks/publish.py), [src/model_tracks/text_report.py](src/model_tracks/text_report.py).

**Telemetry**

[src/model_tracks/live_logs.py](src/model_tracks/live_logs.py), [src/model_tracks/telemetry.py](src/model_tracks/telemetry.py).

**Ablation**

[src/model_tracks/ablation.py](src/model_tracks/ablation.py), [src/model_tracks/ablation_inputs.py](src/model_tracks/ablation_inputs.py), [src/model_tracks/ablation_retrieval.py](src/model_tracks/ablation_retrieval.py), [src/model_tracks/baseline_ablation.py](src/model_tracks/baseline_ablation.py), [src/model_tracks/post_training_ablation.py](src/model_tracks/post_training_ablation.py), [src/model_tracks/staged_ablation.py](src/model_tracks/staged_ablation.py).

**Graph configuration and prepared data**

[src/graph_tracks/config.py](src/graph_tracks/config.py), [src/graph_tracks/data.py](src/graph_tracks/data.py), [src/graph_tracks/prepare.py](src/graph_tracks/prepare.py), [src/graph_tracks/prepared_inputs.py](src/graph_tracks/prepared_inputs.py), [src/graph_tracks/report_attributes.py](src/graph_tracks/report_attributes.py), [src/graph_tracks/setup.py](src/graph_tracks/setup.py), [src/graph_tracks/text_cache.py](src/graph_tracks/text_cache.py).

**Graph execution**

[src/graph_tracks/benchmark.py](src/graph_tracks/benchmark.py), [src/graph_tracks/infer.py](src/graph_tracks/infer.py), [src/graph_tracks/model.py](src/graph_tracks/model.py), [src/graph_tracks/pooling.py](src/graph_tracks/pooling.py), [src/graph_tracks/preflight.py](src/graph_tracks/preflight.py), [src/graph_tracks/train.py](src/graph_tracks/train.py).

**Graph reporting and collection**

[src/graph_tracks/artifacts.py](src/graph_tracks/artifacts.py), [src/graph_tracks/bundle.py](src/graph_tracks/bundle.py), [src/graph_tracks/dvc.py](src/graph_tracks/dvc.py), [src/graph_tracks/report.py](src/graph_tracks/report.py), [src/graph_tracks/report_manifest.py](src/graph_tracks/report_manifest.py), [src/graph_tracks/report_slices.py](src/graph_tracks/report_slices.py), [src/graph_tracks/tracking.py](src/graph_tracks/tracking.py), [src/graph_tracks/worker_package.py](src/graph_tracks/worker_package.py).

**Package initialization**

[src/cli/__init__.py](src/cli/__init__.py), [src/graph_tracks/__init__.py](src/graph_tracks/__init__.py), [src/model_tracks/__init__.py](src/model_tracks/__init__.py).
