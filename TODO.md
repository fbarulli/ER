# TODO — ER

Status: `[ ]` open · `[~]` in progress · `[x]` done.
Added 2026-10-08; refreshed after the consolidation audit (commits `5bc48c9`,
`46811e9`). Anchors are approximate.

## Bundle refactor (artifact + process)
- [~] `core/bundle.py` — `Bundle(role∈{inputs,recovery,result})` + `BundlePipeline` **class written**; single boundary verify via `verify_archive_digest`; accessors `checkpoint/checkpoints/track_inventory/track_complete/ablation_templates/ablation_skipped`; public `bundle_spec()` added (lanes no longer import the private `_bundle_spec`).
- [x] `model_tracks/bundle_steps.py` — exists; `prepare_inputs()` + `finalize(result)` are the single generation/finalize/ablation surface.
- [ ] Role enforcement: result = selected checkpoint only; recovery = all epochs + optimizer; inputs = none. Retire `selected_checkpoint_dirs`/`RESUME_ONLY_FILENAMES` into `Bundle`. (`package._assert_recovery_contract` still re-derives the recovery role contract — centralizing it in `core/bundle.py` is the remaining step.)
- [ ] Transports (git / kaggle-dataset / file) load+save one `Bundle`, one integrity check per VM crossing. No stage re-hashes.
- [~] Migrate call sites — most now read the Bundle spec (`suite_events_file`, `worker_events_file`, `prepared_inputs_dir`, `manifest_name(...)`); remaining inline finalize logic is noted below.
- [ ] Remove the operator-box finalize surface (`snapshot_completion` → local `local_complete`); finalize becomes a remote CPU lane job run from a sparse checkout (Kaggle finalize kernel now wired; colab not).
- [ ] Single-archive handoff (#5): fold `suite_events.jsonl` into the result archive (flush final events before sealing), drop the second `.events.jsonl` sidecar download (`model_tracks/run.py` still copies the sidecar).

## Track training (cascade era)
- [~] `model_tracks/parallel.py` — postprocess-cascade sequential barrier is now the documented SSOT (`run.py` delegates to `parallel.run_postprocess_track` and `parallel.split_tracks`); a real GPU/CPU end-to-end suite run is still unproven.
- [x] `tests/test_tracks_direct_download.py` — green.
- [x] `local_complete` cascade branch present (delegates to `bundle_steps._complete_cascade_track`); `post_training_ablation` + `archive_verification` with cascade remain unproven end-to-end.
- [x] Regenerate stale fixtures: `smoke_200`, `smoke_500`, `data/track_setup` all carry `cascade.yaml` and no `hybrid`.
- [x] `src/training/run_plan.py` — frozen-input materialization maps bare keys → `*_csv` members (`_FROZEN_INPUT_MEMBERS`); `test_smoke_run_plan` green.
- [x] `src/model_tracks/smoke_inputs.py` — repo-relative fixture paths resolved against `TRAIN_ROOT` before `relative_to`.
- [x] `data/prepared/smoke_500` — regenerated (cascade template present).

## Lanes
- [ ] Colab + Kaggle transports load/save `Bundle`; Kaggle finalize kernel is wired, colab finalize is not.
- [ ] Kaggle `embedding_kernel_slug` (`fbarulli/er-embed-gpu`) — **no embed kernel exists on the account**; push one or remove the embed objective.
- [ ] Colab `cohort_label` SSOT — hardcodes `dataset.csv`/`50pct` and returns `dataset_10k` for the 10k export while Kaggle returns `10k`; needs a `colab.*.cohort_tags` config key.
- [ ] `cli/laya_lane.py` kernel-template helper duplication (`log` ×4, `resolve_input` ×4, `sha256_of` ×2): inject shared fragments via the existing token mechanism (artifact-byte-risky; deferred).
- [ ] `_env_dot_value` vs `_env_value` near-duplicates in `cli/`.

## Standalone bundlers (optional, "all surfaces")
- [ ] `graph_tracks/{bundle,worker_package}.py`, `ner/{ner,colab_ner}.py`, `cli/laya_lane.py` dataset payloads — migrate archive/hash to `Bundle`.

## Prepared-layout SSOT (post-audit)
- [x] `core.common.prepared_setup_layout()` — the single layout accessor; all 21 `_setup_layout()` shims delegate to it.
- [~] `PreparationGraphSetupSpec.input_manifest`/`.listings` declared; the ~30 call-site literals (`input_manifest.json`, `listings.json`) still spell the names and should be repointed at the spec.
- [x] Trace batch caps share `core.tracing.TRACE_BATCH_ROWS` / `TRACE_MAX_BATCH_ROWS` (per-stage graph sizes that genuinely differ stay local).
- [~] The per-module `_spec()` wrapper (13 copies in `model_tracks/`) re-wraps `core.bundle.bundle_spec()`; collapse to direct calls.

## Eval balance / data coverage
- [x] `config/model_tracks.yaml` — `report_test: true`.
- [x] Support-floor gate in `model_tracks/preflight.py`.
- [x] `split.negative_fold_policy` re-pinned to `train_side` (evidence-backed; `build_final_validation` measures trained-on-endpoint leakage = 0 under both policies).
- [ ] Balance dev/test negatives (scored halves still positive-dominated).
- [ ] Diet/coverage: compute coverage **per fold**; prefer **generate-only-if-covered** over backfill.
- [x] Augmentation on/off experiment + label-quality check for minted negatives (`training/augmentation_experiment.py`, covered).

## Accel optimization
- [ ] `core/fast_kernels.py` — Triton `_segment_add_kernel` unvalidated on GPU; atomic_add contention → grouped reduction; validate vs `index_add_` fallback on a GPU.
- [ ] Text trainer: remove redundant `.to(device)`/per-step `.item()` syncs; cache re-encodes (respect option shuffle).
- [~] `graph_tracks/pooling.py` — fusion covered by `test_pooling_fusion.py`; verify on GPU.
- [ ] Wire `compile_model`/`segment_reduce_fast`/`autocast_context` into `graph_tracks/model.py`.

## Laya
- [x] Staging `origin/main == HEAD` blocker — `main` was pushed; local is now only the two audit commits ahead.
- [ ] Run **full finetune on the augmented corpus** (14,283 rows: train 7,538 / dev 3,471 / test 3,274) — recipe-driven via `laya.finetune:` YAML.
- [ ] Held-out eval on the **full** test split via `finetune-eval` / `--local-eval`; compare to 0.91.
- [ ] Head-to-head vs the `gnn_only` decider on the same held-out pairs → decide laya's role.
- [ ] Calibration: per-type temperature; abstention (`min_router_confidence`).
- [ ] Per-slice / per-attribute traceability; attribute-ablation (answer-flip per attribute).
- [ ] Corpus semantics: identity ratio ~1:14 (567 pos / 8,462 neg) — rebalance or tune.
- [ ] Optuna HPO over the YAML knobs.
- [ ] New questions to add (labels exist): `field_same:<attr>`, `pack_volume_equal`, `pack_format_equivalent`, `gate_verdict`/`gate_reason`, `counterfactual`, `same_brand_only`, `evidence_sufficient`, pairwise `better_match`.

## Loose ends
- [x] `tests/test_lane_fixes.py::test_finetune_payload_embeds_patch_and_passes_gates` — fixed (supplies `HELD_OUT_BATCH`).
- [x] Pre-existing suite failures — fixed; full suite is green (2442 passed, 12 skipped, 1 xfailed).
- [x] Cascade wiring out-of-scope edits reviewed and committed.
- [x] Landed work committed (`5bc48c9`, `46811e9`) — **not pushed** (2 commits ahead of `origin/main`).
- [ ] `graph_tracks/train.py` hardcodes checkpoint-layout literals (`_checkpoints`, `trainer_state.json`); writer/reader agree via spec-owned globs today, left alone (higher-risk writer contract).
- [ ] `run.py` still completes ablation inline for the local non-GPU path (a parallel of `bundle_steps.finalize`); merging changes lane behavior — kept behavior-preserving.
