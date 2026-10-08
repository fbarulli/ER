# TODO — ER

Status: `[ ]` open · `[~]` in progress · `[x]` done.
Added 2026-10-08. Anchors are approximate.

## Bundle refactor (artifact + process)
- [~] `core/bundle.py` — `Bundle(role∈{inputs,recovery,result})` + `BundlePipeline` **class written**; single boundary verify via `verify_archive_digest`; accessors `checkpoint/checkpoints/track_inventory/track_complete/ablation_templates/ablation_skipped`.
- [ ] `model_tracks/bundle_steps.py` — **does not exist**; implement `prepare_inputs()` + `finalize(result)` (the only place generation/finalize/ablation run).
- [ ] Role enforcement: result = selected checkpoint only; recovery = all epochs + optimizer; inputs = none. Retire `selected_checkpoint_dirs`/`RESUME_ONLY_FILENAMES` into `Bundle`.
- [ ] Transports (git / kaggle-dataset / file) load+save one `Bundle`, one integrity check per VM crossing. No stage re-hashes.
- [ ] Migrate call sites: `model_tracks/{package,run,worker,resume,local_complete,snapshot_completion,publish,incremental,archive_verification}.py`, `model_tracks/colab.py`.
- [ ] Remove the operator-box finalize surface (`snapshot_completion` → local `local_complete`); finalize becomes a remote CPU lane job run from a sparse checkout (Kaggle currently has no finalize).
- [ ] Single-archive handoff (#5): fold `suite_events.jsonl` into the result archive (flush final events before sealing), drop the second `.events.jsonl` sidecar download.

## Track training (cascade era)
- [~] track-training agent running — `model_tracks/parallel.py` still needs the real fix (currently `run.py` runs cascade sequentially as a workaround); reconcile `resume/worker/package/preflight/data_gate/smoke_inputs` on `text/gnn_only/cascade`.
- [ ] `tests/test_tracks_direct_download.py` x4 — fix the `model_tracks/colab.py` `verify_archive_digest` call site (unstubbed core call).
- [ ] `local_complete` has no cascade branch; `post_training_ablation` + `archive_verification` with cascade (+`post_training_ablation=True`) unproven.

## Lanes
- [ ] Colab + Kaggle transports load/save `Bundle`; wire a Kaggle **finalize** step.
- [ ] Kaggle `embedding_kernel_slug` (`fbarulli/er-embed-gpu`) — **no embed kernel exists on the account**; push one or remove the embed objective.
- [ ] Lanes run bundling from a sparse checkout (already true for tracks; verify for finalize).

## Standalone bundlers (optional, "all surfaces")
- [ ] `graph_tracks/{bundle,worker_package}.py`, `ner/{ner,colab_ner}.py`, `cli/laya_lane.py` dataset payloads — migrate archive/hash to `Bundle`.

## Eval balance / data coverage
- [ ] `config/model_tracks.yaml` — flip `report_test: true` (still `false`); held-out test never scored.
- [ ] Balance dev/test negatives (currently dev 1286/9, test 1276/7; train is 38400/38400).
- [ ] Support-floor gate in `model_tracks/preflight.py` (refuse to publish with too few negatives).
- [ ] Diet/coverage: 980/1,200 masked-positive copies never train (no source negative); compute coverage **per fold**; prefer **generate-only-if-covered** over backfill.
- [ ] Augmentation on/off experiment + label-quality check for minted negatives.

## Accel optimization
- [ ] `core/fast_kernels.py` — Triton `_segment_add_kernel` unvalidated on GPU; atomic_add contention → grouped reduction; validate vs `index_add_` fallback on a GPU.
- [ ] Text trainer: remove redundant `.to(device)`/per-step `.item()` syncs; cache re-encodes (respect option shuffle).
- [ ] `graph_tracks/pooling.py` — fuse the two `index_add_` passes.
- [ ] Wire `compile_model`/`segment_reduce_fast`/`autocast_context` into `graph_tracks/model.py` (currently only "intended call sites").

## Laya
- [ ] **Blocker**: staging fails `origin/main == HEAD` — local `main` is 63 commits ahead (consolidation merges, unpushed). Push `main` or wait for the consolidation session.
- [ ] Run **full finetune on the augmented corpus** (14,283 rows: train 7,538 / dev 3,471 / test 3,274) — recipe-driven via `laya.finetune:` YAML.
- [ ] Held-out eval on the **full** test split (1,188 baseline → now 3,274) via `finetune-eval` / `--local-eval`; compare to 0.91 (old 100-item).
- [ ] Head-to-head vs the `gnn_only` decider on the same held-out pairs → decide laya's role (cascade decider / abstaining adjudicator / park).
- [ ] Calibration: overconfident (conf 0.975 vs acc 0.91); per-type temperature; abstention (`min_router_confidence`).
- [ ] Per-slice / per-attribute traceability (laya has only per-question-type today): tag corpus rows with `difficulty_slice`/`gate_reason`/attribute; extend eval; add an attribute-ablation (answer-flip per attribute).
- [ ] Corpus semantics: context masks, not attribute-removed; identity ratio now ~1:14 (567 pos / 8,462 neg) — rebalance or tune.
- [ ] Optuna HPO over the YAML knobs, objective = held-out accuracy + calibration.
- [ ] New questions to add (labels exist): `field_same:<attr>`, `pack_volume_equal`, `pack_format_equivalent`, `gate_verdict`/`gate_reason`, `counterfactual`, `same_brand_only`, `evidence_sufficient`, pairwise `better_match`.

## Loose ends
- [ ] `tests/test_lane_fixes.py::test_finetune_payload_embeds_patch_and_passes_gates` — red; references the removed `FINETUNE_RECIPE`, never substitutes `@PERF_PATCH@`.
- [ ] 7 pre-existing suite failures remain (laya ones now fixed): `test_model_input_contract::test_legacy_profile_reproduces_golden_bytes`, `test_alias_aware_brand_veto::test_frozen_canonical_records_have_no_leading_zero_gtins`, `test_validation_inference::test_the_scored_pair_census_closes_and_is_never_hardcoded`.
- [ ] Cascade wiring left out-of-scope edits in `model_tracks/smoke_inputs.py` + several test files (reported, unreviewed).
- [ ] Commit/push the landed work (currently all uncommitted; a concurrent session is doing git branch ops).
