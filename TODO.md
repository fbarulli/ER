# TODO — ER

Status: `[ ]` open · `[~]` in progress · `[x]` done.
Added 2026-10-08; refreshed after the consolidation audit (commits `5bc48c9`,
`46811e9`, `aa899cc`, `e5bf511`) and again by the 2026-10-09 data-classes audit
(section below). Anchors are approximate.

## Data classes — remaining work (2026-10-09 audit)

Scope: `Dataset` (`src/core/dataset.py`), `HostedRegistry` (`src/core/hosted_dataset.py`),
`Artifacts` (`src/core/artifacts.py`), `Results` (`src/core/results.py`), `Bundle`
(`src/core/bundle.py`). Ordered; every item names its evidence and ONE acceptance
check. At audit the focused classes were green: `test_dataset.py test_hosted_registry.py
test_artifacts.py test_results.py test_structural_identity.py test_bundle.py` (103 passed).

- [ ] **0. Land the in-flight class work.** The worktree holds uncommitted `Artifacts.resolve`/
  `Artifacts.member_name` classmethods + `Artifacts`/`Results` spec `lru_cache`
  (`src/core/artifacts.py:216,566,584`; `src/core/results.py:174,236,250`), the
  `Results.for_root`/`track_dir` public surface, and the 9 `src/model_tracks/*` importers.
  Evidence: `git status` M `src/core/{artifacts,results}.py`, M `config/results.yaml`, 12 `src/model_tracks/*.py`.
  Acceptance: `pytest tests/test_artifacts.py tests/test_results.py -q` green, then commit. (Prerequisite.)
- [ ] **1. `KaggleSpec` must REFERENCE `HostedRegistry`, not re-declare slugs** (dataclass.md §4.2).
  `src/core/schemas.py:3404 embedding_dataset_slug`, `:3408 bundle_dataset_slug` and
  `config/training.yaml:306,308` restate `fbarulli/er-embed-requests` / `fbarulli/er-10k-bundle`,
  already declared in `config/hosted_datasets.yaml` (roles `embeddings`, `bundle`). Add
  `KaggleSpec.hosted_slug(role)` (mirror `LayaSpec.hosted_slug`, `src/core/laya_config.py:266-276`) and route
  `cli/kaggle_kernels.py:209,443,452,474,549`, `cli/kaggle_datasets.py:342`, `cli/kaggle_chain.py:113,129`
  through `hosted_registry().by_role(...)`; delete the two yaml keys.
  Acceptance: `grep -rn "bundle_dataset_slug\|embedding_dataset_slug" src/ config/` → 0 and a new test
  asserts the CLI bundle slug == `hosted_registry().by_role("bundle").slug`.
  Blast radius: `tests/test_kaggle_lane.py` (~12 fixtures), `tests/test_invariants.py:218-219`.
- [ ] **2. Mount root declared twice.** `config/training.yaml:237 kaggle.remote.input_dir: /kaggle/input`
  (`src/core/schemas.py:3313`) vs `config/hosted_datasets.yaml mount_root: /kaggle/input`
  (`src/core/hosted_dataset.py:198`). Make one the source, the other reference it.
  Acceptance: a test asserts `hosted_registry().mount_root == Path(training_cfg().kaggle.remote.input_dir)`
  and `grep -c "/kaggle/input" config/hosted_datasets.yaml` → 0 (or the reverse).
- [ ] **3. Validation size must be FLEX, not a fixed count** (brief; declared but unconsumed).
  `src/core/dataset.py:271 validation_size` + `config/dataset.yaml:34,89` declare it, yet
  `grep -rn validation_size src/` → only `dataset.py`; the REAL validation size is pinned by
  `src/core/schemas.py:791 holdout_component_folds: Literal[4]`, keyed off in
  `src/training/folds.py:487-504`, `src/training/build_final_validation.py:1090-1109`,
  `src/training/evaluate_models.py:213-230`, `src/core/schemas.py:839-869`. Either
  (a) widen the arity and drive the cut from `Dataset.validation_size` (fraction or row count), or
  (b) delete `validation_size` + its test + the yaml claim so no decorative knob remains.
  Acceptance: (a) a test varies `Dataset.validation_size` and the emitted validation row count follows it;
  (b) `grep -rn validation_size src/ config/ tests/` → 0 and the FLEX test is gone.
- [ ] **4. `Bundle` owns its role/member contract in ONE place.**
  (a) `src/model_tracks/package.py:613 _assert_recovery_contract` re-derives the recovery contract that
  `src/core/bundle.py:150 _role_member_violations` already encodes; move the positive assertion onto `Bundle`
  and have package call it.
  (b) `src/core/bundle.py:505-510` hardcodes `.env`, `config.local`, `.publication`, `__payload`,
  `_artifact_publications` inside `is_result_member`, and `src/model_tracks/package.py:646,648` re-spells
  `('.publication','__payload')` / `{'.env','config.local'}`; declare them in `BundleSpec`
  (`src/core/schemas.py:3554`) and read them there.
  Acceptance: `grep -n "\.publication\|__payload\|_artifact_publications" src/core/bundle.py
  src/model_tracks/package.py` → only config-field reads; `pytest tests/test_bundle.py -q` green plus one test
  pinning that a pruned-epoch recovery set is refused by the Bundle method.
- [ ] **5. `Results` leaf re-spelled.** `src/cli/laya_lane.py:125 HOLDOUT_EVAL_REPORT_FILE = "holdout_report.json"`
  and `:3207 WORKING / "holdout_report.json"` (inside the generated `HOLDOUT_EVAL_KERNEL_SCRIPT`) duplicate
  `config/results.yaml names.holdout_report`, owned by `Results.leaf` (`src/core/results.py:250`).
  Carry `Results.leaf("holdout_report")` into the lane constant/script.
  Acceptance: `grep -n "holdout_report.json" src/cli/laya_lane.py` → 0 and a test asserts the constant ==
  `Results.leaf("holdout_report")`.
- [ ] **6. `Artifacts` residual literal.** `src/model_tracks/bundle_steps.py:623` `endswith("__vectors.npz")`
  names the declared `vectors` artifact (`config/artifacts.yaml track_artifacts.vectors`); express it through
  `Artifacts.member_name("vectors", track=…)` or a declared suffix.
  Acceptance: `grep -n "__vectors.npz" src/model_tracks/*.py` → 0; `pytest tests/test_artifacts.py -q` green.
  (Out of Artifacts' declared scope, deferred: `graph_tracks/artifacts.name` still spells graph names.)
- [ ] **7. `Dataset` hygiene.** `src/core/dataset.py:415 as_bundle(self, path, role: Any = "inputs")` defaults a
  bare string and types the role `Any`; use `BundleRole | str` and default `BundleRole.inputs`
  (SSOT = `src/core/bundle.py:113`).
  Acceptance: `pytest tests/test_dataset.py -q` green and the default equals `BundleRole.inputs`.
- [ ] **8. Rule-4 stale hash wording** (hashing itself is gone: `grep -rnE "hashlib|sha256|hexdigest" src/` → 0,
  but prose still claims digests): `config/dataset.yaml:15` ("content DIGEST computed on demand"),
  `tests/test_dataset.py:130,148-155,177,184` ("identity: a content digest", "the digest is the bytes' identity"),
  `dataclass.md:15,24,32,104`, and `src/cli/kaggle_datasets.py:327,353` / `src/cli/kaggle_chain.py:55,82`
  ("sha-verified" / "receipt sha") for what is now a byte-size check.
  Acceptance: `grep -rn "content digest\|content DIGEST\|sha-verified\|receipt sha" config/dataset.yaml
  tests/test_dataset.py dataclass.md src/cli/kaggle_{datasets,chain}.py` → 0; `pytest tests/test_dataset.py -q` green.
- [ ] **External (not code; owner action).** `er-laya-holdout` and `er-laya-finetune-ckpt` do not exist on Kaggle
  (dataclass.md §3.7): declared `on_kaggle: false` / `HostedRegistry.pending_on_kaggle()`. Create them (or make
  holdout-eval fail loud) before the holdout/finetune-eval lanes run. Also `fbarulli/er-embed-gpu` (embed kernel)
  does not exist, so the `embeddings` hosted dataset has no attaching kernel yet.

Assumptions (confirm before acting): the 5 classes are otherwise complete (their pinned tests pass);
`ColabSpec` carries no hosted-dataset slug today, so "ColabSpec references the registry" is a no-op until one
appears; `Dataset.validation_size` and the fold-derived validation size are two homes for one concept (item 3
asks the owner to pick one); `smoke` declaring no splits despite `smoke_200/listing_splits.csv` carrying a real
train/dev/test is intentional (data_surface_map.md §8.6).

## Bundle refactor (artifact + process)
- [~] `core/bundle.py` — `Bundle(role∈{inputs,recovery,result})` + `BundlePipeline` **class written**; single boundary verify via `portable_archive.verify_archive` (member names + byte sizes; the old `verify_archive_digest` symbol is gone — see the data-classes section); accessors `checkpoint/checkpoints/track_inventory/track_complete/ablation_templates/ablation_skipped`; public `bundle_spec()` added (lanes no longer import the private `_bundle_spec`).
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
- [x] Regenerate stale fixtures: `smoke_200` and `data/track_setup` carry `cascade.yaml` and no `hybrid`.
- [x] `src/training/run_plan.py` — frozen-input materialization maps bare keys → `*_csv` members (`_FROZEN_INPUT_MEMBERS`); `test_smoke_run_plan` green.
- [x] `src/model_tracks/smoke_inputs.py` — repo-relative fixture paths resolved against `TRAIN_ROOT` before `relative_to`.
- [x] `data/prepared/smoke_500` — regenerated (cascade template present); since BINNED (owner directive 2026-10-08, only `dataset.csv`/`dataset_3k.csv`/`data/prepared/smoke_200` remain).

## Lanes
- [ ] Colab + Kaggle transports load/save `Bundle`; Kaggle finalize kernel is wired, colab finalize is not.
- [ ] Kaggle `embedding_kernel_slug` (`fbarulli/er-embed-gpu`) — **no embed kernel exists on the account**; push one or remove the embed objective.
- [ ] Colab `cohort_label` SSOT (still open after the 2026-10-08 bin): `cli/colab_lane.py` hardcodes the retired `50pct` branch, so a committed `dataset_3k.csv` is labeled by its stem (`dataset_3k`) while Kaggle returns `3k`; needs a declared tag map (e.g. a `colab.*.cohort_tags` config key).
- [ ] `cli/laya_lane.py` kernel-template helper duplication (`log` ×4, `resolve_input` ×4, `sha256_of` ×2): inject shared fragments via the existing token mechanism (artifact-byte-risky; deferred).
- [ ] `_env_dot_value` vs `_env_value` near-duplicates in `cli/`.

## Standalone bundlers (optional, "all surfaces")
- [ ] `graph_tracks/{bundle,worker_package}.py`, `ner/{ner,colab_ner}.py`, `cli/laya_lane.py` dataset payloads — migrate archive/hash to `Bundle`.

## Prepared-layout SSOT (post-audit)
- [x] `core.common.prepared_setup_layout()` — the single layout accessor; all 21 `_setup_layout()` shims delegate to it.
- [x] `PreparationGraphSetupSpec.input_manifest`/`.listings` declared; every producer/consumer filesystem literal repointed at the spec (producer first) — no `input_manifest.json`/`listings.json` paths remain in code.
- [x] Trace batch caps share `core.tracing.TRACE_BATCH_ROWS` / `TRACE_MAX_BATCH_ROWS` (per-stage graph sizes that genuinely differ stay local).
- [x] The per-module `_spec()` wrapper (11 copies in `model_tracks/`) collapsed to direct `core.bundle.bundle_spec()` calls.

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
