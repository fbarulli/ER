# Data-surface map — sets, splits, ownership, validation code

Status: read-only cartography (2026-10-08/09). No code was edited, no git command run,
no test run, no agent spawned. Every claim is a `file:line` reference into the tree as
found at commit `f893037` + the current dirty worktree.

Authority for this map: `.dataclass-brief.md` (owner directives) and `dataclass.md`
(hosted-dataset requirements).

---

## 0. TL;DR

- **Three official datasets** are declared and reachable: `dataset.csv` (root, bound
  `files.dataset`), `dataset_3k.csv` (root, **bound by a code literal only**),
  `data/prepared/smoke_200` (bound twice: `colab.data_bundle.setup_dir` and
  `preparation.smoke_dir`).
- **Two more root exports still exist and are still bound by config**: `dataset_50pct.csv`
  and `dataset_10k.csv` (the "50pct"/"10k" cohorts). Per the brief they are **to be binned**,
  but they are load-bearing today (`kaggle.export_csvs`, `bundle_prep.export_csvs`,
  `KaggleSpec.export_csvs` default, and a **code literal** `_COHORT_EXPORTS`).
- **The splits are real but have no owner.** `Dataset` (`src/core/dataset.py`) owns the
  member files and the `prepared/` + `track_setup/` tree roots, and **has no `splits()` /
  `sets()` method at all**. The split logic lives in `src/training/folds.py` (a free-function
  module, not a class), and the two split *artifacts* (`final_validation.csv`,
  `validation_fold_map.csv`) are declared as dataset **members**.
- **Validation size is NOT flex.** It is pinned at `Literal[4]` folds →
  train/dev/test = 50/25/25 → the scored validation population is **folds 2+3 = 50 % of the
  component graph**. `split.holdout_component_folds` is the single knob and the schema forbids
  any other arity. `build_final_validation.build(n_folds=...)` accepts an override but it does
  **not** re-shape the split (see §5.3).
- **Validation code** is 6 modules + 1 config block; the requested five entry points all exist
  and are mapped in §5.
- Six splits/surfaces are **unowned** (§7), and the owner appears to be **missing** at least
  three surfaces (§8).

---

## 1. The three official datasets — declaration, binding, consumers

### 1.1 `dataset.csv` — the ONE original / source export

| | |
|---|---|
| File | `/home/opc/ONE/ER/dataset.csv` (54,787,809 B, 2026-10-08) |
| Declared (binding) | `config/paths.yaml:99` → `dataset: "repo:dataset.csv"` |
| Declared (member) | `config/dataset.yaml:31` → `source: {via: files, key: dataset}` |
| Resolved by | `src/core/common.py:808` `F = {...}` → `:809` `DATA_PATH = F["dataset"]` |
| Read spec | `config/paths.yaml:414-417` `dataset_csv_read` (dtype str, keep_default_na, na_filter) |
| Class accessor | `src/core/dataset.py:246` `Dataset.load_source()`; `:224` `Dataset.load(SOURCE)` |
| Read helper | `src/core/common.py:1298` `_load_source_export` → `:1316` `load_dataset()` → `:1356` `load_raw_export()` |
| Identity | `src/core/dataset.py:267` `Dataset.identity()` (sha256 over `source` + sorted members) |

Consumers (who actually reads the raw export):

| Consumer | file:line | How |
|---|---|---|
| Column projection / loader | `src/core/common.py:1316` `load_dataset`, `:1337`, `:1356` `load_raw_export` | `DATA_PATH` |
| `build_reference` (brand reference) | `src/training/build_reference.py:117` | `load_dataset(columns=["brand"])` |
| `dedupe` | `src/training/dedupe.py:262` | `apply_identity_links(load_dataset())` |
| `zero_shot_sims` | `src/training/zero_shot_sims.py:343` (manifest input), `:387` | `F["dataset"]`, `load_dataset()` |
| `data_quality_audit` | `src/training/data_quality_audit.py:20,82,162` | `DATA_PATH` |
| `build_title_attribute_evidence` | `src/training/build_title_attribute_evidence.py:29,45,148` | `DATA_PATH` |
| `attribute_universe` | `src/core/attribute_universe.py:296` | `core.common.load_dataset()` |
| `complete_colab_worker` | `src/training/complete_colab_worker.py:17,78` | `DATA_PATH` (row census) |
| `prepare_all` (bundle member + sparse checkout) | `src/training/prepare_all.py:162,179,187` | `DATA_PATH` |
| Colab lane dataset upload | `src/cli/colab.py:1574-1576`, `src/cli/colab_lane.py:180-183` | `DATA_PATH` |
| Kaggle lane packaging | `src/cli/kaggle_cli.py:210`, `src/cli/kaggle_outputs.py:119` | `F["dataset"]` |
| Cohort detection | `src/core/common.py:308-361` (`mounted_cohort`, `_COHORT_EXPORTS`) | byte-compare vs staged exports |
| `sample_dataset_10k` (stages 10k) | `scripts/sample_dataset_10k.py:26` | `DATA_PATH` |
| audits / scripts | `scripts/audit_identity_context.py:18,98`, `scripts/render_gtin_repair_results.py:6,12` | `DATA_PATH` |
| selftest | `src/training/selftest.py:1130,1376` | `load_dataset()` |
| laya corpus builder | `scripts/laya_build_dataset.py` (via `data/prepared/full/worker_1_baseline.pkl.gz`, §3 set S9) | indirect |
| `Dataset` class | `src/core/dataset.py:246` | `load_source()` |
| Delivery member list | `src/cli/colab_lane_contracts.py` (`DELIVERY_DATA_MEMBERS` is the derived CSVs; `dataset.csv` rides `runtime_inputs.checkout_members`, `src/core/runtime_inputs.py:45`) | `git ls-files` staging |

### 1.2 `dataset_3k.csv` — the 3k set

| | |
|---|---|
| File | `/home/opc/ONE/ER/dataset_3k.csv` (2,188,027 B, 2026-10-07) |
| Declared (binding) | **NONE — no `config/*.yaml` key names it.** |
| Declared (code literal) | `scripts/ablation_timing_3k.py:65` `CATALOG = ROOT / 'dataset_3k.csv'` |
| Producer | **No producer in tree.** Not produced by any `src/` or `scripts/` module (only `sample_dataset_10k.py` exists for the 10k set). Provenance is unrecorded. |
| Class accessor | none |
| Identity | none (not in `Dataset.members`) |

Consumers:

| Consumer | file:line | Note |
|---|---|---|
| `scripts/ablation_timing_3k.py` | `:65` (path), `:114-155` (dev pairs from the catalog), `:73-74` (recorded note), `:277` (`cohort: '3k'`) | The ONLY reader. Its outputs: `artifacts/abl_opt/rounds/round*/…` |
| recorded evidence | `artifacts/abl_opt/rounds/round100/summary.json:4` | frozen `notes` string |

**This is the thinnest surface in the project**: an official dataset with zero config
declaration, zero producer, zero class ownership, and exactly one consumer that reaches it
by a path literal. Flagged in §7 item 6.

### 1.3 `data/prepared/smoke_200` — the smoke set

| | |
|---|---|
| Dir | `/home/opc/ONE/ER/data/prepared/smoke_200/` (23 entries; `eligible_catalog.csv`, `listing_splits.csv` 115/50/45 train/dev/test, `listing_pairs.csv`, `prepared/`, `suite.yaml`, per-track `*.yaml`, `text_prepared.pkl.gz`, `shared_minilm__embeddings.npz`, `setup_manifest.json`, `graph_census.json`, `shared_training_*.json`, `ablation_*`) |
| Declared (binding #1) | `config/training.yaml:435` `colab.data_bundle.setup_dir: "data/prepared/smoke_200"`; `:436` `suite_config` |
| Declared (binding #2) | `config/training.yaml:1388` `preparation.smoke_dir: data/prepared/smoke_200` (schema default `src/core/schemas.py:3678`) |
| Declared (suite matrix) | `src/core/schemas.py:5407-5409` `SuiteMatrixEntry(size="S", name="smoke_200", suite_config="data/prepared/smoke_200/suite.yaml", device="cpu")` |
| **Undeclared duplicate (code literal)** | `src/cli/colab_lane_cpu_provision.py:58` `TRAIN_ROOT / "data/prepared/smoke_200"` — a hardcoded path that duplicates the two yaml keys |
| Track config | `config/model_tracks.yaml:3-4` (that is the `full` suite; smoke uses its own `data/prepared/smoke_200/suite.yaml`) |

Consumers:

| Consumer | file:line | How |
|---|---|---|
| Preparation run | `src/training/prepare_all.py:422` (`smoke=root / prep.smoke_dir`), `:830` | `preparation.smoke_dir` |
| Handoff | `src/training/handoff.py:464-465,486-488,500-501` | `smoke_dir` |
| Runtime inputs (git staging) | `src/core/runtime_inputs.py:56` `training_cfg().preparation.smoke_dir` | |
| Colab data bundle | `src/cli/colab.py:282,331,2207` (`suite.setup_dir`); `src/cli/colab.py:768-771,1304` | `colab.data_bundle.setup_dir` |
| Colab CPU provision | `src/cli/colab_lane_cpu_provision.py:58` | **code literal** |
| Lane delivery | `src/cli/colab_lane_contracts.py:35` `DELIVERY_PREPARED_DIRS = ("full", "smoke_200")` | literal tuple |
| Kaggle kernels | `src/cli/kaggle_kernels.py:305-310` (checkout members) | `kaggle.checkout_paths` |
| Graph/embedding prepare | `src/training/prepare_embeddings.py:192-202` (`graph_tracks.setup.default_setup_dir`) | suite `setup_dir` |
| Graph tracks worker scripts | `scripts/training_profile.py:20,71`, `scripts/smoke_graph_tracks.py` | literals |
| Tests | `tests/test_suite_matrix_defaults.py:26-84`, `tests/test_cascade_fixtures.py:96-129`, `tests/test_colab_spec_contract.py:96-229`, `tests/test_training_handoff.py:65`, `tests/test_prepare_all.py:55`, `tests/test_cohort_coverage_contract.py:4,31`, `tests/test_traceability_finalize.py:692-744`, `tests/test_invariants.py:772` | literals |
| Graph tracks setup/prepare | `src/graph_tracks/setup.py:749-756` `default_setup_dir()` reads the suite `setup_dir` | indirect |

---

## 2. EXACT deletion list (the binned surface)

Each entry: **path** → *yaml keys that bind it* → *code refs that must change with it*.

### 2.1 `data/prepared/smoke_500/` — whole tree

- **yaml keys binding it: NONE.** No config key declares `smoke_500`. It is bound only by its
  own generated files' contents (`data/prepared/smoke_500/suite.yaml:1-2`,
  `gnn_only.yaml:12-19`, `cascade.yaml:12-19`).
- Code / doc refs to fix:
  - `tests/test_cascade_fixtures.py:24` `SMOKE_FIXTURES = ("data/prepared/smoke_200", "data/prepared/smoke_500")`
  - `src/core/text.py:43` — comment "MEASURED (269,867 chars from the smoke_500 catalog…)"
  - `TODO.md:20,23` — done-items referencing regeneration
  - `results/kaggle_lane/train_fetch/failure.manifest.json:302-334` (generated evidence)
  - `results/kaggle_lane/{bundle,train,embed}_kernel/*.py` `_runtime_files` arrays
    (generated kernel snapshots; they still enumerate `smoke_500/*` and
    `smoke_200__clean_shared_inputs/*` → **re-bake required after deletion**)
- **Verdict: safe to delete** (nothing in `src/` reads it; only tests + generated results).

### 2.2 `data/prepared/smoke_200__clean_shared_inputs/`

- **yaml keys binding it: NONE.**
- Producer: `src/model_tracks/shared_graph_data.py:27`
  `CLEAN_BACKUP_SUFFIX = '__clean_shared_inputs'`, written at `:137-146`
  (`backup_root = setup.parent / (setup.name + CLEAN_BACKUP_SUFFIX)`).
- **Reader (this is the problem):** `src/model_tracks/shared_graph_data.py:147-160` reads the
  backup back as the **immutable clean catalog/listings** every projection rebuild starts from;
  `:252` reads `pairs.csv` and `:295-300` the `pair_lineage.json` from the same backup.
- Test consumers: `tests/test_cohort_coverage_contract.py:4,31`;
  `tests/test_traceability_finalize.py:694-695`.
- **Verdict: owner-authorized, but NOT a pure `rm -rf`.** Deleting it removes the clean-input
  backup the `_shared_graph_data` rebuild depends on. Either keep it (it is a *backup*, not a
  "set") or remove the backup/replay path with it. **Delete-with-code-change** (see §7).

### 2.3 `data/track_setup__clean_shared_inputs/`

- Same producer (`shared_graph_data.py:137`) with `setup = data/track_setup`; same reader
  (`:147-160`), same **delete-with-code-change** caveat. yaml keys: **none**.

### 2.4 `dataset_50pct.csv` (root; 26,642,199 B)

- **yaml keys binding it:**
  - `config/training.yaml:276` `kaggle.export_csvs: ["dataset.csv", "dataset_50pct.csv", "dataset_10k.csv"]`
  - `config/training.yaml:343` `bundle_prep.export_csvs: [same three]`
  - `config/training.yaml:180` `kaggle.cohort_tags: ["full", "50pct", "10k"]` (length-checked
    against `export_csvs` at `src/core/schemas.py:3497-3500`)
  - schema defaults: `src/core/schemas.py:3387` (`cohort_tags`) and `:3399-3401`
    (`KaggleSpec.export_csvs`), mirrored by `TrainingConfig` default factory
- **Code literal (SSOT violation):** `src/core/common.py:339-342` `_COHORT_EXPORTS =
  (("dataset_50pct.csv","50pct"), ("dataset_10k.csv","10k"))`, consumed by
  `:345 mounted_cohort()` → `:326 dataset_is_partial_cohort()`.
  **Deleting the CSV without editing this tuple leaves a dead oracle-hook.**
- Code that reads it: `src/cli/kaggle_runtime.py:60-84` (`cohort_label`, `cohort_export_csv`),
  `src/cli/kaggle_datasets.py:24-37,49`, `src/cli/colab_bundle.py:97`,
  `src/cli/colab_lane_cpu_provision.py:40-51` (`_committed_export_name`),
  `src/cli/kaggle_kernels.py:271-324`, `src/cli/kaggle_kernel_templates.py:169-180`.

### 2.5 `dataset_10k.csv` + `dataset_10k.coverage.json` (root)

- Same yaml keys as §2.4, plus:
  - `config/training.yaml:32-37` `masking.balanced_augmentation.cohort_counts.10k`
    (`minted_negatives: 12000`, `masked_minted_negatives: 6000`, `masked_positives: 1200`,
    `vendor_variation_positives: 287`) — consumed at `src/training/train.py:1260-1264`;
    becomes dead config when the 10k cohort is gone.
  - `kaggle.export_csvs` (`training.yaml:276`, `:343`), `kaggle.cohort_tags` (`:180`)
  - `src/core/common.py:341` `("dataset_10k.csv", "10k")`
- Producer: `scripts/sample_dataset_10k.py` (writes both files, `:11-13`, `--output` default
  `TRAIN_ROOT / 'dataset_10k.csv'`).

### 2.6 Other dead / retired set bindings found (bin or repair)

| Surface | Bound by | State |
|---|---|---|
| `data/dataset_deduped_smoke_128.csv` | `config/training.yaml:465` `colab.smoke_dataset_csv` | **File does not exist.** Dead binding; comment at `:463-464` says "Retained legacy schema binding". |
| `data/prepared/smoke_1000/worker_{1,2}_baseline.pkl.gz` | `src/cli/colab.py:2758-2759` (**code literal**, not yaml) | **Dir does not exist.** Dead path on a `sample == 1000` branch. |
| retired `dataset_deduped_sample_3000` / `sample_5000` lanes | referenced only in prose: `src/training/build_final_validation.py:3`, `config/paths.yaml:111-114` | yaml bindings already removed. The only leftover "3000" is `config/paths.yaml:320-324` `layouts.balanced_pairs_sample` template `balanced_pairs_sample_3000.csv` — that is the **balanced-pairs** sample (`owner: training.sample_balanced_pairs`), a different artifact. Do not confuse the two. |
| `data/validation/stratified_holdout.csv` | `scripts/build_stratified_holdout.py:93` (default `--out`, code literal) | Dir `data/validation/` **does not exist**. Legacy producer. |
| `data/validation/slice_review_sample.csv` | `scripts/build_validation_slice_sample.py:55` (code literal) | Same. Outputs go to `data/validation/`; dir absent. |
| `data/prepared/full/` | `config/training.yaml:461-462` `colab.full_prepared_bundles`; `layout` `paths.yaml:390-394` is `data/prepared` (parent) | **This is the "full" set — KEEP, do not bin** (see §8 item 1 for the naming conflict). |
| `data/track_setup/` | `config/model_tracks.yaml:3-4`; `layouts.dataset_track_setup` `config/paths.yaml:395-399`; declared `config/dataset.yaml:44` | **KEEP** — the declared `Dataset.layout.track_setup` tree root, and the `full` suite's `setup_dir`. |
| `data/prepared/smoke_200/ablation_cohort/`, `ablation_settings.yaml`, `ablation_templates/` | untracked additions to smoke_200 | Part of the kept smoke set. |
| `model_tracks_package.json` (root) | references `data/model_tracks/shared__clean_shared_inputs/*` (`:339-343`) | `data/model_tracks/` **does not exist**. Generated package manifest; stale. |

**Not on the deletion list (derived members of the official source, declared in
`config/dataset.yaml`):** `data/dataset_deduped.csv`, `data/canonical_records.csv`,
`data/gate_results.csv`, `data/labeled_pairs.csv`, `data/final_validation.csv`,
`data/number_tokens_reference.csv`, `data/sku_to_rep.csv` — plus the generated
`results/training/validation_fold_map.csv`. These are the pipeline's derived artifacts, not
"sets". If the owner really means "only the three datasets exist", these members must be
re-derived, not deleted.

---

## 3. SET inventory (every set / cohort that exists)

| # | Set | Path(s) | Bound by | Producer | State |
|---|---|---|---|---|---|
| S1 | source / full | `dataset.csv` | `files.dataset` `paths.yaml:99`; `dataset.yaml:31` | external (owner export) | **OFFICIAL** |
| S2 | 3k | `dataset_3k.csv` | code literal `scripts/ablation_timing_3k.py:65` | unknown | **OFFICIAL** |
| S3 | smoke | `data/prepared/smoke_200/` | `training.yaml:435,1388`; `schemas.py:5407`; literal `colab_lane_cpu_provision.py:58` | `graph_tracks.setup` + `prepare_all` | **OFFICIAL** |
| S4 | 50pct cohort | `dataset_50pct.csv` | `training.yaml:276,343,180`; `schemas.py:3399`; literal `common.py:340` | unknown (staged export) | BIN |
| S5 | 10k cohort | `dataset_10k.csv` + `.coverage.json` | same as S4 + `training.yaml:32-37`; literal `common.py:341` | `scripts/sample_dataset_10k.py` | BIN |
| S6 | smoke_500 | `data/prepared/smoke_500/` | **none** | `graph_tracks.setup` (old run) | BIN |
| S7 | smoke_200 clean backup | `data/prepared/smoke_200__clean_shared_inputs/` | **none** (code const `shared_graph_data.py:27`) | `shared_graph_data.py:137` | BIN (with code change) |
| S8 | track_setup clean backup | `data/track_setup__clean_shared_inputs/` | **none** (same const) | same | BIN (with code change) |
| S9 | full prepared text bundle | `data/prepared/full/worker_*_baseline.pkl.gz` | `training.yaml:461-462` `colab.full_prepared_bundles` (`schemas.py:2857`) | `prepare_all` / `prepared_bundle` | KEEP |
| S10 | track_setup (full suite setup) | `data/track_setup/` | `model_tracks.yaml:3-4`; `paths.yaml:395-399`; `dataset.yaml:44` | `graph_tracks.setup.setup()` | KEEP |
| S11 | laya corpus | `data/laya/{train,dev,test}.jsonl`, `unknown_pairs.csv`, `receipt.json` | `laya_config.py:178` `finetune_corpus_dir: "data/laya"` (LayaSpec default; **no `laya:` block in training.yaml**) | `scripts/laya_build_dataset.py` | KEEP (official corpus) |
| S12 | laya holdout | `data/laya/holdout.csv` + `holdout.receipt.json` | `laya_config.py:216` `holdout_csv: "data/laya/holdout.csv"` | `scripts/laya_holdout.py:36` | KEEP |
| S13 | smoke_1000 | `data/prepared/smoke_1000/` | code literal `src/cli/colab.py:2758-2759` | n/a | DEAD (absent) |
| S14 | smoke_128 (deduped) | `data/dataset_deduped_smoke_128.csv` | `training.yaml:465` | n/a | DEAD (absent) |

Suite matrix (canonical S/M/L labels, `src/core/schemas.py:5401-5415`):
`S = smoke_200 (cpu)`, `M = 50pct (cpu)`, `L = full (cuda, `config/model_tracks.yaml`)`.
**Note:** the M entry (`50pct`) has `suite_config=None` and points at the to-be-binned cohort.

---

## 4. SPLIT inventory (every split, end to end)

Legend — **Owner**: `folds-first` = `src/training/folds.py` free functions; `Dataset` =
`src/core/dataset.py` (intended owner per `.dataclass-brief.md`); **UNOWNED** = no class owns it.

### 4.A laya corpus splits — train / dev / test ("laya")

| | |
|---|---|
| Declared (config key) | `src/core/laya_config.py:178` `finetune_corpus_dir: "data/laya"`; `:206` `finetune_eval_split: Literal["train","dev","test"] = "test"`; `:184-188` members `train.jsonl, dev.jsonl, test.jsonl`. **No `laya:` block exists in `config/training.yaml`** → all values are LayaSpec defaults. Ratios are NOT declared in yaml: `scripts/laya_build_dataset.py:453 _split_ratios()` derives them from the listing-pair proportions. |
| Realized sizes | train 7,551 / dev 3,477 / test 3,280 rows (`data/laya/receipt.json` `split_ratios` 0.5278/0.2431/0.2292, `split_sizes`) |
| Produced | `scripts/laya_build_dataset.py:880-1068` (`_assign_splits` `:460`, per-population split loops `:915-1057`, emit `:1059-1068`); holdout by `scripts/laya_holdout.py:36` |
| Consumed | `src/cli/laya_lane.py:93-106` (`FINETUNE_CORPUS_FILES`), `:2181` (`read_jsonl(test_path)`), `:2484`, `:3507-3510` (`finetune_eval_split` validation), `:3970-4010`, `:4188`; `scripts/laya_compare.py:31`; `scripts/laya_metrics_pairs.py`, `scripts/laya_play_sample.py` |
| Owner | **UNOWNED.** Ratios are re-derived from data, not declared; nothing binds `train/dev/test.jsonl` to a yaml key (they are string-joined at `laya_lane.py:101`). |

### 4.B training holdout split — train / dev / test ("full" lane)

| | |
|---|---|
| Declared | `config/training.yaml:693-730` `split:` block. `mode: "holdout"` `:694`, `train_fraction: 0.50` `:695`, `dev_fraction: 0.25` `:696`, `calibration_dev_fraction: 0.50` `:699`, `calibration_seed_offset: 17` `:701`, `holdout_component_folds: 4` `:702` (**`Literal[4]`**, `src/core/schemas.py:793`), `test_fraction: 0.25` `:706`, `negative_fold_policy: "train_side"` `:730`. Schema: `src/core/schemas.py:759-870` (`_shares_sum_to_one` `:812-821`, `_mode_matches_fraction_contract` `:823-870`). |
| Semantics | component-aware; `train = quarters[:-2]`, `dev = quarters[-2]`, `test = quarters[-1]` (`src/training/folds.py:506-516`) |
| Produced (SSOT) | `src/training/folds.py:547 derive_holdout()` → `:458 HoldoutContract` → `:415 component_folds()` → `:362 ComponentIndex`. Graph is built **inside** the entry point: `:605 merged_component_graph()` (training positives ∪ `data/labeled_pairs.csv` positives; **P0 leak fix**) |
| Consumed | `src/training/train.py:1250` (freezes `frozen_holdout` `:1251-1252`), `:1628` (reads back), `:1642` (CV branch); `src/training/build_final_validation.py:1254`; `src/graph_tracks/setup.py:423` (→ `listing_splits.csv`); `src/training/prepared_bundle.py:229-232` (`prepared_holdout`, frozen-parent reuse); `src/training/hpo.py:284` (`_grid_folds`, CV only); `scripts/sid_graph_eval.py:133`, `scripts/sid_hybrid_eval.py:167`, `scripts/sid_phase0_report.py:179` |
| Guard | `src/training/selftest.py:2646-2760` bans direct `holdout_split` / `partition_component_pairs` calls outside `folds.py` |
| Owner | **`folds-first`, not a class.** `Dataset` does not expose it. |

### 4.C validation population — the scored half (folds 2+3) ⟵ THE validation artifact

| | |
|---|---|
| Declared (artifact binding) | `config/paths.yaml:115` `final_validation: "training_data:final_validation.csv"`; `:116-120` `validation_fold_map: "results_training:validation_fold_map.csv"` |
| Declared (member) | `config/dataset.yaml:39` `final_validation: {via: files, key: final_validation}` |
| Declared (stage) | `config/paths.yaml:80` `validation: [final_validation]` (trace-stage join); `config/training.yaml:1413-1414` `preparation.reusable_keys` includes `final_validation`, `validation_fold_map` |
| Produced | `src/training/build_final_validation.py:1221 build()` → emit `:1291-1300`; fold→quarter map `:1100 _resolve_quarter_folds()`; manifest `:1118 write_manifest()` (also writes `results/manifests/final_validation.json`); fold map `:1081 _fold_map()`. Entrypoint `:1465 main()`; registered as stage `'validation'` in `src/training/prepare_all.py:95`. |
| Size | `fold_of[bc] = 0` for train, `n_folds-2` for dev, `n_folds-1` for test (`:1108-1115`) ⇒ **validation = folds 2+3 = 1/2 of the component graph.** Live realized: 6,351 pairs (565 pos / 5,786 neg) over 14,946 entities (7,494 in folds 2+3), per `src/cli/colab.py:225-230` reconciliation comment. |
| Consumed | `src/training/evaluate_models.py:175-220` (DEV=fold 2, TEST=fold 3, derived from `holdout_component_folds`); `src/training/complete_colab_worker.py:86-105,170-180,273-276` (`--validation-input` default `F["final_validation"]`); `src/cli/colab.py:219,353,392,768-771,1304`; `src/cli/colab_validation_upload.py:32`; `src/core/laya_config.py:146-147` (`decision_csv_bindings` identity + laya-cli-eval → `final_validation`); `dashboard/decision_reports.py:631,689`; `dashboard/app.py:333`; `src/core/schemas.py:2505`; tests `tests/test_scored_half_decisions.py` |
| Owner | **UNOWNED as a split.** Owned only as a *file member* by `Dataset` (`dataset.yaml:39`). The split rule lives in `folds.py` + `build_final_validation.py`. |

### 4.D calibration carve (fit / reserved) — a DEV sub-split

| | |
|---|---|
| Declared | `config/training.yaml:699` `calibration_dev_fraction: 0.50`, `:701` `calibration_seed_offset: 17` |
| Produced | `src/training/folds.py:703 derive_calibration_carve()` → `:748 CalibrationReservation.pools()` (`:943`); seed math `:692 calibration_seed()` / `:781` |
| Consumed | `src/training/training.py:5318-5329` (per-fold carve; sample mode skips it, `:5304-5316`) |
| Realized roles | `src/training/folds.py:669-689` `SPLIT_ROLES = ("train","calibration_fit","calibration_reserved","test")` + `EXTERNAL_EVAL_ROLE`; the module docstring states the config's "train/dev/test" is really **four** populations |
| Owner | **`folds-first`.** Not persisted (re-derived each run) — `folds.py:721-723` says so explicitly. |

### 4.E external eval split (labeled_pairs DEV/TEST)

| | |
|---|---|
| Declared | `config/training.yaml:751-753` `evaluation.component_split_k: 2`, `dev_fold: 0`, `test_fold: 1` — **explicitly LEGACY/not selectors** (`src/training/evaluate_models.py:169-174`) |
| Real selector | folds 2/3 of `validation_fold_map.csv` (`src/training/evaluate_models.py:175-230`, `_N_FOLDS = holdout_component_folds`) |
| Produced/consumed | `src/training/evaluate_models.py:96-300` (reads `labeled_pairs` + `embedding_similarities` + `canonical_records` + `final_validation` + `validation_fold_map`) |
| Owner | **UNOWNED** — the deprecated `component_split_k/dev_fold/test_fold` keys still validate at config load but steer nothing. |

### 4.F CV folds

| | |
|---|---|
| Declared | `config/training.yaml:708` `cv_folds: 5`; schema `src/core/schemas.py:796`; CV envelope pinned 1.0/0.0/0.0 (`schemas.py:839-852`) |
| Produced | two independent implementations: (a) `src/core/common.py:1529 kfold_gtins()` (gtin-level, used by `src/training/training.py:5018`), (b) `src/training/folds.py:415 component_folds()` (component-level, used by `src/training/train.py:1642`, `src/training/hpo.py:284`) |
| Consumed | `src/training/training.py:5001-5020` (`folds_override` contract); `src/training/train.py:1642-1644`; `src/training/hpo.py:272-284` |
| Owner | **UNOWNED, and there are TWO implementations** (`kfold_gtins` vs `component_folds`) — a live divergence risk. |

### 4.G robust-validation folds (report lane)

| | |
|---|---|
| Declared | `config/training.yaml:807-818` `evaluation.robust_validation:` `enabled`, `n_folds: 5`, `repeats: 3`, `seed: 42`, `min_slice_size: 25`, `min_test_negatives: 5`, `max_split_attempts: 100`, `dimensions: [brand,category,attribute]`, `operating_thresholds` |
| Produced | `src/training/robust_validation.py:394 run_robust_validation()` (own `DisjointSet` folds over SKU/GTIN components; `_stable_seed` `:76`) |
| Consumed | `src/training/generate_training_report.py:370-381`, `:976`, `:1690`; `src/training/train.py:2734-2768` |
| Owner | **UNOWNED.** `min_test_negatives` is also re-used by `build_final_validation.py:1280-1285` as the emit guard threshold — a cross-module coupling with no single owner. |

### 4.H HPO selection folds

| | |
|---|---|
| Declared | `config/training.yaml:1247` `hpo.calibration_folds: 3`; `:1254-1256` `hpo.objective.{holdout,cv}: rand_index_proxy`; `:1260` `selection_skip_test_eval: true` |
| Produced/consumed | `src/training/hpo.py:272-284 _grid_folds()` (`component_folds`), `src/training/training.py:4886-5191` (holdout/CV dev + leak guards `:5169-5191`) |
| Owner | **UNOWNED** |

### 4.I graph-track listing splits (frozen into the prepared setup)

| | |
|---|---|
| Declared | `config/training.yaml:1417` `preparation.graph_setup.splits: listing_splits.csv`; loaded as `PreparationGraphSetupSpec` (`src/core/schemas.py:3541`) |
| Produced | `src/graph_tracks/setup.py:403 setup()` → `:423-427` `derive_holdout` + `listing_contract`, `src/graph_tracks/setup.py:92 splits.to_csv(Path(output)/layout.splits)`. Also `src/model_tracks/shared_graph_data.py:270-271` rewrites splits for the shared projection. |
| Schema | `src/graph_tracks/data.py:66 SPLITS = {"train","dev","test"}`; `src/graph_tracks/prepare.py:68 ListingSplits` |
| Realized in fixtures | `data/prepared/smoke_200/listing_splits.csv` 115/50/45; `data/track_setup/listing_splits.csv` 2,052/1,003/999 |
| Consumed | `src/graph_tracks/prepare.py:100-165`, `src/graph_tracks/train.py:134`, `src/graph_tracks/report_slices.py:191-283`, `src/model_tracks/preflight.py:229`, `src/model_tracks/data_gate.py:155`, `src/model_tracks/smoke_inputs.py:99`, `src/cli/colab.py:289,334`, `src/graph_tracks/report_attributes.py:47-60` |
| Owner | **UNOWNED as a split.** The `listing_splits.csv` artifact is addressed by the `preparation.graph_setup.splits` name, but no class owns "the train/dev/test of the graph lane". |

### 4.J rand-matching truth splits (calibration + holdout)

| | |
|---|---|
| Declared | `config/training.yaml:851-864` `rand_matching.truth_splits:` `output_dir: "results/rand_truth"`, `calibration_output: "calibration_truth.csv"`, `holdout_output: "holdout_truth.csv"`, **`sample_size: 12986`**, **`calibration_size: 8657`**, `calibration_folds: 3`, `seed: 42`; also `:865-873` `stratum_sweep` (`identities_per_status: 600`, `skus_per_identity: 2`, `calibration_folds: 3`) |
| Schema guards | `src/core/schemas.py:1214-1245` (sizes must reserve ≥3 holdout rows, etc.) |
| Produced | `src/training/generate_rand_truth.py:186 generate_truth_splits()` → `:118 _split_truths()` (`:172` `calibration_fold` = position % `calibration_folds`); CLI `:210-240`. Stratum sweep: `src/training/generate_rand_stratum_sweep.py:47,144` |
| Consumed | `src/training/rand_matching.py:24-25` (`--calibration-input`/`--holdout-input`), `:2056-2266` truth lookup/disagreements; `scripts/sid_graph_eval.py:65` (reads `results/rand_truth/gtin_stratum_sweep.csv`) |
| Owner | **UNOWNED.** The only split whose *size* is declared as fixed counts (`sample_size`, `calibration_size`) — the opposite of the owner's "FLEX" directive. |

### 4.K NER splits (legacy lane inside the training SSOT)

| | |
|---|---|
| Declared | `config/training.yaml:1477-1479` `ner.semantic_training.{TRAIN_FRAC:0.50, VALIDATION_FRAC:0.25, HOLDOUT_FRAC:0.25}`; `:1515-1517` `ner.ner_training.{train_frac:0.50, validation_frac:0.25, holdout_frac:0.25}` |
| Produced | `src/ner/ner.py:206 split_records()` (RNG shuffle + slice, `:217-245`), called at `:840-841`; config `NERTrainingConfig:76-78` |
| Consumed | `src/ner/ner.py` training loop; `src/ner/colab_ner.py` |
| Owner | **UNOWNED** (dict section `TrainingConfig.ner: dict[str, Any]` `src/core/schemas.py:3781` — shape-unvalidated) |

### 4.L per-fold early-stopping carve

| | |
|---|---|
| Declared | `config/training.yaml:1119` `training.dev_fraction: 0.15` ("dev share OF THE TRAIN side") |
| Produced/consumed | `src/training/training.py:5146-5151` (`dev_override` ∩ train side), `:5169-5191` (leak asserts) |
| Owner | **UNOWNED** |

### 4.M post-training held-out inference helpers

| | |
|---|---|
| Module | `src/training/validation_inference.py` (39 lines): `resolve_best_checkpoint()` `:14`, `threshold_assignment_metrics()` `:27`. No split; consumes a checkpoint + scores. |
| Layout | `config/paths.yaml:248-252` `layouts.final_inference` (owner `training.validation_inference`); `config/training.yaml:589-606` `colab.final_inference` |
| Owner | `training.validation_inference` (module-level, not a class). |

---

## 5. The validation code — actual entrypoints

### 5.1 Requested files, confirmed

| Module | Lines | Role | Entrypoints |
|---|---|---|---|
| `src/training/build_final_validation.py` | 1,488 | **The validation artifact owner.** Emits `final_validation.csv` + `validation_fold_map.csv` + `results/manifests/final_validation.json` | `main()` `:1465`; `build()` `:1221`; `write_manifest()` `:1118`; classes `FoldResolver:735`, `ValidationRowAssembler:765`, `LeakGuards:917`, `NegativeFoldPolicy:483`, `SliceFieldGrid:337`, `SliceBagTokenizer:245` |
| `src/training/validation_inference.py` | 39 | Post-training held-out SKU inference helpers (checkpoint resolve + threshold coverage) | `resolve_best_checkpoint()` `:14`; `threshold_assignment_metrics()` `:27` |
| `src/training/folds.py` | 1,019 | **THE split SSOT** (component folds, holdout contract, calibration carve) | `derive_holdout()` `:547`; `holdout_split()` `:519` (banned outside this module, `selftest.py:2662`); `component_folds()` `:415`; `component_ids()` `:439`; `merged_component_graph()` `:605`; `derive_calibration_carve()` `:703`; `partition_component_pairs()` `:987`; `SPLIT_ROLES` `:669` |
| `src/training/robust_validation.py` | 676 | Repeated leakage-aware folds over the scored pair dump (report only) | `run_robust_validation()` `:394` |
| `src/core/bootstrap_ci.py` | 161 | Paired bootstrap CIs over the scored pairs. **`split` is a LABEL, not a selector** | `paired_bootstrap(labels, scores, *, metrics, track, split, spec)` `:65`; `split` only echoed into the result dict `:113,124,157` |
| `src/core/common.py` | 1,557 | The split-SSOT **loader** (not splitter) | `_load_config_cached()` `:262` (the deep-merge/validate reader); `load_config()` `:278`; `config_section()` `:286`; `data_cfg()` `:303`; `training_cfg()` `:399`; `F`/`artifact` `:808`/`:870`; `load_dataset()` `:1316`; `kfold_gtins()` `:1529` |

### 5.2 Every `split` / `validation` / `holdout` / `fold` declaration in `config/*.yaml`

| File:line | Key | Meaning |
|---|---|---|
| `training.yaml:693-730` | `split:` (`mode`, `train_fraction`, `dev_fraction`, `calibration_dev_fraction`, `calibration_seed_offset`, `holdout_component_folds`, `test_fraction`, `fixed_threshold`, `cv_folds`, `negative_fold_policy`) | THE training split SSOT |
| `training.yaml:739` | `evaluation:` block | |
| `training.yaml:751-753` | `evaluation.component_split_k`, `dev_fold`, `test_fold` | **LEGACY, steers nothing** |
| `training.yaml:782-788` | `evaluation.generalization_slices.observed_split: train` | slice semantics; consumed `src/graph_tracks/report_slices.py:191-196` |
| `training.yaml:793-797` | `evaluation.paired_bootstrap` (`resamples`,`seed`,`confidence`) | |
| `training.yaml:807-818` | `evaluation.robust_validation` | folds + thin-cell floor |
| `training.yaml:851-873` | `rand_matching.truth_splits` / `stratum_sweep` | truth calibration/holdout |
| `training.yaml:1119` | `training.dev_fraction` | per-fold ES carve of the train side |
| `training.yaml:1247` | `hpo.calibration_folds` | |
| `training.yaml:1254-1256` | `hpo.objective.{holdout,cv}` | |
| `training.yaml:1260` | `hpo.selection_skip_test_eval` | leak guard |
| `training.yaml:1290-1298` | `calibration_sweep` (**disabled**) | TIER-3 sweep, `enabled: false` |
| `training.yaml:1413-1417` | `preparation.reusable_keys` (`final_validation`,`validation_fold_map`), `graph_setup.splits` | |
| `training.yaml:1477-1479, 1515-1517` | `ner.*.{TRAIN_FRAC,VALIDATION_FRAC,HOLDOUT_FRAC}`, `{train,validation,holdout}_frac` | |
| `paths.yaml:80` | `orchestration_stages.validation: [final_validation]` | |
| `paths.yaml:111-120` | `files.final_validation`, `files.validation_fold_map` | |
| `paths.yaml:125` | `files.fold_metrics: "results:train_fold_metrics.csv"` | worker-scoped pointer |
| `paths.yaml:263-267` | `layouts.checkpoint_repo` template `..._f{fold}` | |
| `config/attribute_ablation.yaml:6` | `split: dev` | ablation lane reads the dev split |
| (no `laya:` block in `training.yaml`) | `LayaSpec` defaults | `finetune_eval_split`, `holdout_csv` |

### 5.3 WHERE VALIDATION SIZE IS SET — and why it is not flex

The validation size is **derived, never configured as a count**:

```
holdout_component_folds = 4            config/training.yaml:702   (Literal[4], schemas.py:793)
   ↓ derive_holdout -> HoldoutContract.cut()
dev  = quarters[-2]                    folds.py:516
test = quarters[-1]                    folds.py:516
validation = folds 2+3                 build_final_validation.py:1100-1115 (_resolve_quarter_folds)
   ↓
final_validation.csv  = "neither side is a training gtin"  (build_final_validation.py:1106)
```

- **The single knob is `split.holdout_component_folds`**, and `src/core/schemas.py:793`
  pins it to `Literal[4]`. Any other arity fails at config load
  (`schemas.py:823-870`), then again at runtime (`folds.py:487-504`).
- `dev_fraction`/`test_fraction` must each equal `1/n_folds` exactly (`schemas.py:856-869`),
  so they cannot be used to tune size either.
- **`build_final_validation.build(n_folds=...)` `:1225` accepts an override but does not use
  it for the split**: `derive_holdout` is called with `dict(split)` `:1254-1256`, i.e. the
  config's `holdout_component_folds`. `n_folds` is only forwarded to
  `_resolve_quarter_folds` `:1261` and `negative_policy_evidence` `:1284` — so an override
  desynchronizes the fold labels from the actual cut. This is a latent bug and the exact
  reason "validation size must be FLEX" cannot be satisfied by the current override.
- Fixed *counts* that do control a validation-ish size, elsewhere:
  `rand_matching.truth_splits.sample_size: 12986` / `calibration_size: 8657`
  (`training.yaml:861-862`), `sweep.smoke_sample: 128` / `sweep_sample: 2000`
  (`training.yaml:1271-1272`), `hpo.n_trials: 20` (`training.yaml:1244`).
- To make validation size flex, the change surface is exactly:
  `schemas.py:788-796` (`Literal[4]`, `ge`/`le` on the fractions) →
  `folds.py:487-504` (`_check`) + `:506-516` (`cut`) →
  `build_final_validation.py:1100-1115` (`_resolve_quarter_folds`) →
  `evaluate_models.py:213-230` (`_DEV_FOLD`/`_TEST_FOLD`) →
  `schemas.py:839-852` (CV envelope). Five sites, all keyed off one yaml integer.

### 5.4 Validation code entrypoints (CLI-level)

| Entrypoint | Command | Notes |
|---|---|---|
| Build validation | `PYTHONPATH=src python -m src.training.build_final_validation` | `build_final_validation.py:1465 main()` |
| Prepare orchestrator stage | `prepare_all` stage `'validation'` → `training.build_final_validation` | `src/training/prepare_all.py:95`; stage order `config/paths.yaml:80` |
| Score the validation split | `python -m src.training.evaluate_models` | `evaluate_models.py` top-level script (no `main()`) |
| Robust folds report | via `training.generate_training_report` | `generate_training_report.py:370-381` |
| Post-training held-out inference | Colab/Kaggle worker | `src/training/validation_inference.py` + `colab.final_inference` `training.yaml:589-606` |
| Validation upload prewarm | `src/cli/colab_validation_upload.py:179 drain_validation_upload_prewarm()`; called `src/cli/colab.py:2840` | |

---

## 6. Who OWNS what (class ownership table)

| Surface | Intended owner (brief) | Actual owner today | Gap |
|---|---|---|---|
| Original export + members | `Dataset` | `Dataset` (`src/core/dataset.py`, `config/dataset.yaml`) | ✅ done |
| `prepared/` + `track_setup/` trees | `Dataset.layout` | `Dataset.layout` (`dataset.yaml:42-44`; `paths.yaml:390-399`) | ✅ done |
| The three official **sets** (source/3k/smoke) | `Dataset` | **UNOWNED**: source is a member; `dataset_3k.csv` is a code literal; `smoke_200` is 2 yaml keys + 1 code literal + 1 test tuple | ❌ |
| Splits (train/dev/test, calibration, folds) | `Dataset` | `UNOWNED` — free functions in `src/training/folds.py`; config block `split:` | ❌ |
| `final_validation.csv` + fold map | `Dataset` (member) | file-level: `Dataset`; **split-level: nobody** | partial |
| Hosted datasets | `HostedRegistry` | `config/hosted_datasets.yaml` + `src/core/hosted_dataset.py` (new, untracked, parallel work) | in flight |
| Run artifacts | `Artifacts` | `config/artifacts.yaml` (new, untracked) | in flight |
| post-training results | `Results` | `src/core/results.py` **does not exist yet** | ❌ |
| Transport / sealing | `Bundle` | `src/core/bundle.py` | ✅ |

> Note: `config/hosted_datasets.yaml`, `config/artifacts.yaml` and
> `src/core/hosted_dataset.py` appeared in the worktree **while this map was being written**
> (untracked, timestamps 22:23-22:24). They are sibling-agent work-in-progress; the map
> treats them as in-flight, not as settled SSOT.

---

## 7. Flagged: splits/surfaces nobody owns

1. **`src/training/folds.py` is the entire split SSOT but is not a class.** `Dataset` was
   designated the owner in `.dataclass-brief.md:39-41`; it has no split API. Every consumer
   calls free functions and re-threads `dict(training_cfg().split)`.
2. **Two CV implementations** — `core/common.py:1529 kfold_gtins` (gtin-level) vs
   `folds.py:415 component_folds` (component-level). `training.py:5018` uses one,
   `train.py:1642` / `hpo.py:284` use the other. Nothing reconciles them.
3. **`evaluation.component_split_k` / `dev_fold` / `test_fold`** (`training.yaml:751-753`)
   are declared, schema-validated, and **steer nothing** (`evaluate_models.py:169-174` says so).
   Dead config that looks live.
4. **`validation SIZE` is pinned at `Literal[4]`** and `build(n_folds=...)` is a decoy
   (§5.3).
5. **Unowned cohort detection**: `_COHORT_EXPORTS` is a hardcoded tuple in
   `src/core/common.py:339-342`, not a yaml key — the only set-membership rule in the tree
   outside config.
6. **`dataset_3k.csv` is unowned end-to-end** (no binding, no producer, one code-literal
   consumer).
7. **NER splits** live in an unvalidated `dict[str, Any]` config section
   (`schemas.py:3781`).
8. **`robust_validation.min_test_negatives`** is read by two unrelated modules
   (`robust_validation.py`, `build_final_validation.py:1280-1285`) with no shared owner.

---

## 8. Sets / splits the owner may be MISSING

1. **The full-lane prepared text bundle (`data/prepared/full/`) and `data/track_setup/` are
   real "sets" with no class.** Bound by `colab.full_prepared_bundles`
   (`training.yaml:461-462`) and `model_tracks.yaml:3-4`, read by 4 scripts + the dashboard
   (`scripts/laya_build_dataset.py:91`, `scripts/build_field_slice.py:19`,
   `scripts/flip_validity_audit.py:22`, `scripts/slice_scale_ladder.py:23`,
   `dashboard/app.py:321-346,519`) **by path literal**. The brief's "full: full + validation
   only" cannot be satisfied by anything in the tree today: there is no declared `full` split
   group anywhere.
2. **The laya holdout (`data/laya/holdout.csv`)** is a *third* split family alongside the
   laya corpus train/dev/test and the training holdout. It is bound
   (`laya_config.py:216`) but its producer (`scripts/laya_holdout.py`) derives components
   from the export, not through `folds.derive_holdout` — worth an owner check.
3. **`data/laya/unknown_pairs.csv`** — an undeclared laya member (written by
   `scripts/laya_build_dataset.py:66`); not in `LayaSpec` and not in `Dataset`.
4. **`artifacts/abl_opt/inputs/*`** — the 3k ablation's frozen inputs
   (`pairs_dev_500.csv`, `ablation_settings.yaml`, `threshold.json`,
   `shared_minilm__ablation_3k.npz`) are a *fourth* split sink; declared only in
   `scripts/ablation_timing_3k.py:65-80` (`split stays 'dev'`).
5. **`results/rand_truth/`** holds `calibration_truth.csv`, `holdout_truth.csv`,
   `gtin_stratum_sweep.csv` — declared in `training.yaml:851-873` but not in the
   `files:` phone book (addressed by raw `output_dir` + `output` names).
6. **No `smoke_200` split is declared**, yet `data/prepared/smoke_200/listing_splits.csv`
   *contains* a train/dev/test (115/50/45). "smoke: no splits" is not true of the artifact
   on disk — the smoke fixture carries a real 3-way listing split.
7. **`data/prepared/smoke_200/ablation_cohort/`** (untracked, new) is a new fixture surface
   with no declared owner.

---

## 9. Cross-checks performed

- Every `files:`/`layouts:` key cited exists in `config/paths.yaml` and resolves through
  `src/core/common.py:790 _resolve_file` / `:870 artifact`.
- Every `config/dataset.yaml` member key was matched against a `paths.yaml` `files:` entry
  (all 7 resolve).
- `grep -rn` sweeps over `src/ scripts/ dashboard/ tests/ config/` for each deleted/kept
  path name; results are the tables above.
- Counts read from the artifacts themselves: `data/prepared/{smoke_200,smoke_500,track_setup}/listing_splits.csv`,
  `data/laya/receipt.json`, `data/laya/{train,dev,test}.jsonl` line counts.
- `git status`/`git log` were read-only (`--porcelain`, `--oneline`); no state was changed.

## 10. Residual uncertainty

- **"laya / full / smoke" in the brief's *Splits* section does not map cleanly onto any single
  existing concept.** Read as *split families*, they are: laya corpus (train/dev/test, §4.A),
  the full/training holdout (§4.B), smoke (no declared split, §8.6). Read as *datasets*, they
  are `dataset.csv`, `dataset_3k.csv`, `smoke_200`. Both readings are recorded; the owner
  should confirm which is meant before any split gets rebuilt.
- `dataset_3k.csv` and `dataset_50pct.csv` have **no producer in the tree** — provenance for
  both is unrecorded, so their regeneration cannot be verified from code alone.
- `src/core/results.py` (named in `.dataclass-brief.md:50-52`) does not exist; `Results`
  ownership is unbuilt.
