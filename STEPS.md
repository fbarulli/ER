# STEPS — every step of the lane, fully defined

One standalone folder for GPU training. The config SSOT is SPLIT into
domain files, each in its owning directory and validated by a pydantic
model (lib/schemas.py) at load — a bad value crashes at import with the
file + field named, never mid-run:

| file | model (lib/schemas.py) | owns |
|---|---|---|
| `00_config.yaml` | `DataConfig` | paths, file names, column mapping, seed, model registry |
| `TRAIN/training.yaml` | `TrainingConfig` | loss, split, masking, **gate thresholds**, training knobs, pair thresholds + eval-pair caps, bands, mining, HPO spaces + selection protocol (`hpo.objective` / `hpo.selection_skip_test_eval`), rerank rule, ablation sweep, plots (dpi), audit (strip-audit sample + blocking-audit budget/min-recall + manifest knobs: `manifest_dir` / `source_export_expected_rows` / `source_drift_threshold_pct` / `manifest_stages`) |

(The EDA dir and its eda.yaml were deleted 2026-09-10 — the lane is
training-only. The five TRAIN-consumed EDA keys — plots.dpi,
pairs.max_pos_per_group/n_neg/neg_oversample, strip_audit_sample —
migrated into TRAIN/training.yaml blocks of the same names.)

Every path, file name, threshold, model name, and split lives in ONE of
these two; the numbered scripts and `lib/` read them through
`lib.common` (deep-merged view + typed accessors `data_cfg()` /
`training_cfg()` / `resolve_model()` / `hpo_cfg()` /
`rerank_cfg()` / `sweep_cfg()` / `plot_dpi()`) — nothing is
hardcoded. Word lists: `lib/pipe_stopwords.json` (data_pipe's
STOPWORDS/MINIMAL_STOPWORDS/CONCEPT_FOLDS) and
`lib/sklearn_stopwords.json` (matching.py's frozen sklearn set) — both in
`lib/` beside their consumers, two files two names (the shared basename
was split 2026-09-08).

### No-fallback completion (audit 2026-09-09, owner Q27)

The LAST inline literals that duplicated config values are gone; every
one below moved to a validated config block, and `TRAIN/selftest.py`
oracle 12b pins their absence from the code (comment-documented history
excepted — the scan matches executable lines only):

| was (file: literal) | now (config) |
|---|---|
| `TRAIN/hpo.py: GRID/QUICK` dict lists | `hpo.grid` / `hpo.quick` |
| `TRAIN/training.py: HPO_SPACE` dict | `hpo.tpe_space` |
| `TRAIN/train.py: --n-trials 20 / --n-jobs 1` | `hpo.n_trials` / `hpo.n_jobs` |
| `TRAIN/training.py: layer_decay=0.9` | `training.layer_decay` |
| `TRAIN/training.py: save_total_limit=2` | `training.save_total_limit` |
| `TRAIN/rerank.py: max_length=512` | `training.rerank_max_length` |
| `TRAIN/rerank.py: > 0.005` A/B margins | `rerank.min_delta_pr_auc` / `min_delta_f1` |
| `TRAIN/train.py: cfg dict 0.05/0.01/linear/1.0` | `training.*` via `runtime()` |
| `TRAIN/train.py + zero_shot_sims.py: batch_size=128` | `training.batch_size_embed` |
| `TRAIN/train.py: mask midpoint 0.10` | derived: `(mask_lo+mask_hi)/2` |
| `TRAIN/plots* 29× dpi=150` | `plot_dpi()` ← `TRAIN/training.yaml plots.dpi` |
| `run_all.py: ("title_only",)/("0.25","0.50","0.75")/2000/CE id` | `sweep.payload_variants` / `train_fracs` / `sweep_sample` / `rerank_model` |
| `colab_backend.py: sample=1000, epochs 2` | `sweep.smoke_sample`, `training.epochs` |
| `lib/blocking.py: n_neg*60` | `pairs.neg_oversample` (TRAIN/training.yaml) |
| `TRAIN/strip_audit.py: or 200` | `audit.strip_audit_sample` (TRAIN/training.yaml) |
| `TRAIN/blocking_audit.py: BUDGET=5M / MIN_RECALL=0.95` | `audit.blocking_budget` / `audit.blocking_min_recall` (round 3) |
| `data_pipe.three_way_gate: 0.05/0.85/0.3` defaults | `gate:` block in TRAIN/training.yaml (round 2 F01) |
| `TRAIN/evaluate_models.py: 2× dpi=150` | `plot_dpi()` (round 3; drift-scan pinned) |
| `TRAIN/hpo.py: tcfg 0.01/linear/1.0` | `training.*` via `runtime()` (round 3; drift-scan pinned) |
| `colab_backend.py: train-frac 0.25` | `sweep.train_fracs[0]` (round 3) |
| `colab_backend.py: artifacts/results + artifacts/data inline` | `RESULTS` / `DATA_DIR` from `lib.common` (round 3) |
| `lib/volume_verified.py: second04_pairs_positive.csv inline` | `files.second04_pairs_positive` via `F[...]` (round 3) |
| `lib/nlp._cosine` (duplicate of pair_similarity) | `lib.common.pair_similarity` — `_cosine` re-exports it (round 3) |
| `lib/nlp.encode_corpus: batch=256/seq=128 defaults` | required keyword-only params — callers pass `runtime()` values (round 3) |
| fixed-threshold metric names `f1_at_0.55` | `f"f1_at_{fixed_threshold:g}"` (config-derived) |

This is the contract. Each step states exactly what enters, what happens,
what leaves. Nothing undefined is allowed to run.

## The core problem

Retailers sell the same physical product, but each describes it differently
(language, formatting, missing units). We need to determine which listings
refer to the same real-world product. The GTIN (barcode) is supposed to
uniquely identify a product, but:

- Many GTINs are missing, invalid, or reused incorrectly.
- The same GTIN can have inconsistent attributes (volume, pack size, flavor)
  across retailers, indicating data errors or barcode reuse.
- Different GTINs can describe the same product because retailers sometimes
  list the same item under different barcodes (private label vs branded,
  local variations).

We need to group listings into true product clusters, using GTIN where it's
valid, and semantic matching where GTIN is missing or unreliable.

## Our approach

1. **Validate GTINs** (length, GS1 check digit — `lib/gtin.py`) to separate
   clean from noisy barcodes. Checksum-invalid barcodes assert NO identity
   anywhere: no canonical forms on them (01), no eval positives/negatives
   certified by them (blocking/hard_negatives), no T1 collapse on them
   (TRAIN/dedupe.py — they fall through to title tiers). Trust-only: grouping
   keys stay RAW gtin strings.
2. **Extract product attributes** (volume, pack count, flavor, type) from
   titles and structured fields.
3. **Use a deterministic three-way gate** on volume/pack/flavor to block
   impossible matches and route uncertain pairs to fallback.
4. **Fine-tune an embedding model** to score the remaining pairs based on
   cleaned text (no size numbers), so it learns product identity beyond
   just brand/type.
5. **Evaluate with component-fold cross-validation** to avoid leakage and
   get honest metrics (PR-AUC, F1).

## The three data sets (config `split:`)

50 / 25 / 25 over **connected components** of the positive-pair graph (two
products linked by any positive chain share one component — a component is
never split across boundaries):

| set | share | used for | model trains on it? |
|---|---|---|---|
| **train** | 50% (q0+q1) | gradient updates | **YES — the only one** |
| **dev** | 25% (q2) | early stopping / metric selection (dev AP) | weights chosen here, never gradient-updated |
| **test** | 25% (q3) | HOLDOUT — final reported metrics only | **NEVER — not in training, not in early stopping, not in any tuning** |

Config SSOT (split, 2026-09-08 — `TRAIN/training.yaml` `split:`):
`train_fraction: 0.50`, `dev_fraction: 0.25`, `test_fraction: 0.25`
(asserted to sum to 1.0 by `SplitSpec`), `fixed_threshold: 0.55` (the
operating threshold for F1/P/R), `cv_folds: 5`.

---

## 1 — Data the model sees: original + augmented, NO NUMBERS

**Sku side** (`clean_sku_text`): title + attributes → normalized (NaN-safe),
volume/pack tokens stripped, minimal stopwords dropped, then the
**number-token strip** driven by `artifacts/data/number_tokens_reference.csv`
(1,745 tokens, ~95.2% occurrence coverage; verdicts: strip / keep_brand /
keep_nutrient / keep_name). Kept digits are semantic only (b12, o2, b6,
alkaline88, good2grow, 12shots). Pure-numeric brands are spelled out
(1724 → seventeen, 1642 → sixteen forty two). Model payload additionally
filters SCHEMA_WORDS ∪ MODEL_PAYLOAD_SOFT_STOP (owner-curated soft-stop
list — format words like packtype/carton/sweetener-class drops; variant
signals like concentrate/powder/sugar deliberately KEPT).

**Canonical side** (`canonical_model_text` — NEW): the model payload gets a
**number-free canonical**. The gate's `canonical_records.csv` KEEPS numbers
(volume/pack drive hard_no) — **the gate CSV and the model payload are not
the same file/purpose**:
- gate canonical: `isostar orange orange_12x500 ... pet_orange_12x500_juice...`
- model payload:  `isostar orange type_plastic_flavour_orange ... pet_orange_juice...`
Measured: 3,562 canonicals carried digits → 13 after the model-strip (12×
`o2`, 1× `9.5ph` — semantic whitelist), gate CSV unchanged.

**Augmented**: for 15% of positive pairs (`masking.frac`), the ANCHOR sku
text gets random token masking at an extent drawn per-copy from U(5%, 15%),
appended as an EXTRA positive (mask token "`", same barcode → same
component, no fold contamination). +4,015 pairs at current settings.

**Pair scheme** (contrastive): positive rows = (sku_text, own-GTIN
canonical) — every row whose barcode has a canonical and both sides
survive the empty-text guard; negative rows = (sku_text, other-GTIN
canonical) from gate hard_no ∩ sim≥0.80 both directions (includes
flavor-mismatch hard-nos). Exact counts print at run time
(`build_training_data` stats: nothing drops silently — empty-text drops
are counted per side).

## 2 — First step of training: the objective function

**Loss: OnlineContrastiveLoss (owner ruling 2026-09-07, config
`training.loss: contrastive`)** — a labeled-pair margin loss over
`(sentence1, sentence2, label)` rows:

- **positives (label 1)** = gate proceed-pairs (sku, own canonical) —
  volume/pack/flavor-verified same product — plus the volume-verified
  cross-country lane when the second04 manifest is present (loud notice
  when absent);
- **negatives (label 0)** = the gate hard-no pairs — same-brand,
  text-similar, gate-proven DIFFERENT size/pack/flavor;
- **hard-pair selection is native to the loss**: per batch it computes
  cosine distances, then optimizes ONLY the hard positives (the farthest
  positive pairs) and hard negatives (the closest negative pairs, hinge
  `margin − d` with `training.contrastive_margin: 0.5`). Hard-pair
  training happens at BOTH layers: the data is hard by construction
  (gate-verified), and the loss mines the hardest subset per batch.

Legacy lanes remain available: `--loss mnrl` (MultipleNegativesRankingLoss,
in-batch ranking, negatives as a 3rd column) and `--loss triplet`
(mined triplets).

- optimizer: AdamW, discriminative LRs (8 layer groups, decay 0.9 bottom→top)
- schedule: linear warmup (5% of steps) + linear decay
- max_grad_norm 1.0, weight_decay 0.01, bf16 on GPU
- early stopping: dev AP every eval interval, patience 3, best checkpoint restored

## 3 — Success metric

**Primary: PR-AUC (average precision) on the holdout test set.** Positives are
rare relative to the pair pool, so PR-AUC is the honest ranking metric.

Reported per fold/run (all on the 25% holdout, never on train/dev):
- **PR-AUC** (primary)
- Precision / Recall / **F1 at the fixed operating threshold** (config
  `split.fixed_threshold`, currently 0.55). The metric column NAMES are
  config-derived (`f1_at_0.55` today follows `fixed_threshold` verbatim —
  a threshold change renames the columns, never lies in them).
- ROC-AUC (secondary sanity)
- accuracy at the Youden point + the Youden threshold itself

**Holdout discipline (owner audit 2026-09-07)**: the Youden threshold is
PICKED ON DEV and APPLIED TO TEST — never fitted on the scores it rates.
The fold rows keep `youden_thr_test_descriptive` purely as the leak
diagnostic (how much a test-fitted threshold would have flattered the
numbers). The same discipline applies to the 07e rerank A/B: thresholds
for both arms (bi-only, hybrid) come from dev, absolutes are test-side.
Pinned by `TRAIN/selftest.py` oracle 6b on the real split: 7,488 train /
3,743 dev / 3,743 test barcodes — pairwise disjoint, zero straddling
positive pairs.

The zero-shot evaluation lane (`TRAIN/evaluate_models.py`, self-fit leak
closed 2026-09-14) follows the SAME discipline: labeled pairs are split
into DEV/TEST by `TRAIN/folds.component_folds` over the positive-pair
barcode graph (knobs `evaluation.component_split_k` / `dev_fold` /
`test_fold` in TRAIN/training.yaml — measured on the real pair set:
6,183 dev / 7,633 test pairs, 6,100 straddling hard-negs dropped loudly
in both directions, 0 positives lost to straddle by construction); the
Youden threshold is fit on DEV and applied verbatim to TEST, ALL
reported metrics are TEST-half numbers, and every summary row carries
`threshold_source=dev_youden` + the leak diagnostic
`youden_thr_test_descriptive`. `evaluate_model(threshold=None)` raises —
fitting the threshold on the scored set is the closed defect.

Fold line example:
```
fold 0: loss=1.35 acc@dev-youden0.71=0.60 AUC=0.62 cross=0.62 PR-AUC=0.28 F1@0.55=0.29 P@0.55=0.17 R@0.55=0.94 | best_dev_ap=0.31
```

(`cross` = `auc_cross`: ROC-AUC restricted to cross-country positive pairs —
positives masked to `country[a] != country[b]` per `cross_mask` in TRAIN/train_one_config,
scored against the same hard negatives; see `auc_cross` in TRAIN/training.py.)

## 4 — Cross-encoder evaluation (stage 2) — the A/B protocol

The cross-encoder (rerank) must PROVE itself against the bi-encoder on the
SAME held-out component folds. Threshold EXCEPTION: this A/B uses a
dev-SELECTED Youden threshold for both arms (the fixed 0.55 is the
fallback only when no dev pool exists — CV mode), while the main
protocol's F1/P/R are always at the fixed config threshold:

1. Same holdout: no barcode in validation was seen in training (component
   split guarantees this for both stages).
2. For every validation pair, compute:
   - **bi-encoder similarity** (stage 1, cosine)
   - **hybrid score** = cross-encoder score for pairs in the confusion band
     ([0.50, 0.75] cosine, per `bands.rerank_band` in TRAIN/training.yaml),
     else the bi-encoder score
3. Compare on the holdout: **PR-AUC** (primary), **Precision/Recall/F1 at a
   threshold chosen on validation**, ROC-AUC (secondary).
4. **Decision rule (quantitative, `rerank:` block in TRAIN/training.yaml)**:
   the hybrid ships only when ΔPR-AUC > `rerank.min_delta_pr_auc` OR
   ΔF1 > `rerank.min_delta_f1` (both 0.005 today) on the holdout —
   otherwise the cross-encoder is not worth its latency; drop it. The rule
   is fixed in config BEFORE any test-side comparison runs.

## 5 — Hyperparameter sweeps: grid + TPE (selection protocol, 2026-09-12)

The `--grid` (second07's 11-config epochs×lr×warmup sweep, `--quick` =
3-config smoke) and `--hpo` (second08 optuna TPE) lanes tune optimizer
knobs — and the signal they select on is owned by the split mode
(`hpo.objective` / `hpo.selection_skip_test_eval`, TRAIN/training.yaml):

| mode | each config trains on | ranked on | per-config test eval |
|---|---|---|---|
| **holdout** (default) | q0+q1 (the 50% train side) | **dev quarter q2 `best_dev_ap`** | **SKIPPED** (`test_eval=skipped_selection_mode`) |
| cv | all-but-one component folds | **mean fold `auc`** (fold test sides are validation folds there) | runs (it IS the validation metric) |

The holdout rule closes the test-side leak: sweeps used to rebuild folds
over ALL barcodes and read a per-config test metric, so the sweep itself
fitted hyperparameters on the test quarter — the test set was read N
times, once per config, by the very lane that was supposed to be blind to
it. Now the test quarter is read exactly once, by the main train lane;
no per-config test number exists to select on, even by accident.

Wiring (TRAIN/train.py passes the SAME component split the main lane
built): holdout sweeps get `folds_override=` q3 (single test fold) +
`dev_override=` q2; cv sweeps get the component-fold list. Loud asserts,
no fallback (owner Q27): `TRAIN/hpo.py` `run_grid`/`run_tpe` assert BOTH
boundaries are present in holdout mode — a missing boundary dies with
`[hpo-grid]`/`[hpo-tpe] holdout split requires the component split's
folds_override (test quarter) + dev_override (dev quarter)` instead of
quietly rebuilding folds over all barcodes. `TRAIN/training.py`
`train_one_config(selection_mode=True)` asserts the boundary again per
fold: dev_override required (no rng carve), dev∩test=∅, no test barcode
in train/dev, and dev == dev_override ∩ train side — a violated boundary
kills the fold with a `[hpo] LEAK:` traceback, never a silent leak.

Console contract (holdout grid):
```
[hpo-grid] holdout selection: 1 test fold(s), dev_override=3,743 barcodes — per-config test eval SKIPPED (test read exactly once)
e1_lr2e-05_w0: devAP 0.3092 (sd 0.0000)
  [hpo] fold 0: test-side eval SKIPPED (selection mode — test read exactly once)
```
`results/hpo_grid.csv` fold rows carry `test_eval=skipped_selection_mode`
+ `objective=best_dev_ap`; the `config_mean` summary row carries
`objective=best_dev_ap` in holdout (`auc` in cv). The TPE lane's
`train_<model><era>_hpo_best.json` names the same signal in its
`selection` field (`best_dev_ap` holdout / `mean_fold_auc` cv).

## 6 — Flavor semantics (transparency contract)

- **Flavor STAYS in the embedding input**: canonical/sku texts keep flavor
  words (orange, apple, ginger) — discriminative signal the model needs.
- **Gate flavor check = hard block ONLY on exact extracted mismatches**:
  `if flavor1 and flavor2 and flavor1 != flavor2 → hard_no`. One side empty →
  NO block (unknown ≠ different). Currently 12,417 hard-no pairs from flavor
  mismatch; the remaining low-sim proceed tail (3,093 pairs) is a
  flavor-EXTRACTION coverage gap (Finnish/Dutch compounds), not gate logic.

## The three-way gate — decision table

The gate's three policy thresholds (volume tolerance ±5%, raw-confidence
cut 0.85, consistency cut 0.3) live in the `gate:` block of
TRAIN/training.yaml (round 2, F01) — `three_way_gate` reads them through
`training_cfg()`; passing a value explicitly still wins (the selftest
does). The decision table below documents the semantics of those knobs:

| Decision | Meaning | Criteria |
|---|---|---|
| **hard_no** | Confidently different products | Any of these: |
| | | • Volume sets have no overlap within ±5% tolerance (e.g., 250ml vs 500ml) |
| | | • Pack sets have no common pack count (e.g., single vs 6‑pack) |
| | | • Both sides have non‑empty flavors and they are different (e.g., apple vs orange) |
| **fallback** | Not sure; needs semantic scoring | Any of these: |
| | | • Volume or pack confidence is below 0.85 |
| | | • Volume and pack overlap, but consistency is very low (<0.3) |
| **proceed** | Likely duplicates; send to embeddings | All of these: |
| | | • Volume confidence ≥0.85 on both sides |
| | | • Pack confidence ≥0.85 on both sides |
| | | • Volume sets overlap within tolerance |
| | | • Pack sets intersect |
| | | • Flavors are compatible (same, or one/both missing) |
| | | • Consistency ≥0.3 on both sides |

## Pydantic boundary contracts (lib/schemas.py, 2026-09-08)

Every transform boundary in the training pipeline crosses a validated
contract — small objects at STAGE edges, never per-row hot loops:

| boundary | model | asserts |
|---|---|---|
| `extract_all` output | `ExtractedAttributes` | confidences in [0,1], pack_qty >= 1, volume >= 0 |
| `three_way_gate` output | `GateResult` | decision in {hard_no, fallback, proceed}, non-empty reason |
| `generate_canonical` output | `CanonicalRecord` | positive volume/pack sets, confidences/consistencies in [0,1], n_titles >= 1 |
| `build_training_data` output | `TrainingData` | payload/row_bc length-locked, every pos/neg index in range, stats complete |
| `augment_positives` output | `MaskingResult` | extended payload locked to row_bc, pos' in range, n_added == audit rows |
| train data tuple | `DataTuple` | country covers payload, emb0 rows == payload, pos/hp in range |
| per-config dict | `TrainConfig` | epochs >= 1, lr > 0, warmup [0,1] — dies BEFORE a fold runs |
| `component_folds` output | `FoldSets` | folds pairwise disjoint (anti-leak) |
| CSV frames | `check_*_frame` | exact columns, decision/label domains, similarity in [0,1], unique GTINs |

Config files are validated the same way: each of the two split files has
a pydantic model (`DataConfig` / `TrainingConfig`) checked at
load in `lib/common.py`. The FIRST live win: `pack_qty >= 1` caught
`extract_pack_from_title` producing pack 0 from "pack 0.5 l" / "0% sugar
… pack" title forms (3 canonicals carried an impossible 0 in pack_set) —
fixed with a zero-guard, 26 gate decisions corrected, +10 honest
hard-negatives in labeled_pairs.csv.

## Pipeline steps (importable modules — renamed from numbered scripts 2026-09-07)

The completion marker for each guarded stage is
`results/manifests/<stage>.json`, atomically published only after its inputs,
outputs, hashes, and row accounting have been recorded. These runtime
manifests are regenerated and never committed.

- **TRAIN/dedupe.py** — RUNS FIRST (historically `06_dedupe.py`, hence the
  `06_*` result filenames): tiered
  exact-duplicate dedupe of the raw export BEFORE canonical formation
  (T1 retailer+barcode — GS1-checksum-VALID barcodes only, invalid ones
  defer to title tiers / T2 retailer+title+price / T3 retailer+title
  price-aggregation with flags) → `dataset_deduped.csv` (61,529 rows;
  T1: 1,943 collapsed + 3,715 checksum-invalid deferred) +
  `sku_to_rep.csv` (audit-only pointer) + `06_dedupe_summary.csv` +
  `06_ambiguous_offer_groups.csv`. Everything downstream reads the
  DEDUPED dataset, never the raw export.
- **TRAIN/build_reference.py** — reproduces the committed
  `number_tokens_reference.csv` (the number-token verdict census; 1,745
  rows, 95.2% coverage). `--verify` asserts byte-equality against the
  committed CSV and exits nonzero on drift. Run order: after 06 (census
  reads dataset_deduped).
- **TRAIN/data_prep.py** — within-brand pipeline: extract volume/pack/flavor
  per row → canonical per GTIN (GS1-checksum-invalid barcodes form NO
  canonical — 1,747 invalid groups dropped loudly; NaN titles now clean to
  "" instead of poisoning canonicals with literal "nan") → three-way gate
  (with flavor check) every candidate pair → `canonical_records.csv`
  (13,250) + `gate_results.csv` (135,769 pairs: hard_no 92,650 /
  proceed 29,351 / fallback 13,768).
  Canonical records also retain per-GTIN `description_evidence` and
  `breadcrumb_evidence` from the source export for review and a future
  component-safe ablation; neither field changes the frozen gate or model
  text without that validation.
- **TRAIN/zero_shot_sims.py** — encode canonical texts with each
  model, score gate pairs → `embedding_similarities.csv` (per-model sim
  columns, incremental per-model writes, resumable).
- **TRAIN/labeled_pairs.py** — gate decisions + sim≥0.8 (SSOT
  `pairs.*_sim_threshold`) → auditable `labeled_pairs.csv`.
- **TRAIN/evaluate_models.py** — per model: ROC-AUC + P/R/F1 on the TEST
  component half, Youden threshold fit on the DEV half (self-fit leak
  closed 2026-09-14; knobs `evaluation.*` in TRAIN/training.yaml),
  per-model plots with absolute n → `model_evaluation_summary.csv`
  (provenance columns `eval_half` / `threshold_source` /
  `youden_thr_dev`, leak diagnostic `youden_thr_test_descriptive`).
- **TRAIN/train.py** — the training entry (masking, holdout/cv folds,
  MNRL, early stopping, plots, mlflow, rerank). ALSO emits the 07-series
  CSVs (owner ruling): 07c/07d per run (payload / train-frac variants,
  append-with-replace), 07b from the `--rerank` lane. Run-tag carries
  model+payload+frac so the ablation series can no longer overwrite each
  other's artifacts. Round 3: `--folds N` builds N folds (no silent cap
  at `split.cv_folds`; N <= CV_FOLDS behavior unchanged), and the
  fold-metrics row carries `lr_groups` ("discriminative" / "single") so a
  discriminative-LR fallback fold is queryable downstream, not just
  visible in stdout. The `--grid`/`--hpo` lanes ride the SAME component
  split as the main lane and follow the section-5 selection protocol.
- **TRAIN/report_plots.py** — the 07_report figure family, per model.
- **TRAIN/composition_plot.py** — training-data composition with absolute n.
- **run_all.py** — orchestrator: embeddings → sweep-sample sweep → full-data
  run → ablation suite. Every axis of the ablation suite (payload
  variants, train-frac curve, sweep/smoke sample sizes, rerank
  cross-encoder id) reads the `sweep:` block of TRAIN/training.yaml via
  `lib.common.sweep_cfg()` — no inline sweep lists. Step logs APPEND
  (run-separator line, never truncate a previous run — round 3 F16);
  `--stop-on-fail` halts the chain at the first failed step (default
  off: record `failed:` in the CSV run ledger and continue, the historical
  resumable behavior).

## CSV reproducibility map — every .csv and its producer

| CSV | Producer | Committed? |
|---|---|---|
| `data/dataset.csv` | raw export (input) | YES — never regenerated |
| `data/number_tokens_reference.csv` | `TRAIN/build_reference.py` | YES (reproducible; `--verify` pins it) |
| `data/dataset_deduped.csv` | `TRAIN/dedupe.py` | regenerated |
| `data/sku_to_rep.csv` | `TRAIN/dedupe.py` | regenerated (audit-only, zero consumers) |
| `results/canonical_records.csv` | `TRAIN/data_prep.py` | regenerated |
| `results/gate_results.csv` | `TRAIN/data_prep.py` | regenerated |
| `results/labeled_pairs.csv` | `TRAIN/labeled_pairs.py` | regenerated |
| `results/embedding_similarities.csv` | `TRAIN/zero_shot_sims.py` | regenerated |
| `results/model_evaluation_summary.csv` | `TRAIN/evaluate_models.py` | regenerated (TEST component half, dev-fit Youden — provenance columns, §3 holdout note) |
| `results/train_fold_metrics.csv` | `TRAIN/train.py` | regenerated (per-run suffixed copies kept) |
| `results/hpo_grid.csv` | `TRAIN/train.py --grid/--quick` | regenerated |
| `results/train_<model><era>_hpo_best.json` (+ `_hpo_trials.csv`) | `TRAIN/train.py --hpo` (TPE lane, run-tagged names) | regenerated |
| `results/06_dedupe_summary.csv` | `TRAIN/dedupe.py` | regenerated |
| `results/06_ambiguous_offer_groups.csv` | `TRAIN/dedupe.py` | regenerated |
| `results/06_dedupe_removals.csv` | `TRAIN/dedupe.py` | regenerated (one reviewable row per removed product) |
| `results/manifests/<stage>.json` | guarded pipeline stages / `run_all.py` | regenerated (atomic completion/integrity record; never committed) |
| `results/07b_four_pop_scores.csv` | `TRAIN/train.py --rerank` | regenerated |
| `results/07c_field_ablation.csv` | `TRAIN/train.py --payload <v>` | regenerated (append) |
| `results/07d_data_scaling.csv` | `TRAIN/train.py --train-frac <f>` | regenerated (append) |
| `results/blocking_feature_audit.csv` | `TRAIN/blocking_audit.py` | regenerated (measured: brand 0.981 recall @ 2.15M cands) |

Full-chain reproduction: `dedupe → build_reference --verify → data_prep →
zero_shot_sims → labeled_pairs → evaluate_models → selftest → train
(--payload full → variants) → blocking_audit → report_plots`.

## Transparency guarantees

- Every count printed at run time: pairs, canonicals, dropped endpoints,
  masked additions, per-fold n_pos/n_neg.
- Every file name and path from the split config SSOT only
  (`00_config.yaml` + `TRAIN/training.yaml`, all through
  `lib.common`).
- Every guarded stage writes an atomic manifest last; it hashes its declared
  files, checks row-accounting closure, and treats a missing expected file or
  interrupted-write residue as failure. Colab download lanes re-hash against
  their remote manifests before accepting artifacts.
- Every plot carries absolute n (titles + per-bar annotations).
- `gate_results.csv` = GATE input (numbers kept). Model payload = NUMBER-FREE
  variant (derived at pair-construction time, never persisted as a second
  CSV — one canonical SSOT, no drift).
- Loss/acc/AUC/PR-AUC/F1 in console + `train_fold_metrics.csv` + mlflow
  (local sqlite backend, `artifacts/mlruns/`).
- **Byte-determinism**: every regenerated CSV is reproducible byte-for-byte
  (PYTHONHASHSEED-proof — set displays sorted at write, keep-token iteration
  sorted, gate pair rows sorted on identity columns). Verified by running
  data_prep twice and cmp-ing.
- **Oracle selftest**: `python TRAIN/selftest.py` — pinned known-good GTIN
  checksums (GS1/Wikipedia entries), cleaning oracles, soft-stop
  keep/strip oracles, reference verdicts, invalid-GTIN exclusion in eval
  pairs and mining, fold component integrity, P@R hand-computed cases,
  config-split oracles (three files, three pydantic models, merged view),
  pydantic boundary-contract oracles (reject the exact shape breaks the
  audits found: unlocked payload/row_bc, out-of-range indices, impossible
  pack/volume values, overlapping folds), the zero-pack guard, the pinned
  real-data counts below, AND the no-fallback SSOT oracle (12b: hpo/rerank/
  sweep blocks load, the module-level sweep lists match the config, the
  drift scan proves the retired inline literals stayed retired, and a
  missing runtime key raises). Round 3 added oracle 12c — the round-3 fix
  pins: gate thresholds == config values (three-way gate behavior pinned on
  proceed/hard_no/fallback cases), `hpo_tpe_best` and `bands.mining_band`
  stay deleted from the config, the F18 dead symbols stay deleted from the
  import surface, colab's train-frac default == `sweep.train_fracs[0]`,
  blocking_audit knobs == the `audit:` block, and
  `lib.nlp._cosine is lib.common.pair_similarity`. The manifest oracle also
  validates the manifest schema, atomic-write failure behavior, dedupe row
  closure, and live output hashes when runtime results are present. Exit 0 =
  green.

Pinned real-data counts (update ONLY alongside an intentional contract
change; `TRAIN/selftest.py` fails loudly on drift):
- canonical_records.csv: 13,250 rows
- gate_results.csv: 135,769 pairs (hard_no 92,650 / proceed 29,351 /
  fallback 13,768)
- number_tokens_reference.csv: 1,745 rows
- labeled_pairs.csv: 19,916 rows (7,409 pos / 12,507 hard-neg; the
  2026-09-08 zero-pack guard — found by the ExtractedAttributes schema —
  moved 10 pairs from proceed to hard_no)

## MLflow (local backend + artifact store)

Every `TRAIN/train.py` invocation = one parent run + nested run per fold. Default
backend LOCAL: `sqlite:///artifacts/mlruns/mlflow.db`, artifacts under
`artifacts/mlruns/artifacts/`. Browse:
`mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db`.
`MLFLOW_TRACKING_URI` overrides; `=off` disables.

## Training loss plot

`training_loss_<split>_payload-<variant>.png` — per-fold TRAIN vs
VALIDATION loss curves (train loss + dev eval_loss per logged step, best
dev-AP step marked); history persisted in `train_fold_metrics.csv`
(`train_loss_hist` / `dev_loss_hist` / `dev_ap_hist`). The validation loss
is computed by the HF Trainer on the dev pair dataset (`eval_dataset`,
same population as the dev evaluator — pos dev pairs + dev hard
negatives); the triplet lane emits train-only (the plot says so on the
axis instead of implying a missing curve).

## Report plots — per model

`TRAIN/report_plots.py` runs the 07_report family for EVERY config model:
`07_report_*_<model_key>.png`. `--models <keys>` selects a subset. Deberta
panels fill in on the GPU pass (CPU: ~2000× slower on this torch build).

## Docker (reproducibility)

`Dockerfile` builds `euromonitor-train-gpu` from the repo root:
```
docker build -t euromonitor-train-gpu \
  -f project/experiments/euromonitor/TRAIN_GPU/Dockerfile .
```
deps pinned in `requirements.txt` with fully pinned `==` versions (no uv — no
`uv.lock` exists in this repo; the image installs `pip install -r requirements.txt`,
per Dockerfile line 22); run the lane with
`--workdir /app/project/experiments/euromonitor/TRAIN_GPU`; mount
`artifacts/` to persist. Verified in-image: lint clean, config SSOT + data
pipe import, exact version pins (torch==2.14.0, transformers 5.16.1,
sentence-transformers 6.0.1, mlflow 3.15.1, optuna 4.4.0, sentencepiece
0.2.2, ruff 0.16.3), all per `requirements.txt`. 10.6GB.

## Deberta CPU warning

deberta-v3 relative attention on this torch CPU build runs ~2000× slower
than MiniLM (3.9 s/text vs 2 ms/text measured). Zero-shot deberta scoring
and deberta training run on the GPU lane.
