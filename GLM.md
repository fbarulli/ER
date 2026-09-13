# GLM.md — Review of Most Recent GH Changes

## Scope

- **Repo:** https://github.com/fbarulli/ER (local: `/home/opc/ONE/EuromonitoR`, branch `training`)
- **Original review range:** `c87d785..e51e5bd` — the initial 8 commits pushed to
  `ER/training` on 2026-09-13
- **Reviewed at:** HEAD `fa08930` (follow-up closure review; working tree was clean
  apart from this review document)
- **Follow-up commits reviewed:** `91f0a1d`, `7e51812`, `5ceb913`, `334fb84`,
  `a1ab904`, `0dd4fe2`, and `fa08930`.
- **Method:** full-range aggregate diff read end-to-end (`/tmp/range_final.diff`),
  then whole-file verification reads of every heavily-changed file at HEAD:
  `src/training/hpo_metrics.py` (702 ln), `src/training/rand_matching.py` (2095 ln),
  `src/core/graph_diagnostics.py` (122 ln), `src/core/common.py` (821 ln),
  `src/training/training.py` (calibration hunks), `src/training/train.py`
  (07-series + tracking hunks), `src/training/hpo.py` (grid/tpe hunks),
  `src/cli/colab.py` (worker-count hunks), `src/core/schemas.py` (spec hunks),
  `config/training.yaml`, `config/paths.yaml`.
- **Prior findings:** `FINDINGS.md` N1–N7 + A/B/C/D/E/F/G + RM1–RM9. This file
  reports what the range FIXED vs what it REGRESSED, plus NEW findings, all
  verified against the current checkout.

## Commit list

| # | Commit | Subject |
|---|--------|---------|
| 1 | 8b642fa | separate lane and dvc worker counts |
| 2 | 79cab23 | add full Rand matching error audit traces |
| 3 | 905ac80 | add Rand group and graph diagnostics |
| 4 | 9641649 | centralize calibration rand metrics across train and hpo |
| 5 | 8440738 | separate train calibration metrics from dev |
| 6 | 6a8d872 | make MiniLM L6 the configured training default |
| 7 | 18b9c81 | make calibration diagnostics fold-safe and traceable |
| 8 | e51e5bd | handle disabled collapse guardrails explicitly |
| 9 | 91f0a1d | preserve folds when calibration is unavailable |
| 10 | 7e51812 | fix fixed-grid HPO fold inputs |
| 11 | 5ceb913 | share candidate gate construction across matching lanes |
| 12 | 334fb84 | unify Rand threshold selection |
| 13 | a1ab904 | avoid duplicate calibration graph diagnostics |
| 14 | 0dd4fe2 | centralize canonical record loading |
| 15 | fa08930 | validate calibration fold and sensitivity rows |

## Findings by Category

### 1. Hardcoded Paths That Should Be in Config

- N3 (LOW): `dvc_jobs`/`dvc_workers` split is config-driven (good), but
  `_publish_local_hpo_model_snapshot` hardcodes `optuna_db=None` and the model
  snapshot scope = `model_dir.name` (colab.py:1429-1432) — behavior knobs, not
  paths; minor.

The remaining N3 item is ANN/DVC publication behavior and is intentionally
deferred. No training or Rand-metric findings remain open in this review.

### 2. Missing Pydantic Validation


### 3. Gaps in Data Traceability


### 4. SSOT (Single Source of Truth) Violations


### 5. Missing Centralized Functions / Duplicated Processes


### 6. Redundant Operations


### 7. Unexpected Behavior / Silent Errors


## Cross-Range Root-Cause Analysis

The original 8-commit range was followed by closure commits that addressed the
calibration availability path, shared candidate gates, threshold selection,
canonical-record loading, duplicate graph diagnostics, and strict intermediate
metric contracts. The remaining review items below are residual design or
performance risks that were not changed by those closure commits.

## What the Range Fixed vs Regressed (vs FINDINGS.md)

FIXED: training/metric config caching and config-owned labels/seed offsets;
training plot/threshold settings; provenance symlink handling; graph and metric
input contracts; calibration-unavailable fold handling; shared component-safe
calibration partitioning; shared candidate gates and GTIN status; shared
canonical-record loading; shared threshold fitting; duplicate graph diagnostics;
strict fold and sensitivity metric contracts; zero-presentation datapoint
lineage; post-train embedding reuse; uniformity pair selection; A1
(`_grid_folds` tuple indexes).

STILL OPEN from FINDINGS (outside this pass):
A3/H1/N3 (ANN gate-slot attribution); N6 (ANN refresh masked-tail encode);
F1 (trainer evaluator and eval-loss encode paths remain separate);
predict_items literals; rerank fold-0 only; NER island.

## Severity Summary

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| HIGH     | 0     | — |
| MED      | 0     | — |
| LOW      | 1     | N3 (ANN/DVC publication behavior) |

## Recommended next actions (owner stance: fail loudly)

1. Defer N3 and the separate trainer evaluator/eval-loss optimization until the
   ANN follow-up pass.

---

# Round 2 — Independent Verification Pass

## Scope of this pass

- **Target:** GH HEAD `b8f16fc` (`ER/training`), read from a CLEAN git worktree
  (`/tmp/er_head`) — deliberately NOT the local checkout, which carries in-flight
  uncommitted fixes. Every line reference below is **HEAD-relative**.
  Note the file lengths differ from Round 1's numbers (HEAD: `rand_matching.py`
  = 2138 ln, `hpo_metrics.py` = 668 ln), so Round 1's line numbers (taken from the
  local tree) are off by a few.
- **Range audited:** `c87d785..b8f16fc` — the whole push batch (worker counts →
  Rand audit traces → graph diagnostics → calibration centralization → MiniLM
  default → fold-safe calibration → closure commits).
- **Method:** aggregate range diff read end-to-end; whole-file reads at HEAD of
  `hpo_metrics.py`, `graph_diagnostics.py`, `folds.py`, the config/spec blocks of
  `common.py` / `schemas.py`, and every changed function of `rand_matching.py`,
  `training.py`, `train.py`, `hpo.py`, `colab.py`; then targeted greps for the
  CONSUMERS of each new field. Two micro-benchmarks were executed against
  `candidate_graph_diagnostics` itself (numbers in X1/X2).
- **Relationship to Round 1:** this pass does not repeat Round 1 items; it records
  what Round 1 missed, plus one correction. Round 1 items re-verified as still open
  at HEAD: N1 (see X11), N2, R1, R3, R4, R5, S4/F7, U2, U5, P4 (see X5/X12).

### 1. Hardcoded Paths / Values That Should Be in Config

- X10 (MED, NEW): the new `training.base_model` SSOT (6a8d872) is bypassed exactly
  where it decides which backbone an audit compares against — `colab.py:975`
  `base_model = Path(resolve_model("minilm_l6"))`, whose failure message claims
  "configured base model is not materialized locally". Change `training.base_model`
  and the local uniformity audit silently scores fine-tuned checkpoints against a
  backbone other than the one trained. Same literal-key class left in the tree:
  `colab.py:2024` (`"deberta_v3_base"`), `strip_audit.py:202`, `selftest.py:618-620`.
- X17 (LOW, NEW): `plot_dpi()` (`common.py:291-298`) documents "Every fig.savefig in
  the tree renders at THIS value", and `selftest.py:729` asserts the accessor, but
  FIVE call sites still hardcode `dpi=150`: `train.py:1123`, `training.py:3507`,
  `training.py:4171`, `colab.py:1159`, `uniformity.py:190`. The drift guard
  (`selftest.py:802-804`) bans the literal in `evaluate_models.py` ONLY — a
  hand-maintained per-file list that is itself a second SSOT. The range's own new
  plot is correct (config-driven via `rand_matching.py:1727, :1743`).

### 2. Missing Pydantic Validation

- X12 (LOW, NEW): the strict-contract coverage added by this range does not extend
  to the artifacts consumers actually read.
  (a) `CalibrationMetricRow.model_validate(result)` IS enforced on the available path
  (`hpo_metrics.py:667`, `extra="forbid"`) — good — but
  `unavailable_calibration_metrics` (`:170-182`) returns a 4-key dict typed
  `dict[str, str | int]`, so ONE fold-metrics CSV carries two mutually incompatible
  row schemas whose only discriminator is the free-text `calibration_status`.
  (b) The new full audit trace has only a column-**SET** contract
  (`_DiagnosticsColumnSpec`, `rand_matching.py:165-177`): no row model, and no domain
  constraint on `error_type` / `gtin_status` / `gate_reason` /
  `evaluation_partition` / `rejection_reason`, while the far smaller calibration
  tables got `extra="forbid"` row models in the same range.
  (c) The typed sensitivity rows are flattened into a JSON **string** column
  (`calibration_sensitivity_table`, `hpo_metrics.py:618-626`; model field `:71`), so
  their pydantic contract is not enforced at the artifact boundary at all.
- X5 (MED, NEW, see SSOT section): `candidate_graph_diagnostics` still consumes a raw
  DataFrame with no column contract (P4, confirmed) — and its key set is now
  duplicated four ways.

### 3. Gaps in Data Traceability

- X11 (LOW, NEW): `calibration_proxy_source` is WRITE-ONLY metadata — one producer,
  **zero consumers** anywhere in the repo (`hpo_metrics.py:574` is the only
  occurrence across `*.py` / `*.yaml`), and it is dropped entirely on the unavailable
  path (`:170-182`). Worse, the label it asserts (`dev_component_safe_split`)
  describes the component-safe DEV carve, while the fit/check split that actually
  produces the metric is ITEM-level: `_fold_ids` (`:366-369`) shuffles
  `true_item_id` and deals `index % n_folds`. A consumer cannot tell the two
  granularities apart from the label.
- X13 (LOW, NEW): `_SubmissionProvenance` (`rand_matching.py:1853-1892`) pins input
  and config file hashes, `final_threshold`, `unmatched_prefix`, `calibration_folds`
  and a lineage string — but records NO derivation parameters: the threshold grid
  (min/max/step), `target_recall`, `plateau_tolerance`/`plateau_min_points`, `top_k`,
  `batch_size` and the structured-features weight are all absent, and none of the
  DERIVED artifacts (threshold_selection_by_fold, threshold_sensitivity,
  plateau_diagnostic, calibration_diagnostics, holdout_*) is fingerprinted.
  `_write_calibration_outputs` had `target_recall` in hand (`:1706`) and wrote it only
  into the un-hashed plateau JSON. Reproduction therefore depends on a mutable single
  config path being identical at the recorded hash. Related: `dataset_deduped`'s
  `rows=len(submission)` (`:1854-1857`) is correct only by virtue of the
  SKU-population assertion in `_write_final_submission` (`:1918-1925`) — it should be
  the dataset's own row count.
- X14 (LOW, NEW, regression): centralizing the candidate gate record silently changed
  the recorded GTIN normalization. Pre-range `_candidate_row` set
  `sku_gtin = self._gtin(row_metadata_text(row, "barcode", "gtin"))`, which blanks
  `nan`/`none`/`null` (that helper still exists, `rand_matching.py:482-485`). The new
  `candidate_gate_fields` (`:321`) uses bare
  `metadata_text(row_metadata_text(row, "barcode", "gtin")).strip()`, so a barcode
  cell reading `nan` is now written to the diagnostics CSV as the literal `"nan"`
  where it used to be `""` (FINDINGS L1–L5 flags the same NaN→"nan" collision class in
  `predict_items`). `sku_gtin_valid` is unaffected. The module now holds THREE GTIN
  normalizations: `_gtin` (used by `_candidate_indexes`), `trusted_gtin`
  (`:287-292`), and this inline one.
- X16 (LOW, NEW): `include_graph_diagnostics=False` returns ZEROS for
  `diagnostic_*` / `plausible_group_count` (`rand_matching.py:1016-1025`), so "not
  computed" is indistinguishable from "computed and zero" for every consumer of the
  metric dict — and that block is a verbatim duplicate of the empty-accepted payload
  in `graph_diagnostics.py:31-40`.

### 4. SSOT (Single Source of Truth) Violations

- X5 (MED, NEW): the diagnostic metric key set is hand-maintained in FOUR places that
  must agree exactly — `METRIC_COLUMNS` (`rand_matching.py:77-106`), the two return
  dicts in `graph_diagnostics.py` (`:31-40`, `:113-122`) and the
  `include_graph_diagnostics=False` fallback (`rand_matching.py:1016-1025`). A
  mismatch is a `KeyError` at `:1048-1052` or a `RuntimeError` from the
  order-sensitive contract check at `:1055`. The SAME range derived
  `CALIBRATION_AGGREGATE_FIELDS` from `CalibrationMetricRow.model_fields`
  (`hpo_metrics.py:146-152`) precisely to stop this drift for the calibration
  fields — the diagnostic set was left hand-rolled. Fix: one
  `DIAGNOSTIC_METRIC_COLUMNS` tuple + one empty-diagnostics payload builder.
- X6 (MED, NEW): the calibration fold split is seeded by the **collapse guardrail's**
  seed — `_fold_ids(truth, n_folds, int(config["hpo"]["collapse_guardrail"]["seed"]))`
  (`hpo_metrics.py:494`). `config/training.yaml` has `hpo.calibration_folds` (`:304`)
  but NO calibration seed; `collapse_guardrail.seed: 42` (`:308`) also seeds
  `select_unrelated_pairs`. Re-tuning the guardrail therefore silently re-deals the
  calibration folds and changes every calibration metric — including in the MAIN
  train lane, where the guardrail is `not_requested` (`:394-400`) yet still
  determines the folds. Fix: add `hpo.calibration_seed`.
- X9 (MED, NEW): stale validation invariant. `schemas.py:190-197`
  (`_registry_has_trainer_base`) RAISES unless `models.multilingual_l12` exists,
  justified as "the trainer base resolves from it (train.py --model default)" — but
  6a8d872 moved that default to `training.base_model: "minilm_l6"`
  (`paths.yaml:163-165`, `training.yaml:154-155`, `train.py:314`). The registry is
  still pinned to a backbone the trainer no longer uses, while the NEW key is only
  cross-checked in `load_config()` (`base_model not in registry_models`).
  `paths.yaml:165` also still comments `# trainer base` on `multilingual_l12` — which
  is why the stale check still looks correct.

### 5. Missing Centralized Functions / Duplicated Processes

- X7 (MED, NEW): the acceptance predicate now exists in THREE copies —
  `_annotate_candidates` (`rand_matching.py:658-661`), `_sweep_assignments`
  (`rand_matching.py:1346`) and the NEW `candidate_graph_diagnostics`
  (`graph_diagnostics.py:20-29`). All three spell
  `gtin_status != "different" AND (exact_gtin OR (rule_ok AND score >= threshold))`.
  A gate change now silently diverges the reported diagnostics from the shipped
  assignment. Fix: one `accepted_mask(frame, threshold)` used by all three.
- X15 (LOW, NEW): the same "true canonical for this SKU" expression appears THREE
  times inside `evaluate_calibration_trial` — `hpo_metrics.py:605`, `:613`
  (pre-existing) and `:657` (ADDED by this range for `attribute_conflict_error_rate`)
  — while `rand_matching` already owns a shared helper for it, `_candidate_labels`
  (`:1182-1196`, merge-based rather than map-based). Four spellings of one label rule.
- X8 (MED, NEW): `hpo_metrics._canonical_record_map` (`:185-190`) and
  `RandMatcher.record_map` (`rand_matching.py:412-415`) build the identical
  gtin→record structure from the same artifact with no shared accessor — see also the
  cost regression in section 6.

### 6. Redundant Operations

- X8 (MED, NEW): "centralize canonical record loading" (0dd4fe2) made this hot path
  MORE expensive, not less. Before: `@lru_cache(maxsize=1)` over the read. Now
  `_canonical_record_map` (`hpo_metrics.py:185-190`) is uncached and calls
  `canonical_records_frame()`, which returns
  `_canonical_records_frame().copy(deep=True)` (`common.py:433-435`) — a 9.4 MB /
  13,251-row frame deep-copied and then re-dictified with `iterrows` — on **every**
  calibration evaluation (per fold, per config, in both grid and TPE). Only the file
  READ is cached. Fix: `@lru_cache(maxsize=1)` on `_canonical_record_map` (it already
  hands out per-call copies).
- X3 (HIGH, NEW): the expensive graph diagnostic is recomputed inside a
  per-threshold × per-stratum loop. `_fold_sensitivity` (`rand_matching.py:1435-1512`)
  already sweeps the grid, and both its `prediction_metrics(...)` call (`:1493`) and
  its `gtin_metrics(...)` call (`:1503` → `:1125`) omit
  `include_graph_diagnostics=False`, although this range introduced that flag and used
  it in the fit lanes (`:1375`, `hpo_metrics.py:550-553`). With 19 grid points ×
  6 GTIN strata rows that is ~114 full graph computations per fold — each one exposed
  to X1/X2. R4 (above) is the same class in the HPO lane.

### 7. Unexpected Behavior / Silent Errors

- X1 (HIGH, NEW): `src/core/graph_diagnostics.py:66-86` runs Tarjan DFS as PYTHON
  RECURSION (`visit()` calling itself), so recursion depth = longest path of the
  accepted candidate component. Measured against the function itself with a synthetic
  frame of the same shape (20 candidates/SKU, `one_missing`, all `rule_ok`):

      skus= 200 edges= 4000 components= 24 max_comp= 1287 time= 0.012s
      skus= 400 edges= 8000 components=  5 max_comp= 2562 time= 0.014s
      skus= 800 edges=16000 components=  1  ->  RecursionError: maximum recursion
                                                depth exceeded

  The dataset is 71,623 rows, so the real calibration/holdout frames are far past
  that point. The function is new in this range and `prediction_metrics`
  (`rand_matching.py:956`, `:1013`) calls it with `include_graph_diagnostics=True`
  **by default**, reachable from the calibration lane (`hpo_metrics.py:530-535`), the
  GTIN strata lane (`rand_matching.py:1125`) and the sensitivity sweep (`:1493`,
  `:1503`). Consequence on a real run: the calibration evaluator raises, X4 converts
  that into "calibration unavailable", and every HPO trial is pruned. Fix: iterative
  DFS with an explicit stack (or union-find).
- X2 (HIGH, NEW): same file — `edge_scores` (`graph_diagnostics.py:93-101`) rebuilds a
  per-component score list by scanning ALL edges once per component ⇒ O(C × E).
  Measured on the sparse regime (each SKU its own GTIN, so C == E; recursion limit
  raised so only cost is visible):

      edges=  4000 components=  4000 time=  1.287s
      edges=  8000 components=  8000 time=  5.733s    (4.5x per doubling)
      edges= 16000 components= 16000 time= 27.055s    (4.7x per doubling)

  Sparse acceptance is exactly the high-threshold end of the configured grid
  (`threshold_min 0.80` / `max 0.98` / `step 0.01`), i.e. where most sensitivity
  points and the holdout threshold sit. Fix: one `groupby(component_id)` pass.
- X4 (HIGH, NEW): `training.py:3540-3564` wraps `evaluate_calibration_trial` in
  `except Exception` and substitutes `unavailable_calibration_metrics(...)` with only
  a print; `training.py:3588-3592` then maps the fold status to
  `"calibration_unavailable"`, which (a) `hpo.py:204-209` filters out of `vals`, so
  the grid CSV is written with a NaN objective and the run continues, and
  (b) `training.py:4323-4336` leaves `proxy_rows` empty, so EVERY Optuna trial raises
  `optuna.TrialPruned("no fold completed")` and the study dies at
  `study.best_params` (`training.py:4476`) with Optuna's "no completed trials" error
  only after the full GPU budget is spent. A programming error in the metric evaluator
  is thus reported as a legitimate SELECTION outcome — the opposite of the
  "fail loudly" stance this file is written to. Fix: catch narrow expected errors only
  and re-raise the rest; never let an evaluator crash become a selection status.
- X6 delivers the same class of silent coupling (guardrail seed drives the calibration
  folds) — see section 4.

## Corrections to Round 1

- **N5's mechanism is wrong.** `_sha256_path` (`rand_matching.py:1822-1833`) cannot hit
  a symlink loop: pathlib's `rglob` does not descend into symlinked directories. The
  real exposure is narrower — a symlinked FILE is hashed through its target, and a file
  plus a symlink to it are both counted. (`path.is_file()` / `is_dir()` on the
  top-level argument do follow symlinks.)
- **N1's framing is incomplete.** The proxy literal is not merely a magic string: it is
  DEAD metadata with a misleading granularity claim (see X11). A config knob alone
  would not make it verifiable.

## Pre-existing, verified while auditing (NOT introduced by this range)

- **Two threshold-grid builders for one config knob:** `_thresholds`
  (`hpo_metrics.py:193-200`) rounds to 10 decimals; `rand_matching.main`
  (`:2093-2101`) rounds to **2**. Identical today at `step=0.01`, but any finer step
  silently collapses the final lane's grid while the calibration lane keeps it.
- **Mislabeled provenance counts:** `hpo_best.json`'s `"n_trials": len(study.trials)`
  (`training.py:4476`) and W&B's `hpo_completed_trials` (`:4496`) both count
  pruned/failed trials as completed (git blame: `16fc531`, before this range).
- **The final lane's calibration input has NO in-repo producer:** nothing in `src/`,
  `scripts/` or `run_all.py` writes a CSV carrying `calibration_fold` — only
  `rand_matching.py` and `notebooks/final_submission.ipynb` consume it. Its column
  contract lives inline in `_load_calibration_frame` (`rand_matching.py:1234-1248`)
  instead of in `schemas.py`, and it is not a `paths.yaml files:` binding. Provenance
  hashes it (good) but records nothing about how it was produced — while `hpo_metrics`
  builds its OWN calibration folds internally. Two calibration-fold constructions, one
  of them untraceable.

## Severity Summary (round 2)

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| HIGH     | 4     | X1, X2, X3, X4 |
| MED      | 6     | X5, X6, X7, X8, X9, X10 |
| LOW      | 7     | X11, X12, X13, X14, X15, X16, X17 |

## Working-tree delta (NOT in GH)

The local checkout currently carries uncommitted fixes for Round 1 items N1, N2, R1,
R3, R5, U2, U5, N5, plus `calibration_non_finite_*` fields, an `lru_cache` on the raw
config read (with a `deepcopy` per call) and a new `folds.partition_component_pairs`.
**None of X1–X17 is addressed there.** Two cautions on that work:
`partition_component_pairs` still re-derives `n_folds = max(2, ceil(1/fraction))` (so
D3's duplicate fold-count rule merely moved module) and contains
`int(round(n_folds * fraction),)`; and the per-call `deepcopy` in `load_config()`
preserves S4/F7's hot-path cost in HPO loops (the YAML parse is cached, the copy is
not). `_sha256_path` gained a symlink guard — correct hardening even though N5's
stated failure mode is not real.

## Recommended next actions (round 2, ordered by blast radius)

1. **X1/X2/X3** — make `candidate_graph_diagnostics` iterative and single-pass
   (`groupby(component_id)`), then pass `include_graph_diagnostics=False` in
   `_fold_sensitivity` and `gtin_metrics` so the diagnostic is computed once per
   reported population instead of ~114× per fold.
2. **X4** — narrow the `except Exception` around `evaluate_calibration_trial`; an
   evaluator crash must fail the fold loudly, not become `pruned`.
3. **X6** — add `hpo.calibration_seed` (a guardrail knob must not re-deal the
   calibration folds, and the main lane does not even request the guardrail).
4. **X5/X7** — single `DIAGNOSTIC_METRIC_COLUMNS` + empty-payload builder, and one
   shared `accepted_mask` for all three gate lanes.
5. **X8** — `@lru_cache` the canonical-record MAP, not just the file read.
6. **X9/X10** — validate `training.base_model` at schema level and make
   `colab.py:975` read it instead of the `"minilm_l6"` literal.

## Root cause & attribution (round 2, per finding)

**Supersedes the `(NEW)` tags above** for X1, X2, X6, X10, X11, X13, X17: digging for
the originating commit proved those are *inherited*, not introduced by this range.
Mechanism was proven for all 17 (by execution for X1/X2, by call-site/status-flow
traces for X3/X4, by whole-file comparison elsewhere); "root cause" below means the
originating decision, with its commit as evidence.

| ID | Mechanism (how proven) | Root cause | Origin | Introduced here? |
|----|------------------------|-----------|--------|------------------|
| X1 | executed the function; `RecursionError` at 800 SKUs | hand-rolled recursive Tarjan written as exploratory code in `hpo_metrics._graph_diagnostics`, then carried into a shared core module without a scaling review | `c87d785 hpo_metrics.py:286-297` | **No — inherited** (905ac80 moved it; the range only widened the graph: `rule_ok` + bridge set + size distribution) |
| X2 | executed; 4.5–4.7× per doubling | same origin: one edge pass per component instead of a group-by; the move added `or component_ids[right] == component_id`, making it strictly more expensive | `c87d785 hpo_metrics.py:311-314` | **No — inherited, slightly worsened** |
| X3 | call-site inventory + grid arithmetic (19 × 6 × folds) | `include_graph_diagnostics: bool = True` promoted a COST-BEARING diagnostic into the shared metric API; a1ab904 then deduplicated only the one instance it could see (2 commits after creating the exposure) | 18b9c81 (flag, default True), extended by 334fb84 | **Yes** |
| X4 | status-flow trace `3554 → 3592 → hpo.py:207 → training.py:4323` | "record unavailable calibration without dropping train folds" was implemented at the CALLER with a blanket `except` plus an overloaded `status` field, so "completed but unmeasured" became indistinguishable from "did not complete" for every status-filtering consumer | 91f0a1d | **Yes** |
| X5 | four-way key-list comparison at HEAD | the diagnostic payload has NO pydantic model (P4), so the derive-from-model technique applied to calibration had nothing to derive from; the range added the keys to `METRIC_COLUMNS` + two payload dicts + a fallback | `METRIC_COLUMNS` gained the 8 diagnostic keys in 905ac80 | **Yes** (the 19-key tuple itself is pre-range) |
| X6 | grep of the seed's readers + config keys | no calibration seed existed, and `_fold_ids` + the collapse guardrail were introduced by the SAME commit, which reused the guardrail's seed | `efcd064 hpo_metrics.py:354` | **No — inherited** |
| X7 | layering check: `src/core/*` never imports `training/*`; the predicate is private inside the frame-mutating `_annotate_candidates` | structural: core CANNOT import the predicate, and no `accepted_mask` API exists — so any new consumer must re-spell it | 2 copies pre-range, 3rd in 905ac80 | 3rd copy **yes** |
| X8 | pre/post comparison of `_canonical_record_map`; 9.4 MB / 13,251 rows | cache responsibility moved DOWN to the reader while the expensive DERIVED structure lost its cache in the same commit | 0dd4fe2 | **Yes** |
| X9 | schema invariant vs the new config key | the default-model change was a call-site migration: no inventory of the other places that encode "the base model" | 6a8d872 | **Yes** |
| X10 | `git log -S` on the literal | same incomplete migration | literal from f8f36f5; inconsistency created by 6a8d872 | inconsistency **yes**, literal no |
| X11 | repo-wide consumer grep + full `-S` history | unowned metadata: authored as a report label, never read by any commit | efcd064 (value renamed in 8440738) | **No — inherited** |
| X12 | contract coverage compared across artifacts | (a) the degraded row was modeled as a DIFFERENT dict instead of the same model with Optional fields (the `collapse_*` fields already used that pattern); (c) the sensitivity rows were serialized into a JSON string column | 91f0a1d (a), 18b9c81 (c) | **Yes** |
| X13 | `git show c87d785:...` proves model + writer pre-exist | provenance was designed as INPUT pinning only — no output/derivation manifest, so the chain stops at the threshold | fdc547a | **No — pre-existing** (the range refactored it and added config/holdout/calibration pins) |
| X14 | pre/post `_candidate_row` comparison | the new helper re-derives the field from the RAW row instead of receiving the caller's already-normalized value, so the caller's `_gtin` normalizer was silently dropped | 5ceb913 | **Yes** |
| X15 | 3 occurrences in one function + the shared helper | interface mismatch: `_candidate_labels` serves threshold fitting (scores + labels arrays), not "is this row the true candidate", so the expression was pasted a third time | 3rd copy in-range | 3rd copy **yes** |
| X16 | both payloads compared, plus the in-file convention | no NA convention applied: the file's own convention for "no rows" is `np.nan` (`rand_matching.py:1090`), but the flag's false branch used typed zeros because the payload mixes ints with a JSON-string field | 18b9c81 + 905ac80 | **Yes** |
| X17 | grep + guard history | the dpi migration was executed file-by-file AND guarded file-by-file (a hand-maintained banned-literal list), so 5 sites escaped; FINDINGS.md:135 already documented "5 literals remain" before this range | 16fc531 | **No — pre-existing and already known** |

### Systemic root causes shared by the batch

1. **Instance-fix instead of class-fix.** Every cost/consistency defect that the batch
   *did* notice was fixed at a single call site: the duplicate diagnostics (a1ab904,
   one instance), the base-model default (6a8d872, entry point only), the candidate
   gate (5ceb913, record built but caller normalizer dropped), the fold carve (D3).
   The class-level audit that would have caught X3/X9/X10/X14 was never run.
2. **Diagnostics were treated as free metadata.** A diagnostic that was previously
   computed once per calibration trial was promoted to a DEFAULT-ON argument of the
   shared metric function (18b9c81) in the same range that was fixing duplicate
   diagnostics — so cost and duplication were managed in opposite directions.
3. **Availability conflated with completion.** One `status` field carries both
   "the fold ran" and "the calibration was measurable" (91f0a1d), and every consumer
   filters on it — which is exactly how X3/X1 turn into "pruned trial", not an error.
4. **Missing contracts at the payload boundary.** The diagnostic payload has no model
   (X5/X12), the seeds/offsets have no registry (X6, U2, FINDINGS E), and the gate
   predicate has no shared API (X7). Each gap forces the next copy.

### Where the root cause is NOT established

- **X2/X1 authorship.** That the algorithm is inherited is proven; *why* one edge pass
  per component was written that way is not recoverable from the repo (no commit
  message, no comment, and the notebook contains none of this code).
- **X12 (a)/(c)** are design choices, not provable defects — I can show the resulting
  inconsistency, not the author's intent.
- **X15's "interface mismatch"** is my reading of the helper's return type; there is no
  commit evidence for the reason it was pasted rather than shared.
- **X9/X10's "no inventory was done"** is inferred from the 4-file list of 6a8d872
  (`config/training.yaml`, `common.py`, `schemas.py`, `train.py`) plus the absence of
  any record in `TODO.md`, `notes.md` or `FINDINGS.md`. A decision recorded outside
  the repo would invalidate this one.

### Correction to Round 1

- Round 1's "FIXED: provenance gap (new `_SubmissionProvenance` with sha256)" is
  **factually wrong**: `_SubmissionProvenance` and its writer exist at `c87d785`
  (`rand_matching.py:112`, `:1447-1468`). The range refactored the writer into
  `_file_provenance`/`_write_provenance` and added config/holdout/calibration input
  pins — an improvement, but not the introduction of provenance (see X13).

## Remediation tracking — current pass

Re-verified independently at `b29fa93` (see **Round 3** at the end of this file
for the executed evidence, root-cause attribution, fix explanation and residual
gaps). Status below is what execution showed, not what the fix commits claim.

| ID | Status at `b29fa93` | Fixed by | Independent evidence |
|----|---------------------|----------|----------------------|
| X1 | **FIXED** | `89d7e41` | at the real retrieval population (71,623 SKUs × `top_k: 20`) the pre-fix code raises `RecursionError`; the fixed code completes it (`max_comp=91,623`, 4.397 s) |
| X2 | **FIXED** | `89d7e41` | old growth 4.0–5.0× per doubling → new ≈linear; 152.8× faster at 16k edges with an identical payload |
| X3 | **FIXED-BUT-INCOMPLETE** | `89d7e41` | Rand lane measured 98 → 0 graph calls per fold; the HPO lane still omits the flag at `hpo_metrics.py:579` and `:1032` → **95 graph calls per trial** (default `True`) |
| X4 | **FIXED (one residual path)** | `89d7e41` | an unexpected evaluator error now raises out of the fold loop and out of `study.optimize`; only the legitimately-empty split still records `unavailable` |

Root cause for X1/X2 was misattributed in Round 2 — the origin is `efcd064`,
not `c87d785` (proof in Round 3).








I reviewed the 6 commits after the last audit round (`41cd50e`), pinned at **`df736a8`** ("reconcile training and rand review fixes"). Method: full diff read, then empirical verification (live repros + `python -m training.selftest`), plus a clean lint delta (HEAD vs `41cd50e`) to avoid reporting pre-existing noise.

**Two important context notes first:**
1. `python -m training.selftest` at `df736a8` → **exit 0, all oracles green** (255 lines, no skips). The new oracles do run.
2. **Another writer is editing this repo right now** — `build_second04_pairs.py`, `rand_matching.py`, `training.py` became dirty at 18:15–18:16 while I worked (nothing was dirty at session start). One of my findings is already being fixed in-flight; I flag that below. Line numbers are for `df736a8`.

---

## HIGH — verified regressions

**H1. `resolve_model` lost its "key *or* subdir" contract; 5 call sites still pass registry *values* → unconditional `KeyError`.**
`common.py:619` now only accepts a registry key or an existing path. Callers that pass the map's *value*:
- `src/training/zero_shot_sims.py:43` — `MODELS = {k: resolve_model(sub) for k, sub in _cfg["models"].items()}`
- `run_all.py:140, 168, 186, 215, 257` — `resolve_model(sub)` / `resolve_model(MODELS["multilingual_l12"])`

Verified live: importing `training.zero_shot_sims` → `KeyError: unknown model registry key 'all-MiniLM-L6-v2'`. This fires **regardless of whether DVC bundles exist** (the value is never a key, so it falls to the `TRAIN_ROOT/<value>` probe at `common.py:632`). The old code did `sub = MODELS.get(key_or_sub, key_or_sub)`, which is why this worked before. Impact: `--what sims` (`colab.py:2052`) and every `run_all.py` step die at import.

**H2. `colab.py:977` — the graceful "uniformity skipped" branch is now dead, and local finalization hard-fails.**
```python
base_model = Path(resolve_model(str(training_cfg().training.base_model)))
if not base_model.is_dir():      # ← unreachable: resolve_model raises instead
    ... "status": "skipped_base_model_unavailable"
```
`resolve_model` can no longer return a non-directory, so `base_model.is_dir()` is always true. Verified: `artifacts/models/` is empty locally and `resolve_model('minilm_l6')` raises `FileNotFoundError`. Since `uniformity.enabled: true`, `finalize_local_training_run` (`colab.py:1421`) now aborts before `publish_local_training_results`/`publish_local_wandb_artifacts`, and `main()`'s `except BaseException` keeps the VM alive for recovery — the same "quota burn" class as the previously-reported N1. Before this series: hub id → `Path(hub).is_dir()` false → skip branch → report still generated.

**H3. 07c/07d are now un-appendable against any existing artifact (`train.py:159, 193-226`).**
`_append_csv` raises when a key field in `key_fields` is absent from the *old* file. The new calls pass `["variant", *_provenance]` / `["fraction", "payload", *_provenance]`, and pre-existing files lack those columns. Verified against the repo's own files:
```
training_results/20260912T152315Z/worker_1/07c_field_ablation.csv →
ValueError: ... is missing required replace-key fields
['split','holdout_component_folds','calibration_seed_offset','seed']
```
Same for `07d_data_scaling.csv`. There is no backfill/migration, so the 07c/07d emission crashes mid-run on every pre-existing results tree. (The fresh-file path works; only pre-existing files break.) Note the existing `model` column holds `sentence-transformers/all-MiniLM-L6-v2` — evidence of the old resolution contract.

---

## MED

**M1. The recall-label fix is lossy and the provenance key omits the value that matters.** `recall_column_suffix` (`common.py:313`) rounds to whole percent — verified: `0.895/0.899/0.904 → '90pct'`, `0.995/0.999 → '100pct'`. `_provenance` (`train.py:94`) records `split`, `holdout_component_folds`, `calibration_seed_offset`, `seed` but **not `target_recall`**. So a sub-percent retune (0.90 → 0.904) keeps the same column names *and* the same replace key → the new numbers silently overwrite the old row under the old header. That is exactly the mislabeling the change claims to have closed (it only closes it for changes that cross a whole-percent boundary).

**M2. `holdout_split` silently yields an EMPTY train split for `n_folds=2` (`folds.py:93-125`).** The guard is `if n_folds < 2: raise`, but the derivation is `set().union(*quarters[:-2])` — for `n_folds=2` that is `set()` (empty). Verified: `n_folds=2 → train=0 dev=4 test=4`, no error. `Literal[4]` pins it in config today, but the helper documents/accepts any `n_folds` and both `train.py` and the selftest pass the config value straight through. Guard should be `< 3` or assert a non-empty train.

**M3. The registry is now heterogeneous but two consumers iterate it wholesale.** `paths.yaml` adds `rerank_minilm_l6` (ms-marco **cross-encoder**), `ner_semantic_base`, `ner_transformer_base` to the same `models` map that `report_plots.py:66` and `zero_shot_sims.py:43` expand over and feed to `encode_corpus`/`SentenceTransformer`. Consequences: (a) `report_plots` now requires all 6 bundles at import — verified it fails at import today; (b) NER/cross-encoder bundles get encoded as bi-encoders (wrong panels or a load failure). Compounding it, `_validate_materialized_model` accepts `modules.json` **or** `config.json`, so a plain HF dir (`xlm-roberta-base`) passes the gate the consumers actually need.

**M4. `materialize_remote_models` docstring ≠ behavior (`colab.py:1710-1722`).** "Pull and validate only the model bundles required by this lane", but the body runs `dvc pull artifacts/models` (the whole tree); the per-key loop only *validates* resolution afterwards. The stated locality contract isn't enforced.

**M5. `sims_model` is the only model knob not registry-checked at config load.** `ColabSpec.sims_model` is `Field(min_length=1)` (`schemas.py:942`), while `training.base_model` (`common.py:176`), `hpo.models` (`:181`) and `sweep.rerank_model` (`:188`) are all validated against the registry. A typo now surfaces late (on the VM path via `materialize_remote_models`) instead of at load.

**M6. Pydantic coverage gaps in the *new* code:**
- `CalibrationPartition` (`schemas.py:1446`) has no `extra="forbid"`, and `zip(("positive_fit", …), self.pools())` (`:1498`) lacks `strict=True` — if `pools()` ever returned fewer than 4 arrays, the per-pool shape check would silently skip the tail (this is one of the two *new* lint errors).
- `check_cross_country_pair_frame`'s `if len(rows) != len(df)` (`schemas.py:1611-1614`) is **vacuous** — `rows` is built 1:1 from `df.to_dict("records")`, so it can never fire. It is precisely the "guard that cannot fail" that the same commit deleted from `folds.py` with a comment about vacuous checks.
- `CalibrationMetricRow` embeds the new diagnostics as JSON **strings** (`calibration_sensitivity_by_gtin_status`, `calibration_fold_collapse`, `calibration_threshold_tie_break`), so a malformed payload is unvalidatable at the aggregate boundary.
- `CALIBRATION_UNAVAILABLE_REASON_CODES` (`hpo_metrics.py:237`) maps numeric codes from free-text prefixes duplicated from `training.py:3563/3586`. Any wording change silently degrades to code 0 ("unclassified") — no failure, no warning.

---

## LOW / hygiene

- **`threshold_tie_break` is a decorative config knob.** `RandMatchingSpec` (`schemas.py:413`) hard-rejects any order other than `["rand_index","fewest_unmatched_skus","lowest_threshold"]`, and `_threshold_selection_key` hardcodes that exact mapping (`values[name]`, `KeyError` on any new name). Configurable in name only.
- **3 new lint errors** (clean HEAD-vs-base comparison on the changed files): `colab.py` F401 + F811 (module-level `resolve_model` at `:67` is unused because `:929` re-imports it locally), `schemas.py` B905. Two were fixed (`hpo_metrics` GTIN_STATUSES, `train.py` training_cfg now used).
- **Dead branch:** `_validate_materialized_model`'s inner `if not resolved.is_dir(): raise FileNotFoundError` is unreachable — callers only invoke it on `candidate.is_dir()`.
- **Metric provenance drift:** `args.model` is now an absolute local path (`train.py:521`), so `row_07c["model"]`, the MLflow `model` param and the run tag now embed machine-specific paths instead of a portable id.
- **CLI contract divergence:** `train.py --model` accepts a key *or* a local path; `colab.py --what train --model` accepts a registry **key** only (and passes the key through). Same flag, two contracts.
- **Solid, not a finding:** `_write_datapoint_usage`'s A4 fix is correct (coverage aggregated from presentations only; `missing` status + `n_missing_datapoint_populations`); `_usage_row`'s "fill only absent keys" merge is correct; `build_second04_pairs` is deterministic (`mergesort` on `barcode,product_id,country`); `volume_verified` now counts unresolved ids and the volume gate is counted (no silent drops there); RM1/RM3/RM8/RM9 are genuinely closed; `_fold_collapse_stats`'s payload/row alignment and its `source_rows` guard are sound.

---

## On the uncommitted, concurrent edits

`build_second04_pairs.py`, `rand_matching.py` and `training.py` were modified (18:15–18:16) while I reviewed. That in-flight work adds: an `ExclusionCensus` closing the invalid-GTIN/blank-field drops I had flagged as an unaccounted silent filter (so that item is already being fixed), `source_row_index` identity preservation through merges, a `Decimal`-based threshold grid, and `RequiredCalibrationError` making unavailable calibration **fatal in non-selection mode** (`training.py:3573-3612`). That last one is a significant new fail-loud path — worth confirming it can't strand a holdout run when DEV has too few positives, and none of it is covered by the oracles I ran (those were at `df736a8`).

Suggested order if you want fixes: **H1/H2** (broken lanes, VM-strand risk) → **H3** (un-appendable artifacts) → **M1/M2** (silent mislabel + degenerate split) → M5/M6 (schema coverage). I did not modify any repository file; all repro artifacts are under `/tmp/repro/`.

---

# Round 3 — HIGH findings X1–X4 independently re-verified

## Scope, revision, method

- **Target:** branch `training` at **`b29fa93`** (three commits landed after the pass that
  produced X1–X4: `121c3fc`, `89d7e41`, `b29fa93`).
- **Working-tree state at verification:** dirty in `FINDINGS.md`, `src/cli/colab.py`,
  `src/training/hpo_metrics.py`, `src/training/selftest.py`, `src/training/zero_shot_sims.py`.
  None of those edits touches the graph diagnostic, the diagnostics-flag discipline or the
  calibration error path, so X1–X4 were verified against **committed** code.
- **Fix attribution:** all four closed by `89d7e41` "harden graph diagnostics and calibration
  failures". `b29fa93` additionally collapsed the diagnostic key set to ONE definition
  (`CANDIDATE_GRAPH_METRIC_COLUMNS` + `empty_candidate_graph_diagnostics()`, consumed by
  `rand_matching.METRIC_COLUMNS`, the empty-accepted payload and the `include_graph_diagnostics=False`
  fallback — the X5/X16 class), and added four oracles, **none of which covers X1–X4**.
- **Method:** the PRE-fix implementation was extracted from `df736a8`
  (`git show df736a8:src/core/graph_diagnostics.py`) and run **side by side** with the fixed one.
  That gives both the failure reproduction and a semantic-equivalence check. Plus: real-data runs on
  the repo's own artifacts, `git log -S`/`git show` provenance, and a full selftest run.

## Verdict

| ID | Status at `b29fa93` | Fixed by | Decisive evidence |
|----|---------------------|----------|-------------------|
| X1 | **FIXED** (active at production scale) | `89d7e41` | at the real retrieval population (71,623 SKUs × `top_k: 20` = 1,432,460 rows) the pre-fix code raises `RecursionError` and the fixed code returns `max_comp=91,623` in 4.397 s; 1,058-frame correctness check, 0 mismatches |
| X2 | **FIXED** | `89d7e41` | on the real 271,538-edge frame: old 3.524 / 2.341 / 2.358 s vs new 0.097 / 0.112 / 0.072 s (36× / 21× / 33×) |
| X3 | **FIXED-BUT-INCOMPLETE** | `89d7e41` | Rand `_fold_sensitivity` measured 98 → 0 graph calls/fold; `hpo_metrics.py:579` + `:1032` omit the flag → **95 graph calls/trial** (default `True`) |
| X4 | **FIXED-BUT-INCOMPLETE** | `89d7e41` | unexpected evaluator error now raises out of the fold loop **and out of `study.optimize`**; the legitimately-empty split is still silent-and-prunable in selection mode (independent-agent probe, §X4) |

## Real data used (read-only)

The repo carries real artifacts, so the two performance/crash claims were re-measured on real data,
not only on synthetic frames:

- `results/gate_results.csv` — 135,769 real rows
  (`gtin1,gtin2,canon1,canon2,gate_decision,gate_reason,similarity`).
- Rebuilt as the real candidate frame: **271,538 edges** (both orientations), **12,952 SKUs**,
  **12,279 canonicals**, real `similarity` scores, `rule_ok` from the real `gate_decision`
  (58,038 proceed / 213,500 other).
- `results/canonical_records.csv` — 13,250 real canonicals. `dataset.csv` — the real source dataset:
  **71,623 rows** (14,998 GTINs) spread over 198,870 physical lines (text fields contain embedded
  newlines, so a line count is not a row count).

### X2 on real data (decisive)

| threshold | accepted edges | components | max component | OLD | NEW | speedup |
|-----------|----------------|------------|---------------|-----|-----|---------|
| 0.80 | 14,660 | 2,449 | 84 | 3.524 s | 0.097 s | 36.3× |
| 0.90 | 9,964 | 2,250 | 40 | 2.341 s | 0.112 s | 20.9× |
| 0.98 | 9,964 | 2,250 | 40 | 2.358 s | 0.072 s | 32.8× |

The old cost tracks `components × edges` (2,449 × 14,660 ≈ 3.5 s), which is exactly the O(C×E)
signature the finding predicted; the new cost is flat in the component count. Payloads agree key by key.

### X1 on real data — and a correction to this pass's own first conclusion

My first real-data run (the gate-derived frame above: 12,952 SKUs, 271,538 edges, **max component 84
nodes**) did not crash the old code, and I initially recorded X1 as "latent, not active". **That was
wrong for the population the lane actually grades.** At the real retrieval population —
**71,623 SKUs × `top_k: 20` (config/training.yaml:143) = 1,432,460 rows** — the old implementation
raises `RecursionError`, while the new one returns `max_comp=91,623` in **4.397 s** (independent-agent
run). The crash is a property of the RETRIEVAL candidate graph (20 candidates per SKU), not of the
gate-pair graph I first measured, which is why my first frame looked benign.

The failure tracks **DFS depth, not component size**: the old code survived one draw with
`max_comp = 7,087` yet died on a smaller component. Depth evidence:

| shape | old | new |
|-------|-----|-----|
| chain, 100 SKUs | ok | ok |
| chain, 500 SKUs | **`RecursionError`** | ok |
| chain, 100,000 SKUs (200,001 nodes) | — | ok, 1.035 s, bridges=199,999 |
| star (1 GTIN, 100k SKUs) | — | ok, 0.428 s |
| ring, 100k × 2 edges | — | ok, 1.550 s |

No `RecursionError` handler and no `sys.setrecursionlimit` call exists anywhere in `src/`, so the old
defect propagated uncaught all the way up.

**Correctness of the replacement (independent-agent, exhaustive):** 1,058 frames compared against a
self-written reference (union-find components + brute-force "remove the edge, re-test connectivity"
bridges) — 24 randomized, 6 targeted multigraph, 300 random small (230 with parallel edges, 214 with
bridges), and 728 exhaustive multigraphs (3 SKUs × 2 GTINs, multiplicities 0–2): **0 mismatches**.
Parallel edges are handled correctly (the second parallel edge is a back edge, so
`low[child] > discovery[parent]` is false); self-loops are structurally impossible because nodes are
namespaced `sku:`/`gtin:`, which is also why the single `parent_edge` slot is sufficient.

**Stale consequence description (cross-item):** X1's write-up ends "the calibration evaluator raises,
X4 converts that into *calibration unavailable*, and every HPO trial is pruned". That chain no longer
holds — `89d7e41` fixed X4, so a `RecursionError` would now abort the lane loudly through
`CalibrationEvaluatorError`. The consequence is gone; the defect it described is not.

### Semantic equivalence (both fixes)

- Pre-fix vs fixed implementation on **180 randomized (frame, threshold) pairs**, all 8 metric keys
  (`diagnostic_edge_count`, `_component_count`, `_component_size_distribution`, `_max_component_size`,
  `_score_diameter`, `_bridge_edge_count`, `_weakest_bridge_score`, `plausible_group_count`):
  **0 mismatches**, no `NaN != NaN` artifacts.
- On the real frame, old and new agree on `component_count`, `max_component_size` and `edge_count`
  at every threshold tested. The fixes changed cost, not semantics.

### X3 on the current code — FIXED-BUT-INCOMPLETE

The **Rand lane is fixed; the HPO lane is not.** `include_graph_diagnostics=False` is passed explicitly
at `rand_matching.py:1438`, `:1563`, `:1575` (the `_fit_fold_threshold` / `_fold_sensitivity` sites) and
at `hpo_metrics.py:887`, `:921` — but **two `gtin_metrics` calls still omit the flag** and therefore
fall back to the `bool = True` default (`rand_matching.py:1158`):

| site | flag | default | graph calls |
|------|------|---------|-------------|
| `hpo_metrics.py:579` `_stratified_sensitivity_rows` → `gtin_metrics` | **omitted** | `True` | **95 per trial** (19 thresholds × 5 strata) |
| `hpo_metrics.py:1032` trial `strata_rows` | **omitted** | `True` | **5 per trial** |
| `rand_matching.py:1895` holdout `gtin_metrics` | omitted | `True` | 5 per holdout (arguably intended) |
| `rand_matching.py:1438/1563/1575`, `hpo_metrics.py:887/921` | explicit `False` | — | 0 ✓ |

Measured with an instrumented `candidate_graph_diagnostics` (same frame, same call, flag forced `True`
to emulate `89d7e41^` — independent-agent probe):

```
RAND _fit_fold_threshold (19 thr)        calls=0
RAND _fold_sensitivity (19 thr, 3 alts)  calls=0      << the X3 fix
HPO per-trial threshold x strata loop    calls=95     graph_s=0.678  (8,000-row frame)
BEFORE (flag forced True, same call)     calls=98  = 19x5 + 3 alternatives per fold
```

Per-call cost of the real function: 20,000 rows **20.88 ms**; 100,000 **210.71 ms**;
400,000 **486.17 ms**; **1,432,460 rows (= 71,623 SKUs × `top_k: 20`) 2,057.82 ms**.
**Measured:** 95 calls = 0.678 s on an 8,000-row frame. **Extrapolated (not measured):** ≈**195 s of pure
graph time per HPO trial** at full-population frame size, ≈**65 min across the configured 20 trials**.

**Correction to X3's arithmetic:** the strata count is **5**, not 6 — `GTIN_STATUSES` is a 4-tuple and
`gtin_metrics` appends one `"ALL"` row (confirmed by execution: BOTH_EQUAL / DIFFERENT / ONE_MISSING /
BOTH_MISSING / ALL). The pre-fix Rand-lane figure is the **measured 98 per fold**, not ~114.

**Transparency regression introduced by the partial fix:** because `empty_candidate_graph_diagnostics()`
is substituted whenever the flag is off, `threshold_sensitivity_by_gtin_status.csv` and
`threshold_comparison.csv` (config/training.yaml:135/137) now carry **zeros**
(`0/0/"{}"/0/0.0/0/NaN/0`) where the diagnostic used to be computed. Nothing breaks — 28 keys are still
returned with the flag on or off, the `tuple(metrics) != METRIC_COLUMNS` assertion holds,
`CalibrationSensitivityRow` has no diagnostic fields, and no in-repo consumer reads those columns — but
"not computed" is again indistinguishable from "computed and zero" in shipped output (the X16 class,
now realized in real artifacts). Real values survive only on the trial `overall` row and the holdout report.

**Also still redundant (independent of the flag):** `gtin_metrics` rebuilds
`candidates[candidates.SKU_ID.isin(group.SKU_ID)]` for all 5 strata on *every* threshold
(`rand_matching.py:1203-1208`) although those masks are threshold-independent — 5 redundant boolean
passes × 19 thresholds per trial. Hoisting them out of the loop is the second half of a complete fix.

### X4 on the current code

`89d7e41` replaced the blanket `except Exception` around `evaluate_calibration_trial` with a named
failure: `CalibrationEvaluatorError` (`training.py:142`, raised at `:3609`) with fold context, and the
fold loop re-raises it instead of writing a `status="failed"` row (`training.py:4253`). `study.optimize`
(`training.py:4515`) passes **no `catch=`**, so an evaluator crash now escapes the study rather than
being recorded as a trial outcome — the fail-loud intent of X4. `TrialPruned("no fold completed")`
(`:4352`) is now reachable only for folds that completed with a non-`ok` status, i.e. the
legitimately-empty calibration split, which is the one case that *should* be a selection outcome.

**Residual path (why this item is FIXED-BUT-INCOMPLETE):** the legitimately-empty split is still
silent and prunable in selection mode. Independent-agent probe at `b29fa93` (the block under
`training.py:3568-3611` was extracted from the on-disk file and exec'd, so the shipped text ran):

| outcome | selection mode | holdout lane |
|---------|----------------|--------------|
| empty split | row `status="calibration_unavailable"` + `**calibration_metrics`; **no print** (the print sits inside `if not selection_mode:` at `:3579`) | print + `raise RequiredCalibrationError` (`:3579-3587`) |
| unexpected error (`KeyError`/`TypeError` from the evaluator) | `raise CalibrationEvaluatorError(...) from exc` (`:3608-3611`) with `__cause__` preserved | same |

and the consumer side, also probed: `hpo.py:204-208` drops every non-`"ok"` row, so the grid summary
row is written with an **empty** objective (`mean,,,,,,,h0,calibration_rand_index,`) and the grid
continues; the Optuna lane reaches `TrialPruned("no fold completed")` (`:4352`) and, with all trials
pruned, dies at `study.best_params` (`:4534`) **after the whole budget**, with
`ValueError: No trials are completed yet.` A probe confirmed `study.optimize` (no `catch=`, `:4515`)
propagates an unexpected `KeyError` immediately — so the fix does work for real bugs; only the
empty-split path still spends the budget before failing.

**X4 in the REAL artifacts — the class did fire in production, just not as `unavailable`.**
Independent-agent sweep plus my own re-verification of the repo's shipped outputs:

- Aggregate across every `*fold_metrics*.csv` in the checkout (**73 files**): **52 `ok`, 16 `failed`,
  5 `skipped`**. `calibration_status` exists in **5** files, and in all 5 it reads `available`;
  `calibration_reason` / `calibration_reason_code` exist in **0** files, and in `artifacts/mlruns/mlflow.db`
  (104 runs, 645 metrics, 49 distinct `calibration*` keys) `calibration_available` matches **0** rows.
  So the post-`91f0a1d` *availability status* never shipped — but the swallow underneath it did.
- The smoking gun is a real run artifact:
  `artifacts/mlruns/artifacts/e1a66d8e2c4e4e8ba0105fce1561734e/artifacts/train_checkpoint-5_holdout_full_full_sample100_fold_metrics.csv`
  — **1 row, `status="failed"`, zero calibration columns**, traceback ending

      File ".../src/training/training.py", line 3539, in train_one_config
        calibration_metrics = evaluate_calibration_trial(
      File ".../src/training/hpo_metrics.py", line 484, in evaluate_calibration_trial
        collapse = _collapse_stats(model, df, payload, config, ba...
      File ".../src/training/hpo_metrics.py", line 362, in _collapse_stats
        raise RuntimeError(
      RuntimeError: collapse guardrail could not construct its configured unrelated-pair sample: required=100 selected=36

  i.e. a calibration-evaluator crash recorded as a fold status while the MLflow run finished
  **FINISHED** with **0 metrics logged** (source commit `6a8d872`, i.e. **before** `91f0a1d` — the
  pre-fix shape of the same conflation). That exact `RuntimeError` condition was later softened to
  `collapse_status="insufficient_pairs"` + print, and now fires only at `selected=0`.
- **The residual live mechanism is not the calibration branch any more — it is
  `training.py:4248-4254`**, which still converts any non-calibration fold crash into a
  `status="failed"` row. **16** such rows exist in the real artifacts, and both consumers
  (`hpo.py:204-208`, `training.py:4350`) filter non-`ok` rows out → same NaN-objective / pruned-trial
  path. Closing X4 completely means handling that envelope too.
- **Reader caution (real schema):** NaN is *not* an unavailability marker. A healthy shipped row
  (`results/train_all-MiniLM-L6-v2_holdout_full_full_sample100_fold_metrics.csv`, 173 columns, 58
  `calibration_*`) carries `calibration_status=available`, `calibration_non_finite_count=9` and 9 NaN
  cells in the `calibration_different_*` / `calibration_both_missing_*` strata. The real marker is the
  `calibration_status` value, plus the `calibration_reason*` columns when a row carries them at all.

**Correction to this pass's own M6/residual claim:** the reason-code lookup is NOT wording-dependent at
`b29fa93`. `hpo_metrics._unavailable_reason_code` now reads the typed `calibration_reason_code` field
and falls back to `CALIBRATION_UNAVAILABLE_REASON_UNCLASSIFIED`; the prefix-matching version I flagged
was real at `df736a8` (`reason.startswith(prefix)`, 1 occurrence) and was replaced in **`121c3fc`**
(0 occurrences from that commit on). No `startswith` on reason text remains anywhere — the only
`startswith` calls left in that file are metric-key prefix filters. The docstring still calls the
reason "free text", which is now stale. Two related leftovers:

- `CALIBRATION_REASON_EVALUATOR_FAILED` (`hpo_metrics.py:54`) now has **zero producers** — dead code,
  so a crash is no longer distinguishable by reason code at all.
- `numeric_calibration_metrics`'s `calibration_available=0` / `calibration_unavailable_reason_code`
  is **unreachable from the training lane**: its only caller is `train.py:1532/1534` inside
  `_emit_07_series`, fed exclusively by `status == "ok"` rows (`train.py:1440`). The "never measured
  vs measured clean" distinction that 03-3 set out to create therefore does not reach the artifacts.

Other blanket `except` blocks that still convert a metric/evaluator error into a status:
`training.py:4248-4254` (any fold crash that is not a `RequiredCalibrationError`/`CalibrationEvaluatorError`
becomes a `status="failed"` row — which the Optuna lane then prunes), `training.py:3083`
(discriminative-LR → `lr_groups="single"`), `training.py:793` (telemetry, harmless),
`train.py:1330` / `:1471` (mask-effect and report dumps, traceback printed/written post-training).

## Root cause (with the Round-2 misattribution corrected)

| ID | Root cause | Origin | Round 2 said |
|----|-----------|--------|--------------|
| X1 | a hand-rolled **recursive** Tarjan written as exploratory code inside the HPO metrics module, then promoted into a shared core module without a scaling review | **`efcd064`** (`src/training/hpo_metrics.py:286` `def visit`), moved to core by `905ac80` — which ADDED `src/core/graph_diagnostics.py` (122 lines, `--diff-filter=A`) carrying the recursion with it | `c87d785` — **wrong** |
| X2 | same origin: one full edge scan **per component** instead of one bucketing pass; the move to core added the reverse-orientation test, making it strictly more expensive | **`efcd064`** (`src/training/hpo_metrics.py:311-315`) | `c87d785` — **wrong** |
| X3 | a cost-bearing diagnostic was promoted into the shared metric API and then switched off one call site at a time, so the FIX inherited the same per-site discipline that created the exposure | `18b9c81` introduced the flag with default `True` and **explicitly passed `True`** in the HPO per-threshold loop (so that exposure was an opt-in, not a default); `gtin_metrics` received the parameter only in `89d7e41` **with default `True`**, which fixed the Rand sites and left `hpo_metrics.py:579`/`:1032` (and `rand_matching.py:1895`) on the default; `334fb84`/`41cd50e` closed the fit and trial-sensitivity sites | `18b9c81` — **partly confirmed** |
| X4 | "record unavailable calibration without dropping train folds" was implemented at the CALLER as a blanket `except` plus an overloaded `status`, so "completed but unmeasured" became indistinguishable from "did not complete" for every status-filtering consumer | `91f0a1d` | `91f0a1d` — **confirmed** |

**Proof of the X1/X2 misattribution:** `c87d785` ("align configured git remote name") touches only
`config/training.yaml`, `src/cli/colab.py` and `src/core/schemas.py` — not `hpo_metrics.py`. And the
file at `c87d785~1` **already contains** `def visit` (`grep -c` = 1), so the function cannot have been
introduced there. The line numbers Round 2 quoted (286-297, 311-314) match the file as it stands in
`efcd064`, which is the commit that ADDED `hpo_metrics.py` with 456 lines.

## The fixes, explained

- **X1** — the recursive `visit(node, parent_edge, component)` is replaced by an explicit frame stack
  `(node, parent_edge, next_adjacency)` that emulates the call stack. Entering a node pushes a frame;
  a frame whose adjacency is exhausted is popped and performs the parent's post-order low-link update
  (`low[parent] = min(low[parent], low[node])`, bridge test `low[node] > discovery[parent]`) before
  control returns. Depth is now heap-bounded, so component size can no longer hit Python's recursion
  limit; semantics are byte-identical (180-pair equivalence check).
- **X2** — instead of rescanning all edges once per component, each edge is bucketed into its
  component in a single pass (`edge_scores[component_ids[left]].append(score)`), which turns the
  component-score aggregation from O(C×E) into O(E) and is what produces the measured 21–36×
  real-data speedup.
- **X3** — rather than changing the shared default (which would have silently dropped diagnostics
  where they are wanted), the sweep and strata call sites now request `include_graph_diagnostics=False`
  explicitly, so the diagnostic is computed once per reported population instead of once per
  threshold × stratum.
- **X4** — the blanket `except Exception` is replaced by a named `CalibrationEvaluatorError` carrying
  fold context, re-raised through the fold loop so it escapes `study.optimize`; only the explicit
  empty-population branch remains a legitimate `unavailable` status.

## Verification-coverage gap (applies to all four)

`python -m training.selftest` at `b29fa93`: **241 PASS, 0 FAIL, 0 SKIP, exit 0**. But the four oracles
added by `b29fa93` (`oracle_holdout_validation`, `oracle_recall_provenance_identity`,
`oracle_model_resolution_contract`, `oracle_embedding_consumer_allowlist`) cover unrelated ground.
**Nothing in the suite exercises the graph traversal, the edge-to-component mapping, the
`include_graph_diagnostics` discipline, or the calibration error path** — so X1–X4 are fixed but
unguarded, and a future refactor can reintroduce any of them silently. Recommended: a small oracle
that (a) runs `candidate_graph_diagnostics` on a chain component larger than the recursion limit,
(b) compares it against a union-find/brute-force reference, and (c) asserts an evaluator exception
propagates instead of becoming a status.

**Also uncovered by construction:** no real retrieval-derived candidate frame is persisted anywhere
(`results/` holds `gate_results.csv`, `canonical_records.csv`, fold metrics — no candidate/diagnostics
CSV with the gate columns), so the real per-fold cost of X3 and the real maximum component depth of X1
can only be estimated from a reconstructed frame, never verified from the artifacts the lane actually
produces. Persisting one real candidate frame per run would close that gap.