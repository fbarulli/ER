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

- N1 (LOW): `calibration_proxy_source: "dev_component_safe_split"` is a hardcoded
  string literal in hpo_metrics.py:608. It is a data label, not a path, but it is
  a magic string that changed value mid-range ("dev_component_safe_subsplit" →
  "dev_component_safe_split") — a config knob or constant would prevent silent
  drift between the two calibration producers (hpo_metrics vs rand_matching).
- N2 (LOW): `_sweep_assignments` / `_assignments_with_trace` tie-break column
  order `["SKU_ID", "exact_gtin", "score", "attribute_matches", "candidate_gtin"]`
  is duplicated in TWO places (rand_matching.py:689-692 and :1292-1295) as an
  inline literal — should be a shared constant (SSOT for the assignment sort).
- N3 (LOW): `dvc_jobs`/`dvc_workers` split is config-driven (good), but
  `_publish_local_hpo_model_snapshot` hardcodes `optuna_db=None` and the model
  snapshot scope = `model_dir.name` (colab.py:1429-1432) — behavior knobs, not
  paths; minor.
- N5 (LOW, NEW): `_sha256_path` determinism relies on `sorted(child.rglob("*"))`
  (rand_matching.py:1786) — symlinks/duplicate paths inside a checkpoint dir can
  silently change the digest semantics across machines (dir fingerprint is
  path+content-ordered, but a symlink loop would recurse). Guard with a file-only
  filter + no-follow; current code `is_file()` follows symlinks.

### 2. Missing Pydantic Validation

- P4 (LOW, residual): `candidate_graph_diagnostics` accepts any DataFrame and
  accesses `gtin_status/exact_gtin/rule_ok/score` columns with raw pandas —
  no pydantic column contract; a wrong-shaped candidates frame fails with
  KeyError mid-function rather than a named contract error. Consistent with
  prior FINDINGS stance; LOW.

### 3. Gaps in Data Traceability

- T5 (MED, residual): the audit trace merge guards are loud (raise on population
  mismatch, rand_matching.py:746-755, 767-768, 1252-1260) — GOOD; but
  `_audit_trace(..., truth=None)` for the final submission labels every row
  `error_type="unlabeled"` (rand_matching.py:1329) — the FINAL submission CSV is
  the one artifact that will never carry a label audit; acceptable but the
  diagnostics CSV for "final" partition has empty `true_*`/`prediction_correct`
  columns that a downstream consumer could mistake for "correct" — column values
  are `""` (empty string), not NA — mildly misleading. LOW-MED.

### 4. SSOT (Single Source of Truth) Violations

- S4 (LOW, NEW): `calibration_fraction` is read from config INSIDE
  `train_one_config` (training.py:2261-2263 `calibration_config = load_config()`)
  on every call — `load_config()` is UNCACHED (F7 from FINDINGS still open) and
  now called per-config in HPO loops (hpo.py grid/tpe → train_one_config per
  trial). The config re-read is a hot-path redundancy; the value itself could be
  a module constant like `_TRAIN_CFG` (common.py already caches `_CFG`).

### 5. Missing Centralized Functions / Duplicated Processes

- D3 (MED, NEW): `_partition_calibration_pairs` (training.py:2142-2189)
  re-implements a component-safe carve using `component_folds` with a
  RE-DERIVED fold count `n_folds = max(2, ceil(1/calibration_fraction))` —
  the calibration "50% of dev" is approximate (rounding + component deal) and
  duplicates the fold machinery used elsewhere; the actual calibration share
  can be anywhere near 0.5; empty calibration is now represented explicitly.
  A single `partition_by_fraction(pairs, barcodes, fraction, seed)` helper
  shared with the split block would be the SSOT.
- D4 (LOW, NEW): `_write_calibration_outputs` + `write_outputs` +
  `_write_final_submission` each re-validate frame contracts with three
  different spec classes (`_SubmissionColumnSpec`, `_DiagnosticsColumnSpec`,
  `_MetricColumnSpec`) — all in rand_matching.py:151-189; contract DRY is fine
  (three distinct shapes), but `_MetricColumnSpec.validate_frame` only checks
  REQUIRED columns (missing) while `_DiagnosticsColumnSpec` checks exact set —
  asymmetric strictness; LOW.

### 6. Redundant Operations

- R1 (MED, NEW): `evaluate_calibration_trial` computes `overall` via
  `_assignment_metrics(validation_candidate_frame, ..., final_threshold,
  include_graph_diagnostics=True)` (hpo_metrics.py:577-582) AND THEN the
  `sensitivity` sweep recomputes `_assignment_metrics(...)` for EVERY threshold
  INCLUDING final_threshold again (hpo_metrics.py:583-593). The final_threshold
  row is computed twice per trial.
- R3 (MED, NEW): `_emit_07_series` calls `agg(field)` for ~55
  CALIBRATION_AGGREGATE_FIELDS per row across TWO rows (07c + 07d,
  train.py:109-114 and :137-142) — `agg` is O(n_folds) per field, so the whole
  block is O(fields × folds) twice; trivial per-run, but the two rows duplicate
  the entire loop (could compute the means once and reuse).
- R4 (LOW, residual): RM5 (choose_assignments re-annotates whole frame per
  threshold) — VERIFIED STILL OPEN in rand_matching: `_sweep_assignments`
  (rand_matching.py:1263) exists for the calibration lane, but
  `hpo_metrics._fit_threshold` (hpo_metrics.py:400-403) still calls
  `_assignment_metrics` → `choose_assignments` → `_assignments_with_trace`
  (full annotate+sort) PER THRESHOLD inside the fold loop — the HPO lane pays
  O(thresholds × frame) annotation cost while the final lane got the optimized
  sweep. Not unified.
- R5 (LOW): `calibration_threshold_fold_median` is assigned the same value as
  `calibrated_threshold` (hpo_metrics.py:613-614) — redundant field, always
  equal by construction.

### 7. Unexpected Behavior / Silent Errors

- U2 (MED, NEW): `_partition_calibration_pairs` uses `seed + fold_i + 17`
  (training.py:2514) — a magic RNG offset constant (+17) with no config or
  constant name; same class as the FINDINGS E "RNG offsets" item.
- U5 (LOW, NEW): `CalibrationMetricRow` allows NaN floats (e.g.
  `calibration_youden_threshold` when a fold has no positives → `_youden_threshold`
  returns NaN, hpo_metrics.py:634-641), and `numeric_calibration_metrics` filters
  non-finite values SILENTLY (hpo_metrics.py:132-133) — a NaN calibration metric
  disappears from tracking without a flag. Same class as FINDINGS D "mlflow/wandb
  drop non-finite silently" — still open, now on the calibration fields.
- U6 (LOW, residual): the plateau diagnostic flags
  `degenerate_unmatched_plateau` (rand_matching.py:1498-1501), but selection
  still prefers the LARGEST threshold on ties (rand_matching.py:1342). The
  degenerate case is surfaced but not prevented.

## Cross-Range Root-Cause Analysis

The original 8-commit range was followed by closure commits that addressed the
calibration availability path, shared candidate gates, threshold selection,
canonical-record loading, duplicate graph diagnostics, and strict intermediate
metric contracts. The remaining review items below are residual design or
performance risks that were not changed by those closure commits.

## What the Range Fixed vs Regressed (vs FINDINGS.md)

FIXED: N5-ish dpi (new plot uses config); RM1, RM2, RM6, RM7, RM8, RM9;
provenance gap (new _SubmissionProvenance with sha256); calibration-unavailable
fold handling; shared candidate gates and GTIN status; shared canonical-record
loading; shared threshold fitting; duplicate graph diagnostics; strict fold and
sensitivity metric contracts; A1 (`_grid_folds` tuple indexes).

STILL OPEN from FINDINGS (not addressed, verified):
A3/H1/N3 (gate-vs-ann_finetuned mislabel); A4 (datapoint_usage RuntimeError);
F1 (dev double-encode); F5 (post-train tail re-encode); F7 (uncached
load_config — now additionally called per-config in train_one_config);
N6 (refresh encodes masked tail); N7 (uniformity O(n²)); E RNG offsets
(+new +17); predict_items literals; rerank fold-0 only; NER island.

## Severity Summary

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| HIGH     | 0     | — |
| MED      | 4     | S4/F7, T5, R1, U2 |
| LOW      | 9     | N1, N2, N5, P4, R3, R4/RM5-residual, R5, U5, U6 |

## Recommended next actions (owner stance: fail loudly)

1. Cache `load_config()` (lru_cache) and hoist `calibration_fraction` to a
   module-level config access to eliminate S4/F7 hot-path re-reads.
