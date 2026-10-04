# Training surfaces — outstanding work

Closed in this audit (do not re-open without new evidence): youden-SSOT
rewiring, DisjointSet consolidation, sha256_file consolidation,
config-driven `gate.pair_families`, identity_policy dead conditionals,
canonical-records read checks in pipeline.py, plot_dpi() literals,
CWD-relative balanced-pair paths (now `config/paths.yaml`), title_only blank
tuple (now `COLUMN_ALIASES`-driven), and all lane-test regressions from those
changes (`55cf9c7`, `772dd40`).

## Stage 1: dedupe tiers -> dataset_deduped.csv / sku_to_rep.csv

- Per-load dedupe-lineage guard deliberately NOT kept in
  `load_dataset_deduped` (reverted in `772dd40`; a default-path coupling of
  two artifacts plus the census pin broke fixture-lane tests). Lineage is
  enforced at the prepare boundary (stage-manifest provenance) and the colab
  worker census; a consumer reading `dataset_deduped` out-of-band between
  dedupe and the worker is unguarded by design. If a per-load guard is
  wanted, it must be opt-in, not default-path.

## Stage 2: canonical records + data quality audit

- No open items.

## Stage 3: gates + pair building (blocking, hard_negatives, labeled_pairs, build_training_data)

- `LEGACY_REASON_PREFIXES` in `sample_balanced_pairs.py` keeps a family for
  the retired "Flavor mismatch:" wording. Remove it (and the collision
  check) only when that wording can no longer appear;
  `test_every_hard_no_gate_reason_maps_to_a_known_family` still enforces the
  legacy classification until then.
- `labeled_pairs.py` runs `main()` at module level (line 190) — documented as
  intentional because two importers rely on it; converting to a guarded
  entry point is an open refactor if the import-time flow is ever revisited.

## Stage 4: truth/folds/validation (generate_rand_truth, build_final_validation, folds, robust_validation)

- Pinned live-data tests are tied to the regenerated validation frame
  (gitignored `data/final_validation.csv` +
  `results/manifests/final_validation.json`, re-pinned to the 2026-10-04
  frame). On a fresh clone they skip until the pipeline regenerates the
  artifacts, and every regeneration re-pins
  `test_scored_half_decisions` per its attribution convention.

## Stage 5: augmentation + prepared bundle

- No open items. (The deferred `surfaces_20261004` preparation regeneration
  is tracked in `COLAB_SURFACES.md`.)

## Stage 6: training entry (train.py, training.py, hpo) + outputs

- HPO lane stays out of scope (owner directive "ignore hpo"). The one seam:
  `src/training/hpo_metrics.py:34` imports `_youden_threshold` from
  `rand_matching.py` (re-export alias at `rand_matching.py:2116`). When hpo
  is in scope: import `youden_threshold` from `core.ranking_metrics`
  directly and delete the alias.
- No GPU run has exercised the rewired paths. Live-CPU validation confirmed
  byte/row identity for folds, union-find components, youden thresholds,
  sha256 manifests, and the regenerated `final_validation` artifact; the
  next authorized training run should confirm the GPU lifecycle
  (`training.py`, `evaluate_models.py`, `rerank.py`).
- 41 pre-existing test failures remain in the training lane; none are from
  this audit's changes (verified: zero new failures vs the 18287fc baseline,
  and the audit's two regressions are fixed). They belong to in-flight
  work: 14 train.py provenance-block pins in `test_mining_hypotheses`,
  the `_EXPECTED_SOURCE_EXPORT_ROWS` import in `test_validation_inference`
  (census-to-config move), GPU-driver environment failures, and stale
  gate/dashboard pins.
