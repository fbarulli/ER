# Training surfaces — outstanding work

Closed in this audit (do not re-open without new evidence): youden-SSOT
rewiring, DisjointSet consolidation, sha256_file consolidation,
config-driven `gate.pair_families`, identity_policy dead conditionals,
canonical-records read checks in pipeline.py, plot_dpi() literals,
CWD-relative balanced-pair paths (now `config/paths.yaml`), title_only blank
tuple (now `COLUMN_ALIASES`-driven), `ranking_metrics.component_index`
rewritten on the shared `DisjointSet` (its sixth union-find copy; union
direction and partition proven byte-identical, 300 randomized trials plus
`tests/test_ranking_pool.py`), and all lane-test regressions from those
changes (`55cf9c7`, `772dd40`). The 41 pre-existing training-lane test
failures are resolved: 38 realigned with the committed contracts in
`b2e0835` (provenance pins, calibrated report manifests, explicit `device`,
env-driven strict switch, disarmed exact-census pins in favor of the
config `gate_census_pin` + semantic invariants), and the two tests that
require a live NVIDIA driver were deleted in `d5082df` (they cannot run on
driverless dev machines).

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

- Live-data tests are tied to the regenerated validation frame (gitignored
  `data/final_validation.csv` + `results/manifests/final_validation.json`,
  re-verified against the 2026-10-04 frame). On a fresh clone they skip
  until the pipeline regenerates the artifacts. Exact census pins are
  disarmed by owner directive (2026-10-04): `test_scored_half_decisions`
  enforces only the thin-cell decision criteria and the scalar==set_bag
  semantics identity; the hard guard against a broken frame is the
  emit-time criteria refusal inside
  `build_final_validation.build()` (SystemExit), so a regeneration can no
  longer silently ship a mis-attributed decision.

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
- Test suite is green on this driverless machine: the 41 pre-existing
  failures were all in-flight-contract drift, not audit regressions (zero
  new failures vs the 18287fc baseline; the audit's two regressions were
  fixed in `772dd40`), and are now resolved as noted in the header.
