# DATA_PATH.md — one data path, one artifact contract (2026-09-28)

Reference: the last training run `20260913T123559565190Z` (worker_1,
masking_only, multilingual-L12) — the submission-branch lineage. Its
`publication_manifest.json` (41 DVC pointers, recovered via
`dvc_refs/20260913T123559565190Z/worker_1/`) is the authoritative list of
what a COMPLETE run publishes. Partial recovery lives under
`training_results/20260913T123559565190Z/worker_1/`.

## Stage map (source -> submission, one path)

| stage | producer | inputs | outputs (the contract) |
|---|---|---|---|
| S0 source | dedupe (src/training/dedupe.py) | dataset.csv (71,623 rows, sha-pinned in config audit.source_export_expected_sha256) | data/dataset_deduped.csv (61,529) |
| S1 splits | sample_deduped_dataset | dataset_deduped.csv | data/dataset_deduped_train_minus_5000.csv (56,529 train), data/dataset_deduped_sample_5000.csv (5,000 holdout inference) |
| S2 gate/pairs | pipeline + labeled_pairs | deduped CSV | data/canonical_records.csv, data/gate_results.csv, data/labeled_pairs.csv, results/training/balanced_pairs_*.csv |
| S3 bundle build (LOCAL, pre-training) | `training.train --prepare-bundle` | S1 CSV + canonical + gate + labeled pairs | data/prepared/full/worker_{1,2}_baseline.pkl.gz + .json manifests (pos/neg/train_neg + sources, mask audits, structured features) |
| S4 training (Colab VM) | `training.train_prepared` -> training.train_one_config | S3 bundle + config | results/train_<model>_holdout_<variant>_fold_metrics.csv; results/logs/<run_tag>/{see contract below}; _checkpoints/<model>/checkpoint-N (safetensors + trainer_state.json) |
| S5 report | generate_training_report | S4 outputs | report_.../{report.json, metrics_aggregate.json, metrics_summary.csv, confusion_matrices.csv, ranking_hits_at_k.csv, attribute_error_breakdown.csv, random_easy_metrics.csv, score_distribution_overlap.csv, PNGs} |
| S6 checkpoint eval (NEW contract, per checkpoint) | scripts/build_field_slice.py + minimal_flip_slice.py + flip_validity_audit.py | S3 bundle + S4 checkpoint | results/eval_slice_by_field.json (overall P@R95), results/minimal_flip_slice_<name>.json (twin margins per field, donor uniformity), results/flip_validity_audit.json (prose contradiction + transplant concentration per bundle) |
| S7 publication | DVC publisher + HF artifacts + W&B mirror | S4-S6 outputs | dvc_refs/<run>/<worker>/ pointers; HF fbarulli/e-r-training-artifacts; W&B project e-r (active since 2026-09-28: WANDB_API_KEY in .env); submission/SKU_ITEM_submission.csv + diagnostics + provenance |

Gates on the path (a stage may not hand off without them):
- S3->S4: `scripts/diet_manifest.py BUNDLE` exit 0 (neg_aug_frac >= 0.30,
  pos/neg ratio <= 1.50; arithmetic becomes loss-aware per TODO fix F2)
  + `load_prepared_bundle` re-validation (feature lineage, twin
  conflicts) + flip_validity_audit re-run.
- S4->S5: fold status "ok", coverage identity asserts (usage rows cover
  every registered population; MNRL lane restored per TODO fix F5).
- S4 checkpoint -> S6: twin P@R95 >= 0.500 floor, margin mean lifting
  off 0.0074 (experiments.md EXP-03 decision rule).

## Per-fold artifact contract (results/logs/<run_tag>/)

From the reference manifest — every future run must publish:

1. `datapoint_usage_fold{i}.csv` — per-pair population/augmentation/
   presentations/lineage. MISSING under MNRL today (contrastive-only
   writer); restored by TODO fix F5.
2. `datapoint_type_coverage_fold{i}.csv` — per-population ladder
   (ok/missing/not_reached/unavailable/unregistered) + identity asserts.
   Same restoration as (1).
3. `mnrl_subset_loss_by_epoch_fold{i}.csv` — NEW (EXP-03): per-population
   subset losses per epoch (twin/swap/masked/gate/cross-brand/easy);
   lands via the background agent's `_tracking_mnrl_loss` in
   train_one_config (shared with train_prepared, so production gets it).
4. `loss_backprop_fold{i}.csv` — logging-step loss telemetry (contrastive;
   MNRL subset totals mirror it via pop_tracking_stats).
5. `pair_backprop_fold{i}.csv` — contrastive per-pair selection/backprop
   counters (contrastive-only by design; contract notes it absent for
   MNRL, superseded by (3) for MNRL).
6. `mask_visibility.csv`, `mask_hard_negative_visibility.csv`,
   `masking_per_epoch_fold{i}.csv` — per-copy realized extents, static +
   per-epoch dynamic (dynamic rows exist only when contrastive).
7. `train_rows_fold{i}.csv`, `payload_pairs.csv`,
   `negative_resolution_manifest.csv` — which rows/pairs trained, payload
   fingerprint, negative-resolution census.
8. `live_status.json`, `training.log` — VM heartbeat + run log.

## Bundle + gate artifacts (S3, checked before any training)

- worker manifest .json (schema_version, masking_config echo, sha256,
  static_view_ratio) — the REBUILT bundles must show frac 0.80 and ratio
  <= 1.50.
- diet verdict (diet_manifest.py stdout -> archived next to the bundle).
- flip_validity_audit.json re-run on the fresh bundle (per-field prose
  contradiction + concentration tables; flavor-policy arm recorded in
  experiments.md EXP-01).

## Runbook — data prep + training, step by step (execution order)

Preconditions: tracking fixes F1-F5 + smoke-sampler fix landed; agents 1
(MNRL telemetry) + 3 (alias expansion) landed; full suite green.

1. Sanity: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` — all
   green (baseline 523 + new tests).
2. Rebuild bundles (ONE rebuild, inherits aliases + all fixes):
   `PYTHONPATH=src .venv/bin/python -m training.train --model minilm_l6
   --dataset data/dataset_deduped_train_minus_5000.csv --payload full
   --masking-profile baseline --collapse-guardrail-profile threshold_80
   --prepare-bundle data/prepared/full/worker_1_baseline.pkl.gz
   --no-mask-effect --no-plot` (repeat for worker_2 with --seed
   variance per the existing worker scheme). ~531 s/worker measured.
3. Gate the bundles: `PYTHONPATH=src .venv/bin/python
   scripts/diet_manifest.py data/prepared/full/worker_1_baseline.pkl.gz`
   — expect PASS with the corrected (projection-free) arithmetic; then
   `scripts/flip_validity_audit.py --bundle ... --out
   results/flip_validity_audit_fresh.json` — record per-field
   contradiction + concentration tables on the fresh mint.
4. Verify the contract columns: fresh manifest shows frac 0.80,
   static_view_ratio <= 1.50, no swap_agreed_frac; `load_prepared_bundle`
   drift warning gone.
5. Launch training (Colab lane): `er-colab` train run as configured
   (train_prepared via colab.py:1283 swap) — W&B mirror now active.
6. Per-checkpoint eval loop (the EXP-03 contract): after each
   checkpoint, `scripts/build_field_slice.py --model <ckpt>` +
   `scripts/minimal_flip_slice.py --model <ckpt> --bundle
   data/prepared/full/worker_1_baseline.pkl.gz --out
   results/minimal_flip_slice_ckpt<N>.json` — append to the twin
   P@R95/margin curve; watch the pre-registered failure signatures
   (sweetener margin ~0, P@R95 < 0.500, epoch-1-2 spikes, gate slice
   regression).
7. Consolidation: collect contract artifacts (DATA_PATH.md list), run
   the coverage-identity check, generate the training report, then
   final inference on the 5,000 holdout (device: cuda enforced).



- Tracking fixes section: F1 '+aug' registration, F2 loss-aware diet
  projection, F3 swap-copy accounting, F4 masked-positive conditioning,
  F5 MNRL coverage restoration. Until F5 lands, items (1)-(2) are absent
  from any MNRL run — do not launch training before they are restored.
- EXP-03 adds (3). Everything else in the contract is produced by
  existing code paths.
