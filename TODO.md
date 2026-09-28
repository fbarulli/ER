# TODO (2026-09-28, branch training-sid-hybrid)

## Checkpoint eval contract (twin training curve)
- After each epoch/checkpoint: `build_field_slice.py --model <ckpt>` +
  `minimal_flip_slice.py --model <ckpt>` on the live bundle.
- Evaluation target: overall P@R95 0.355 -> ~0.500 by epoch 3;
  verify improvement on trained checkpoints.
- Invariance floor: per-bucket twin P@R95 must hold >= 0.500; overall up +
  twin down = over-smoothing -> raise counterfactual_frac or guarantee
  twin triples per batch (seeded stratified sampler, component-safe).
- Twin margin mean must lift off ~0.008; if ~= 0 post-training, reopen
  field markers as an ablation. Twin P@R95 0.500 is a floor; expect
  improvement above the zero-shot baseline.

## Next
- [ ] Full-data twin P@R95 curve per checkpoint; revisit markers only if
      twin margin ~= 0. No embedding mixup (dynamic; label undefined).

## Defects found in today's changes (source fixes applied; not test-run)
- [x] Update the pinned launcher's row-count preflight for the 5k split.
      `run_ann_full_data.py` points at the new 5k files but still declares
      `EXPECTED_TRAIN_ROWS = 58_529` and `EXPECTED_INFERENCE_ROWS = 3_000`.
      The new split is 56,529/5,000, so `check_settings` refuses to launch.
- [x] Extend payload and pair lineage to symmetric counterpart copies.
      `src/training/training.py` `_build_payload_metadata` records only
      `copy_payload_idx -> anchor_payload_idx`; it treats the new
      `copy_pair_payload_idx` as an original canonical instead of a copy.
      `_build_pair_lineage` likewise keys augmented positives by the original
      counterpart, so `(copy, copy_pair)` falls back to unaugmented lineage.
      Preserve both endpoints' source indices and augmented status in traces.
- [x] Include every tied score when reporting threshold-based P@R95.
      `scripts/minimal_flip_slice.py` `precision_at_recall` counts only the
      sorted prefix ending at the target positive. Other rows with that same
      score are excluded even though the reported threshold accepts them.
      Precision/recall therefore depend on input order for ties; this also
      affects `build_field_slice.py`, which imports the helper. Count all
      scores meeting the chosen threshold and cover ties in regression tests.
- [x] Keep cross-brand scores and labels aligned for unequal sample sizes.
      `scripts/minimal_flip_slice.py` slices positives to the cross-brand
      count but creates that many positive labels even when fewer positive
      scores exist (`--pos-sample` can be smaller than `--gate-sample`).
      Negative scores can consequently receive positive labels. Derive label
      counts from the actual selected score arrays or cap both populations.
- [x] Reject malformed canonical attribute sets in the precision guard.
      `scripts/check_proceed_precision.py` `_as_set` converts parse errors
      and unexpected value types into empty sets; `pair_agrees` treats
      absent evidence as compatible. Corrupted records can therefore pass
      admission. Distinguish genuinely absent evidence from invalid input.
- [x] Isolate the result-sync listing-failure test from the remote session.
      `tests/test_incremental_result_sync.py`
      `test_a_listing_failure_never_propagates` patches only `_list_remote`,
      but the syncer now reads its heartbeat first through the real Colab
      subprocess. A failed heartbeat returns early, so the test may pass
      without exercising a listing failure. Mock the heartbeat/read path
      and assert that the failing listing was actually reached.
- [x] Fail the proceed-precision admission guard on missing canonical records.
      `scripts/check_proceed_precision.py:79-90` excludes missing endpoints
      from `checked`, and the final gate checks only agreement among the
      remaining rows. It can report 100% agreement and PASS while some
      candidate pairs were never verified. Require complete endpoint coverage.
- [x] Align the holdout sampler's default size with its output paths.
      `src/training/sample_deduped_dataset.py:148` still defaults to 3,000
      while output/remainder/manifest defaults now name the 5k split.
      Running without `--size` overwrites those paths with the wrong split.
- [x] Respect selected training negatives when constructing MNRL twins.
      `src/training/training.py` `_build_mnrl_training_triples` appends twins
      from the full audit without checking that `(copy, pair)` survives in
      `train_neg`. This can reintroduce negatives discarded by balancing
      or caller filtering. Require membership in the selected negative pool.
- [x] Count retained augmented negatives in the diet gate.
      `scripts/diet_manifest.py:73` uses every negative audit row as the
      numerator but selected `train_neg` rows as the denominator. After
      trimming augmentations, this overstates exposure and can falsely pass
      the gate. Match audit pairs to the retained training pairs first.
- [x] Validate entity-key coverage against payload indices, not pair count.
      `src/training/masking.py` `_resolve_entity_keys` compares key count
      with the number of sampled pairs. A valid pool with more pairs than
      payload rows is rejected; a smaller pool can still reference an
      uncovered high payload index. Check the selected pairs' endpoints.
- [x] Review twin handling for anchors with multiple positive counterparts.
      `_build_mnrl_training_triples` checks twins against the first positive
      stored for an anchor, rather than membership in all selected positive
      pairs. Twins for other valid counterparts can be silently skipped.

## Additional review findings (2026-09-28; source fixes applied)
- [x] Require a semantic conflict before labeling a counterfactual twin 0.
      `src/training/masking.py` `augment_counterfactual_twins` tests donor
      token-list inequality, not field compatibility. Reordered equal sets,
      overlapping flavors, and volumes within tolerance can become negatives.
      Earlier inspection confirmed 12 volume twins in worker_1 compatible
      under the existing 5%/5ml rule (e.g. 480/500ml and 325/330ml).
      Apply the field's conflict semantics, including order-independent sets.
- [x] Revisit positive supervision for value-swapped hard-negative anchors.
      `_build_mnrl_training_triples` maps every negative copy back to its
      source's unchanged positive. This is safe for masking, but a volume or
      flavor transplant can make that positive incompatible with the copy.
      Earlier inspection found 2,146 live triples with disjoint values in
      the transplanted field, including a 1500ml copy paired positively with
      1000ml. Use a compatible rewritten positive or exclude these triples;
      the current supervision can oppose the counterfactual objective.
- [x] Make diet projections depend on the actual loss and realized quota.
      `scripts/diet_manifest.py` and `prepared_bundle.py` always credit the
      configured easy-negative quota, but `training.py` calls
      `_mix_random_easy_training_negatives` only for `loss == "contrastive"`.
      The active config uses MNRL, so its claimed extra negatives never join.
      Even contrastive can return no easy candidates. Do not report a
      projected ratio as an enforced train-time contract in these cases.
- [x] Include easy negatives in the augmentation-exposure denominator.
      Independently of the retained-audit numerator issue above,
      `diet_manifest.py` divides augmented views by static `train_neg`
      while its ratio gate credits additional easy views. For contrastive
      at a 1:1 easy quota, static 40% augmentation becomes 20% of negative
      presentations, below the configured 30% minimum despite a PASS.
- [x] Use the production scoring representation in checkpoint slice reports,
      or explicitly identify them as encoder-only diagnostics.
      `build_field_slice.py` and `minimal_flip_slice.py` score raw normalized
      encoder vectors. Training evaluation fuses bundle numeric features
      through `fuse_numpy` with the configured weight (currently 0.35).
      In particular, volume/pack metrics and thresholds do not measure the
      same scorer whose checkpoint quality they are intended to track.
- [x] Enforce the new ambiguous-volume flag downstream.
      `pipeline.py` flags out-of-range winners but keeps their values;
      `critical_attribute_evaluation` and structured feature construction
      do not consult `attribute_consistency_flags`. A flagged 25000ml value
      still becomes resolved model evidence and can drive a volume conflict.
      The new comment's promised ambiguity behavior is therefore absent.
- [x] Make the cluster ratio breaker safe for small sample runs.
      `check_cluster_sizes` divides the largest component by clustered
      records only and runs unconditionally during preparation. A healthy
      sample with one two-record component gets ratio 1.0 and fails the
      0.05 cap, regardless of how many isolated rows exist. Use a population
      denominator/minimum-size policy that does not reject tiny valid pools.

## Screening coverage (2026-09-28)
- Completed inspection of the final code state from pre-day commit
  `8394602` through `9889c32`, including the current uncommitted changes
  and new `repair_augmented_features.py` / `test_augmented_features.py`.
- Covered all changed production code, scripts, configuration, tests and
  golden-fixture changes; inspected bundle manifests and split integrity.
- The seven entries above extend the twelve existing defect entries.
  All nineteen source findings were addressed by six Luna agents and
  combined source review. Tests were not run for these fixes.
  Final continuation was inspection-only, as requested; no further tests
  were started after that instruction. Screening does not establish that
  the code is free of other defects.

## Regeneration after review fixes
- [ ] Rebuild both full prepared bundles from corrected code. Existing bundles
      contain semantically compatible twins and are rejected by the loader.
- [ ] Record the corrected diet gate result without relaxing its thresholds.

## Open gaps (2026-09-28 audit — all items)
Parser / extraction:
- [ ] Volume unit anomalies (`0.33 ml` title vs `330 ml` attribute, attribute
      wins with an inconsistency flag) — downstream blast radius unmeasured.
- [ ] No MPN parsing; entity guard best-effort on unmapped rows (52.1%
      cluster coverage; barcode-less duplicates need record linkage).
Data / augmentation:
- [ ] Positive/type coverage: current balanced sample has 1,414 positives
      at sim>=0.50 (530 at >=0.80) and 1,414 pack-blocker negatives only.
      Current 5k-holdout training bundles have 261 hard-positive pairs;
      expand reviewed positive coverage and negative-family diversity.
- [ ] Low-cardinality fields concentrate transplants structurally; monitor
      field/value distributions under the existing concentration caps.
- [ ] `swap_agreed` lane yields 0 structurally — remove or keep as invariant.
- [ ] Swaps alter structured tokens while prose keeps the old word;
      measure this contradiction and review field-specific rewriting.
      Numeric volume/pack vectors now follow the swapped tokens.
- [ ] Counterfactual validity assumed: decorative flavor words -> label noise;
      no per-field flip-validity measurement.
- [ ] Swap/twin fracs and caps hand-picked (0.20/0.10, 0.35/0.03); HPO never
      swept them.
Training / eval:
- [ ] No checkpoint trained with twins — hypothesis unvalidated (zero-shot
      P@R95 0.50, margin 0.009 is the baseline to beat).
- [ ] No train-time twin monitoring (MNRL loss has no per-subset hooks;
      offline slice curve per checkpoint not yet drawn).
- [ ] No twin loss warmup (watch-item: spikes epochs 1-2; guards in place).
- [ ] Pooling vs single-token flips unvalidated (markers deferred, not dead).
- [ ] Contrastive/triplet paths consume new audits generically — untested.
Process / repo:
- [ ] Metrics unversioned (`results/` gitignored — reports live locally only).
- [ ] Smoke unrepresentative (`--sample 1000` = first rows, not stratified;
      e.g. entity coverage 11.8% vs 52.1%).
- [ ] Easy-negative replenishment samples with replacement when pool is thin
      (silent re-weighting at small scale).
- [ ] Dynamic masking invisible to diet (ephemeral views excluded by design).
- [ ] 41MB new CSVs in git while DVC sits disabled (bloat policy question).
