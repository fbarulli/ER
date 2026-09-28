# TODO (2026-09-28, branch training-sid-hybrid)

## Done
- [x] Regex review: fixed low/reduced-sugar inversion + bare-soda overfire,
      refreshed 2 golden rows (853 byte-identical). Suite green.
- [x] 5k holdout: dataset_deduped_sample_5000 + train_minus_5000 + manifest
      (seed 42, superset of the 3k). 3k lane still live.
- [x] Value-swap augmentation (coconut->lime): symmetric positives,
      anchor-side negatives. Static, pre-training, corpus donors only.
- [x] Counterfactual twins: agreed-field flips labeled 0, registered
      `counterfactual` source, MNRL triple join fix (twins as explicit
      negatives or they never train).
- [x] Guards: entity-disjoint donors (GTIN-normalized), identical-text skip.
- [x] Caps: swap_max_field_share 0.35 (soft), swap_max_value_share 0.03 (hard).
- [x] Base-priority class balance: trim aug copies before base pairs, report
      discards by kind (smoke: discarded_base=0, discarded_aug=476).
- [x] Stress-slice script: twin P@R95 + margins, cross-brand/gate slices,
      donor uniformity, subset distance split.

## In progress
- [ ] Diet gate + slice baseline on smoke probe4 (caps build).
- [ ] Full test-suite regression sweep.

## Next
- [ ] Rebuild full prepared bundles (regex fix + new lanes stale them).
- [ ] Rewire 5k holdout into final_inference/colab (review first).
- [ ] Full-data twin P@R95 curve per checkpoint; revisit markers only if
      twin margin ~= 0. No embedding mixup (dynamic; label undefined).
