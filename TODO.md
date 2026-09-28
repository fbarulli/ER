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
- [x] Diet gate + slice baseline on smoke (caps build): PASS.
- [x] Full test-suite regression sweep: 488 passed, 7 pre-existing fails
      (verified identical on clean tree).
- [x] Shared cross-lane donor-value counter (footprint cap binds bundle).
- [x] Entity clusters wired (52.1% rows covered full-scale, 13,783 clusters).
- [x] Full bundle rebuilt (worker_1+2): 51,153 pos / 30,341 neg, 2,245
      twins, 4,413 + 2,864 value swaps. Zero-shot slice baselined.
- [ ] Diet ratio FAIL at full scale: 1.686 > 1.500 (old bundle was 2.23 —
      inherited, improved). Owner call: recalibrate cap or raise neg
      static fracs. Train-time ratio ~= 0.85 after 1:1 easy negatives.

## Next
- [ ] Rewire 5k holdout into final_inference/colab (review first).
- [ ] Full-data twin P@R95 curve per checkpoint; revisit markers only if
      twin margin ~= 0. No embedding mixup (dynamic; label undefined).
