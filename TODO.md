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
- [x] Guarded bare-soda rule v2 (still-evidence + dry-type guards):
      3,635 fire / 451 suppressed of 4,084; 12 golden rows refreshed
      (all true sodas; syrups + still-declared rows excluded).
- [x] Donor overlap guard (swap_max_donor_overlap 0.95; min-overlap floor
      rejected: costs 43% yield, transfers nothing domain-specific).
- [x] Barcode degeneracy check: no placeholder dominance (top values are
      legit 11-digit GTINS x11-13 rows); 53,601 rows share barcodes
      (normal multi-retailer); 34,636 empty.
- [x] T2 end-to-end: proceed 0.80->0.65, precision gate 1.0000, labeled
      rebuilt 8,641 (919/7,722), oracle pins reconciled, balanced pool
      rebuilt 1,838 (pack_blocker-only — negatives bind).
- [x] T3: aerosol/tray -> rejected_by_ontology with reason codes; miss
      queue re-run: 44 -> 19 candidates + 25 rejected.
- [x] T4: volume anomaly flag (attribute wins, text-neutral); 197 rows.
- [x] T5: manifest ratio contracts (ratio_to_hard, static/effective views,
      note); diet gates on projected train-time ratio.
- [x] T1: field-sliced harness 33/33/34 (234 twins/bucket, package binds;
      2,106 rows with gate negatives; zero-shot P@R95 0.35 / 0.50 per-bucket).
- [x] Failure audit: all 7 pre-existing fails are test-side drift (5x mock
      of removed colab._remote_file_size, 2x stale MNRL call signature) —
      zero pipeline/null/shape breakage.
- [x] Twin-triple regression test (twins as explicit MNRL negatives).
- [x] Sampling audit: smoke --sample takes first-1000 rows (63/280
      retailers, water-heavy vs juice-heavy full) — smoke yields are
      directional only, never selection-grade.

## Checkpoint eval contract (twin training curve)
- After each epoch/checkpoint: `build_field_slice.py --model <ckpt>` +
  `minimal_flip_slice.py --model <ckpt>` on the live bundle.
- Convergence: overall P@R95 0.355 -> ~0.500 by epoch 3 (organic
  disambiguation learned).
- Invariance floor: per-bucket twin P@R95 must hold >= 0.500; overall up +
  twin down = over-smoothing -> raise counterfactual_frac or guarantee
  twin triples per batch (seeded stratified sampler, component-safe).
- Twin margin mean must lift off ~0.008; if ~= 0 post-training, reopen
  field markers as an ablation.
- [x] T2 end-to-end: proceed 0.80->0.65, precision gate 1.0000, labeled
      rebuilt 8,641 (919/7,722), oracle pins reconciled, balanced pool
      rebuilt 1,838 (pack_blocker-only — negatives bind).
- [x] Donor overlap guard (swap_max_donor_overlap 0.95; min-overlap floor
      rejected: costs 43% yield, transfers nothing domain-specific).
- [x] Bundle pins masking + easy config; drift warns on load.
- [x] Balanced pair sample at max feasible (1,060, single pack_blocker
      family — positives bind; per-type diversity needs twin-sourced
      eval sets next).

## In progress
- [x] Diet gate + slice baseline on smoke (caps build): PASS.
- [x] Full test-suite regression sweep: 488 passed, 7 pre-existing fails
      (verified identical on clean tree).
- [x] Shared cross-lane donor-value counter (footprint cap binds bundle).
- [x] Entity clusters wired (52.1% rows covered full-scale, 13,783 clusters).
- [x] Full bundle rebuilt (worker_1+2): 51,153 pos / 30,341 neg, 2,245
      twins, 4,413 + 2,864 value swaps. Zero-shot slice baselined.
- [x] Diet ratio at full scale: gate now projects the train-time easy
      quota (ceil(hard * ratio), same SSOT as the trainer) — DIET PASS
      (train-time 0.843; bundle-only 1.686 printed, informational).

## Next
- [x] 5k rewire event (tagged rewire-5k): pointers switched, bundle rebuilt
      on train_minus_5000 (56,529 rows; encoder never sees inference rows),
      breaker passed live (max 12), diet PASS (0.810), harness rebuilt
      (203 twins/bucket, P@R95 0.355) + twin baselines reset.
- [x] Recall-loss bands: agreement 1.0000 in every 0.05 band to 0.50;
      floor set 0.50 (+884 pairs). Boundary-20 inspected (genuine matches).
- [ ] Full-data twin P@R95 curve per checkpoint; revisit markers only if
      twin margin ~= 0. No embedding mixup (dynamic; label undefined).

## Open gaps (2026-09-28 audit — all items)
Parser / extraction:
- [x] 55 unresolved regex candidates -> 19 + 25 ontology-rejected (rerun).
- [x] Bare "soda" resolved with guards (see Done).
- [ ] Volume unit anomalies (`0.33 ml` title vs `330 ml` attribute, attribute
      wins silently) — blast radius unmeasured.
- [ ] No MPN parsing; entity guard best-effort on unmapped rows (52.1%
      cluster coverage; barcode-less duplicates need record linkage).
- [ ] Sweetener veto excluded from gates (6 false merges vs 74 true lost).
- [ ] Low-cardinality fields (carbonation 3 values, pulp 2, sweetening 1)
      concentrate transplants structurally — bounded, monitored, not fixable.
Data / augmentation:
- [ ] Positive scarcity binds everything: 530 gate positives @sim>=0.8,
      balanced-pair ceiling 1,060; cross-country hard positives only 299.
- [ ] `swap_agreed` lane yields 0 structurally — remove or keep as invariant.
- [ ] Swaps are tags-only: prose keeps old word, copies inherit anchor
      structured vector (text/vector disagreement unmeasured).
- [ ] Counterfactual validity assumed: decorative flavor words -> label noise;
      no per-field flip-validity measurement.
- [ ] Swap/twin fracs and caps hand-picked (0.20/0.10, 0.35/0.03); HPO never
      swept them.
- [ ] 5k holdout built but not wired (final_inference + colab still on 3k).
- [ ] `balanced_pairs_sample_3000` never built; threshold sweeps lack artifact.
- [ ] Bundles don't record ratio_to_hard: gate verdicts shift if it changes
      without rebuild.
Training / eval:
- [ ] No checkpoint trained with twins — hypothesis unvalidated (zero-shot
      P@R95 0.50, margin 0.009 is the baseline to beat).
- [ ] No train-time twin monitoring (MNRL loss has no per-subset hooks;
      offline slice curve per checkpoint not yet drawn).
- [ ] No twin loss warmup (watch-item: spikes epochs 1-2; guards in place).
- [ ] Pooling vs single-token flips unvalidated (markers deferred, not dead).
- [ ] Contrastive/triplet paths consume new audits generically — untested.
Process / repo:
- [x] Result-sync and MNRL regression tests updated for current interfaces;
      the related Colab setup expectation now uses CLI defaults (47 targeted
      tests pass). Full suite: 515 passed, 2 skipped with pinned
      `hnswlib==0.8.0` installed in the local `.venv`.
- [ ] Metrics unversioned (`results/` gitignored — reports live locally only).
- [ ] Smoke unrepresentative (`--sample 1000` = first rows, not stratified;
      e.g. entity coverage 11.8% vs 52.1%).
- [ ] Easy-negative replenishment samples with replacement when pool is thin
      (silent re-weighting at small scale).
- [ ] Dynamic masking invisible to diet (ephemeral views excluded by design).
- [ ] 41MB new CSVs in git while DVC sits disabled (bloat policy question).
