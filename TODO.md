# TODO (updated 2026-09-29, branch training-sid-hybrid)

## PRIORITY ORDER (owner ruling 2026-09-29)
- **P1 — FINALIZE before training (data alignment etc.)**: bundle rebuild
  (Questions #1), MNRL telemetry port (reference-contract CSVs), swap-copy
  diet accounting (owner decision), masked-positive minting survival,
  coverage-analysis remainder, 4a39fdc verification remainder, flavor-twin
  policy decision, smoke-sampler code committed.
- **P2 — remaining open gaps** (parser/extraction, augmentation, training/
  eval, process) in "Open gaps".
- **DEAD LAST — graph/linkage additions** (record linkage, GNN/RQ-VAE
  semantic IDs — SKIPPED by owner verdict, graph-construction tiers, queued
  measurement). Do not start until P1/P2 closed.

## P1 — FINALIZE before training (data alignment etc.)
- [ ] **Bundle rebuild (blocks training)**: diet gate passes on paper
      (frac=0.80) but worker_1/worker_2 bundles were built at frac=1.00
      (sidecars confirm frac=1.0). Rebuild worker_1/worker_2 bundles from
      the pushed code; who triggers: `_build_local_training_bundles`
      (src/cli/colab.py) runs `training.train` on the Colab VM. Test on a
      small sample first, then full.
- [ ] Port per-population coverage to the MNRL fold path so the next run
      publishes the reference contract's tracking CSVs (usage + type
      coverage per fold; populations from triples twin/masked/base).
      Reference: publication_manifest (DVC 20260913T123559565190Z) lists
      datapoint_usage_fold0.csv, datapoint_type_coverage_fold0.csv,
      pair_backprop_fold0.csv, loss_backprop_fold0.csv,
      masking_per_epoch_fold0.csv, mask_hard_negative_visibility.csv,
      train_rows_fold0.csv — the MNRL lane produces NONE of these
      (coverage writer is contrastive-only): telemetry regression to close
      before training.
- [ ] Swap-copy diet accounting for MNRL: RESOLVED IN CODE (ead6966
      excludes swap negatives from neg_aug_views when loss=mnrl) — owner
      sign-off pending to close the item.
- [ ] Masked-positive minting: 76% (16,345/21,373) never train — no source
      negative in the fold. Diet now REPORTS real survival (ead6966).
      Remaining: condition minting on fold-negative availability (code) —
      or accept + document. Owner decision.
- [ ] Coverage analysis remainder (agent 2 handoff, NOT yet written to
      results/coverage_expansion_analysis.json): (a) gate band x decision
      x canonical-agreement table incl. the 323 proceed rows below 0.50
      (+22.8% pos headroom); (b) max_pos_per_group surplus (blocking.py
      identity lane, group inventory on the 56,529 split); (c) hp_pairs
      yield: second04_pairs_positive.csv row count, strict volume equality
      vs the gate's 5% tolerance, unknown-volume rows; (d) confirm bundle
      hp_pairs == 261. KNOWN: 1,414 = ENTIRE gate-verified positive
      population at sim>=0.50; balanced-pool negative family = pack_blocker
      only; cross_brand funnel 30,372 clear the floor vs target 6,000
      (~4.9x headroom); labeled_pairs.csv = 9,136 rows exact;
      PINNED_GATE_FALLBACK_PAIRS pins gate universe (labeled_pairs.py:85-95,
      core/common.py:608) — changing gate universe requires pin updates.
- [ ] Verification of 4a39fdc remainder: claims 1-7 verified (5 PASS,
      1 partial, 1 FAIL); outstanding: claim 8 (flip_validity_audit.py
      self-review — numeric surfaces, global denominator), 49->30
      ambiguous_volume arithmetic, sweep for other sites assuming frac
      1.00/1.622.
- [ ] Flavor-twin policy decision (owner): flavor twins 100%
      prose-contradicted (717/717) — exclude flavor from twin flips
      (config allowlist) vs accept. Bundle rebuild inherits the decision.
- [ ] Checkpoint eval contract execution: after each epoch/checkpoint,
      build_field_slice.py + minimal_flip_slice.py on the live bundle;
      P@R95 0.355 -> ~0.500 by epoch 3; twin-bucket floor >= 0.500;
      twin margin must lift off ~0.008. (Harness ready; blocked on the
      bundle rebuild + first training run.)

## Checkpoint eval contract (twin training curve)
- After each epoch/checkpoint: `build_field_slice.py --model <ckpt>` +
  `minimal_flip_slice.py --model <ckpt>` on the live bundle.
- Evaluation target: overall P@R95 0.355 -> ~0.500 by epoch 3;
  verify improvement on trained checkpoints.
- Invariance floor: per-bucket twin P@R95 must hold >= 0.500; overall up +
  twin down = over-smoothing -> raise counterfactual_frac or guarantee
  twin triples per batch (seeded stratified sampler, component-safe).
- Twin margin mean must lift off ~0.008; if ~= 0 post-training, reopen
  field markers as an ablation.

## Open gaps (P2 — remaining)
Data / augmentation:
- [ ] Positive/type coverage: current balanced sample has 1,414 positives
      at sim>=0.50 (530 at >=0.80) and 1,414 pack-blocker negatives only.
      Current 5k-holdout training bundles have 261 hard-positive pairs;
      expand reviewed positive coverage and negative-family diversity.
      IN PROGRESS: per-lever headroom measured on the training split, report
      to results/coverage_expansion_analysis.json.
      Constraint: pos/neg ratio 1.474 vs 1.50 ceiling — positive
      expansion must be paired with negative expansion.
- [ ] Counterfactual validity: MEASURED 2026-09-28 (flavor twins 100%
      prose-contradicted 717/717; volume 0.55, package_type 0.93 opaque).
      Remaining: owner policy decision (flip allowlist vs prose rewrite vs
      accept); bundle rebuild at the end inherits the decision.
- [ ] Swap/twin fracs and caps hand-picked (0.20/0.10, 0.35/0.03); HPO never
      swept them.
- [ ] Swap-copy accounting for MNRL (diet): 2,709 swap copies diet-counted
      as augmented views but 0% MNRL-train — ead6966 implemented the
      exclusion for MNRL; needs owner sign-off + bundle-side verification.
Training / eval:
- [ ] No checkpoint trained with twins — hypothesis unvalidated (zero-shot
      P@R95 0.50, margin 0.009 is the baseline to beat).
- [ ] Pooling vs single-token flips unvalidated (markers deferred, not dead).
- [ ] Contrastive/triplet paths consume new audits generically — untested.
Process / repo:
- [ ] Metrics unversioned (`results/` gitignored — reports live locally only).
- [ ] 41MB prepared bundles in git while DVC disabled (bloat policy).
- [ ] Duplicate-code hunt (owner directive 2026-09-29): consolidate
      remaining copies of the per-row model composition — DONE for the
      4-copy loop (see Completed). Remaining known dupes: duplicated
      regexes (STOPWORDS/brand_tokens islands), NER dead cluster,
      hpo_persistence PG machinery — LOW priority, triage deferred unless
      they touch the active path.

## DEAD LAST — recent additions (2026-09-29; do NOT start until P1/P2 done)

### Standing rules for all new code (owner)
- SSOT config loading for EVERYTHING, paths included (RESULTS root,
  files./layouts. bindings; honors EUROMONITOR_RESULTS_DIR).
- graphify blast radius before/after every change (rebuild when HEAD moves).
- Every new code path EXECUTED (synthetic -> smoke -> real-data sandboxed).
- Limit test writing: guard tests only where behavior could silently regress.

### 1. Barcode-less record linkage — OPERATIONAL, residuals documented
- src/core/record_linkage.py + scripts/build_barcode_less_linkage.py;
  guard tests tests/test_record_linkage.py (5). Suite 561+2 green.
- Design (final): finalized-title comparison (SSOT build_sku_texts,
  title-side — attributes blanked, mirroring the payload title_only
  variant), brand blocks, cross-retailer-only, exact OR IDF-weighted
  Jaccard >= 0.7 on pack-stripped tokens, AVERAGE-LINKAGE agglomeration
  over DISTINCT titles (chain-drift guard), per-item uniqueness (corpus
  token-IDF) as measured feature, low-coherence cluster flag (margin 0.05).
- Full-run census (2026-09-29): 38,159 eligible; 3,867 multirow clusters;
  11,150 rows in them; 2,498 exact + 8,511 fuzzy links; 406 merges refused
  by the internal-mean guard; 500 low-coherence clusters flagged; ~2m5s
  (single finalized-text build, deterministic).
- Defect trajectory: union-find flavor merge (Obsesso 61-row cluster)
  -> fixed by block-IDF weighting + average-linkage over distinct titles.
  Verified clean on real data: Obsesso (Black/Mocha/Caramel/Latte
  separated), International Delight, Clearly Canadian, Aspire (variety
  packs reduced to same-flavor pairs).
- Residuals (documented, review-queue): Wandering Bear bl-020834-style
  mixes (cross-flavor sims 0.71-0.78 — too close to threshold for any
  global value; short titles, weak block IDF); Aspire multi-flavor variety
  listings need flavor-SET extraction (future work). 500 low-coherence
  clusters flagged for review. Linkage is candidate generation — the
  gate/verifier keeps the final say.
- GTIN-14 indicator-digit folding: MEASURED NEGATIVE — 46 GTIN-14 rows,
  0 fold to checksum-valid GTIN-13s, 0 match existing units; EAN-8 padding
  collisions 0; UPC-12<->13 padding collisions 0. NOT built (no yield).
- Retailer alias normalization: DONE (core.text.normalize_retailer SSOT,
  wired into blocking.py build_pairs/build_true_pairs, common.py
  kfold_barcodes, record_linkage). 4 alias groups merged (Voila/Voilà
  1,533 rows; publix/Publix 938; El Corte Inglés/Ingles 695;
  Shop Apotheke/shop-apotheke 184); 7 fake multi-retailer eval groups
  removed; blocking ground truth + k-folds now alias-safe.
- Brand variants (within-GTIN measurement, decision gate): 62/62 groups
  survive case/accent folding — variants are SEMANTIC (a shoc/
  adrenaline shoc/accelerator rebrand; olvi/kevytolo parent-co;
  fitaid/lifeaid sister brands), not typos. DECISION: config-owned
  brand alias map (vocabulary.json, like FLAVOR_ALIASES), NOT edit-distance
  fuzzy. TO DO: seed vocabulary.json brand_aliases from the measured
  62-group list (~30 distinct pairs) + wire into brand blocking/veto with
  veto-asymmetry doctrine.
- Uniqueness score: implemented as measured feature (corpus token-IDF
  mean over finalized text; CLI emits product_id,cluster_id,uniqueness).
  Signal direction confirmed: linked rows 4.53 vs singletons 5.77 (fuzzy
  matches concentrate on generic listings). NOT gated on it yet.

### Consolidation + duplicate hunt (owner directive 2026-09-29)
- [x] Consolidated the per-row model composition loop (sku_info ->
      model_input_info -> build_sku_text) into ONE source:
      core.model_input.build_sku_texts(frame, structured_enabled=...) ->
      (texts, infos). Refactored call sites: pipeline.build_training_data
      (byte-identical payload; golden-byte contract tests pass),
      predict_items, rand_matching, record_linkage. Guard test updated
      (test_both_lanes_call_the_shared_builder counts the consolidated
      builder).
- [ ] Hunt remaining duplicate code (grep for parallel loops, second
      normalizations, re-implemented helpers). Known suspects: duplicated
      regexes (TODO B10), brand/fold helpers (core.attribute_conflicts
      vs critical_attributes both define NFKD accent-fold — consolidate
      into core.text), STOPWORDS islands.

### GNN + RQ-VAE Semantic IDs: SKIPPED (owner verdict 2026-09-29)
Wrong architecture for a 61k-row matching pipeline (no generative
consumer, no scale pressure, quantization hides the exact distinctions the
vetoes need — mocha vs latte). Revisit only if: catalog ~100x, generative
ranker adopted, or real-time constraints appear. Cheap substitute if a
semantic bucket feature is ever needed: category_macros + linkage cluster
id (deterministic, auditable).

## Completed log (one-liners; verbose root causes in git history)
- 230fb13 record-linkage lane + finalized TODO priority order
- 9c10c71 flag census persisted in data_prep manifest (9c10c71)
- 3695c90 B8 triplet lane graceful skip (kept optional)
- 78fa96d MNRL subset monitoring + twin warmup (item 7, EXP-03)
- ead6966 honest MNRL diet accounting (swap negatives excluded, masked-
  positive survival reported)
- 5c012d1/4e8db93 pre-existing test failures fixed (title-wins, golden)
- 7a6af8b A3 ann_finetuned attribution; f401601 fix
- 06920ce B6 _optuna_mlflow_cb deleted + wiring guard
- 54f889f hard_negative_swap_frac dead knob removed
- 051c906 stale swap-mode comment/fixture fixed
- 65b270a/23f9115 F1 <source>+aug normalized; 5c012d1/4e8db93 test+golden
  fixes; 033c4ab reviewed source fixes
- Diet gate: frac 1.00->0.80, swap_agreed deleted; bundle rebuild pending
- Smoke 128 stratified regeneration; easy-negative replace=False;
  dynamic-mask diet projection removed
