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
- [ ] **Bundle rebuild (blocks training) — RECHECKED 2026-09-29, the gate does
      NOT pass and a rebuild alone will NOT fix it.** Ran
      `scripts/diet_manifest.py data/prepared/full/worker_1_baseline.pkl.gz`:
      real exit code **2 (FAIL)**. Measured:
        clause 1 `neg_aug_frac` = 6,111/28,844 = **0.2119** < 0.30  -> FAILS
        clause 2 `pos/neg` (MNRL-surviving) = 30,438/28,844 = 1.0553 -> passes
      The stale bundle is masking.frac=1.00 (config now 0.80). The
      frac=0.80 rebuild moves only the BUNDLE-ONLY ratio (1.622 -> ~1.474,
      unverifiable from the repo since the ratio is computed live at
      prepared_bundle.py:155). It does NOT touch clause 1, which lives on
      the NEGATIVE side (`hard_negative_frac` + the MNRL swap exclusion).
      Note `diet_manifest` gates the MNRL-SURVIVING ratio (1.0553), not the
      bundle-only one — "under the 1.50 ceiling" is not a claim the gate
      ever evaluates. **Rebuild only after TIER 1 lands**; a rebuild now
      still exits 2. **STRUCTURAL: the gate is NOT wired into the rebuild
      path** — `_build_local_training_bundles` (colab.py:2914-3007) only
      calls `load_prepared_bundle` for shape validation, so a rebuild will
      COMPLETE SUCCESSFULLY while the gate still exits 2. Wire
      `scripts/diet_manifest.py` into the rebuild path, otherwise the
      rebuild stays a silent no-op. Who triggers: `_build_local_training_
      bundles` (src/cli/colab.py) runs `training.train` locally. Test on a
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
      1 partial, 1 FAIL). **2026-09-29 audit DISPUTED both remaining items.**
      (a) 49->30 ambiguous_volume arithmetic is WRONG: shipped code moved
          49 -> **79** raw rows (the 30 is 79-49, misread). Unit is also
          wrong — `_flag_census` (data_prep.py:136-157) counts per CANONICAL
          RECORD; real effect is **3 records (was 2)**: 8722200964525
          de-flags, 5021554989646 + 7311676670255 newly flag. 45 of the 49
          pre-fix hits have a MISSING gtin and are dropped by the guard
          (pipeline.py:1722-1738) before any canonical exists. THIRD
          inconsistent number lives at data_prep.py:59 ("49 -> 69") —
          not reproducible under any of 8 predicate variants. Replace all
          three with the measured 3 records.
      (b) claim 8 `flip_validity_audit.py` self-review: TWO real defects.
          - `_value_strings` rule R3 (centilitre, ml//10) emits a BARE
            integer the matcher accepts unitless: 671/817 of its hits (82%)
            land on unrelated numbers (pct100, caffeine 15 25, 3x). R2 is
            dead code (subsumed by R4). So volume/pack figures are
            CONTAMINATED: twin volume contradicted 0.5459 -> **0.4521**;
            swap_positive 0.5361 -> 0.4689; swap_negative 0.5802 -> 0.4987.
            pack is contaminated the OTHER way (new-side hits push rows
            into `both`): twin pack 0.4118 -> 0.4314.
            **`TODO.md`'s "volume 0.55" cites the contaminated figure.**
          - `_classify` (lines 122-123) probes only the FIRST token of a
            multi-token value: 13.1% of rows affected, so the metric can
            only UNDER-detect. sweetener_type disagrees on 74% of them.
          - All NON-numeric fields are bit-identical under every policy,
            so **the 717/717 flavor figure is TRUSTWORTHY** and the closed
            flavor verdict (P1) stands on it.
          - `global` section is a statistic the code never computes — see
            the withdrawn TIER 1(c).
      VERIFIED CLEAN: the frac-1.00/1.622 sweep found NO load-bearing site
      assuming either. `diet_manifest.py` is frac-agnostic (recomputes from
      the bundle's own arrays, reproduced 0.2119/1.0553 exactly);
      `prepared_bundle.py:155,193-199` interpolates live counts;
      `colab.py:2826-2839` `_tree_digest()` hashes config/* into the bundle
      cache key so the frac change invalidates the cache; no test pins 1.622
      or any frac. 2 cosmetic defects: diet_manifest.py:206-224 computes
      `pos_neg_ratio` and `effective_ratio` from the IDENTICAL expression
      yet labels the first "bundle-only (informational)" — it is not, and it
      is the value the gate consumes; diet_manifest.py:120-121,226-231
      print ACTIVE-config masking values beside bundle numbers that enter
      no arithmetic.
- [x] **Flavor-twin policy: CLOSED — accept flavor twins, no allowlist
      (owner verdict 2026-09-29, evidence-based).** The "717/717
      prose-contradicted" figure does NOT argue for exclusion:
        - the metric measures whether the old flavor's surface form survives
          in the PROSE, and 78% of flavor anchors name the old flavor 2x+ with
          37% multi-flavor anchors — that residue is expected mechanically,
          because the transplant lives in the STRUCTURED-TOKEN channel, not
          in prose. The metric is pointed at the wrong channel.
        - against intent, flavor is the BEST-separated twin field zero-shot:
          margin **0.0113, 89.3% ranked correct** — ahead of package_type
          (0.0081) and volume (0.0052). Excluding it would delete the
          strongest counterfactual pressure the lane has.
      REOPEN ONLY on a TRAINED checkpoint if either objective trigger fires:
      (a) per-bucket flavor twin P@R95 falls below the 0.500 invariance floor
      (checkpoint eval contract), or (b) flavor twin margin drops below its
      zero-shot 0.0113 baseline. Do not reopen on contradiction-rate evidence
      alone. Bundle rebuild inherits this verdict (nothing to implement).
- [ ] Checkpoint eval contract execution: after each epoch/checkpoint,
      build_field_slice.py + minimal_flip_slice.py on the live bundle;
      P@R95 0.355 -> ~0.500 by epoch 3; twin-bucket floor >= 0.500;
      twin margin must lift off ~0.008. (Harness ready; blocked on the
      bundle rebuild + first training run.)

## AUGMENTATION TIERED ACTIONS (added 2026-09-29; HPO dropped — not in use)
Decision doctrine: sort every augmentation change by whether it needs a
TRAINING RUN to adjudicate. A row that is minted, audited, diet-counted and
then silently dropped is a LEAK, not a tradeoff — leaks are fixed, not
measured. Metric decisions apply only to rows that actually train, and the
deciding metric must be measured on a population the decision did NOT touch
(held-out P@R95), never on the augmented rows being questioned.

- [x] **TIER 0 — unblock attribution (DONE 2026-09-29)**: `mnrl_monitoring.
      enabled: true` (config/training.yaml). `twin_loss_warmup` deliberately
      left OFF — it changes loss WEIGHTING, so it is a TIER 3 calibration
      decision, not observability. Shipping it here would have been an
      unevidenced metric call.
- [x] **TIER 1(a) — counterpart positives for swapped negatives (DONE
      2026-09-29)**: all 2,709 real `swap_values` hard-negative copies now
      reach a gradient; measured on the real bundle, 100% coverage.
      Mechanism: an anchor-only transplant invalidates the source's unchanged
      positive, so the triple builder omitted every swap row. New
      `masking.mint_swap_counterpart_positives` registers a compatible
      positive per copy in two real sub-cases:
        - 2,216 rows — source positive CARRIES the transplanted field, so the
          same donor transplant is replayed onto it (new payload row; feature
          lineage derives from the source positive, never re-claiming the
          already-extended swap copy).
        - 493 rows — source positive is SILENT on that field (481) or already
          AGREES with the donor value (12), so the transplant cannot
          contradict it and the source positive is registered against the copy
          directly (no new payload row).
      Gate effect: `neg_aug_frac` 6,111/28,844 = 0.2119 -> 8,820/28,844 =
      **0.3058**, clearing `diet_min_neg_aug_frac=0.30`. `diet_manifest` was
      updated to count a swap row ONLY when its copy anchor actually owns a
      positive (counted per row from the bundle), so the stale bundle honestly
      still reports 6,111 and the gate never assumes a population label.
      A test asserts a reused source positive can never CONTRADICT the copy.
- [x] **TIER 1(b) — WITHDRAWN 2026-09-29: redundant, and unsafe to "fix".**
      The 16,345/21,373 dead masked-positive measurement is REAL, but the diet
      gate ALREADY accounts for it: `diet_manifest.effective_pos_views`
      subtracts them, so `pos_views` is reported as surviving (46,783 - 16,345
      = 30,438) and the 1.0553 ratio is computed on the surviving count.
      Pruning them from `pos` would make the gate subtract the same 16,345 a
      SECOND time, under-reporting positives and risking a spurious clause-2
      failure. It also breaks the payload-suffix feature invariant
      (`prepared_bundle._validate_augmented_features` re-derives features from
      the audit and requires it to cover every appended payload row), so the
      only correct prune is a full payload reindex — not worth it for a
      cosmetic bundle-size win on rows the gate already excludes. TIER 2 must
      re-derive the floor from the post-1(a) numbers.

- [ ] **TIER 1(c) — WITHDRAWN 2026-09-29: the premise was false.**
      The "volume realized 1.428x
          the 0.35 cap (1.549x swap_positive)" figures came from
          `flip_validity_audit.py`'s `global` section, whose denominator is
          WRONG on two counts: (i) it divides by realized audit rows
          (8,790) while the caps bind against `_swap_pick_total`
          (train.py:1008-1011) = 10,415, inflating every share 1.185x; and
          (ii) the FIELD cap is enforced PER LANE (fresh Counter per
          function, masking.py:629,:827 — only `shared_value_counts` threads
          across lanes), so a "global" field share is a statistic the code
          never computes. Recomputed on the code's own arithmetic: field cap
          peaks **0.600x**, value cap **0.875x** — COMPLIANT, zero breaches.
          The soft fallback at masking.py:680-682 did fire but never pushed
          a lane past the cap. Residue is a soft/hard SEMANTIC mismatch
          (config wording vs behavior) needing a decision, NOT an
          enforcement fix. Fix the audit's denominator/reporting separately.
      (d) Prune dead knobs `swap_agreed_frac` / `hard_negative_swap_frac`
          (gone in 7967ccc/54f889f) from any surviving sidecar, so drift
          detection is not reading a stale schema generation.
      (e) Tighten bundle drift from WARNING to hard failure
          (prepared_bundle.py:283-299) — the current bundle loaded cleanly
          while carrying two deleted knobs. **SEQUENCING: land AFTER the
          rebuild.** Hardening drift makes the current bundle UNLOADABLE,
          which strands every open audit finding measured on it (the 717/717
          flavor figure, the volume contamination numbers, diet 0.2119).
          Rebuild first, then harden. Wire `scripts/diet_manifest.py` into
          `_build_local_training_bundles` (colab.py:2914-3007) at the same
          time — it is the only diet gate and today nothing calls it, so a
          rebuild completes "successfully" while the gate still exits 2.
- [ ] **TIER 2 — re-derive the diet floor AFTER Tier 1** (do not tune the
      data to hit a stale number): `diet_min_neg_aug_frac: 0.30` was
      calibrated against a denominator the gate now correctly rejects.
      Measured: honest MNRL = 0.2119; counting the 2,709 never-trained
      swap copies = 0.3058 (would have passed by a hair). `ead6966` made
      the denominator honest and dropped it under the floor. A floor has to
      be re-derived when its denominator's definition changes. Sequence:
      Tier 1 changes the numerator by construction, so re-derive once,
      after, not before.
- [ ] **TIER 3 — genuine calibration; requires training runs; decided on
      held-out P@R95 only**:
      (a) `counterfactual_frac: 0.10` and `hard_negative_frac: 0.30` —
          unswept, and the entire counterfactual hypothesis rests on the
          former. (Config-only route for hard_negative_frac alone is
          ~0.33, but that is a band-aid: prefer Tier 1(a), which makes the
          rows real rather than merely numerous.)
      (b) SWEETENER is the weak slice and gets the first run: margin
          **0.0017, 62.4% ranked correct**, 35.8% flip-SUPPORTED, n=109.
          Flavor second (0.0113/89.3%). `swap_max_donor_overlap: 0.95` has
          never fired (0/490) — an untested net, not a tuned value.
          CAVEAT: the 0.0017/62.4% margins come from
          `minimal_flip_slice.py`, which has NOT been self-reviewed (only
          `flip_validity_audit.py` was). The 35.8% flip-supported rate is
          from flip_validity_audit and is on a non-numeric field, so it is
          clean. Review minimal_flip_slice before treating the margin
          ranking as settled.

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
      Flavor policy CLOSED (accept — see P1). Remaining: build the explicit
      flip-policy mechanism, which does NOT exist today — there is no
      allowlist in config, `MaskingSpec` or `MaskingProfileSpec`; only the
      implicit parseability whitelist `_FIELD_PREFIXES` (masking.py:38-48).
      `experiments.md:61`'s "one-line config change" is wrong: it needs
      masking.py + both schemas + config. Purpose is the SWEETENER slice
      (Tier 3b), not flavor.
- [ ] Swap/twin fracs and caps hand-picked (0.20/0.10, 0.35/0.03) and never
      swept. HPO is NOT in use — calibration is by explicit Tier 3 A/B on
      held-out P@R95 instead.
- [x] Swap-copy accounting for MNRL (diet): 2,709 swap copies diet-counted
      as augmented views but 0% MNRL-train — ead6966 implemented the
      exclusion for MNRL. Diet side closed; the underlying 2,709 inert rows
      are now TIER 1(a).
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
