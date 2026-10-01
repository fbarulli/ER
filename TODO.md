# TODO (updated 2026-10-01, branch main)

## SESSION LEDGER — GTIN + attribute capture + veto (2026-09-30, this branch)
### Landed (all measured; suite 840 passed / 2 skipped; selftest 279 oracles green)
### Open (owner calls / next training cycle)
- [ ] TIER 2 — re-derive diet floor on the fresh bundle's numbers.
- [ ] TIER 3 — calibration fracs (counterfactual/hard-negative; sweetener
  slice first) — needs training runs; sweetener exclusion precedent (12:1)
  stays in the ledger.
- [ ] +116 clause-level attribution could move beyond commits-level if a
  consumer asks (per-clause manifest diff between worktree states).


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
- **P0 — VALIDATION REBUILD (owner ruling 2026-09-29, supersedes the 3k/5k
  lanes)**: emit ONE final validation CSV from a single merged component graph.
  Blocks any TIER 3 decision. Scoped below.

## P0 — VALIDATION REBUILD (owner ruling 2026-09-29: one final validation CSV) — COMPLETE 2026-09-30/10-01
Status: single merged component graph + `folds.derive_holdout` SSOT entry point;
`data/final_validation.csv` (6,351 = 565 pos / 5,786 neg, folds 2+3);
`results/training/validation_fold_map.csv` (14,946 entities); leak guards
raise-before-write; 9 leak regression tests; regen byte-identical after the
2026-09-30 wiring waves; selftest 279 oracles green. 3k/5k lanes: keys deleted
(paths.yaml + DataFilesSpec), producers deleted, final_inference retargeted to
the scored-pair population (schema-first); diet_manifest verified lane-free.
Residual open decision (NOT validation): scored-half thinness + slice-flag
set semantics — see Open gaps.

**Owner ruling.** Do not keep the 3k/5k lanes. Add everything back to training
and keep ONE final validation CSV. Fix the gates so they are measured on a
population the decision did not touch.

### The blocker that was found: 74.7% of current validation is contaminated
- `row_bc` (barcodes) and `data/labeled_pairs.csv` (gtins) looked like disjoint
  namespaces: **intersection 0/5,428**. The training-side component split was
  structurally blind to the validation set, so neither side protected the other.
- Root cause of the apparent disjointness is missing normalization, not two
  identifier worlds. `norm(s) = strip non-digits, zfill(14)` gives:
  - validation gtins that ARE `row_bc` barcodes: **5,428/5,428 (100%)**
  - positive pairs with both sides resolvable: **1,414/1,414 (100%)**
  - `row_bc` matching `dataset.gtin` (normalized): 14,901/14,921
  - raw `dataset.gtin` lengths are messy (7..14 digits), which is why the
    unnormalized join returned zero.
- Measured contamination of the CURRENT protocol (train = barcode folds 0+1):
  - POSITIVES: both gtins in train **23.3%** / one in train **51.4%** /
    clean **25.3%** -> only 358/1,414 are clean.
  - NEGATIVES: both **24.2%** / one **49.2%** / clean **26.6%**.
  - So the test-quarter P@R95 measures memorization as well as generalization.

### The fix: single merged component graph (feasible, verified)
- Union-find over the normalized entity namespace, with edges from BOTH
  `bundle['pos']` base positive pairs AND `labeled_pairs` positive pairs.
- Merged graph is safe to split (no giant component):
  - entities 14,921 -> **components 14,080**
  - **largest component = 15 entities (0.1%)**; nothing >= 50 entities
  - validation positives in components >= 50 entities: **0**
- Guarantee: because validation positive pairs contribute edges, both gtins of
  a positive always land in the SAME fold. Verified: **0 straddles in 1,414/1,414**
  positives. A test-fold positive therefore has neither side in train -> full
  leak 0% by construction.
- Negatives CAN straddle folds (mined negatives are not identity links). This
  matches current behaviour; treat it as a known, documented property, not a
  regression. Do not assert no-straddle on negatives.

### Resulting single validation CSV (folds 2+3, seed 1337)
- **758 positives + 3,873 negatives** (vs 1,414/7,722 total today).
- Training cost is effectively zero: training base positive pairs retained
  (folds 0+1) = **10,682 vs 10,692 today = -10 pairs (-0.09%)**.
- Per-field measuring power, old test quarter (~354 pos) -> new validation:

  | field | valPairs | distinct | singletons | largest bucket | old test | gain |
  |---|---|---|---|---|---|---|
  | carbonation | 654 | 3 | 0 | 362 | ~177 | 3.7x |
  | pack | 758 | 30 | 7 | 358 | ~142 | 5.3x |
  | volume | 758 | 39 | 9 | 170 | ~84 | 9.0x |
  | package_type | 267 | 7 | 3 | 166 | ~55 | 4.9x |
  | **sweetener** | **165** | 8 | 2 | 75 | ~38 | **4.3x** |
  | **flavor** | **390** | 34 real values | 4 | 42 | ~11 | **35x** |
  | pulp | 2 | 1 | 0 | 2 | ~1 | n/a |

- Fold balance is already adequate and needs no stratification to fix:
  `component_folds` gives barcodes 3730/3730/3730/3730 and base positive pairs
  5299/5453/5249/5372 against an ideal of 5343 (within +/-4%). Stratified
  assignment is therefore OPTIONAL here, not the fix — the leak and the
  gtin/barcode join are the real defects. Revisit only if a slice is still thin.

### Gate realignment required (the "align our gates" item)
- `build_field_slice.py` buckets by TWIN (1 bucket per field, ~34% each from
  2,044 twins) while `labeled_pairs` slices by CANONICAL VALUE (163 flavors).
  These are different notions of a bucket, yet the checkpoint contract compares
  a twin-bucket P@R95 floor of 0.500 against field-value P@R95 as if equivalent.
  **Reconcile to one definition before either number is a decision gate.**
- **Flavor gates — CORRECTED 2026-09-29 (an earlier entry here was WRONG).**
  An earlier note claimed "107/163 flavor values are singletons, drop per-flavor."
  That counted FUSED PAIR COMBINATIONS (`{orange,lemon}` treated as one unit),
  not flavor values, and manufactured fake sparsity. The truth:
    - `flavor_set` has **34 real values**; true singletons = **4**.
    - Per-value coverage in the 758-pair validation: ginger 83, fruit 68,
      apple 59, coffee 45, lemon 40, aloe 30, peach 19, strawberry 19,
      **orange 18**, tonic 18, berry 13, coconut 11.
    - **6 of 31 values have n >= 30** -> per-flavor IS supportable for the top 6.
  Correct policy: **gate on the top-6 flavor aggregate; treat the tail
  (peach/orange/tonic and below) as INFORMATIONAL ONLY** — no floor. Orange at
  n=18 is a usable coarse signal but far below sweetener's largest bucket (75),
  so it must not carry a standalone floor.
- **Drop pulp as a gate entirely**: 2 pairs in validation. Unmeasurable at any
  budget that does not also break the component constraint. Root cause is
  POPULATION SCARCITY, not split or parsing weakness — do not "fix" an extractor:
    - `pulp_set` populated in only **302/13,250 (2.3%)** canonical records vs
      flavor 74.2% / carbonation 81.8% / sweetener 43.9% (~30x rarer).
    - 140/5,428 (2.6%) validation gtins; positives are the SAME product so
      "both sides flagged" reduces to "is this a pulp product" -> only
      **7/1,414 (0.5%)** of verified positives are pulp -> 2 in the val half.
    - Provenance is SOURCE ATTRIBUTES, not canonical text (hence the two-way
      disagreement: 301 flagged records whose text never says "pulp", and 25
      texts that do but are unflagged). Unioning the attribute signal recovers
      only 26 more gtins (+9%); 302/324 = 93% already captured. No real gap.
  Pulp becomes a gate only if the verified positive population grows for that
  category — better allocation cannot fix a 0.5% category.
- Per-slice gates that ARE supportable after this rebuild: volume, pack,
  carbonation, package_type, sweetener (aggregate per value, 8 values, largest
  bucket 75, only 2 singletons).

### The 16,345 dead masked-positive anchors — NOT validation (owner asked)
- Confirmed they must not enter the validation CSV. Reasons in order of force:
  1. **Distribution shift** — they are masked text (20-30% `[MASK]` tokens);
     P@R95 on them measures dropout robustness, not clean-text matching.
  2. **Circularity** — they are the OUTPUT of the augmentation knobs being
     tuned (`frac`, swap fields, twin composition). TODO's own doctrine: the
     deciding metric must be measured on a population the decision did NOT touch.
  3. **Not independent** — they derive from the same source anchors as training.
- Correct use: a **dedicated augmentation-invariance probe**, kept strictly
  separate from P@R95, answering "did augmentation damage the model?". The
  harness already exists in `build_field_slice.py` (twin buckets + the
  twin-bucket floor >= 0.500 in the checkpoint contract); it currently reads
  only `target_mode == "counterfactual"` and could gain a masked-anchor bucket.
- "We don't need 8k, add them all back": there is no 8k to add back. The 3k/5k
  files DO NOT EXIST (`training_data/` is absent), so nothing is currently
  reserved from them. The 7,722 figure is validation NEGATIVES, not reserved
  positives. The 16,345 dead anchors are a training-side issue the gate already
  excludes.

### Implementation checklist (LANDED 2026-09-30 — measured, not estimated)
**All targets below were re-measured, because the census drifted (1,414 -> 1,223
positives) and every number in this section was written against the old census.
Re-deriving them was the first task; a stale target is a wrong target.**

- [ ] **STILL OPEN — the scored halves are thin, and this is a real decision.**
      Withholding straddling negatives is the honest choice (one side is a
      trained-on entity), but it costs 4,741 of 5,781 negatives:
      **DEV 315 pos / 585 neg, TEST 249 pos / 455 neg.** A Youden fit and a
      P@R95 on 249 test positives is thin. The alternative is to assign each
      negative by the fold of its TRAIN-side entity so all 5,781 survive
      whole — but that changes what a negative measures and needs an owner call.
- [ ] **STILL OPEN — slice flags are set-valued and per side; gates must not
      compare `v1 == v2`.** Measured disagreement across the 564 positives:
      flavor 366, package_type 173, sweetener 155, carbonation 32, pack 31,
      volume 9. Same product, two feeds, different extracted sets. Treat each
      side as a bag of values. Flavor has 52 real values (10 with n>=30, 8
      singletons) after splitting the stored list-literal — the 129 "distinct"
      in the manifest are FUSED combinations, exactly the trap noted above.
- (closed 2026-10-01 — informational: `normalize_gtin` kept for CROSS-NAMESPACE
      edge resolution only; measured to contribute nothing to the split itself.)
- (closed 2026-10-01 — `scripts/diet_manifest.py` verified lane-free: bundle-agnostic, zero 3k/5k references; gate wired into the rebuild path.)

## P1 — FINALIZE before training (data alignment etc.)
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

## Open gaps (P2 — remaining) — STATUS AUDIT 2026-10-01
### NOT SOLVED — needs work (4)
- [ ] Positive/negative coverage EXPANSION (the datagen action wave): 1,414 positives / 1,414 pack-blocker negatives is the floor, not the target — evidence exists in the census + veto audit; expansion executes after the running rebuild lands.
- [ ] 4a39fdc incorrect numbers: 3 inconsistent figures to replace with the measured "3 records" statement (being measured by the decision-briefs agent now).
- [ ] MNRL telemetry port: fold-census CSVs the reference contract expects (code-only; queued next wave).
- [ ] Duplicate-code hunt: remaining islands (NER dead cluster, hpo_persistence PG machinery, STOPWORDS residues) — LOW priority.
### DECIDED IN CODE (owner posture change: the system decides from evidence, 2026-10-01) — (4)
- [x] Thin scored halves: DECIDED `split.negative_fold_policy: "train_side"` (policy B) —
      evidence/measured at tests::test_scored_half_decisions + the DECISION block in
      src/training/build_final_validation.py: with policy B every negative scores by
      train-side entity fold: DEV 592 -> 1,087 negs (+83.6%), TEST 466 -> 957 (+105.4%),
      min_test_negatives=5 thin-cell share DEV 67.6% -> 64.9% / TEST 70.2% -> 67.2%,
      no trained-on endpoint enters the scored half under either policy. Policy A
      ("withhold_straddle") stays reachable via the same config key. Artifact regen:
      emitters support config; `PYTHONPATH=src .venv/bin/python -m src.training.build_final_validation`
      is the next regen command for the rebuild chain.
- [x] Slice-flag gate semantics: DECIDED `evaluation.slice_agreement: "set_bag"` (bag
      equality) — implemented at the count sites via count_slice_disagreements()
      (src/training/build_final_validation.py, imported by the consumer gate);
      measured on data/final_validation.csv ALL disagree counts byte-identical
      (volume 13, pack 48, package_type 153, sweetener 127, flavor 363,
      carbonation 38 — attribution recorded in write_manifest's comment;
      legacy "scalar" reachable via the same config key). Pinned in
      tests/test_scored_half_decisions.py.
- [x->SPEC] TIER 3 calibration fracs: FORMALIZED as documented-but-disabled
      `calibration_sweep:` block in config/training.yaml + CalibrationSweepSpec
      (src/core/schemas.py) — fracs sweep declared only when a training-run lane
      exists (fail-loud schema: enabled=false refuses non-null fractions; slice
      order pinned sweetener first), acceptance = the twin-bucket floor contract
      (per-bucket twin P@R95 >= 0.500; overall up + twin down = over-smoothing =
      reject; twin margin must lift off ~0.008). NOT run here (no training).
- [x->VERIFIED] Veto-eligibility: END-TO-END CONFIG-SOURCED — runtime veto consumes
      rand_matching.targeted_veto_gates.veto_dimensions only
      (src/training/rand_matching.py:320,422,450), schema allow-list gates what may
      be configured (src/core/schemas.py TargetedVetoGatesSpec._veto_dimensions_are_critical),
      and veto_eligibility_ledger (src/core/attribute_conflicts.py) reports the same
      config surface + the exact config delta per dimension; sweetener stays
      audit-column-only; no orphan/hard-coded veto logic found. (The veto-admission
      rule itself is being landed by the running veto-admission wave.)
### NOT SOLVED — process debts, do not block training (2)
- [ ] Metrics unversioned (results/ gitignored — today's evidence lives locally only).
- [ ] 41MB prepared bundles in git while DVC disabled (bloat policy).

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
- 2026-10-01 - [x] GTIN integrity: longest-digit-run extraction (0 cells changed on this corpus — census regression byte-identical; the "3,715 invalid" now closes exactly as 3,646×11 + 50×10 + 22×7); `gtin_equivalent()` sibling equality (UP...
- 2026-10-01 - [x] model_input dead-def consolidation + description alias (`description_short_eng` OR `description`): deduped lane regains description evidence — 52,856/63,079 rows (was 0); +330 carbonation / +545 sweetener / +67 pulp fille...
- 2026-10-01 - [x] brand_aliases seeded in vocabulary.json (8 entries, 6 families: shoc chain, hi/hiball, fitaid/lifeaid, olvi/kevytolo, biotech, dg/ting) + 27 false-veto dissolutions, 22 declined groups filed; alias-aware fold wired at ran...
- 2026-10-01 - [x] AttributeUniverse (src/core/attribute_universe.py): all 37 raw keys registered (FieldSpec kinds/parser/conflict), census (rows/distinct sets/same-GTIN conflict rates; verify_census self-check), datagen_budget() (donor/vet...
- 2026-10-01 - [x] Capture wiring (additive, byte-prefix contract): pack material title∪attributes (canonical material sets 774 → 7,433 populated), juice content bands (27 canonical bands), [FIELD_PACK_MATERIAL] / [FIELD_JUICE_CONTENT_BAND]...
- 2026-10-01 - [x] Merge re-run + attribution: 71,623 → 63,079 (+116 vs last refresh — ATTRIBUTED to commits 0452692..2d3ac4b (GLN quarantine/collapse repairs), NOT to wave-1 fixes (verified byte-identical dedupe at HEAD-clean via worktree...
- 2026-10-01 - [x] labeled_pairs + gate pins: fallback 42,039 → 41,748 (−291 pairs now resolvable on evidence; proceed 1,239); pins updated together (common.py + selftest.py) with attribution cascade (proceed 1,737→1,506→1,395→1,239 narrati...
- 2026-10-01 - [x] Diet gate wired INTO the rebuild path (`_run_diet_gate`, colab.py:2886/3031): rebuild refuses to ship on gate failure; cached gate-failing bundles are never reused. Fresh worker_1: neg_aug_frac 0.3136 OK, ratio 1.1664 OK...
- 2026-10-01 - [x] TIER 1(e) drift hardened: `prepared_bundle_drift_strict` (default now TRUE post-rebuild; env override honored); stale bundle hard-fails, fresh loads; audit reproducibility kept via env=0.
- 2026-10-01 - [x] `pack_material` ADDED to veto_dimensions (owner go 2026-09-30). Evidence: hard_no 92,259 / both-populated 67,899 / disjoint 32,641 (48%). Identity-safe by construction (canonical-level union; verified byte-identical train...
- 2026-10-01 - [x] 3k/5k lane retirement: paths.yaml + DataFilesSpec keys deleted; run_ann_full_data.py + sample_deduped_dataset.py deleted; STOPPED live consumers: sample_balanced_pairs.py (miner/gate test pins), colab.py training_csv (own...
- 2026-10-01 - [x] Dashboard: /gate route (original-columns samples, per-dimension mismatch evidence w/ deciding-clause gloss, full strings in details), /datagen (session ledger + census auto-render) and /graphs track pages; all six routes...
- 2026-10-01 - [x] BLOCKER FIRST — single source of truth for split derivation. DONE. `folds.derive_holdout` is the entry point and now builds the GRAPH internally, so no caller can bypass the leak fix. `holdout_split` / `partition_componen...
- 2026-10-01 - [x] `normalize_gtin()` added as the single entity key — and MEASURED TO BE THE WRONG FIX. The "0/5,428 intersection / missing normalization" root cause above is incorrect. Measured on the live data: the deduped set holds 14,9...
- 2026-10-01 - [x] Validation positive edges folded in behind the single entry point. `folds.merged_component_graph` = training positives UNION labeled positives. Negatives are never unioned (a similarity claim is not an identity claim). Me...
- 2026-10-01 - [x] Leak closed, and the leak guarantee is a hard stop, not a log line. 0/1,223 positives straddle a fold; 0 test-fold positives have either side in train. `build_final_validation` raises BEFORE writing if either holds. Repro...
- 2026-10-01 - [x] ONE validation CSV emitted: `data/final_validation.csv`, 6,345 rows = 564 positives / 5,781 negatives (folds 2+3), 23 columns — `gtin1, gtin2, gtin1_norm, gtin2_norm, true_label, fold, fold_2, component_id, component_id_2...
- 2026-10-01 - [x] Complete split accounting emitted: `results/training/validation_fold_map.csv`, 14,981 entities -> fold + component (7,508 train / 3,742 dev / 3,731 test). Required, not optional: the validation CSV holds only the scored h...
- 2026-10-01 - [x] `evaluate_models.py` retargeted at the artifacts. It built its OWN graph from the labeled census ALONE with `component_split_k=2` — the leak site. It now reads the fold map; accounting closes 8,889/8,889; 0 scored pairs t...
- 2026-10-01 - [x] Leak regression tests in `tests/test_validation_leak.py` (9 tests). Includes `test_merged_graph_is_not_vacuous`, which asserts the bare graph DOES straddle before asserting the merged one does not — a first draft named on...
- 2026-10-01 - [x] RETIRED — the 3k/5k lanes (2026-09-30). paths.yaml + DataFilesSpec keys deleted; `run_ann_full_data.py`, `sample_deduped_dataset.py`, and its lane test deleted. Deliberately left in place (live consumers, evidence filed):...
- 2026-10-01 - [x] DONE (earlier this P0) — single SSOT split entry point. `folds.derive_holdout` builds the graph internally; `holdout_split`/`partition_component_pairs` banned outside `folds.py` and enforced by the selftest guard.
- 2026-10-01 - [x] DONE — `normalize_gtin()` scope fixed wrong: kept as CROSS-NAMESPACE bridge only (see the measured reversal at the top of this P0 — zfill on `row_bc` silently empties 8,559/14,981 filters; the real fix was the merged edge...
- 2026-10-01 - [x] DONE — validation positive edges folded in behind the SSOT (24,420 training + 1,223 validation over 14,981 entities; 0 straddles).
- 2026-10-01 - [x] DONE — ONE validation CSV emitted (`data/final_validation.csv`, folds 2+3, slice flags for volume/pack/package_type/sweetener/flavor/ carbonation; pulp excluded 2.3% population).
- 2026-10-01 - [x] DONE — leak regression tests (9 tests; straddle + both-sides-in-train hard stops inside `build_final_validation`).
- 2026-10-01 - [x] DONE — flavor reopen trigger restated on aggregate/twin buckets (top-6 aggregate gate; tail informational-only; policy CLOSED 2026-09-29).
- 2026-10-01 - [x] Bundle rebuild — LANDED 2026-09-30 with the gate wired in. The 0.2119 clause-1 FAIL was the STALE bundle (frac=1.00, pre-TIER-1(a) mint). Fresh worker_1 (built on data/dataset_deduped.csv, since the retired train_minus_50...
- 2026-10-01 - [x] Swap-copy diet accounting for MNRL: RESOLVED IN CODE (ead6966 excludes swap negatives from neg_aug_views when loss=mnrl) — fresh bundle diet numbers confirm the accounting (neg_aug 0.3136 on the mint-inclusive denominator...
- 2026-10-01 - [x] Masked-positive minting survival — stale figure refreshed on the fresh bundle: DEAD masked positives 16,345 → 14,832 @ frac 0.80 (quantized census high 9,660 / low 9,828). Diet REPORTS real survival. Remaining owner decis...
- 2026-10-01 - [x] Coverage analysis remainder — SUPERSEDED by results/attribute_ universe_census.json + the AttributeUniverse datagen_budget() class (measured per-key headroom, donor/veto/eval/channel classes). The agent-2 handoff items (a...
- 2026-10-01 - [x] Flavor-twin policy: CLOSED — accept flavor twins, no allowlist (owner verdict 2026-09-29, evidence-based). The "717/717 prose-contradicted" figure does NOT argue for exclusion: - the metric measures whether the old flavor...
- 2026-10-01 - [x] TIER 0 — unblock attribution (DONE 2026-09-29): `mnrl_monitoring. enabled: true` (config/training.yaml). `twin_loss_warmup` deliberately left OFF — it changes loss WEIGHTING, so it is a TIER 3 calibration decision, not ob...
- 2026-10-01 - [x] TIER 1(a) — counterpart positives for swapped negatives (DONE 2026-09-29): all 2,709 real `swap_values` hard-negative copies now reach a gradient; measured on the real bundle, 100% coverage. Mechanism: an anchor-only tran...
- 2026-10-01 - [x] TIER 1(b) — WITHDRAWN 2026-09-29: redundant, and unsafe to "fix". The 16,345/21,373 dead masked-positive measurement is REAL, but the diet gate ALREADY accounts for it: `diet_manifest.effective_pos_views` subtracts them,...
- 2026-10-01 - [x] Swap-copy accounting for MNRL (diet): 2,709 swap copies diet-counted as augmented views but 0% MNRL-train — ead6966 implemented the exclusion for MNRL. Diet side closed; the underlying 2,709 inert rows are now TIER 1(a)....
- 2026-10-01 - [x] T1.5 shipped (21705f9): same retailer + same checksum-invalid barcode + same product -> collapse. 108 groups collapsed, 11 escalated.
- 2026-10-01 - [x] T3 IDENTITY LOSS FIXED (data deletion, not noise). T3 keyed its collapse on (retailer, title) ALONE. T2 explicitly DEFERS rows whose trusted barcodes disagree, and T3 then merged them anyway: 692 groups, 1,778 rows, 1,223...
- 2026-10-01 - [x] Verified after the fix: deduped 61,414 -> 62,963 rows; trusted barcodes present 12,986 -> 13,250; canonical products orphaned 264 -> 0; 0 trusted barcodes lost their last row (new hard invariant gate); closure 71,623 == 6...
- 2026-10-01 - [x] New hard gate: the dedupe now REFUSES to finish if any trusted barcode present in the input is absent from the output. This failure mode is silent and unrecoverable downstream, so it must never be a report.
- 2026-10-01 - [x] `core/product_identity.py` (new SSOT): one descriptor bundle with every field a SET, so a descriptor restated across title/attribute/category collapses to one token and cannot move a comparison. `identity_conflict` is the...
- 2026-10-01 - [x] price / url / image_url removed from identity: `price` is a seller attribute, not a product description. T2 no longer keys on it, and representative choice uses DESCRIPTOR completeness (`descriptor_completeness`) instead...
- 2026-10-01 - [x] `results/training/dedupe_conflicts.csv` (new): the durable review queue. Anything the descriptor bundle cannot settle is ESCALATED, not guessed — 68 proven splits + 11 unresolved across 3,504 malformed-barcode groups. "Th...
- 2026-10-01 - [x] Fixed a silently DEAD dimension: `package_material` used the regex `pack\s*material`, which compiles to `pack\s*material\s*:` and never matches the real corpus key `Pack Material Type:` — it read empty across all 35,571 n...
- 2026-10-01 - [x] 17 new regression tests in `tests/test_dedupe_identity.py` pinning the identity partition, the T1.5 verdict table, completeness independence from price/urls, and absence-is-not-contradiction. Full suite: 586 passed, 2 ski...
- 2026-10-01 - [x] Consolidated the per-row model composition loop (sku_info -> model_input_info -> build_sku_text) into ONE source: core.model_input.build_sku_texts(frame, structured_enabled=...) -> (texts, infos). Refactored call sites: p...
