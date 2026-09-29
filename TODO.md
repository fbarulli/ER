# TODO (updated 2026-09-29, branch training-sid-hybrid)

## PRIORITY ORDER (owner ruling 2026-09-29)
Work top-down. The recent graph/linkage additions are DEAD LAST — do not
start them until every P1/P2 item is closed and training is unblocked.
- **P1 — FINALIZE before training (data alignment etc.)**: bundle rebuild
  (Questions #1), MNRL telemetry port (reference-contract CSVs), swap-copy
  diet accounting (owner decision), masked-positive minting survival,
  coverage-analysis remainder, 4a39fdc verification remainder, flavor-twin
  policy decision, smoke-sampler code committed.
- **P2 — remaining open gaps** (parser/extraction, augmentation, training/
  eval, process) below in "Open gaps".
- **DEAD LAST — recent additions (2026-09-29)**: barcode-less record
  linkage (KNOWN DEFECT: flavor merge), two-stage graph matching
  architecture (GNN bi-encoder retrieval + cross-graph alignment
  verification), graph-construction improvements Tier 1-3, and the queued
  neighborhood-context recall measurement. See the DEAD LAST section at the
  end of this file.

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

## FINDINGS.md triage (submission branch audit vs HEAD — 2026-09-28)
Historical audit reports exist (origin/submission FINDINGS.md). Triage
status per item, verified against HEAD:
- A1 hpo._grid_folds data[2]/data[4] swap (cv+grid crash): FIXED —
  current code reads data[3]/data[5] (hpo.py:282-283).
- A2 rerank country out-of-bounds for canonical endpoints: FIXED —
  padding present before use (rerank.py:267-273 -> :293).
- [x] A3 gate slots labeled ann_finetuned without realized replacement —
      FIXED (agent 2, merge 7a6af8b; fix f401601 on
      fix/ann-replacement-attribution, pushed). The one-line move applied:
      base_population overwrite now INSIDE `if replacement is not None:`
      (src/training/training.py:3154-3167). Guard:
      tests/test_ann_replacement_attribution.py (2 tests: unreplaced slot
      keeps its population; realized replacement -> ann_finetuned).
      Suite: 532 passed + 2 skipped + 2 pre-existing failures.
- [x] A4 _write_datapoint_usage brittle on 0-presentation populations —
      CONFIRMED FIXED, no change needed (agent 2, verified vs HEAD).
      Evidence: docstring "audit A4 — this is no longer a hard failure"
      (training.py:2699); status ladder ok/missing/eval_only/not_reached/
      unavailable/unregistered (training.py:2786-2798); missing ->
      WARNING print + n_missing_datapoint_populations, fold continues
      (training.py:2827-2843). The only remaining raise is the INTENDED
      UnregisteredDatapointPopulationError in
      _assert_datapoint_coverage_identity (training.py:2929) — provenance
      guard for unregistered producer tags, unrelated to 0-presentation.
      Regression pins pre-exist and pass (23/23):
      tests/test_datapoint_coverage.py:269-277 (non-empty registered,
      zero presentations -> missing + no raise), :247-267 (0-pair
      registered -> unavailable/not_reached), :233-245 (unregistered ->
      loud).
- [x] B6 _optuna_mlflow_cb never registered: FIXED (agent 1, commit
      06920ce). ROOT CAUSE: superseded in-commit by _optuna_tracking_cb
      (registered at study.optimize, training.py:6091-6095) which already
      logs the MLflow metric under the correct objective name
      (trial_N_objective; objective is discriminative-LR calibration Rand,
      training.py:6125) + wandb; _optuna_mlflow_cb would have logged a
      misleading trial_N_auc on a non-AUC objective. Blast radius:
      graphify affected = 0 nodes, zero consumers of trial_*_auc anywhere
      -> DELETE chosen (register would add lying telemetry). Guard test:
      tests/test_optuna_callback_wiring.py (fails if any _optuna_*_cb is
      defined but unregistered).
- [x] PRE-EXISTING suite failures found while verifying B6 (NOT agent-1
      queue, flagging for triage): (1) tests/test_model_input_contract.py
      ::test_legacy_profile_reproduces_golden_bytes — golden fixture
      expects [FIELD_VOLUME] volume_ml_250 tokens the producer no longer
      emits (golden drift vs code change); (2) tests/
      test_unit_canonicalization.py
      ::test_order_of_magnitude_title_attribute_disagreement_flags_not_flips
      — expects volume_ml 330.0, gets 3300.0 (title-volume-on-10x-
      disagreement change at TODO line 312-314 likely didn't update the
      test, or flipped the wrong side). Both fail on base 4bfe3d1 too.
- B7 Initial-fold ANN/attribute-conflict negatives impossible: DESIGNED
  (document, no fix). Static TARGETED attribute-conflict negatives DO join
  the fold from gate similarity (train.py:1291-1305; fresh bundle +466
  targeted static); dynamic ANN + dynamic attribute-conflict mining defer
  to the refresh callback because emb0 is empty pre-checkpoint
  (train.py:1284 `emb0 = np.empty((0,0))`; training.py:1326 gate on
  emb0.size). Telemetry not_reached/unavailable is the DESIGNED initial-fold
  state, not a bug.
- [x] B8 Triplet lane dead-by-construction: FIXED (2026-09-29, commit
      3695c90, pushed). KEPT OPTIONAL (owner ruling — do not remove the
      lane). Root cause: hard_train_all mined from emb0 ONCE before the
      fold loop, emb0 empty pre-checkpoint -> hard_train always empty ->
      build_triplets returned [] -> RuntimeError "no triples built for
      fold". FIX: when build_triplets returns no examples, train_one_config
      appends a status='skipped' fold row with a clear reason and continues,
      mirroring the contrastive/MNRL skip paths. Lane stays functional when
      ANN mining provides hard negatives; main path records the skip; HPO
      still fails loudly via require_no_failed_folds (no calibration
      evidence to tune on). Blast radius verified via graphify (rebuilt at
      HEAD): change localized to the fold-loop dataset-build branch of
      train_one_config; no other loss lane touched.
- B9 HPO lane items: NOT APPLICABLE — HPO is not used (no hpo runs).
- B10 dead-code list: partially resolved already (hard_negative_swap_frac,
  swap_agreed, _optuna_mlflow_cb removed). Remaining (NER island, colab dead
  downloader cluster, hpo_persistence/fencing PG machinery, STOPWORDS,
  brand_tokens, duplicated regexes) are LOW-priority dead code; triage
  deferred unless it touches the active training/data path.
- C/D/E sections: HPO-adjacent + LOW dead-code items not in the active
  path are deferred. HIGH E item (dpi=150 -> plot_dpi()) verified below in
  pytorch/colab optimization pass.


Phases: (1) scour entire project (agents + main thread) for
redundancies / dead code / unexpected behavior; (2) root causes WRITTEN
here, clustered; (3) fixes in worktrees per cluster (no collisions);
(4) consolidate + verify end-to-end against DATA_PATH.md contract, then
launch training.
Audit lenses: SSOT violations, config-owned paths, dead code, silent
drops, phantom accounting, reuse-don't-duplicate.

Main-thread sweep results (2026-09-28):
- Dead knob: hard_negative_swap_frac read at train.py:722, zero
  consumers; key still in config + schemas — remove. FIXED (agent 1,
  commit 54f889f): ROOT CAUSE — the agreed-surface swap lane was deleted
  (positive-side swap_agreed, TODO "swap_agreed DELETED") but its
  hard-negative-side sibling survived: read bound to a local with zero
  consumers (train.py:722), key in config/training.yaml:28, field in
  MaskingSpec (schemas.py:924) + MaskingProfileSpec slot (schemas.py:989).
  SSOT defect: a shipped knob implying deleted behavior. Blast radius:
  config -> masking_cfg() -> dead local only; old bundle sidecars
  (data/prepared/full/worker_*.pkl.gz.json masking_config) record it as a
  historical build record, left untouched (prepared_bundle.py:287 drift
  check only warns; bundle rebuild already pending). Removed: YAML key,
  MaskingSpec field + comment, MaskingProfileSpec override, train.py read
  + comment. Guard: tests/test_masking_cfg_dead_knobs.py (dead-local scan
  of every mask_cfg[...] binding + knob pin).
- [x] Consistency flags (volume_inconsistency / ambiguous_volume) were
      IN-MEMORY ONLY — FIXED (2026-09-29, commit 9c10c71, pushed): flag
      census persisted in the data_prep manifest (row_accounting.flags_census,
      gtin level). Verified end-to-end on the real 71,623-row export with
      sandboxed outputs: 13,250 canonicals -> {ambiguous_volume: 3,
      description_conflict:carbonation: 101, description_conflict:pulp: 4,
      description_conflict:sweetener: 73, no_added_sugar_with_cane_sugar: 30,
      unsweetened_with_declared_sweetener: 3, volume_inconsistency: 42};
      closure 71,623 == 13,250 + 45,260 holds. Counts now verifiable from
      results/manifests/data_prep.json and drift-tracked.
- second04: 948 candidates -> 261 bundle hp_pairs (strict volume
  equality; lever = the gate's 5% tolerance).

## Duck hunt protocol (rate-limited: ONE agent at a time)
Relay model, orchestrated by the main thread:
1. Agent finds unexpected behavior -> investigates ROOT CAUSE + BLAST
   RADIUS (graphify graph: `graphify affected <node>`, god nodes).
2. In a WORKTREE (`git worktree add ../ER-fix-<cluster>`): replicate the
   error as a failing test, apply the minimal fix, run the suite.
3. Merge back to training-sid-hybrid, push (protocol: every finding ->
   TODO + push).
4. Agent ends; a NEW agent runs and continues from the TODO queue.
One agent at a time (provider rate limits); main thread orchestrates,
triages, and integrates.

Queue state (updated at stand-down 2026-09-28, agent 1 session end):
- DONE+PUSHED item B6: _optuna_mlflow_cb deleted (commit 06920ce, guard
  tests/test_optuna_callback_wiring.py). Root cause + blast radius above.
- DONE+PUSHED dead knob hard_negative_swap_frac: removed from
  config/training.yaml, MaskingSpec + MaskingProfileSpec, train.py read
  (commit 54f889f, guard tests/test_masking_cfg_dead_knobs.py).
- DONE item 3 stale comment/fixture: train.py extent-halves comment
  rewritten; swap_agreed fixture -> "targeted" (commit 051c906, guard
  tests/test_stale_mode_references.py).
- DONE item F1: <source>+aug normalized to base population (commit
  65b270a, docs 23f9115; tests/test_aug_source_population_registration.py,
  3 tests).
- Item A3 (gate slots labeled ann_finetuned without realized
  replacement): DONE+PUSHED (merge 7a6af8b on training-sid-hybrid). See
  triage checkbox above.
- Item A4 _write_datapoint_usage: DONE — CONFIRMED already fixed vs HEAD,
  regression pins pre-exist (see triage checkbox above).
- [x] Item 7 (MNRL subset monitoring + twin warmup): DONE+PUSHED (commit
      78fa96d, EXP-03). MnrlMonitoringSpec + TwinLossWarmupSpec in the
      training config block (config training.mnrl_monitoring +
      training.twin_loss_warmup), _build_mnrl_triple_populations per-triple
      tags, _tracking_mnrl_loss override attributing per-row loss to its
      population and ramping twin weight across warmup_epochs; pair_id
      threaded via PairIdDataCollator. Both DISABLED by default — plain run
      is bit-identical. Guard tests: tests/test_mnrl_monitoring_warmup.py
      (9). Suite at that commit: 548 passed + 2 skipped.
- Suite state (2026-09-29): 561 passed + 2 skipped. Growth since the
  532 baseline: +9 MNRL monitoring/warmup guards (78fa96d), +5 diet MNRL
  accounting guards (ead6966), +5 record-linkage guards (uncommitted at
  this writing). No pre-existing failures remain (the two flagged above
  were fixed by commits 4e8db93 golden refresh + 5c012d1 title-wins test
  update).
- Agent 3 (data-prep track) still running: aliases, flag census, smoke
  sampler, coverage JSON.
- Main thread: FINDINGS B7-B10 + C/D/E triage continues.


Source: sample-tracking audit (experiments.md, chain-of-custody table).
Reference for the artifact contract: the last training run's
publication_manifest (DVC 20260913T123559565190Z) — that run published
datapoint_usage_fold0.csv, datapoint_type_coverage_fold0.csv,
pair_backprop_fold0.csv, loss_backprop_fold0.csv,
masking_per_epoch_fold0.csv, mask_hard_negative_visibility.csv,
train_rows_fold0.csv. The upcoming MNRL lane produces NONE of these
(coverage writer is contrastive-only) — that is the telemetry regression
to close before training.
- [x] Register/normalize the `<source>+aug` tag (6,776 rows/fold,
      train.py:1122 f-string, invisible to the static producer scan) in
      _training_pair_populations — FIXED (agent 1, commit 65b270a). PATH
      CHOSEN: normalize (not register). ROOT CAUSE: train.py mints static
      masked/swap copies of hard negatives with an "<source>+aug"
      provenance label via f-string concat — invisible to the static
      producer scan, absent from DATAPOINT_POPULATION_SPEC; both coverage
      checks raised on ANY contrastive fold with masked hard negatives:
      _training_pair_populations passed compound tags through verbatim
      (training.py:2614-2619) and _negative_source_accounting enumerated
      the raw source array (N3 breach). Blast radius: 6,776 rows/fold,
      every contrastive fold on current bundles (hard crash); zero other
      consumers of +aug (diet gate counts from the audit, not tags).
      FIX: _base_population_tag() helper (training.py:215-226) strips
      "+aug" -> base in _training_pair_populations and folds the copies
      into their base source in the census (N1/N2/N3 identities hold);
      static minting mode rides the usage row's lineage block
      (_build_pair_lineage adds target_mode from
      hard_negative_mask_audit -> flows into every usage row). NOTE vs
      plan: mode carried as lineage column target_mode, NOT overwriting
      the presentation-time `augmentation` column (that column is part of
      the presentation key and reconciliation identities; overwriting
      would corrupt them). Registry stays honest — no combinatorial
      <base>+aug tag family. Tests: tests/
      test_aug_source_population_registration.py (fold with masked hard
      negatives writes coverage rows without raising; census folds +aug
      into base; strip guard).
- [x] Fix diet_manifest.py's dynamic-mask projection: it is conceptually
      wrong, not just loss-aware — dynamic masking is IN-PLACE
      replacement (training.py:3096 rewrite, "no static negative copies
      are added" :3371), so it adds NO views for ANY loss; the +30%
      projection inflates the denominator ~1.3x, biasing neg_aug_frac
      LOW (stricter, can flip PASS->FAIL at 0.30) and pos_neg_ratio LOW
      (more lenient, can flip FAIL->PASS at 1.50). Remove the projection
      (and the docstring claim), then re-run the gate on fresh bundles.
      FIXED (commit 5c012d1+): `project_dynamic_mask_views` removed;
      projected_neg_views = bundle's own view counts; easy-projection
      remains an upper bound only. Fresh full bundle rebuilt at
      frac=0.80 -> DIET PASS (pos_views 42,477 / neg_views 28,852 =
      1.4722 < 1.50; neg_aug_frac 0.3060 >= 0.30).
- [ ] Align swap-copy accounting for MNRL: 2,709 swap hard-negative
      copies are diet-counted as augmented views but 0% MNRL-train
      (by design, _build_mnrl_training_triples omits them). Either
      exclude never-trained populations from diet neg_aug_views or gate
      swap minting on loss. Owner decision.
- [ ] Masked-positive minting: 76% (16,345/21,373) never train — no
      source negative in the fold. Condition minting on fold-negative
      availability OR report real survival in the diet; also ~20k dead
      payload rows per bundle.
- [ ] Port per-population coverage to the MNRL fold path so the next run
      publishes the reference contract's tracking CSVs again (usage +
      type coverage per fold, populations from triples: twin/masked/base
      — classifier verified in the audit).
- [x] MNRL subset-loss telemetry (per-population, per-epoch) — being
      built by background agent in train_one_config, which train_prepared
      shares, so production reaches it automatically. New contract file:
      mnrl_subset_loss_by_epoch_fold{i}.csv.

## Recalled-agent tasks (main thread owns these)
Agent slots capped at 2 (MNRL monitoring + alias normalization kept);
recalled mid-work, partial handoffs received — remainder done by main
thread.

- [ ] Coverage analysis remainder (agent 2 handoff, NOT yet written to
      results/coverage_expansion_analysis.json): (a) gate band x decision
      x canonical-agreement table incl. the 323 proceed rows below 0.50
      (+22.8% pos headroom); (b) max_pos_per_group surplus (blocking.py
      identity lane, needs group inventory on the 56,529 split — note:
      caps build_pairs identity population, NOT the gate lane);
      (c) hp_pairs yield: second04_pairs_positive.csv row count (52,283
      bytes on disk), strict volume equality vs the gate's 5% tolerance
      (volume_verified.py:67-71), unknown-volume rows; (d) confirm
      bundle hp_pairs == 261. KNOWN from handoff: 1,414 = ENTIRE
      gate-verified positive population at sim>=0.50 (not a sampling
      cap; balanced sample already contains all); balanced-pool negative
      family = pack_blocker only (1,414/1,414; all 7,722 hard_no>=0.80
      rows are Pack blocker); cross_brand funnel 30,372 clear the floor
      vs target 6,000 (~4.9x headroom); labeled_pairs.csv = 9,136 rows
      exact; PINNED_GATE_FALLBACK_PAIRS pins gate universe
      (labeled_pairs.py:85-95, core/common.py:608) — changing gate
      universe requires pin updates.
- [ ] Verification of 4a39fdc remainder (agent handoff): claims 1-7
      verified (5 PASS, 1 partial, 1 FAIL); outstanding: claim 8
      (flip_validity_audit.py self-review — numeric surfaces, global
      denominator), 49->30 ambiguous_volume arithmetic, sweep for other
      sites assuming frac 1.00/1.622.

## Verification findings from the recall (fixes below, actions in Tracking fixes)
- HIGH smoke_128 "stratified" claim FAILS: committed CSV is skewed
  (amazon 23.5%->55.5%, Netherlands/Italy ABSENT, Turkey 18.75%); no
  sampler code committed (not reproducible); config comment still says
  "Deterministic 128-row prefix". Must implement + commit a real
  stratified sampler and regenerate in the rebuild.
- HIGH diet projection overcounts phantom views even under contrastive:
  dynamic masking is IN-PLACE replacement (training.py:3096, "no static
  negative copies are added"), not added views — the +30% projection is
  conceptually wrong, not just loss-aware (sharpens TODO F2).
- MED dead knob: train.py:722 reads hard_negative_swap_frac, never uses
  it (consumer deleted); key still in config:28 + schemas — remove.
- MED ambiguous_volume multi-pack exclusion is a silent drop (no
  counter/manifest; 49->30 unverified) — add a census counter.
- LOW stale comment train.py:1057-1059 ("both swap modes"); stale
  fixture "swap_agreed" target_mode in tests/test_mnrl_pair_selection.py:22.
  FIXED (agent 1, commit 051c906): comment rewritten — only swap_values
  is excluded from the extent halves (swap_agreed lane deleted); fixture
  renamed to "targeted" (registered mode; NOT swap_values — that mode is
  special-cased in _build_mnrl_training_triples training.py:676-686 as
  omit-swap-copies, which would have inverted the test's assertion).
  Guard: tests/test_stale_mode_references.py (registered-mode set derived
  from masking.py producers; deleted-mode pin; stale-claim pin).


- [x] Full-data twin P@R95 curve per checkpoint — harness READY
      (build_field_slice.py + minimal_flip_slice.py on the live bundle);
      curve itself requires the first trained checkpoint (P1 blocks on
      bundle rebuild). No embedding mixup (dynamic; label undefined).
- [x] Train-time per-subset MNRL monitoring + twin-loss warmup — DONE
      (commit 78fa96d, disabled by default; see Queue state item 7).

## Data gaps measured (2026-09-28, flip_validity_audit.py)
- [x] Per-field flip validity measured (results/flip_validity_audit.json,
      scripts/flip_validity_audit.py; stale bundle but minting logic
      unchanged). Twins: flavor 717/717 (100%) prose-contradicted — the
      decorative-flavor -> label-noise hypothesis CONFIRMED; volume 0.55
      contradicted / 0.42 opaque; package_type 0.93 opaque (cleanest);
      sweetener 0.54/0.36. Policy decision pending: exclude flavor from
      twin flips (config allowlist) or accept (structured channel still
      flips; veto alignment argument).
- [x] Swap/prose contradiction measured: flavor/carbonation swaps leave
      the old word in prose 89-93% of the time (positive and negative
      lanes); volume 54-58%; package_type ~10%. Same report.
- [x] Transplant concentration monitored: global volume share 0.500
      (soft field cap 0.35 — fallback path working as designed), top
      (field,value) 250ml 0.031 / bottle 0.031 / still 0.031 = at the
      0.03 hard value cap (rounding-compliant). Per-lane tables in the
      JSON; re-run per bundle.
- [x] Swap-metric correctness verified per field (delegated audit):
      volume -> volumes_compatible (0.05 rel / 5 ml abs), flavor ->
      overlap == 0.0, sweetener -> sweetener_conflict (sugar vs
      no_sugar/diet), rest -> categorical disjoint sets. 0 dot-tokens in
      108,046 payload rows; every field yields eligible swaps; bundle
      twins re-validate under the conflict rules at load. Latent seams
      only (dot-form sweetener, noise-token flavor, mixed-prefix
      agreement) — unreachable with the single emitter; no code change.

## Review fixes pushed
- Commit `033c4ab` contains the reviewed source fixes.

## Follow-up: training diet ratio
- [x] Regenerated worker_1 and worker_2 baseline bundles from the pushed
      code and 56,529-row training split. Both contain 2,044 semantically
      checked twins and load-time lineage is rebuilt.
- [x] Resolve the static positive/negative ratio before training: diet gate
      reports 46,783 positives / 28,844 negatives = 1.6219, above the
      configured 1.50 ceiling. Root cause: `masking.frac=1.00` doubles
      positive count via random masking copies while negative augmentation
      is lighter (30%). Fixed: reduced `masking.frac` 1.00->0.80 and
      `swap_agreed_frac` 0.50->0.00 (swap_agreed is structurally empty —
      same values always produce same token order, 0 swappable fields
      measured on 8,893 positive pairs). New ratio at frac=0.80:
      ~42,508/28,844 = 1.474 < 1.50. Bundle must be rebuilt for gate to pass.
- [x] Volume unit anomaly blast radius measured: 275 records (0.38%)
      carry volume_inconsistency (232) or ambiguous_volume (49) flags.
      Only 2/1,414 positive pairs (0.14%) and 74/7,722 hard-negative
      pairs (0.96%) involve flagged GTINs. Downstream impact is negligible.

## Open gaps (2026-09-28 audit — remaining)
Parser / extraction:
- [ ] MPN parsing: NOT APPLICABLE to this dataset (full 61,529-row scan:
      only 29 rows mention any MPN-style key, all false positives —
      beverages carry no manufacturer part numbers). Owner ruling
      2026-09-29: keep the triplet-style optionality mindset — the real
      gap (barcode-less identity) moved to the DEAD LAST section below as
      the record-linkage lane.
Data / augmentation:
- [ ] Positive/type coverage: current balanced sample has 1,414 positives
      at sim>=0.50 (530 at >=0.80) and 1,414 pack-blocker negatives only.
      Current 5k-holdout training bundles have 261 hard-positive pairs;
      expand reviewed positive coverage and negative-family diversity.
      IN PROGRESS (background): per-lever headroom measured on the
      training split, report to results/coverage_expansion_analysis.json.
      Constraint: pos/neg ratio 1.474 vs 1.50 ceiling — positive
      expansion must be paired with negative expansion.
- [x] Low-cardinality fields concentrate transplants structurally; monitor
      field/value distributions under the existing concentration caps.
      Monitor implemented: scripts/flip_validity_audit.py concentration
      section (global + per-lane). Volume dominates by soft-cap design.
- [x] `swap_agreed` lane yields 0 structurally — DELETED (frac 0.50->0.00).
      Root cause: text composition is deterministic; same values always
      produce same token order. 0 swappable fields measured on 8,893
      positive pairs and 3,638 negative pairs. Code removed.
- [x] Swaps alter structured tokens while prose keeps the old word;
      measure this contradiction and review field-specific rewriting.
      Measured 2026-09-28 (scripts/flip_validity_audit.py): flavor
      89-93%, carbonation 89%, volume 54-58%, package_type ~10%. Numeric
      volume/pack vectors follow the swapped tokens. Rewriting review
      pending the flavor-twin policy decision.
- [ ] Counterfactual validity assumed: decorative flavor words -> label noise;
      no per-field flip-validity measurement. MEASURED 2026-09-28:
      flavor twins 100% prose-contradicted (717/717) — hypothesis
      confirmed; volume 0.55, package_type 0.93 opaque. Remaining:
      owner policy decision (flip allowlist vs prose rewrite vs accept);
      bundle rebuild at the end inherits the decision.
- [ ] Swap/twin fracs and caps hand-picked (0.20/0.10, 0.35/0.03); HPO never
      swept them.
Training / eval:
- [ ] No checkpoint trained with twins — hypothesis unvalidated (zero-shot
      P@R95 0.50, margin 0.009 is the baseline to beat).
- [ ] No train-time twin monitoring (MNRL loss has no per-subset hooks;
      offline slice curve per checkpoint not yet drawn). IN PROGRESS
      (background): _tracking_mnrl_loss per-population hooks + per-epoch
      subset-loss CSV + config training.mnrl_monitoring.
- [ ] No twin loss warmup (watch-item: spikes epochs 1-2; guards in place).
      IN PROGRESS (background): training.twin_loss_warmup knob
      (warmup_epochs: 2, twin_weight: 0.25), weighted inside the MNRL
      wrapper, bit-identical when disabled.
- [ ] Pooling vs single-token flips unvalidated (markers deferred, not dead).
- [ ] Contrastive/triplet paths consume new audits generically — untested.
Process / repo:
- [ ] Metrics unversioned (`results/` gitignored — reports live locally only).
- [x] Smoke unrepresentative (`--sample 1000` = first rows, not stratified;
      e.g. entity coverage 11.8% vs 52.1%). Fixed: regenerated smoke_128
      with stratified sampling (retailer+country+category+attribute-signature).
- [x] Easy-negative replenishment samples with replacement — FIXED (replace=False).
- [x] Dynamic masking invisible to diet — FIXED: diet_manifest.py now projects dynamic mask views.
- [x] 41MB new CSVs in git while DVC sits disabled (bloat policy question).
      41MB is from prepared bundles (pkl.gz) in recent commits. DVC disabled.

## Verdict: 3 good, 7 bad

### GOOD (working as intended)
- Diet gate fix (frac 1.00→0.80, swap_agreed 0.50→0.00) — ratio now 1.474 < 1.50
- Volume anomaly blast radius — negligible impact (0.14% pos, 0.96% neg)
- Counterfactual validity — twins validated, semantic conflict verified
- MPN coverage — 52.1% acceptable, entity guard works
- Twin fracs — counterfactual_frac=0.10 reasonable
- Value swaps — working correctly, numeric vectors follow swapped tokens
- All tests pass (561 passed + 2 skipped, 2026-09-29)

### BAD (needs fixing)
- Smoke not stratified — FIXED: regenerated smoke_128 with stratified sampling
- ambiguous_volume flag — FIXED: multi-packs excluded
- Bundle rebuild — existing bundles built at frac=1.00, new config needs rebuild
- Low-cardinality field concentration — transplants concentrate structurally
- Counterfactual validity assumption — decorative flavor words may be label noise
- Easy-negative replenishment — FIXED (replace=False)
- Dynamic masking invisible to diet — FIXED: diet_manifest.py projects dynamic mask views
- Metrics unversioned — results/ gitignored
- 41MB CSVs in git — DVC disabled, bloat policy unclear
- **Who builds bundles**: `_build_local_training_bundles` in `src/cli/colab.py`
  runs `training.train` as a subprocess on the Colab VM.
- **Twin fracs**: `counterfactual_frac: 0.10` — 10% of positives minted as twin negatives.
- **Ambiguous volume**: 49 records flagged, but most are legitimate multi-packs
  (12L, 20L, 33L total volumes), NOT ambiguous. The flag is overly aggressive
  for multi-pack titles like "10 x 0, 20l" -> 20000ml. Only 1 of 5 flagged
  GTINs is truly ambiguous (barcode-less, no volume evidence).
- **MPN coverage**: 52.1% GTIN-based cluster coverage. No MPN parsing exists.
  Acceptable for training (entity guard handles unmapped rows), but barcode-less
  duplicates need record linkage.

## Questions to resolve before training
1. **bundle rebuild required**: diet gate passes on paper (frac=0.80) but existing
    bundles were built at frac=1.00. Must rebuild worker_1/worker_2 bundles
    before training. Who triggers the rebuild?
2. **volume_inconsistency reliability**: 232 records have
    `volume_inconsistency` (title/attribute disagree >=10x). FIXED: now defaults
    to title volume on 10x+ disagreements instead of attribute. Is the title-side
    volume reliable enough?

## DEAD LAST — recent additions (2026-09-29; do NOT start until P1/P2 done)

Owner ruling: the graph/linkage work below is parked. P1 finalize items and
P2 open gaps come first; training launch outranks everything here.

### Standing rules for all new code (owner, 2026-09-29)
- SSOT config loading for EVERYTHING, paths included: no bare "results/"
  literals or CWD-relative defaults — resolve through core.common (RESULTS
  root, files./layouts. bindings; honors EUROMONITOR_RESULTS_DIR on Colab
  workers). Applied: build_barcode_less_linkage.py + flip_validity_audit.py
  defaults now RESULTS-resolved (behavior identical locally).
- graphify blast radius before/after every change (rebuild the graph when
  HEAD moved: it was stale at 4a39fdc once already).
- Every new code path is EXECUTED (synthetic -> smoke sample -> real-data
  sandboxed run) before it is called done. Evidence recorded here.
- Limit test writing: guard tests only where behavior could silently
  regress; no coverage theater.

### 1. Barcode-less record linkage — BUILT, KNOWN DEFECT OPEN
- Reusable rule: src/core/record_linkage.py (link_barcode_less) + thin CLI
  scripts/build_barcode_less_linkage.py; guard tests
  tests/test_record_linkage.py (5). Suite green (561 passed + 2 skipped).
- Verified executed: synthetic cross-retailer/pack-variant cases,
  smoke_128 (99 barcode-less rows, all singles), real-data sandboxed run
  (census: 38,159 eligible rows; 3,511 multirow clusters; 12,613 rows in
  them; 3,089 exact + 11,481 fuzzy links; 609,590 pair checks; ~4s).
- [ ] DEFECT (root cause NOT yet fixed): flavor merging survives the
      pack-multiplicity strip. Verified on real data 2026-09-29: bl-000583
      Clearly Canadian 71 rows (peach/raspberry/cherry merged), bl-012115
      Obsesso 68 rows (mocha/latte/caramel/black merged), bl-000909
      International Delight 66 rows. Jaccard >= 0.7 on short brand-blocked
      titles links distinct flavors sharing brand/size tokens; transitive
      union-find then chains them. NOT FIXED by strip_pack_multiplicity
      (flavor words are not pack tokens). Fix direction: average-linkage
      clustering with a margin (replace raw union-find transitive closure)
      + lower fuzzy ceiling / discriminating-token check. Numbers in
      census are provisional until this lands.
- [ ] Retailer alias normalization (defect, Tier-1): "Voila" vs "Voila"
      (accent) normalize differently -> same-retailer duplicates can be
      treated as cross-source evidence. Accent-fold + alias table.
- [ ] Hold-out edge validation (cheap, do first when resumed): hide known
      GTIN edges, measure linkage precision/recall against that ground
      truth — honest estimate of barcode-less linkage quality.

### 2. Graph-construction improvements (Tier 1-3, parked)
- Tier 1 (defects): retailer alias normalization (above); GTIN-14
  indicator-digit folding (case<->unit packaging links, zero fuzzy risk);
  average-linkage clustering instead of raw union-find.
- Tier 2 (recall on ~25.5k singletons): IDF-weighted token similarity
  (NgramIDF exists in pipeline.py); unit canonicalization before matching
  (unit_canonicalization.py exists); embedding-proposed candidates via the
  existing ANN band (bands.eval_mining 0.35-0.90) verified by linkage rules.
- Tier 3 (GNN-ready graph): typed+weighted edges (gtin_identity 1.0,
  link_exact 0.9, link_fuzzy 0.7, gate_hard_no anti-edge) with per-edge
  provenance; description_evidence / breadcrumb_evidence as node features
  or weak edges (collected in canonical_records.csv, excluded from frozen
  canonical text by design); image evidence last (image_url exists).

### 3. Two-stage graph matching architecture (directive, DESIGN ONLY — nothing built)
- Stage 1 retrieval: Bi-Encoder GNN over graph neighborhoods (GraphSAGE/GAT,
  <=3 layers — component cap 15 is the over-smoothing guard) initialized
  from existing sentence-transformer embeddings + structured features;
  trained with a LINK-PREDICTION objective (positives = identity edges;
  negatives = gate-verified hard-no anti-edges + in-batch MNRL pressure);
  cold-start via GraphSAGE h_v^0 concat (isolated nodes fall back to text
  embedding). Single encode pass + ANN = fast candidate retrieval.
- Stage 2 verification: Graph Alignment Network over top-K candidate
  sub-graph pairs — cross-graph node matching scores + edge-consistency
  scores + existing structured vetoes -> calibrated score; thresholds on
  the component-safe dev split, test read once (folds.py discipline).
  Existing cross-encoder rerank lane is the degenerate single-node case.
- Key measurement queued (owner approved): neighborhood-context recall
  vs plain text bi-encoder — the GNN's win comes from the 12.6k linked
  rows, ~25.5k singletons get nothing from edges; measure before building.
- Implementation doctrine: optional lane, disabled by default, bit-identical
  when off (mirror 78fa96d MnrlMonitoringSpec pattern); config SSOT — exact
  YAML block + pydantic Spec (extra="forbid") proposed for owner review
  BEFORE schemas.py/config change; no edits to the live training path.
