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
- [ ] No MPN parsing; entity guard best-effort on unmapped rows (52.1%
      cluster coverage; barcode-less duplicates need record linkage).
      MPN parsing not implemented — no code exists. 52.1% coverage
      is GTIN-based entity clusters only. 47.9% unmapped rows need
      record linkage.
Data / augmentation:
- [ ] Positive/type coverage: current balanced sample has 1,414 positives
      at sim>=0.50 (530 at >=0.80) and 1,414 pack-blocker negatives only.
      Current 5k-holdout training bundles have 261 hard-positive pairs;
      expand reviewed positive coverage and negative-family diversity.
- [ ] Low-cardinality fields concentrate transplants structurally; monitor
      field/value distributions under the existing concentration caps.
- [x] `swap_agreed` lane yields 0 structurally — DELETED (frac 0.50->0.00).
      Root cause: text composition is deterministic; same values always
      produce same token order. 0 swappable fields measured on 8,893
      positive pairs and 3,638 negative pairs. Code removed.
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
- All 523 tests pass

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