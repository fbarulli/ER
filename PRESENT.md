# SET UP
- GPU : Colab + DVC
## Initial diagnosis (train/test loss gap) `SOLVED`
- Started from train loss 0.5 vs test loss 1.7, large score overlap between label 1/0 in holdout.
- Established: overlap ≠ overfitting by default; the diagnostic question was whether train also showed overlap (ceiling) or only holdout did (generalization gap).

## Root causes found, in order of discovery `SOLVED`

**1. Contamination in mined negatives (~5.6%)** — mining pipeline didn't exclude same-canonical pairs as negatives; ~521-1,345 "negative" pairs were actually true matches. Fixed by adding `canonical_map[gtin1] == canonical_map[gtin2]` exclusion. Partial recovery: AUC 0.558 → 0.644.

**2. Augmentation asymmetry** — positives got masking augmentation every epoch (`mask_n` matched positive count exactly); negatives got zero augmentation, training on a static, repeatable, memorizable 6,051-9,323-pair pool. Confirmed via `n_train_neg == n_train_neg_total` (no larger pool being subsampled).

**3. Numeric tokens stripped from model payload** — `clean_sku_text()`/`canonical_model_text()`/`strip_schema_words()` removed volume/pack numeric tokens (e.g. "500ml", "1L") before the model ever saw them. Audit found **zero usable signal** across all 908 volume and 193 pack hard-negative examples. This was the single biggest fix: volume error rate 22.6%→1.65%, pack 25.9%→2.6%, holdout overlap coefficient 0.658→0.070, hard-negative AUC 0.644→0.991.

**4. Missing-value-as-conflict labeling bug** — 169/176 `volume/1` errors traced to SKU-side volume parsing as missing/zero and being wrongly treated as a conflict rather than "unknown." Never fully confirmed fixed.

**5. Embedding-space collapse** — the model's raw cosine scores compressed upward globally after fine-tuning. Confirmed via deliberately-unrelated-pairs test: zero-shot median cosine 0.34 (0/100 pairs ≥0.70) vs fine-tuned median 0.83 (100/100 pairs ≥0.70). Root cause hypothesis: 100% negative masking on every presentation, every epoch. **Still unresolved** — multiple checkpoints since have failed the collapse guardrail (crossing rates 0.15, 0.06, 0.28, 0.03, one borderline at 0.02).

## Evidence-based theories tested and ruled out
- Tokenizer breaking up brand names — ruled out (brand+category already separated cleanly per field-ablation; problem was numeric-token deletion, not tokenization quality).
- "Transformers can't handle numbers" — ruled out as a myth for this task; the task is text discrimination, not arithmetic reasoning.
- Loss function (OnlineContrastiveLoss) needing replacement — never confirmed necessary; coverage stats (69% hard-selected, 62% margin-active) were adequate once other fixes landed.
- Chaining/wrong-checkpoint causing a bad run — investigated and ruled out twice (early stopping + restore-best-checkpoint mechanics confirmed working correctly via trainer_state.json).
- "Report is stale/cached" — investigated at length via checkpoint hashes; ultimately explained by ANN-mined pairs feeding eval only, not training loss (`_train_neg_source` = gate + attribute-conflict only).

## Rand Index / calibration pipeline findings
- Design (fold-safe, canonical-disjoint threshold calibration, GTIN-stratified reconciliation, direct SKU-to-canonical assignment instead of connected-components to avoid chaining) was sound and implemented largely as specified.
- **Critical flaw found twice**: `both_equal` GTIN stratum is locked/bypasses the cosine threshold entirely — a "perfect" Rand Index of 1.0 there tests nothing about the model. All calibration positive-pair sources were found to require GTIN presence by construction, making `different`/`one_missing`/`both_missing` strata initially untestable.
- `both_missing` confirmed **structurally impossible** — canonical records always have GTIN (0/13,250 missing), so every missing-GTIN SKU is `one_missing`, not `both_missing`.
- `labeled_pairs.csv` (gate-derived, 0.80-threshold-filtered) was wired into calibration for the `different` stratum — yielded 3,118 usable source GTINs, but this positive source is gate-derived, not independent ground truth.
- **Degenerate Rand Index caught twice**: Rand=1.0 can occur with precision/recall = 0.0 (all-unmatched) — a real methodological trap. On the collapsed checkpoint, `different` also failed the opposite way: 92/94 matched, still 0% precision — evidence of collapse-driven false positives, not threshold miscalibration.

## Live submission incident
- Manual 20-example `one_missing` spot-check (not random, hand-picked "likely recoverable" cases) led to lowering threshold 0.8→0.60/0.62 and adding a brand-conflict veto (confirmed fixing a live Dia→Premier false positive, 379 rejections).
- This was applied more broadly than validated; actual submission score dropped from 20→12, most likely from the small-sample threshold change combined with an unconfirmed-collapse checkpoint being used globally rather than stratified.
- Fix in progress: per-stratum threshold config now wired (`both_equal: 0.80, different: TBD, one_missing: 0.60, both_missing: 0.80`), brand veto retained.

## Currently open / blocking

1. `different`-stratum threshold sweep needs to be redone once a collapse-healthy checkpoint exists — current sweep was contaminated by a collapsed checkpoint.



