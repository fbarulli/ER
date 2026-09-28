# experiments.md — A/B registry (2026-09-28, branch training-sid-hybrid)

Every change to minting, normalization, or loss goes through an entry here:
hypothesis, instrument, decision rule, status. Instruments produce the
numbers; the decision rule says what number means "ship it". No training
has validated any twin yet — everything below starts from the zero-shot
baseline.

## Shared baseline (zero-shot minilm_l6, worker_1 bundle)

Source: `results/minimal_flip_slice_zeroshot.json`
(script: `scripts/minimal_flip_slice.py --bundle data/prepared/full/worker_1_baseline.pkl.gz --model minilm_l6`).

- twin slice: n=4,088 (2,044 twins + sources), P@R95 **0.503**,
  margin mean **0.0074**, median 0.0051, frac_positive 0.854
- per-field twin margins (the field-bias watch):
  flavor 0.0113 (89.3% ranked correctly), package_type 0.0081 (81.2%),
  volume 0.0052 (85.5%), pack 0.0036 (90.2%), sweetener 0.0017 (62.4%)
- subset mean cos sim: twin_0 **0.8117** vs pos_1 0.8234 (twins sit 0.012
  below positives — the razor the training must widen) vs gate_0 0.7525,
  cross_brand_0 0.6364
- cross-brand slice P@R95 0.756 (n=1,000)
- donor uniformity: max donor-row share 0.001, top-10 rows 0.009 —
  no donor memorization risk; within-field value shares dominated by
  carbonation `still` (0.486 of carbonation transplants, 3 unique values)

Shared invariants (per-bundle flip audit,
`scripts/flip_validity_audit.py` -> `results/flip_validity_audit.json`):
flavor twins 100% prose-contradicted (717/717; 78% name the old flavor
2x+ in prose, 37% multi-flavor anchors), volume 0.55, package_type 0.93
opaque (cleanest), sweetener 0.54/0.36. Global transplant concentration:
volume 0.500 of picks (soft field cap fallback), top (field,value) all
<= 0.031 vs 0.03 hard cap — cap-compliant.

Decision rule for every training A/B: twin P@R95 must hold >= 0.500
(invariance floor), twin margin mean must lift off 0.0074, and no
regression on the gate/cross-brand slices. Winner = higher twin P@R95 at
equal-or-better margin; ties broken by margin mean.

---

## EXP-01: flavor-twin policy — block vs accept vs rewrite

- **Hypothesis.** H-block: 717 prose-contradicted flavor twins are label
  noise and drag the twin margin down. H-accept: the structured token is
  load-bearing (zero-shot: flavor = the BEST-separated field, margin
  0.0113, 89.3% ranked correctly) and training lifts them like the rest.
- **Evidence so far.** Zero-shot margins refute "unlearnable": the model
  already separates flavor twins better than any other field despite the
  prose contradiction. Sweetener, not flavor, is the weak slice (margin
  0.0017, 62.4%). The label-noise worry is not zero-shot measurable —
  it is a training-time quantity: watch the flavor twin-margin curve.
- **Arms.** (a) block flavor flips (config allowlist), (b) accept +
  monitor (needs EXP-03's per-subset hooks), (c) prose rewriting
  (minting-semantics change; own A/B, defer).
- **Instrument.** `minimal_flip_slice.py` per-checkpoint, flavor slice;
  EXP-03's per-subset MNRL loss (flavor population) per epoch.
- **Decision rule.** If flavor twin margin under (b) lifts at the same
  rate as volume/package_type through epoch 3 -> accept stands. If it
  stalls while other fields lift -> switch to (a) at the next rebuild.
- **Status:** decision deferred to first training curve; blocklist
  implementation is a one-line config change, held in reserve.

## EXP-02: low-hanging-fruit normalization A/B (aliases + plurals)

- **Hypothesis.** Normalizing variant flavor words (tamarindo->tamarind,
  fruits->fruit, apples->apple) expands the flavor signal available to
  every lane: more flavor tokens minted, more twin/swap eligibility,
  fewer false flavor conflicts in the veto. Cost: an over-eager alias
  could merge distinct flavors (pearl vs pear — flagged, exclude).
- **Candidate inventory (measured 2026-09-28, /tmp/opencode/alias_mine.py).**
  2,300/108,046 rows (2.1%) carry a lexicon-missed near-flavor word;
  71 distinct pairs (41 co-occur in one text). Top: fruits~fruit 1,025,
  fruity~fruit 497, apples~apple 138, oranges~orange 103, grapes~grape
  97, tonica~tonic 68, lemons~lemon 56, pears~pear 33, lemoni~lemon 8,
  strawberr~strawberry 8. `pearl~pear` (28) is a FALSE pair — exclude.
  FLAVOR_LEXICON currently 41 entries, FLAVOR_ALIASES 2 entries.
- **Arms.** (a) baseline bundle (current lexicon/aliases), (b) normalized
  bundle: plural/variant alias map applied at flavor extraction
  (`core/critical_attributes.py` FLAVOR_ALIASES/lexicon expansion),
  same seed, same fracs.
- **Instrument.** Same as every training A/B: per-checkpoint
  `build_field_slice.py` + `minimal_flip_slice.py`, flavor-slice margins
  + overall twin P@R95 curve; plus mint-yield diff (twin counts per field
  pre/post — the normalized bundle should mint MORE flavor twins).
- **Decision rule.** Ship normalization only if (b) twin P@R95 >= (a) at
  equal-or-better margin AND mint yield increases; if yield increases but
  curves are flat, keep (data coverage improved, no cost).
- **Status:** candidates mined; implementation staged behind
  config/SSOT (blocked until MNRL-monitoring agent lands its
  config/training.yaml + schemas.py edits).

## EXP-03: twin training + train-time monitoring (primary)

- **Hypothesis.** Twins train as explicit negatives of their source —
  margin mean lifts off 0.0074 and twin P@R95 holds >= 0.500 by epoch 3
  (TODO eval contract). Watch-item: loss spikes epochs 1-2 -> warmup
  (twin_weight 0.25, epochs 2) guards; per-subset hooks make the spike
  visible per population instead of pooled.
- **Arms.** (a) no warmup, (b) warmup on (config
  `training.twin_loss_warmup`). One arm for the first run (b) — the
  spike watch is the historical reason.
- **Instrument.** EXP-03 landing code: `_tracking_mnrl_loss` per-population
  hooks -> `mnrl_subset_loss_by_epoch_fold{i}.csv` +
  `loss_backprop_fold{i}.csv`; per-checkpoint slices.
- **Decision rule.** Warmup stays only if (b) reaches floor P@R95 with
  fewer/no epoch-1-2 margin regressions than the zero-shot-anchored
  expectation; if warmup caps the lift, rerun (a) and compare curves.
- **Status:** monitoring + warmup implementation in progress
  (background agent); bundle rebuild (frac=0.80) is the LAST step and
  inherits EXP-01/EXP-02 decisions.

## EXP-04: positive-coverage / negative-family expansion

- **Hypothesis.** Raising reviewed positive coverage (1,414 balanced
  sample; 261 hard-positive pairs in 5k bundles) and negative-family
  diversity improves twin P@R95 without breaking the diet ratio.
- **Constraint.** pos/neg view ratio 1.474 vs 1.50 ceiling — every
  positive-side lever must be paired with negative-side headroom.
- **Instrument.** Background agent measuring per-lever headroom on the
  training split -> `results/coverage_expansion_analysis.json`; lever =
  config key + measured yield + risk.
- **Decision rule.** Adopt a lever only with measured headroom AND a
  paired negative expansion keeping the diet gate green; verify via
  bundle manifest ratio + diet pass before rebuild.
- **Status:** analysis in progress (background agent).

## 2026-09-28 synthesis — conclusions from recovered history + today's measurements

History recovered (W&B runs deleted, API-unrecoverable; substance came
from DVC pull + HF artifacts + origin/submission):

1. The finetune trajectory is 0.611 -> 0.679 -> 0.967 AUC (0911 ablation
   -> 0913 masking_only -> full minilm run) and the jumps were driven by
   DATA fixes, not hyperparameters: 0911 had n_train_hp=0 (hard positives
   did not exist), 0913 ran masking_only on a 1000-row sample, the full
   run inherited the diet/gate/payload fixes. Validates the data-items-
   before-training sequencing.
2. NO finetuned checkpoint has ever had a twin slice computed — the twin
   margin is a quantity this repo has never observed on a trained model.
   Zero-shot (margin 0.0074, P@R95 0.503) is an anchor, not a verdict.
3. Previous-model bar (ANN lane, full minilm run): adj Rand 0.9993, pair
   recall 0.9987, over-merge 0%, under-merge 0.13% on the 3,000-SKU
   validation; GTIN gate alone decides 2,915/2,928 strata. The new
   checkpoint must hold this while adding the twin-margin instrument.
4. Flavor is NOT the weak twin slice despite 100% prose contradiction —
   zero-shot it is the BEST-separated field (margin 0.0113, 89.3% ranked
   correctly). SWEETENER is the weak slice (margin 0.0017, 62.4%,
   highest flip-support rate, n=109). EXP-01 leans accept+monitor; the
   first training curve gets watched at sweetener first, flavor second.
5. Minting hygiene is healthy: donor uniformity max row-share 0.001 (no
   memorization surface), transplant caps compliant (top values <=0.031
   vs 0.03 hard cap; volume dominance 0.500 is the soft-cap fallback by
   design), and the per-field conflict metrics are wired correctly for
   every attribute type (verified end-to-end; 3 latent seams all
   unreachable).
6. Normalization headroom is bounded and real: 2.1% of rows carry
   lexicon-missed flavor variants (71 pairs; tamarind absent entirely).
   EXP-02 proceeds behind the alias expansion; training-time A/B judges.
7. Open question the first monitored checkpoint answers: does training
   move per-field twin margins — expected direction is UP from 0.0074;
   the failure signatures to watch are (a) sweetener margin staying ~0,
   (b) twin P@R95 < 0.500 floor, (c) loss spikes epochs 1-2 (warmup
   armed), (d) gate/cross-brand regression.

## Sample-tracking audit (2026-09-28, done in main thread)

Question: is every minted sample registered, presented, trained, and
per-epoch visible after the diet change? Computed on the stale worker_1
bundle with the production functions (`_build_mnrl_training_triples`,
coverage-writer code paths).

Chain of custody (MNRL, n=32,126 triples):

| population | minted | trains | silent drop |
|---|---:|---:|---|
| gate | 13,558 | 13,558 | 0 |
| gate+aug masked | 4,067 | 4,067 | 0 |
| gate+aug swap_values | 2,709 | 0 | 100% (by design: no compatible positive) |
| cross_brand | 6,000 | 6,000 | 0 |
| targeted_attribute | 466 | 466 | 0 |
| counterfactual twins | 2,044 | 2,044 | 0 |
| masked positives | 25,410 | 5,991 | 76% — 16,345 "no source negative" |

Findings:

1. HIGH `gate+aug` (6,776 rows, train.py:1122 f-string tag) is NOT in
   DATAPOINT_POPULATION_SPEC and is invisible to the static producer scan.
   Live crash (UnregisteredDatapointPopulationError) if the CONTRASTIVE
   lane trains on these bundles; dormant under MNRL because the coverage
   writer is contrastive-guarded.
2. HIGH The production lane has no per-population sample telemetry:
   `_write_datapoint_usage` + coverage CSVs are contrastive-only
   (training.py:4634). Under loss=mnrl nothing tracks which populations
   trained, per epoch.
3. HIGH Diet-vs-trainer mismatches: (a) dynamic-mask projection adds
   +30% phantom views for MNRL (dynamic masking is contrastive-only,
   training.py:4004) — real MNRL neg_aug_frac = 8,820/28,844 = 0.306
   (passes the 0.30 floor); with the phantom projection 0.235 (would
   FAIL). (b) 2,709 swap copies are diet-counted as augmented negative
   views but never MNRL-train. (c) 76% of masked positives are
   diet-counted as positive views but never train (minting unconditioned
   on fold-negative availability) — also ~20k dead payload rows.
4. GOOD Twins and masked hard-negative copies reach training 100%; every
   base population 100%.

Fixes for the findings above live in ER/TODO.md ("Tracking fixes —
before rebuild"); this file holds measurements only.



- W&B mirror (WandbCtx, src/core/wandb_ctx.py) — activates whenever
  WANDB_API_KEY is in .env: config + live metrics + result artifacts per
  run, project "e-r" under the key's default entity. NOTE 2026-09-28:
  historical runs are GONE from fbarulli-none/e-r (runCount 0; ub5q40js
  "not found"; check UI Trash). Recovered history instead: DVC pull
  (training_results/20260913T123559565190Z/worker_1: AUC 0.679, AP 0.846,
  report.json, training.log), HF fbarulli/e-r-training-artifacts (0911
  ablation checkpoints 48/50 + fold metrics), origin/submission
  report.json (full minilm run: AUC 0.9666, AP 0.9914, adj Rand 0.9993).
- `scripts/minimal_flip_slice.py` — per-field twin margins, P@R95,
  donor uniformity (zero-shot or any checkpoint; re-run per checkpoint
  for the curve).
- `scripts/build_field_slice.py` — overall P@R95 per checkpoint.
- `scripts/flip_validity_audit.py` — prose contradiction + transplant
  concentration per bundle (re-run on every rebuilt bundle).
- MNRL per-subset hooks (landing) — per-population loss per epoch.
- Paired A/B harness pattern (MODEL_INPUT_FIX_REPORT.md): each variant
  judged at its own operating point on the same labeled rows; standing
  bar "keep retrieval recall at or above baseline" (the shipped
  redundancy-removal A/B: Youden +0.0083 but recall@1 -3.15pp -> kept).
