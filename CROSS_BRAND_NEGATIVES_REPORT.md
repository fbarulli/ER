# Cross-brand hard negatives: mining, integration and evidence

**Status: implemented, tested, measured on real data, committed.** No training was
run (CPU only). This is the DATA/pair-mining half of the brand-separation fix
specified in `MODEL_INPUT_FIX_REPORT.md` §15/§20.4.

The defect, restated with the number that identifies it: the gate generates its
candidate pairs **inside a brand block** (`src/pipeline.py`, "Brand blocking"), so
brand agreement is 100 % in *both* classes of the training population and measured
pair separation for brand is exactly **0.000** while volume separates at +0.837.
The encoder was never given a pair in which brand is the discriminating evidence.
This change mines that population — **different brand, every other critical
attribute agreeing** — verifies each candidate is not a true match, and feeds it
into the existing pair-building path as one more registered negative source.

---

## 1. What was mined — the funnel, on real data

Command (real artifacts: `artifacts/data/dataset_deduped.csv` 61,529 rows,
`results/canonical_records.csv` 13,250 canonicals, `results/gate_results.csv`
135,769 rows; config `mining.cross_brand` as committed):

```
PYTHONPATH=src .venv/bin/python -m training.train \
  --dataset artifacts/data/dataset_deduped.csv --prepare-bundle /tmp/xbundle.pkl
```

`candidate_generation` is the miner's own blocking census; each following row is
one filter, cumulative — every step's `out_count` is the next step's `in_count`
(the miner's `MiningFunnelBase.stages()`, written into `results/logs/training_trace.csv`
as `mining.cross_brand_funnel.<step>`).

| step | in | out | why |
|---|---:|---:|---|
| `candidate_generation` | 13,250 canonicals | **133,127 candidates** | blocking on the exact `(volume, package_type)` values: 273 blocks, 2,592 canonicals carry both, 10,658 carry no package_type evidence and generate none |
| `endpoint_resolution` | 133,127 | 130,339 | 264 canonical GTINs have no row in the deduped frame |
| `same_canonical_guard` | 130,339 | 130,297 | **42** candidates were same-canonical TRUE MATCHES |
| `label_error_guard` | 130,297 | 130,293 | 4 known conflicting-barcode label errors |
| `brand_pair_distinct` | 130,293 | 124,512 | 5,781 candidates share (or miss) a brand |
| `brand_surface_variant_guard` | 124,512 | 124,486 | 26 candidates are one brand written twice |
| `attribute_agreement` | 124,486 | 46,251 | 78,235 attribute conflicts (flavor 46,954 / carbonation 25,750 / pack 24,765 / sweetener 6,802) |
| `similarity_floor` | 46,251 | 30,372 | 15,879 candidates below the configured 0.10 gate-similarity floor |
| `endpoint_diversity_cap` | 30,372 | 29,634 | 738 candidates blocked by `max_per_canonical`/`max_per_brand` |
| `target_cap` | 29,634 | **3,000** | deterministic truncation at `target/2` |
| `direction_expansion` | 3,000 | 6,000 | one candidate emits both `(row A → canonical B)` and `(row B → canonical A)` |
| `baseline_deduplication` | 6,000 | 6,000 | **0** overlap with the same-brand baseline (by construction) |
| `emitted` | 6,000 | **6,000 pair rows** | |

**Final cross-brand negative count: 6,000 pair rows = 3,000 distinct unordered
cross-brand pairs**, covering 363 distinct brands and 1,351 distinct brand pairs.

Funnel closure (asserted in the evidence run):
`133,127 candidates = 130,127 dropped + 3,000 accepted`; `6,000 rows = 2 × 3,000 − 0`.

### Config (`config/training.yaml` → `mining.cross_brand`)

| key | value | why this value |
|---|---|---|
| `enabled` | `true` | the lane is the fix; `mining_profiles.masking_only` sets `cross_brand_enabled: false` |
| `target` | `6000` | ≈29 % of the resulting negative population (20,829). Measured coverage curve below |
| `min_similarity` | `0.10` | strict `>` on the gate's own short-token Jaccard. **0.20 was measured and rejected**: it drops pairs the review band cites (Piacelli/Premier 0.143–0.250) |
| `require_agreement` | `["volume", "package_type"]` | the blocking key. Volume alone blocks 5,693,447 candidates (43× the work for the same answer); package_type alone exists on only 2,712 canonicals |
| `max_per_canonical` | `20` | endpoint diversity (measured max degree 253 without it) |
| `max_per_brand` | `100` | endpoint diversity (measured max brand degree 2,242 without it) |

Measured target sweep (same miner, same data; `target | rows | test-fold cross-brand | queries with ≥1 cross-brand distractor | train-fold cross-brand | share of train negatives`):

```
  3000 |   3000 |  296 | 116/5847 (0.020) |  668 | 0.152
  6000 |   6000 |  522 | 206/5847 (0.035) | 1388 | 0.271
 12000 |  12000 |  928 | 316/5847 (0.054) | 2850 | 0.433
 20000 |  17504 | 1262 | 387/5847 (0.066) | 4276 | 0.534
 30000 |  17504 | 1262 | 387/5847 (0.066) | 4276 | 0.534   <- saturation
```

The lane **saturates at 17,504 rows** with the configured endpoint caps: raising
`target` past that emits nothing more unless `max_per_canonical`/`max_per_brand`
rise. That ceiling is visible in the funnel (`endpoint_diversity_cap`), not
implied. `6000` was chosen so the negatives stay a minority-but-meaningful share:
the same-brand gate population still carries the volume/pack/type signal.

---

## 2. Proof the negatives are correct

Command: `PYTHONPATH=src .venv/bin/python /tmp/final_evidence.py` (reads the real
bundle from `pipeline.build_training_data`, checks **every** emitted row).

```
=== 0-LABEL-CONFLICT CHECK ===
  undirected pos pairs           : 12,986
  undirected gate-negative pairs : 7,702
  undirected targeted pairs      : 284
  undirected cross-brand pairs   : 3,000
  pos ∩ gate_neg                 : 0
  pos ∩ targeted                 : 0
  pos ∩ cross_brand              : 0     <-- no pair carries both labels
  targeted ∩ cross_brand         : 0
  row-index collisions across populations: 0

=== TRUE-MATCH CHECK on the emitted cross-brand rows ===
  rows=6,000 | same_brand=0 | same_canonical=0 | attribute_conflict=0 | label_error=0
```

* **No true matches.** Every one of the 6,000 rows has two *different* canonical
  texts (the same-canonical guard) — the 154-of-350 defect cannot recur: 42
  candidates that would have reintroduced it were caught and counted.
* **No same-brand rows, no attribute conflicts** under the shared evaluator
  (`core.attribute_conflicts.critical_attribute_evaluation`) with the training
  gate's own `vol_tolerance` — i.e. brand is the only discriminating evidence.
* **No label errors**: neither the conflicting-barcode population (same title,
  conflicting barcode) nor a pair sharing a canonical identity.
* Attribute-agreement census over the 6,000 emitted rows (how many rows agree
  explicitly on each dimension): volume 6,000 · package_type 6,000 · carbonation
  5,714 · flavor 1,944 · sweetener 1,484 · pack 618 · pulp 20. Emitted similarity:
  min 0.333 / mean 0.396 / max 0.714.

### Sample rows (brand | title | GTIN <> brand | title | GTIN | similarity)

```
vitae kombucha | 'VITAE kombucha green tea infused with lemon…' (651973512269)
   <> mun kombucha | 'MUN kombucha fermented drink green tea taste…' (8437017259817)  sim=0.714
zingo | 'orange Zero soda, can' (7310070006776)
   <> premier | 'Orange Zero Sugar soda can' (7611612524957)                       sim=0.667
true fruits | 'TRUE FRUITS smothie yellow 250 ml bottle…' (4260122391615)
   <> romantics | 'ROMANTICS pampering smoothie mango and fruit…' (8437006671217)   sim=0.700
london essence co | 'London Essence Pomelo Pink Pepper Tonic…' (5010102243828)
   <> the london essence co | 'The London Essence Co. Original Indian Tonic' (5010102242241)
      ^ DROPPED by the brand-spelling guard (one brand written twice), not emitted
```

**The brand-spelling guard, measured.** Over the 30,388 candidates above the
similarity floor, 16 are *spelling variants* of one brand (`the london essence co`
/ `london essence co`, `mont roucous` / `mont`, `kiju organic` / `kiju`,
`jones` / `jones soda co`) and 12 of those sit in the hardest 3,000 — exactly the
rows a hardest-first target keeps. The guard is the token-subset relation, not a
fuzzy ratio, because the data says so: a `ratio ≥ 0.85` rule matches **0**
candidates, while the 0.60–0.85 band (169 candidates) was inspected and is
*genuinely different brands sharing a word* (`vitae kombucha`/`mun kombucha`,
`thick it`/`thick easy`, `carola`/`cabreiroa`, `eska`/`isklar`) — real hard
negatives a ratio guard would destroy.

**Same-name pairs are kept, deliberately.** 30 emitted rows have *identical*
normalized product names — all inspected as different brands selling the same
generic product (Saint Amand vs Bezoya "mineral water"; Aquabona vs Lanjaron).
Those are the ideal cross-brand negatives; a same-name guard would delete them.
0 emitted rows have byte-equal titles.

---

## 3. Nothing duplicated, nothing dropped silently

* **No sampling at all.** The miner has no RNG: it ranks the surviving candidates
  (highest gate similarity first, ties by GTIN) and takes a deterministic prefix
  under the configured caps. The `replace=True` duplication class is therefore
  structurally impossible here, and the miner asserts its own output is
  duplicate-free before returning.
* Measured: `pos` 23,370 rows / 23,370 distinct; `neg` 14,829/14,829;
  `targeted` 568/568; **`cross_brand` 6,000/6,000 — duplicates 0**.
* **Every candidate is accounted for**: generation census → 8 named filters →
  direction expansion → baseline dedup → emitted (13 steps), each with in/out
  counts that chain. Emitted to `results/logs/training_trace.csv` on every run.
* **Registered population.** `cross_brand_conflict` is in
  `training.training.DATAPOINT_POPULATION_SPEC` (`role="negative_source"`), so the
  per-fold coverage audit visits it: a `cross_brand_conflict` row appears in
  `datapoint_type_coverage_fold*.csv` and in the negative-source census with
  `present/selected/backprop` counters. The static producer-scan test
  (`tests/test_datapoint_coverage.py`) fails if the tag and the registry diverge.

### The adjacent `replace=True` defect, fixed

The brief names a live defect in this same area: negative sampling with
`replace=True` duplicated rows while downstream counters reported them as
distinct pairs. Measured on the real train fold (`/tmp/dup_and_target_probe.py`):

```
BEFORE train_pos=11,686 train_neg=3,734
  class-balance: target=11,686 pool=3,734 -> rows replace=True would DUPLICATE: 7,952
AFTER  train_pos=11,686 train_neg=5,122
  class-balance: target=11,686 pool=5,122 -> rows replace=True would DUPLICATE: 6,564
random-easy (both): unique pool == target -> 0 duplicates (measured, not assumed)
```

`src/training/train.py` now samples **without replacement**, keeps the whole pool
when it is the smaller side, and reports the shortfall as its own number
(`n_class_balance_shortfall`, `n_class_balance_duplicated_rows = 0`, and the
realized ratio). The end-to-end run prints:

```
[class-balance] training positives=46,740 negatives=21,397 ratio=0.458
                (without replacement; unavailable rows=25,343, duplicated rows=0)
```

i.e. the old code would have duplicated 25,343 rows per epoch on this
configuration; the new code reports 0 and says what is missing instead. This is
also why the negative-source census and `distinct_pairs_presented` are now
truthful. Pinned by
`tests/test_mining_hypotheses.py::test_class_balance_never_duplicates_rows`.

---

## 4. Fold isolation

The miner emits the **same shape every other negative lane emits** —
`(source SKU row, other canonical payload index)` — so the existing fold filter
(`core.hard_negatives.pairs_in_set`, both endpoints required) applies unchanged.
Measured on the real holdout split (`holdout_split`, 4 component folds,
7,488/3,743/3,743 barcodes):

| side | negatives BEFORE | negatives AFTER | cross-brand after | share |
|---|---:|---:|---:|---:|
| train | 3,734 | 5,122 | 1,388 | 27.1 % |
| dev | 1,076 | 1,404 | 328 | 23.4 % |
| test | 886 | 1,408 | 522 | 37.1 % |

No mined pair can be trained on one side and evaluated on the other: a pair is
admitted to a side only when **both** endpoints are in it. Pinned by
`tests/test_cross_brand_negatives.py::test_mined_pairs_are_fold_local_when_both_endpoints_share_a_fold`
and `::test_holdout_split_keeps_mined_pairs_on_one_side`, which also assert the
identity `admitted(train) + admitted(dev) + crossing = total`, so the pairs the
fold filter drops are counted rather than assumed away.

---

## 5. Before/after pair-population composition

### 5.1 The training bundle (real, 61,529 source rows)

Command: `PYTHONPATH=src .venv/bin/python /tmp/pair_probe.py {before|after}`.

| population | rows | brand agrees | brand differs | share cross-brand |
|---|---:|---:|---:|---:|
| positives (unchanged) | 23,370 | 23,370 | 0 | 0.0000 |
| gate negatives (unchanged) | 14,829 | 14,829 | 0 | 0.0000 |
| targeted attribute negatives (unchanged) | 568 | 568 | 0 | 0.0000 |
| **cross-brand negatives (new)** | **6,000** | **0** | **6,000** | **1.0000** |
| **negatives total** | 15,397 → **20,829** | | **0 → 6,000** | **0.0000 → 0.2881** |

The brand-constant property is broken by 6,000 rows; nothing else moved
(positives, batch shapes and every pre-existing population are byte-identical).

### 5.2 The repo's own separation metric

`training.attribute_separation` over the real labelled-pair artifact (19,918
pairs) plus the mined pairs as label-0 rows (22,918 pairs). Command:
`PYTHONPATH=src .venv/bin/python /tmp/separation_before_after.py`.

| attribute | BEFORE | AFTER |
|---|---:|---:|
| **brand** | **0.0000** (negative class saturated, flagged weak) | **+0.1925** (not flagged, not saturated) |
| volume | +0.8370 | +0.6751 |
| pack | +0.5836 | +0.5494 |
| package_type | +0.2532 | −0.0003 |
| flavor | −0.0869 | −0.0623 |
| carbonation | +0.0291 | +0.0226 |
| sweetener | +0.0617 | +0.0651 |
| pulp | −0.0514 | −0.0235 |

`P(brand agrees | negative)` moves 1.0000 → 0.8075; the saturation flag clears.
**The dilution of the other attributes is mechanical and must be stated**: the
new negatives agree on volume and package_type *by construction* (they are the
blocking key), so `P(agree | negative)` rises for exactly those dimensions —
package_type most (0.3663 → 0.6198, separation ≈ 0). If the owner wants
package_type to keep separating, drop it from `require_agreement` (volume-only
blocking is available and measured: 5,693,447 candidates) or lower `target`; both
are config, no code change.

**Unverified hypothesis, labelled as such:** the *negative* flavor separation
(−0.0869) was flagged in §15 as possibly sharing the same mining artefact. This
change moves it to −0.0623, i.e. slightly less negative, but it does **not**
explain it: the flavour figure comes from the same-brand gate population, which
this lane does not touch. I have no evidence that the mining artefact causes it.

---

## 6. Candidate-pool effect (secondary benefit, measured)

The "90.5 % of holdout queries have exactly one candidate" degeneracy was already
fixed by the ER-346 retrieval-pool protocol (`competitors_per_query` +
`build_evaluation_pool`), so **this lane does not change the pool size**: every
query receives `1 + 99 = 100` candidates before and after (measured: 5,847
queries, pool size 100–100 in both states). What it changes is the pool's
**composition** — the hard negatives the lane actually knows about are seated
first (`priority_pairs=hard_test`, and `hard_test` carries `neg_pairs`, which now
includes the cross-brand rows):

| measure (test fold, 5,847 queries) | BEFORE | AFTER |
|---|---:|---:|
| real mined distractors seated in the pool | 886 | **1,408** (+58.9 %) |
| queries with ≥1 real mined distractor | 575 (9.83 %) | **717 (12.26 %)** |
| queries with ≥1 **cross-brand** distractor | **0 (0.0000)** | **206 (3.52 %)** |
| mean real distractors per query | 0.152 | 0.241 |

So the honest statement is: the ranking metric's **information content improves**
(more real, verified distractors, and the first cross-brand ones), not its size.

---

## 7. What the OWNER must run, and what pass/fail looks like

Training is yours; nothing here needs a GPU.

**Step 1 — the DATA check (CPU, ~1 min, no training):**

```bash
cd /home/opc/ONE/EuromonitoR
PYTHONPATH=src .venv/bin/python -m training.train \
  --dataset artifacts/data/dataset_deduped.csv --prepare-bundle /tmp/xbundle.pkl
```

PASS if all four lines appear:

```
[cross-brand-negatives] 6,000 label-0 pair rows from 3,000 candidates (133,127 generated -> 30,372 survived every filter; target 6,000, reached=True)
[cross-brand] +6,000 cross-brand static label-0 pairs (enabled=True; 6,000 carry the cross_brand_conflict source in this fold's negative pool)
[class-balance] … (without replacement; unavailable rows=…, duplicated rows=0)
[prepared-bundle] wrote /tmp/xbundle.pkl (… 21,397 negatives)
```

and the bundle carries the provenance:

```bash
PYTHONPATH=src .venv/bin/python -c "import gzip,pickle,collections; d=pickle.load(gzip.open('/tmp/xbundle.pkl','rb')); print(collections.Counter(d['neg_sources']))"
# Counter({'gate': 14829, 'cross_brand_conflict': 6000, 'targeted_attribute_conflict': 568})
```

FAIL if: `emitted=0` / `[cross-brand-negatives] disabled`, or
`cross_brand_conflict` missing from `neg_sources`, or a non-zero
`duplicated rows`. The per-run funnel is always in
`results/logs/training_trace.csv` (`mining.cross_brand_funnel.*`).

**Step 2 — your training run (Colab, unchanged command).** The prepared bundle is
regenerated by the launcher (`python src/cli/colab.py --what train …`); existing
bundles on disk/Colab are **stale** and must be rebuilt (see §9).

PASS if, per fold:
* `logs/<run>-<worker>-matcher/datapoint_type_coverage_fold<i>.csv` contains a
  `cross_brand_conflict` row with `registered=True`, `status="ok"`,
  `presentations > 0`;
* the fold metrics carry non-zero
  `n_train_neg_source_cross_brand_conflict{,_present,_selected,_backprop}` and the
  census `n_train_neg_source_total` still equals the fold's negative count;
* `n_presented_cross_brand_conflict > 0`.

FAIL if that row is `missing`, `unavailable` or `unregistered`, or if the census
no longer sums to the fold total — both mean the population stopped reaching
training.

**Step 3 — the MODEL-level effect (yours, since it needs a trained encoder).**
The pre-change measurement was: 585-pair human-review band, mean score **0.6687
cross-brand vs 0.6936 same-brand → a gap of 0.0249**. Success is a **larger**
same-brand-minus-cross-brand gap on the same band with a re-trained encoder, and
a reduction in cross-brand false merges. I could not measure this here (no
training, and the `brand_analysis` module is not in this tree).

---

## 8. Config, schemas, tests, files

**Config keys added** (`config/training.yaml`):
`mining.cross_brand.{enabled,target,min_similarity,require_agreement,max_per_canonical,max_per_brand}`
and `mining_profiles.*.cross_brand_enabled` (both profiles).

**Pydantic models added** (`src/core/schemas.py`):
`CrossBrandMiningSpec` (`extra="forbid"`, validates that `require_agreement` names
critical dimensions, uniquely) + wired into `MiningSpec` and `MiningProfileSpec`;
`TrainingStats.n_cross_brand_candidates` / `n_cross_brand_resolved`;
`TrainingData.cross_brand_neg` (same `(N,2)` int validator and range check as the
other pair arrays). The funnel readback is a dataclass, like the existing
`MiningFunnel`.

**Tests added** — `tests/test_cross_brand_negatives.py` (22): emitted pairs are
cross-brand and attribute-compatible; same-canonical true matches are never
emitted; the label-error guard; the brand-spelling guard (variant dropped,
LIFEWTR/ZenWTR-class kept); attribute-conflict census; no duplicate rows and both
directions; target cap; endpoint caps; baseline dedup; byte-identical repeat call;
generation census; missing-evidence blocking; strict similarity floor; invalid
`require_agreement`; zero target; **fold locality** (+ the split identity); funnel
cumulative monotonicity; trace-ready readback; wrapper signature drift; the
targeted funnel unchanged by the base-class refactor.

**Tests updated** (each justified):
* `tests/test_mining_hypotheses.py` — the H5 provenance test enumerates the
  producers of negative rows; the new lane is now scanned and executed in the
  sandbox (+3 cases), and a new test pins that class balancing never duplicates
  rows (+2 cases).
* `tests/test_datapoint_coverage.py` — the derived negative-source tuple now
  contains `cross_brand_conflict` (registry change).
* `tests/test_consolidated_trace.py` — the pair census identity now includes the
  new population and asserts its key is present, not silently zero.

No test was deleted or weakened; per-file test-function counts vs `HEAD`:
total 361 → 384, **no file has fewer** (`test_cross_brand_negatives.py` +22,
`test_mining_hypotheses.py` +1).

**Files changed:** `src/core/hard_negatives.py` (funnel base + cross-brand funnel
+ miner + wrapper), `src/core/schemas.py`, `config/training.yaml`,
`src/pipeline.py` (integration, stats, trace), `src/training/train.py` (source
assembly + counters + the no-replacement fix), `src/training/training.py`
(population registry), `tests/test_cross_brand_negatives.py` (new),
`tests/test_mining_hypotheses.py`, `tests/test_datapoint_coverage.py`,
`tests/test_consolidated_trace.py`, `CROSS_BRAND_NEGATIVES_REPORT.md` (this file).

**Suite:** `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` →
**386 passed, 2 skipped**. Ruff (`--select F,E9`, the repo's rule set):
**4 findings before, 4 after, 0 introduced**; the new test file is clean under
the full default rule set.

---

## 9. Reused vs newly created

**Reused (nothing re-implemented):** `core.attribute_conflicts.canonical_attribute_info`
and `critical_attribute_evaluation` (the shared attribute SSOT and the gate's own
volume tolerance); `core.critical_attributes.normalized_attribute_text` (the
accent/punctuation-folding brand identity); `core.hard_negatives.conflicting_barcode_pairs`
(the label-error guard); `core.hard_negatives.pairs_in_set`; `pipeline.jaccard_similarity`
(the gate's own similarity); `pipeline.load_canonical_map`/`build_training_data`
index maps; `MiningFunnel`'s contract (extended, not replaced);
`core.tracing.TraceRun`; `core.ranking_metrics.component_index` /
`competitors_per_query` / `build_evaluation_pool`; `training.folds.holdout_split` /
`component_folds`; `training.attribute_separation`; `core.common.load_config` /
`load_dataset_deduped` / `F`; `config/*.yaml` + the `TrainingSpec` nested-spec style.

**Newly created (all justified):** `MiningFunnelBase` (one cumulative-walk
implementation shared by two lanes instead of a second funnel style),
`CrossBrandMiningFunnel`, `mine_cross_brand_negatives` +
`mine_cross_brand_negatives_with_funnel`, `_brand_identity`,
`_required_agreement_values`, `_brand_surface_variant`, `_canonical_similarity`
(private helpers with one call site each), `CrossBrandMiningSpec`, the two
`TrainingStats` counters, `TrainingData.cross_brand_neg`. The canonical-identity,
label-error, brand-distinctness and attribute-agreement rules are *calls*, not
copies.

**Grep evidence that the miner is new:** `mine_cross_brand|require_agreement|cross_brand`
appeared nowhere in `src/` before this change; the only cross-brand logic that
existed was `mine_hard_negatives` (ANN-based, needs a fine-tuned checkpoint and
an embedding matrix, unavailable pre-training) and the reverted `brand_analysis`
module.

## 10. EXECUTED vs READ

**EXECUTED (real data, commands above):** the full baseline suite before and after
(334 → 386 passed); the miner's funnel on 13,250 canonicals / 133,127 candidates;
the target sweep; every emitted row re-checked for same-canonical / same-brand /
attribute-conflict / label-error (all 0); the 0-label-conflict and duplication
identities over the whole bundle; the fold split and per-fold counts; the two
`replace=True` duplication sites; the separation metric before/after over 19,918
and 22,918 labelled pairs; the retrieval pool composition before/after (5,847
queries ×2); the end-to-end `--prepare-bundle` run and the resulting bundle's
`neg_sources`; ruff on the changed files.

**READ only:** `src/training/prepared_bundle.py` / `train_prepared.py` (to confirm
`neg`/`train_neg`/`neg_sources` are serialized — confirmed, and the concurrent
prepared run observed in the shared trace shows `neg_cross_brand: 6000`);
`src/core/ranking_metrics.py`'s pool builder; the concurrent agents' colab edits.

**Not run:** any training, fine-tuning, GPU or Colab VM; the model-level
separation re-measurement.

## 11. Blast radius

* **Encoder input text: unchanged.** No composition change, so existing
  checkpoints remain valid for *scoring*; but every checkpoint trained before
  this change was trained without cross-brand negatives, so **the separation
  improvement requires a re-train**.
* **Prepared bundles are stale**: `/tmp/xbundle.pkl` and every bundle on disk /
  Colab predate the lane and contain no `cross_brand_conflict` rows. Regenerate
  (the launcher does this) before the next run.
* **Artifacts that become non-comparable:** any future `fold_metrics` row, the
  per-fold coverage CSVs, and the attribute-separation numbers are computed over
  a *larger* negative population than before (20,829 vs 15,397), so per-fold
  negatives and the separation table are not byte-comparable with older runs —
  compare populations, not single numbers.
* `dataset.csv`, `results/*.csv` and `training_results/` were not modified
  (verified: `git status --short -- results/ dataset.csv training_results/` is empty;
  the only trace writes land in the gitignored `results/logs/training_trace.csv`).

## 12. Stated plainly — what this does NOT establish

1. **No model-level evidence.** Whether brand separation improves *in the
   encoder* is unmeasured here; that needs the owner's training run (§7 step 3).
2. **Blocking is not exhaustive with respect to the volume tolerance.** The
   blocking key is the *exact* `(volume, package_type)` value, so a pair whose
   volumes agree only within the gate's tolerance (e.g. 480 vs 500 ml) generates
   no candidate. Measured cost of that exclusion: a coarser volume bucket adds
   only +3,613 (100 ml) / +4,107 (200 ml) survivors to the 47,210, so the
   population is nearly complete — but the exclusion is real and is a property of
   the generation rule, not a bug I silently accepted.
3. **The lane saturates at 17,504 rows** under the configured endpoint caps.
4. **The flavour separation anomaly is not explained** (§5.2) — only slightly
   moved.
5. **A concurrent agent was editing `src/cli/colab.py` and
   `tests/test_colab_setup_path.py`** in this checkout throughout this work (they
   committed 4 commits on top of the base while I worked). Their files are
   **not** part of my commit; their earlier work-in-progress snapshot, which I
   set aside before starting so the tree was clean, is preserved as
   `git stash@{0}` ("FOREIGN WIP (colab runtime packages) preserved by
   cross-brand agent") — its `config/training.yaml` and `src/core/schemas.py`
   parts are already byte-identical to `HEAD`, and its `colab.py` part is an
   older form of code now committed/live.
