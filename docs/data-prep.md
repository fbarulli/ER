# Data preparation

One command rebuilds every training input from `dataset.csv`:

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all
```

It does not train and does not touch Colab. Output lands in
`results/training_prep/<timestamp>/`, ending in a verified `all_tracks_inputs` package.

Expect roughly 40–60 minutes end to end. `suite_inputs` alone is ~40 min and is
the memory-hungry step.

## The stages

| # | Stage | What it does | Output |
|---|---|---|---|
| 1 | `dedupe` | Collapse duplicate listings to one representative | `data/dataset_deduped.csv`, `data/sku_to_rep.csv` |
| 2 | `cross_country_pairs` | Same GTIN in two countries → hard positives | `results/training/second04_pairs_positive.csv` |
| 3–4 | `number_reference` | How to read digits in names (`7up` is a brand, `250ml` is a volume) | `data/number_tokens_reference.csv` |
| 5 | `canonical_and_gates` | Extract attributes per GTIN, then decide every candidate pair | `data/canonical_records.csv`, `data/gate_results.csv` |
| 6 | `gate_census` | Re-measure the gate census as a measured record (no config rewrite; the pin system was removed 2026-10-06) | run dir `gate_census.json` |
| 7 | `labeled_pairs` | Turn gate decisions into training labels | `data/labeled_pairs.csv` |
| 8–9 | `negative_supply` + `discriminator` | Diagnostic lane; generated but **not** used for training in gate mode | `results/negative_supply/<tag>/` |
| 10 | `validation` | The held-out scored population and its fold map | `data/final_validation.csv` |
| 11 | `graph_inputs` | Catalog, splits, clean graph pairs, tensors | `data/track_setup/` |
| 12 | `full_bundle` | Text training bundle: tokens, objective rows, frozen epoch batches | `data/prepared/full/worker_1_baseline.pkl.gz` |
| 13 | `suite_inputs` | Ablation templates, then package everything | `all_tracks_inputs.tar.zst` |
| 14 | `verify_handoff` | Verify all three tracks together | run dir `handoff.json` |

Every stage writes `<stage>.log`, `<stage>.timing.json`, and updates `manifest.json`.

## Two views of the source

`data_prep` reads **raw** columns (`gtin`, `sku_name_eng`, `attribute`). Training
reads the **deduped catalog** (`gtin`, `title`, attributes). They meet only at
`canonical_records.csv` and `gate_results.csv`. Do not treat them as
interchangeable.

## Dedupe

Duplicate listings are collapsed in four tiers. Price **never** picks the
representative — it is a seller attribute, not product identity.

| Tier | Key | Purpose |
|---|---|---|
| **T1** | retailer + GTIN | GTIN is ground truth. Only checksum-valid GTINs. |
| **T1.5** | retailer + *malformed* GTIN | Same product with a broken barcode. Descriptor review decides. |
| **T2** | retailer + title + identity partition | Lossless collapse when all trusted GTINs agree. |
| **T3** | retailer + title + identity partition | Price aggregation, flagged not silent. |

The **identity partition** (`_ident`) is load-bearing. Keying T3 on
`(retailer, title)` alone once deleted 1,086 product-listings and erased 264
products from the corpus entirely. Two different products sharing a title string
at one retailer must never collapse together.

T1.5 escalation: the descriptor bundle alone merges **59%** of provably-different
pairs, because a missing descriptor reads as agreement. So absence of conflict is
not evidence. A byte-identical malformed GTIN at one retailer *plus* a descriptor
verdict is; anything undecided goes to the owner-adjudicated table rather than
being guessed.

Hard invariants, all asserted:

- No `(retailer, title, identity)` duplicate survives.
- Every raw SKU maps to a representative, and `rep_id` covers `0..n-1`.
- **No product loses its last row.** A GTIN present before must be present after.

## The gate

Every candidate pair gets exactly one decision. Nothing is dropped.

- `proceed` — compatible, a training positive
- `hard_no` — provably different products, the hard negative
- `fallback` — contradictory or missing evidence; excluded from labels, sent to review

Tolerances (from `config/training.yaml`):

| Key | Value | Meaning |
|---|---|---|
| `vol_tolerance` | 0.05 | relative volume overlap, so 250 ml ≈ 260 ml |
| `vol_abs_tolerance` | 5.0 | absolute ml; **whichever is wider applies** |
| `raw_conf_threshold` | 0.85 | below this, volume/pack parse is not trusted |
| `consistency_fallback_threshold` | 0.3 | below this → fallback, never `hard_no` |

Two ordering rules carry the weight:

- A definite negative always beats a review flag. Packaging-level one-sidedness is
  checked *after* categorical conflicts — moving it earlier downgraded 79 genuine
  flavour conflicts from `hard_no` to `fallback`.
- Absent evidence is never a veto. Unknown stays unknown and routes to
  `fallback`; it is never fabricated into agreement or into a rejection.

Gate reason strings live in `config/training.yaml` under `gate.reasons` and are the
single source — the gate, the replay audit and the pool miners all read them, so
wording changes land everywhere at once.

## Gate census

**Removed 2026-10-06 (owner ruling).** The census tripwire system — the
source-export rows+sha gate, `gate_census_pin`, and `dedupe_census_pin`, each
with per-site copies in `selftest.py` and on-VM rewriters — is deleted. Drift
controls now rest entirely on per-stage manifest size, the preparation
provenance identity, closure asserts (`input == output + Σ dropped`) and the
Colab package freshness gate.

What still happens at measurement time: the `gate_census` stage writes
`run_dir/gate_census.json` and the stage manifest as **measured records** (no
config rewrite, no pinned equality). The selftest prints the live census
without gating on it. Consequence accepted: an out-of-band consumer run (direct
`labeled_pairs`, standalone dedupe, or a Colab launch against the wrong cohort
export) no longer fails loudly on a changed universe — verify the cohort
*before* launching.

## Labeled pairs

| Class | Rule | Threshold |
|---|---|---|
| positive | `gate_decision == proceed` and similarity ≥ | 0.50 |
| hard negative | `gate_decision == hard_no` and similarity ≥ | 0.80 |
| fallback | excluded — would inject noise into both classes | — |

Similarity is Jaccard over canonical words. The positive threshold came down
0.80 → 0.65 → 0.50; canonical agreement was 1.0000 in every 0.05 band down to 0.50,
so the extra pairs are safe.

The exclusion is counted, never silent. Rows partition into exactly four buckets —
kept, fallback, below-threshold, other — and the closure
`input == output + sum(dropped)` is asserted before the manifest is published.

## Splits

50 / 25 / 25 over **connected components**, four folds: `train` = folds 0+1,
`dev` = 2, `test` = 3. Splitting on GTIN directly was the original defect: 7,489
positives produced ~1,500 straddling pairs per boundary and test shrank to ~130
pairs.

The fix is a **merged component graph**: training positives *unioned with* labeled
positives before components are cut. A positive is one edge, so both endpoints land
in the same component and therefore the same fold — which is what makes it
impossible for a test positive to have a trained-on side. Negatives are never
unioned; they are a similarity relation, not an identity claim.

Both leak guards are asserted *before* anything is written:

- no positive may straddle a fold
- a positive's two endpoints must share a component

### Negative fold policy

Negatives with mismatched endpoint folds need a rule. Two exist:

- **A `withhold_straddle`** — keep raw folds; a mismatched pair scores nowhere
- **B `train_side`** — give the pair one whole fold, preferring the train side

**A is pinned.** The decision was originally B, but at the 2026-10-01 regeneration
the evidence reversed: A scored more thin-heavy negatives (65.7% vs 63.8% of cells
below `min_test_negatives=5`). The rule is that no artifact may ship under a policy
its own evidence rejects, so the config moved with the numbers.

`build_final_validation` re-measures both policies at every emit and refuses to
write if the pinned one stops winning.

### One trap

**Never normalize `row_bc`.** Fold sets are keyed by the caller's *raw* strings.
8,559 of 14,981 GTINs are 13-digit; zero-padding them to 14 makes every lookup miss
and drops the pair silently. Try the raw spelling, then the normalized one, and
count the misses.

## Attribute extraction

`canonical_records.csv` carries 31 columns: the `*_set` fields
(`volume_set`, `pack_set`, `flavor_set`, `sweetener_set`, `made_from_set`,
`carbonation_set`, `package_type_set`, `package_material_set`, …), confidences,
consistency scores and an evidence ledger.

Evidence is drawn from all seven source columns, including `sku_url` — path slugs
carry product tokens. `sku_last_price` is captured for row completeness but is
**never** read as attribute evidence.

Vocabularies are config-owned in `config/vocabulary.json` and validated fail-closed:
a flavor alias pointing at a non-existent lexicon entry fails at config load.

Units come from one table in `config/paths.yaml`. `oz` is marked `ambiguous` at 0.75
confidence because it spans mass and volume.

`made_from` feeds the supporting-review channel only — it can produce `fallback`,
never a veto, and is deliberately not model-visible. It is excluded from veto
dimensions because same-GTIN feed noise is 21.2%.

## Row accounting

Every stage writes a manifest asserting its own closure: `input == output + dropped`.
Each dropped population gets its own key, and deferred populations are recorded
outside `dropped` so the audit trail stays complete without breaking the invariant.

If a stage dies, its log and manifest say which population failed. That is the
whole point of the accounting — a silent drop is a bug, not a statistic.

## Resume

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all --resume-from validation
```

| Resume from | Requires |
|---|---|
| `dedupe` (default) | nothing; full run |
| `validation` | CSV stages done and their manifests verify. Fresh log directory. |
| `full_bundle` | same run dir, identical provenance, this run's `graph_inputs` done |
| `suite_inputs` | same run dir, identical provenance, this run's `full_bundle` done |

Any change to source, config, raw input or checkpoint bytes during the run poisons
resume and is rejected. That is intentional.

Preparation **mutates its inputs**, so it sets `ER_DATA_GATE_ENFORCE=1` and pops
`WANDB_API_KEY`. Smoke inputs are hashed before and after and must be unchanged.
