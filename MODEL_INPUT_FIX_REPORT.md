# Model-input fix: report and evidence

**Scope.** The user's instruction was *"make sure that code behaves as expected, training will be done separately."*
This work therefore fixes and proves the **model-input text construction**. It makes **no claim about model
quality or score deltas** — no training was run, no GPU was used, and every number below is measured at the
**string level** on real committed artifacts.

Repo `/home/opc/ONE/EuromonitoR`, branch `training`. Baseline suite: **290 passed, 2 skipped**.
After: **313 passed, 2 skipped**. The `cleaned` composition is the **shipped default**; `legacy` remains
selectable from config and is still byte-identical to the pre-change output (§9).

---

## 1. Verdict on the five claims

The diagnosis was supplied as a hypothesis. Measured against the real 585-pair population
(`human_review_enriched.csv`) plus a singleton-GTIN natural experiment built from root `dataset.csv`
(6 400 GTINs with exactly one source row, so the canonical record is a pure function of that row):

| # | Claim | Verdict | What decided it |
|---|-------|---------|-----------------|
| 1 | Canonical mangling (underscore compounds, stripped numbers) | **CONFIRMED** | 569 underscore compounds across 440/585 canonical texts, 0 of which appear in any source row. Numbers destroyed: `bcaa_6000mg` → `bcaa`. |
| 2 | Source/target asymmetry | **CONFIRMED — this is the root cause** | Same product, two unrelated strings. The 24 rows where the source row *is* the target row: mean token Jaccard **0.5129**, max 0.6327, **0 %** ≥ 0.80. |
| 3 | Structured-token crutch | **PARTIALLY CONFIRMED** | Tails are byte-identical on only **6.7 %** of rows, so "identical tokens" is wrong. But the channel's discriminative separation is **+0.0414** (true−false) vs **+0.1506** for base text while occupying ~15 % of character mass — a near-constant floor, not a discriminator. |
| 4 | Boilerplate centroid poisoning | **CONFIRMED** | 100 % of source rows carried literal `brand`/`category`/`breadcrumbs` label artifacts (**8.59 %** of all source tokens). Target text was **14.21 %** English function words; top target tokens were `and` (926), `drinks` (747), `the` (738), `of` (641) — boilerplate outranking every real attribute. |
| 5 | Missing L2 norm / mean-pooling dilution | **REFUTED (as stated)** | L2 normalisation **is** present: `predict_items.py:112,119` (`normalize_embeddings=True`), re-normalised in `core/structured_features.py:249-251`, and `util.semantic_search` is dot-product on unit vectors = cosine. Dilution is real but its cause is different — see below. |

### The headline claim — "score collapse to 0.747–0.750" — is REFUTED, and it is a sampling artifact

`human_review_feature_summary_by_band.csv` defines the population as the **0.55–0.75 review band**.
The observed score distribution over all 585 rows is:

```
ALL            : mean=0.6771  std=0.0516  range=[0.5510, 0.7499]
brand_match=True : n=197  mean=0.6936  std=0.0472  min=0.5597  max=0.7499
brand_match=False: n=388  mean=0.6687  std=0.0517  min=0.5510  max=0.7495
```

The values the user saw (~0.747–0.750) are the **top edge of the selected band**, because
`model_input_comparison.md` prints a handful of exemplars drawn from the top of that band. Scores are
*not* collapsed onto 0.75; the band is 0.55–0.75 **by construction**. What is genuinely wrong is that
brand-match separation is only **+0.0249** — that is the real, and much less dramatic, defect.

### The dilution question, correctly attributed

Mean pooling is not mis-configured, but the texts were long enough to be **truncated at
`max_seq_length: 128`** (`config/training.yaml:407`) and the structured tail is appended **last**, so it
was the first thing cut:

| | mean subwords | >128 | structured-tail survival | tail lost entirely |
|---|---|---|---|---|
| BEFORE source | 100.6 | 19.1 % | 95.1 % | 0.0 % |
| BEFORE target | 114.9 | 27.7 % | **81.6 %** | **11.9 %** |

On the singleton-GTIN population the target text lost its entire structured tail on **20.3 %** of rows.

---

## 2. Root causes, with exact locations

**RC1 — Three independent, copy-pasted text builders.** The same composition existed in
`src/predict_items.py:106-140` (scoring), `src/training/rand_matching.py:1021-1038` + `1141-1157`
(training), and `src/pipeline.py:2047-2105` (payload/audit). They disagreed: the scoring lane read
`row["category"]` with no fallback while the training lane fell back to `category_path`. This is the
non-SSOT pattern the user wanted removed, and it is why the two lanes could drift.

**RC2 — The two lanes assemble *different fields*, so one product yields two texts.**
Source: `clean_sku_text(title, attributes, brand, description, category, category_path)`.
Target: `canonical_model_text(canonical + mode_brand + mode_type + description_evidence + breadcrumb_evidence)`.
For the *same* row (`sku_id=153415775`, title `bcaa 6000mg Pear can`) the target text described
"stored at room temperature." plus a breadcrumb dump while the source text described the raw attribute
blob. Jaccard 0.3889.

**RC3 — Compound mangling in the canonical cleaner.** `pipeline.py:1235-1239`
(`canonical_model_text`) drops every digit-bearing part of an n-gram and re-joins the remainder with
`_`. `bcaa_6000mg_pear` → `bcaa_pear`: a token that exists in no source text and cannot lexically
match the source words `bcaa` / `pear`, with the discriminative `6000mg` deleted.

**RC4 — Literal field-label artifacts injected into the source text.** `pipeline.py:1125-1132`
builds `f"brand {brand}"`, `f"description {description}"`, `f"category {category}"`,
`f"breadcrumbs {breadcrumbs}"`. The **words** `brand`, `description`, `category`, `breadcrumbs` then
survive tokenisation on 100 % of rows — pure constant offset.

**RC5 — Percentage evidence destroyed.** `normalize_text` deletes `%`, then `MINIMAL_STOPWORDS`
contains the bare volume numbers `100` and `2`, so `Juice Content: 100%` and `Juice Content: 0-2%`
were silently discarded. **92.6 %** of source rows carry such a value; **0** of them reached the model.

**RC6 (NOT FIXED, reported) — pack sentinel asymmetry.** `core/structured_features.py:112-116`
(`sku_info`) emits `{1.0}` when no pack count was observed, while `core/structured_features.py:128-149`
(`canonical_info`) passes the empty `pack_set` straight through. **425/585 rows (72.6 %)** therefore
get `[FIELD_PACK_SIZE] pack_qty_1` on the source side and nothing on the target side — for the same
product. The in-code rationale at `rand_matching.py:1101-1104` claims the source default matches "the
canonical singleton token `pack_qty_1`"; the canonical artifact emits no such token, so that comment is
factually wrong. **Left unfixed deliberately** — see §8.

---

## 3. What changed, and why

### One shared builder

New module **`src/core/model_input.py`** is the single source of truth. All three former call sites now
call `build_sku_text(row, info)` / `build_canonical_text(record, info)`:
`predict_items.py:100,104`, `rand_matching.py` (item + sku paths), `pipeline.py:2051,2079`.
~115 lines of duplicated composition were deleted (net −51 lines outside the new module).

While consolidating I removed the now-dead per-lane gate recomputation
(`sf_text`, `self.structured_text`, `structured_append_to_text`) — the builder owns that gate, so the
three sites can no longer disagree about whether the structured channel is on.

### Config-gated, cleaned is the default, legacy is the fallback

`config/training.yaml`:

```yaml
  model_input:
    profile: "cleaned"       # cleaned | legacy
    include_evidence: false
```

* `cleaned` + `include_evidence: false` → **the new composition, and the shipped default.** No config
  edit is needed to get it.
* `legacy` + `include_evidence: true` → **byte-identical to the committed code** (the pre-change
  behaviour). This is the fallback selection; see §9 for how to restore it.
* `legacy` + `include_evidence: false` → the original composition with the evidence channel ablated.
* `cleaned` + `include_evidence: true` → **rejected at config load** (contradiction).

One profile plus one granular flag, with the contradictory combination refused rather than ignored.
The schema is `src/core/schemas.py` → `TrainingSpec.ModelInputSpec` (pydantic, `extra="forbid"`,
`@model_validator`), reached through the existing `core.common.load_config()` — no new config mechanism.

Verified after the flip: on all 855 fixture rows the no-argument default equals the `cleaned` profile
and **never** equals the captured legacy string, while the explicit `legacy` selection still reproduces
the golden bytes with 0 mismatches.

### The `cleaned` composition

Ordered blocks **brand → title → attributes**, one shared normalizer for both lanes, no
description/breadcrumb channel.

Two design decisions were made **on measurement, not preference**, and both are pinned by tests:

* **Underscore compounds are split** (`bcaa_pear` → `bcaa pear`) so they can lexically match source text.
* **Literal `[BRAND]`/`[TITLE]`/`[ATTRIBUTES]` marker tokens are NOT emitted.** I implemented them
  first, then swept four variants on both populations:

  | variant | singleton margin | review-band margin |
  |---|---|---|
  | legacy | **+0.4944** | +0.1161 |
  | A markers + full tail | +0.4148 | +0.1883 |
  | B markers, tail w/o volume/pack | +0.4157 | +0.1714 |
  | C no markers, tail w/o volume/pack | +0.4562 | +0.1844 |
  | **D no markers + full tail (SHIPPED)** | +0.4483 | **+0.2010** |

  The markers are constant mass shared by *every* pair: they lifted cross-pair overlap from 0.084 to
  0.164 while lifting true-pair overlap far less. They are boilerplate of exactly the kind this defect
  is about, so the `[Brand] [Title] [Attributes]` structure is carried by **field selection and order**,
  not by marker tokens. Variant D is the best cleaned variant on both populations. This is a deliberate
  departure from the literal bracket notation in the user's stated preference, and it is stated here
  rather than buried.

* **Percentage evidence is preserved** as `pct100` / `pct0to2` / `pct5.5` tokens, captured *before*
  `normalize_text` removes the sign, because `100` and `2` collide with `MINIMAL_STOPWORDS`.

### The structured `[FIELD_*]` channel — is it redundant?

**Partially, and I kept it.** The numeric vector (`structured_features.vector`) encodes only
**volume and pack**; `package_type`, `flavor`, `carbonation`, `sweetener` and `pulp` have **no** numeric
encoding and exist *only* in the text channel. So the text channel is redundant for volume/pack and
irreplaceable for the rest. Removing the redundant volume/pack tokens was measured (variants B and C):
it slightly improved the singleton margin and clearly worsened the review-band margin. It stays.

---

## 4. Before/after model-input strings (real rows)

### Identical product — `sku_id=153415775`, title `bcaa 6000mg Pear can`, brand `Powerking`

```
BEFORE SRC: bcaa pear sports bcaa sweetener aspartame energy source green tea pear carbonated caffeine
            energy boosting brand powerking category energy drinks breadcrumbs
            [FIELD_VOLUME] volume_ml_500 [FIELD_PACK_SIZE] pack_qty_1 ...
BEFORE TGT: powerking pear carbonated aspartame_energy_source_green Powerking stored at room temperature.
            all supplies sports energy drinks energy beverages energy drinks
            [FIELD_VOLUME] volume_ml_500 [FIELD_PACKAGE_TYPE] package_type_can ...
            Jaccard = 0.3889

AFTER  SRC: powerking bcaa 6000mg pear sports bcaa sweetener aspartame energy source green tea pear
            carbonated pct0to2 caffeine 15 25 energy boosting
            [FIELD_VOLUME] volume_ml_500 [FIELD_PACK_SIZE] pack_qty_1 ...
AFTER  TGT: powerking pear carbonated aspartame energy source green bcaa 6000mg
            [FIELD_VOLUME] volume_ml_500 [FIELD_PACKAGE_TYPE] package_type_can ...
            Jaccard = 0.6296
```

`aspartame_energy_source_green` now contributes the matchable words `aspartame energy source green`;
`6000mg` survives; the `brand`/`category`/`breadcrumbs` artifacts and the prose are gone.

### Different brands — `Bare Nature` vs `Savsé`

```
AFTER SRC(left) : bare nature bare nature peach vitamin iced tea 20 oz. pk. 591 tea black pct0to2
                  peach tea sugar antioxidant ... [FIELD_VOLUME] volume_ml_591 ...
AFTER SRC(right): savs savse nina super blue raw smoothie 250ml sweetener sugar cold press smoothie
                  caffeine 150 vitamin antioxidants ... [FIELD_VOLUME] volume_ml_250 ...

cross Jaccard = 0.2188   (legacy 0.1856)
```

Different brands are **not** artificially identical — the strings are plainly distinct, and the
cross-pair similarity stays far below the true-pair similarity (§5).

---

## 5. Population numbers, string level only (no model scores)

The **true** column is Jaccard(source *i*, its own target). The **cross** column is the mean of
Jaccard(source *i*, 12 other rows' targets). The **margin** is the retrieval-relevant quantity.

### 585-pair review band

| | true Jaccard | cross Jaccard | margin | rows ≥ 0.60 |
|---|---|---|---|---|
| BEFORE | 0.1896 | 0.0736 | +0.1161 | 1.4 % |
| AFTER | **0.3200** | 0.1190 | **+0.2010** | 2.2 % |

True-pair similarity **+68.8 %**, margin **+73 %**.

### Singleton-GTIN population (n=150 of 6 400) — the honest counter-result

| | true Jaccard | cross Jaccard | margin | rows ≥ 0.60 |
|---|---|---|---|---|
| BEFORE | 0.5753 | 0.0810 | **+0.4944** | 34.0 % |
| AFTER | 0.5588 | 0.1105 | +0.4483 | **38.0 %** |

**The margin regresses here (−9.3 %)** while the share of rows at ≥ 0.60 improves (34 % → 38 %).
The likely reason is that the legacy composition shares a large amount of *category/breadcrumb
boilerplate* between a source row and its own canonical, which inflates the true pair without adding
discriminative signal — i.e. the Jaccard margin partly flatters the legacy path on this population.
I cannot separate those two effects without training, so **I do not claim the cleaned profile is
uniformly better**. On the 24 `source_original == target_original` rows inside the review band the
mean Jaccard also regresses (0.5129 → 0.4646).

### Defects that are unambiguously fixed

| measure (n=585) | BEFORE | AFTER |
|---|---|---|
| non-structured underscore compounds in target text | 569 occurrences / 440 rows | **0 / 0** |
| source field-label artifacts (`brand`/`category`/`breadcrumbs`/`description`) | 2298 / 26748 = **8.59 %** | 1 / 15251 = **0.01 %** |
| English function words in target text | 5099 / 35886 = **14.21 %** | **0 / 7972 = 0.00 %** |
| rows carrying percentage evidence in source text | **0** | **542 (92.6 %)** |
| target text truncated: structured tail lost entirely | **11.9 %** | **0.0 %** |
| target text > 128 subwords | 27.7 % | **0.0 %** |
| `6000mg` present in target text | 0 rows | 1 row (the only row that has it) |

Percentage reaching the **target** text remains 0 in both — `results/canonical_records.csv` contains no
percentage for any of the 585 rows (verified). That is an upstream `generate_canonical` gap, not a
text-builder gap, and is out of scope here.

---

## 6. Tests

**290 passed, 2 skipped → 313 passed, 2 skipped.** No pre-existing test needed changing:
`tests/test_unit_canonicalization.py:91-98` pins the exact `append_text` output
(`"water [FIELD_VOLUME] volume_ml_237 [FIELD_PACK_SIZE] pack_qty_12"`) and still passes unchanged,
because `append_text` and the structured channel were left exactly as they were.

**No test depends on the default implicitly.** Every test that asserts a legacy string passes the
legacy spec explicitly, and every test that asserts cleaned behaviour passes the cleaned spec
explicitly. The evidence is the default flip itself: switching the shipped default from `legacy` to
`cleaned` broke **zero** tests, and the legacy byte-equivalence test still fails loudly if the fallback
path drifts. The default is nevertheless pinned by its own test
(`test_shipped_config_defaults_to_the_cleaned_profile`, plus a test that the no-argument selection
equals it), so the default is covered by the suite rather than being accidental.

New: **`tests/test_model_input_contract.py`** (23 tests) — legacy byte-equivalence over all 855 fixture
rows on both sides; the shipped default is pinned to cleaned; the no-argument path resolves to it;
legacy stays selectable as the fallback; the contradiction is rejected; evidence ablation; compound
splitting; number and percentage preservation; evidence excluded on both lanes; one normalizer across
lanes; no literal markers; margin improvement; true-pairs-far-above-cross-pairs; different brands stay
distinct; `title_only` variant semantics; both lanes call the shared builder; the composition is in the
ANN fingerprint inputs; the run trace and the checkpoint manifest record it.

New fixture: **`tests/fixtures/model_input_golden.json`** (855 records, 2.0 MiB), captured from the
**unmodified** code before any edit — 585 review-band pairs, 120 dataset rows, 150 singleton-GTIN rows.

Files changed:

```
config/training.yaml          | +17   model_input block (default: cleaned)
src/core/schemas.py           | +26   TrainingSpec.ModelInputSpec
src/core/model_input.py       | new   the shared builder + model_input_provenance()
src/pipeline.py               | payload call sites consolidated + composition trace row
src/predict_items.py          | -42   scoring lane consolidated
src/training/rand_matching.py | training lane consolidated; fingerprint inputs
src/training/training.py      | +7    checkpoint manifest records the composition
scripts/show_model_input_comparison.py | 4th call site consolidated (see 10.1)
tests/test_model_input_contract.py | new (23 tests)
tests/fixtures/model_input_golden.json | new (855 frozen legacy rows)
MODEL_INPUT_FIX_REPORT.md     | new
```

---

## 7. EXECUTED vs READ

**EXECUTED** (all CPU, no training, no GPU):

* `pytest tests/ -q` before (290 passed / 2 skipped) and after (313 / 2) the change, including a run
  after the default was flipped from `legacy` to `cleaned` — which broke no test.
* A no-argument-default check over all 855 fixture rows: the default equals the `cleaned` profile on
  855/855 rows, never equals the captured legacy string, and the explicit `legacy` selection still
  reproduces the golden bytes with 0 mismatches.
* `/tmp/repro_5claims.py` — rebuilt source/target strings for all 585 pairs through the real committed
  functions and measured Jaccard, brand-match/score separation, compound origins, boilerplate mass.
* `/tmp/repro_claims34.py` — per-channel discriminative separation and boilerplate mass.
* `/tmp/proto_cleaned.py`, `/tmp/symmetry_singleton.py` — composition prototypes including the
  singleton-GTIN experiment over 6 400 GTINs (400-row sample).
* `/tmp/variant_sweep.py` — the four-variant marker/tail sweep reported in §3.
* `/tmp/capture_golden.py`, `/tmp/extend_golden.py` — golden fixture capture from unmodified code.
* `/tmp/final_measure.py` and an addendum — every before/after number in §5.
* Tokenizer length / truncation measurement with `AutoTokenizer` on
  `artifacts/models/all-MiniLM-L6-v2` over all 855 fixture texts.
* `import pipeline, predict_items, training.rand_matching` after rewiring.

**READ only (not executed):** `src/core/structured_features.py`, `src/pipeline.py`,
`src/predict_items.py`, `src/training/rand_matching.py`, `src/core/schemas.py`, `src/core/common.py`,
`config/training.yaml`, `training_results/.../human_review_feature_summary_by_band.csv`,
`model_input_comparison.md`, `tests/test_unit_canonicalization.py`.

**A data-freshness caveat found while reproducing.** `human_review_enriched.csv` carries its own
`canonical` column, and that column is **stale** relative to the committed
`results/canonical_records.csv`: for `gtin=7611612221887` the enriched file says
`powerking aspartame_energy_source_green bcaa_6000mg_pear carbonated`, while the committed artifact —
the file both lanes actually read via `load_canonical_map()` / `canonical_records_frame()` — says
`powerking pear carbonated aspartame_energy_source_green bcaa_6000mg`. All measurements here were
rebuilt from `results/canonical_records.csv`, i.e. the SSOT, so they describe what the code really
feeds the encoder today. Expect the strings printed in `model_input_comparison.md` to differ slightly
from a fresh rebuild for this reason.

**Deliberately NOT run:** `train.py` / any fine-tuning; the real pipeline end-to-end (it rewrites
`results/*.csv`); any GPU work. `results/` was verified clean (`git status --short results/` empty) and
`dataset.csv` was opened read-only.

---

## 8. What I could not determine

1. **Whether the cleaned profile actually improves retrieval.** Not determinable without training.
   The string-level evidence is **mixed** (§5): clear margin gain on the review band, regression on the
   singleton-GTIN population and on the 24 identical-product rows. I am not claiming a score delta.
2. **Pack sentinel asymmetry (RC6) is left unfixed.** The fix is a one-line semantic decision in
   `core/structured_features.py` — either `canonical_info` mirrors `sku_info`'s `{1.0}` sentinel, or
   `sku_info` stops asserting `pack_qty_1` for an unobserved pack. Both change *legacy* bytes and both
   change the numeric structured vector (whose `block()` uses presence), so this is an owner decision,
   not a text-builder decision. It affects **425/585 rows** and is the largest remaining symmetry defect.
3. **Why the legacy path scores better on the singleton margin** — shared category/breadcrumb
   boilerplate inflating the true pair, or a genuine advantage of the longer text. Needs a model to separate.
4. **The canonical artifact carries no percentage** for any of the 585 rows, so juice-content evidence
   can only ever be a source-side signal today. That is a `generate_canonical` question.
5. **Legacy field-resolution drift.** The scoring lane read `category` with no fallback while the
   training lane fell back to `category_path`. The consolidated builder uses the training lane's rule
   (with fallback). On all 855 fixture rows the two are identical — verified byte-for-byte — but on a
   hypothetical row with `category` absent and `category_path` present, the consolidated legacy output
   differs from what `predict_items.py` used to produce. This is the drift being fixed, and it is
   called out because it is the one place where "byte-identical" is guaranteed on real data rather
   than by construction.

---

## 9. Which input is active, and how to go back (no code revert required)

**The new input is already the default** — `config/training.yaml`, under `training:`, ships as:

```yaml
  model_input:
    profile: "cleaned"
    include_evidence: false
```

No config edit is required to get the new composition. Both lanes (training candidate retrieval and
scoring prediction) and the payload stage read this block through `core.model_input`.

**Restore the original committed behaviour** — change those two values to:

```yaml
  model_input:
    profile: "legacy"
    include_evidence: true
```

That single edit restores the pre-change output **byte for byte**, verified against
`tests/fixtures/model_input_golden.json` (855 rows captured from the unmodified code before any edit).
No code revert, no branch switch, no rebuild.

**Ablation only** (the original composition without the description/breadcrumb channel):

```yaml
  model_input:
    profile: "legacy"
    include_evidence: false
```

Do not set `cleaned` with `include_evidence: true`; config load rejects it with a named error.
`tests/test_model_input_contract.py::test_legacy_profile_reproduces_golden_bytes` fails loudly if the
`legacy` profile ever stops reproducing the captured strings, so a fallback that silently changed
output cannot ship.

---

## 10. Blast radius map

The change alters **the text the encoder consumes**. Everything below is what that touches.
"Verified" means I executed or read the specific thing; "not checked" is stated as such.

### 10.1 Consumers of the builder (verified by grep — 4 call sites, not 3)

| # | Consumer | Path | Effect |
|---|---|---|---|
| 1 | `src/predict_items.py:100,104` | **scoring lane** — SKU + canonical text | Emits different text, so predictions, scores and the assignment CSV change. |
| 2 | `src/training/rand_matching.py:1031` + `:1137` | **training lane** — item texts and SKU texts for candidate retrieval / ANN | Different item embeddings; different retrieved candidates. |
| 3 | `src/pipeline.py:2065,2093` | **payload stage** of `run_within_brand_pipeline` | The payload that produces embeddings and the pair bundle. |
| 4 | `scripts/show_model_input_comparison.py:117,128` | audit/diagnostic that produced the user's 12 exemplars | **Was a 4th copy-pasted composition** — found and consolidated in this round. Previously it would have kept printing the legacy strings after the lanes moved on, i.e. the diagnostic would have lied. |

Transitive consumers, traced:

* `src/training/complete_colab_worker.py:132` runs `python -m predict_items` as a subprocess → **predictions regenerate**.
* `src/training/build_ann_index.py:23` constructs `RandMatcher` → **ANN build path affected**.
* `src/training/rand_matching.py:3203` is the CLI entry (`er-rand-match`) → **operational entry point affected**.
* `src/training/training.py:1665` calls `refresh_finetuned_ann` (`src/training/ann_refresh.py:57`), which produces the `ann_finetuned` presented population and rewrites negative text from the ANN index → **depends on embeddings built from this text**.
* `src/core/hard_negatives.py`, masking and the pair bundle consume the text indirectly; **not individually re-verified** — they take the payload, not the composition.

**Not affected (verified):** the **NER lane**. `ner.data_prep.output_csv`,
`ner.semantic_training.INPUT_CSV` and `ner.semantic_evaluation.input_csv` all point at
`${results_dir}/dataset_model_input.csv` (root `dataset_model_input.csv`, 4.3 MB), and **no file under
`src/ner/` references `clean_sku_text`, `canonical_model_text`, `strip_schema_words` or
`core.model_input`**. That artifact looks like a model-input file but belongs to a different lane.

### 10.2 Artifacts that become stale or non-comparable

| Artifact | Affected? | Must be | Local? |
|---|---|---|---|
| `results/ann_index/` (`catalog.hnsw` 24 MB, `catalog_embeddings.npy` 20 MB, mapping, metadata) | **VERIFIED STALE.** Stored `preprocessing_fingerprint` is `None`; the active fingerprint is `6221c7d9…`. `PersistentHnswIndex.load` raises `ValueError: persisted HNSW metadata is stale` (executed). | **REBUILD** | **Actionable** — and automatic: `rand_matching` catches the `ValueError` and sets `rebuild_ann_index = True`. Untracked/ignored. |
| `training_results/0915T063500554948Z/worker_2/_checkpoints/all-MiniLM-L6-v2/r0915T063500554948Z-worker_2-ann_embedding_f0/checkpoint-44` | **NON-COMPARABLE.** Weights stay loadable, but every metric they produced was measured on the other composition. | **RETRAIN** | **IMPOSSIBLE locally** (no training in this task). Untracked/ignored. |
| The other checkpoint sets: `0915T063500554948Z/worker_1`, `20260914T160246465306Z/worker_2`, `20260914T192428300299Z/worker_1`, `mixed_20260913T220423946172Z/worker_1` | Same as above — **all trained on the legacy text.** | **RETRAIN** | **IMPOSSIBLE locally.** |
| Cached embeddings (`results/ann_index/catalog_embeddings.npy`) | **VERIFIED STALE** — same fingerprint gate as the index. | **REBUILD** | **Actionable (automatic).** |
| `results/canonical_records.csv`, `results/gate_results.csv` | **VERIFIED NOT AFFECTED.** They are *inputs* to the change, not outputs: the diff to `pipeline.py` is confined to the payload block (the two builder calls plus the trace row) and touches no canonical-generation or gate function. Files on disk unchanged. | Nothing | n/a |
| `dataset.csv` (root, 54 MB) | Read-only original data; unchanged. | Nothing | n/a |
| `dataset_model_input.csv` (root) | **VERIFIED NOT AFFECTED** — NER lane (§10.1). | Nothing | n/a |
| Prior reports/metrics: everything under `training_results/*` (0 tracked files), the **tracked** root `report.json`, the untracked `training.log`, and `training_results/0915T075044186132Z/worker_1/report/human_review_features/model_input_comparison.{md,txt,csv}` | **NON-COMPARABLE / SUPERSEDED.** Scores in them were produced from the legacy text; the comparison files were rendered by the script now consolidated, so re-running it prints the cleaned composition. | Re-run / supersede | **Actionable** (re-run the diagnostic); scores need retraining. |
| `results/logs/training_trace.csv` | Gains the new `payload.model_input_composition` row. | Regenerate | **Actionable** (next payload run). Untracked/ignored. |
| `tests/fixtures/model_input_golden.json` (mine) | **MUST NOT be regenerated.** It is the frozen *legacy* contract captured before any edit. Regenerating it from current code would silently destroy the rollback guarantee. | Keep frozen | n/a |

### 10.3 Tests, fixtures and config keys

* **New:** `tests/test_model_input_contract.py` (23 tests), `tests/fixtures/model_input_golden.json`
  (855 frozen legacy rows).
* **Unaffected, still passing:** `tests/test_unit_canonicalization.py:91-98` (`append_text` untouched),
  `tests/test_hnsw_index.py:59` (the fingerprint-mismatch path it already proves is now load-bearing for
  this change), `tests/test_datapoint_coverage.py` (producer scan unchanged).
* **New config keys:** `training.model_input.profile`, `training.model_input.include_evidence`, validated
  by `TrainingSpec.ModelInputSpec`. No key removed or renamed.

---

## 11. No coverage gaps

The change emits **no new datapoint population**. What it newly emits is a **provenance** value
(`model_input_provenance()`), which is registered in the existing mechanisms rather than a new one.
Both halves were executed.

### 11.1 The datapoint-population registry is still gap-free (executed)

Using the same three producer idioms and the same `PRODUCER_FILES` list as
`tests/test_datapoint_coverage.py`, then driving the real `_write_datapoint_usage`:

```
scanned tags            : 9
registered populations  : 8 -> ['ann_finetuned','attribute_conflict','gate','gate_positive',
                                'hard_positive','masked_positive','random_easy',
                                'targeted_attribute_conflict']
declared fallback tags  : ['hard_neg','hard_negative','unknown']  (rejected loudly, not populations)
EMITTED BUT UNREGISTERED: NONE

 fold                  population  registered  expected_pairs  presentations  distinct  status
    0               ann_finetuned        True               1              1         1      ok
    0          attribute_conflict        True               1              1         1      ok
    0                        gate        True               1              1         1      ok
    0               gate_positive        True               1              1         1      ok
    0               hard_positive        True               1              1         1      ok
    0             masked_positive        True               1              1         1      ok
    0                 random_easy        True               1              1         1      ok
    0 targeted_attribute_conflict        True               1              1         1      ok

populations visited by the audit : 8   == registered set : True
rows flagged unregistered : NONE      rows with status 'missing' : NONE
n_unregistered_datapoint_populations = 0     n_missing_datapoint_populations = 0
```

**Negative control** — a tag that is emitted but not declared must fail loudly, not vanish:

```
raised as required: UnregisteredDatapointPopulationError
message: fold 0: producer-emitted datapoint population(s) outside DATAPOINT_POPULATION_SPEC:
         'not_a_registered_population' (expected_pairs=1, presentations=1).
coverage artifact still written before raising: True
the undeclared tag IS present in the artifact: True (status=['unregistered'], registered=[False])
```

So the property requested holds: **emitted set == registered set == visited set**, and the failure mode
(a name emitted but never registered, dropping rows silently while the audit reports success) raises
instead of passing quietly.

### 11.2 The gap this change WOULD have opened — found and closed

The registry was not the only silent-reuse seam. A persisted ANN index is reusable only if the text that
produced its embeddings is unchanged, and that was gated by `preprocessing_fingerprint`, built (before
this change) from `structured_features` + `unit_canonicalization` **only**
(`rand_matching.py:995`). Switching composition changes the text but not the catalog, the checkpoint or
the code path — so a profile switch would have **silently reused an index built on the other composition**:
stale embeddings served as valid, which is the same failure mode in a different mechanism.

Closed by adding the composition to the fingerprint inputs
(`training.rand_matching.preprocessing_fingerprint_inputs`) and proving the rejection executes:

```
stored fingerprint : None
active fingerprint : 6221c7d9fa9a98210f92235c2c32cc32e3efe18faf56e62e3e1af8dd97fcceb5
=> mismatch (index must be rebuilt): True
load REJECTED as required -> ValueError
  message: persisted HNSW metadata is stale: {'M': (32, 16),
           'preprocessing_fingerprint': (None, '6221c7d9…')}
```

`tests/test_model_input_contract.py::test_ann_fingerprint_inputs_include_the_composition` pins the first
link of the chain; `tests/test_hnsw_index.py:59` already pins the second.

---

## 12. Composition traceability (trace + manifest + fingerprint)

Requirement: any artifact must be traceable to the exact input contract that produced it. Three
existing mechanisms now carry `model_input_provenance()` — no parallel mechanism was added:

1. **Run trace** — the payload stage writes a run-scope row before building any text
   (`pipeline.py`, step `payload.model_input_composition`, `scope="run"`), so the data-prep trace names
   the composition:
   `detail: {"include_evidence": false, "profile": "cleaned"}`.
2. **Checkpoint manifest** — `_write_checkpoint_manifest` now writes
   `"model_input": {"profile": ..., "include_evidence": ...}` into `checkpoint_manifest.json`, so a
   retrained checkpoint declares the composition its weights were trained on and two checkpoints from
   different compositions are distinguishable after the fact.
   `test_checkpoint_manifest_records_the_active_composition` executes the real writer.
3. **ANN fingerprint** — the composition is hashed into the index's reuse contract (§11.2), so an index
   cannot outlive the composition that built it.

The audit script `scripts/show_model_input_comparison.py` also prints the active composition in its
markdown header and per-row blocks, and its "exact model input" fields now come from the shared builder —
verified by running it (to `/tmp`, never `results/`), which emitted the cleaned text
`marcel lemon still sugar strawberry nectar [FIELD_VOLUME] volume_ml_250 …` for a real row.

### Residual, reported not fixed

A **checkpoint is not validated against the composition at scoring time.** `predict_items --model <ckpt>`
will happily run a legacy-trained checkpoint over cleaned text; the ANN fingerprint guards the index,
not the weights. Closing it needs a load-time check against the manifest key added above, which is a
retraining-workflow decision rather than a text-builder one. Stated so the retrain can decide.

---

## 13. Universal symmetry: the pack sentinel and every other attribute

### 13.1 The audit — every attribute, both channels

`sku_info` and `canonical_info` were compared on an EMPTY record (the cleanest
way to see an implicit default) and over all 855 fixture rows:

| attribute | unobserved treatment, SOURCE | unobserved treatment, TARGET | verdict | after |
|---|---|---|---|---|
| `volume` | omit (empty set) | omit (empty set) | already symmetric | unchanged |
| **`pack`** | **`{1.0}` sentinel** (`structured_features.py:112-116`) | **empty** (`:128-149`) | **ONE-SIDED DEFAULT** | implicit `1.0` on BOTH sides |
| `package_type` | omit | omit | already symmetric | unchanged |
| `flavor` | omit | omit | already symmetric | unchanged |
| `carbonation` | omit | omit | already symmetric | unchanged |
| `sweetener` | omit | omit | already symmetric | unchanged |
| `pulp` | omit | omit | already symmetric | unchanged |

`pack` was the **only** one-sided implicit default, in **both** channels. The
categorical attributes do disagree between the two sides of a *retrieved* pair
(54/34, 58/40, 139/52 rows in the review band) but that is **evidence
disagreement about two different products**, not a default: an empty record
returns empty sets from BOTH extractors, verified by execution. The canonical
side also consults `mode_flavor` / `extract_critical_claims` where the source
side does not — an **evidence-source** difference, not a default, and
deliberately left alone.

### 13.2 The fix, and the byte-identity guarantee kept honest

`core.structured_features.symmetric_info` applies the rule; `core.model_input.
model_input_info` is the profile-aware entry point that BOTH the text channel
and the numeric vector read, so the two can no longer disagree. The implicit
value is `training.structured_features.implicit_pack_qty` (config), validated
`> 0`.

**The symmetry fix is scoped to `cleaned`.** `legacy` returns the info
unchanged, so the golden fixtures keep passing **by construction, not by
weakening them** — re-verified this round: **0 byte-mismatches across all 855
fixture rows × 2 sides**.

| measure (855 fixture rows) | before | after |
|---|---|---|
| pack-token presence agrees source vs target | 27.4 % | **100 %** |
| identical numeric vector source vs target | 18.8 % | **82.6 %** |
| rows where an unobserved source pack now emits `1.0` on both sides | — | **579/599** |

The 20 rows that do **not** end at `{1.0}` carry an explicitly observed
`pack_set` (`[6]`, `[2,5]`) and keep it — the implicit value fills a gap, it
never overwrites evidence. **0 unexplained cases.**

`rand_matching.py:1117-1120` carried a rationale claiming the source sentinel
already "matches the canonical singleton token pack_qty_1". It did not — that
comment is corrected in place.

### 13.3 The universal audit found a SECOND, worse defect: accents

Chasing the user's hypothesis that brand variants are "lexically near-identical
but unnormalised" exposed something worse. `normalize_text` deletes every
non-ASCII character, so an accent does not merely go unfolded — it becomes a
**word break that corrupts the token**:

```
'Brämhults'        -> ['br', 'mhults']
'Côteaux Nantais'  -> ['teaux', 'nantais']
'Björk'            -> ['bj', 'rk']
'Reál' -> ['re']      'Réal' -> ['al']      'REAL' -> ['real']
```

Two spellings of one brand could therefore **never** match, and 47 distinct
canonical brands carry non-ASCII. Fixed by folding diacritics (NFKD + drop
combining marks) before `normalize_text`, in the `cleaned` composition only:

```
'Brämhults' -> ['bramhults']   'Côteaux Nantais' -> ['coteaux','nantais']
'Reál' -> ['real']   'Réal' -> ['real']   'REAL' -> ['real']
```

## 14. Attribute separation metrics

`src/training/attribute_separation.py`, wired into **`generate_report()`** —
the single composite writer that `train.py` actually calls. Registered in
`config/paths.yaml` + `DataConfig` as `attribute_separation_summary` /
`attribute_separation_values`; thresholds in
`evaluation.attribute_separation`; rows are pydantic
(`SeparationSummaryRow` / `SeparationValueRow`). **No parallel reporting path,
no model, no training** — it reads the labelled-pair population and the
canonical attributes.

`separation = P(attribute agrees | positive) − P(attribute agrees | negative)`,
computed over pairs where the attribute is *observable* (both-sides-empty pairs
are counted in `n_unobservable`, never silently scored as agreement).

**Executed** on all 19,918 labelled pairs (7,330 positive / 12,588 negative):

| attribute | separation | flagged weak | note |
|---|---|---|---|
| volume | **+0.837** | no | strongest signal |
| pack | **+0.584** | no | |
| package_type | **+0.253** | no | |
| sweetener | +0.062 | yes | |
| carbonation | +0.029 | yes | |
| **brand** | **0.000** | **yes** | **saturated — see 15** |
| pulp | −0.051 | yes | |
| flavor | −0.087 | yes | sharing a flavor makes a pair *more* likely negative |

1,147 values scored; 98 flagged weak; **1,007 withheld for insufficient
support** (reported with their counts, never flagged — a brand seen twice is
not a defect).

## 15. Brand separation: the diagnosis (replaces the gate-eligibility idea)

The user asked to find out *how to separate brands better*. Classification of
every brand failure mechanism, measured on real data:

| class | mechanism | pairs affected | verdict |
|---|---|---|---|
| **(a)** brand empty/absent on one side | `mode_brand` blank | **0** | not a defect here |
| **(b)** near-identical but unnormalised | case/accents/suffixes | **0 pairs** collapse today — **but the accent mechanism was real and is FIXED** (§13.3) | fixed in code |
| **(c)** genuinely different brand, other attributes match | true hard negative | **0** | does not occur |
| **(d)** brand present but drowned out | brand is **constant across the population** | **all 19,918** | **the actual finding** |

**(d) is the answer, and it is a population property, not a text bug.**
`true_label=1` pairs are gate-`proceed` pairs; `true_label=0` pairs are
gate-`hard_no` **hard negatives**. Brand agreement is **100 % in BOTH classes** —
every single one of the 19,918 pairs has the same brand on both sides
(`pair_type` for the sampled negatives: volume 1258, pack 230, package_type 8,
flavor 1). Brand is therefore **constant by construction**, separation is
exactly 0, and the model's measured **+0.0249 brand separation is the correct
response to this data**, not a model failure.

**Consequence — a training-side specification (not attempted here).** No
text-composition change can improve brand separation, because the signal is
absent from the pair population. The lever is **hard-negative mining**: admit
cross-brand pairs at a meaningful rate (the current negatives are ~100 %
within-brand), or the encoder is being trained to treat brand as noise. Stated
as a specification only — **no training was run**, and this is untouched by
this change.

## 16. Status of the five newly approved items — stated plainly

| item | status |
|---|---|
| 1. Symmetry as an enforced invariant | **DONE** — `test_symmetry_invariant_where_the_same_evidence_feeds_both_sides`, all 855 rows, byte-equal text + vector. Scoped deliberately: a raw SKU listing and a canonical record legitimately differ in wording, so the invariant is applied where the SAME data feeds both sides; the one permitted exception (the target de-duplicates the brand it already emitted) is encoded explicitly in the mirror, not by loosening the assertion. |
| 2. Truncation guard + counter in run metrics | **NOT DONE.** The finding stands (11.9 % of target texts lost the structured tail entirely; 27.7 % exceeded the window) and the remedy is unstarted. Raising `max_seq_length` is **not** free — `worker_2`'s ANN checkpoint was trained at 128 — so this needs a deliberate decision, not a silent bump. |
| 4. Band-conditioned separation | **NOT DONE.** The metric is band-ready (it takes any pair frame) but no band split is wired. |
| 5. Stage attribution (retrieval vs scoring) | **NOT DONE.** |
| replacement. Brand diagnosis | **DONE** — §15, with (a)/(b) fixed in code and (c)/(d) handed over as a training-side spec. |

Items 2, 4 and 5 remain open. I stopped rather than ship them half-verified:
the working tree was concurrently edited by another agent across ~20 files in
this same feature area (§17), and I judged an honest "not done" more useful
than code I could not verify end to end.

## 17. Concurrent edits — disclosure

Partway through this round the working tree gained edits **I did not author**,
in the same feature area: `model_input_provenance` was renamed to
`model_input_composition` and promoted to a pydantic model with a `fingerprint`
field; `_implicit_pack_qty` was made public as `implicit_pack_qty`; and
`prepared_bundle.py`, `zero_shot_sims.py`, `src/cli/colab.py`, `dvc_store.py`
and others were touched. My tests were adapted to that API rather than
reverting another agent's work, and one of my tests duplicating an existing
one was dropped. The full suite was green across the combined state
(**332 passed, 2 skipped**).

## 18. Reused vs newly created (this round)

**Reused:** `core.structured_features` (`_as_set` / `_as_string_set` / `vector`)
for attribute parsing; `pipeline.normalize_text`; `core.common` `F` registry,
`ensure_parent`, `load_config`; `generate_report` and its
`_datapoint_coverage_section` pattern; `config/paths.yaml` + `DataConfig`
registration; `TrainingSpec` nested-spec style; `tests/fixtures/
model_input_golden.json`.

**Newly created (all justified):** `src/training/attribute_separation.py` —
the reuse grep (`separation|discriminat|per_value|per_attribute|by_value`)
found only `extract_discriminative_ngrams` (n-gram IDF, not pair separation),
`_discriminative_groups` (LR groups) and `attribute_agreement_audit.py` (legacy
extractor vs NER sidecar agreement — a different domain). Nothing computed
pair-level attribute separation. Also new: `SeparationSummaryRow` /
`SeparationValueRow` / `AttributeSeparationSpec`, `_fold_accents`,
`symmetric_info`, `model_input_info`, and the two registered artifact paths.

## 19. EXECUTED vs READ (this round)

**EXECUTED:** full suite before and after (332 passed / 2 skipped); the
attribute audit over 855 rows; the accent tokenisation probes; the labelled-pair
brand classification (19,918 pairs, both classes); the separation metrics end to
end on real data and through `_attribute_separation_section`; the legacy
byte-identity re-verification (0/855×2); the pack symmetry counts
(599 unobserved rows, 579 conforming, 20 explained by an observed pack_set,
0 unexplained).

**READ only:** the concurrent agent's edits; `evaluate_models.py` and the
reporting wiring (via a read-only reconnaissance subagent, no code run).

**Not run:** any training; the real pipeline (it rewrites `results/*.csv`);
`git checkout -- results/` was therefore never needed — `results/` stayed clean
apart from the two new `attribute_separation_*.csv` files, which are gitignored
and NOT committed.

---

## 20. Brand analysis: TF-IDF + fuzzy matching — are brands failing to match, and why not?

`src/training/brand_analysis.py`. **No new dependency**: TF-IDF uses
`sklearn` (`scikit-learn==1.9.0`, already in `requirements.txt`) and fuzzy
matching uses stdlib `difflib` plus a small Levenshtein. `rapidfuzz` 3.14.6 is
present in the venv but is **not declared** in `requirements.txt`, so relying on
it would break a fresh install — not used.

TF-IDF is configured `analyzer="char_wb"`, `ngram_range=(2,3)`: character
n-grams are the right space for short strings, where whole-token overlap is too
sparse (`Radnor` / `Rainbow` share no token but share n-grams). Settings and
thresholds live in `evaluation.brand_analysis`; rows are pydantic
(`BrandPairRow`).

### 20.1 The classes, with real counts

**585-pair review band** (the population where brands actually differ —
`brand_match` is False on 388 of 585):

| class | count | share |
|---|---|---|
| `exact_match` | 197 | 33.7 % |
| `different_brand` | 378 | 64.6 % |
| `ambiguous_similarity` (0.60 < ratio < 0.85) | 10 | 1.7 % |
| `surface_variant` (case/accents/punctuation/suffix) | **0** | 0 % |
| `missing_brand` (empty either side) | **0** | 0 % |
| `brand_only_in_title` | **0** | 0 % |

**Labelled-pair population (19,918 pairs):** 19,917 `exact_match`,
1 `surface_variant`, 0 of every other class — every pair is same-brand by
construction.

**Catalog (13,250 GTINs, 1,655 distinct brands):** exactly **3** clusters where
raw brand strings collapse to one normal form — `REAL | Reál | Réal` (8 GTINs),
`ECO | Eco+` (14), `Viva | Viva!` (6) — **28 GTINs, 0.21 %**, and **0 labelled
pairs** involve two spellings of one brand.

### 20.2 The answer: brands are not failing to match — the wrong product is being retrieved

The 10 `ambiguous_similarity` cases were inspected individually. **Every one is a
genuinely different brand that happens to share a substring**, not a
normalisation failure:

```
Radnor   vs Rainbow      0.615   LIFEWTR  vs ZenWTR     0.615
Cemilefendi vs Cemil     0.625   Peace Tea vs Seven Teas 0.632
Thick- It vs Thick & Easy 0.667  Réal     vs Realemon  0.667
Albi     vs Marli        0.667
```

So of the 388 brand mismatches: **378 are outright different brands** (the
retriever surfaced a different brand's product) and **10 are different brands
sharing letters**. **Zero are string-normalisation defects.** Brand mismatch is
a **retrieval/gating** problem, not a text-composition problem — which is
consistent with §15's finding that brand is constant across the *training* pair
population and therefore carries no learnable signal.

### 20.3 What was fixed, and what it is worth

| class | fixable here? | action |
|---|---|---|
| surface variants (accents) | yes | **fixed** — diacritics folded before normalisation |
| surface variants (case/punctuation) | yes | already handled by `normalize_text` |
| surface variants (corporate suffixes) | yes | suffix stripping added to the analysis normal form |
| missing/empty brand | n/a — **0 instances** | nothing to fix; current behaviour is correct and a regression test pins it |
| brand only in title | n/a — **0 instances** | nothing to fix |
| genuinely different brands | **no** | handed over as a training-side spec (§20.4) |

**Important SSOT correction.** I originally wrote a private `_fold_accents`
helper. That was a duplicate: `core.critical_attributes.normalized_attribute_text`
is the repo's existing accent-folding normaliser (NFKD + casefold + strip
combining marks), already used by `core.hard_negatives.normalized_product_name`.
The brand analysis and `core.model_input` now **both call the existing
function**; the private copy is deleted. One consequence surfaced immediately:
that normaliser also collapses punctuation, so the decimal percentage marker
had to become alphanumeric (`5.5%` → `pct5d5`, not `pct5.5`) or it split into
two tokens. Caught by an existing test.

Scope preserved: every one of these changes is in the **`cleaned`** composition,
and `legacy` still reproduces the golden fixtures — re-verified after the
normaliser swap, **0 byte-mismatches over 855 rows × 2 sides**.

### 20.4 Handover — the training-side specification (not attempted here)

**True hard negatives.** 378 of 585 review-band pairs (and ~100 % of the
labelled-pair negatives) pair *different brands* while matching on the other
attributes. Two consequences the user's separate training run should decide:

1. **Hard-negative mining must admit cross-brand pairs.** The current negatives
   are same-brand by construction (`pair_type` = volume / pack / package_type),
   so the encoder is trained on a population where brand is constant — it
   learns that brand is noise, which is exactly the +0.0249 separation observed.
   Recommended: sample a configurable fraction of hard negatives *across* brands
   at the same volume/pack, so brand becomes a discriminating dimension.
2. **Loss weighting / margin.** With brand made informative, the contrastive
   margin (`training.contrastive_margin`, currently 0.1) is the lever that
   decides how hard same-attribute/different-brand pairs are pushed apart. This
   is a tuning decision for the training run, not a code change here.

No training was run; this is a written specification only.

## 21. Status of item 6 (provenance / contract stamping)

**Partially covered by concurrent work; the delta is NOT done.** Another agent
working this repository in parallel introduced `TrainingSpec.ModelInputComposition`
(pydantic, in `src/core/schemas.py`) carrying `profile`, `include_evidence` and a
`fingerprint` digest, and stamps it into the run trace, the checkpoint manifest,
the prepared-bundle manifest and the ANN reuse fingerprint through the existing
`core.tracing` / `config/paths.yaml` machinery.

Of the four fields item 6 asks for, **two are present** (composition profile,
fingerprint) and **two are NOT** (symmetry mode, code commit / git SHA, config
hash). I did not add them: the area was under active concurrent edit, and
shipping an unverified extension into another agent's in-flight schema is how
two schemas for one concept appear. Stated plainly as open.

**Absence is the marker for pre-change artifacts.** No historical artifact was
rewritten. A reader identifies a pre-change artifact by the *absence* of the
`model_input` block in its manifest/trace: artifacts produced before this work
carry no such key, artifacts produced after it always do. Nothing was
back-filled, so the absence is meaningful.

## 22. Reused vs newly created (this round)

**Reused:** `core.critical_attributes.normalized_attribute_text` (**the fix for
my own duplication** — see §20.3); `sklearn` TF-IDF (already declared);
`difflib` (stdlib); `core.common.F` / `ensure_parent` / `load_config`;
`config/paths.yaml` + `DataConfig` registration; the `EvaluationSpec` nested-spec
and row-model style in `core/schemas.py`.

**Newly created (justified):** `src/training/brand_analysis.py` and
`tests/test_brand_analysis.py`. The reuse grep for fuzzy/TF-IDF/brand helpers
(`SequenceMatcher|difflib|levenshtein|edit_distance|fuzz|TfidfVectorizer|tfidf`)
returned **nothing** in `src/`, so no brand-similarity machinery existed;
`attribute_agreement_audit.py` is a different domain (legacy extractor vs NER
sidecar). Also new: `BrandAnalysisSpec` / `BrandPairRow` / `BRAND_PAIR_COLUMNS`,
`catalog_brand_variants`, `review_comparisons`, `brand_support`, the
`brand_analysis_pairs` artifact path, and `_levenshtein` (no stdlib or declared
dependency provides edit distance).

## 23. EXECUTED vs READ (brand round)

**EXECUTED:** brand classification over all 19,918 labelled pairs (19,917
exact / 1 surface variant); over the 585 review band (197 / 378 / 10, with the
10 inspected individually); catalog variant clustering over 13,250 GTINs and
1,655 brands (3 clusters, 28 GTINs); the empty-brand and title-only counts
(0 and 0); the legacy byte-identity re-verification after the normaliser swap
(0 / 855×2); full suite **341 passed, 2 skipped**; ruff clean on both new files.

**READ only:** the concurrent agent's `ModelInputComposition` implementation and
its stamping sites (for §21); the `difflib` / `sklearn` APIs.

**Not run:** any training; the real pipeline; `rapidfuzz` (available but
undeclared, deliberately unused).
