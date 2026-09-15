# Model-input fix: report and evidence

**Scope.** The user's instruction was *"make sure that code behaves as expected, training will be done separately."*
This work therefore fixes and proves the **model-input text construction**. It makes **no claim about model
quality or score deltas** — no training was run, no GPU was used, and every number below is measured at the
**string level** on real committed artifacts.

Repo `/home/opc/ONE/EuromonitoR`, branch `training`. Baseline suite: **290 passed, 2 skipped**.
After: **309 passed, 2 skipped**. The `cleaned` composition is the **shipped default**; `legacy` remains
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

**290 passed, 2 skipped → 309 passed, 2 skipped.** No pre-existing test needed changing:
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

New: **`tests/test_model_input_contract.py`** (19 tests) — legacy byte-equivalence over all 855 fixture
rows on both sides; the shipped default is pinned to cleaned; the no-argument path resolves to it;
legacy stays selectable as the fallback; the contradiction is rejected; evidence ablation; compound
splitting; number and percentage preservation; evidence excluded on both lanes; one normalizer across
lanes; no literal markers; margin improvement; true-pairs-far-above-cross-pairs; different brands stay
distinct; `title_only` variant semantics; both lanes call the shared builder.

New fixture: **`tests/fixtures/model_input_golden.json`** (855 records, 2.0 MiB), captured from the
**unmodified** code before any edit — 585 review-band pairs, 120 dataset rows, 150 singleton-GTIN rows.

Files changed:

```
config/training.yaml          | +17   model_input block
src/core/schemas.py           | +26   TrainingSpec.ModelInputSpec
src/core/model_input.py       | new   the shared builder
src/pipeline.py               | -50/+... third call site consolidated
src/predict_items.py          | -42      scoring lane consolidated
src/training/rand_matching.py | -44      training lane consolidated
tests/test_model_input_contract.py | new
tests/fixtures/model_input_golden.json | new
MODEL_INPUT_FIX_REPORT.md     | new
```

---

## 7. EXECUTED vs READ

**EXECUTED** (all CPU, no training, no GPU):

* `pytest tests/ -q` before (290 passed / 2 skipped) and after (309 / 2) the change, including a run
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
