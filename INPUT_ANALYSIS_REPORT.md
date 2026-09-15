# Model-input analysis — what the encoder actually receives

**Revision analysed: `c0b4d352ce12698440dd8c0c7145cd5aae3cc83f`** (worktree
`/home/opc/ONE/ER-analysis-brand-input`, branch `analysis/brand-and-input`).

> The main checkout has since advanced to `27b1cb0`. That delta is provenance plumbing
> (`model_input_provenance`, `preprocessing_fingerprint_inputs`), tests and reports;
> **the composition functions are byte-identical** (`git diff c0b4d35 27b1cb0 --
> src/core/model_input.py` shows only additions). Every number here describes both revisions.

**Scope.** Corpus-level, offline measurement of the string `core.model_input` hands to the
encoder. It is deliberately **not** the productionised run-report metric a teammate is wiring
up; nothing here writes to a run tree. CPU only, no training, no GPU.

**Corpora measured** (all via `core.common.load_dataset()` and
`core.common.canonical_records_frame()` — the SSOT loaders, so the shared
`column_mapping` in `config/paths.yaml` is applied):

| corpus | rows | side |
|---|---:|---|
| `canonical_corpus` (`results/canonical_records.csv`) | 13 250 | target |
| `source_corpus` (`dataset.csv`, deduped on `product_id`) | 71 623 | source |
| `review_pairs` (the 0.55–0.75 band) | 585 | both |

**Encoder contract read from config, not hardcoded:** `max_seq_length =
core.common.runtime("max_seq_length")` → **128**; tokenizer =
`core.common.resolve_model("minilm_l6")` → `artifacts/models/all-MiniLM-L6-v2`;
`training.structured_features` → `enabled: true`, `append_to_text: true`,
`feed_to_loss: true`, `embedding_weight: 0.35`.

> **Correction made during this analysis, disclosed because it changed results.** My first
> implementation read `dataset.csv` directly. That bypasses `COLUMN_MAPPING`, so
> `row_metadata_text(row, "title")` found no `title` column and returned `""` — the source
> text silently collapsed to brand-only (~3 tokens) and every source-side number was wrong.
> The scripts now call `core.common.load_dataset()`. The brand analysis was unaffected (it
> used `brand`, which the mapping leaves unchanged, and read the title under its raw name);
> re-running it after the fix produced a **byte-identical** `brand_analysis_summary.json`.

---

## 1. Token budget against `max_seq_length: 128`

`analysis_outputs/input/token_budget.csv`

| population | side | profile | records | mean | median | p95 | max | truncated | share | tokens lost | headroom when it fits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_corpus | target | **cleaned** | 13 250 | 51.4 | 51 | 79 | 201 | **7** | **0.05 %** | 216 | 76.6 |
| source_corpus | source | **cleaned** | 71 623 | 78.4 | 76 | 119 | 229 | **1 889** | **2.64 %** | 27 478 | 51.3 |
| review_pairs | source | cleaned | 585 | 76.0 | 73 | 117 | 213 | 14 | 2.39 % | 419 | 54.0 |
| review_pairs | target | cleaned | 585 | 49.5 | 49 | 77 | 115 | **0** | **0 %** | 0 | 78.5 |
| review_pairs | source | legacy | 585 | 100.6 | 96 | 153 | 228 | 112 | 19.15 % | 2 207 | 38.5 |
| review_pairs | target | legacy | 585 | 54.8 | 54 | 83 | 121 | 0 | 0 % | 0 | 73.2 |
| review_pairs | target | **legacy+evidence** | 585 | **114.9** | 104 | **216.6** | **446** | **162** | **27.69 %** | 9 104 | 39.7 |
| review_pairs | source | legacy+evidence | 585 | 100.6 | 96 | 153 | 228 | 112 | 19.15 % | 2 207 | 38.5 |

**Headroom is real on the target side and thin on the source side.** A target record uses
51 of 128 tokens (76.6 spare); a source listing uses 78 (51.3 spare) and its p95 is 119, i.e.
the tail is already at the cap. The `cleaned` profile sits comfortably inside 128; the
pre-consolidation `legacy+evidence` composition did not — 27.7 % of target records were
truncated, losing 9 104 tokens.

### What truncation actually costs

`analysis_outputs/input/truncation_loss_by_group.csv` — tokens past the cap, attributed to the
field group that overflowed, by walking the builder's own emission order.

| population | side | group | tokens lost | truncated records attributed |
|---|---|---|---:|---:|
| canonical_corpus | target | **structured** | **216** (100 %) | 7 |
| source_corpus | source | **structured** | **27 478** (99.99 %) | 1 889 |
| source_corpus | source | attributes | 2 | 1 889 |

**Every token lost to truncation is a structured-channel token.** This is structural, not
coincidence: `brand`, `title`, `attributes` are emitted first and `[FIELD_*] volume_ml_*`
tokens are appended last (`core.structured_features.append_text`), so the discriminative
attribute channel is exactly what falls off the end. The brand/title/attributes the model
still sees are never the casualty; the volume/pack/flavour evidence is.

The 1 889 truncated source rows are the ones with the largest attribute payloads — i.e.
truncation hits hardest precisely where there is most to compare.

*Disclosed imprecision:* source attribution sums to **27 480** against the budget table's
**27 478** (+2 tokens, 0.007 %). The cumulative-prefix attribution re-tokenises group
boundaries, so a two-token merge at one boundary can migrate. The target side reconciles
exactly (216 = 216). The `truncated_records_attributed` column states how many truncated rows
were walked (capped at `TRUNCATION_ATTRIBUTION_CAP = 4096`), so the number is never silently
partial.

---

## 2. Composition — how much of the text can discriminate at all

`analysis_outputs/input/group_composition.csv`. `filler` = the share of a group's token
*instances* whose document frequency exceeds `UBIQUITOUS_DOC_FRACTION = 0.5` — a token in more
than half the corpus cannot separate two of its records. `mean idf` is the TF-IDF idf averaged
over token instances, fitted on the full corpus each group belongs to.

| population | side | group | token share | filler share | mean idf | distinct tokens |
|---|---|---|---:|---:|---:|---:|
| canonical_corpus | target | brand | **9.8 %** | 0.0 % | **7.33** | 1 846 |
| canonical_corpus | target | canonical | 35.9 % | **22.3 %** | 4.44 | 3 745 |
| canonical_corpus | target | mode_type | 1.4 % | 0.0 % | 3.76 | **5** |
| canonical_corpus | target | **structured** | **53.0 %** | **40.5 %** | **2.57** | 357 |
| source_corpus | source | brand | **5.1 %** | 0.0 % | **7.34** | 2 422 |
| source_corpus | source | title | 24.7 % | 0.2 % | 5.69 | 12 799 |
| source_corpus | source | attributes | **40.0 %** | 10.7 % | 3.51 | 891 |
| source_corpus | source | **structured** | **30.1 %** | **56.4 %** | **2.34** | 501 |
| review_pairs | source | brand | 5.4 % | 0.0 % | 7.33 | 496 |
| review_pairs | source | title | 26.8 % | 0.1 % | 5.78 | 1 329 |
| review_pairs | source | attributes | 36.0 % | 13.1 % | 3.46 | 357 |
| review_pairs | source | structured | 31.8 % | 56.3 % | 2.38 | 149 |
| review_pairs | target | brand | 10.0 % | 0.0 % | 7.44 | 486 |
| review_pairs | target | canonical | 34.5 % | 23.9 % | 4.39 | 547 |
| review_pairs | target | mode_type | 1.4 % | 0.0 % | 3.76 | 5 |
| review_pairs | target | structured | 54.0 % | 41.7 % | 2.57 | 141 |

**This is the mechanism behind the +0.0249 brand separation reported in
`BRAND_ANALYSIS_REPORT.md`.** Brand is the *most* discriminative group by idf (7.34 source /
7.33 target) and simultaneously the *smallest* by mass — **5.1 % of source tokens, 9.8 % of
target tokens**. The structured channel is the largest group and the least discriminative:
**30.1 % of source tokens at idf 2.34, of which 56.4 % are corpus-ubiquitous**. On the target
side the structured channel is **53.0 % of all tokens at 40.5 % filler**.

So two different-brand isotonic drinks of the same volume, in the same package type, with the
same flavour, share well over half their composed text, and the one token that would separate
them carries ~5 % of the mass. That is a text-composition fact, measured, and it does not
require the model to be at fault.

`mode_type` deserves a note: 5 distinct values across 13 250 records, idf 3.76, 1.4 % of mass.
It is nearly a constant — a category label the model can read but which separates almost
nothing.

---

## 3. Field presence / coverage across the corpus

`analysis_outputs/input/field_coverage.csv`. Attribute presence is taken from the composition's
own parsers (`canonical_info` / `sku_info`), **not** from the raw column text — the canonical
artifact stores an absent set as the literal string `[]`, which a naive non-empty test scores
as 100 % populated. That mistake was made and corrected here.

### Canonical corpus (target side, n = 13 250)

| field | populated | field | populated |
|---|---:|---|---:|
| **pulp** | **2.3 %** (304) | mode_flavor / flavor | 74.2 % (9 830) |
| package_type | 20.5 % (2 712) | carbonation | 81.8 % (10 843) |
| **pack** | **25.2 %** (3 342) | volume | 93.2 % (12 348) |
| sweetener | 43.9 % (5 813) | mode_brand | 100 % (13 250) |
| mode_type | 54.4 % (7 205) | salient_ngrams | 100 % |
| | | description_evidence | 100 % |
| | | breadcrumb_evidence | 100 % |

### Source corpus (n = 71 623)

| field | populated | field | populated |
|---|---:|---|---:|
| **pulp** | **1.7 %** (1 241) | flavor | 69.7 % (49 956) |
| package_type | 18.7 % (13 378) | carbonation | 78.6 % (56 267) |
| sweetener | 30.8 % (22 094) | volume | 91.0 % (65 194) |
| ~~pack~~ | **100 % — artefact, see below** | brand / title / attributes | 100 % |

**`pulp` is confirmed rare: 2.3 % of canonical records, 1.7 % of source SKUs.** Any rule that
depends on pulp agreement can only apply to ~1 record in 50, and the structured channel spends
a `[FIELD_PULP]` marker on the 98 % of rows where it is empty — a marker that is then
ubiquitous and therefore filler.

**`pack` on the source side is a measurement trap.** It reads 100 % populated against 25.2 %
on the canonical side. That is not real coverage: `core.structured_features.sku_info` uses
`{1.0}` as its explicit "no pack count observed" sentinel (documented in its own docstring),
so an unobserved pack is indistinguishable from a genuine pack of 1. Section 4 measures the
consequence.

`mode_brand`, `title`, `attributes`, `brand` and `barcode` are populated on **100 %** of both
corpora. **Input completeness is not the problem** — on the brand axis specifically, see
`BRAND_ANALYSIS_REPORT.md` §2: zero absent brands.

---

## 4. Per-field symmetry

### 4a. The structured channel across all 585 pairs

`analysis_outputs/input/structured_channel_asymmetry.csv` — a `[FIELD_*]` marker present on one
side of a pair and absent on the other is a field the comparison cannot make:

| marker | source present | target present | both | **source only** | **target only** | neither |
|---|---:|---:|---:|---:|---:|---:|
| VOLUME | 554 | 555 | 548 | 6 | 7 | 24 |
| **PACK_SIZE** | **585** | **160** | 160 | **425** | **0** | 0 |
| PACKAGE_TYPE | 137 | 124 | 56 | **81** | **68** | 380 |
| FLAVOR | 393 | 413 | 359 | 34 | 54 | 138 |
| CARBONATION | 454 | 472 | 414 | 40 | 58 | 73 |
| SWEETENER_DIET | 138 | 225 | 86 | 52 | **139** | 308 |
| PULP | 7 | 11 | 4 | 3 | 7 | 571 |

**The known unfixed defect is reproduced exactly: `PACK_SIZE` is source-only on 425/585 rows
= 72.6 %.** The source side emits `[FIELD_PACK_SIZE] pack_qty_1` on *every* row (the sentinel);
the canonical side reads `pack_set` directly and emits nothing when it is empty. The target is
never the only side.

`VOLUME` is the counter-example and shows the fix is achievable: 6 source-only / 7 target-only,
i.e. **well aligned**, because both sides derive volume through the same extractor and both
emit a marker only on real evidence.

Two further asymmetries the brief did not name, both material:

- **`SWEETENER_DIET` is 139 target-only against 52 source-only** — a 2.7:1 skew. The canonical
  record knows a sweetener the source listing never states. Any "sweetener disagrees →
  conflict" penalty would fire on the side that is simply less informed.
- **`PACKAGE_TYPE` disagrees in direction on 149 rows** (81 source-only, 68 target-only) while
  agreeing on only 56 of 585.

### 4b. The 24 rows where one product feeds both sides

`analysis_outputs/input/pair_symmetry_same_product.csv` — identified as
`source_original_product_id == target_original_product_id` (24 of 585).

| composition | mean token Jaccard | mean shared tokens | mean union |
|---|---:|---:|---:|
| before (`legacy+evidence`) | **0.5129** | 27.17 | 52.75 |
| legacy, no evidence | 0.2338 | 10.88 | 44.96 |
| **cleaned (shipped)** | **0.4646** | 12.71 | **27.33** |

And, on the same 24 rows: **22/24 emit `[FIELD_PACK_SIZE]` on the source side and not on the
target side** — the §4a defect, on the rows where both sides are literally the same product.

**Which fields differ, and why.** Even for one product the two texts are built from different
derivations: the source reads `title`/`attributes` from a retailer listing, the target reads
`canonical`/`mode_brand`/`mode_type` from a merged canonical record. `brand` agrees by
construction. The groups that differ are `title` vs `canonical` (different strings for the same
product — a retailer's phrasing against a normalised name) and the structured channel
(asymmetric per §4a). That asymmetry is the confirmed root cause and it survives in `cleaned`.

---

## 5. `cleaned` vs `legacy` at input level — what changes, what is gained, what is LOST

`analysis_outputs/input/profile_comparison.csv`. Three compositions are reported, because two
are not enough to be honest:

- **`before`** = `legacy+evidence` — the shipped pre-consolidation composition. This is what
  every reported "BEFORE" number means.
- **`legacy_no_evidence`** = today's `legacy` profile (config `include_evidence: false`).
- **`cleaned`** = the shipped default.

| metric | cleaned | before | legacy (no evidence) | Δ cleaned − before |
|---|---:|---:|---:|---:|
| true_pair_jaccard | **0.3200** | 0.1896 | 0.1873 | **+0.1303** |
| cross_pair_jaccard | 0.1190 | 0.0736 | 0.0816 | +0.0454 |
| **retrieval_margin** | **0.2010** | 0.1161 | 0.1058 | **+0.0849** |
| **identical_product_24_jaccard** | **0.4646** | **0.5129** | 0.2338 | **−0.0483** |
| mean_source_tokens | 76.0 | 100.6 | 100.6 | −24.6 |
| mean_target_tokens | 49.5 | 114.9 | 54.8 | −65.3 |
| target_truncated_records | **0** | **162** | 0 | −162 |
| evidence_channel_target_tokens | 0.0 | **60.0** | 0.0 | −60.0 |
| evidence_channel_source_tokens | 0.0 | 0.0 | 0.0 | 0.0 |

(I re-derived the teammate's `BEFORE`/`AFTER` numbers independently: 0.1896 → 0.3200 true,
0.0736 → 0.1190 cross, and 0.5129 → 0.4646 on the 24 rows. **They reproduce exactly.**)

### What is GAINED

- **Truncation eliminated on the target side.** 162 truncated records (27.7 %) → **0**. The
  evidence channel alone added 60.0 target tokens on average; dropping it brings the mean from
  114.9 to 54.8 tokens. On the source side, `cleaned` cuts truncated rows from 112 (19.1 %) to
  14 (2.4 %).
- **Retrieval margin up 73 %** (+0.1161 → +0.2010) and true-pair similarity up 69 %
  (0.1896 → 0.3200) — because the union shrank far more than the intersection, not because
  more tokens now match.
- **Underscore compounds, field-label artefacts, function words and the percentage evidence
  channel**: fixed as reported by the teammate; not re-derived here (string-level claims already
  pinned by `tests/test_model_input_contract.py`).

### What is LOST — the regression, investigated

**The 0.5129 → 0.4646 regression is real, reproducible, and fully attributable to one thing:
removal of the description/breadcrumb evidence channel.** No hand-waving needed — the
shared/union split accounts for it token by token:

| composition | mean shared | mean union | Jaccard |
|---|---:|---:|---:|
| before (`legacy+evidence`) | **27.17** | 52.75 | 0.5129 |
| legacy, evidence off | **10.88** | 44.96 | 0.2338 |
| cleaned | 12.71 | **27.33** | 0.4646 |

On those 24 rows the evidence channel supplies **27.17 − 10.88 = 16.29 of the 27.17 shared
tokens — 60.0 %**. Those rows are the same product on both sides, so the evidence channel
(free prose plus breadcrumbs) is near-duplicate content that matches itself across the two
derivations almost by construction.

Two consequences worth stating plainly:

1. **Measured against legacy without evidence, `cleaned` nearly doubles the 24-row Jaccard
   (0.2338 → 0.4646).** The "regression" only exists relative to the evidence channel.
2. **The regression does not transfer to the retrieval-relevant quantity.** The margin improves
   against *both* baselines: +0.1161 (before) and +0.1058 (legacy, no evidence) → **+0.2010**.
   And `cleaned`'s 24-row **maximum** is higher than before (0.6875 vs 0.6327) — the best
   same-product pairs got better while the mean fell, because the union shrank slightly less
   than the intersection did.

So the honest reading: **on this 24-row population the metric is degenerate** — it rewards
carrying a channel on both sides whose content is the same prose, which is exactly what a
retrieval system should not rely on. `cleaned` is better on margin, better on truncation,
better on token economy, and worse only on a metric that the evidence channel was inflating.
I am **not** claiming `cleaned` is uniformly better: the teammate's singleton-GTIN counter-result
(margin −9.3 % on that population) stands, and I did not re-derive it.

**What is unambiguously lost and *not* recovered:** the description/breadcrumb prose is gone
from the encoder input entirely. If any downstream consumer relied on the encoder having seen
that text, `cleaned` removes it — that is a real capability removal, mitigated only by the fact
that it was also the largest truncation and filler contributor.

---

## 6. Ranked recommendations

The decisive constraint: **any change to the composed text invalidates every checkpoint trained
on the previous text** (`AGENTS.local.md` §10). So the split below is not "easy vs hard" — it is
"can ship against the current checkpoint" vs "needs a new one".

### Fixable locally — no retraining, no encoder-input change

| # | Action | Expected impact | Evidence |
|---|---|---|---|
| 1 | **Stop letting a one-sided structured channel drive a conflict.** The gate consumes parsed sets, not the composed text, so a rule "a channel absent on one side cannot be a conflict" is local. `SWEETENER_DIET` is target-only on 139 rows and source-only on 52; `PACKAGE_TYPE` skews 81/68. | Blocks false conflicts on up to 139/585 = 23.8 % of pairs. Needs a false-merge audit before shipping. | §4a |
| 2 | **Add a brand-disagreement veto on the scoring path** (see `BRAND_ANALYSIS_REPORT.md` §7 rec. 3). 380/585 pairs here carry irreconcilable brands yet score 0.668 mean. | High; rule-level, so it does not touch the checkpoint. | §2, `BRAND_ANALYSIS_REPORT.md` §5 |
| 3 | **Record the model-input composition on every artifact that depends on it.** A profile switch changes the text without touching the catalog or the checkpoint, so a persisted embedding index looks reusable when it is not. | Correctness/auditability, no metric change. | `core.model_input.model_input_provenance()` (added at `27b1cb0`); also `rand_matching.preprocessing_fingerprint_inputs` |
| 4 | **Expose these thresholds in `config/training.yaml`** (block at §8). | Mechanical. | §8 |

### Requires retraining — the fix changes the encoder text

| # | Action | Expected impact | Evidence |
|---|---|---|---|
| 5 | **Fix the pack sentinel asymmetry — the single largest field defect.** `sku_info` emits `{1.0}` for an unobserved pack while `canonical_info` emits nothing, so `PACK_SIZE` is source-only on **425/585 (72.6 %)** rows and 22/24 same-product rows. Either emit nothing when unobserved, or emit the sentinel on both sides. | Highest of the text fixes: 72.6 % of the population currently compares a channel that exists on one side only. | §4a, §4b |
| 6 | **Cut or shrink the structured TEXT channel.** It is 30.1 % of source tokens / 53.0 % of target tokens, 56.4 % / 40.5 % of it ubiquitous filler, mean idf 2.34 / 2.57 — and it is **100 % of what truncation destroys**. A numeric structured vector already exists (`core.structured_features.vector`, fused at `embedding_weight: 0.35`, `feed_to_loss: true`), so the text channel is partly redundant. | Largest text-mass reduction available. Must be A/B'd — removing a channel is exactly the change that can reduce the margin, as the evidence-channel removal did on the 24 rows. | §1, §2 |
| 7 | **Raise brand's token mass, or give it a marker.** Brand is the highest-idf group (7.34) at the lowest mass (5.1 % source). | Directly targets the +0.0249 separation. Caveat: `core/model_input.py:94-104` records that literal `[BRAND]` markers were measured and *reduced* the true-vs-cross margin on two fixture populations — so the naive version is already known not to work. | §2, `BRAND_ANALYSIS_REPORT.md` §5 |
| 8 | **Train with brand-disagreement hard negatives** built from pairs that share category, volume and flavour but not brand (378 available here). | The only lever that changes what the encoder represents rather than what the text looks like. | `BRAND_ANALYSIS_REPORT.md` §7 |
| 9 | **Reconsider `max_seq_length: 128`.** 2.64 % of source rows exceed it today, and the loss is entirely the attribute channel. | Only after 5–7, since raising the cap changes every embedding. | §1 |

**What I would not do:** pad the structured channel with more markers, or add a brand marker,
without an A/B. Both add ubiquitous filler mass, and §2 shows filler is already 40–56 % of the
largest group.

---

## 7. Reused vs newly created

**Reused (called, not re-implemented):**

| existing function / module | used for |
|---|---|
| `core.model_input.build_sku_text` / `build_canonical_text` | **every string measured here.** The composition is never forked. |
| `core.model_input._normalized_tokens` | per-group token lists. Imported deliberately: re-deriving it is the duplication the project forbids. Its correctness is enforced, not assumed — `cursor_attribution` raises unless the reconstructed groups reproduce the SSOT string's own token count. |
| `core.model_input._structured_text_enabled` | whether the structured channel participates |
| `core.structured_features.sku_info` / `canonical_info` / `text_tokens` / `append_text` | field parsing and the structured token channel |
| `core.common.load_dataset()` | the SSOT dataset loader — applies `COLUMN_MAPPING` and validates the export |
| `core.common.canonical_records_frame()` | the validated canonical-record artifact |
| `core.common.resolve_model("minilm_l6")` | tokenizer path, from the model registry |
| `core.common.runtime("max_seq_length")` | the 128-token cap, from config |
| `core.common.row_metadata_text` / `metadata_text` | missing-value-safe field reads |
| `core.schemas.TrainingSpec.ModelInputSpec` | profile selection contract |
| `pipeline.jaccard_similarity` | token Jaccard, the existing SSOT |
| `transformers.AutoTokenizer` | real encoder token counts (not a word-count proxy) |
| `sklearn.feature_extraction.text.TfidfVectorizer` | idf and document frequency; no prior TF-IDF use exists in the repo (`grep` → 0 hits) |

**Newly created, with justification:**

| new thing | justification |
|---|---|
| `token_counts` | wraps the tokenizer on the SSOT model path; no existing helper returns encoder token counts |
| `group_tokens_for_source` / `group_tokens_for_target` | maps the builder's own emission order onto named groups for attribution; no existing function exposes per-group tokens |
| `cursor_attribution` + `truncation_loss` | per-group truncation attribution with a hard assertion against the SSOT string |
| `fit_idf_reference` | corpus idf/df lookup; TF-IDF had no prior use in this repository |
| `StructuredAsymmetryRow`, `TruncationLossRow`, `TokenBudgetRow`, `GroupCompositionRow`, `FieldCoverageRow`, `ProfileComparisonRow`, `ModelInputProvenance`, `ModelInputSummary` | pydantic boundary models (`AGENTS.local.md` §5). Declared in the script because this analysis may not edit `src/`; §8 carries the promotion patch. |
| `UBIQUITOUS_DOC_FRACTION`, `TFIDF_TOKEN_PATTERN`, `TRUNCATION_ATTRIBUTION_CAP`, `STRUCTURED_MARKERS`, `CLEANED_SOURCE_GROUPS`, `STRUCTURED_GROUP`, `PROFILE_*` | named constants; `TFIDF_TOKEN_PATTERN = r"\S+"` is load-bearing — sklearn's default word pattern would split `[FIELD_VOLUME]`, silently disaggregating the boilerplate this analysis is about |

`scripts/analyze_model_input.py` is new. It does not duplicate
`scripts/show_model_input_comparison.py` (which renders a per-row audit view for a human) or
`scripts/analyze_human_review_features.py` (which joins the queue to raw evidence): this script
measures distributions over the whole corpus.

---

## 8. Proposed config block (productionisation only — NOT applied)

`config/` is owned by another agent, so these remain script constants. When productionised:

```yaml
# ── model-input corpus analysis (scripts/analyze_model_input.py) ───────────
model_input_analysis:
  ubiquitous_doc_fraction: 0.5      # df above this = a token cannot discriminate
  tfidf_token_pattern: '\S+'        # must match _normalized_tokens' split()
  truncation_attribution_cap: 4096  # bounded per-row tokeniser work
  structured_markers: ["VOLUME", "PACK_SIZE", "PACKAGE_TYPE", "FLAVOR",
                       "CARBONATION", "SWEETENER_DIET", "PULP"]
```

If the pydantic models are promoted, `TokenBudgetRow`, `GroupCompositionRow`,
`FieldCoverageRow`, `TruncationLossRow`, `StructuredAsymmetryRow`, `ProfileComparisonRow`,
`ModelInputProvenance` and `ModelInputSummary` belong in `core.schemas`, following the style of
`CanonicalRecord` and `GateResult`; the script then imports them.

---

## 9. EXECUTED vs READ

### EXECUTED (commands run, output observed)

```bash
git -C /home/opc/ONE/ER-analysis-brand-input rev-parse HEAD     # c0b4d352...
git diff --stat c0b4d35 27b1cb0                                  # composition unchanged

# SSOT contract read through the loaders, not hardcoded
PYTHONPATH=src python -c "from core.common import runtime, resolve_model; ..."
  -> runtime('max_seq_length') = 128 ; resolve_model('minilm_l6') =
     .../artifacts/models/all-MiniLM-L6-v2
  -> structured_features: enabled=True append_to_text=True feed_to_loss=True
     embedding_weight=0.35
  -> AutoTokenizer loads; vocab 30522; do_lower_case=True; tokenizer.model_max_length=512

# group attribution is asserted against the SSOT builder, not assumed
PYTHONPATH=src python -c "<reconstruct groups, append structured, compare>"
  -> source reconstruction matches build_sku_text: 60/60
  -> target reconstruction matches build_canonical_text: 60/60

# the full corpus analysis (final run)
PYTHONPATH=src python -u scripts/analyze_model_input.py \
  --review training_results/0915T075044186132Z/worker_1/report/human_review_features/human_review_enriched.csv \
  --out-dir analysis_outputs/input
  -> token budget, truncation attribution, composition, coverage,
     asymmetry and profile comparison exactly as tabulated in §1–§5

# independent re-derivation of the teammate's BEFORE/AFTER numbers
PYTHONPATH=src python -c "<585-pair and 24-row Jaccard, three compositions>"
  -> 585 true: legacy+evidence 0.1896 | legacy 0.1873 | cleaned 0.3200
  -> 585 cross:               0.0736 |           0.0816 |         0.1190
  -> 585 margin:             +0.1161 |          +0.1058 |        +0.2010
  -> 24-row mean:             0.5129 |           0.2338 |         0.4646
  -> 24-row shared/union:     27.17/52.75 | 10.88/44.96 | 12.71/27.33
  -> 24-row max:              0.6327 |           0.3833 |         0.6875

# SSOT-existence greps (reuse-before-write evidence)
grep -rn "TfidfVectorizer\|tfidf" src/ scripts/ tests/ --include=*.py   # 0 hits
grep -rn "def jaccard" src/ --include=*.py        # pipeline.py:623, strip_audit.py:97
grep -n "def load_dataset" src/core/common.py     # :816, the COLUMN_MAPPING loader
```

### READ only (not executed by me)

- `MODEL_INPUT_FIX_REPORT.md` — read for its BEFORE/AFTER table and the `[BRAND]`-marker
  rationale. Its headline numbers **were** independently reproduced (above); its singleton-GTIN
  counter-result was **not**.
- `AGENTS.local.md` — project conventions (§10 blast radius shapes §6).
- `src/core/model_input.py`, `src/core/structured_features.py`, `src/core/common.py` docstrings.
- `config/training.yaml` (`training.structured_features`, `training.model_input`,
  `max_seq_length`) and `config/paths.yaml` (`column_mapping`, `models`).
- `artifacts/models/all-MiniLM-L6-v2/1_Pooling/config.json` — `pooling_mode_mean_tokens: true`
  (consistent with the known finding; not re-derived).
- `training_results/0915T063500554948Z/worker_2/report/` — listed only; ANN baseline metrics
  were not needed.

---

## 10. What I could not determine

- **No embedding-level measurement.** I did not load the checkpoint to embed text (the brief
  permits it but does not require it). Every claim here is about the *string*; the link from
  token mass to cosine is argued from the measured +0.0249 separation, not demonstrated
  end-to-end. A token-share-to-cosine causal claim would need encoding.
- **The singleton-GTIN population.** The teammate reports a −9.3 % margin regression there. I
  did not re-derive it, so I cannot say whether the §5 mechanism explains it. **Unresolved.**
- **Whether the structured text channel is net-positive.** §6 rec. 6 argues it is a large
  filler cost, but nothing here shows removing it would help — the evidence-channel removal
  is a worked example of exactly that going badly on one population. Needs an A/B.
- **Truncation attribution for the 14 truncated review-population source rows** is not in the
  artifact; only `canonical_corpus` and `source_corpus` were attributed. The corpus-level
  picture (§1) covers the same mechanism at 1 889 rows, so I did not add it.
- **`pulp` behaviour at runtime.** 2.3 % coverage is measured; whether the gate or the encoder
  does anything harmful on the other 97.7 % I did not trace.

---

## 11. Reproduce

```bash
cd /home/opc/ONE/ER-analysis-brand-input
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python scripts/analyze_model_input.py \
  --review training_results/0915T075044186132Z/worker_1/report/human_review_features/human_review_enriched.csv \
  --out-dir analysis_outputs/input
```

Artifacts written to `analysis_outputs/input/`: `model_input_summary.json` (all tables),
`token_budget.csv`, `truncation_loss_by_group.csv`, `group_composition.csv`,
`field_coverage.csv`, `structured_channel_asymmetry.csv`,
`pair_symmetry_same_product.csv` (the 24 rows), `profile_comparison.csv`.
Runtime is roughly five minutes on CPU. The script prints nothing until it finishes, so use `-u` and expect a quiet log rather than an error.
