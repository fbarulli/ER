# Brand analysis — are brands not matching, and why not?

**Revision analysed: `c0b4d352ce12698440dd8c0c7145cd5aae3cc83f`** (`training`, worktree
`/home/opc/ONE/ER-analysis-brand-input`, branch `analysis/brand-and-input`).

> The main checkout has since advanced to `27b1cb0`. The delta between the two revisions
> touches only provenance plumbing (`model_input_provenance`,
> `preprocessing_fingerprint_inputs`), `scripts/show_model_input_comparison.py`, tests and
> `MODEL_INPUT_FIX_REPORT.md` — **the composition functions `_cleaned_sku_text`,
> `_cleaned_canonical_text`, `_legacy_sku_text`, `_legacy_canonical_text` are byte-identical**
> (`git diff c0b4d35 27b1cb0 -- src/core/model_input.py` shows only additive changes).
> Every number below therefore describes both revisions.

**Population:** the 585-pair `0.55–0.75` review band
(`training_results/0915T075044186132Z/worker_1/report/human_review_features/human_review_enriched.csv`).

**Brand sources (SSOT, not the review CSV):**
- source brand = `dataset.csv.brand` for `SKU_ID` (bound via `core.common.F["dataset"]`) —
  this is the value `core.model_input` actually tokenises;
- target brand = `results/canonical_records.csv.mode_brand` for `NEAREST_ITEM_ID`, read
  through `core.common.canonical_records_frame()`.

The review CSV's own `canonical` column is **stale for 542/585 rows (92.6%)** against the
SSOT artifact, so it was not used for any target-side value. `canonical_brand` happens to
agree with the SSOT on all 585 rows (0 differences), but the SSOT was still read directly.

---

## 1. Verdict

**Brands are not matching, but almost never because of how they are spelled.**

388 of 585 pairs (66.3%) carry different brands on the two sides. Of those 388,
**384 share not a single token** — not one word in common. Only **4** are recoverable by any
string normalisation (case, diacritics, punctuation, spacing, corporate suffix, token order,
truncation). The remaining **380 (97.9% of mismatches) are genuinely different brands**: true
hard negatives that no normaliser can or should repair.

The defect is therefore **not a brand-normalisation defect**. The brand fields are clean and
fully populated; the matcher is retrieving different-brand products and scoring them 0.55–0.75
because brand is a small part of the text it sees. Fixing brand *strings* cannot fix this —
the fix has to be in how much weight the brand axis carries (see §7).

---

## 2. Class counts

`analysis_outputs/brand/brand_class_summary.csv` — classifier reproduced by
`scripts/analyze_brand_matching.py`.

| class | n | share of 585 | mean cosine | what it means |
|---|---:|---:|---:|---|
| `absent_both` | 0 | 0.0% | — | both brand fields empty |
| `absent_source` | 0 | 0.0% | — | source brand empty |
| `absent_target` | 0 | 0.0% | — | target brand empty |
| `identical_normalized` | **197** | 33.7% | 0.6936 | brands agree (under the repo normaliser) |
| `surface_variant_same_brand` | **0** | 0.0% | — | same brand, different spelling |
| `containment_or_truncation` | **4** | 0.7% | 0.6818 | one brand key is a prefix/substring of the other |
| `brand_present_only_in_title` | **4** | 0.7% | 0.6922 | brand recoverable from the other side's product name |
| `distinct_brand_true_hard_negative` | **378** | 64.6% | 0.6682 | genuinely different brands |
| `partial_overlap_needs_review` | **2** | 0.3% | 0.6856 | partial token overlap, human judgement needed |
| **total** | **585** | 100% | | |

Three of the classes the brief anticipated are **empty, and the emptiness is falsifiable**:
`scripts/analyze_brand_matching.py --selftest` proves all nine classes are reachable from 15
synthetic cases (`Côteaux`/`Coteaux`, `S.A. Dampt`/`SA Dampt`, `Quellbrunn GmbH`/`Quellbrunn`,
`Coca Cola`/`Cola Coca`, `Mont Roucous`/`Mont`, `Cemilefendi`/`Cemil`, the three absence
cases, `Albi`/`Marli`, `River City`/`City River Drink`, `Ting`/`Dg`). The detector works; the
population contains no instances.

### The normalisation ladder is empty

Every step that would make two differently-spelled brands compare equal was measured
separately. None of them fires on this population:

| normalisation step | pairs newly resolved |
|---|---:|
| casefold + strip | 197 (already equal) |
| + NFKD diacritic folding | **+0** |
| + punctuation / spacing flattening | **+0** |
| + corporate-suffix stripping (`GmbH`, `S.A.`, `Group`, …) | **+0** |
| + token-order insensitivity | **+0** |
| + prefix/substring containment | **+4** |

Diacritics are the instructive case. **29 source brands carry a diacritic** (`PureThé`,
`José Cuervo`, `Ramlösa`, `Çamlica`, `Içim`, `Côteaux Nantais`, …), **3 019/71 623 source
SKUs (4.2%)** and **619/13 250 canonical records (4.7%)** have a diacritic-bearing brand, and
**34/585 review pairs carry a diacritic on exactly one side** — but folding resolves **zero**
of them, because in all 34 the two brands are different brands anyway. The diacritic exposure
is real and widespread; it is simply not what is causing these 388 mismatches.

---

## 3. Worst offending brand pairs, with support

Ranked by character TF-IDF cosine (both vectors fitted on the union of the two brand columns,
so idf describes this population's brand vocabulary). `n` = rows in the review population
carrying that exact brand pair; `src n` / `tgt n` = corpus-wide rows carrying each brand.

| source brand | target brand | fuzzy ratio | token-set ratio | TF-IDF cos | n | src n | tgt n | class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Mont Roucous | Mont | 50.0 | **100.0** | **0.4993** | 1 | 63 | 8 | containment (prefix) |
| Cemilefendi | Cemil | 62.5 | 62.5 | **0.4451** | 2 | 110 | 2 | containment (prefix) |
| Anna's Best | Cool Best | 50.0 | 61.5 | 0.3929 | 1 | 29 | 66 | **hard negative** |
| Thick- It | Thick & Easy | 66.7 | 76.9 | 0.3874 | 2 | 99 | 25 | partial overlap |
| LIFEWTR | ZenWTR | 61.5 | 61.5 | 0.2165 | 1 | 125 | 8 | **hard negative** |
| Fruit Fast | Frutti | 62.5 | 50.0 | 0.1678 | 1 | 28 | 13 | **hard negative** |
| Peace Tea | Seven Teas | 63.2 | 63.2 | 0.1382 | 2 | 232 | 10 | **hard negative** |
| Monchique | Mondariz | 47.1 | 47.1 | 0.0930 | 1 | 24 | 8 | **hard negative** |
| Aksu Vital | Akmina | 50.0 | 50.0 | 0.0863 | 1 | 171 | 7 | **hard negative** |
| Sir Up | Hip Syrups | 50.0 | 50.0 | 0.0768 | 1 | 51 | 17 | **hard negative** |
| Oca | A SHOC | 44.4 | 44.4 | 0.0743 | 1 | 23 | 7 | **hard negative** |
| Produits U | Fruity King | 47.6 | 47.6 | 0.0624 | 1 | 33 | 35 | **hard negative** |
| Obsesso | Coldpress | 50.0 | 50.0 | 0.0574 | 1 | 146 | 16 | **hard negative** |
| Radnor | Rainbow | 61.5 | 61.5 | 0.0570 | 1 | 100 | 81 | **hard negative** |
| Clever | Premier | 46.2 | 46.2 | 0.0540 | 1 | 44 | 121 | **hard negative** |
| Bullit | Bragulat | 57.1 | 57.1 | 0.0485 | 1 | 82 | 2 | **hard negative** |
| Venom | NOS | 50.0 | 50.0 | 0.0457 | 1 | 58 | 29 | **hard negative** |
| Réal | Realemon | 66.7 | 66.7 | 0.0302 | 1 | 102 | 4 | containment (prefix) |
| Albi | Marli | 66.7 | 66.7 | 0.0273 | 1 | 115 | 110 | **hard negative** |
| Fior di Loto | Pingo Doce | 45.5 | 36.4 | 0.0214 | 1 | 23 | 7 | **hard negative** |
| PureThé | Fresh Tea | 50.0 | 50.0 | 0.0154 | 1 | 23 | 12 | **hard negative** |
| ESI | Savia | 50.0 | 50.0 | 0.0000 | 1 | 22 | 8 | **hard negative** |

Highest-impact pairs **by support** (`brand_mismatch_pairs_by_support.csv`) — the ones worth
acting on first:

| source brand | target brand | n | mean ratio | mean cos | mean score | src n | tgt n |
|---|---|---:|---:|---:|---:|---:|---:|
| Big 8 | Peapod | **4** | 0.0 | 0.0000 | 0.6820 | 104 | 96 |
| Everfresh | Xyience | 2 | 25.0 | 0.0000 | 0.6716 | 33 | 19 |
| North Coast | Central Market | 2 | 40.0 | 0.0170 | 0.7048 | 48 | **1** |
| BOB | Prosain | 2 | 20.0 | 0.0000 | 0.7131 | 85 | 8 |
| Peace Tea | Seven Teas | 2 | 63.2 | 0.1382 | 0.6118 | 232 | 10 |
| Rougemont | Bravo | 2 | 28.6 | 0.0000 | 0.6466 | 41 | 34 |
| Farm Boy | Rieme | 2 | 30.8 | 0.0000 | 0.7222 | 143 | 7 |
| Thick- It | Thick & Easy | 2 | 66.7 | 0.3874 | 0.6856 | 99 | 25 |
| Cemilefendi | Cemil | 2 | 62.5 | 0.4451 | 0.6922 | 110 | **2** |
| Mr. Fitzpatrick's | Fiovana | 2 | 26.1 | 0.0534 | 0.7196 | 34 | 3 |

`North Coast → Central Market` and `Cemilefendi → Cemil` are the cautionary cases: the
**target** brand has corpus support of **1** and **2** rows. A "fix" aimed at those two pairs
would be fitting two rows. The two pairs with genuinely large joint support are
`Big 8 / Peapod` (n=4, 104 vs 96 corpus rows) and `Peace Tea / Seven Teas` (232 vs 10).

---

## 4. The 8 non-hard-negative rows, enumerated

These are the only rows any normalisation or field-recovery work could touch. Listed in full
so nobody has to trust an aggregate.

**Containment / truncation (n=4)** — `containment_or_truncation`

| SKU_ID | target GTIN | source brand | target brand | rule | partial ratio | char cos | score | src n | tgt n |
|---|---|---|---|---|---|---|---:|---:|---:|
| 125342983 | 8680462025654 | Cemilefendi | Cemil | prefix truncation | 100.0 | 0.4451 | 0.7105 | 110 | 2 |
| 517371316 | 8024884248299 | Mont Roucous | Mont | prefix truncation | 100.0 | 0.4993 | 0.6797 | 63 | 8 |
| 125267583 | 8680462025654 | Cemilefendi | Cemil | prefix truncation | 100.0 | 0.4451 | 0.6739 | 110 | 2 |
| 67878005 | 5410233710228 | Réal | Realemon | prefix truncation | 100.0 | 0.0302 | 0.6631 | 102 | 4 |

**Brand recoverable only from a product name (n=4)** — `brand_present_only_in_title`

| SKU_ID | target GTIN | source brand | target brand | rule | char cos | score |
|---|---|---|---|---|---|---:|---:|
| 963809178 | 855323002596 | Stewart's | River | target brand inside source title | 0.0000 | 0.7231 |
| 726315819 | 858629001065 | Ting | Dg | source brand inside target title | 0.0609 | 0.7112 |
| 907683998 | 3272030001235 | Green's | Moulin de Valdonne | source brand inside target title | 0.0000 | 0.6786 |
| 552423542 | 850027473291 | Mix | VUE | source brand inside target title | 0.0000 | 0.6559 |

**Partial overlap needing judgement (n=2)** — `partial_overlap_needs_review`

| SKU_ID | target GTIN | source brand | target brand | token-set ratio | char cos | score | src n | tgt n |
|---|---|---|---|---:|---:|---:|---:|---:|
| 78046727 | 696850061607 | Thick- It | Thick & Easy | 76.9 | 0.3874 | 0.7162 | 99 | 25 |
| 956091711 | 695145142519 | Thick- It | Thick & Easy | 76.9 | 0.3874 | 0.6550 | 99 | 25 |

`Thick-It` (Kent Precision Foods) and `Thick & Easy` (Hormel) are **different companies**;
"thick" is a shared generic category word, not a shared brand token. These two should be
classified as hard negatives, not repaired.

Only **4** of the 388 mismatches share *any* brand token at all: `thick` (×2),
`best` (`Anna's Best` / `Cool Best`), `mont` (`Mont Roucous` / `Mont`). The other **384 share
nothing**.

---

## 5. Impact quantification

From `analysis_outputs/brand/brand_analysis_summary.json`:

| quantity | value |
|---|---:|
| population | 585 |
| brand agrees under the crude casefold rule | 197 (33.7%) |
| brand disagrees | 388 (66.3%) |
| **resolved by normalisation** | **4** |
| **remains a true hard negative** | **380** |
| resolved ÷ population | **0.68%** |
| resolved ÷ mismatches | **1.03%** |
| hard negatives ÷ mismatches | **97.94%** |

So: **4 cross-brand candidate pairs would be resolved by normalisation; 380 would remain true
hard negatives.** A perfect brand normaliser — one that folded every diacritic, every
corporate suffix, every punctuation and spacing variant, every token order and every
truncation — would move 0.7% of this population. There is no string-level fix here.

### The score separation, reproduced

| group | n | mean cosine |
|---|---:|---:|
| brand agrees | 197 | 0.6936 |
| brand disagrees | 388 | 0.6687 |
| **separation** | | **+0.0249** |
| (of which) true hard negatives | 380 | 0.6683 |

The known **+0.0249** is reproduced exactly. The hard negatives alone sit at 0.6683 — i.e.
products whose brands have *nothing* in common still score within 0.001 of the cross-brand
mean and only 0.025 below same-brand pairs. That is the quantitative statement of the defect:
**the encoder text barely encodes brand.**

This is not a paradox. `INPUT_ANALYSIS_REPORT.md` measures the composed text directly and
finds brand contributes only a small share of its tokens, while category, volume, flavour and
the `[FIELD_*]` boilerplate markers are shared by a large fraction of the corpus. Two
different-brand isotonic drinks of the same volume have almost identical composed text
except for one or two tokens.

---

## 6. Latent defect found (not the cause here, but real)

`src/training/rand_matching.py:207-210` — the gate's brand-equality normaliser:

```python
def _normalize_brand(value: object) -> str:
    """Normalize a brand for equality without treating missing as a value."""
    normalized = unicodedata.normalize("NFKC", metadata_text(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())
```

It uses **NFKC**, which *composes* rather than *decomposes* accents, so the combining-mark
strip its three sibling normalisers all perform is missing here:

```
  'Côteaux Nantais'    _normalize_brand='côteauxnantais'     NAT-derived='coteauxnantais'
  'PureThé'            _normalize_brand='purethé'            NAT-derived='purethe'
  'Ramlösa'            _normalize_brand='ramlösa'            NAT-derived='ramlosa'
  'Lanjarón'           _normalize_brand='lanjarón'           NAT-derived='lanjaron'
  Coteaux == Côteaux under _normalize_brand: False
  Coteaux == Côteaux under NAT-derived key: True
```

`ô` is `isalnum()`, so it survives. **`_brand_conflict("Coteaux", "Côteaux")` returns `True`** —
the gate reports a brand conflict for a pure diacritic variant. `core/attribute_conflicts.py:45` and `core/critical_attributes.py:44` do
NFKD + `not unicodedata.combining(char)` directly, and `core/hard_negatives.py:40-43`
delegates to `normalized_attribute_text` (so it inherits the correct behaviour);
`_normalize_brand` is the outlier.

**Measured impact on this population: zero resolved pairs** — the 34 single-sided-diacritic
review pairs are different brands regardless. It matters at corpus scale (3 019 source SKUs /
619 canonical records carry diacritic brands), where it can veto a legitimate merge. It is a
latent correctness bug, not the explanation for the 388 mismatches, and I am **not** claiming
otherwise.

**Proposed patch** (`src/training/rand_matching.py`, replace lines 207-210) — deliberately
the same idiom already used by the three sibling normalisers:

```python
def _normalize_brand(value: object) -> str:
    """Normalize a brand for equality without treating missing as a value.

    NFKD + combining-mark strip, matching core.attribute_conflicts /
    core.critical_attributes / core.hard_negatives.  NFKC (the previous form)
    leaves ``Côteaux`` != ``Coteaux``, so a pure diacritic variant was reported
    as a brand conflict by the targeted veto gate.
    """
    normalized = unicodedata.normalize("NFKD", metadata_text(value)).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return "".join(character for character in normalized if character.isalnum())
```

Any change here invalidates nothing trained (it is gate logic, not encoder input), but it does
change `targeted_brand_conflict` for diacritic pairs — so it needs a regression test and a
re-run of the hard-negative mining counts before landing.

---

## 7. Ranked recommendations

### Fixable locally (no retraining)

| # | Action | Expected impact | Evidence |
|---|---|---|---|
| 1 | **Do not invest in brand string normalisation.** Fix the `NFKD` bug in `_normalize_brand` (§6) because it is a latent correctness defect, then stop. | **≤0.68%** of this population; **4 rows**. Explicitly *not* a solution to the separation defect. | §5 |
| 2 | **Make the brand axis explicit in the composed text.** Brand occupies a small token share and no structural marker; `[FIELD_*]` markers exist for volume/pack/type but none for brand. Give brand a marker and/or repeat it, so its token mass matches its decision weight. | Requires an A/B on the encoder to size; **cannot be quantified from strings alone** — stated, not guessed. Note `core/model_input.py:94-104` documents that literal `[BRAND]` markers were already tried and *reduced* the true-vs-cross margin, so this must be re-tested, not assumed. | `INPUT_ANALYSIS_REPORT.md` §2 |
| 3 | **Add a brand-conflict veto on the scoring path**, mirroring gate logic that already exists (`rand_matching._brand_conflict`, `targeted_brand_conflict`). 380/585 pairs here carry irreconcilable brands and the encoder still scores them 0.668. | High — it is a rule, not a model change; the whole point is that the encoder cannot separate them. Needs a false-merge audit against true pairs. | §3, §5 |
| 4 | **Recover the 4 title-only brands** by reading brand from the product name when the brand field disagrees. | 4 rows (0.7%). Low value; listed for completeness. | §4 |
| 5 | **Add the proposed `config/training.yaml` block** (§10) so the thresholds stop being script constants. | Mechanical; no behaviour change. | §10 |

### Requires retraining

| # | Action | Why |
|---|---|---|
| 6 | **Train with explicit brand-disagreement hard negatives** — pairs that share category, volume and flavour but not brand (378 of them are already sitting in this population). | This is the only path that changes what the encoder represents. No amount of text normalisation substitutes for it (§5). |
| 7 | **Only after (6), reconsider any change to the composition.** Any change to the encoder input contract invalidates every checkpoint trained on the previous text (`AGENTS.local.md` §10). | Sequencing, not a separate task. |

**What I would not do:** build a corporate-suffix list, a diacritic folding table, or a
fuzzy-brand-similarity gate as a general fix. This population contains 0 suffix variants and
0 diacritic-resolvable pairs; the measured ceiling for all of that work combined is 4 rows.

---

## 8. Reused vs newly created

**Reused (called, not re-implemented):**

| existing function / module | used for |
|---|---|
| `core.common.F`, `canonical_records_frame()` | SSOT paths and the validated canonical-record artifact |
| `core.common.metadata_text` | missing-value-safe field reading |
| `core.critical_attributes.normalized_attribute_text` | **the** brand normaliser (NFKD + casefold + combining strip + punctuation flattening). The compact brand key is this output with separators removed — the only added step. |
| `pipeline.jaccard_similarity` | token Jaccard, the existing SSOT |
| `core.schemas.TrainingSpec.ModelInputSpec` | profile selection contract |
| `training.rand_matching._normalize_brand` | imported **only to demonstrate its NFKD gap in §6** |

**Newly created, with justification:**

| new thing | justification |
|---|---|
| rapidfuzz (`ratio`, `token_set_ratio`, `partial_ratio`, `Levenshtein.distance`) | required by the brief. `grep -rn "rapidfuzz\|fuzz\." src/ scripts/ tests/` returns **0 hits** — no prior use, nothing to reuse. |
| `sklearn.feature_extraction.text.TfidfVectorizer` | required by the brief. `grep -rn "TfidfVectorizer\|tfidf" src/ scripts/ tests/` returns **0 hits** — no prior use. |
| `CORPORATE_SUFFIX_TOKENS` | `grep -rni "gmbh\|\bsarl\b\|corporate" src/ config/` returns **0 hits** — no corporate-form vocabulary exists. Declared so the class is *testable*; it fires on **0** population rows. |
| `compact_brand_key`, `spaced_brand_text`, `strip_corporate_suffix`, `brand_tokens` | thin compositions of `normalized_attribute_text`. **Not** a second normaliser: no NFKD/combining/casefold logic is written here. |
| `BrandPairClassifier` rule ladder + `BrandClass` | the classification the brief asks for; no existing equivalent. |
| pydantic boundary models (`BrandPairRecord`, `BrandClassSummary`, `BrandImpact`, `BrandAnalysisProvenance`, `BrandAnalysisSummary`) | `AGENTS.local.md` §5 wants pydantic at boundaries. They live in the script because this analysis is forbidden from editing `src/`; the patch below moves them to `core.schemas`. |

`scripts/analyze_brand_matching.py` is new; no existing script computed brand classes
(`scripts/analyze_human_review_features.py` computes only a crude `casefold().strip()`
equality, which this analysis supersedes for the brand axis).

---

## 9. EXECUTED vs READ

### EXECUTED (commands run, output observed)

```bash
# setup and revision
git worktree add -b analysis/brand-and-input /home/opc/ONE/ER-analysis-brand-input training
git -C /home/opc/ONE/ER-analysis-brand-input rev-parse HEAD      # c0b4d35...
git -C /home/opc/ONE/EuromonitoR rev-parse HEAD                  # 27b1cb0 (advanced)
git diff --stat c0b4d35 27b1cb0 -- src/core/model_input.py       # additive only

# join integrity and staleness (the SSOT decision)
#   NEAREST_ITEM_ID -> canonical_records.gtin                         585/585
#   review.canonical text differs from SSOT canonical                 542/585
#   review.canonical_brand differs from SSOT mode_brand               0/585
#   dataset sku_id duplicated rows                                    0/71623
#   sku_id carrying >1 distinct brand                                 0/71623
#   canonical mode_brand empty                                        0/13250
#   source brand empty on review SKUs                                 0/585

# SSOT-existence greps (reuse-before-write evidence)
grep -rn "rapidfuzz\|fuzz\." src/ scripts/ tests/ --include=*.py          # 0 hits
grep -rn "TfidfVectorizer\|tfidf" src/ scripts/ tests/ --include=*.py     # 0 hits
grep -rni "gmbh\|\bsarl\b\|corporate" src/ config/                        # 0 hits
grep -rn "unicodedata\|NFKD" src/ --include=*.py                          # 4 normalisers found
grep -rn "def jaccard" src/ --include=*.py                                # pipeline.py:623, strip_audit.py:97

# the analysis itself
PYTHONPATH=src python scripts/analyze_brand_matching.py --selftest
  -> [selftest] all brand classes reachable
PYTHONPATH=src python scripts/analyze_brand_matching.py \
    --review training_results/0915T075044186132Z/worker_1/report/human_review_features/human_review_enriched.csv \
    --out-dir analysis_outputs/brand --worst-top 25
  -> commit: c0b4d352...; population: 585 pairs
  -> counts and impact as tabulated in §2 and §5

# the latent-diacritic defect, and the corpus exposure
PYTHONPATH=src python -c "<_normalize_brand vs NAT-derived key on Côteaux/PureThé/Ramlösa/Lanjarón>"
  -> Coteaux == Côteaux under _normalize_brand: False ; under NAT-derived key: True
  -> source SKUs with diacritic brand 3019/71623 ; canonical 619/13250
  -> review pairs with a diacritic on exactly one side 34/585
```

### READ only (not executed by me)

- `MODEL_INPUT_FIX_REPORT.md` — the teammate's report. I read its §5 population table, its
  "24 rows, 0.5129 → 0.4646" claim and its `[BRAND]`-marker rationale. I did **not** re-derive
  its Jaccard numbers here; the input analysis does that independently.
- `AGENTS.local.md` — project conventions.
- `src/core/model_input.py` docstrings (the `[BRAND]`-marker result and the marker rationale).
- `config/training.yaml` `training.model_input` block and its comments.
- `artifacts/models/all-MiniLM-L6-v2/1_Pooling/config.json` (`pooling_mode_mean_tokens: true`)
  and `artifacts/models/all-MiniLM-L6-v2/sentence_bert_config.json`
  (`max_seq_length: 256`, `do_lower_case: False`). Both **read**; the effective cap is the
  config SSOT `training.max_seq_length: 128`, which I read via `core.common.runtime`.
- `training_results/0915T063500554948Z/worker_2/report/` — listed only; the ANN baseline
  metrics were not needed for the brand question.

---

## 10. Proposed config block (productionisation only — NOT applied)

`config/` is owned by another agent, so the constants above are script-level. When this is
productionised, add to `config/training.yaml` and read through `core.common.load_config()`:

```yaml
# ── brand-axis analysis (scripts/analyze_brand_matching.py) ────────────────
brand_analysis:
  containment_partial_ratio_min: 95.0   # partial_ratio floor for truncation
  containment_min_chars: 4              # shortest key allowed to "contain"
  title_recovery_min_chars: 3           # shortest key creditable to a title
  distinct_token_set_max: 70.0          # below this = distinct-brand hard negative
  char_ngram_range: [2, 5]              # TF-IDF character n-grams
  word_ngram_range: [1, 2]              # TF-IDF word n-grams
  min_df: 1
  corporate_suffix_tokens: ["gmbh", "sarl", "sas", "sa", "srl", "spa", "ltd",
                            "limited", "inc", "llc", "plc", "bv", "nv", "ag",
                            "kg", "kgaa", "ab", "as", "oy", "aps", "sro", "co",
                            "company", "corp", "corporation", "group", "groupe",
                            "grupo", "holding", "pte"]
```

If the pydantic models are promoted, `BrandPairRecord` / `BrandClassSummary` /
`BrandImpact` / `BrandAnalysisProvenance` / `BrandAnalysisSummary` belong in `core.schemas`
alongside `CanonicalRecord` and `GateResult`, following their style; the script then imports
them instead of declaring them.

---

## 11. What I could not determine

- **Whether the 380 hard-negative pairs are truly non-matches.** The review population is
  defined by a *score band*, not by human labels. "Different brand" is strong evidence of a
  true negative but is not a label. Every claim above is a string-level claim, and I have
  not converted it into an accuracy claim.
- **The score-level effect of any fix.** Without training or re-encoding I cannot say how
  much cosine separation a brand veto or a brand marker would buy. I have deliberately not
  estimated it.
- **Whether a `[BRAND]` marker helps or hurts.** `core/model_input.py:94-104` records that
  markers were measured and *reduced* the true-vs-cross margin on two fixture populations.
  That is a READ result from the teammate's measurement, not mine, and it argues against the
  naive version of recommendation 2. Recommendation 6 (retraining with brand-disagreement
  hard negatives) is the path I am confident in.
- **Two rows of the population carry a target brand with corpus support of 1–2**
  (`North Coast → Central Market`, `Cemilefendi → Cemil`). For those, target-side brand
  quality is suspect and I cannot tell whether the canonical record's `mode_brand` is wrong
  or the pair is genuinely cross-brand. Flagged, not resolved.

---

## 12. Reproduce

```bash
cd /home/opc/ONE/ER-analysis-brand-input
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python scripts/analyze_brand_matching.py --selftest
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python scripts/analyze_brand_matching.py \
  --review training_results/0915T075044186132Z/worker_1/report/human_review_features/human_review_enriched.csv \
  --out-dir analysis_outputs/brand --worst-top 25
```

Artifacts written to `analysis_outputs/brand/`:
`brand_pair_classification.csv` (585 rows, per-pair), `brand_class_summary.csv`,
`brand_worst_pairs.csv`, `brand_mismatch_pairs_by_support.csv`, `brand_analysis_summary.json`.
