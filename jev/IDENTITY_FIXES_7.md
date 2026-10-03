JEV identity fixes — offline validation

Implemented: remove category-derived flavors; recognize anise/cassis/blackcurrant/exotic flavor vocabulary; capture juice With Bits and smooth texture; preserve source-grounded flavor additions, carbonation strength, cola/mate family and named arishta variants as review evidence. The shared review contract is wired into both three_way_gate and the SKU/canonical targeted inference gate. Explicit configured vetoes retain priority; missing declarations require review, not invented rejection.

Rebuilt features for 1,364 canonical records from original sources and replayed the same 900 unique round-7 pairs. No additional JEV calls and no full candidate census or training-label rewrite.

| Prior → new gate | Pairs |
| --- | ---: |
| fallback->fallback | 217 |
| hard_no->hard_no | 300 |
| proceed->fallback | 167 |
| fallback->hard_no | 14 |
| fallback->proceed | 27 |
| hard_no->fallback | 6 |
| proceed->proceed | 148 |
| proceed->hard_no | 21 |

Of 336 total prior approvals (335 stored positives plus one stored negative), 167 move to review and 21 to rejection. Of 275 low-scoring prior approvals, 185 cease being approved and 90 remain. All six high-scoring prior approvals remain approved. Removing spurious category flavors also releases 27 prior review pairs: 18 score different and nine uncertain, so category removal alone does not solve identity approval. The new approval group totals 175, including 108 low-score pairs.

Inspected regressions now blocked/reviewed: anise/exotic and cassis/anise syrup, lemon-lime/lemon lemonade, light/strong sparkling water, Musta/Khadira, cola/mate, smooth/with-bits orange juice, and the medium-labelled water pair. Matching Starbucks, Hals, Bare Nature, Ozarka and Romerquelle controls remain approved.

Golden model-input changes are limited to 11/855 SKU examples gaining newly recognized flavor tokens and any needed FIELD_FLAVOR marker. Prior bytes and review notes are retained per affected record. Canonical golden bytes remain unchanged.

Validation: new regression coverage plus attribute, schema, extraction, model-input and inference suites. The historical frozen-population ANN test remains a pre-existing failure: it assumes all committed proceed pairs contain no current conflicts, while 6,443 are rejected by the already-existing flavor policy (documented in AUDIT_NOTES.md before these changes). It was not re-pinned to stale artifacts.

Additional unresolved issues are documented in ADDITIONAL_FINDINGS_7.md: source measurement disagreement, nested pack counts, named variants, configured carbonation policy and mode-flavor false review. The shipped CSV census and labels still require a controlled regeneration before training on updated features.
