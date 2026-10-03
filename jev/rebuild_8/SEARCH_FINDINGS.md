Remaining positive-label failures: source and code search

Searched all 640 current-positive pairs in the final audit. Inspected full source listings for concrete examples and retained the 43 high-scoring matches as controls. No new JEV calls; this search changes no labels.

| Signal | Different <=0.2 | Uncertain | Same >=0.8 |
| --- | ---: | ---: | ---: |
| Conflicting extracted carbonation | 41 | 10 | 0 |
| No extracted flavor on either side | 212 | 48 | 12 |
| Description flavor language plus incomplete flavor sets | 46 | 7 | 2 |
| Different populated mode_type values | 39 | 25 | 1 |
| Similarity 1.0 despite different compound tokens | 127 | 83 | 28 |

These flags overlap. They identify review work, not independent error counts.

1. The pair similarity explicitly removes every token containing an underscore (src/pipeline.py, within-brand gate-and-similarity loop). Named product identity often survives only in these discriminative compound tokens. Lohilo Pink Beach BCAA versus Glow collagen becomes similarity 1.0 after those tokens are removed. Compatibility of shared generic attributes then supplies no positive proof of exact identity. Removing the filter alone is unsafe: 28 high-scoring matches also have differing compounds, often describing catalog metadata.

2. Carbonation is absent from configured targeted_veto_gates.veto_dimensions (config/training.yaml). Still versus sparkling therefore can remain approved. Strathmore Twist raspberry/apple still versus sparkling and Deep River Rock still versus sparkling are direct title-supported examples. Route unresolved or conflicting source evidence to review before considering any stronger policy; structured carbonation also contradicts titles in some records.

3. Bar-le-Duc records expose a second carbonation failure: the title says "without carbonic" while attributes say "carbonated". The shared extractor does not recognize that title phrase, so the canonical record retains only carbonated and has no consistency flag. Shared titles elsewhere in that record do not resolve the contradictory listing.

4. Description flavor claims are deliberately omitted by extract_description_claims (src/core/critical_attributes.py). This avoids interpreting every ingredient as the advertised flavor, but also discards explicit "flavored" and "taste of" declarations. Lohilo descriptions distinguish flavor and functional formula while both extracted flavor sets are empty. Capture explicitly anchored product flavor language as review evidence; do not import arbitrary ingredients.

5. Named formulations and product families remain outside decisive identity evidence: Peapod Dr. Bob versus Cream Soda; Battery Black versus Pearberry; lemon soda versus lemon tea drink; gingerbread syrup versus chai concentrate. The current declared_identity rules cover a small phrase list. Product-type modes are too weak to veto indiscriminately: one high-scoring match also has differing modes. Preserve source-grounded named variant, subtype and functional-formula evidence.

Do not solve this with a blanket missing-flavor veto, arbitrary title-token mismatch, or automatic rejection of compound differences. The controls contain valid-looking plain waters, reordered titles and catalog aliases that those shortcuts would withhold.

Recommended implementation order: source-aware carbonation review and contradiction capture; explicit description-flavor/formulation capture; named product identity comparison with documented alias handling. Validate offline against the saved final scores, all high-score controls and the complete item/partner census. No additional JEV run is needed.

SEARCH_FINDINGS.json retains the full captured listings for eight inspected examples and source hashes. The remaining_* JSON files hold all flagged examples, including control overlaps.
