Final JEV review — rebuilt pairs, fixed item catalog

All 1,000 calls completed and their saved request hashes and raw scores were verified. 900 unique new pairs; 100 swapped-order checks. No overlap with previous tested pairs. Full comparison of old/new partner sets covers all 13,216 canonical items, including isolated items.

| Cohort | Unique pairs | Different (<=0.2) | Uncertain | Same (>=0.8) |
| --- | ---: | ---: | ---: | ---: |
| current_positive | 640 | 387 | 210 | 43 |
| current_negative | 160 | 158 | 2 | 0 |
| lost_positive_partners | 100 | 92 | 8 | 0 |

The weighted low-score fraction in the fresh eligible positive population falls from 93.6% in round 7 to 60.6% in round 8. These are different partner populations from the same catalog, sampled with recorded inclusion probabilities. This is JEV disagreement, not measured ground-truth accuracy. Previously tested pairs are excluded, so the estimates do not cover every shipped positive row.

Removed-positive checks provide no high-score evidence of a lost valid link in this sample. They do not prove that every removal was safe. Weighted diagnostics and training-population estimates remain separate.

Remaining clear examples, all still approved:

| Items | Captured titles | JEV score |
| --- | --- | ---: |
| 7391881837636 / 7398818389848 | lohilo Pink beach bcaa drink 330ml / lohilo Glow 2022 collagen containing carbonated sugar free beverage 330ml | 0.02 |
| 688267001222 / 688267001505 | Peapod Dr. Bob Soda / Peapod Cream Soda | 0.03 |
| 6415600579292 / 6415600590358 | Battery Black 0.5l box / Battery Pearberry 0.5l box | 0.02 |

These show that named formulations and flavors still disappear when the canonical comparison retains only shared generic attributes. Negative controls look sound; positive labels need further source-backed review before training. JEV opinions alone must not flip labels.

Order checks: 8/100 change score band; 5/100 have score gaps >=0.2.

No further JEV calls are scheduled. Saved evidence supports the next offline corrections and training decision. See ALL_ITEM_COMPARISON.md for every-item census, all_item_partner_sets.jsonl for actual old/new partners, and RESULTS_REVIEW.json for complete example listings.
