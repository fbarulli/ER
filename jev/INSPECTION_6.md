Paired JEV round-6 inspection, 2026-10-02

Inspected all 10 category changes and all approved pairs consistently low under both formats. No new live calls or gate changes.

| Pair | Gate | Processed scores | Original scores | Finding |
| --- | --- | --- | --- | --- |
| 8693354001223 / 8693354004231 | proceed | 0.48 / 0.33 | 0.04 / 0.04 | Named variant missing from structured comparison |
| 352154336161 / 811130031631 | proceed | 0.66 / 0.64 | 0.92 / 0.82 | Likely wording/typo equivalence |
| 4104450005571 / 4104450005588 | proceed | 0.13 / 0.36 | 0.05 / 0.05 | Carbonation strength collapsed |
| 856472002055 / 856472002086 | proceed | 0.12 / 0.26 | 0.05 / 0.03 | Category contamination plus flavor subset approval |
| 7310867561402 / 7310867562706 | fallback | 0.15 / 0.23 | 0.06 / 0.06 | Captured pulp contradiction, correctly routed to review |
| 5705010079224 / 5705010080176 | proceed | 0.08 / 0.29 | 0.05 / 0.07 | Multiword flavor qualifier lost |
| 868235000451 / 868235000475 | hard_no | 0.15 / 0.11 | 0.26 / 0.23 | Measurement/source contradictions, not a clear false rejection |
| 8711900018874 / 8711900018935 | fallback | 0.06 / 0.26 | 0.05 / 0.06 | Ingredient/category bag obscures declared variant |
| 7313619000181 / 7313619001201 | fallback | 0.15 / 0.24 | 0.12 / 0.06 | Recipe evidence differs; reporting incomplete |
| 752697964546 / 758918255318 | proceed | 0.87 / 0.79 | 0.89 / 0.88 | Threshold crossing, little evidence of extraction failure |

The ten changes comprise six approved pairs, three fallback pairs, and one rejected pair. Two approved pairs look like wording-equivalent matches (Twix/Twiix and identical Proud Source titles), rather than extraction failures. The other four approvals expose spicy/plain variants, carbonation strength, blood-orange specificity, and category-contaminated flavor containment. Three fallback cases are already held for review; the rejected Rise case contains contradictory volume and sugar claims.

Both-format low approvals: 26. Flavor relationships: {'flavor_subset': 11, 'empty_flavor_on_at_least_one_side': 13, 'equal_flavor_sets': 2}. These are not all lexical misses; missing evidence and permissive containment both contribute.

Offline category ablation reproduced category-created flavor tokens. Savia Original loses coconut when category inputs are removed; Wicky loses coffee. Taika Matcha remains classified as coffee in the saved processed snapshot, but this ablation did not isolate its source. This confirms the broad-category promotion path rather than merely assuming it from the title. Detailed before/after records are in inspection_6.json.

Priority candidates for a future fix:

1. Stop broad category/breadcrumb terms becoming specific flavor identity evidence.
2. Separate declared product flavor/variant from ingredients and keep phrase-level distinctions (blood orange, matcha, plain/spicy, carbonation strength).
3. Route added specific flavor and unresolved measurement contradictions to review instead of approving on any subset/intersection.
4. Preserve readable source titles and source provenance so typo-equivalent matches remain possible.

The evidence supports targeted investigation. It does not justify declaring every JEV disagreement a gate error or turning every missing field into a conflict.
