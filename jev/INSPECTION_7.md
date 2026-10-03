JEV round-7 source inspection

Inspected 16 surviving approvals: ten low-score examples across different brands, three intermediate-score cases, and three high-score controls. Selection prioritized high similarity and is diagnostic, not representative.

| Pair | Score | Finding | Evidence |
| --- | ---: | --- | --- |
| 3272030008234 / 3272036007460 | 0.04 | Clear declared variant difference | Anise syrup versus exotic-fruit syrup. Both canonical flavor sets collapse to fruit; title-specific anise/exotic differences remain outside the comparison. |
| 3274490970274 / 3274490970281 | 0.02 | Clear declared flavor difference | Cassis/blackcurrant versus anise. Descriptions name blackcurrant concentrate versus natural anise flavor, yet both canonical flavor sets are empty. |
| 3292481350010 / 3292482350026 | 0.05 | Confirmed category contamination | Lemon-lime versus lemon-only lemonade. Removing category inputs changes the lemon-only source flavor sets to lemon alone; categories introduce lime, and sometimes berry/fruit. The merged sets become identical. |
| 3361730001253 / 3361730001352 | 0.07 | Clear carbonation-strength difference | Lightly sparkling versus strongly sparkling 6 x 500ml water. Both carbonation sets contain only carbonated; strength disappears from structured comparison. |
| 4004191908448 / 4004191908806 | 0.13 | Distinct named formulation; likely different | Immune Strong versus Morgenstark. Descriptions distinguish zinc/vitamin C formulation from multivitamin juice, while both flavor sets are empty. |
| 4032108138688 / 4032108138800 | 0.07 | Distinct named herbal variant | Musta versus Khadira herbal drinks, both 500ml. Product-specific names survive in canonical text but neither record has a recognized flavor or product type. |
| 4088600388816 / 4088600388823 | 0.19 | Explicit texture difference missed | Smooth orange juice versus orange juice whose description explicitly says With Bits. Both pulp sets are empty despite the description. |
| 4101130003445 / 4101130003544 | 0.16 | Source contradiction; identity requires review | Medium versus Natural mineral water. Titles suggest differing carbonation variants, but captured attributes mark both still; do not infer the exact intended carbonation from names alone. |
| 4260107220831 / 4260107223177 | 0.02 | Clear product-type difference; confirmed category contamination | Stevia cola versus mate tea. Removing categories removes cola from the mate record. Other Non-Cola Carbonates contributes cola despite the negation. Both canonical mode_type fields are empty. |
| 4260183210122 / 4260183212072 | 0.19 | Ambiguous named coconut variant | Ordinary organic coconut drink versus king-coconut drink. Titles and descriptions distinguish king coconut, but translation and source reliability warrant review; low JEV score alone does not prove a mismatch. |
| 192397070268 / 363756405971 | 0.76 | Plausible same product below reporting threshold | Ozarka spring water 3 litre versus 3 L. Brand, water type, volume, and titles align; score 0.76 falls below the arbitrary 0.80 same band. |
| 2020003787690 / 2020005429079 | 0.63 | Plausible match with source differences | Identical Romerquelle Emotion apricot/elderflower titles, both 250ml. Sweetener attributes differ (fructose versus fructose/sugar); score 0.63 warrants review rather than a categorical mismatch. |
| 231302714337 / 747519475737 | 0.26 | Possible mixed pack versus single variant | Blenheim Spicy Sampler versus Hot, both 12 x 12oz. Sampler may indicate a mixed selection; source text is insufficient to establish equivalence. |
| 100141564826 / 100142347862 | 0.93 | Strong source match | Identical Starbucks vanilla Frappuccino titles and descriptions, both 4 x 9.5oz. JEV score 0.93 is consistent with the source evidence. |
| 642709037477 / 736983891518 | 0.83 | Strong source match | Identical Hals black-cherry sparkling-water titles and descriptions, both 24 x 20oz. JEV score 0.83 aligns with identity; some captured packaging fields differ or are missing. |
| 686082455671 / 689128697950 | 0.87 | Strong wording-equivalent match | Bare Nature peach vitamin iced tea, 12 x 20oz, expressed as 12 Pack versus 12 ct. JEV score 0.87 aligns with matching product details. |

Confirmed extraction ablations: removing categories strips lime from every lemon-only Rieme listing and strips cola from Fritz-mate. Cassis flavor and the With Bits pulp wording remain unrecognized even without categories.

Priorities supported by these examples: prevent broad/negated categories creating specific product identity; preserve declared named variants and carbonation strength; extract explicit pulp/texture statements; do not treat agreement on generic fields or jointly missing fields as sufficient identity evidence. Keep source contradictions and uncertain sampler/king-coconut interpretations available for review.

Low similarity-score disagreement cannot be equated with gate error universally: Ozarka is a plausible exact match at JEV 0.76, and Romerquelle may also match at 0.63. Identical/equivalent Starbucks, Hals, and Bare Nature controls receive high scores.

Complete original listings and selected structured fields are preserved in inspection_7.json. No gate policy changed and no additional API calls made.
