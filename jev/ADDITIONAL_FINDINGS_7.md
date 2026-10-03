Additional findings from saved JEV round-7 results

Inspected 12 additional cases using original listings, stored features, saved JEV scores and the offline feature replay. No new API calls.

| Pair | JEV score | Finding | Evidence |
| --- | ---: | --- | --- |
| 8437007759273 / 8437007759433 | 0.03 | Conflicting sizes inside one identity record | 200ml versus 1l titles. The second canonical unions five 1000ml rows and one 200ml row into {200,1000}; overlap with the first canonical {200} permits approval. No source-conflict flag is set. |
| 721867271111 / 721867277267 | 0.03 | Nested retail-pack counts lost | Titles and descriptions assert 4 x 250ml sold as pack of 2 versus pack of 4, as well as Lite versus Original. Extraction records the outer counts but selects pack_qty=4 for both, with no hierarchy flag. |
| 5060510930018 / 5060510930124 | 0.03 | Within-record pack conflicts treated as compatible | Pu-Erh versus Linden kombucha; one barcode has 5/6/7-pack listings and the other 5/7. Intersection permits approval despite conflicting listings and distinct named tea variants. |
| 5060128500801 / 5060128501105 | 0.03 | Cultivar identity collapsed | Gala versus Braeburn apple juice. Both collapse to apple. Source pack counts also vary (1/2/12 versus 2/3/5/12); their overlap is accepted. |
| 876063811910 / 876063811927 | 0.03 | Named commercial flavor missing | Cool Blue versus Glacier Freeze across six listings per product. Both flavor sets are empty while generic energy-drink fields agree. |
| 8690327892222 / 8690327892420 | 0.02 | Functional formulation identity missing | Women Fit versus C Mix vitamin water, both 500ml. Source functional ingredients differ, while flavor sets are empty and the gate approves. |
| 8710624308360 / 8710624345631 | 0.03 | Light versus regular variant absent | First Choice cola regular versus light, both 500ml. Both sweetening and sweetener sets are empty; shared cola does not distinguish the variant. |
| 8693354001032 / 8693354001230 | 0.03 | Spicy versus plain turnip variant absent | Titles explicitly distinguish plain/simple from spicy/hot turnip juice, both 2l. Flavor sets are empty and ingredient sets broadly overlap. |
| 6410270009353 / 6410270009384 | 0.03 | Untranslated compound names and declared-input loss | Marjex titles differ as mustikkamehujuoma versus puolukkamehujuoma. One source explicitly says Made From: blueberry, but neither canonical has flavor evidence. Exact intended translation is not externally verified here. |
| 6415712509408 / 6415712509422 | 0.03 | Captured carbonation conflict excluded by policy | 6 x 500ml spring water: first carbonation set is {still}, second {carbonated}; gate proceeds because carbonation is absent from configured veto_dimensions. |
| 894357002042 / 894357002844 | 0.88 | Mode flavor overrules equal complete sets | Super Fruit 7 1l versus 33.8fl oz. Full flavor sets agree, but mode_flavor is cherry versus fruit, influenced by per-listing generic wording; review persists despite JEV 0.88. |
| 642709089940 / 642709101987 | 0.79 | Matching-looking titles with contradictory pack descriptions | Identical Ingrilli organic lemon squeeze 4oz pack-of-6 titles; first description lists Unit Count 12, second Unit Count 6. JEV 0.79 is plausible uncertainty, not a demonstrated model mistake. |

Structural flags across the 900 saved unique pairs: five saved approvals involve a record with incompatible observed volumes; ten approvals have disjoint known carbonation sets; sixty approvals have empty flavor sets on both sides; fourteen review pairs have equal full flavor sets but different mode_flavor. These are diagnostic flags, not independently proven error counts.

Order sensitivity: the largest score gaps are 0.57 versus 0.31 for the multilingual apple/pineapple/lime/ginger/mint pair, and 0.50 versus 0.72 for the May Tea green-tea/mint pair. Unequal source-listing counts, translations and incomplete descriptions accompany these cases; the saved results do not isolate a causal explanation for order sensitivity.

Priorities: (1) conflicting source measurements and nested pack counts, (2) named product variants beyond ingredient lexicons, (3) captured carbonation differences excluded by policy, (4) lossy mode_flavor causing unnecessary review. These extend the first set of category, syrup-flavor and pulp fixes.

The identity inspection also corrects a tempting interpretation: identical Ingrilli titles do not establish a clean match because descriptions disagree on unit count. Smart Juice is a stronger candidate for needless review, although external product verification was not performed.

Detailed source evidence: additional_findings_7.json. No extra matching-policy changes were made specifically for these additional cases.
