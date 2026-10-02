# Feature extraction and gate usage — 2026-10-02

This review distinguishes extracted evidence, canonical persistence, diagnostic comparison, and branches that actually decide `three_way_gate`. Runtime configuration is the authority; comments and census rates are not accuracy estimates. No dataset or training bundle was rewritten.

## Actual decision paths

`src/pipeline.py` contains both `pack_gate` and `three_way_gate`. The latter consumes configured volume/pack conflicts, package type/material conflicts, configured categorical decision-engine verdicts, explicit sugar/pulp claim conflicts, source-conflict flags, missing/weak numeric evidence, packaging level, and numerical consistency. Decision order preserves trusted contradictions before review. `full_attribute_evaluation` computes the full registry evidence later when gate rows are written; its full-dimension output does not itself change the previously selected gate decision.

| Feature | Extraction and persistence | Gate use |
|---|---|---|
| Volume | Title, attributes, URL and image readers; selected scalar, confidence, canonical set and consistency | Configured overlap veto, confidence/consistency review, source disagreement and ambiguous-volume review |
| Count per unit | Declared count and title/URL/image count evidence; explicit canonical pack set | Configured pack veto, missing/confidence/consistency review; registry `count per unit` now correctly maps to pack, and the direct numeric pack path consumes it |
| Package type | Title NER then attributes; canonical union | Configured disjoint-set veto |
| Pack material | Title and attributes; canonical union plus universe evidence | Configured disjoint-set veto |
| Packaging level | Title case/retail claims; canonical union | Two-sided disjoint veto and one-sided review, independently of `veto_dimensions` |
| Flavor | Title/attributes, description fallback and category tokens; canonical union | Configured decision-engine veto |
| Sugar claims | Explicit positive/negative title/attribute claims; description conflict flags | Configured direct sugar-claim contradiction veto, with review for conflicting source evidence |
| Sweetener ingredients | Title/declared ingredients; canonical union and universe evidence | Registry ingredient conflict verdict maps to configured `sweetener`; ingredient negation flags review |
| Pulp | Explicit claims, description fallback and flags | Configured direct contradiction veto |
| Carbonation | Extracted and persisted | Currently excluded from `veto_dimensions`; remains diagnostic |
| Organic | Extracted and persisted | No configured veto mapping; remains evidence |
| All other registry fields | Attribute-universe evidence persisted as JSON | Full census/decision metrics, without a configured veto route |
| Price, retailer, country, source IDs | Source metadata, with some listing ledger context | No direct identity veto; price text must not invent package counts |
| Dates | Additive role-aware evidence from title, attributes, description and both category fields; original spans and normalized candidates persist through listing ledger | Stock review context; no identity veto |

## Registry coverage

The runtime registry has 37 fields. Its configured categorical gate projection includes `flavour` and `sweetener`; volume, pack and material use the direct paths above. The former `pack type`→pack mapping was a confirmed defect: current mapping is `count per unit`→pack and `pack type`→package type. Package type and pulp still have direct paths outside the categorical registry projection.

Other registered diagnostic fields are: juice content, weight, caffeine, carbonization, health claims, naturally derived, made from, water type, immune support ingredients, sustainable sourcing, no artificial ingredients, juice features, geographic origin, contains minerals, free from, energy source, diets, sports ingredients, concentrate format, RTD coffee style, tea type, sustainable packaging, environmentally friendly, botanicals and functional ingredients, coffee type, sports positioning, sports drinks style, roast type, nutri score, special edition and giftbox. `count per unit` is additionally consumed directly as pack count; its registry mapping now reflects that numeric meaning.

## Why exclusions require evidence

`config/training.yaml` documents same-GTIN noise for water type (14.9%), made from (21.2%), and caffeine (3.2%). Registry notes identify concentrate format (48.5%) and RTD coffee style (44.4%) as review channels, sports ingredients (3.6%) and coffee type (2.6%) as below the cited donor/veto floor, roast type as folded into flavor identity, and special edition/giftbox as constant at the cited measurement. Those notes describe historical observed rates; they do not independently prove safe automatic rejection.

The config prose says removing sweetener avoided 74 lost true matches on a 6,898-pair evaluation, but the current `veto_dimensions` list includes `sweetener`. The stale comment was corrected to distinguish historical measurement from current runtime policy; the actual veto list was preserved. Carbonation is absent from the actual list; the historical comment reports zero removed false merges and zero lost true matches for it.

## Confirmed repairs and remaining usage risks

Numeric configured-veto bypass and nonfinite confidence were reproduced against HEAD and repaired. Invalid confidences/consistency now review, and disabled volume/pack dimensions cannot hard reject.

Pack parser repairs preserve complete grouped-thousands quantities, exclude decimal/price tails and list ordinals, multiply nested counts, recover small word-number counts and BT bottle shorthand, and derive omitted counts only when explicit total-volume arithmetic proves it. Inner/outer hierarchy remains distinct: generic pack labels preserve bounded derived inner quantities and emit `pack_hierarchy_ambiguous` for review; explicit outer-box × sticks-per-box quantities can assert a trusted inner total. Declared volume now shares the measurement grammar, including fractional/whitespace-decimal inputs; fractional count declarations stay unknown.

Cross-listing ingredient negations were previously discarded during canonical aggregation. The canonical now intersects the aggregated negative ingredients with positive ingredients and emits `sweetener_source_conflict:<ingredient>` for gate review; it does not fabricate a positive ingredient from a negative-only listing.

A confirmed repaired usage error: a title `Vanilla water no sugar 330ml` with description `Contains sugar` generates `description_conflict:sweetener`, yet the original gate returns `proceed` against itself. Likewise `with pulp` versus description `No pulp` generates `description_conflict:pulp` and originally proceeds. The gate now reviews affected dimensions and preserves unrelated definite contradiction precedence. Explicit unsweetened-with-ingredient and sweetening-status conflict flags also trigger review. Canonical critical claim unions additionally flag both polarities across listings.

The additive listing evidence ledger preserves negative ingredient claims and measurement/count spans. Numeric selected sets/confidence are persisted as gate inputs; richer roles and original per-reader evidence are diagnostic ledger entries. All-source extraction does not imply every extracted field should automatically become a hard veto.

Date extraction now distinguishes expiry, manufacture, and unspecified dates, retaining ambiguous candidates and month/year precision as stock review context; neither absent dates nor differently captured listing dates establish product identity conflict. The root agent is measuring actual date field/text availability separately.
