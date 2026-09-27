# Regex capture and parser gaps — 2026-09-27

## Current result

The refreshed review has **55 unresolved candidate mentions across 55 products**. These are candidates, not confirmed extraction errors. They break down into **25 package ontology questions** and **30 residual title signals**. The title signals include unresolved semantics and remaining context cases.

| Remaining group | Mentions | Why it remains |
| --- | ---: | --- |
| `pack type: aerosol` (package_type) | 24 | The source declares a package type the live package ontology rejects. Aerosol occurs on ordinary beverages, so accepting it blindly could create false conflicts. |
| `pulp` (pulp) | 10 | Remaining product/brand wording or nonstandard claims need a safer distinction from actual pulp content. |
| `sugar` (sweetener) | 7 | Title mentions do not state a supported sugar claim or a known ingredient with enough confidence. |
| `bubble` (carbonation) | 4 | Product/brand wording such as Bubble Up or sparkle does not establish carbonation. |
| `sweetener` (sweetener) | 3 | Title mentions do not state a supported sugar claim or a known ingredient with enough confidence. |
| `sweeteners` (sweetener) | 3 | Title mentions do not state a supported sugar claim or a known ingredient with enough confidence. |
| `pack type: tray` (package_type) | 1 | The source declares a package type the live package ontology rejects. Aerosol occurs on ordinary beverages, so accepting it blindly could create false conflicts. |
| `effervescent` (carbonation) | 1 | Product/brand wording such as Bubble Up or sparkle does not establish carbonation. |
| `sparkle` (carbonation) | 1 | Product/brand wording such as Bubble Up or sparkle does not establish carbonation. |
| `pack of 6` (pack) | 1 | The residual count phrase still needs parser confirmation. |

## What caused the inflated queue

1. **Evidence capture was mistaken for parser assignment.** Previously, 9,785 sweetener declarations were recorded as regex misses because the audit looked for ingredient values in a field intended for sugar claims. There are now separate `sweetener_type_set` and `sweetening_set` fields that flow through extraction, canonical records, and model input.
2. **Residual cleanup joined separated text.** The audit now verifies each residual title phrase against a contiguous span in the original title and rejects cleanup-created phrases.
3. **Title words lacked product context.** N-gram context and targeted exclusions remove known boba/bubble-tea, Pulp & Press, Rocket Fizz, and effervescent-tablet contexts from the relevant claims.
4. **Descriptions can resolve clear claims.** Explicit low-sugar, no-sweetener, ingredient, carbonation, or pulp phrases populate an unknown field from description. Conflicting title/attribute and description evidence leaves the field unresolved and adds a consistency flag.

## Captures now assigned

- **Flavors:** 88 reviewed values including `tea` and `latte`, accepted from explicit `Flavour:`/`Flavor:` declarations.
- **Sweetener ingredients:** 18 declared types (including sucralose and stevia) stored separately from claims such as `no_sugar`.
- **Sweetening status:** explicit states such as unsweetened, low sugar, reduced sugar, sweetened, and no sweeteners.
- **Carbonation and pulp:** non-tablet effervescent water and explicit pulp phrases such as “juice and pulp” are recognized; “effervescent tablets” and “Pulp & Press” are treated as context, not claims.
- **Volume:** British millilitre/centilitre spelling and `1 000 ml` are parsed. The volume source still has low-unit anomalies; a parsed value is not proof the source unit is correct.

Across the full dataset, **32,668 ingredient mentions** and **632 unsweetened declarations** now have dedicated fields. 14 source rows have contradictory sweetener declarations and retain consistency flags.

## N-gram and description review

NLTK bigrams/trigrams are counted by product for the remaining title signals in `title_signal_ngrams`. Examples include:

- **carbonation**: bubble up (3), bubble up lemon (2), white sparkle (1), sparkle fl (1), white sparkle fl (1).
- **pulp**: pulp l (3), pulp l cevita (3), m pulp (2), m pulp l (2), juice m pulp (2).
- **sweetener**: sweeteners only (2), sweeteners only clean (2), real sugar (2), sweetener iced (2), luzianne sweetener (2).

Description evidence for the remaining title candidates:

- `not_needed_for_declared_field`: 25
- `no_explicit_cue`: 28
- `no_description`: 2

The remaining descriptions have no matching explicit claim cue; two candidates have no description. The parser uses descriptions only for narrow, explicit cues. This review does not certify ambiguous titles.

## Known source-data questions

- The 24 `Pack Type: aerosol` values occur on beverage records whose other packaging evidence says glass or plastic. They are held for source ontology review; they were not forced into the package parser. The single `tray` value may describe a multipack arrangement rather than the primary container.
- Several titles say `0.33 ml` while the same record’s `Volume` attribute says `330 ml`. The attribute currently wins, so this source-unit discrepancy is outside the unresolved-phrase count and should be checked before trusting volume consistency.
- Remaining cases are review candidates, not a gold-standard false-negative count. The report distinguishes them from parser-assigned values and rejected contextual phrases.

## Verification and activation

Python compilation and direct spot checks completed for the updated extraction paths. No test suite was run for this follow-up. Existing canonical and embedding artifacts have not been rebuilt; old canonical records read the new fields as unknown until regenerated from source.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python scripts/regex_miss_review.py
PYTHONPATH=src .venv/bin/python scripts/regex_miss_evidence.py
```

`results/regex_miss_summary.json` gives the current unresolved candidate count; `results/regex_miss_evidence.csv` preserves original spans and description cues. Generated files under `results/` are ignored by Git.
