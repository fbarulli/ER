# Regex residual review (2026-09-27)

Corpus: `data/dataset_deduped.csv`, 61,529 rows. The audit removes matched
text in four cumulative rounds: live attribute/extraction regex spans,
volume/pack cleanup, model stopwords, and diagnostic candidate phrases.
Titles and attribute blobs are counted separately. Match counts are lexical
coverage, not a precision/recall score for structured parser output.

| Pass | Title tokens left | Attribute tokens left |
| --- | ---: | ---: |
| Live regex | 307,099 | 1,541,464 |
| Volume/pack cleanup | 306,741 | 1,520,504 |
| Model stopwords | 240,128 | 454,630 |
| Diagnostic candidates | 236,225 | 390,415 |

Before each pass, a comparison run removed exact repeated phrases of at least
four tokens when the later phrase began within 64 original token positions of
an earlier one. It removed 651 raw title tokens across 100 rows and 634 raw
attribute tokens across 146 rows. At the final pass, the residual changed by
only 254 title tokens and 154 attribute tokens. Nearby exact repetition is a
minor cause of the remaining text. Distinct variants in one title, such as the
grapefruit and white-grape entries in product `575074082`, stay distinct.

The row-linked candidate review compares explicit source evidence with the
actual `sku_attribute_info` output. It generated 59,230 candidate mentions:
33,888 declared fields whose parser dimension is empty, 14,593 unrecognized
declared flavors, 9,799 sweetener ingredient mentions outside the current
claim classes, and 950 residual title signals with an empty parser dimension.
Examples include declared `Pack Type: Bottle` (20,522 rows), `Flavour:
Blueberry` (927), and `Sweetener: Cane Sugar` (2,345). These are review leads,
not confirmed errors: the flavor field also contains type-like values such as
`tea`, and the sweetener claim schema deliberately has fewer classes than the
catalog's ingredient vocabulary.

The scripts leave production regexes untouched. Reproduce the outputs with:

```sh
PYTHONPATH=src .venv/bin/python scripts/regex_residual_audit.py
PYTHONPATH=src .venv/bin/python scripts/regex_residual_audit.py --compare-repeats --out results/regex_repeat_comparison.json --rows-out results/regex_repeat_rows.csv
PYTHONPATH=src .venv/bin/python scripts/regex_miss_review.py
```

Outputs under `results/` are local generated files ignored by Git. The row
level candidate CSV includes product ID, original title/attributes, candidate
phrase, and parser value; the JSON reports aggregate counts and example IDs.
