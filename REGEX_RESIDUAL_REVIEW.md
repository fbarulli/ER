# Regex review (2026-09-27)

The review uses the 61,529 rows in `data/dataset_deduped.csv` and the active
SKU fields `brand`, `title`, and `attributes`. Lexical capture is not the same
as a structured parser value. In particular, brand regex hits are diagnostic.

`results/regex_attribute_captures.json` is the compact, **complete** list of
all 291 distinct attribute capture groups, with counts and example product
IDs. `results/regex_capture_summary.json` contains all 3,590 groups across the
three working fields. These are not top-N previews. The row-level capture CSV
is available for drill-down but is much larger. Full row-level JSON is opt-in
via `--detail-json` because it is impractically large to inspect.

For product `783327667`, `results/regex_capture_783327667.json` shows exact
title and attribute lexical matches plus raw numeric spans. The raw spans
preserve `100%`, `12x355ML`, `Caffeine: 0-15 mg`, and `Juice Content: 0-2%`,
which the ordinary text normalizer would otherwise render without punctuation.
`100% Natural` is a title claim, not an attribute value; it is captured
lexically as `100 natural` but is not a structured parser class.

`results/regex_miss_summary.json` likewise contains all 319 distinct candidate
groups. Its 25,367 row-linked candidate mentions break down as follows:

| Review reason | Mentions | Interpretation |
| --- | ---: | --- |
| Declared flavor unrecognized | 14,593 | Some are type-like values (`tea`, `latte`), others are likely vocabulary gaps (`blueberry`, `banana`). |
| Sweetener ingredient outside claim classes | 9,799 | The current claim schema has fewer classes than ingredient vocabulary (`cane sugar`, `sucralose`). |
| Title signal, parser dimension empty | 950 | Includes `bubble` carbonation and standalone `sugar`; inspect context. |
| Declared field, parser dimension empty | 25 | 24 `Pack Type: aerosol`, one `Pack Type: tray`. |

`Pack Type: Bottle` no longer appears in the declared-field miss category:
the extractor now uses declared package type when the title has none. The
attribute lexical capture has 24,863 `bottle` spans; lexical counts do not
assert structured acceptance for every span.

The residual audit applies five cumulative diagnostic passes: live lexical
regexes, legacy volume/pack cleanup, model stopwords, candidate phrases, and
per-product stem uniqueness in `brand`, `title`, `attributes` order. The last
pass gives `unique_description` in `results/regex_residual_rows.csv`; English
singular/plural variants such as `electrolyte` and `electrolytes` share a key,
while the first observed spelling remains visible. The 61,529 combined
descriptions contain 578,200 retained tokens and no repeated stem within a
product. The JSON audit is a ranked summary, not a full row dump. Full rows
can be emitted as JSON with `--rows-json-out`, but this is opt-in due to size.

The nearby-repeat comparison drops a later exact phrase of at least four
tokens if it begins within 64 original token positions of an earlier copy.
It found 651 raw title tokens and 634 raw attribute tokens in such repeats.
This is a diagnostic comparison; no production cleaning behavior is changed
by the audit script.

Reproduce with ER's uv environment:

```sh
PYTHONPATH=src .venv/bin/python scripts/regex_residual_audit.py
PYTHONPATH=src .venv/bin/python scripts/regex_residual_audit.py --compare-repeats --out results/regex_repeat_comparison.json --rows-out results/regex_repeat_rows.csv
PYTHONPATH=src .venv/bin/python scripts/regex_capture_review.py
PYTHONPATH=src .venv/bin/python scripts/regex_capture_review.py --inspect-product-id 783327667 --inspect-out results/regex_capture_783327667.json
PYTHONPATH=src .venv/bin/python scripts/regex_miss_review.py
```

Optional audit dependencies are listed in `requirements-audit.txt`. The files
under `results/` are generated locally and ignored by Git.
