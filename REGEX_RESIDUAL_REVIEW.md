# Regex review (2026-09-27)

The review uses the 61,529 rows in `data/dataset_deduped.csv` and the active
SKU fields `brand`, `title`, and `attributes`. Lexical capture is not the same
as a structured parser value. In particular, brand regex hits are diagnostic.

`results/regex_attribute_captures.json` is the compact, **complete** list of
all 287 distinct attribute capture groups, with counts and example product
IDs. `results/regex_capture_summary.json` contains all 3,595 groups across the
three working fields. These are not top-N previews. The row-level capture CSV
is available for drill-down but is much larger. Full row-level JSON is opt-in
via `--detail-json` because it is impractically large to inspect.

For product `783327667`, `results/regex_capture_783327667.json` shows exact
title and attribute lexical matches plus raw numeric spans. The raw spans
preserve `100%`, `12x355ML`, `Caffeine: 0-15 mg`, and `Juice Content: 0-2%`,
which the ordinary text normalizer would otherwise render without punctuation.
`100% Natural` is a title claim, not an attribute value; it is captured
lexically as `100 natural` but is not a structured parser class.
The same focused JSON includes `model_payload_review`, built with ER's active
`cleaned` composition and current structured tokens. That payload still has
repeated plain words; audit dedup is deliberately a separate comparison.
For this row it keeps `pct100 natural` and `pct0to2`, but the caffeine range
`0-15 mg` appears only as `caffeine 15` in the plain payload; the raw evidence
record keeps both endpoints and the unit.

The focused JSON also has a versioned `semantic_profile`: accepted parser
values are separate from raw numeric evidence and candidate-only typed values.
For `783327667`, trusted flavor is `lime`, while explicit declarations are
`lime` and `maple`; the latter is retained with its source span but does not
enter the gate or swap pipeline yet. The `sweetener_type` candidate slot keeps
declared ingredient types (e.g. `cane_sugar`, `stevia`) separate from the
existing `sweetener` claim classes (`no_sugar`, `no_added_sugar`, `sugar`,
`diet`). A sugar ingredient alongside a `no_sugar` claim gets a consistency
flag and is excluded from the current swap-compatible field list. Raw numeric
and lexical-only entries are never marked swap-eligible. This review schema
does not itself assign Semantic IDs or augment training pairs.

Attribute package-type lexical captures now require an explicit `Pack Type`
field. `can` in `can be recycled` and `carton` in `Pack Material Type` are no
longer treated as package types. The claim extractor and lexical audit now
share their sugar patterns; `0 sugar`, `0g sugar`, and the explicit typo
`no dugar` map to `no_sugar`, while `0 sugar added` maps to `no_added_sugar`.
The separate positive claim `Made with Sugar` maps to `sugar`.

`results/regex_miss_summary.json` likewise contains all 319 distinct candidate
groups. Its 25,334 row-linked candidate mentions break down as follows:

| Review reason | Mentions | Interpretation |
| --- | ---: | --- |
| Declared flavor unrecognized | 14,593 | Some are type-like values (`tea`, `latte`), others are likely vocabulary gaps (`blueberry`, `banana`). |
| Sweetener ingredient outside claim classes | 9,785 | The current claim schema has fewer classes than ingredient vocabulary (`cane sugar`, `sucralose`). |
| Title signal, parser dimension empty | 931 | Includes `bubble` carbonation and standalone `sugar`; inspect context. |
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
descriptions contain 578,547 retained tokens and no repeated stem within a
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
