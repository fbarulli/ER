# Dataset loading review — 2026-10-02

Scope: latest eight commits through `28a2f5a`, plus existing working changes.
Three agents reviewed raw data flow, training inputs, and performance. Existing
data/configuration edits were preserved; generated datasets were not rebuilt.

## Raw source and handoff

`config/paths.yaml` binds `files.dataset` to `repo:dataset.csv`.
`core.common.DATA_PATH` resolves to `/home/opc/ONE/ER/dataset.csv`.
The inspected file has 54,787,809 bytes, 71,623 rows, and all 13 declared columns.
The public raw loaders check the configured row census and SHA-256.

`load_raw_export()` retains source column names for
`training.data_prep -> pipeline.run_within_brand_pipeline -> identity corrections
-> eligible-GTIN filtering -> grouped canonical evidence -> gates`.
`load_dataset()` renames all columns through the configured mapping for
`training.dedupe -> identity corrections -> tiered dedupe -> dataset_deduped.csv`.
Training loads the deduped artifact; canonical/gate/label CSVs are separate inputs
that join this flow later. An explicit training dataset must have deduped column
names, so the raw CSV cannot substitute directly.

A full parse measured about 0.9 seconds and 56.2 MiB of dataframe memory.
The brand-only projection measured 0.381 seconds and 1.14 MiB versus 1.009
seconds and 56.2 MiB for the full load; values and vocabulary were identical.
Chunking followed by concatenation would retain the same final memory and add
overhead. Column projection is useful for the raw brand vocabulary reader; the
main preparation and dedupe stages need all columns and global grouping.

## Missing values and incorrect merges

The existing pandas parser recognizes missing-value tokens even with `dtype=str`.
The actual export contains 41,545 `NA` GTINs, 1,420 `NA` prices, and 33 `None`
description cells. It also has 19 missing titles. Missing-GTIN conversion is
part of current guard/accounting behavior; a global parser-policy change needs
its own evidence review.

Confirmed bug: title-based T2/T3 dedupe grouped absent titles together.
Committed `sku_to_rep.csv` maps Amazon products `69869228`, `80350488`,
`466263333`, and `500150117` to representative `28003`, despite different brands
and attributes. MatHem juice products `905519389`, `905525613`, `905561811`,
`905632962`, and `905771390` similarly map to representative `63078`.
The fix preserves missing-title rows in title-based tiers. Trusted-gtin T1
continues to establish identity independently of title availability.
Existing generated CSVs still contain the old mappings until preparation is rerun.

Descriptor completeness also counted floating NaN as a populated string and
could fail on pandas NA. Scalar and vectorized scoring now exclude missing and
whitespace-only fields consistently. This can change which listing survives;
the old generated representative mappings were not rewritten here.

## Consolidation and configuration

All three dedupe tiers use `core.deduplication.collapse_representatives` for
stable ranking, grouping, and source-row lineage. T2 gtin agreement uses the
built-in grouped `nunique` reduction. Batch completeness operates by descriptor
column and avoids allocating a dictionary for every source row.

`config/paths.yaml` now declares a typed `dataset_csv_read` contract used by
raw, projected, deduped, and training-override loaders. Existing string dtype
and default NA policy are preserved. Override validation derives canonical
columns from the column mapping rather than reading a second CSV header;
reviewed identity filtering runs once per load. Descriptor columns and all ten
reviewed malformed-gtin keep/collapse decisions are also YAML-owned,
with duplicate/unknown-field checks and review reasons. Product-identity
relative volume tolerance reads the existing gate config instead of restating
its numeric value.

## Other fixes from the review

- Prepared-bundle cache identity includes the labels, canonical records, and gate
  CSV bytes actually frozen into the bundle.
- Prepared training rejects an explicit guardrail profile that would otherwise
  be silently ignored.
- Gate replay keeps global drift separate from its stage display filter, passes
  replayed verdicts into per-sample reporting, and bounds outstanding chunk work.
- Missing previous labels can be recovered from git using a repository-relative
  path and explicit repository working directory.
- ANN mining gathers each embedding group once and uses views for query chunks.

## Remaining review findings

- Deduped training does not independently verify artifact lineage against the
  current raw export. The raw-source checks apply when raw loaders run.
- ANN directed top-k followed by lower-index filtering can omit a pair retrieved
  only from its higher-index endpoint. Changing this changes the candidate diet.
- ANN candidates still accumulate before final target/cap selection.
- One-sided low-confidence pack evidence can hit the early blocker before the
  later confidence fallback. This requires gate/census review before changing.

Regression and equivalence checks cover loading/projection, missing-title dedupe,
bundle identity, guardrail handling, replay drift, and chunked mining. Existing
model-input, column, URL, volume, and categorical gate suites were also run.
