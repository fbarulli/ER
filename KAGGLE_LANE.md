# KAGGLE_LANE.md — runbook for the Kaggle dataset/export transport lane

Parallel dataset/transport lane to the Colab GPU lanes. Contract + evidence:
`KAGGLE_LANE_COMPLETION.md`. NEW-files only; the colab lane is untouched.

## Ownership shape

| file | role |
|---|---|
| `src/cli/kaggle_lane.py` | the lane (packaging, upload, download, submission), dry-run by default |
| `kaggle_backend.py` | repo-root shim mirroring `colab_backend.py` |
| `src/core/schemas.py` | ADDITIVE: `KaggleSpec` + default-factory `kaggle:` root field |
| `config/training.yaml` | ADDITIVE top-level `kaggle:` block (the SSOT; no default flips) |
| `tests/test_kaggle_lane.py` | offline pins (no network, staged subprocess fakes) |

Isolation: imports only `core.common` + `core.manifest` + the existing
`scripts.format_submission.format_submission`; never `cli.colab*`, never
`training.train*`, never GPU runtime modules.

## What it does

1. **package** — a cohort CSV (`dataset.csv`, `dataset_50pct.csv`, …) becomes
   `results/kaggle_lane/<cohort>/<cohort>.kaggle.zip` + `dataset_metadata.json`
   + a `.<cohort>.receipt.json` with the measured census (rows, bytes, sha256,
   columns — measured, never hardcoded).
2. **upload** — `kaggle datasets create` (vs `version` for an existing slug)
   via the configured executable; requires `--execute`, credentials at
   `~/.kaggle/kaggle.json`, and `kaggle.slug` set. Fail-loud otherwise.
3. **download** — pull-back of the published dataset verified against the
   package receipt's sha256 (transport-identity contract).
4. **submission** — a finished prediction frame is validated through the
   existing `format_submission` SSOT into the external two-column contract
   (`sku_id,item_id` — lowercase), with a receipt (rows / unique items /
   unmatched count under the configured `unmatched_prefix`).

## Owner launch steps (live, when credentials exist)

```bash
# 1. name the dataset once in config/training.yaml (kaggle.slug: "user/slug")
# 2. package + upload (0. comparison of census vs the printed receipt first):
   PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane --what upload \
       --dataset-csv dataset_50pct.csv --execute
# 3. verify fetch-back:
   PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane --what download \
       --dataset-csv dataset_50pct.csv --execute
# 4. package a finished submission:
   PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane --what submission \
       --submission-input submission/predictions.csv \
       --submission-output submission/SKU_ITEM_submission.csv
```

Without `--execute` every command is a local dry run (this box has no Kaggle
credentials; the default never touches the network).

## Known pre-existing failure (not this lane)

`tests/test_dvc_streaming_publish.py` fails 6 tests at the lane's fork point
(902689e) — DVC was retired 2026-10-05 (HANDOFF §7) and those pins have not
been revisited on main. No dvc byte changed in this lane
(`git diff 902689e HEAD -- src/training/dvc_store.py` = empty).
