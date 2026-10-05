# KAGGLE_LANE_COMPLETION.md — reconstruction + completion contract

Written 2026-10-05 in `/tmp/opencode/er-kaggle` (branch `kaggle-lane` @ 902689e,
exclusive worktree). Every claim cites evidence. Status markers: [x] done,
[~] partial (contract states what "done" means), [ ] open.

## 0. Lane definition reconstructed from evidence

Named "kaggle-lane" by the orchestrator (lane intent was NOT in writing). The
evidence base for what this lane is:

- **Repo groundwork already in main**: `scripts/format_submission.py`
  (2a15852, "external submission contract" — lowercase `SKU_ID`,`ITEM_ID`
  two-column CSV, missing/duplicate `ValueError`s) and
  `submission/SKU_ITEM_submission_run_note.md` (a full-deduped-cohort
  top-1-cosine assignment run at threshold 0.80 via
  `src/training/rand_matching.py`, the `er-rand-match` entry in
  pyproject.toml, config SSOT `rand_matching:` block in config/training.yaml
  with `output_dir: "submission"` and `unmatched_prefix: "UNMATCHED_"`).
  The repo's end product is a retail item-resolution submission graded
  externally — the same surface a Kaggle competition consumes.
- **Parallel lanes**: the box runs one lane per worktree/agent (TRAINING_LANE_
  COMPLETION.md, COLAB_CPU_PREP_PARITY.md on main record two already-adopted
  sibling lane contracts authored the same way as this file). The Colab GPU
  lane is `src/cli/colab.py` + `colab_backend.py`; the CPU bundle lanes are
  `src/cli/colab_bundle.py` (standalone, 8ddc614→902689e) and
  `src/cli/colab_data_bundle_prep.py` (ruling 8, main 65a26d1 — after this
  tree's fork point 902689e).
- **Dataset/transport context**: the repo transports prepared inputs by
  immutable archives (tar.zst via `core.portable_archive`, Git-published via
  `model_tracks.publish.push_artifacts`; DVC retired 2026-10-05 per HANDOFF
  §7). A Kaggle dataset/competition lane is the transport counterpart for
  pushing/pulling cohort exports and prepared artifacts via Kaggle's dataset
  API — the same role `git publisher + tar.gz` plays for Colab today.
- **cohort CSVs present in the box**: `dataset_50pct.csv` (49,225 rows,
  /tmp/opencode/er-50pct + repo root) alongside full `dataset.csv` — the
  bundle lanes already re-pin on-VM `audit.source_export_expected_rows/
  _sha256` to ANY uploaded cohort (99c7ce8). Evidence the cohort/transport
  half of this lane is a real, in-progress production concern.

### What the lane is (definition)

> The **Kaggle lane** is the dataset/export/competition transport lane:
> package and upload this repo's cohort exports and prepared artifacts to
> Kaggle datasets (and retrieve them back under hash identity), parallel to
> the Colab GPU lane, reusing — never modifying — the repo's existing
> preparation and submission machinery. It is the DVC-replacement transport
> candidate for Kaggle-hosted artifacts (HANDOFF.md §7 leaves publication
> transport TBD).

### Explicitly OUT of scope (owner rulings)

- No edits to working colab lanes: `src/cli/colab.py`,
  `src/cli/colab_bundle.py`, `src/cli/colab_data_bundle_prep.py`,
  `colab_backend.py` (owner reminder, standing rule 8).
- No defaults flip anywhere; no config key loses its current value.
- No new DVC runs (owner 2026-10-05); Kaggle is the transport, not DVC.
- No diet-gate change, no `negative_supply.mode` flip (TODO owner rulings).
- `colab/colab_retrieved/` and `colab_cli_state/` are CONSUMED evidence
  (existing Colab-lane state), not surfaces this lane builds. colab_cli_state/
  does not exist as a tracked dir in this branch checkout (launcher lock
  files are gitignored); colab_retrieved/ ships only live_status.json
  (the 2fdd44e "tree ships code, not data" ruling untracked the heavy CSVs).

## 1. Completion contract (what "done" means per surface)

All code is NEW files only (ruling 8 precedent: one lane file per surface +
one test file + additive SSOT config block + additive schema spec). Config
and schema additions are the SSOT for every name-like value (no literal
paths/magic fractions in code).

### S1 — `src/cli/kaggle_lane.py` Kaggle dataset transport lane [~]

New standalone lane file (copy-assembled from cli.colab's proven transport
is NOT needed — it uploader is the kaggle CLI/API, not the Colab notebook
transport). Surfaces:

- `package_export(csv_path) -> Path`: package a cohort export
  (dataset_50pct.csv / dataset.csv / testing.csv fixture) into a Kaggle-
  dataset upload payload (zipped csv + staged `dataset_metadata.json`),
  recording rows + sha256 (the same census shape the audit pins use, from
  `core.common._validate_source_export`-measured values, NOT hardcoded).
- `upload_dataset(csv)`: drive `kaggle datasets create/version` via
  subprocess with the packaged archive; fails loudly on missing
  credentials (`~/.kaggle/kaggle.json`), never silent-falls.
- `download_dataset(slug, destination)`: hash-verified fetch back
  (sha256 checked against the packaged manifest written at upload time).
- `main()` argparse: `--dataset-csv`, `--slug`, `--dee` … minimal flag set;
  config-owned values read from the new `config/training.yaml kaggle:` block.
- DONE = module imports dry; a --dry-run end-to-end packaging run on the
  testing.csv fixture (36 rows) produces the staged archive + metadata +
  printout of rows/sha without network; unit tests pin package/hash/fail-
  loud paths (no network in tests).

### S2 — submission packaging surface [~]

`scripts/kaggle_submission.py` (new): validate/format a finished
`submission/SKU_ITEM_submission.csv`-shaped prediction frame into the
external contract using the EXISTING `scripts/format_submission.py`
`format_submission()` (imported — never copied), plus a provenance writer
(rows, unique items, unmatched count with the config `unmatched_prefix`).
DONE = pytest over validation/fail-loud paths + a golden small-fixture run
writing a real two-column CSV.

### S3 — SSOT + schema (additive only) [~]

- `src/core/schemas.py`: new `KaggleSpec` (extra=forbid) + one additive
  root field `kaggle` on `TrainingConfig` (default-factory ⇒ existing YAML
  without the block still validates; no default flip).
- `config/training.yaml`: new top-level `kaggle:` block ONLY (dataset slug,
  export paths list, metadata fields, submission contract keys) — all
  VALUES mirrored from the existing SSOT meanings (e.g. export default =
  `dataset.csv` per paths.yaml `dataset:` binding comment; submission dir =
  existing `rand_matching.output_dir` value "submission"). I am the single
  config writer in this tree.

### S4 — doc surface [x drafted → finalize at end]

This file + a `KAGGLE_LANE.md` runbook (mirrors COLAB_CPU_PREP_PARITY.md
shape: ownership table, isolation boundary, owner launch steps) committed
at the end state.

### S5 — tests [~]

`tests/test_kaggle_lane.py` (new): pins S1 packaging/validation/fail-loud,
S2 contract validation. No network in tests (kaggle CLI call sites faked
via subprocess monkeypatch — same staged fake precedent as
tests/test_colab_bundle_delivery_root.py / test_colab_cpu_prep_parity.py
which use staged fakes only).

### S6 — verification + push (standing) [ ]

- PYTHONPATH=src .venv/bin/python -m pytest tests/test_kaggle_lane.py
  tests/test_colab_bundle_delivery_root.py tests/test_colab_stream_status.py
  -x -q → green.
- Existing colab-lane test files pass (no regression).
- Fresh verifier subagent checklist (no self-approval): duplicated code vs
  src/+scripts/ siblings; test-coverage gaps; redundancies/dead code; SSOT
  literal scan (grep src+scripts for hardcoded paths, suffixes, magic
  fractions, duplicates of config keys). Runs tests itself. PASS or
  findings looped.
- `git push -u origin kaggle-lane` after EACH green commit (owner push-
  often directive): fast-forward only.

## 2. Commits evidencing this contract (append as they land)

(to be appended granularly)

## 3. Open questions for the owner

1. Is there an actual Kaggle competition handle / username + slug for the
   target competition (evidence found NO `kaggle` API usage, credentials,
   or competition references anywhere in the repo history — the lane
   definition is inferred from format_submission + the transport gap left
   by DVC retirement)? If the intended lane was instead "Kaggle-notebook
   training," say so and S1/S2 pivot accordingly.
2. Should S1 upload the FULL 50pct/full cohort CSVs (57MB raw), or the
   prepared delivery archives the CPU bundle lane produces? Contract above
   supports both; the config block decides.
3. DVC is retired; is Kaggle the chosen replacement transport for
   `results/` training-artifact publication as well, or exports only?
