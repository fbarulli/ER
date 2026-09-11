# SILENT_DROPS — work plan for this session

Repo: EuromonitoR (local git only — commits stay local, never pushed).
Baseline: `5e3595a` (src/euromonitor package migration committed).

This file is the coordination artifact for closing the silent-drop guarantee
gap. Each task is scoped for one agent, under ~90 minutes. Observability and
guardrail work only — no model-behavior changes, no new dependencies
(stdlib + pydantic/pandas/torch already present), house conventions
(config knobs in the SSOT via schemas.py, selftest oracles, results/ is a
runtime tree and never committed).

## Ground rules (apply to every task)

- All code lives under src/euromonitor/ (post-migration layout).
- Config knobs go in 00_config.yaml / src/euromonitor/training/training.yaml
  via src/euromonitor/core/schemas.py pydantic models — nothing hardcoded.
- New regression pins go into src/euromonitor/training/selftest.py following
  the existing check()/oracle_* pattern.
- Results/manifests/ and results/ generally are runtime artifacts — never
  commit them.
- Run the relevant selftest oracles or a direct module smoke before
  reporting done.
- Commits are LOCAL ONLY. Do not push, do not create remotes.

## Design sketch: per-stage manifest

`src/euromonitor/core/manifest.py` (stdlib hashlib/json/os/tempfile +
pandas only) provides: `sha256_file(path)` (chunked read), `atomic_write`
(`<path>.tmp-<pid>` in the same dir then `os.replace` — a reader never sees a
partial file), and a `StageManifest` pydantic model written to
`results/manifests/<stage>.json` LAST. The manifest's presence with
`status: "complete"` IS the completion marker; an interrupted stage leaves
at most the previous run's manifest plus `.tmp-*` residue, which
`verify_manifest(stage)` treats as failure.

| field | type | content |
|---|---|---|
| `schema_version` | `"1"` | bump on incompatible change |
| `stage` | str | `dedupe`, `data_prep`, `sweep_full`, `train`, … |
| `started` / `finished` | ISO-8601 UTC | `finished` absent while running = incomplete |
| `status` | `"complete" \| "failed" \| "running"` | written only via final atomic rename |
| `inputs[]` | {path, sha256, rows, cols?} | every file read |
| `outputs[]` | {path, sha256, rows, expected} | every file written; `expected=false` = unexpected extra |
| `row_accounting` | {input_rows, output_rows, dropped: {reason: count}} | input == output + Σ dropped asserted |
| `environment` | {git_sha, config_sha, seed, host} | `git_sha` = `dirty` fallback when worktree unclean |
| `expected_outputs[]` | [str] | cross-checked against `outputs` |

## Task list

| # | Task | Files | Status |
|---|------|-------|--------|
| 1 | `core/manifest.py`: sha256_file + atomic_write helpers | `src/euromonitor/core/manifest.py` | done |
| 2 | AuditSpec manifest knobs in schemas.py + training.yaml | `src/euromonitor/core/schemas.py`, `src/euromonitor/training/training.yaml` | pending |
| 3 | StageManifest model + write/read/verify in manifest.py | `src/euromonitor/core/manifest.py`, `schemas.py` | pending |
| 4 | Pilot manifest on training/dedupe.py | `src/euromonitor/training/dedupe.py` | pending |
| 5 | selftest oracle_manifest | `src/euromonitor/training/selftest.py` | pending |
| 6 | Manifests for data_prep/pipeline.py canonical+gate stage | `src/euromonitor/pipeline.py`, `src/euromonitor/training/data_prep.py` | pending |
| 7 | Manifests for labeled_pairs / evaluate_models / zero_shot_sims | `src/euromonitor/training/{labeled_pairs,evaluate_models,zero_shot_sims}.py` | pending |
| 8 | run_all.py step manifests + atomic npz writes | `run_all.py` | pending |
| 9 | Source-export drift gate in common.py loaders | `src/euromonitor/core/common.py` | pending |
| 10 | Per-row dedup removal review table | `src/euromonitor/training/dedupe.py`, `schemas.py`, `selftest.py` | pending |
| 11 | Hash-verify Colab downloads in cli/colab.py | `src/euromonitor/cli/colab.py` | pending |
| 12 | Hash-verify NER Colab downloads + remote manifest | `src/euromonitor/ner/colab_ner.py`, `src/euromonitor/ner/ner.py` | pending |
| 13 | External-library row-loss guards at silent call sites | `src/euromonitor/core/{blocking,volume_verified}.py`, `training/data_quality_audit.py` | pending |
| 14 | Sync STEPS.md with the manifest layer | `STEPS.md` | pending |

## Task details (verified file anchors from the research pass)

### 1. core/manifest.py — sha256_file + atomic_write

Port the chunked `_sha256` pattern from
`src/euromonitor/training/data_quality_audit.py:28`; `atomic_write(path,
data)` writes `path.tmp-<pid>` in the SAME directory then `os.replace`s onto
the final path; `atomic_write_csv(df, path)` wraps `df.to_csv` through the
same mechanism (serialize to the temp path, fsync, rename). Pure helpers,
no manifest logic yet. Depends: nothing. Blocks everything.

### 2. AuditSpec manifest knobs

Extend the `AuditSpec` pydantic model in `src/euromonitor/core/schemas.py`
with `manifest_dir` (default `results/manifests`), `source_export_expected_rows`
(71,623 — the current raw export census), `source_drift_threshold_pct`
(default 0.0), `manifest_stages` (list of stage names that must produce
manifests). Mirror defaults into the `audit:` block of
`src/euromonitor/training/training.yaml` so the SSOT stays explicit.
`extra="forbid"` keeps the contract tight. Depends: none (parallel-safe
with 1). Blocks 3+.

### 3. StageManifest model + write/read/verify

Add the `StageManifest` pydantic model (field table above) to schemas.py's
boundary-contract section; in manifest.py add `write_manifest(stage, …)`
(atomic, written LAST — presence with status complete is the marker),
`read_manifest(stage)`, and `verify_manifest(stage)` (re-hash outputs,
check row-accounting closure input == output + Σ dropped, fail on `.tmp-*`
residue or missing manifest). Depends: 1, 2.

### 4. Pilot the manifest on training/dedupe.py

Wrap `main()` with started/finished; hash the raw-export input; record
`row_accounting.dropped` from the existing tier summary (T1/T2/T3 +
`deferred_to_t3` + `skipped_checksum_invalid` reasons; current census
10,094 dropped → 61,529 of 71,623); atomically write the four outputs
(dataset_deduped, sku_to_rep, 06_dedupe_summary, 06_ambiguous_offer_groups)
via helper 1; write `results/manifests/dedupe.json` last. Capture + write
only — no dedupe logic edits. Depends: 1–3.

### 5. selftest oracle_manifest

Add `oracle_manifest()` to `src/euromonitor/training/selftest.py`,
registered in the main list (~line 1260), following the existing
check()/oracle pattern (cf. oracle_pinned_counts at :487). Pins: manifest
parses against StageManifest; dedupe accounting closes
(71,623 − 10,094 == 61,529); every output sha256 matches the on-disk file;
a deliberate write failure (bad tempdir) leaves no complete marker.
Depends: 4.

### 6. Manifests for the canonical+gate stage

data_prep/pipeline.py: manifest inputs (raw export, dataset_deduped) and
outputs (canonical_records 13,250, gate_results 135,769, gate_visibility,
payload_pairs) with dropped reasons from the existing gtin-guard counts and
gate decision census (hard_no/proceed/fallback).
`pipeline.py:run_within_brand_pipeline` already prints all of these — the
change is capture+write. Depends: 1–3.

### 7. Manifests for labeled_pairs / evaluate_models / zero_shot_sims

labeled_pairs: dropped = fallback exclusion (already asserted == pin).
evaluate_models: pour the existing merge/straddle/parked accounting into
row_accounting (the assert-exact split `dev+test+straddle+parked == len(df)`
becomes manifest-enforced). zero_shot_sims: per-model `.model_fp` hash
stamps become manifest output hashes. Depends: 1–3 (independent of 4–6).

### 8. run_all.py step manifests + atomic npz

run_all.py step1's `np.savez_compressed` (~line 112) becomes atomic_write
(kills the skip-if-exists-on-a-partial-.npz re-run bug); each step writes
`results/manifests/<step>.json`. The JSON complements the existing
train_manifest.csv, not replaces it. Depends: 1–3.

### 9. Source-export drift gate in common.py loaders

`load_raw_export`/`load_dataset` in `src/euromonitor/core/common.py`
compare `len(df)` against `audit.source_export_expected_rows` and the
pinned sha256 of the export path; mismatch beyond
`source_drift_threshold_pct` raises SystemExit with the exact observed vs
expected numbers. Closes the "changed source export" gap. Depends: 1–3.

### 10. Per-row dedup removal review table

In dedupe.py, emit `results/06_dedupe_removals.csv` — one row per dropped
raw product_id: `{product_id, rep_id, tier}` (tier from the per-tier
`parent` updates before transitive resolution; product_id→rep from the
existing sku_to_rep frame), so removals are reviewable without joining
sku_to_rep + summary. Register the filename in `DataFilesSpec` (`removals`
key) and pin the count oracle in selftest (10,094 rows). Depends: 4.

### 11. Hash-verify Colab downloads in cli/colab.py

After remote lanes, fetch the remote-generated `results/manifests/*.json`
(the manifest helpers ship in the uploaded src tree) via the existing
`_list_remote`/colab download channel; `download_results`/
`download_checkpoints` re-hash each local file against the manifest sha256
and re-raise on mismatch or missing expected file — extending the existing
fail-loud ruling (colab.py:438) from presence to integrity. Depends: 1–3
and 6–7 (remote stages must produce manifests first).

### 12. Hash-verify NER Colab downloads + remote manifest

`download_if_exists` (ner/colab_ner.py:573) currently logs "unavailable"
and continues — change to fail-loud for expected artifacts (ner_errors.csv,
training_metadata.json, ner_model_final.zip) and verify sha256 against a
small remote manifest; ner.py's finalization already writes
training_metadata.json (ner.py:1133) — add file hashes there. Depends: 1
and the 11 pattern.

### 13. External-library row-loss guards

Wrap the silent pandas shrinkage sites with pre/post len reporting via a
small `count_drop(before, after, reason)` helper in manifest.py:
`src/euromonitor/core/blocking.py:61` (`drop_duplicates("_t")`),
`src/euromonitor/core/volume_verified.py:38-60` (manifest filter +
`pairs[agrees]`), `src/euromonitor/training/data_quality_audit.py:146`
(`explode`). evaluate_models already demonstrates the assert-exact pattern.
Depends: 1 (helper) only.

### 14. Sync STEPS.md

Document `results/manifests/` in the pipeline-steps and
transparency-guarantees sections, add the manifest to the CSV
reproducibility map (regenerated, never committed), note the new oracle in
the selftest bullet — keeping the repo's doc-honesty convention (cf.
commit 2fc3e63 "docs never lie"). Depends: 4–5 landed; do LAST.

## Sequencing (one agent at a time, commit per task)

1 → 2 → 3 → 4 → 5 (foundation chain, strictly ordered)
then: 6 → 7 → 8 → 9 → 10 → 13 (independent-ish fan-out, still serialized)
then: 11 → 12 (need 6–7 landed)
last: 14 (STEPS.md truth sync after everything real is in place)

Status column above is updated as tasks land.
