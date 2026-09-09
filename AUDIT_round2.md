# EuromonitoR — Audit Round 2 (2026-09-10)

Scope: full-tree read-only audit of the training lane at `/home/opc/ONE/EuromonitoR` (every file walked; every finding verified against consumers/callers, not pattern-matched alone). During the audit the owner issued two live directives that were executed and are reported here as **EXECUTED CHANGES** (§E): delete EDA (training-only lane) and add the train-vs-validation loss plot. Everything else was observed, not modified.

Categories: 1 hardcoded-value-that-should-be-config · 2 fallback · 3 unexpected behavior (doc/comment contradictions, misleading logs, wrong variables, stale comments) · 4 type-hint gaps · 5 silent errors · 6 silent data dropping · 7 generated-but-never-used.

## Executive summary

- **18 open findings**: 2 HIGH, 6 MED, 10 LOW.
- By category: cat 1: 7 · cat 3: 5 · cat 5: 2 · cat 6: 1 · cat 7: 2 · cat 1+2: 1. (F11 — plot ylabel and the EDA-migration items — were fixed during the executed changes; they are listed in §E, not counted here. Cat 4 gaps are bundled into one LOW finding.)
- **Verdict**: the lane is structurally sound — the no-fallback SSOT doctrine (Q27) genuinely holds at the accessor level (`runtime()` raises on missing keys; pydantic validates every block at import; selftest pins the discipline). The HIGH findings are a misleading log that names a file that is never written (F02) and the three-way gate's policy thresholds living as function-signature defaults outside any config (F01). Nothing silently drops rows: every gate/dedupe/pairs drop is counted and printed. The previously-fixed items (audit round 1) were verified still-fixed.
- Gates at audit close: **ruff green** (`All checks passed!`), **selftest green** (`SELFTEST PASSED — all oracles green`, pinned counts intact: canonicals 13,250 · gate pairs 135,769 (hard_no 92,650 / proceed 29,351 / fallback 13,768) · labeled 19,916 (7,409 pos / 12,507 hard-neg) · reference 1,745).

---

## Executed owner directives (this session, §E)

**E1 — EDA deleted; lane is training-only.** Verified first that nothing imports the EDA scripts and that `EDA/eda.yaml` (one of the three config files) fed **five** TRAIN-side keys. Then executed:
- Migrated `plots.dpi` (150), `pairs.max_pos_per_group` (4) / `n_neg` (10000) / `neg_oversample` (60), and `strip_audit_sample` (200) into `TRAIN/training.yaml` (`plots:`, extended `pairs:`, new `audit:` blocks).
- `lib/schemas.py`: `EDAConfig`/`EDAPlotsSpec`/`EDASamplingSpec`/`EDAPairsSpec` removed; `PairsSpec` extended; `TrainingPlotsSpec` + `AuditSpec` added; `TrainingConfig` gains `plots` + `audit`.
- `lib/common.py`: EDA file loading/merge/accessor removed; `plot_dpi()` now reads `TRAIN/training.yaml plots.dpi`; merged-view docstrings updated. Verified merged view: `pairs` = both threshold + cap keys, `plots` = {dpi:150}, `audit` = {strip_audit_sample:200}.
- Consumers rewired: `lib/blocking.py` → `training_cfg().pairs.neg_oversample`; `TRAIN/strip_audit.py` → `training_cfg().audit.strip_audit_sample`; comments fixed in `TRAIN/rerank.py`, `TRAIN/report_plots.py`, `TRAIN/zero_shot_sims.py`.
- `TRAIN/selftest.py`: config-split oracle now pins the 2-file SSOT **and asserts the EDA dir stays deleted**; migrated-key checks moved into the no-fallback oracle.
- `EDA/` deleted outright (8 scripts + `eda.yaml`). Docs/config headers updated (`STEPS.md`, `README.md`, `00_config.yaml` comments, `ruff.toml` per-file-ignores, `colab_backend.py` upload list, `run_all.py` comment). EDA-only keys (`sampling`, `rare_cutoff`, `top_per_dim`, `probe_cap`, `model_key`) died with the dir — verified zero TRAIN-side consumers first.

**E2 — Train-vs-validation loss plot.** The trainer logged train loss and dev AP but never validation loss, so the requested plot could not exist.
- `TRAIN/training.py`: builds `eval_ds` in the trainer's own column shape (contrastive: `sentence1/sentence2/label`; mnrl: `anchor/positive`) over the **same dev population** the `BinaryClassificationEvaluator` scores (pos dev pairs + dev hard negatives), passes it as `eval_dataset` (HF now emits `eval_loss` natively per eval step), sets `per_device_eval_batch_size=runtime("batch_size_eval")` (SSOT), extracts `dev_losses` from `log_history`, and writes a `dev_loss_hist` json column beside `train_loss_hist`/`dev_ap_hist`. **Verified live in a 1k-sample smoke run**: eval logs show `{'eval_loss': '22.33', 'eval_dev_cosine_ap': '0.2356', ...}` per eval step.
- `TRAIN/plots.py` `training_loss_plot`: now draws **train loss + val loss** per fold (val dashed), keeps the best-dev-AP vline, labels the y-axis with `SSOT_LOSS` (fixes finding F11), prints an on-axis "no val-loss history" notice for the triplet lane instead of implying a flat line. Render-verified with a metrics CSV in the real emitted shape (both branches).
- `STEPS.md` "Training loss plot" section documents the new contract.

**E3 — pre-existing ruff error cleared.** `dont_delete_me.ipynb` (untracked, predates this session — confirmed the same error exists at HEAD) had an unsorted import block; fixed the cell's import formatting (content untouched). Ruff green.

---

## Findings (open, by severity)

### F01 — HIGH · cat 1 · certain · `data_pipe.py:321-326`
**Gate policy thresholds live as signature defaults, outside the config SSOT.**
```python
def three_way_gate(
    attrs1, attrs2,
    vol_tolerance=0.05,
    raw_conf_threshold=0.85,
    consistency_fallback_threshold=0.3,
):
```
Why: these three values *are* the three-way gate's decision table (±5% volume tolerance, the 0.85 raw-confidence cut, the 0.3 consistency cut) — `STEPS.md` documents them as the gate's semantics. They are used in the body (L330/337/350/376-379) and the sole call site (`data_pipe.py:1313`) never overrides them, so they are effectively constants. No `gate:` block exists in any config file; pydantic never sees them; a change requires editing code. This is the exact "second declaration the config cannot steer" pattern the owner banned (Q27).
Fix: add a `gate: {vol_tolerance, raw_conf_threshold, consistency_fallback_threshold}` block to `TRAIN/training.yaml` + a `GateSpec` in `lib/schemas.py`, read in `run_within_brand_pipeline` via `training_cfg()`, and pass explicitly to `three_way_gate`. (If the owner prefers them frozen, an explicit comment block declaring them domain constants would still be better than silent defaults.)

**STATUS: FIXED (round 3)** — `gate:` block in TRAIN/training.yaml (0.05/0.85/0.3, same values), `GateSpec` in lib/schemas.py wired into TrainingConfig, `three_way_gate` reads `training_cfg().gate` via None-defaults (explicit arg still wins); selftest oracle 12c pins the config values + all three gate behaviors.

### F02 — HIGH · cat 3+7+1 · certain · `TRAIN/training.py:1184-1187`
**Misleading log: print names an HPO-best file that was never written; the config's file key for it has zero consumers.**
```python
with open(RESULTS / f"train_{model_tag}{era}_hpo_best.json", "w") as f:
    ...
print(f"wrote {RESULTS / 'train_hpo_best.json'}", flush=True)
```
Why: the actual write is `train_<model_tag><era>_hpo_best.json` (e.g. `train_paraphrase-multilingual-MiniLM-L12-v2-dlr_hpo_best.json`); the print names `train_hpo_best.json` — a path that is never written. Anyone (or any tooling) chasing the logged path finds nothing. Compounding: `00_config.yaml files.hpo_tpe_best: "hpo_tpe_best.json"` and its `lib/schemas.py:79` declaration have **zero consumers** — `F["hpo_tpe_best"]` is never read anywhere; `STEPS.md`'s reproducibility map even documents `hpo_tpe_best.json` as a produced artifact. Three defects in one site: misleading log (cat 3), dead SSOT key (cat 7), and the real filename re-declared inline instead of via F (cat 1).
Fix: either write to `RESULTS / F["hpo_tpe_best"]` (making the config key live and the print true), or update the config key to the real run-tagged name and print `out_path` verbatim. Also correct the STEPS.md CSV-map row.

**STATUS: FIXED (round 3)** — print now names the actual `out_path` (round 2); the dead `files.hpo_tpe_best` key + `DataFilesSpec.hpo_tpe_best` field deleted (grep-verified zero consumers first); STEPS.md CSV-map row corrected to the real run-tagged names (`train_<model><era>_hpo_best.json` + `_hpo_trials.csv`); hpo.py module docstring corrected; selftest 12c pins the key's absence.

### F03 — MED · cat 1 · certain · `TRAIN/evaluate_models.py:171,196`
**The plot_dpi() fix regressed here: two `dpi=150` literals remain.**
```python
fig.savefig(out1, dpi=150)   # L171
fig.savefig(out2, dpi=150)  # L196
```
Why: the audit-round-1 fix moved 29 savefig sites to `plot_dpi()`; these two were missed (verified: they are the only `dpi=150` occurrences left in the tree besides the `lib/common.py` docstring). The selftest drift-scan's `banned` map has no `evaluate_models.py` entry, so the regression is unpinned.
Fix: `dpi=plot_dpi()` at both sites; add `"TRAIN/evaluate_models.py": [r"dpi\s*=\s*150"]` to the selftest banned map.

**STATUS: FIXED (round 3)** — both savefigs read `plot_dpi()`; the drift-scan banned map gains the `dpi\s*=\s*150` entry (and F06's tcfg-literal entries) so the regression is pinned.

### F04 — MED · cat 1 · certain · `TRAIN/report_plots.py:181`
**Hardcoded operating threshold duplicates the SSOT.**
```python
ax.axvline(0.55, ..., label="operating threshold 0.55")
```
Why: `split.fixed_threshold: 0.55` is the SSOT (`FIXED_THR` is read from config elsewhere, e.g. the EDA probe used it); this line re-declares the value inline, so a config change leaves the plot's marked threshold stale.
Fix: read `training_cfg().split.fixed_threshold` and use it for both the line and the label.

**STATUS: FIXED (round 3)** — `report_plots.py` reads `training_cfg().split.fixed_threshold` for both the axvline and its label.

### F05 — MED · cat 1 · certain · `TRAIN/report_plots.py:96,121,162`
**Hardcoded 07-series CSV names bypass the F map.**
```python
"07c_field_ablation.csv", "07d_data_scaling.csv", "07b_four_pop_scores.csv"  # (literal names)
```
Why: `F["field_ablation"]`, `F["data_scaling"]`, `F["four_pop_scores"]` exist in the config files map and are used by `train.py`/`rerank.py`; `report_plots.py` re-spells the names inline — a rename through config would silently desynchronize this consumer.
Fix: read the three names through `F[...]`.

**STATUS: FIXED (round 3)** — all three 07-series reads go through `F["field_ablation"]` / `F["data_scaling"]` / `F["four_pop_scores"]`.

### F06 — MED · cat 1 · certain · `TRAIN/hpo.py:81-85`
**run_grid re-declares three training knobs inline.**
```python
tcfg = { ..., "weight_decay": 0.01, "lr_scheduler": "linear", "max_grad_norm": 1.0, ... }
```
Why: audit round 1 moved `train.py`'s per-config knobs to `runtime()`, but `hpo.py`'s grid lane kept its own copies. The values currently match `training.weight_decay/lr_scheduler/max_grad_norm`, but nothing keeps them aligned — the HPO grid can silently train under different regularization than the lane it tunes.
Fix: build `tcfg` from `runtime("weight_decay")` etc. (or pass the same cfg dict `train.py` builds).

**STATUS: FIXED (round 3)** — `tcfg` reads `runtime("weight_decay"/"lr_scheduler"/"max_grad_norm")` (same pattern as train.py); drift-scan entries ban the literals' return.

### F07 — MED · cat 1 · certain · `colab_backend.py:54`
**`_TRAIN_FRAC_DEFAULT = 0.25` inline despite the comment saying the literals were moved.**
```python
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = 0.25          # <-- stayed inline
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)
```
Why: its own comment (L47-49) says the 1000/0.25/2 literals "could silently diverge from the configs" — 1000 and epochs were migrated, 0.25 was not. `sweep.train_fracs: [0.25, 0.5, 0.75]` already lives in config; the colab default should be its first element, not a re-declaration.
Fix: `_TRAIN_FRAC_DEFAULT = float(sweep_cfg()["train_fracs"][0])`.

**STATUS: FIXED (round 3)** — reads `float(sweep_cfg()["train_fracs"][0])`; selftest 12c pins the equality.

### F08 — MED · cat 3 · certain · `TRAIN/masking.py:43`, `TRAIN/hpo.py:8`, `TRAIN/train.py:357`, `TRAIN/training.yaml:20`
**Stale masking-band claims: comments still say 5-15% while the config band is 20-35%.**
```python
# masking.py:43  "spec: masking done to different extents varying from 5-15%"
# hpo.py:8      "variable 5-15% extent"      # train.py:357 "the ANCHOR text gets 15% random token masking"
mask_lo: 0.20       # owner spec: 5-15%   <- training.yaml comment contradicts its own value
mask_hi: 0.35
```
Why: the band was changed to 0.20-0.35 but four comments still describe the old spec. Anyone tuning from the docs trains the wrong band; the midpoint derivation (0.275) is what the code actually uses.
Fix: update the four comments to the 20-35% band.

**STATUS: FIXED (round 3)** — training.yaml comment fixed in round 2; masking.py:43, hpo.py:8 and train.py:357 ("15% random token masking" → "config-band variable token masking, U(0.20, 0.35)") updated; the GUARANTEE note keeps its measured U(0.05,0.15) history figure (that was the measurement era, now labeled as such).

### F09 — MED · cat 5 · probable · `TRAIN/training.py:740-742`
**Discriminative-LR failure degrades to single-LR with only a print.**
```python
except Exception as exc:
    print(f"    [optim] single LR fallback ({exc})", flush=True)
    groups = [{"params": model.parameters(), "lr": base_lr}]
```
Why: an unexpected `_discriminative_groups` error silently switches the optimizer contract mid-run — training continues under a different learning-rate geometry with no metrics flag, no mlflow tag, no row annotation. Under the owner's no-fallback doctrine this is the one sanctioned-looking escape hatch that isn't config-sanctioned. (It is *visible* in stdout, hence "probable" not certain: the doctrine's visibility guarantee is met, but the run is not distinguishable downstream.)
Fix: either raise (strict doctrine), or record `lr_groups: "single"` in the fold-metrics row + a mlflow tag so a fallback run is queryable.

**STATUS: FIXED (round 3)** — fallback kept (visibility-sanctioned) but the fold-metrics row now carries `lr_groups` ("discriminative" normal / "single" fallback) and the fallback print names the field, so a degraded fold is queryable downstream without scraping stdout.

### F10 — LOW · cat 3 · probable · `TRAIN/training.py:424-427`
**`--folds` > CV_FOLDS silently caps at CV_FOLDS.**
```python
n_folds = cv_folds if cv_folds is not None else CV_FOLDS
all_folds = kfold_barcodes(df, CV_FOLDS, SEED)   # always CV_FOLDS
...
folds = all_folds[:n_folds]
```
Why: `kfold_barcodes` is always called with `CV_FOLDS` (5); `[:n_folds]` then caps — a user asking for `--folds 8` silently gets 5. The `[:n_folds]` slice compensates only for n_folds < CV_FOLDS.
Fix: `kfold_barcodes(df, n_folds, SEED)` when `cv_folds` is given, or assert `n_folds <= CV_FOLDS`.

**STATUS: FIXED (round 3)** — `kfold_barcodes(df, n_folds, SEED)` builds the requested fold count (identical behavior for n_folds <= CV_FOLDS: same strided permutation split; the `--quick` prefix slice is unchanged); `--folds 8` now really runs 8.

### F11 — FIXED in §E2 · `TRAIN/plots.py:73`
~~`ax.set_ylabel("MNRL loss" if i == 0 else "")`~~ — the lane's default loss is `contrastive`; the label now uses `f"{SSOT_LOSS} loss"`. (Recorded as fixed, not counted in the open tally.)

### F12 — LOW · cat 1+2 · probable · `lib/nlp.py:65`
**`encode_corpus(..., batch_size: int = 256, max_seq_length: int = 128)`** — signature defaults duplicate `training.batch_size_embed`/`max_seq_length`. Every caller passes `runtime()` values today, so the defaults are unreachable fallback literals; per the no-fallback doctrine they should be removed (keyword-only, required) so a future caller cannot silently get the default.
Fix: make both params required keywords.

**STATUS: FIXED (round 3)** — `batch_size` / `max_seq_length` are required keyword-only params (all callers already pass `runtime()` values — verified by grep before the change).

### F13 — LOW · cat 1 · probable · `lib/volume_verified.py:6`
**`RESULTS / "second04_pairs_positive.csv"` hardcoded** — not in the config `files:` map; the repo-side history manifest is absent in this lane (the code prints a loud notice and runs gate-only, documented). Low priority: add it to the files map (or a `manifests:` block) so the name is at least declared SSOT-side.

### F14 — LOW · cat 6 · certain (zero impact today) · `TRAIN/evaluate_models.py:42`
**`df = labeled.merge(emb_sim, on=["gtin1","gtin2"], how="inner")` with no drop accounting.**
Probed on the real CSVs: 19,916 → 19,916 (0 rows lost). But an inner join that silently shrinks on drift would contradict the "nothing drops silently" guarantee.
Fix: `n_before/n_after` print (one line) or an explicit length assert.

**STATUS: FIXED (round 3)** — loud `[merge]` drop accounting (n_before/n_after/dropped) added; an empty merge is a hard SystemExit (upstream sims missing entirely); the zero-rows-lost-today behavior is unchanged and now visible.

### F15 — LOW · cat 1 · probable · `colab_backend.py:44-45`
**`RESULTS = HERE / "artifacts" / "results"`, `DATA = HERE / "artifacts" / "data"`** re-derive the paths that `00_config.yaml paths.results_dir/data_dir` already own (they happen to match today). Fix: import from `lib.common` (module already imports `lib.common` for `sweep_cfg`).

**STATUS: FIXED (round 3)** — `RESULTS` / `DATA_DIR` imported from lib.common (00_config.yaml paths block); `DATA` kept as the `DATA_DIR` alias for its call sites.

### F16 — LOW · cat 3 · probable · `run_all.py` (`_sh`, main loop)
Two nits in the orchestrator: (a) `_sh` opens step logs with `"w"` — a re-run truncates the previous log while the docstring sells resumability (the *manifest/artifacts* are append-safe; logs are not); (b) the `except SystemExit` handler records `failed: ...` and **continues to the next step** — defensible for a resumable orchestrator, but a failed `data_prep` means downstream steps burn GPU on stale inputs. Fix: `--stop-on-fail` flag or fail-fast default for the chain steps.

**STATUS: FIXED (round 3)** — (a) `_sh` opens logs in append mode with a run-separator header (a re-run no longer truncates); (b) `--stop-on-fail` (default False, historical record-and-continue behavior preserved) halts the chain at the first failed step; the manifest write moved to a `finally` so a stop still records the row.

### F17 — LOW · cat 4 · certain · assorted signatures
Hot-path functions with missing annotations (returns and/or params): `data_pipe.py` `three_way_gate`, `jaccard_similarity`, `generate_ngrams`, `extract_discriminative_ngrams`, `generate_canonical`, `run_within_brand_pipeline`, `parse_attribute_volume_pack`, `reference_path`; `lib/common.py` `runtime` (return), `write_visibility_log` (`df`); `lib/gtin.py` `_canonicalize_gtin`; `lib/blocking.py` `eval_blocking`; `TRAIN/evaluate_models.py` `evaluate_model`; `TRAIN/train.py` `_append_csv`; `lib/text.py` `get_measurement_type(category)`, `attributes_keys(series)`; `TRAIN/masking.py` `mask_text`, `augment_positives`; `TRAIN/hpo.py` `_grid_folds`, `run_tpe` params; `TRAIN/plots.py` both fns. No mutable-default-arg bugs found (AST scan clean); no import-shadowing found (AST scan clean).

**STATUS: FIXED (round 3)** — Python 3.12 hints added to every listed function (three_way_gate attrs/thresholds were already typed by the F01 work): data_pipe `parse_attribute_volume_pack`, `jaccard_similarity`, `generate_ngrams`, `extract_discriminative_ngrams`, `generate_canonical`, `reference_path`, `run_within_brand_pipeline`; lib/common `runtime` (`-> Any`, honest open type — the block is heterogeneous) + `write_visibility_log`; lib/gtin `_canonicalize_gtin`; lib/blocking `eval_blocking`; evaluate_models `evaluate_model`; train `_append_csv`; lib/text `attributes_keys`; masking `mask_text` + `augment_positives`; hpo `_grid_folds` + `run_tpe`; plots both. Hints only — zero behavior change.

### F18 — LOW · cat 7 · certain · `lib/text.py`, `lib/common.py`
**Dead code (zero consumers anywhere — exhaustive all-file-type grep, no dynamic dispatch):**
- `lib/text.py`: `get_measurement_type` (+ `CATEGORY_MEASUREMENT_TYPE`, `DEFAULT_MEASUREMENT_TYPE`, consumed only by it), `is_pack_multiple`, `DRY_MIX_HINTS`, `SUSPECT_ROUND`, `FLAVOR_VOCAB`/`FLAVOR_RE` (lib/text's copy — note `data_pipe.py` has its own `FLAVOR_PATTERN`, a near-duplicate list), `_LITER_UNITS`, `MULTIPACK_RE`; `norm_unit`/`bucket_ml`/`extract_volume_measurement` are internal-only (single in-module caller) — the public wrapper `extract_volume_ml` is the live API.
- `lib/common.py`: `multi_retailer_mask` — defined, never called.
- `TRAIN/blocking_audit.py`: `BUDGET = 5_000_000`, `MIN_RECALL = 0.95` are inline policy knobs (cat 1), audit-only script.
Fix: delete the dead symbols (selftest pins none of them) or, for `FLAVOR_VOCAB`, unify with `data_pipe.FLAVOR_PATTERN` to drop the duplicate list.

**STATUS: FIXED (round 3)** — every listed symbol grep-re-verified dead then deleted (lib/text.py: get_measurement_type, CATEGORY_MEASUREMENT_TYPE, DEFAULT_MEASUREMENT_TYPE, is_pack_multiple, DRY_MIX_HINTS, SUSPECT_ROUND, FLAVOR_VOCAB/FLAVOR_RE, _LITER_UNITS, MULTIPACK_RE; lib/common.py: multi_retailer_mask). `data_pipe.FLAVOR_PATTERN` untouched. Judgment call: `BUDGET`/`MIN_RECALL` moved into the EXISTING `audit:` block (`audit.blocking_budget` / `audit.blocking_min_recall`, AuditSpec extended) — the audit lane's knobs belong in the audit block rather than a new `blocking_audit:` block; blocking_audit.py reads them via `training_cfg().audit` (module constants `BUDGET`/`MIN_RECALL` kept as names). Selftest 12c pins the symbols' absence and the knob equality.

### F19 — LOW · cat 3 · certain · `lib/nlp.py:65` vs `lib/common.py:394`
**Duplicate cosine helpers**: `lib/nlp._cosine(emb, pairs)` and `lib.common.pair_similarity(emb, pairs_idx)` are the same function; `TRAIN/training.py` aliases `_cos = pair_similarity` while `TRAIN/report_plots.py` imports `_cosine` from `lib.nlp`. One implementation should call the other (or one die).

**STATUS: FIXED (round 3)** — minimal-churn choice documented: `lib.common.pair_similarity` is canonical (the SSOT module); `lib/nlp._cosine` is now a re-export alias (`from lib.common import pair_similarity as _cosine`) so the historical import surface keeps working with ONE implementation; `TRAIN/report_plots.py` imports `pair_similarity` from lib.common directly. Selftest 12c pins `_cosine is pair_similarity`.

### F20 — LOW · cat 7 · certain · `00_config.yaml` files map / `TRAIN/evaluate_models.py:124`
**`model_evaluation_summary` written, never read** — `evaluate_models.py` writes the summary CSV via `F[...]`; no reader exists anywhere (STEPS.md documents it as a produced artifact, so it is a report artifact, not a bug — but nothing consumes it, including the plots). Borderline: keep as the audit artifact it is documented to be, or drop it.

**STATUS: KEPT (round 3, no code change)** — kept as the documented report artifact per the audit's own recommendation; the STEPS.md CSV-map row (`results/model_evaluation_summary.csv` → `TRAIN/evaluate_models.py`, regenerated) and the producer description were re-verified accurate.

### F21 — LOW · cat 7 · certain · `TRAIN/training.yaml:85` (was `bands.mining_band`)
**Dead config key**: `bands.mining_band: [0.35, 0.90]` — `band("mining_band")` is never called; only `eval_mining` and `rerank_band` are consumed. The band actually used for mining is `mining.band "0.45-0.80"` (the string form). The `[0.35, 0.90]` list is validated, documented as "eval-pair mining", and read by no one. Fix: delete the key (and its schema field) or wire it to where the comment claims it is used.

**STATUS: FIXED (round 3)** — key + schema field removed in round 2; round 3 swept the remaining references: lib/common.py `band()` docstring now names only `eval_mining`/`rerank_band` (with the removal note); grep confirms no `mining_band` remains in any doc/config/schema; selftest 12c pins its absence and the exact band-key set.

---

## Checked and clean

- **Gate/canonical drops**: `run_within_brand_pipeline`'s GTIN-guard and `build_training_data`'s neg-endpoint drops are counted in `stats` and printed loudly (`n_neg_dropped`, `n_pos_empty_dropped`, empty-text guards); payload visibility dump (`payload_pairs.csv`) prints counts + kinds.
- **Dedupe tiers**: every tier's drops land in `06_dedupe_summary.csv` (T1 checksum-invalid deferred counted, T2 conflicts deferred to T3 counted, T3 ambiguous groups flagged + `06_ambiguous_offer_groups.csv`).
- **No-fallback SSOT discipline (Q27)**: `runtime()` raises on missing keys; `SSOT_LOSS`/`SSOT_CONTRASTIVE_MARGIN` read-once; the selftest drift-scan pins the retired literals; the 07-series axes read `sweep_cfg()`; rerank band + A/B margins read config; `kfold_barcodes` prints fold composition; strip-audit sample reads `audit.strip_audit_sample` (post-migration).
- **Boundary contracts**: `ExtractedAttributes`/`GateResult`/`CanonicalRecord`/`TrainingData`/`MaskingResult`/`DataTuple`/`TrainConfig`/`FoldSets` all validated at the edges; zero-pack guard pinned by selftest.
- **Sanctioned domain constants (judged clean, not findings)**: extraction-confidence ladder (0.75/0.85/0.90/0.95/0.98 in the volume/pack extractors — hand-calibrated ruling values; note they feed F01's threshold comparison), `SCHEMA_WORDS`/`MODEL_PAYLOAD_SOFT_STOP` (owner-curated word lists), `hard_negatives` chunk=2048 (memory constant, documented ~300MB/chunk), `MINIMAL_STOPWORDS`/`CONCEPT_FOLDS`/`KEEP_TOKENS` (SSOT json + documented sets), `lib/text.py` `BUCKET=5`/`MIN_PACK/MAX_PACK` (documented domain bounds).
- **Documented no-consumer artifacts**: `artifacts/embeddings/*.npz` (analysis dump), `sku_to_rep.csv` (audit-only pointer) — both in STEPS.md's reproducibility map.
- **Deliberate non-fatal try/excepts** (owner-ruling documented): train.py mask-effect probe ("non-fatal — training results stand"), ProgressCallback (display-only).
- **`zero_shot_sims` resume contract**: pair-sequence + per-model canonical-text fingerprint stamps; stale columns re-score loudly; resumed columns carried on incremental writes (the old column-dropping bug is fixed and documented).
- **`build_pairs` GS1 discipline**: positives need a checksum-VALID group barcode; negatives need BOTH barcodes valid (matches `mine_hard_negatives`); `neg_oversample` from SSOT; undersampled negatives raise, never pad silently.
- **Holdout discipline**: Youden picked on dev, applied to test; dev/train/test barcode sets disjoint via component folds; `auc_cross` reported separately.

## Gate results (audit close)

- `ruff check .` → **All checks passed!** (includes the §E3 notebook fix; the error was confirmed pre-existing at HEAD before being fixed)
- `TRAIN/selftest.py` → **SELFTEST PASSED — all oracles green** (post-migration: 2-file config-split oracle, EDA-deleted pin, migrated-key checks, all 14 oracles, pinned counts unchanged)
- Live smoke (1k sample, 2 folds, 1 epoch): `eval_loss` emitted per eval step (`{'eval_loss': '22.33', 'eval_dev_cosine_ap': '0.2356', ...}`), fold metrics CSV carries the history columns; the train-vs-val plot renders (both branches verified).

*Previously-fixed items from audit round 1 were re-verified and are not re-reported. No repository code was modified for findings F01-F21 — they are open items for the owner.*

---

## Round 3 remediation (2026-09-10) — all findings applied

Round 3 applied the fixes for F01-F21 (F01/F02-print/F08-yaml/F21-key were
already half-applied in round 2's working tree; round 3 finished them and
did the rest). Every deletion was grep-re-verified for zero consumers
first; every config move kept the exact same values so the pinned
real-data counts could not shift. New selftest oracle **12c** pins the
round-3 fixes (gate==config on all three decision paths, `hpo_tpe_best`
absent, `mining_band` absent, the ten F18 dead symbols absent from the
import surface, colab train-frac default == `sweep.train_fracs[0]`,
blocking_audit knobs == the `audit:` block, `_cosine is
pair_similarity`), and the 12b drift-scan banned map gained the
evaluate_models `dpi=150` and hpo tcfg-literal entries.

## Round 3 verification

Run on the main tree at `/home/opc/ONE/EuromonitoR` after all fixes:

1. **ruff** — `.venv/bin/python -m ruff check .` → **All checks
   passed!** (plus a full-import sweep: every touched module — data_pipe,
   colab_backend, run_all, lib.*, TRAIN.* — imports clean).
2. **selftest** — `.venv/bin/python TRAIN/selftest.py` → **SELFTEST
   PASSED — all oracles green**, 180 PASS / 0 FAIL, rc=0. Pinned
   real-data counts unchanged: canonicals 13,250 · gate pairs 135,769
   (hard_no 92,650 / proceed 29,351 / fallback 13,768) · labeled 19,916
   (7,409 pos / 12,507 hard-neg) · reference 1,745. The new oracle 12c
   contributed 24 PASS lines.
3. **data_prep determinism** — `.venv/bin/python TRAIN/data_prep.py`
   regenerated both pinned CSVs **byte-identical** (md5):
   - `artifacts/results/canonical_records.csv` →
     `039f05eac0a28e8cd715edecb5267461` (pinned: same)
   - `artifacts/results/gate_results.csv` →
     `c70e5189051e10ac560d6e6b2d728d77` (pinned: same)
4. **1k smoke train** — `MLFLOW_TRACKING_URI=off .venv/bin/python
   TRAIN/train.py --sample 1000 --split holdout --no-plot` → rc=0, fold
   row status=ok, `lr_groups=discriminative` in the fold-metrics row
   (early stopping engaged at step 198/epoch 4.1 as configured; test
   AUC 0.8490 / PR-AUC 0.7984 — identical to the historical 1k run,
   deterministic).
