# Finalization report — model-input composition (Part 1) and the ANN arm (Part 2)

Finalizer session, 2026-09-15. Repo `/home/opc/ONE/EuromonitoR`, branch `training`.
Owner's instruction for this session: **"im going back on my word, make all changes final"** —
commit everything meaningful, leave the tree clean, push to `ER/training`.

Companion document: `MODEL_INPUT_FIX_REPORT.md` (the implementer's report, sections 1-12).
This file is the **independent** account: what I reproduced, what I corrected, what I found that
the implementer's map missed, and the ANN-arm measurement.

---

## 0. Verdict up front

| Question | Answer |
|---|---|
| Is the implementer's model-input work correct? | **CONFIRMED, with 3 corrections.** Legacy byte-identity holds (855/855 rows, re-derived from the *committed* pre-change code, not from its fixture). The cleaned composition is the shipped default. Three things its report got wrong or understated, all fixed here. |
| Is the default what the owner decided? | **YES** — `profile: cleaned`, one config value restores legacy. |
| Suite green? | **YES** — see §2 for the exact final count; the committed tree was **red** when I took it over. |
| Is the composition mode traceable? | **YES** — via the existing `core.tracing` SSOT, the checkpoint manifest and the ANN reuse fingerprint. Executed evidence in §4. |
| Any coverage gaps? | **NO** — emitted set == registered set == visited set, proved by execution in §5. |
| Does the composition change help or harm the ANN arm? | **HELPS.** Measured on CPU against the recovered `checkpoint-44`; every metric and every attribute error bucket improves *once each composition is judged at its own operating point*. The checkpoint is nonetheless **stale and must be retrained**. §6. |

---

## 1. What I CONFIRMED vs what I CORRECTED

### Confirmed (reproduced myself, not trusted from the report)

**1.1 Legacy byte-identity — CONFIRMED, and verified on both ends.**
The implementer's golden fixture is only evidence if the *fixture* is genuine, so I re-derived it
from the committed pre-change code in an isolated extraction of `2a15852`
(`git archive 2a15852 | tar -x -C /tmp/ER-head-check`, no worktree, no repo mutation):

```
$ cd /tmp/ER-head-check && PYTHONPATH=/tmp/ER-head-check/src python /tmp/verify_legacy_head.py
records=855
HEAD-code  legacy_sku_text mismatches     : 0
HEAD-code  legacy_canonical_text mismatches: 0
```

Then the other end — the new builder's `legacy` profile against the same fixture:

```
$ PYTHONPATH=src python /tmp/verify_agent_a.py
[1] legacy byte-identity over 855 rows: sku mismatches=0 canonical mismatches=0
VERDICT: legacy byte-identical
```

So the chain `committed 2a15852 code == fixture == new code legacy profile` is closed on 855 real
rows (585 review-band pairs + 150 singleton-GTIN + 120 dataset rows). **The rollback contract is
real.**

**1.2 The cleaned default and the one-value rollback — CONFIRMED.**
`config/training.yaml` ships `profile: "cleaned"` / `include_evidence: false`; the pydantic
validator rejects `cleaned + include_evidence: true` at config load. Rollback lines are in §3.

**1.3 The string-level claims on the 585-pair review band — CONFIRMED.**
Re-measured independently (`/tmp/verify_agent_a.py`): true-pair Jaccard `0.3200` vs cross-pair
`0.1195`, margin **+0.2005** (implementer reported +0.2010 on a different cross-pair sample —
same conclusion); legacy margin +0.1157. Discriminative digits survive on **215/215** rows whose
legacy target carried one. **75078** different-brand source pairs contain **0** identical strings,
and the 585 cleaned source strings collapse to 0 collisions across brand groups. The implementer's
own honest counter-result reproduces too: on the 24 `source_is_target_row` rows the cleaned Jaccard
*regresses* (0.5129 → 0.4646), which is stated in its report and is not hidden here.

> One nuance I checked because a high number looked wrong: the maximum cross-"brand_match=False"
> source Jaccard is 0.9688, but the two rows are the *same* brand and the *same* title (`BOB`,
> "lingon & cranberry soft drink") from different retailers. `brand_match` compares source against
> canonical, not source against source, so that is not a collapse. The correct cross-brand test
> (group by the actual brand value) yields 0 collisions.

### Corrected (three things the implementer's report got wrong, fixed here)

**C1 — The committed tree was RED.** The report claims *"290 passed, 2 skipped → 313 passed, 2 skipped"*
and *"No pre-existing test needed changing"*. On the committed tree I measured:

```
$ PYTHONPATH=src python -m pytest tests/ -q
FAILED tests/test_model_input_contract.py::test_structured_token_channel_is_identical_across_profiles
1 failed, 312 passed, 2 skipped
```

Cause: the same commit series added the universal pack symmetry (`structured_features.symmetric_info`
+ `model_input_info`, config key `training.structured_features.implicit_pack_qty`), which
**deliberately** makes the structured tail profile-dependent for `pack`. The old test asserted the
channel was profile-independent. I rewrote it to assert what is actually true and worth pinning —
every shared field group is profile-independent, only `[FIELD_PACK_SIZE]` may differ, and the
symmetry is visible on the text — and added a second test proving the implicit default reaches the
**numeric vector** as well as the text. This is a legitimately-affected test, updated with that
justification; no coverage was deleted (the file went 23 → 33 tests).

**C2 — The report omits its own most consequential fix.** `MODEL_INPUT_FIX_REPORT.md` §3 does not
describe the universal pack symmetry, and §6's file list omits
`src/core/structured_features.py` and the `implicit_pack_qty` config key. It is a real behaviour
change (425/585 rows previously carried `pack_qty_1` on the source side only); it deserves to be in
the record rather than only in a commit message.

**C3 — `model_input_provenance()` returned a loose dict across three artifact boundaries.**
The project rule is pydantic at boundaries (`AGENTS.local.md` §5) and the task asked for the
composition/config object and any new trace record to be modelled in `core.schemas`. I replaced it
with `TrainingSpec.ModelInputComposition` (`profile`, `include_evidence`, plus a stable `fingerprint`
digest), and rewired all six consumers. The digest is what lets an artifact name its input contract
without carrying the text.

---

## 2. Final state

| Item | Value |
|---|---|
| Default mode | `training.model_input.profile: "cleaned"`, `include_evidence: false` |
| Config file | `config/training.yaml` |
| Legacy rollback | `profile: "legacy"` + `include_evidence: true` — byte-identical to `2a15852` on 855/855 fixture rows |
| Evidence ablation | `profile: "legacy"` + `include_evidence: false` |
| Rejected combination | `profile: "cleaned"` + `include_evidence: true` → `ValidationError` at config load |
| Test count | **343 collected: 341 passed, 2 skipped** at commit time (was 290 passed, 2 skipped at `2a15852`) |
| Branch / remote | `training` → `git push ER training` |

Files in the composition change (commits, not a working tree):

```
config/training.yaml                    model_input block (default cleaned) + implicit_pack_qty
config/paths.yaml                       attribute-separation artifact bindings
src/core/schemas.py                     TrainingSpec.ModelInputSpec, .ModelInputComposition
src/core/model_input.py                 NEW — the single composition point + provenance record
src/core/structured_features.py         symmetric_info (universal implicit-default rule)
src/pipeline.py                         payload call sites + composition trace row
src/predict_items.py                    scoring lane
src/training/rand_matching.py           training lane + composition in the ANN fingerprint
src/training/training.py                checkpoint manifest records the composition
src/training/prepared_bundle.py         bundle manifest records it; load refuses a mismatch
src/training/zero_shot_sims.py          was a 4th composition — routed through the SSOT
src/training/attribute_separation.py    NEW — per-attribute separation metric
scripts/show_model_input_comparison.py  was a 5th composition — routed; prints the active mode
scripts/export_atlas_embeddings.py      was a 6th composition — routed; stamps the mode
tests/test_model_input_contract.py      the contract tests
tests/fixtures/model_input_golden.json  855 frozen legacy rows (the rollback contract)
```

---

## 3. Independent blast-radius audit — what the implementer's map MISSED

The implementer's §10 map is good but incomplete. I audited the repo from the changed builder
outward myself (and re-derived every line reference before writing it here). Findings its map does
**not** contain:

| # | Surface | What it is | Affected? | Needs |
|---|---|---|---|---|
| M1 | `src/training/zero_shot_sims.py:354` | built the canonical encoder text itself, `strip_scope_words(canonical_model_text(...))`, under a comment claiming it was *"the SAME text the trainer encodes"*. It also had a **self-blinding staleness guard**: its fingerprint hashes the text *it* builds, so a profile flip left the fingerprint byte-identical and `results/training/embedding_similarities.csv` was **silently resumed** — stale sims served as valid, consumed by `evaluate_models.py`. | **YES — silently stale** | **FIXED**: routed through `core.model_input`; the fingerprint now moves with the text, so the sims invalidate correctly. |
| M2 | `scripts/export_atlas_embeddings.py:62` | built a 6th composition, calling the six-argument `clean_sku_text` with **two** arguments, so brand/description/category/breadcrumbs silently defaulted to `""` — matching *neither* profile. Wrote `embeddings.npy` + `metadata.csv` with no composition stamp. | **YES** | **FIXED**: routed through the builder, and the metadata CSV now stamps `model_input_profile` / `model_input_include_evidence`. |
| M3 | `result_prepared` bundles — `src/training/prepared_bundle.py` | `PreparedBundleManifest` had no composition field; `load_prepared_bundle` validated file sha, counts, `payload_variant` and `masking_profile` only. `train_prepared.py` **never calls the builder** — it takes `payload` straight from the pickle. All 55 bundles on disk predate the flip. | **YES — silently reusable** | **FIXED**: manifest is `schema_version 3` and carries the composition; `load_prepared_bundle` now refuses a composition mismatch by name. Older bundles are rejected **loudly** (`extra="forbid"`), not silently accepted. |
| M4 | `tests/test_mining_hypotheses.py:681-699` | re-implemented the payload composition to measure *"the texts the MODEL ingests"* — its docstring claim became false the moment the default moved. | test-only | **FIXED**: routed through the shared builder. |
| M5 | `tests/test_model_input_contract.py` (shared-builder guard) | asserted only that `predict_items` and `training.rand_matching` call the builder — which is exactly why M1/M2 survived. | test-only | **FIXED** by routing M1/M2, which the guard could not have caught. Noted as a residual: the guard is still module-scoped. |
| M6 | Nothing pins the **bytes of the shipped default** | the fixture pins legacy; every cleaned assertion is a property or inequality. `_cleaned_sku_text` could drift silently and the suite would stay green. | test-only | **OPEN, reported.** See §8. |

Verified **not** affected, with the reason (these are the "not affected" entries the implementer's
map lacks): `artifacts/data/embeddings_cache/*.npy` (content-addressed by the exact payload string —
`core/nlp.py:42-47`), `results/ann_index/catalog_id_mapping.csv` (identity map), `results/rand_truth/*`
(identity splits), `notebooks/*`, and the whole `src/ner/` lane (it consumes
`dataset_model_input.csv`, which belongs to a different lane).

**Inputs vs outputs, stated plainly:** `dataset.csv`, `artifacts/data/dataset_deduped.csv`,
`results/canonical_records.csv` and `results/gate_results.csv` are **inputs** — this change alters
how they are rendered into text, never their bytes. Everything under `training_results/`,
`submission/` and `results/ann_index/` is an **output**.

---

## 4. Traceability — executed evidence

Nothing new was invented: the composition rides the existing mechanisms (`core.tracing` + the
`training_trace` layout in `config/paths.yaml`, the checkpoint manifest, the ANN reuse fingerprint,
and now the prepared-bundle manifest).

Executed against the REAL payload stage on a 200-row frame under a throwaway project root
(`EUROMONITOR_PROJECT_ROOT` / `EUROMONITOR_RESULTS_DIR` redirected to `/tmp`; the repo's `results/`
was never written — confirmed clean afterwards):

```
$ PYTHONPATH=src python /tmp/traceability_smoke.py
active composition        : {"profile":"cleaned","include_evidence":false,
                             "fingerprint":"0d3c1d518e4424e0bd4c22ab245c73a5ffe4a023b70285e69c6b330ecf0e2694"}
config selected           : {"include_evidence": false, "profile": "cleaned"}
[payload-stage] building variant=full rows=200
[trace] pairs steps written -> /tmp/ER-trace-smoke/results/logs/training_trace.csv
composition rows on trace : 1
  stage=pairs step=payload.model_input_composition scope=run run_id=run-composition-smoke
  detail={"fingerprint": "0d3c1d51...", "include_evidence": false, "profile": "cleaned"}

PASS: the run trace names the exact active composition it produced.
persisted index           : results/ann_index
  stored preprocessing_fingerprint : None
  active fingerprint inputs include model_input: True
  load REJECTED as required -> ValueError: persisted HNSW metadata is stale
RESULT: traceability evidence complete.
```

So: a reader of the run trace can state which encoder-text contract produced a given run, and a
persisted index built from a different composition is **rejected**, not reused.

---

## 5. No coverage gaps — executed evidence

Run against the live tree (`/tmp/coverage_gap_proof.py`), all checks PASS:

```
== A. datapoint-population registry vs producer sources ==
  scanned producer tags (9): ann_finetuned, attribute_conflict, gate, gate_positive, hard_positive,
                             masked_positive, random_easy, targeted_attribute_conflict, unknown
  registry               (8): the same minus 'unknown'
  declared fallback tags (3): hard_neg, hard_negative, unknown
  [PASS] scanned - registry - fallbacks is empty
  [PASS] registry == scanned - fallbacks (no stale entry)
  [PASS] registry knows every emitter
  [PASS] negative-source census enumerates every registered negative producer
  [PASS] dynamic populations are derived, not literal

== B. runtime guard on an unregistered population ==
  [PASS] unregistered tag raises UnregisteredDatapointPopulationError
  [PASS] coverage artifact was still written before the raise (rows=7)

== C. the audit visits EVERY registered population ==
  coverage rows = 8; populations = [the 8 registered]
  [PASS] every registered population has a coverage row

== D. the composition vocabulary is closed and recorded ==
  [PASS] cleaned + include_evidence=false validates
  [PASS] unknown profile is rejected
  [PASS] cleaned+evidence is rejected
  active composition: {'profile': 'cleaned', 'include_evidence': False, 'fingerprint': '0d3c1d51...'}
  [PASS] active composition is a validated boundary object

RESULT: ALL CHECKS PASSED
```

Interpretation, stated precisely rather than as a slogan:
* the **emitted set equals the registered set**, in both directions (a scanned-but-unregistered tag
  and a registered-but-unemitted tag are both failures);
* `'unknown'` is scanned but is a **declared fallback**, not a population — the negative control in
  B shows the runtime raises loudly for a genuinely undeclared tag, **after** writing the coverage
  artifact so the evidence survives;
* the per-fold coverage audit writes a row for **every** registered population, so a registered
  population that contributes zero pairs still appears (status `missing`/`not_reached`), which is
  the property that makes "nothing drops silently while the audit reports success" checkable;
* the composition vocabulary is closed by the same discipline: an unknown `profile` cannot pass
  config load.

**The change emits no new datapoint population.** What it newly emits is a *provenance* value, and
that value is registered in the existing mechanisms (§4) rather than in a new registry.

---

## 6. PART 2 — the ANN arm

### 6.1 What was measured, and how

No training, CPU only. The recovered early-stopped checkpoint
`training_results/0915T063500554948Z/worker_2/_checkpoints/all-MiniLM-L6-v2/*/checkpoint-44/`
was loaded and used to embed real rows (`trainer_state.json` shows `best_model_checkpoint` = that
directory, `best_global_step` = 44, `max_steps` = 170 → the run early-stopped at step 44, so this
**is** the run's best model).

Harness `/tmp/ann_variant.py`, one pass per composition over the run's own artifacts:
* the 5 518 source rows and all 13 250 canonicals of `worker_2/canonical_records.csv` are embedded
  with the checkpoint at `max_seq_length: 128` (the training SSOT value);
* structured features are fused exactly as the lanes do (`model_input_info` → text **and** vector,
  `fuse_numpy` at `embedding_weight: 0.35`);
* every one of the 6 898 labelled holdout pairs is re-scored;
* retrieval recall@k is computed over the **full** 13 250-canonical catalog with each source row's
  own canonical as the target;
* the attribute-bucket error rates use the report's own definition
  (`generate_training_report.py`'s bucket rule) — the baseline `attribute_error_breakdown.csv` was
  reproduced **exactly** from the run's pair dump (all six rows, error rates to 6 decimals) before
  the harness was used to judge anything.

Compositions compared: `legacy + evidence` (what the checkpoint was trained on) vs
`cleaned` (the shipped default), both from the committed tree at `e326c46`.

### 6.2 The result, threshold-fair

Judging a composition at a threshold fitted to the *other* composition is the trap here, so each is
shown at its **own** Youden-optimal operating point as well as at the legacy-fitted threshold.

**At each composition's own Youden optimum:**

| metric | legacy | cleaned | Δ |
|---|---|---|---|
| Youden J | 0.6874 | **0.7688** | **+0.0814** |
| AUC | 0.9271 | **0.9520** | +0.0249 |
| precision | 0.9576 | **0.9747** | +0.0171 |
| recall | 0.8353 | **0.8579** | +0.0226 |
| F1 | 0.8923 | **0.9126** | +0.0203 |
| false merges (FP) | 204 | **123** | **−81 (−39.7 %)** |
| missed merges (FN) | 909 | **784** | −125 (−13.8 %) |
| operating threshold | 0.6909 | 0.7317 | **higher, not lower** |

**Attribute error rates at that operating point** — the acceptance-criteria buckets:

| bucket | n | legacy | cleaned | Δ |
|---|---|---|---|---|
| `pack/0` | 172 | 0.3779 | **0.2500** | −0.1279 |
| `volume/0` | 925 | 0.1005 | **0.0519** | −0.0486 |
| `none/0` | 111 | 0.2523 | **0.2252** | −0.0270 |
| `none/1` | 5477 | 0.1636 | **0.1413** | −0.0223 |
| `flavor/1` | 41 | 0.3171 | **0.2439** | −0.0732 |
| `multiple/0` | 172 | 0.1047 | **0.0407** | −0.0640 |

**Retrieval recall over the full catalog** (5 518 queries):

| k | legacy | cleaned | Δ |
|---|---|---|---|
| @1 | 0.7555 | **0.7769** | +0.0214 |
| @5 | 0.9067 | **0.9543** | +0.0476 |
| @10 | 0.9368 | **0.9732** | +0.0365 |

### 6.3 The verdict on the composition change

**It HELPS the ANN arm, and it does not require lowering the threshold** — the optimal operating
point moves *up* (0.6909 → 0.7317), which is the opposite of buying recall by lowering the bar.
Every acceptance bucket improves, false merges drop 39.7 %, and retrieval recall rises at every k.

Two honest qualifications, because they matter:

1. **Absolute levels are not comparable to the published baseline.** My local stack
   (torch 2.14 / sentence-transformers 6.0.1 / transformers 4.53.2) does not reproduce the Colab
   numbers (`transformers_version: 5.16.1` in the checkpoint's own `config.json`); a control run of
   the production entry point on the *unmodified* legacy code reproduced only 42.5 % of the baseline
   `NEAREST_ITEM_ID` assignments. So the **paired A/B under one harness is the evidence**; the
   absolute error rates are not. This is stated rather than papered over.
2. **The published retrieval-recall baseline is degenerate.** `ranking_hits_at_k.csv` reports
   `0.99837 @1`, but the handoff document already records why: 90.5 % of holdout queries have
   exactly **one** candidate in the pool, so an oracle and a constant scorer produce identical
   numbers. My harness computes recall against the full 13 250-item catalog, where the honest
   legacy number is **0.7555 @1**. The acceptance criterion "keep retrieval recall at or above
   baseline" is therefore measured against a metric that cannot fail; the replacement above is the
   meaningful one.

### 6.4 The checkpoint: stale and non-comparable — RETRAINING required

`checkpoint-44` was trained on the **old** text. Nothing in the code refuses it: `predict_items` /
`RandMatcher` will encode the new text and score it against those weights (the ANN fingerprint
guards the *index*, not the weights). The engineering consequence:

* every metric in `training_results/0915T063500554948Z/worker_2/report/` describes a model+text pair
  that no longer ships → **retrain, then regenerate**;
* the measurement in §6.2 is exactly that mismatched configuration, and it is *still* an improvement,
  which is the strongest available local statement: the composition change is not a regression even
  before retraining;
* the checkpoint manifest now records `model_input`, so after the retrain the two checkpoints are
  distinguishable.

### 6.5 Acceptance criteria — status

| Criterion (`ANN_ERROR_PRESENTATION.md`) | Status | Evidence |
|---|---|---|
| Improve `pack/0` | **IMPROVED (local)** | 0.3779 → 0.2500 at each composition's own operating point |
| Improve `none/0` | **IMPROVED (local)** | 0.2523 → 0.2252 |
| Improve `volume/0` | **IMPROVED (local)** | 0.1005 → 0.0519 |
| Preserve over-merge at 0 % | **NOT MEASURABLE LOCALLY** | over-merge is an assignment-level statistic; it needs the calibration gate + full pipeline, which is a training-lane run. The nearest local proxy, false merges (FP) at the matched operating point, **falls 204 → 123**. Claiming 0 % preserved would be a claim I did not run. |
| Do not lower the global threshold | **SATISFIED** | the optimal threshold rises (0.6909 → 0.7317); no recall was bought by lowering it |
| Keep retrieval recall ≥ baseline | **IMPROVED (paired, same harness)** | @1 0.7555 → 0.7769, @5 0.9067 → 0.9543, @10 0.9368 → 0.9732. The published 99.84 % @1 is degenerate (single-candidate pool); see §6.3(2). |
| Improve `flavor/1`, `multiple/0` (not in the list, reported anyway) | **IMPROVED (local)** | 0.3171 → 0.2439, 0.1047 → 0.0407 |

**Achieved locally:** every string-level and scoring-level improvement above, with the existing
checkpoint, CPU only.
**Requires Colab retraining (out of scope, not faked):** any claim about the fine-tuned model's
quality, the assignment-level over/under-merge rates, and the final submission.

---

## 7. What I committed, and the push

Committed by this session (in addition to the implementer's `bfe539d`, `c0b4d35`, `27b1cb0`,
`e326c46`):

* `FINALIZATION_REPORT.md` — this document.
* `.gitignore` — **decision:** the pnpm footprint (`node_modules/`, `pnpm-lock.yaml`,
  **and `package.json`**) is ignored, not tracked, with an explanatory comment. `package.json`
  contains only a `packageManager` pin and nothing in this Python project reads it; committing it
  would make the repo look like a Node project. The owner said "yes on strays"; **if you would
  rather keep the pin, delete the `package.json` line from `.gitignore` and commit the file.**
  `node_modules/` is never tracked.
* `SESSION_HANDOFF_2026-09-15.md` — restored to the real handoff document (the working tree had it
  overwritten by a stray agent brief) and **updated with a new final-state section**.
* `AGENT_BRIEF_MATCHER_ROUTING.md` — the stray brief preserved under its own name.
* `PRESENT.md`, `ANN_OLD_NEW_RUN_MANIFEST.md`, `colab_retrieved/` — the remaining meaningful
  untracked files, now committed.
* The blast-radius fixes M1-M4 and the corrections C1-C3.

`results/` is **clean** (`git status --porcelain -- results/` empty); no regenerated
`results/*.csv` was committed, and `dataset.csv` was only ever opened read-only.

Push: `git push ER training` — see the final report line below.

---

## 8. EXECUTED vs READ

**EXECUTED** (CPU only; no training, no GPU, no Colab):
* `pytest tests/ -q` on the committed tree before my fixes (**1 failed**, 312 passed, 2 skipped) and
  after (**341 passed, 2 skipped**).
* `/tmp/verify_legacy_head.py` — re-derived the golden fixture from an isolated `2a15852`
  extraction: 855/855 byte-identical.
* `/tmp/verify_agent_a.py` — legacy byte-identity through the new builder (0/0 mismatches) and the
  585-pair string claims.
* `/tmp/coverage_gap_proof.py` — the registry / guard / coverage / vocabulary proof in §5.
* `/tmp/traceability_smoke.py` — the real payload stage on a 200-row frame, trace read back from
  disk; plus the stale-index rejection in §4.
* `/tmp/ann_variant.py` — the CPU A/B in §6 on the recovered checkpoint (two full passes, ~9 min
  each); the baseline `attribute_error_breakdown.csv` reproduced exactly from the run's pair dump
  first.
* `/tmp/ann_out/AB_SUMMARY.txt` — the threshold-fair comparison artifact.
* An independent read-only blast-radius sweep by a separate agent, re-checked line by line before
  inclusion in §3.

**READ only (not executed):** `ANN_ERROR_PRESENTATION.md`, `ANN_ERROR_REVIEW_0915T085806480410Z.md`,
`config/training_ANN.yaml`, `src/core/ann_config.py`, the `worker_2/report/*` artifact set,
`training_results/0915T075044186132Z/worker_1/report/human_review_features/*`,
`src/training/hnsw_index.py`'s load contract, `src/training/complete_colab_worker.py`.

**NOT run, deliberately:** `train.py`, any fine-tuning, any GPU work, the real end-to-end pipeline
(it rewrites `results/`), and `colab_retrieved/0915T085806480410Z` (that run was torn down at step
102/700, so it has no post-training predictions or ANN report — not a defect).

---

## 9. Open, uncertain, or needing an owner decision

1. **Retraining is required** before any ANN-arm claim is final. §6.4.
2. **Nothing pins the bytes of the shipped `cleaned` default.** `tests/fixtures/model_input_golden.json`
   pins the *legacy* contract; the cleaned assertions are properties. A silent drift in
   `_cleaned_sku_text` / `_cleaned_canonical_text` would pass the suite. Closing it means capturing a
   second frozen fixture from `e326c46` and asserting byte equality against it; I did not do that
   because regenerating fixtures from the code under test is exactly the anti-pattern the first
   fixture exists to avoid, and it deserves the owner's explicit sign-off on *what* is frozen.
3. **The shared-builder guard is still module-scoped.** `test_both_lanes_call_the_shared_builder`
   counts calls in `predict_items` and `training.rand_matching` only. It could not catch M1/M2, and
   it still would not catch a new hand-built composition elsewhere. A repo-wide scan for the legacy
   idioms outside the legacy arm would close it.
4. **Absolute ANN numbers do not reproduce the Colab baseline** (stack version skew, §6.3(1)). If
   the owner wants exact comparability, the harness should be run on the Colab image.
5. **`results/canonical_records.csv` (tracked) and the run-scoped
   `worker_2/canonical_records.csv` are different files** (22 vs 18 columns, different `canonical`
   strings). Every report regenerated from a run bundle therefore describes a different canonical
   text than the committed input. Pre-existing, not introduced here, but it means "regenerate the
   report" is not by itself enough — the input has to be pinned too.
6. **`artifacts/embeddings/*.npz`** are orphaned: no writer and no reader anywhere in the tree, and
   stale under the new composition. Harmless today; worth deleting or documenting.
7. Multiple agents were writing to this checkout during the session (a third was running in
   `/home/opc/ONE/ER-analysis-brand-input`). The tree was re-verified green at the moment of commit;
   if another agent continues afterwards, that verification no longer covers its output.
