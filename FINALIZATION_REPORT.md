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
| Youden J | 0.6874 | **0.7690** | **+0.0816** |
| AUC | 0.9271 | **0.9517** | +0.0246 |
| precision | 0.9576 | **0.9715** | +0.0139 |
| recall | 0.8353 | **0.8711** | +0.0358 |
| F1 | 0.8923 | **0.9186** | +0.0263 |
| false merges (FP) | 204 | **141** | **−63 (−30.9 %)** |
| missed merges (FN) | 909 | **711** | −198 (−21.8 %) |
| operating threshold | 0.6909 | 0.7249 | **higher, not lower** |

**Attribute error rates at that operating point** — the acceptance-criteria buckets:

| bucket | n | legacy | cleaned | Δ |
|---|---|---|---|---|
| `pack/0` | 172 | 0.3779 | **0.2791** | −0.0988 |
| `volume/0` | 925 | 0.1005 | **0.0627** | −0.0378 |
| `none/0` | 111 | 0.2523 | **0.2432** | −0.0091 |
| `none/1` | 5477 | 0.1636 | **0.1282** | −0.0354 |
| `flavor/1` | 41 | 0.3171 | **0.2195** | −0.0976 |
| `multiple/0` | 172 | 0.1047 | **0.0465** | −0.0582 |

**Retrieval recall over the full catalog** (5 518 queries):

| k | legacy | cleaned | Δ |
|---|---|---|---|
| @1 | 0.7555 | **0.7760** | +0.0205 |
| @5 | 0.9067 | **0.9565** | +0.0498 |
| @10 | 0.9368 | **0.9734** | +0.0366 |

### 6.3 The verdict on the composition change

**It HELPS the ANN arm, and it does not require lowering the threshold** — the optimal operating
point moves *up* (0.6909 → 0.7249), which is the opposite of buying recall by lowering the bar.
Every acceptance bucket improves, false merges drop 30.9 %, and retrieval recall rises at every k.

*(Consistency check: an earlier pass on a tree frozen just before the accent-folding commit gave
J 0.7688, FP 123, recall@1 0.7769 — the same direction and the same conclusion on every row of both
tables. Only the committed-`e326c46` numbers above are quoted as the result.)*

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

### 6.3b A related silent drop, closed after my measurement

A later commit (`f996718`) added `token_budget_report()` + a `payload.token_budget` trace row: the
`[FIELD_*]` tail is appended last, so at `max_seq_length` it was truncated first and silently
(11.9 % of target texts lost the whole tail, 27.7 % exceeded the window, and nothing recorded it) —
the same defect class as a coverage gap. It is **observational only**: the guard names and counts
every dropped field group and never rewrites the text, so the strings measured in §6.2 are still the
shipping strings, and the measurement is not invalidated. It does raise one honest caveat for the
retrain: the remedy (raising `max_seq_length`) is deliberately left unapplied because the worker_2
checkpoint was trained at 128.

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
| Improve `pack/0` | **IMPROVED (local)** | 0.3779 → 0.2791 at each composition's own operating point |
| Improve `none/0` | **IMPROVED (local)** | 0.2523 → 0.2432 |
| Improve `volume/0` | **IMPROVED (local)** | 0.1005 → 0.0627 |
| Preserve over-merge at 0 % | **NOT MEASURABLE LOCALLY** | over-merge is an assignment-level statistic; it needs the calibration gate + full pipeline, which is a training-lane run. The nearest local proxy, false merges (FP) at the matched operating point, **falls 204 → 141**. Claiming 0 % preserved would be a claim I did not run. |
| Do not lower the global threshold | **SATISFIED** | the optimal threshold rises (0.6909 → 0.7249); no recall was bought by lowering it |
| Keep retrieval recall ≥ baseline | **IMPROVED (paired, same harness)** | @1 0.7555 → 0.7760, @5 0.9067 → 0.9565, @10 0.9368 → 0.9734. The published 99.84 % @1 is degenerate (single-candidate pool); see §6.3(2). |
| Improve `flavor/1`, `multiple/0` (not in the list, reported anyway) | **IMPROVED (local)** | 0.3171 → 0.2439, 0.1047 → 0.0407 |

**Achieved locally:** every string-level and scoring-level improvement above, with the existing
checkpoint, CPU only.
**Requires Colab retraining (out of scope, not faked):** any claim about the fine-tuned model's
quality, the assignment-level over/under-merge rates, and the final submission.

---

## 6b. Verification of the implementer's LATER work (symmetry + separation metrics)

Added after the interruption described in §10. **Measured at HEAD `3fdc039`** — after this point
other agents continued committing (`4f5be42`, `c3cac78`, …) and began editing `src/pipeline.py`,
`src/training/training.py`, `config/training.yaml` again, so these numbers are pinned to that HEAD
rather than to whatever the tree holds now.

### Symmetry on REAL data (not fixtures)

`/tmp/symmetry_realdata.py`, 3 000 real source rows that have a canonical, both sides built through
the shared builder:

```
real rows audited: 3000
  presence agreement (both sides carry a pack token, or neither):
      legacy  :  119/3000 =   4.0%
      cleaned : 3000/3000 = 100.0%
  both sides UNOBSERVED : 2881
    (a) text   same implicit token on both sides : 2881/2881
    (a) vector same presence bit on both sides   : 2881/2881
  exactly one side OBSERVED : 60   (b) evidence never overwritten: 60/60
  both sides OBSERVED       : 59
RESULT: symmetric on real data, evidence preserved
```

So the claimed properties hold on real data, in **both channels**: where neither side observed a
pack, both now emit the same implicit token and both numeric vectors set the same presence bit; and
where one side *did* observe a pack, that observation is never overwritten by the default.

**The trap, recorded because it is the reason this defect survived so long:** "unobserved" must be
read from the parser's own confidence (source) and from the canonical `pack_set` (target) — **not**
from `sku_info`'s output. `sku_info` collapses "nothing observed" into a hardcoded `{1.0}`, so its
output is never empty and a symmetry check derived from it measures nothing. My first attempt made
exactly that mistake and reported 0 rows in the symmetric bucket; the numbers above are from the
corrected check.

**Legacy byte-identity was NOT sacrificed.** On the stabilised tree, `legacy` still reproduces the
frozen golden bytes on **855/855 rows × 2 sides** (0 mismatches), so the rollback contract survives
the symmetry rework intact. Measured, not assumed.

Side effect worth recording: on the final tree the earlier honest counter-result is **gone**. On the
24 `source_is_target_row` rows the cleaned Jaccard is now `0.5377` against legacy `0.5129` (it was
`0.4646` before accent folding + symmetry), and the review-band margin rose from `+0.2005` to
`+0.2157` (true `0.3754` vs cross `0.1597`).

### Separation metrics

`/tmp/separation_verify.py`, run on the real labeled-pair and canonical artifacts, **no model and no
training**:

```
spec: enabled=True min_pairs_per_class=30 min_value_support=20 flag_below=0.1
inputs: labeled_pairs=19918 canonical_records=13250
[separation] 8 attributes, 1,147 values scored, 98 flagged weak, 1,007 withheld for support

values under the support floor (20 in BOTH classes): 1007
  of those, flagged as defects: 0
    brand  100         n_pos=0 n_neg=1  separation=-1.0  reportable=False flagged_weak=False
    brand  5 alive     n_pos=0 n_neg=1  separation=-1.0  reportable=False flagged_weak=False
    brand  abant       n_pos=0 n_neg=1  separation=-1.0  reportable=False flagged_weak=False

real reporting path: generate_training_report._attribute_separation_section exists=True
  files written by the real path: ['attribute_separation_summary.csv', 'attribute_separation_values.csv']
RESULT: separation metrics verified
```

Answering the three things I was asked to check:
* **wired into the real reporting path** — yes, `generate_training_report._attribute_separation_section`
  runs and writes both artifacts into the report directory (executed, above);
* **a low-support brand is not reported as a defect** — 1 007 values are under the support floor and
  **0** of them are flagged; each is still reported with its counts. That is the property, and it holds;
* **runs without training** — yes: labelled pairs + canonical attributes only, CPU, no model.

## 7. What I committed, and the push

### Commits on `training` for this work

| commit | what |
|---|---|
| `bfe539d` | one config-gated builder for the encoder text in both lanes (the implementer) |
| `c0b4d35` | make the cleaned composition the default |
| `27b1cb0` | map the blast radius, close the coverage gap, stamp provenance |
| `e326c46` | universal pack symmetry, accent folding, attribute separation |
| `38358bf` | revert the brand analysis (handed to its dedicated agent) |
| `f996718` | truncation guard with a counted, traceable budget |
| `056fab8` | **this session** — finalize the session record; commit the remaining meaningful files |
| `addc705` | **this session** — the final ANN measurement + the truncation-guard caveat |
| `7c9f0ae` | **this session** — verify the symmetry rework on real data + the separation metrics |
| `95b330d` | **this session** — the project-wide rules passes and the reuse audit |
| `16e04e3` | **this session** — the encoder-text fingerprint did not cover the vocabulary |

Corrections C1-C3 and blast-radius fixes M1-M4 were in the working tree while the implementer was
still committing and were swept into `e326c46` by its `git add`; the content is what is described
here and the code is in that commit, not in a later one. Everything else in this table is committed
under its own message.

Committed by this session's own commits:

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

Push result (final):

```
$ git push ER training
To https://github.com/fbarulli/ER.git
   88e70b4..16e04e3  training -> training
```

`git rev-list --count ER/training..HEAD` afterwards: **0**. No force-push, no history rewrite.

---

## 7b. Project-wide rules — verification passes (all executed)

The instruction was to check the whole change against the project's rules. The premise given was that
this repo has **no** written conventions file. **It does.** `AGENTS.local.md` exists (5 230 bytes,
untracked by its own design — its header says the `.local.` form "lives on disk without being
committed. Do not commit it"). It is invisible to `git ls-files` and to a search for `AGENTS.md`,
which is presumably why it looked absent. I verified against **that** file, and I did **not** commit
it. (It is locally excluded via `.git/info/exclude`, which is not a tracked file.) The implementing
agent flagged the same discrepancy in its own report §25.

| # | Rule | Result | Evidence |
|---|---|---|---|
| 1 | **SSOT** — no second implementation left behind | **PASS, one documented residual** | `sf_text`, `self.structured_text` and `structured_append_to_text` (the per-lane gate recomputations) are **gone: 0 hits**. `_structured_text_enabled()` is the single gate and is reused by `scripts/analyze_model_input.py`. Residual: `scripts/show_model_input_comparison.py:107-126` still hand-builds the *legacy intermediate steps* for its before/after columns — labelled "explanatory only", and its two "exact model input" columns do come from the builder, so it cannot misreport the shipped payload; reported, not silently duplicated. |
| 1b | **SSOT** — the named SSOTs were reused, not re-implemented | **PASS** | see the reuse audit in §7c |
| 1c | one path derivation | **PASS with one justified duplicate** | `src/cli/colab_cli_entry.py:17` has its own `_find_project_root`, duplicating `core.common._find_project_root`. Its docstring states why: that wrapper is launched with the Colab CLI's interpreter, "which has neither the repo on `sys.path` nor its deps, so `core.common/TRAIN_ROOT` cannot be imported here". Justified and documented, not a defect. |
| 2 | **Transparency** — every number has its command | **PASS after one reproduction failure of my own** | I re-derived the golden fixture from `2a15852` (855/855), the 585-band margins, pack presence agreement, the separation counts, and agent A's `27.4% → 100%`. Its `18.8% → 82.6%` did **not** reproduce on my first two readings — it is **full 10-dimensional structured-vector equality over all 855 fixture rows** (`legacy 18.8% → cleaned 82.6%`), which I then reproduced exactly. My failure, not a fault in the number; the report simply does not state the definition, which is a one-line gap worth closing. |
| 2b | no silent drop / default / fallback in the changed paths | **PASS** | the three control-flow sites that looked suspicious are all sound: `token_budget_report` skips only in-budget texts and **counts** every out-of-budget one and every dropped `[FIELD_*]` group; `attribute_separation` skips only values with zero support in **both** classes and reports `n_values_withheld_for_support`; `zero_shot_sims:86` catches a `pd.isna` type error and still returns the value. |
| 3 | **No dead code / no bloat** | **PASS after 2 fixes** | `ruff check --select F,E9` over `src/ scripts/ tests/` diffed against the same run on an extraction of `2a15852`: **2 findings introduced by this work, 0 removed by it** (22 pre-existing repo-wide, unchanged). Both new ones — an unused `json` import and an unused `row` binding, both mine — are fixed. I also removed a genuinely dead accumulation in the same script (`targets`, 7 lines) whose only reader was an unused variable, plus two pre-existing dead symbols in `rand_matching.py` (an unused `sklearn` import and an unused local) while fixing §7d there. The repo-wide F/E9 count is now **17, below the 20-finding `2a15852` baseline, with zero new findings**. |
| 4 | **Pydantic at boundaries** | **PASS** | every new boundary structure is a pydantic model in `core.schemas` and is genuinely used: `TrainingSpec.ModelInputSpec`, `TrainingSpec.ModelInputComposition`, `AttributeSeparationSpec`, `TokenBudgetReport` (+ the two column tuples). `PreparedBundleManifest` gained a typed `model_input` field. No loose dict or tuple crosses a boundary. |
| 5 | **Config over constants** | **PASS** | every new tunable is in `config/*.yaml` read through `core.common.load_config`: `training.model_input.{profile,include_evidence}`, `training.structured_features.implicit_pack_qty`, `evaluation.attribute_separation.{enabled,min_pairs_per_class,min_value_support,flag_below}`, plus the two `paths.yaml` bindings. Scanned the new modules for numeric literals: the hits are indices, guards (`max_seq_length < 1`), docstring statistics and a CLI display width — no domain constant. |
| 6 | **Tests** — green, no coverage deleted | **PASS** | `pytest tests/ -q` → **383 passed, 2 skipped** (run with other agents' in-flight edits present, so it is a floor, not a ceiling). Per-file test-function counts versus `2a15852`: **no file has fewer**; totals 200 → 290. The one test I rewrote asserts a strictly stronger pair of invariants than the one it replaced, and a second was added. |
| 7 | **Repo safety** | **PASS** | `git diff HEAD -- results/` empty; `git diff HEAD -- dataset.csv` empty; no training, no GPU, no Colab (the only `cuda` string in the changed surface is a pre-existing CLI choice defaulting to `cpu`); other agents' uncommitted files were never staged. |
| 8 | **Reuse audit** | **PRESENT** — §7c | required by rule 9 of `AGENTS.local.md` |

Things I could **not** check, stated plainly: the suite result is a floor because other agents were
mid-edit; the ANN A/B was not re-run after their later commits; and rule 10's "every artifact that
becomes stale" can only be answered for the artifacts that exist on disk today.

## 7c. Reused vs newly created

**Reused** (existing SSOT, extended in place — nothing parallel was written):

| Existing thing | Where | Used for |
|---|---|---|
| `core.common.load_config` / `row_metadata_text` / `F` / `runtime` / `resolve_model` | `core.common` | every config read, every field fallback, every artifact path, model loading |
| `TrainingConfig` / `TrainingSpec` validation at load | `core.schemas` | fail-loud config, no call-site defaults |
| `core.tracing.TraceRun` + the `training_trace` layout | `core.tracing` + `config/paths.yaml` | the composition row and the token-budget row — **no second tracing mechanism** |
| `append_text` / `vector` / `fuse_numpy` | `core.structured_features` | the structured channel; `symmetric_info` was added **inside that same module**, not beside it |
| `normalize_text` / `MINIMAL_STOPWORDS` / `strip_schema_words` | `pipeline` | the one normaliser, the stopword vocabulary, the schema-word strip |
| `UNIT_CANONICALIZATION_VERSION` | `core.unit_canonicalization` | fingerprint input |
| `PersistentHnswIndex` | `training.hnsw_index` | the index reuse gate that now rejects a stale composition |
| `_attribute_separation_section` in the existing writer | `training.generate_training_report` | the separation metrics ride the real end-of-run report path |
| `load_ann_config` | `core.ann_config` | ANN settings |

**Newly created**, each with the grep that justified it:

| New | Why nothing existing would do |
|---|---|
| `src/core/model_input.py` | the composition existed as **five** copy-pasted builders; the `pipeline` primitives are the *steps*, never a composition point. Consolidation, not addition — net lines went down outside the new module. |
| `core.structured_features.symmetric_info` | no implicit-default rule existed anywhere; placed in the structured-features SSOT because that is where the one-sided sentinel lived. |
| `TrainingSpec.ModelInputSpec`, `TrainingSpec.ModelInputComposition` | config contract + artifact provenance record; `core.schemas` is the declared boundary SSOT. Replaced a loose dict. |
| `training.attribute_separation` (+ `AttributeSeparationSpec`) | the previous brand-separation figure (+0.0249) was produced once, by hand; a grep found no repeatable per-attribute/per-value metric. |
| `token_budget_report` + `TokenBudgetReport` | no truncation/token-budget helper existed; the truncation was previously invisible. |
| `preprocessing_fingerprint_inputs()` | a named accessor over a dict that was already inline — it makes the fingerprint's inputs auditable and testable. It is **not** a second fingerprint. |
| `implicit_pack_qty()` made public | it was `_implicit_pack_qty`, and the contract test needs to state the expected value; making it public avoided duplicating the config read in the test. |

## 7d. New gap found while running the rules checklist

### The ANN reuse fingerprint did not cover the normalisation vocabulary

`preprocessing_fingerprint_inputs()` documents itself as *"everything that changes the encoder TEXT
a persisted index was built on"*. It covered `structured_features`, `model_input` and
`unit_canonicalization` — but the composed text also depends on **`config/vocabulary.json`** twice
over: `core.model_input._normalized_tokens` filters with `MINIMAL_STOPWORDS`
(`config/vocabulary.json:MINIMAL_STOPWORDS`), and `strip_schema_words` uses the same file. So a
vocabulary edit changes the encoder text while leaving every fingerprint input identical, and a
persisted index built before the edit is **silently reused** — the same seam that was closed for
`model_input`, in a second input. It is not hypothetical: commit `88e70b4` had just changed
`config/vocabulary.json`.

Reproduced **before** the fix (`/tmp/vocab_fingerprint_proof.py`):

```
MINIMAL_STOPWORDS entries  : 99
composed-text hash         : ee18d78e99f1db58
ANN fingerprint            : 3bbb1f1c4b36c162
after a vocabulary edit:
composed-text hash         : d1be608c9d5746f2
ANN fingerprint            : 3bbb1f1c4b36c162
text changed               : True
fingerprint changed        : False
RESULT: SILENT STALE-INDEX REUSE REPRODUCED
```

**Fixed** in `16e04e3`: the vocabulary's sha256 joins the fingerprint inputs, reusing the existing
`core.common.VOCABULARY_CONFIG_PATH` and the existing `core.manifest.sha256_file` — no new path
constant, no new hasher. Pinned by
`test_ann_fingerprint_inputs_cover_the_normalisation_vocabulary`, which asserts both halves (the file
is in the fingerprint, and the vocabulary really does change the composed text).

### The conventions document — resolved

A `CONVENTIONS.md` was briefly believed to exist in the main checkout; `find / -maxdepth 6 -name
'CONVENTIONS.md'` returned nothing, and I declined to invent it (writing a file the owner believes
exists, or copying the rules into a second document, is the duplication §2 of those rules forbids).
The owner has since **consolidated it away to stop two copies of the same rules drifting apart**, and
the conventions live at their original home:

**`AGENTS.local.md`** — and that filename is load-bearing, not arbitrary. The harness probes
`AGENTS.md` / `CLAUDE.md` / **`AGENTS.local.md`** / `CLAUDE.local.md` at the project root and injects
a hint into every agent, subagents included, telling it to read and follow them. The `.local.` form
is the conventionally untracked variant, so the rules are **auto-discovered** by agents rather than
depending on a prompt happening to mention them.

`.gitignore` therefore ignores **`AGENTS.local.md` only** — `CONVENTIONS.md` is deliberately NOT
ignored, because it no longer exists and re-adding the rule would resurrect the duplicate that was
just consolidated away:

```
$ git check-ignore -v AGENTS.local.md
.gitignore:88:AGENTS.local.md	AGENTS.local.md          # resolves  (rc=0)
$ git check-ignore -v CONVENTIONS.md
                                                         # does not resolve (rc=1) - correct
$ git status --porcelain | grep AGENTS.local             # -> no output
$ ls -la AGENTS.local.md                                 # 5230 B, still on disk, not deleted
```

`AGENT_BRIEF_MATCHER_ROUTING.md` is a **different** file — the stray agent brief preserved under its
own name — and stays tracked, unchanged. The earlier machine-local `.git/info/exclude` entry was
removed now that the rule lives in the tracked `.gitignore`, so the behaviour is reproducible on any
clone instead of only on this machine.

## 7e. Single destination — `analysis/brand-and-input` is already on `training`

Owner rule: all work stays on the `training` branch of `/home/opc/ONE/EuromonitoR`; nothing may rest
anywhere else. The analysis agent works in an isolated worktree on a scratch branch
(`/home/opc/ONE/ER-analysis-brand-input`, `analysis/brand-and-input`).

**Result: nothing is stranded — the branch is already fully merged into `training`, and `training`
is already pushed.** Executed, not assumed:

```
$ git branch --list analysis/brand-and-input
+ analysis/brand-and-input
$ git log --oneline training..analysis/brand-and-input
                                        # -> EMPTY: no commits on the branch that training lacks
$ git merge-base --is-ancestor analysis/brand-and-input training && echo ancestor
ancestor
$ git rev-list analysis/brand-and-input --not training
                                        # -> EMPTY: no object on the branch is unique to it
$ git cherry training analysis/brand-and-input
                                        # -> no unique commits (checked by patch content too)
$ git merge --no-ff analysis/brand-and-input
Already up to date.                     # rc=0, no commit created
```

It landed through the merge commit **`98333b5 merge: brand and model-input analysis onto training`**.
Its deliverables are present and tracked in `training`:

| Artifact | State in `training` |
|---|---|
| `scripts/analyze_brand_matching.py` | tracked |
| `scripts/analyze_model_input.py` | tracked |
| `scripts/show_model_input_comparison.py` | tracked |
| `BRAND_ANALYSIS_REPORT.md` | tracked |
| `INPUT_ANALYSIS_REPORT.md` | tracked |

The analysis worktree itself carries **no uncommitted work** (`git status --porcelain` shows only an
untracked `training_results` scratch directory, i.e. data, not output). The branch was **not deleted**:
git refuses to delete a branch that a worktree has checked out, and the agent may still be running.
Nothing depends on it remaining, which is the property that mattered.

### The same sweep across every other branch (reported, not acted on)

Rule 4 taken literally — "do not leave any branch as the resting place for anything" — I checked
every local and remote branch. Everything that follows **predates this task** and is **not** part of
the analysis workstream, so I did not merge it; merging six unrelated WIP branches plus `main` on my
own initiative would be exactly the kind of blind resolution the instruction warned against.

| Branch | Commits not in `training` | By patch content (`git cherry`) |
|---|---|---|
| `analysis/brand-and-input` | **0** | no unique commits — **fully merged** |
| `fix/346-pydantic`, `fix/346-routing`, `fix/346-verify`, `verify/346-capture` | 1 each | **already in `training`** — the tips are WIP checkpoint commits whose changes landed via other commits |
| `fix/346-minershape` | 1 | **not in `training`** — prior-session minershape audit |
| `fix/review-41cd50e` | 1 | **not in `training`** — prior-session review checkpoint |
| `main` | 4 | README / notebook / visualization commits on the default branch |
| `ER/main` | 5 | as `main` plus a README link |

`fix/346-minershape` and `fix/review-41cd50e` are the only two carrying genuinely unlanded work, and
the handoff document already records them as the previous session's WIP (its ownership map lists
minershape as owning `core/hard_negatives.py` + `sample_balanced_pairs.py` + `rand_matching.py`).
**Owner decision, not mine:** merge them, or delete them as superseded. `main`/`ER/main` divergence is
a separate question about the default branch.

## 7f. The main agent's four files — verified, and already committed

The heads-up listed four files as **uncommitted**. They are **not**: all four are clean against
`HEAD` and landed in `3fdc039 fix(colab,dvc): keep-alive is CPU-only; .env is the single credential
origin` (with follow-ups `4f5be42`, `c3cac78`). So there was nothing left to sweep into a
finalization commit — **nothing was reverted, and nothing can be lost**: they are already in
`training` and already pushed. I verified the described behaviour is actually present rather than
trusting either the description or the commit message.

### Executed verification — 20/20 (`/tmp/verify_main_agent_files.py`)

```
== 1. syntax / imports ==
[PASS] parses: src/cli/colab.py
[PASS] parses: src/cli/colab_cli_entry.py
[PASS] parses: run_ann_full_data.py
[PASS] parses: src/training/dvc_store.py
[PASS] imports: cli.colab, cli.colab_cli_entry, training.dvc_store

== 2. keep-alive invariant ==
[PASS] GPU + --keep-alive is refused — ValueError: --keep-alive is CPU-only (a retained GPU VM
       consumes accelerator quota indefinitely); requested --gpu T4. Use --gpu CPU or drop it.
[PASS] CPU + --keep-alive passes the guard and publishes the marker
       — preflight reached=True  EUROMONITOR_KEEP_ALIVE_ALLOWED='1'
[PASS] CPU without --keep-alive publishes 0 — marker='0'
[PASS] _spawn_keep_alive denied when marker='0'      — "not marked keep-alive-eligible"
[PASS] _spawn_keep_alive denied when marker unset    — same

== 3. dvc credential ==
[PASS] _remote_owner derives the account from the configured url
       — 'https://dagshub.com/fbarulli/ER.dvc' -> 'fbarulli'
[PASS] _remote_owner raises on 'https://dagshub.com'   (no guessing)
[PASS] _remote_owner raises on 'https://dagshub.com/'
[PASS] _remote_owner raises on 'not-a-url'
[PASS] first call writes the credential (auth + derived user + token)
[PASS] second call re-applies a ROTATED token (the reported bug)
[PASS] the workspace was not re-initialised

== 4. no hardcoded account / no duplicate block ==
[PASS] no hardcoded 'fbarulli' in dvc_store.py
[PASS] exactly one remote-auth block (single _configure_workspace helper)
[PASS] both call sites use the shared helper

RESULT: 20/20 passed
```

**The keep-alive invariant fires, observed directly.** A GPU launch with `--keep-alive` raises
`ValueError: --keep-alive is CPU-only ... requested --gpu T4`; the identical CPU launch passes the
guard, reaches the preflight and publishes `EUROMONITOR_KEEP_ALIVE_ALLOWED=1`. The spawn point is
independently default-deny: with the marker `'0'` **and** with it unset, `_spawn_keep_alive` raises
`"refusing to start Colab keep-alive: this session was not marked keep-alive-eligible"` — and it
raises *before* `subprocess.Popen`, so no VM can be retained by a caller that skips the launcher.

**The rotation fix is proved against a real temp workspace** (local `dvc init` / `remote modify`
only — no network, the real remote was never contacted). `config.local` after the first call:

```
['remote "dagshub"']
    auth = basic
    user = fbarulli          <- derived from colab.dvc_remote_url, not hardcoded
    password = TOKEN_FIRST
```

and after a **second** call with a rotated token, on the same already-initialised workspace:

```
    password = TOKEN_ROTATED
```

The old token is gone and `.dvc/` was not re-created, which is exactly the reported bug: the
credential used to be applied only when `.dvc/` did not yet exist, so a rotated `DVC_API_KEY` in
`.env` was ignored for the life of the workspace.

*(One `FAIL` in my first run was **my** assertion, not the code — I grepped `config.local` for the
literal string `.dvc`, which is the directory name, not file content. Corrected and re-run: 20/20.)*

### `run_ann_full_data.py`, all three claims checked

```
-        "--keep-alive",                                              <- removed
-    print("... epochs=10 keep_alive=true", ...)                     <- was wrong
+    print("... epochs=10 keep_alive=false", ...)                    <- corrected
-It does not download or tear down the VM; use manual_download_results.py
+launcher tears the GPU VM down on every outcome: a retained GPU VM consumes
+accelerator quota indefinitely, so keep-alive is CPU-only ...       <- docstring updated
```

The only surviving `keep_alive` strings in that launcher are the docstring explanation and the
corrected printed line.

### Suite with these four files in the tree

`pytest tests/ -q` → **386 passed, 2 skipped**. `results/` and `dataset.csv` untouched.
`git status --porcelain` for all four files is empty, i.e. **committed and safe**.

## 7g. Concurrency resolution — verified, with three briefing claims corrected

The arrangement says agent A is paused, `e326c46` is the shared baseline and must not be
rewritten. **Confirmed on all three counts**: `e326c46` is an ancestor of `HEAD` and untouched, and
`HEAD` is `5b4c232`. Nothing of agent A's was reverted.

### Three claims in the briefing are stale — checked against the tree, not accepted

| Claim | Reality |
|---|---|
| "Still mine and still uncommitted: `.gitignore`, `run_ann_full_data.py`, `src/training/dvc_store.py`, `src/cli/colab_cli_entry.py`" | All four are **already committed and clean**. `.gitignore` → my `8d698e4`; the other three → `3fdc039` (and follow-ups). There was nothing left to sweep into a finalization commit, and nothing was rewritten. |
| "Agent A explicitly did NOT deliver the truncation guard" | **It did.** `token_budget_report` is live at `src/core/model_input.py:77`, delivered by `f996718 feat(model-input): truncation guard with a counted, traceable budget`, and it is the subject of §6.3b. |
| "suite 332 passed / 2 skipped" | Stale. Current: **386 passed, 2 skipped**. |

Band-conditioned separation and stage attribution (retrieval vs scoring) are genuinely **not**
present — noted as outstanding, not filled in (see §9).

### What I confirmed of agent A's claims — 9/9 executed (`/tmp/verify_agentA_claims.py`)

```
== (a) is pack the ONLY one-sided implicit default? ==
[PASS] pack is the only attribute with a one-sided implicit default — one-sided: ['pack']
       volume/package_type/flavor/carbonation/sweetener/pulp all return set() on BOTH sides
       when nothing is observed; only `pack` returns {1.0} on the source and set() on the canonical.
== (b) symmetric_info is scoped to cleaned; legacy untouched ==
[PASS] legacy leaves the source sentinel as-is
[PASS] cleaned applies the implicit default to an EMPTY canonical pack
[PASS] cleaned never overwrites an observed pack
== (c) text channel and numeric vector cannot diverge ==
[PASS] pack presence agrees between text and vector on all 585 pairs (x2 sides) — divergences=0
== (d) legacy byte-identity re-run NOW ==
[PASS] legacy reproduces the frozen bytes on 855x2 sides — mismatches=0
```

`pack-token agreement 27.4% → 100%` I had already reproduced **exactly** (§6b), and on 3 000 real
rows presence agreement goes 4.0% → 100.0%.

### (e) The accent-folding claim is TRUE AS HISTORY AND FALSE AS SHIPPED — and it left a lie in the code

I tested it directly and it failed:

```
'Brá̈mhults'  normalize_text -> ['br', 'mhults']   _normalized_tokens -> ['br', 'mhults']
'Reál'       normalize_text -> ['re', 'l']         _normalized_tokens -> ['re']
'Réal'       normalize_text -> ['r', 'al']         _normalized_tokens -> ['al']
```

The reason is a commit the briefing does not mention: **`38358bf revert(brand): hand the brand
analysis to its dedicated agent`** deliberately reverted the interim accent fix —

> *"revert the accent-folding normalisation in core.model_input: brand-string normalisation is not
> applied ahead of the diagnosis. The composition is back to normalize_text alone … folding it in
> today would change only 3 catalog clusters / 28 GTINs / 0 labelled pairs, while 378 of the 388
> review-band brand mismatches are outright different brands. Brand mismatch is a retrieval problem,
> not a string-composition one."*

So the corruption agent A measured is real and still present **by decision**, not by oversight — the
numbers are kept in its report §20.1 as input to that ruling. That part is fine.

**What is not fine:** the revert left the old explanatory comment behind, so `_normalized_tokens`
contained a comment claiming *"Fold accents BEFORE normalize_text. Without this every non-ASCII
letter becomes a word break … it CORRUPTS it"* sitting directly above code that folds nothing, while
its own docstring three lines above correctly says *"Diacritics are NOT folded here (owner ruling)"*.
A comment that contradicts its function and misdescribes the code is a defect under §3 (transparency)
and §4 (a comment that lies about what the code does). **Fixed** — the comment now states the
deliberate non-folding, names the ruling and the commit, and quotes the measured consequence. No
behaviour was changed: whether to fold is the owner's call and stays reverted.

### Its two headline numbers — both CONFIRMED, and both explained (`/tmp/sep_scrutiny.py`)

| attribute | n_pos | n_neg | separation | reportable | flagged |
|---|---|---|---|---|---|
| brand | 7330 | 12588 | **0.000000** | True | True |
| volume | 7330 | 12588 | +0.836969 | True | False |
| pack | 1086 | 4918 | +0.583639 | True | False |
| package_type | 1916 | 4499 | +0.253216 | True | False |
| **flavor** | 5997 | 9729 | **−0.086930** | True | True |
| carbonation | 6686 | 11622 | +0.029128 | True | True |
| sweetener | 4672 | 7214 | +0.061668 | True | True |
| pulp | 336 | 516 | −0.051426 | True | True |

**brand = 0.000 is exactly zero, and the reason is structural, not a metric bug.** Brand is observed
on both sides of **100%** of pairs in *both* classes and agrees on **100%** of them:
`P(agree | positive) = 7330/7330 = 1.0000`, `P(agree | negative) = 12588/12588 = 1.0000`. The
candidate population is same-brand **by construction**, so brand has zero variance and therefore
zero discriminative power here. Consequence worth acting on: **brand can never be a useful gate
feature on this population**, and the earlier "+0.0249 brand separation" figure was measuring noise
in a population where the answer is fixed by the sampling.

**flavor = −0.087 is confirmed, and it should NOT be read as "sharing a flavour makes a pair more
likely to be a mismatch".** I split each class by whether agreement was even possible
(`/tmp/flavor_mechanism.py`):

```
flavor  positive  both-observed=4661 (63.6%)  one-sided=1336  neither=1333  P(agree|both)=0.7775
        negative  both-observed=8059 (64.0%)  one-sided=1670  neither=2859  P(agree|both)=0.8345
        -> separation over the BOTH-OBSERVED subpopulation: -0.056955   (headline -0.086930)
```

So the headline decomposes into two parts:
* **≈ −0.030 is a coverage artefact.** A one-sided pair can never agree, and positives are one-sided
  *more often* (18.2%) than negatives (13.3%), which drags the positive agreement rate down for
  reasons that have nothing to do with flavour.
* **≈ −0.057 is real but is a property of the sampled population, not of flavour.** Negatives were
  mined as same-category, flavour-sharing near-neighbours, so *by construction* they share flavour
  more often than cross-country positives do. The same effect shows on `carbonation`: headline
  +0.029 collapses to +0.004 once both sides are observed.

Verdict: **CONFIRMED as arithmetic, REFUTED as an interpretation.** It is evidence about how the
negative pool was built, not evidence that flavour is anti-predictive — and it should be quoted with
the both-observed number beside it. (Per-value, the strongest negatives are `ginger` −0.2479,
`peach` −0.1502, `strawberry` −0.1017 — all popular flavours, consistent with the mining explanation.)

### Current state after this round

Suite **386 passed, 2 skipped**. `results/` and `dataset.csv` untouched. `e326c46` untouched.

## 7h. `rapidfuzz` declared — a latent import break closed

`scripts/analyze_brand_matching.py:66-67` imports `from rapidfuzz import fuzz` and
`from rapidfuzz.distance import Levenshtein`, and **the dependency was declared nowhere**:
`git show HEAD:requirements.txt | grep -ci rapidfuzz` → **0**. It worked on this machine only
because the venv happens to carry it, so any environment built from `requirements.txt` — including
the Colab VM path the file's own header says it drives — would fail at that script's import.

Approved by the owner and fixed. The header states *"The Dockerfile installs EXACTLY this file"*, so
an undeclared import is a real image break, not a formality.

Added, in the file's existing commented style, as its own section:

```
# ── analysis / audit scripts ───────────────────────────────────────────────
rapidfuzz==3.14.6      # scripts/analyze_brand_matching.py imports `fuzz` and
                       # `distance.Levenshtein`. AUDIT 2026-09-15: it was
                       # imported but declared nowhere, so the script ran only
                       # on machines whose venv happened to carry it and any
                       # image built from this file failed at import.
```

Verified, executed:

```
rapidfuzz pinned : 3.14.6          (the version actually installed in the venv)
rapidfuzz installed: 3.14.6        MATCH
occurrences of 'rapidfuzz' in the file: 1 pin — no second, conflicting declaration
any Levenshtein / fuzzywuzzy / thefuzz pin: none
fuzz.ratio 100.0 · fuzz.token_set_ratio 100.0 · Levenshtein.distance 1
  · Levenshtein.normalized_similarity 1.0      (the exact symbols the script uses)
analyze_brand_matching.py imports resolve OK
every importer of rapidfuzz in src/ scripts/ tests/: scripts/analyze_brand_matching.py (declared)
Suite: 386 passed, 2 skipped
```

### A pre-existing environment drift this surfaced (reported, NOT "fixed")

Validating the whole file against the venv, **all 16 other pins match**, but three do not — and they
are pre-existing, describing the **image** rather than this dev venv:

| pin | installed here |
|---|---|
| `pandas==3.0.5` | 2.3.3 |
| `transformers==5.16.1` | **4.53.2** |
| `psycopg[binary]==3.3.5` | not installed |

`uv pip check` adds the reason: *"sentence-transformers requires transformers>=5.0.0,<6.0.0, but
4.53.2 is installed"* (plus an aarch64/CUDA platform warning). I did **not** touch those pins — the
file documents the pinned image and the local venv is simply not that image. It does, however,
independently confirm the caveat already recorded in §6.3(1): **the local stack is not the declared
stack**, which is why the ANN A/B reproduced only as a paired comparison and not in absolute terms.

## 7i. Redundant-word removal in the model payload — measured, shipped, config-gated

Owner task: cut redundant tokens, canonical side first, config-gated, nothing assumed.

### The brief's numbers verified on the real 13,250 canonicals
```
canonicals 13,250 · total tokens 209,134 · mean 15.8 tokens/text
[FIELD_*] markers : 55,100 tokens = 26.3%          (brief: 55,100 = 26.3%)   ✓
still 57.1% + carbonation_still 57.1%,  P(twin|plain) = 100.0%              ✓
carbonated 29.1% + carbonation_carbonated 29.0%, P(twin|plain) = 99.8%      ✓
pack_qty_1 : 75.8% of texts (10,042 tokens)                                 ✓
```
One refinement: **8** tokens sit in the ≥50% band, not 4 — five are `[FIELD_*]`. Excluding markers it is exactly **4**, so the brief is right for non-marker tokens.

**`sugar` is NOT removable, by measurement not assumption.** There is no `sweetener_sugar` twin (0.0%); the corpus carries `sweetener_diet_sugar` (22.3%) and `sweetener_diet_no_sugar` (15.8%), whose suffix is `diet_sugar`, so plain `sugar` (57.3%) is not covered and is left alone.

### The paired ANN A/B — six variants, one protocol
Same harness as §6: recovered `checkpoint-44`, `max_seq_length: 128`, structured features fused
exactly as the lanes do, every one of the 6,898 labelled holdout pairs re-scored, **each variant
judged at its own Youden optimum** (judging at a threshold fitted to another variant flips the sign
of the conclusion — that is how §6 nearly went wrong). Both sides transformed, so the pack symmetry
is not silently reintroduced.

| variant | tok/text | AUC | Youden | thr | P | R | FP | FN | r@1 | r@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `cleaned` (shipped baseline) | 18.1 | 0.9520 | 0.7688 | 0.7317 | 0.9747 | 0.8579 | 123 | 784 | 0.7769 | 0.9543 |
| + no markers | 14.2 | 0.9553 | 0.7806 | 0.6871 | 0.9736 | 0.8755 | 131 | 687 | 0.7740 | 0.9516 |
| + no `pack_qty_1` | 16.6 | 0.9512 | 0.7702 | 0.7177 | 0.9733 | 0.8652 | 131 | 744 | 0.7811 | 0.9551 |
| **+ no markers, no `pack_qty_1` (SHIPPED)** | **13.4** | **0.9541** | **0.7786** | 0.6879 | **0.9751** | 0.8670 | **122** | **734** | **0.7784** | 0.9522 |
| + no redundant words | 16.0 | 0.9547 | 0.7771 | 0.7135 | 0.9764 | 0.8605 | 115 | 770 | 0.7454 | 0.9473 |
| + all three | 11.3 | 0.9545 | 0.7815 | 0.6598 | 0.9743 | 0.8735 | 127 | 698 | 0.7470 | 0.9402 |

Deltas against the shipped baseline:

```
no markers              tokens -3.9 (-21.6%)  J +0.0118  AUC +0.0033  FP  +8  FN -97  r@1 -0.0029
no pack_qty_1           tokens -1.6 ( -8.7%)  J +0.0015  AUC -0.0008  FP  +8  FN -40  r@1 +0.0042
no markers + no pack1   tokens -4.7 (-26.0%)  J +0.0098  AUC +0.0021  FP  -1  FN -50  r@1 +0.0014
no redundant words      tokens -2.1 (-11.8%)  J +0.0083  AUC +0.0027  FP  -8  FN -14  r@1 -0.0315
all three               tokens -6.9 (-37.9%)  J +0.0127  AUC +0.0025  FP  +4  FN -86  r@1 -0.0299
```

### The decision, and the one removal I did NOT ship

**Shipped: `emit_field_markers: false` and `emit_singleton_pack_token: false`.**
The combination dominates: **−26.0% tokens**, Youden `0.7688 → 0.7786`, AUC `0.9520 → 0.9541`,
FP `123 → 122`, FN `784 → 734`, precision up, and **recall@1 `0.7769 → 0.7784` — above baseline**,
so the standing "keep retrieval recall at or above baseline" bar is met.

**Not shipped: `keep_redundant_attribute_words` stays `true`.** Dropping the plain word whose
structured twin is present improves separation (Youden +0.0083) and lowers FP, but it costs
**recall@1 −0.0315, i.e. 3.15 pp BELOW baseline**, which breaks that same standing bar. The plain
flavor words (`lemon`, `apple`, `orange`, …) evidently carry lexical signal the `flavor_*` twin does
not fully replace. The flag remains independently selectable so the trade can be revisited — it is
turned off, not removed. This is the "measure with and without; if removing them costs separation,
keep them and say so with the number" instruction, answered with the number.

### Final measured reduction with the SHIPPED config (real corpus, end-to-end)

```
CANONICAL side (13,250 texts): 209,134 tokens (15.8/text) -> 143,992 (10.9/text)
                               removed 65,142 = 31.1%
SOURCE   side (3,000 real rows): 82,354 tokens (27.5/text) -> 67,072 (22.4/text)
                               removed 15,282 = 18.6%

before: monini coffee concentrate new improved version [FIELD_VOLUME] volume_ml_1000 [FIELD_PACK_SIZE] pack_qty_1 [FIELD_FLAVOR] flavor_coffee
after : monini coffee concentrate new improved version volume_ml_1000 flavor_coffee
```

### Config-gating, identity, and the ANN fingerprint

* Extends the **existing** `TrainingSpec.ModelInputSpec` — no second switch mechanism. Three
  positive-polarity booleans with today's bytes as their field defaults, each independently
  selectable so every removal can be A/B'd and rolled back by config alone.
* `legacy` **cannot** select a removal: the validator rejects it, because a flag silently ignored
  under the byte-for-byte stream is worse than one that fails. Pinned by
  `test_a_reduction_flag_is_refused_under_legacy`.
* The flags are part of **`ModelInputComposition`**, so they ride the run trace, the checkpoint
  manifest, the prepared-bundle manifest and the ANN reuse fingerprint. Verified: flipping any one
  moves the ANN fingerprint (`440a45e3191cd2e5` → `51822a03f8b00d63` / `07a75ccd3112a6ac` /
  `62198ed181f8516b`), pinned by `test_a_reduction_flag_changes_the_composition_fingerprint` at both
  the composition and the ANN level. Had they been left out, a flag flip would have changed the text
  while the index stayed "valid" — the exact seam §7d/§9-6b closed.
* `legacy` stays byte-identical: `test_legacy_profile_reproduces_golden_bytes` passes untouched
  against the frozen 855-row fixture.

### Test movement
**420 passed, 2 skipped** (was 407 before this change). One pre-existing test was legitimately
affected and updated with justification: `test_default_selection_is_reachable_without_any_config_argument`
asserted the no-argument path equals a constructor constant, which stopped being true once the
shipped config deliberately differs from the field defaults; it now asserts the built TEXT is
identical with and without an explicit spec — a stronger statement of its actual intent (that the
path resolves through `load_config`). No coverage was removed; the shipped-default pin was extended
to the three new flags.

## 7j. `validation_inference` → `final_inference` (config key), and final inference on GPU

Both owner changes landed after the payload/redundancy work was committed (`2b14035`), which was
the stated gate. **No collision materialised**: `src/cli/colab.py` and
`tests/test_colab_setup_path.py` were reported as held by the Colab agent, but at the time I got
here every target file was clean and committed, so I proceeded.

### A) The rename

The name was wrong: this lane scores the **whole** deduped catalog (61,529 rows) and uses the 3,000
held-out rows only for the threshold view.

| Site | Change |
|---|---|
| `config/training.yaml` | key `validation_inference:` → `final_inference:`; `output_dir` → `"final_inference"` |
| `config/paths.yaml` | layout `validation_inference:` → `final_inference:`, `template` → `"final_inference/{name}"` |
| `src/core/schemas.py` | `ValidationInferenceSpec` → `FinalInferenceSpec`; field `validation_inference` → `final_inference` |
| `src/cli/colab.py` | `_VALIDATION_INFERENCE` → `_FINAL_INFERENCE`, the `validation_inference: bool` parameters, the `..._enabled` preflight key, the guard message, the shell-completion clause |
| `src/training/complete_colab_worker.py` | 3 config accesses + the `trace_artifact` layout keys |
| `run_ann_full_data.py` | 2 config accesses |
| `tests/test_colab_setup_path.py` | the mock's `output_dir` + 4 `mock.patch.object(colab, "_FINAL_INFERENCE", …)` |

**The `trace_artifact` key had to move with the layout**: `trace_artifact(key, …)` records
`"layout": key`, so leaving it as `validation_inference` would have pointed the artifacts trace at a
layout that no longer exists. Verified: `artifact("final_inference", {"name": …})` resolves and
`"validation_inference" not in LAYOUTS`.

**Every surviving occurrence, named explicitly** (the acceptance test was "zero *config* hits, with
identifiers excepted and named"):

```
config/paths.yaml:114   owner: training.validation_inference      <- names the MODULE; must not rename
config/training.yaml:171  a comment saying why the block is NOT called that
src/core/schemas.py:1612  the same explanatory sentence in the docstring
src/training/complete_colab_worker.py:16   from training.validation_inference import ...
tests/test_validation_inference.py:12,15,86  module import, module-scoped class name,
                                             and the negative assertion that pins its absence
```

No `config/` hit remains that is a **key**. I agree with leaving the module
`src/training/validation_inference.py` alone: it holds genuinely validation-scoped helpers
(`resolve_best_checkpoint`, `threshold_assignment_metrics`), and renaming it would churn imports and
the DVC/artifact ownership declared in `paths.yaml` for no behavioural gain. Its test file keeps its
name for the same reason.

### B) Final inference on GPU, batched for 12 GB

```
device: "cuda"        # REQUIRED, not a preference
batch_size: 512       # 12 GB VRAM, inference only
max_batch_size: 1024  # enforced ceiling
min_vram_gb: 10.0     # refuse a smaller card
```

**Why 512.** The existing config assumes a 14.6-GB T4 for *finetune* batches and uses 64 there
(`training.batch_size_cuda`), because a finetune step must hold activations for the backward pass
plus Adam moments. Inference carries **no gradients and no optimiser state** and runs forward-only,
so the same class of card holds far more: MiniLM-L6 is ~22 M parameters (~90 MB fp32) and the
per-sample activation at `max_seq_length: 128` is small, so 512 — 8× the finetune batch — is a
conservative operating point on 12 GB, with 1024 as the hard ceiling.

**Three guards, so a mistake fails loudly instead of OOMing part-way through 61,529 rows:**
1. **At config load** — `FinalInferenceSpec._batch_fits_the_declared_ceiling` rejects
   `batch_size > max_batch_size`. A typo cannot reach a run.
2. **Before the encoder starts** — `complete_colab_worker._resolve_final_inference_device(cfg, override)`
   re-checks the ceiling, then requires a visible GPU, then requires the card to meet `min_vram_gb`.
3. **At the point of use** — `predict_items.py` refuses `--device cuda` when no GPU is visible rather
   than dying inside torch or quietly encoding on the CPU.

**The device cannot silently fall back.** `device: "cuda"` is treated as a requirement: with no GPU
the worker raises `RuntimeError: colab.final_inference.device is 'cuda' but no GPU is visible. Final
inference must not silently fall back to CPU; run on a GPU runner or set device: 'cpu'
deliberately.` The `device=` parameter on `complete_worker` remains the documented escape hatch, and
it excuses **only** the GPU requirement — the batch ceiling is still checked.

### Tests
**427 passed, 2 skipped** (was 420). New `FinalInferenceContractTests` (7 tests) pins: the old key is
absent and the new one resolves; the batch is larger than the finetune batch and within the ceiling;
an over-ceiling batch fails at config load; CUDA is used when a GPU is present (12 GB faked); a
sub-floor card is refused before encoding; no GPU produces the loud error naming the CPU override;
and the CPU override still enforces the ceiling. One legitimately-affected test was updated —
`test_completion_orders_inference_reports_provenance_before_dvc` exercises the completion ordering on
a CPU box, so it now asks for CPU through the documented override, which is precisely what the
override exists for. No coverage was deleted. Repo-wide lint: **0 new findings vs `2a15852`**, 17 vs
the baseline's 20.

## 7k. Training inputs moved to a single current root — plus two things I had to fix to make it true

Owner task: one root (`training_data/`) holding ONLY files we are certain are current, config pointed at
it, old copies gone. No collision on the files I needed (`src/cli/colab.py` was clean), so I proceeded.

### The root, and the resolver extension

```
config/paths.yaml
  paths:
    training_data_dir: "training_data"        # declared like every other root
  files:
    dataset_deduped:                  "training_data:dataset_deduped.csv"
    dataset_deduped_sample_3000:      "training_data:dataset_deduped_sample_3000.csv"
    dataset_deduped_train_minus_3000: "training_data:dataset_deduped_train_minus_3000.csv"
    canonical_records:                "training_data:canonical_records.csv"
    gate_results:                     "training_data:gate_results.csv"
```
`DataPathsSpec` gained `training_data_dir` (the config contract), and `_BINDING_ROOTS` in
`src/core/common.py` gained `"training_data": TRAIN_ROOT / _CFG["paths"]["training_data_dir"]` — ONE rule
beside the existing five, no hardcoded path anywhere. `dataset.csv` keeps its `repo:` binding and did
not move.

Tracked files moved with `git mv` so history follows; `git diff --cached -M` confirms the renames were
detected (`{results => training_data}/canonical_records.csv`, and 0-change pure renames for the other
two). The two derived splits were plain `mv` and stay untracked by the same policy as before, now with
matching rules for their new home; `.gitignore`'s allow-list entries for the old `results/` and
`artifacts/data/` locations were deleted as dead.

Consumers updated: `common.load_dataset_deduped` reads `F["dataset_deduped"]` (unchanged code, new
binding), `training/sample_deduped_dataset.py` now derives its defaults from `F[...]` instead of
spelling `TRAIN_ROOT / "artifacts/data/..."` by hand, `training/sample_balanced_pairs.py` likewise
(and gained the `F` import it needed), `run_ann_full_data.py`, `training/dedupe.py`'s docstring, and
`tests/test_validation_inference.py`.

**Verified:** zero live references to the old paths
(`git grep "results/canonical_records|results/gate_results|artifacts/data/dataset_deduped.csv|..."` over
`config/ src/ scripts/ tests/ run_ann_full_data.py` → empty). A real load through the config:
```
load_dataset_deduped() -> 61529 rows   from training_data/dataset_deduped.csv
canonical_records      -> 13250 rows
gate_results           -> 135769 rows
dataset.csv (raw)      -> still at the repo root
```
Suite **427 passed, 2 skipped**.

### FIX 1 — the "regenerated" artifacts were still the committed copies

The brief said `results/canonical_records.csv` and `results/gate_results.csv` had been regenerated. They
had **not**: both were **byte-identical to HEAD** (`gate d91363ce…`, `canon b044f2ef…`), and `gate_results`
still carried the OLD split `hard_no 88,683 / fallback 45,494 / proceed 1,592` — a fresh mtime from a
`git checkout -- results/`, which restores content without changing it. Moving them into a root defined as
"only files we are certain are current" would have put stale files in the one directory that exists to
prevent exactly that.

So I regenerated them (`python -m training.data_prep`), which now writes **through the binding** and landed
directly in `training_data/`. Result, matching the brief's expected NEW numbers exactly:

```
gate decisions: {'hard_no': 87241, 'fallback': 46791, 'proceed': 1737}   (expected 87,241/46,791/1,737 ✓)
canonical rows: 13,250      manifest closure 71,623 == 13,250 + 45,260
gate     d91363ce… -> 24478ab0…      canon b044f2ef… -> 566d6d83…
```

I also regenerated `dataset_deduped.csv` as instructed; it came out **byte-identical** to the 09-14 copy
(`3050b456…`), which confirms dedupe is deterministic on the unchanged raw export rather than merely
being assumed current.

### FIX 2 — ⚠️ a concurrent agent had staged the deletion of `dataset.csv`

While preparing the commit the index contained **16 staged deletions, 474,418 lines, including
`dataset.csv`** — plus `colab_retrieved/*`, `artifacts/data/number_tokens_reference.csv` and the
`analysis_outputs/*` CSVs. The files were all present on disk; only the index said deleted.

Cause: another agent added a blanket rule to the **same** `.gitignore` I was editing —
```
# Every CSV is generated or local-only; the tree ships code, not data.
*.csv
```
— and ran `git rm --cached` over the CSVs. That is incompatible with this task (`git mv` the frozen
inputs so history follows; `dataset.csv` "must not move") and with the standing repo-safety rules.

What I did, deliberately and without destroying anyone's work:
* `git restore --staged .` — the index is back to HEAD; **no file was lost**, all 16 are on disk and
  tracked again;
* staged **only** my paths, and verified `dataset.csv`, `colab_retrieved/`,
  `artifacts/data/number_tokens_reference.csv` and `analysis_outputs/` are **not** in the staged set;
* the other agent's `*.csv` line is **left exactly where it is in the working tree, uncommitted** — I
  staged a `.gitignore` without it rather than committing their in-flight change or reverting it.

**Owner decision needed:** the blanket `*.csv` rule and the four frozen inputs in `training_data/` are in
direct conflict. If "the tree ships code, not data" is the real policy, this migration must be reverted
in favour of it; if the four frozen inputs stay tracked, the `*.csv` line should be dropped deliberately.

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
6a. **Not delivered by agent A, outstanding:** band-conditioned separation (separation computed
   within score bands rather than over the whole pair pool) and stage attribution (whether an error
   originates at retrieval or at scoring). The briefing states these remain with agent A; I did not
   implement them and did not silently substitute anything for them.
6b. **CLOSED — the fingerprint now covers the composition CODE.** (Was: "data inputs but not the
   composition code".) The owner ruled for the **coarse** granularity, so
   `preprocessing_fingerprint_inputs` gained a `composition_code` component holding the digest of the
   two modules that produce the encoder text — `pipeline` (`SCHEMA_WORDS` at `src/pipeline.py:1249`,
   `_MODEL_STOP` at `:1292`, `normalize_text`, `strip_schema_words`) and `core.model_input`
   (`_normalized_tokens`). It reuses the existing `core.manifest.sha256_file`; no new helper, no new
   path constant, no second hashing mechanism.

   The new fingerprint input set is:

   ```
   structured_features   {enabled: ...}
   model_input           {profile, include_evidence, fingerprint}
   unit_canonicalization <UNIT_CANONICALIZATION_VERSION>
   vocabulary            sha256(config/vocabulary.json)
   composition_code      {core.model_input: sha256(...), pipeline: sha256(...)}
   ```

   End-to-end evidence (`/tmp/fingerprint_code_proof.py`, hermetic — the edited copies live in `/tmp`
   and are wired in via the modules' own path attributes, because `src/pipeline.py` was held by
   another agent at the time):

   ```
   baseline fingerprint : d381d84369d7066b
   (1) pipeline.py edited (file)  -> b892ea7447781dcc   changed=True
       the schema-stop symbol drives the text: ['cola','auditmarker'] -> ['cola']  changed=True
   (2) vocabulary.json edited     -> 8444598e22f8bfbd   changed=True
   (4) digests equal the real files: pipeline=True core.model_input=True vocabulary=True
   RESULT: a code/vocabulary FILE edit invalidates the index
   ```

   The old vocabulary reproduction was itself corrected while doing this: it originally simulated the
   edit by monkeypatching `pipeline.MINIMAL_STOPWORDS` **in memory**, which is not a file edit and so
   never moved a file-hash component. Re-run as a real file edit it now behaves:

   ```
   after a vocabulary FILE edit:
   composed-text hash : d1be608c9d5746f2   ANN fingerprint : 80f8dbb307f440c2
   text changed: True   fingerprint changed: True   -> no gap
   ```

   Pinned by `test_ann_fingerprint_inputs_cover_the_composition_code`, which asserts the digests equal
   the real files' digests, that they are stable across calls (a stable tree must not churn the
   index), that a modified copy digests differently, and that the schema-stop symbol genuinely drives
   the composed text. Cost accepted deliberately: an unrelated edit inside either module forces one
   **visible** rebuild — over-invalidation costs time, under-invalidation costs correctness silently.
7. Multiple agents were writing to this checkout during the session (a third was running in
   `/home/opc/ONE/ER-analysis-brand-input`). The tree was re-verified green at the moment of commit;
   if another agent continues afterwards, that verification no longer covers its output.

---

## 10. Contention episode — disclosed in full

The Phase-0 wait condition I was given was *"`MODEL_INPUT_FIX_REPORT.md` exists AND `git log` shows a
commit newer than `2a15852` touching the model-input code"*. **Both were already true when I started
waiting** (`bfe539d`), and agent A was in fact still working — it went on to make four more
substantial commits (`27b1cb0`, `e326c46`, `f38358bf`-series and `f996718`) and other agents were
active in the same checkout and in sibling worktrees. The condition was therefore too loose, and I
began writing before agent A had stopped.

What that cost, stated plainly:

* I edited files agent A was also editing (`src/core/model_input.py`, `src/core/schemas.py`,
  `src/training/prepared_bundle.py`, `src/training/zero_shot_sims.py`, `scripts/*`,
  `tests/test_mining_hypotheses.py`, `tests/test_model_input_contract.py`). None of my edits
  corrupted its work and none of its edits reverted mine, but that was luck, not process.
* Agent A's `git add` swept my working-tree edits into **its** commit `e326c46`, so the blast-radius
  fixes M1-M4 and the corrections C1-C3 are authored under its message rather than a commit of mine.
  The content is present and is described in this report.
* After the interruption I switched to committing **only files agent A could not be touching**:
  `.gitignore`, `SESSION_HANDOFF_2026-09-15.md`, `AGENT_BRIEF_MATCHER_ROUTING.md`,
  `FINALIZATION_REPORT.md`, `PRESENT.md`, `ANN_OLD_NEW_RUN_MANIFEST.md`, `colab_retrieved/`. Those are
  commits `056fab8` and `addc705`. I never staged `src/`, `tests/` or `config/` once the new rule
  was in force.
* I then re-ran the strict stability rule — **three consecutive checks at least 3 minutes apart with
  `git log -1` unchanged, no modified files under `src/`/`tests/`/`config/`, and no `.git/index.lock`**
  — before re-verifying and finalising. The tree did **not** stabilise on the first attempt
  (another agent began editing `config/training.yaml`, `src/cli/colab.py`, `src/core/schemas.py`
  again at 16:42); verification and the final commit were done after it did.
* Everything in §6b and the re-run byte-identity check were executed on the **stabilised** tree, not
  on a moving one.

Honest residual: the ANN A/B in §6 was measured on the tree committed at `e326c46`. Later commits
(`f996718` truncation guard, and the colab/dvc work) do not change the composed strings — the
truncation guard only counts and traces, and the colab/dvc work is outside the composition — so the
measurement stands, but it was not re-run after them.
