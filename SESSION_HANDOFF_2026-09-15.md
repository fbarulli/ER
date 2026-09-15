# Session handoff — commit 346f401 review, fixes, and consolidated traceability

Owner session date: 2026-09-15. Base of all work: **`2cfd772`** (branch `verify/ssot-346f401`,
worktree `/home/opc/ONE/ER-verify-346f401`), whose parent is **`346f401`** (the commit under review,
branch `training`).

**FINAL CONSOLIDATED TIP: `09a8ca8`+ (tag `session-2026-09-15-final`), 247 tests passing (2 skipped).**
Every agent branch is merged into `verify/ssot-346f401`; `results/canonical_records.csv`,
`results/gate_results.csv` and `dataset.csv` are verified byte-identical to `346f401`, and
`results/` plus every worktree are clean. The bundle is current at 112M.

## How to resume in a fresh session

```bash
# primary working worktree (has the fixes; 72 tests green)
cd /home/opc/ONE/ER-verify-346f401
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python -m pytest tests/ -q

# recover every branch from the bundle if anything is lost
git clone /home/opc/ONE/ER-346-branches.bundle /tmp/restore && cd /tmp/restore && git branch -a
```

Physical backup of every worktree **including uncommitted work**:
`/home/opc/ONE/_session_backup_20260915T132844Z` (path also in `/tmp/session_backup_path.txt`).

## Branches (all based on 2cfd772 unless stated)

| Branch | Worktree | State |
|---|---|---|
| `verify/ssot-346f401` | `ER-verify-346f401` | **`2cfd772`** — the reviewed/fixed tip, 72 tests green |
| `fix/346-trace` | `ER-346-trace` | `b882c16` WIP — CSV consolidation into `core.tracing` |
| `fix/346-pydantic` | `ER-346-pydantic` | `00e303c` WIP — schemas + trace row model + frame checker |
| `fix/346-minershape` | `ER-346-minershape` | at `2cfd772` (agent work not yet committed) |
| `fix/346-verify` | `ER-346-verify` | at `2cfd772` (read-only verifier) |
| `fix/346-ranking` | `ER-346-ranking` | at `2cfd772` (agent work not yet committed) |
| `fix/346-coverage` | `ER-346-coverage` | at `2cfd772` (agent work not yet committed) |
| `fix/346-routing` | `ER-346-routing` | at `2cfd772` (agent work not yet committed) |
| `verify/346-capture` | `ER-346-capture` | read-only verifier |
| `training` | `EuromonitoR` | `346f401` — untouched original |

Agents still running at handoff time: trace, pydantic, minershape, verify, ranking, coverage, routing,
capture. All work in **isolated worktrees with disjoint file ownership**; see "Ownership map" below so a
resumed session can re-dispatch without collisions.

---

## 1. Root causes found and fixed in `2cfd772`

Each was found by independent audit agents and re-verified by live-data replay before being fixed.

| # | Defect | Evidence |
|---|---|---|
| 1 | `pipeline.pack_gate` ran **before** the confidence checks, hard-rejecting pairs whose evidence the gate itself does not trust | 1,300 pairs `hard_no → fallback`; 80,777 of 87,804 blockers had pack/volume confidence < 0.85 |
| 2 | `pack_gate` compared pack counts and package types by set **inequality** | 145 pairs `proceed → hard_no` on overlapping packs (`{12,24}` vs `{12}`) |
| 3 | **Two** functions named `pack_gate`, opposite answers on identical input; `rand_matching` read the other one | `pipeline` True vs `core.attribute_conflicts` False on the same record |
| 4 | Targeted miner used **zero** volume tolerance, disagreeing with the label gate | `8002267004212/8002267025644` gate `proceed` but mined as a conflict |
| 5 | Miner re-added **same-canonical true matches** as label-0 pairs | 154 of 350 (44%) |
| 6 | `"Pack blocker:"` gate reason never registered in `REASON_PREFIX_TO_TYPE` | `sample_balanced_pairs` **hard-crashed** on the committed artifacts (7,967 rows) |
| 7 | `ranking_at_k_by_query` silently reported `hits_at_1 = 0.0` when `1 ∉ ks`; accepted empty/non-positive K | `ks=(5,10)` returned 0.0 for a population whose top candidate is relevant |
| 8 | `pulp` regex matched across phrase boundaries, **inverting meaning** | `"with Added Pulp, No Sugar Added"` → `no_pulp`; `"pulp 0.33l"` → `no_pulp` |
| 9 | Two path derivations used `__file__/__parents__` | `complete_colab_worker.py`, `run_raw_tcp_bridge_probe.py` |
| 10 | Three tests red at the parent commit (field markers + widened missing-attribute contract) | parent 66 passed → commit 3 failed; now 72 passed |

`2cfd772` also introduces `src/core/tracing.py` (the single consolidated trace), the
`config/paths.yaml` layout `training_trace`, and removes the scattered
`results/logs/*.csv` diagnostic writers.

## 2. Open defect NOT yet fixed — ranking metric is information-free

`ranking_at_k_by_query` is **arithmetically correct** (verified against an independent implementation over
4,000 randomized fixtures, and the call-site row alignment is provably right). The problem is the population
it is fed:

- **5,292 of 5,847 (90.5%) holdout queries have exactly ONE candidate** — their own positive. Max: 6.
- Therefore a **perfect oracle and an informationless constant scorer produce identical numbers**:
  `hits_at_1 = precision_at_1 = recall_at_1 = 1.0`, `recall_at_5 = recall_at_10 = 1.0`.

Mechanical cause: positives exist for every row, but negatives anchor only at the single representative row
per barcode (`gtin_to_row`, longest title), and `pairs_in_set` requires both endpoints in the test fold, so
only ~878 of 15,179 negatives survive. **Owner has asked for this to be fixed** (agent `ER-346-ranking`):
build a real per-query candidate pool using connected components — a query's fold-safe competitors are
canonicals from *other* components in the same test fold — with pool size strictly greater than
`max(evaluation.retrieval_ks)` or recall@K is trivially 1.0 again.

## 3. Open defect NOT yet fixed — matcher routes 99.94% to human review

`rand_matching.targeted_veto_gate` builds `missing` by looping over **all seven** `CRITICAL_ATTRIBUTE_DIMENSIONS`,
so a clean pair with any unknown dimension defers instead of auto-merging. Measured on the live artifacts,
over the 1,592 pairs the gate labelled `proceed`: **`human_review` 1,591 (99.94%), `auto_merge` 1 (0.06%)**.
`pulp_set` is populated on only ~2.3% of canonicals, so the joint requirement is nearly unsatisfiable and
`auto_merge` is effectively dead — despite the setting being named `missing_pack_or_volume_route`.
Assigned to agent `ER-346-routing` (verdict + fix, or a pinning test if it is intended).

## 4. Coverage gaps proven, assigned, not yet closed

- **`targeted_attribute_conflict` is invisible to the coverage audit.** `train.py:1054` emits that
  provenance name (350 rows/fold) but `training.py:109 KNOWN_DATAPOINT_POPULATIONS` does not list it, and the
  coverage loop only visits names in `configured_populations | expected | by_population`. The audit that
  exists to prove nothing drops silently is itself dropping them. → agent `ER-346-coverage`.
- **The two dataset loaders disagree on column contract.** `load_raw_export()` yields
  `gtin/sku_name_eng/attribute`; `load_dataset_deduped()` yields `barcode/title/attributes`;
  `run_within_brand_pipeline` requires `gtin` while `build_training_data` requires `barcode`, so the
  documented `deduped → pipeline` flow breaks. Assigned to agent `ER-346-trace` (instrument as a trace step).
- **`precision_at_k` means two different things** under the same column names: the per-query macro value in
  `training.py` vs the global micro value still live at `evaluate_models.py:278`. → agent `ER-346-ranking`.

## 5. Other agent findings worth acting on (from the four audits)

- **Column-schema name collision**: `canonical_attribute_info` returns `"flavor"` as a **str**;
  `structured_features.canonical_info` returns it as a **set**. Two builders of the same 7-dim record mapping.
- `canonical_attribute_info` **raises** on `volume_set=[0.0]` / `pack_set=[0]`, while
  `structured_features.canonical_info` correctly returns unknown — an asymmetry between two implementations
  of one mapping.
- **Round-trip is not reproducible**: 23.29% of committed canonical rows disagree with
  `extract_critical_claims` re-extraction (2,955 rows lose `sugar`; 41 rows fabricate claims from flattened
  token adjacency, e.g. `sweetener` + `sugar` n-grams read as the phrase "sweetener sugar").
- `pipeline.PHRASE_VARIANTS` remains a **second, un-consolidated** source for the same sweetener/pulp claims
  as `extract_critical_claims`.
- Honest uncaptured diet phrasings in live data: `low sugar` 890 rows, `less sugar` 42, `reduced sugar` 18,
  `0 sugar`/`sugar 0` 23, `sin azucar` 1. And `no sugar added` (261 rows) classifies as the stronger
  `no_sugar` although the commit's stated intent was to keep `no_added_sugar` separate.
- `scalar flavor` changed value on ~27% of source rows because the pick moved from first-in-text to
  alphabetically-first; the canonical **token set** is unchanged, so no gate decision flips from it.
- Dead code introduced by the commit inside `three_way_gate`: `_tokens` local, and the `No volume overlap` /
  `No pack overlap` branches are unreachable once `pack_gate` runs first.
- `KNOWN_DATAPOINT_POPULATIONS` and the `[negative-source]` print still omit
  `targeted_attribute_conflict` (folded into item 4).

## 6. Ownership map (keep disjoint when re-dispatching)

| Agent | Owned files |
|---|---|
| trace | `src/core/tracing.py`, pipeline trace call sites, `config/paths.yaml` layouts, `src/training/data_prep.py` |
| pydantic | `src/core/schemas.py`, `src/training/selftest.py`, `tests/test_trace_schema_contracts.py` |
| minershape | `src/core/hard_negatives.py`, `src/training/sample_balanced_pairs.py`, `src/training/rand_matching.py` |
| ranking | `src/training/training.py` holdout-eval region, `src/core/ranking_metrics.py`, `tests/test_ranking_pool.py` |
| coverage | `src/training/training.py` registry/coverage regions, `src/training/generate_training_report.py` |
| routing | `src/training/rand_matching.py`, `config/training.yaml` veto block, `tests/test_targeted_ann_gates.py` |
| verify / capture | read-only (`/tmp` scratch only) |

**Conflict to watch:** `minershape` and `routing` both need `src/training/rand_matching.py`; `ranking` and
`coverage` both touch `src/training/training.py` but in disjoint regions (eval pool ~2900-4600 vs registry
~100-125 and coverage ~2400-2470).

## 7. Verification recipe after integration

```bash
cd /home/opc/ONE/ER-verify-346f401
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python -m pytest tests/ -q          # expect 72+ passed
# live data-prep + trace, then RESTORE the artifacts (it rewrites results/)
PYTHONPATH=src /home/opc/ONE/EuromonitoR/.venv/bin/python -c "
from core.common import load_raw_export
from pipeline import run_within_brand_pipeline
g,c = run_within_brand_pipeline(load_raw_export())
print(g['gate_decision'].value_counts().to_dict(), len(c))"
git checkout -- results/     # ALWAYS restore afterwards
```

Expected regenerated live counts with the fixes: `hard_no 87,241 / fallback 46,791 / proceed 1,737`,
13,250 canonicals, from 71,623 raw rows (45,260 dropped by the GTIN guard, 3,715 of them on a failed GS1
checksum). The commit's own artifacts are `88,683 / 45,494 / 1,592` — the delta is exactly the 1,445 pairs
the confidence-aware gate re-routes.

**Never commit `results/*.csv` from a verification run** — they are regenerated artifacts; restore them.


---

## 8. Later findings (after the first handoff)

### Verified by the adversarial verifier (falsified two of my own claims)
- **BLOCKER (mine)**: deleting the `gate_visibility.csv` writer left `data_prep.py` still declaring it as an
  expected manifest output, so `finish_manifest` raised `FileNotFoundError` AFTER the stage did all its work.
  FIXED (`54a65c3`); verified `[manifest] data_prep complete … closure 71,623 == 13,250 + 45,260`.
- **The `pairs` trace block was unreachable**, placed after `build_training_data`'s only `return`. This is why
  the trace had no stage-2 rows. FIXED (`ad1214d`).
- **`sample_balanced_pairs` still fails** after the reason-registration fix: `Requested 3,000 rows, but balanced
  pool has only 986` — the POSITIVE side binds (`2*min(493, 7,967)`), not the composite reason.
- **The ranking degeneracy is total, not 90.5%**: a constant scorer scores `hits@1 = 1.0000` (tying an oracle)
  and the worst possible scorer still scores 0.9051.

### Mining hypotheses (minershape agent)
- **H1 label conflict REFUTED** — 0 (text_a, text_b) tuples carry both labels.
- **H2 confirmed + FIXED**: flavour conflicts were structurally unmineable (5,549 candidates, 0 reachable).
  Added a gate-confirmed flavour-variant name rule → emitted pairs 196 → **568**, base set fully retained, 0
  `proceed` rows relabelled. Pulp remains unreachable (78 candidates; a pulp rule would reach only 2).
- **H2b FIXED**: `normalized_product_name` leaked fused pack notation (`18x33cl`), blocking 115 candidates.
- **H3 CONFIRMED**: `target: 12000` is unreachable by ~21×; true ceiling **568**.
- **H6 FIXED**: the miner now returns a typed `MiningFunnel`; the trace emits one row per real filter.

### Capture delta (capture agent) — the premise was wrong, and it corrected me
The extraction changes belong to **`346f401`**, not `2cfd772`; the pinned pair `346f401 → 2cfd772` has exactly
ONE extraction change (the `no_pulp` branch). Measured both deltas on 71,623 rows:
- `8c44e72 → 346f401`: flavour **10,795 gained / 0 lost**; 2,502 pure pick-moves (not 8,462 — the rest also
  changed the token set).
- `346f401 → 2cfd772`: pulp 0 gained / 5 stopped / 11 changed — **16/16 CORRECT REJECTIONS, 0 regressions**,
  with an identity proof that the affected population IS the removed branch's match set (16 of 18).
  A **third inversion** was found: `"with … Pulp"` + a following `No-` claim across a `|` boundary (3 rows).
- **Sweetener split is faithful**: 0 false `no_sugar`; 826/826 co-emissions literally justified.
- **`gate_results.csv` never had `*_set` columns** (7 cols only) — corrects an earlier note.

### Open items the owner must decide on
1. **The committed artifacts are stale.** `canonical_records.csv` matches the BEFORE code 13,250/13,250 on all
   four dimensions and still carries 7 inverted `no_pulp` tokens; `gate_results.csv` is reproducible only from
   `346f401` code (proceed 1,592 vs 1,737 now). **Regeneration is required before any report from `results/` is
   trustworthy.**
2. **Blob flavour pollution**: only 2,824 of 22,659 blob-only flavour tokens come from a `Flavour:` field;
   2,756 come from non-flavour fields (`Water Type`, `Energy Source`, `Sweetener`, …) — e.g.
   `Caffeine + Water, Unflavored` → flavour `coffee`. A wrong capture is worse than a gap.
3. **Carbonation both-states**: 563 canonicals emit both `still` and `carbonated`; **485 are manufactured purely
   by union over member rows**, which no per-row guard can see.
4. **Honest uncaptured sweetener gaps**: `low sugar` 575 rows, `light/low calorie` 720, `unsweetened` 478,
   `less sugar` 16, `reduced sugar` 14, `no sweetener` 19.
5. **`rng.choice(..., replace=True)`** duplicates 11,602 negative rows at live scale while reporting the
   duplicated count as distinct data (`train.py:1107-1115`).
6. **Composite `Pack blocker` reason** collapses pack/volume/package_type (and 8.6% categorical) into one string;
   the balanced pool's `pair_type` stratum is a 3-dimension mixture no trace can decompose.


---

## 9. Calibration-gate review queue (owner question: which dataset features should gate)

**The question's population was the wrong one, and the queue is already gone.**

| population | size | what it actually is |
|---|---|---|
| pipeline `fallback` | 45,494 | pipeline LABEL gate undecideds (untrusted parsed volume/pack). NOT calibration items. |
| calibration candidates (pipeline `proceed`) | 1,592 | the real calibration population |
| calibration `human_review` BEFORE the routing fix | 1,591 (99.94%) | caused by deferring when ANY of 7 critical dimensions was unknown |
| calibration `human_review` AFTER the routing fix | **0** | `DEFERRAL_DIMENSIONS = ("pack", "volume")` — deferral now matches what `missing_pack_or_volume_route` names |

Verified on the committed artifacts: `auto_merge` is now **1,592/1,592**, 0 rejects, 0 review. The all-seven
requirement was unsatisfiable because `pulp_set` is populated on ~2.3% of canonicals; it was a rule bug, fixed
by the routing agent, not a missing-feature problem.

### The six unused dataset features do NOT qualify as gate criteria — for two independent reasons

**(1) There is no ground truth to validate any criterion.** Of the 43,684 evaluable pipeline-fallback pairs:
- identical GTIN both sides: **0**
- identical canonical text (the pipeline's own true-match rule): **55 (0.126%)**
- → **43,629 pairs (99.87%) carry no label of any kind.**

Statistical power, Wilson 95%: a 55-positive sample cannot certify a 0.5% error floor — the upper bound is
**0.065 even at ZERO observed errors**. Certifying ≤0.5% needs ~**600** labels (bound 0.0064) or ~**1,000**
(bound 0.0038). Any "feature X is a safe gate criterion" claim on current data is unfalsifiable.

**(2) The features do not separate the populations anyway.** Measured on 43,684 gate-fallback pairs vs a
20,000-pair gate-hard_no sample:

| signal | gate fallback | gate hard_no | verdict |
|---|---|---|---|
| `brand` equal | 99.6% | 99.9% | no signal — candidate pairs are same-brand by construction |
| `category_path` equal | 18.4% | 17.2% | no separation |
| `retailer` equal | 39.1% | 41.7% | no separation; and a cross-retailer pair is the SAME product |
| `country` equal | 88.4% | 77.3% | weak; same trap |
| `description` Jaccard | median 0.093 (p90 0.714) | median 0.077 (p90 0.500) | overlapping; p90 is the only hint |

**`price` is absent from the table deliberately: it cannot gate.** Price differences encode retailer/country
and promotions, not product identity, so a price criterion is a false-merge machine.

**Conclusion:** wiring any of `description`, `category_path`, `price`, `retailer`, `country`, `url`, `image_url`
into the calibration gate would add risk without resolving a queue that no longer exists. The preconditions are
(a) a populated calibration review population, and (b) ~600-1,000 human labels to certify a precision floor.
Until both hold, the correct action is **no new gate criteria**.

**Still worth doing independently:** the `both_equal` GTIN stratum bypasses the cosine threshold entirely, so a
perfect Rand Index there tests nothing — the calibration design flaw already recorded in `PRESENT.md`.

---

# 10. FINAL STATE — model-input composition session (2026-09-15, later)

This section supersedes §"FINAL CONSOLIDATED TIP" above for the current tip. The earlier sections are
kept because they still describe the `346f401` review and its open items.

## What shipped

The **encoder-text composition** is now one config-selectable contract instead of three copy-pasted
builders that had drifted apart (the source and target of the same product shared only ~0.51 of their
tokens).

```
config/training.yaml
  training:
    structured_features:
      implicit_pack_qty: 1.0     # unobserved pack, applied to BOTH sides by 'cleaned'
    model_input:
      profile: "cleaned"         # legacy | cleaned   <- SHIPPED DEFAULT
      include_evidence: false    # 'cleaned' excludes the evidence channel by definition
```

**Rollback is one config edit, never a code revert.**

| Want | Set |
|---|---|
| Shipped default (symmetric, compound-split, number-preserving) | `profile: "cleaned"` + `include_evidence: false` |
| Today's pre-change behaviour, byte for byte | `profile: "legacy"` + `include_evidence: true` |
| Pre-change composition without the description/breadcrumb channel | `profile: "legacy"` + `include_evidence: false` |
| *(rejected at config load)* | `profile: "cleaned"` + `include_evidence: true` |

`profile: legacy` + `include_evidence: true` is pinned byte-identical to the pre-change code on
**855/855** fixture rows (`tests/fixtures/model_input_golden.json`, captured from `2a15852` before any
edit and independently re-derived from that commit during verification). If the legacy path ever
drifts, `tests/test_model_input_contract.py::test_legacy_profile_reproduces_golden_bytes` fails loudly.

## Where the composition is recorded (traceability)

`core.model_input.model_input_composition()` returns a validated
`TrainingSpec.ModelInputComposition` (`profile`, `include_evidence`, `fingerprint`) and is written to
**four existing** places — no new mechanism was introduced:

1. the run trace, as the `payload.model_input_composition` run-scope row (`core.tracing`);
2. `checkpoint_manifest.json`, so a checkpoint names the contract its weights were trained on;
3. the prepared-bundle manifest (`schema_version 3`) — and `load_prepared_bundle` now **refuses** a
   bundle built under a different composition instead of training on the wrong payload;
4. the ANN index reuse fingerprint (`training.rand_matching.preprocessing_fingerprint_inputs`), so an
   index cannot outlive the composition that built it.

## Verified state at handoff

* Suite: **341 passed, 2 skipped** (baseline at `2a15852`: 290 passed, 2 skipped). The committed tree
  was briefly **red** (1 failed) when the pack-symmetry change landed without its test being updated;
  that test was rewritten to assert the true invariant and a second one added. Fixed, not silenced.
* `results/` clean; no regenerated `results/*.csv` committed; `dataset.csv` read-only throughout.
* Independent verification, blast-radius audit and the ANN measurement: see `FINALIZATION_REPORT.md`.

## What is now stale, and what needs retraining

* **Every checkpoint trained on the old text is stale and non-comparable** — including
  `training_results/0915T063500554948Z/worker_2/_checkpoints/.../checkpoint-44`. It still loads and
  still scores, but every metric it produced describes the other composition. **Retrain on Colab.**
  CPU measurement on that checkpoint anyway shows the switch is not a regression (see
  `FINALIZATION_REPORT.md` §6): every attribute error bucket and retrieval recall@1/5/10 improve once
  each composition is judged at its own operating point, and the optimal threshold moves **up**
  (0.6909 → 0.7249), so no recall was bought by lowering the bar.
* **Every report under `training_results/*`** — retrain, then regenerate.
* `results/ann_index/` — **rebuild** (automatic: the fingerprint check rejects the stale index).
* `results/prepared_training/*` (55 bundles) — **re-prepare**; the new manifest rejects them loudly.
* `results/training/embedding_similarities.csv` — **regenerate**; the zero-shot lane was a 4th
  hand-built composition with a self-blinding fingerprint, now routed through the SSOT.

## Corrections to the record

* The stray "MATCHER-ROUTING agent" brief that had overwritten this file is preserved as
  `AGENT_BRIEF_MATCHER_ROUTING.md`. This file is the real handoff.
* The pnpm footprint (`node_modules/`, `pnpm-lock.yaml`, `package.json`) is **ignored, not tracked**.
  `package.json` holds only a `packageManager` pin and nothing in this Python project reads it. To
  keep the pin instead, remove the `package.json` line from `.gitignore` and commit it.

## Open items carried forward

1. **Retrain the ANN arm** on the cleaned composition; the local measurement says it should improve,
   but that is a prediction, not a result.
2. **Nothing pins the bytes of the shipped `cleaned` default** — only its properties. A second frozen
   fixture captured from the shipping commit would close this; it needs an owner decision on what is
   frozen.
3. The **shared-builder guard is module-scoped** (it counts calls in `predict_items` and
   `rand_matching` only). It could not catch the zero-shot or atlas compositions, which were routed
   by hand during this session. A repo-wide scan would close it.
4. All items from §§2-5 and §9 above remain open unless explicitly closed in this session: the
   information-free ranking metric, the dead `auto_merge` route, the stale committed
   `canonical_records.csv` vs the run-scoped one, blob flavour pollution, carbonation both-states,
   and the uncaptured sweetener phrasings.
