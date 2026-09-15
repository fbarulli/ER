# Session handoff — commit 346f401 review, fixes, and consolidated traceability

Owner session date: 2026-09-15. Base of all work: **`2cfd772`** (branch `verify/ssot-346f401`,
worktree `/home/opc/ONE/ER-verify-346f401`), whose parent is **`346f401`** (the commit under review,
branch `training`).

**FINAL CONSOLIDATED TIP: `377676e`, tag `session-2026-09-15-final`, 247 tests passing (2 skipped).**
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
