# LAYA_ADDON — where the laya classifier plugs into the ER pipeline

Design doc, grounded read-only in the tree at `main` (2026-10-10) plus the
in-flight `.worktrees/laya` lane refactor. Every claim carries `file:line`
evidence or a measured artifact census. No generic ML advice: each proposal
names the stage, the owner class, the concrete input→output, the cost, the
validation, and the role (blocker / verifier / reranker / calibrator).

---

## 1. TL;DR

**Laya is not a blocker and must not become one.** It is a typed-question
decision model over a *composed pair-state string* — it scores a pair (or an
anchor-vs-two-candidates choice), it cannot retrieve, and it emits no
embeddings on any surface this repo uses (§2). The rule gate it would "block"
for is free, deterministic, auditable, and is the label SSOT the whole
training side consumes (§3, §5F). The three highest-value plugs are:

1. **Fallback adjudicator** — the gate's `fallback` tier (18,592 pairs,
   measured) is quarantined from labels today; laya's fine-tuned
   `identity_claim` P(same) + its own abstention calibration turns part of
   that unknown mass into supervised pairs (§5A).
2. **Cascade decision verifier** — a Decorator over the GNN decider's scores
   in the near-threshold band only, catching GNN FP/FN before clusters form
   (§5B).
3. **Negative-supply auditor** — a sampled Strategy over the gate's
   `hard_no` population (115,432 pairs) that estimates label contamination
   of the negative class with a component-clustered CI, cheap (~2k pairs,
   one short T4 session) (§5C).

Calibration: **augment, never replace** — laya's `fit_temperature_map` /
`fit_abstention_thresholds` calibrate *laya's own* scores (already wired in
the lane, `EvalCalibrationSpec`); the ER gate/threshold path (Youden-on-DEV,
`fixed_threshold` 0.55, calibration carve) stays the tracks' SSOT (§4.5, §5D).

---

## 2. What laya ACTUALLY is (verified inputs/outputs)

Evidence from the lane that runs it — this is the contract every proposal
below is built on:

| surface | call | what it returns | evidence |
|---|---|---|---|
| decision agent | `laya.load(checkpoint, device)` → `agent.predict_batch(states, questions, min_confidence=…)` | one typed answer per (state, question); `min_confidence` gates low-confidence answers | `src/cli/laya_lane.py:1381-1385`, `:1396` |
| pair score | `laya.train.load_checkpoint` → `items_from_rows` → `calibration_records(model, tok, items, …)` → softmax over the `identity_claim` logits, `p[1]` | **P(same item)** per pair — the `gtin1,gtin2,score` convention `scripts/laya_compare.py` joins on | `src/cli/laya_lane.py:3201-3220` |
| calibration | `laya.train.fit_temperature_map(records)` → per-type temperature + per-bucket map; `fit_abstention_thresholds(records, temperature, target_error, min_bucket_n)` → per-bucket `min_confidence` gate | temperature map + abstention thresholds; consumed, never reimplemented | `src/cli/laya_lane.py:2494-2506`; spec `src/core/laya_config.py:91-135` |
| metrics | `laya.train.evaluate_records(records)` | accuracy / loss / **ECE / Brier** on a split | `src/cli/laya_lane.py:2225-2237` |
| local CPU twin | `LayaLocalEvalRunner` (`load_checkpoint` on CPU + `calibration_records` + `fit_eval_calibration`) | same `eval_report.json` offline | `.worktrees/laya/src/cli/laya_local_eval.py:24-50` |

**The state is the input.** A pair becomes ONE string: the six frozen
identity-slice fields (volume / pack / package_type / sweetener / flavor /
carbonation) rendered side-by-side `field: v1=[…] v2=[…]`, joined with `"; "`
— composed by `scripts/laya_metrics_pairs.py` `compose_side`/`compose_state`,
reused (not copied) by the lane via `_pairs_composer()`
(`src/cli/laya_lane.py:2980-2988`, holdout use `:3030-3032`). A field absent
from a side renders `''` — **unmeasured, never invented**. Consequence: a
pair whose decisive difference lives outside the six fields (package_material,
pulp, sweetening…) is *invisible to laya* — the corpus builder already counts
and skips exactly those counterfactuals (`scripts/laya_build_dataset.py`
docstring, "collapses to two identical sides and is SKIPPED and COUNTED").

**The questions are typed** (`config/laya.question.json`, SSOT; 17 questions):
the headline three are `attribute_alignment` (choice:
aligned/mismatched/obscure), `identity_claim` (**noul: "does the paired
attribute evidence support the two rows carrying the same grocery item"** —
the owner's core question, verbatim), `package_state` (noul). Also declared:
`field_same:<attr>` ×6, `pack_volume_equal`, `pack_format_equivalent`,
`gate_verdict` (choice: proceed/hard_no/fallback), `gate_reason`,
`counterfactual`, `same_brand_only`, `evidence_sufficient`, and
**`better_match`** (choice: "one anchor listing and two candidate matches —
which candidate is the better match") — the only native *group-vs-single*
question (§4.4).

**No embeddings.** Nothing in `src/cli/laya_lane.py`, the worktree lane
factories, or `scripts/laya_*` calls an encode/embed surface of the laya
package (grep over both trees). Laya cannot populate a `CascadeIndex`
(`src/model_tracks/cascade.py:64`) or an ANN. "Verify the embeddings"
therefore means: verify the *decisions/scores* the embeddings produce (§5B),
not the vectors.

**Cost anchors (measured, not guessed).** Single pinned T4
(`LayaSpec.gpu: Literal["T4"]`, `src/core/laya_config.py:253`; device pin
`src/cli/laya_lane.py:1361-1369`); base checkpoint 647 MB travels as a hosted
dataset (`laya_config.py:172-180`); batch 8 (decision) / 16 (holdout-eval)
(`laya_config.py:229,238,264`); the 2026-10-08 run measured **512.7 s for
the 1,758-row holdout eval + tar tail** on 2xT4 (`laya_config.py:255-262`
comment) — order ~0.3 s/pair all-in at batch 16, so 18.6k pairs ≈ 1.5–2.5 h
in one T4 session. Decision rows are capped at 2,500 per run today
(`laya_decision_max_rows`, `laya_config.py:267`).

**Training data (why circularity matters).** The fine-tune corpus is built
FROM the gate: `hard_no` rows are the identity negatives (552 sampled of
6,947 available), `proceed` rows are positives (22 — all of them),
`fallback` rows are **quarantined to `data/laya/unknown_pairs.csv`, never in
the corpus**; growth from the pipeline's own mask/counterfactual audits
(1,200 mask cases + 7,901 aug pairs, 7,875 counterfactual) and 26
labeled_pairs rows (`data/laya/receipt.json` census; builder docstring
`scripts/laya_build_dataset.py:1-60`). Laya is thus a *student of the gate* —
using it to re-decide gate output is partially self-referential and must be
validated on the component-disjoint holdout, never on gate agreement alone
(§8, open question 3).

---

## 3. The pipeline as laya sees it (stage map, real numbers)

```
brand blocking → rule gate → gate_results.csv → labeled_pairs.csv → tracks → cascade → clusters
  (pipeline)      (pipeline)    134,365 pairs      pos/neg + buckets   text|gnn   (worker)  (masking)
```

| stage | owner | what flows | evidence |
|---|---|---|---|
| candidate generation | `PipelineStages.brand_blocking_gate` — all-pairs inside a brand | 134,365 candidate pairs (measured on `data/gate_results.csv`) | `src/pipeline.py:3910-3953` |
| **the gate** | `three_way_gate` → `_ThreeWayGate.from_config(...).decide()` — deterministic volume/pack/flavor decision table; **every training label flows through it** | verdicts measured: `hard_no` 115,432 / `fallback` 18,592 / `proceed` 341 | `src/pipeline.py:2139-2156`; the scalar-loop + label-SSOT ruling `src/pipeline.py:3938-3950` |
| gate write | `sort_validate_write` — `pair_id` via `PairIdentity.column`, frame contract `check_gate_results_frame`, atomic CSV | `gate_results.csv` keyed by the direction-independent pair id | `src/pipeline.py:4034-4083`; `src/core/pair_identity.py:26-71` |
| pair similarity | `identity_similarity` — canonical-word Jaccard | the `similarity` column the label split cuts on | `src/core/pair_policy.py:51-56` |
| **labeled pairs** | `GateSplit`/`SplitLedger` — exact four-bucket partition: pos = `proceed & sim≥0.50`, hard-neg = `hard_no & sim≥0.80`, **`fallback` stays OUT by design** (counted, never labeled), below-threshold counted | `labeled_pairs.csv` | `src/training/labeled_pairs.py:1-27,80-92`; thresholds `config/training.yaml:1161,1166` |
| gate replay | `gate_replay.replay(fired=…)` — re-runs the gate over `gate_results.csv` for counterfactual reason analysis | moved-pair census | `src/training/gate_replay.py:168-232` |
| text track | trained encoder → ANN retrieval geometry | embeddings + `PersistentHnswIndex` | `src/model_tracks/cascade.py:16-20` |
| GNN track | two-hop typed **listing→attribute→listing** message passing; `PairScorer` = cosine calibration over `gnn_only` embeddings, `forward(embeddings, pairs, text) → logits` | the precision-oriented decider | `src/graph_tracks/model.py:1-5,193-230`; pairs contract `sku_id1,sku_id2,label,split` `src/graph_tracks/train.py:143,202` |
| negative supply for scored halves | `ScoredNegativeSampler` — samples REAL gate `hard_no` pairs into the graph scored dev/test pairs (the labeled source ships only twelve negatives); **`gate_results.csv` owns the FPR floor** | scored pair files | `src/graph_tracks/setup.py:374-395,545-549` |
| **cascade** | retrieve-then-rerank COMBINATOR: text ANN `rank()` → `pair_batch()` → GNN `decide()` (sigmoid → score + order); runs per query with `k = len(ids)-1` (whole catalog) | `Decisions(query_ids, candidate_ids, scores, order)` | `src/model_tracks/cascade.py:152,189,219-240`; caller `src/model_tracks/worker.py:144-189,262-289`; track vocabulary `src/model_tracks/resume.py:24` |
| threshold calibration | DEV carved into `calibration_fit`/`calibration_reserved` by whole components; Youden threshold picked on DEV applied to TEST; ship threshold is the config SSOT `fixed_threshold: 0.55`; ANN mining band is `calibrated_ann_band` | operating point(s) | `src/training/folds.py:768-846`; `src/training/training.py:7047-7060,7169`; `config/training.yaml:694,713`; `src/core/ranking_metrics.py:81`; `src/core/hard_negatives.py:1857`; `src/training/ann_refresh.py:137` |
| **clusters** | `build_entity_cluster_map` — transitive closure (`nx.connected_components`) over positive pairs + shared-GTIN chains → `CLUSTER_xxxxxx`; fold/validation components via `DisjointSet` union-find | cluster ids, component folds | `src/training/masking.py:507-546` (caller `src/training/train.py:965`); `src/training/robust_validation.py:147-157` |
| post-training ablation | inference-only ablations after publication, frozen-threshold binding | per-track reports | `src/model_tracks/post_training_ablation.py:1-40` |
| laya holdout (already built) | `scripts/laya_holdout.py` → `data/laya/holdout.csv` (1,758 rows / 1,042 components; 604 labelled = 568 pos / 36 neg; strata: real_listing 576, gate_fallback 1,132, gate_proceed 22, p0_* 28); scored IN-SESSION by the `fbarulli/er-laya-holdout-eval` kernel; compared with component-clustered bootstrap CIs per gate stratum | the honest laya-vs-tracks read | `data/laya/holdout.receipt.json`; `src/cli/laya_lane.py:3275-3350`; `src/core/holdout_eval.py:79-193`; `scripts/laya_compare.py:1-30`; `scripts/laya_verify.py:1-30` |

### 3.1 Data freshness (measured today — affects every number below)

`data/gate_results.csv`, `data/labeled_pairs.csv`, `data/final_validation.csv`
and `data/track_setup/listing_pairs.csv` were all **regenerated 2026-10-10**
(file mtimes 10:38–10:47); every laya artifact predates that regeneration —
`data/laya/receipt.json` + `data/laya/holdout.receipt.json` are dated
2026-10-09 and were built against the PREVIOUS gate file (their census:
`gate_hard_no_available: 6,947`, `gate_fallback_quarantined: 1,132`,
`gate_proceed_available: 22` — vs the current file's measured 115,432 /
18,592 / 341). Consequences, in order:

1. The fine-tuned checkpoint was trained on the OLD gate population; the
   current fallback tier (18,592) is a *larger, partly unseen* population.
2. The holdout (`data/laya/holdout.csv`, gitignored + DVC-backed JSONLs)
   must be rebuilt (`scripts/laya_holdout.py`) and the corpus re-minted
   (`scripts/laya_build_dataset.py`) before any §5 validation number means
   anything against the current data.
3. The gate tallies cited in this doc (134,365 / 115,432 / 18,592 / 341) are
   the CURRENT measured artifact; the laya receipt censuses are historical.
   Both are labeled as such wherever cited.

This is a prerequisite line item in §7, not a footnote: rebuild → re-run
`holdout-eval` → only then trust 5A/5C numbers.

---

## 4. The owner's framing, answered directly

### 4.1 "Are these items related?" — the core question

Laya answers it *literally*: the `identity_claim` noul question is worded
"does the paired attribute evidence support the two rows carrying the same
grocery item (same physical product, not merely the same brand or category)"
(`config/laya.question.json`). Operationally the answer is
`softmax(identity_claim logits)[1]` = P(same) (`src/cli/laya_lane.py:3213-3220`),
thresholded at `holdout_eval_threshold: 0.5` (`src/core/laya_config.py:240`)
or gated by a fitted abstention `min_confidence` (§5D). Two honesty limits,
both structural:

- **Evidence limit:** laya sees only the six composed slice fields (§2). It
  answers "does the *evidence in the state* support identity" — a pair whose
  decisive attribute is outside the six fields is unanswerable, and
  `evidence_sufficient` exists precisely to say so.
- **Truth limit:** the P0 population (`data/final_validation.csv`) remains the
  only fully product-disjoint truth; laya scores join it through
  `scripts/laya_compare.py`'s order-insensitive pair key. **Join-key ruling
  for any integration:** use `PairIdentity.of/column` (`src/core/pair_identity.py:54-71`)
  as the ONE key — `laya_compare._pair_key` re-spells the ordering with
  `normalize_gtin` (`scripts/laya_compare.py:34-37`); a landed integration
  should route through `PairIdentity`, not add a third spelling.

### 4.2 NODES and EDGES — what laya can verify

- **Nodes:** nothing. Laya's unit of work is a *state string*; a single
  listing state only supports the per-row questions (`attribute_alignment`,
  `package_state`) — useful as a node-quality census (is this listing's
  attribute evidence aligned/measurable at all?), not as node identity.
- **Edges:** everything. Every pair-bearing artifact in the pipeline is an
  edge set keyed by `PairIdentity`: `gate_results.csv` (134,365 edges),
  `labeled_pairs.csv` (pos/neg edges), the graph tracks' scored pair files
  (`sku_id1,sku_id2,label,split`, `src/graph_tracks/train.py:143`), and the
  cascade's per-query decided candidate edges (`Decisions`,
  `src/model_tracks/cascade.py:123`). Laya verifies an edge by scoring its
  composed state; it **proposes** edges only over a population someone else
  enumerated (the quarantine file, the cascade candidate set) — it has no
  retrieval surface (§2).

### 4.3 CLUSTERS / NEIGHBORS — where laya assists

Clusters form by **transitive closure over accepted positive edges**
(`src/training/masking.py:507-546`) and union-find components
(`src/training/robust_validation.py:147-157`, `src/training/folds.py`). So:

- one laya-**verified** positive edge can merge two components — a false
  positive is an *over-merge* that propagates transitively; a false negative
  is an *under-merge*. Both rates are already measured metrics
  (`calibration_over_merge_rate` / `calibration_under_merge_rate`,
  `src/training/cluster_quality_plot.py:25-28`) — that is the validation
  surface for every laya edge proposal below.
- The asymmetry to respect: laya verdicts should enter cluster formation
  **only through a labeled/verified edge artifact with provenance**, never as
  a direct union-find call — the closure stays deterministic and replayable.
- **Neighbors:** the cascade's `decide()` produces the neighbor order
  (`scores`, `order` per query, whole-catalog k, `src/model_tracks/worker.py:178-189`).
  Laya reranks/verifies the *band*, not the matrix (§5B) — scoring all
  k=|catalog|−1 candidates per query with laya would be the expensive
  duplicate of what the GNN decider already does for free at cosine cost.

### 4.4 GROUP vs SINGLE — using laya efficiently

- Native group question: **`better_match`** — one anchor + two candidates,
  ONE call picks the better neighbor (`config/laya.question.json`). That is
  1 forward instead of 2 `identity_claim` calls *and* it forces a comparative
  judgment the pairwise score cannot express. **Blocked today:** the corpus
  carries `better_match_cases: 3` (`data/laya/receipt.json`) — the checkpoint
  is effectively untrained on it (open question 1).
- Batch economics: states compose per pair; the kernel already batches
  (`predict_batch`, `calibration_records` batch_size 8/16). Efficiency lever
  that matters: **score populations, not matrices** — fallback tier (18.6k),
  sampled hard_no audit (~2k), cascade threshold band (~k per query at δ),
  never the 134k full gate (the rule gate already decided it for free) and
  never all-pairs retrieval (laya has no ANN).

### 4.5 Gate calibration — replacement or add-on?

The current "gate calibration" is three distinct mechanisms; laya maps onto
each differently:

| mechanism | today | laya as replacement? | laya as add-on? |
|---|---|---|---|
| rule gate decision table (`three_way_gate`) | deterministic, config-owned vetoes, label SSOT (`src/pipeline.py:2139`, ruling `:3938-3950`) | **No** — replacing it makes every training label a model output (circularity, §2) and loses auditability | Yes: adjudicate its `fallback` tier (§5A) and audit its `hard_no` tier (§5C) |
| operating threshold (Youden-on-DEV → TEST; ship `fixed_threshold: 0.55`; calibration carve `calibration_fit`/`calibration_reserved`) | `src/training/training.py:7047-7060,7169`; `src/training/folds.py:768-846` | **No** — it calibrates the *tracks'* cosine scores; laya has no say in their scale | Yes: when laya scores enter a decision (§5A/B), calibrate *them* with laya's own `fit_temperature_map` + `fit_abstention_thresholds` (already lane-wired: `EvalCalibrationSpec`, `src/core/laya_config.py:91-135`; kernel path `src/cli/laya_lane.py:2494-2506`) and report ECE/Brier via `evaluate_records` |
| abstention semantics | none in the gate; laya side already models three states `passed/abstained/unevaluated` with a pydantic accounting invariant (`src/core/eval_trace.py:315-335`) | — | Yes: the abstain tier is the honest output of §5A/B — an abstained pair stays in the quarantine bucket, never masquerades as a decision (fail-soft *with a recorded reason*, contract §3) |

Verdict: **add-on**. Laya's calibration stack is real and already consumed
(not reimplemented) — but it calibrates laya; the gate's determinism is a
feature the training side depends on.

---

## 5. Integration catalog

Role vocabulary: **blocker** (cheap pre-filter shrinking the candidate set),
**verifier** (second opinion on an existing decision), **reranker** (reorders
an existing candidate list), **calibrator** (fits the operating point /
confidence scale). Pattern vocabulary per `patterns.md`.

### 5A. Fallback adjudicator — laya over the gate's uncertain tier ★ top pick

- **Stage:** gate → labeled_pairs (between `sort_validate_write` and
  `GateSplit`).
- **Role:** verifier of the gate's non-decision; label *proposer* (never label
  SSOT).
- **Pattern:** **Strategy** — a `FallbackAdjudicator` selected by config
  beside the deterministic split; constructed by a factory in the laya lane's
  new DI structure (`.worktrees/laya/src/cli/laya_lane.py:24-30`: behavior
  lives in injected `laya_recipe`/`laya_runtime`/… factories — land it there,
  not in the monolithic `main`-tree copy).
- **Input:** the `fallback_gate_pairs` bucket (18,592 pairs measured). The
  composition half ALREADY EXISTS: the corpus builder's quarantine writer
  emits `data/laya/unknown_pairs.csv` with one **composed `attribute_pairs`
  state per fallback row** (`_gate_state`,
  `scripts/laya_build_dataset.py:974,1095-1111`), and the lane's reusable
  pair composer is `_pairs_composer()` (`src/cli/laya_lane.py:2980-2988`).
  Flow: quarantine rows → `calibration_records` (batch 16) →
  `identity_claim` P(same) + fitted per-bucket `min_confidence` (§5D).
- **Output:** a NEW artifact (immutability contract — never mutate
  `gate_results.csv`): `adjudicated_pairs.csv` keyed by `PairIdentity.column`,
  columns `pair_id, gtin1, gtin2, laya_score, laya_confidence, verdict
  ∈ {same, different, abstained}`, plus receipt. `GateSplit`'s partition gains
  one bucket (`labeled_pairs` frames already carry provenance discipline via
  the manifest, `src/training/labeled_pairs.py:12-23`); adjudicated pairs
  enter training labels ONLY with a `label_source=laya` provenance column and
  only above the fitted confidence — abstained rows stay quarantined and
  counted.
- **Cost/latency:** ~18.6k states ≈ 1.5–2.5 h on ONE Kaggle T4 session at the
  measured holdout pace (§2); one-shot per data-prep regeneration, same
  cadence as the gate loop itself (`src/pipeline.py:3938-3944`). Batch 16,
  `holdout_eval_batch_size` knob already exists (`src/core/laya_config.py:238`).
- **Validate — honestly, in three steps.** The catch (measured): the holdout's
  gate strata are **label-less** — its 604 labelled rows are exactly
  `real_listing` 576 + `p0_overlap` 19 + `p0_disjoint` 9
  (`data/laya/holdout.receipt.json`; staging skips unlabelled rows by ruling,
  `src/cli/laya_lane.py:3012-3014,3025-3029`) — so **no ground truth exists
  for fallback pairs anywhere today**; that is precisely why they are
  quarantined. Therefore: (1) *general trust*: rebuild the holdout (§3.1),
  run `fbarulli/er-laya-holdout-eval` and read labelled-stratum
  precision/recall/F1/PR-AUC with component-clustered CIs
  (`src/core/holdout_eval.py:121-193`) — plus `gate_verdict`/`gate_reason`
  agreement on the fallback stratum as a *consistency probe, never truth*
  (the holdout builder's own ruling, `scripts/laya_holdout.py:10-13`).
  (2) *probe truth*: an owner-labelled probe of ~100–200 sampled fallback
  pairs scored with clustered CIs → the adjudicator's measured error at the
  fitted abstention threshold (open question 9). (3) *downstream*: one
  ablation with/without the adjudicated labels — track PR-AUC at
  `fixed_threshold` + over/under-merge rates
  (`src/training/cluster_quality_plot.py:25-28`); the post-training ablation
  lane already reports per track
  (`src/model_tracks/post_training_ablation.py`).
- **Why it wins:** the fallback tier is 13.8% of the pair universe, is
  deliberately label-free today ("would inject label noise into both
  classes", `src/training/labeled_pairs.py:5-8`), and laya's abstention
  calibration makes "I don't know" a first-class output — the exact shape of
  the gap. Proceed-tier positives are scarce (341 gate rows; only 22 in the
  holdout); fallback adjudication is the largest *honest* label supply left.

### 5B. Cascade decision verifier — Decorator over the GNN decider's band

- **Stage:** cascade (`decide()` output) → report / cluster formation.
- **Role:** verifier, and reranker inside the disagreement band.
- **Pattern:** **Decorator** over `Decisions` — wraps
  `model_tracks/cascade.decide` (`src/model_tracks/cascade.py:219-238`) at
  the caller (`_cascade_roles`, `src/model_tracks/worker.py:144-189`) or at
  report time in `graph_tracks/report.report_cascade`; the cascade itself
  stays a pure combinator (its docstring ruling: "Nothing is re-fused",
  `src/model_tracks/cascade.py:1-13`).
- **Input:** ONLY the threshold band — candidates with
  `|score − fixed_threshold| < δ` (config knob, `config/training.yaml` beside
  `fixed_threshold: 0.55`) plus the top-m per query; each (query, candidate)
  composed into an identity state (needs the query/candidate catalog
  attribute strings — the same `eligible_catalog.csv` join the holdout
  staging uses, `src/cli/laya_lane.py:3019-3032`).
- **Output:** per banded pair: `laya_score`, `agrees ∈ {confirm, contradict,
  abstain}`; a reranked `order` inside the band when laya contradicts with
  confidence ≥ fitted threshold. Join key: `PairIdentity` (§4.1).
- **Cost/latency:** band-sized, not matrix-sized: k=|catalog|−1 per query is
  the GNN's job at cosine cost; laya touches O(queries × band). At the
  measured ~0.3 s/pair, a 2k-pair band ≈ 10–20 min T4.
- **Validate:** the honest read already exists — `scripts/laya_compare.py`
  joins laya vs tracks predictions to the component-disjoint holdout with
  clustered CIs per gate stratum (`scripts/laya_compare.py:1-30`). Add the
  cascade decider as a third `--predictions` source; the experiment: on the
  `real_listing` stratum (576 labelled pairs), does laya catch GNN FP/FN in
  the band (disagreement-cell precision/recall)? Then
  `decider_report`-style precision@k with/without band reranking
  (`src/model_tracks/cascade.py:335`).
- **Why second, not first:** higher value per pair than 5A (these decisions
  directly form clusters) but needs the band knob + state composition in the
  worker path; 5A reuses the holdout machinery almost as-is.

### 5C. Negative-supply auditor — sampled Strategy over `hard_no`

- **Stage:** gate_results → labeled_pairs hard-neg class AND
  `ScoredNegativeSampler` pool (`src/graph_tracks/setup.py:374-395` — the
  graph tracks' scored negatives come from this population).
- **Role:** verifier (data-quality read; proposes NO labels by itself).
- **Pattern:** **Strategy** — a seeded sampler + laya scorer beside the
  existing sampler; same factory DI as 5A.
- **Input:** seeded sample of n≈2,000 `hard_no` pairs (the corpus builder
  already samples this population with a cap: `gate_hard_no_sampled: 552` of
  `6,947` — pre-regeneration census, `data/laya/receipt.json`; the current
  pool is 115,432) → composed states → `identity_claim`.
- **Output:** estimated contamination rate of the negative class —
  `P(laya says same | gate says hard_no)` with a component-clustered
  bootstrap CI (`cluster_bootstrap_ci`, `src/core/holdout_eval.py:79-118`),
  stratified by `gate_reason` (**589 distinct reasons measured on the
  CURRENT `data/gate_results.csv`**; the `src/pipeline.py:125` comment's 138
  is the pre-regeneration census). Pairs laya confidently calls `same` route
  to `gate_replay.replay(fired=<reason>)` for the rule-level post-mortem
  (`src/training/gate_replay.py:168`).
- **Cost:** ~2k pairs ≈ 15–25 min T4 — the cheapest item in this doc.
- **Validate:** self-validating: it IS the measurement. Sanity floor: on the
  holdout the same composition is already runnable (the `holdout-eval` kernel
  excludes `hard_no` from truth by ruling — `scripts/laya_holdout.py:10-13` —
  so this audit is deliberately *estimate + replay*, never "laya overrides
  the gate").
- **Why:** the negative class is 115k of the 134k universe and BOTH the text
  track's hard negatives and the GNN's scored negatives drink from it; a
  measured contamination rate with CIs is prerequisite knowledge before 5A/5B
  numbers can be trusted.

### 5D. Laya-side calibrator — wire what already exists (enabler, not a plug)

- **Stage:** any laya scoring run (5A/5B/5C).
- **Role:** calibrator — of laya's scores only (§4.5).
- **Pattern:** **Strategy** already landed: `EvalCalibrationSpec`
  (`temperature: true`, `abstention: false` by default,
  `src/core/laya_config.py:117-123`) selects `fit_temperature_map` (+ opt-in
  `fit_abstention_thresholds` with `target_error: 0.10`, `min_abstain_n: 10`);
  the abstention validator enforces temperature-before-abstention
  (`laya_config.py:125-135`). Trace model with the three-state accounting
  invariant: `src/core/eval_trace.py:315-335`.
- **Action:** for 5A, flip `laya.eval_calibration.abstention: true` in
  `config/training.yaml` (config-only change; validator already guards it) so
  the adjudicator's verdicts come with a per-bucket `min_confidence` fitted
  at `target_error` — i.e., the abstain rate is a *declared* error budget,
  not a vibe.
- **Validate:** ECE/Brier before/after from `evaluate_records`
  (`src/cli/laya_lane.py:2225-2237,2538-2543` — the `before`/`after` blocks
  are already the receipt shape); abstention accounting is pydantic-enforced.
- **NOT proposed:** replacing `youden_threshold`/`fixed_threshold`/the
  calibration carve with laya's temperature map — different score spaces
  (tracks' cosine vs laya's logit softmax); a replacement would be an
  unmeasured capability swap (contract §7 replace-before-remove).

### 5E. Neighbor chooser — `better_match` before transitive closure (parked)

- **Stage:** cluster formation (`build_entity_cluster_map` input edges,
  `src/training/masking.py:507`) / cascade neighbor order.
- **Role:** reranker (group-vs-single: one anchor, two candidates, ONE call).
- **Pattern:** **Strategy** over candidate merge edges: when an anchor has
  ≥2 above-threshold cascade candidates that would merge different
  components, `better_match` picks which edge to admit first.
- **Input/output:** anchor+2-candidate composed state (a NEW composer shape —
  today's `compose_state` joins exactly two sides,
  `scripts/laya_metrics_pairs.py` docstring) → choice `candidate_1` /
  `candidate_2` → edge admission order for the closure.
- **Cost:** one call per contested merge — bounded by the number of
  component-boundary collisions, not the pair universe.
- **Validate:** over/under-merge rates
  (`src/training/cluster_quality_plot.py:25-28`) with/without the chooser.
- **Why parked:** `better_match_cases: 3` in the training corpus
  (`data/laya/receipt.json`) — the checkpoint cannot answer it credibly until
  the corpus builder grows anchor-triplet states (open question 1). Building
  the plumbing before the training data would violate rung 1 of the laziness
  ladder.

### 5F. Laya as full blocker — evaluated and REJECTED (red-team, contract §7)

The framing "laya as a cheap pre-filter" was red-teamed before anything was
built. Capability cost of doing it anyway:

1. **It is not cheap here.** The gate's candidate space is 134,365 pairs
   (measured); the rule gate decides all of them deterministically in one
   scalar loop that runs ONCE per data-prep regeneration
   (`src/pipeline.py:3938-3950` — two vectorization attempts were abandoned
   to protect the pinned counts). Laya on the same set is ~11 h of T4
   (at the measured 0.3 s/pair) *per regeneration*, for a decision the rule
   table already makes for free.
2. **Wrong layer.** A blocker shrinks the candidate space *before* scoring;
   laya needs a fully composed pair state to score at all — it consumes the
   canonicalization the pipeline produces, so it can only ever sit *after*
   the gate's evidence assembly, i.e., as verifier/adjudicator, never as
   pre-filter. And it cannot pre-filter retrieval either: no embeddings, no
   ANN surface (§2) — the cascade's recall role belongs to the text track by
   measured evidence ("text ranker leads retrieval geometry… gnn_only is a
   weak retriever", `src/model_tracks/cascade.py:9-13`).
3. **Circularity + determinism loss.** `three_way_gate` is the label SSOT
   ("every training label flows through its decision table",
   `src/pipeline.py:3944-3946`); laya was fine-tuned ON its outputs (§2). A
   laya blocker would make labels a function of a model trained on the
   previous labels, with no deterministic replay (`gate_replay`) equivalent.

What survives of the blocker idea: laya IS the cheap pre-filter **for the
expensive thing** — human/P0 review of the fallback tier (5A) and GNN-band
contradictions (5B). That is the honest reading of "blocker": it blocks
*downstream expensive attention*, not upstream pair generation.

---

## 6. Questions laya can answer (17 typed questions → pipeline use)

Source: `config/laya.question.json` (SSOT; the live set is whatever the file
declares — `docs/laya-lane.md:68-70`). "Trained?" = corpus support in
`data/laya/receipt.json`.

| question | type | pipeline question it answers | plug | trained? |
|---|---|---|---|---|
| `identity_claim` | noul | **"are these two items the same physical product?"** — the core edge question | 5A/5B/5C score | yes (567 pos / 8,462 neg with growth) |
| `evidence_sufficient` | noul | "does this state carry enough measured evidence to decide at all?" — the honest abstain precondition | 5A pre-check; quarantine triage | corpus-wide `expected` |
| `gate_verdict` | choice | "what would the rule gate say?" — gate *agreement/distillation* probe, e.g. replay-free what-if on fallback pairs | 5A secondary column; `gate_replay` companion | schema-declared |
| `gate_reason` | choice | "which gate reason explains this pair?" — attributes a laya verdict to the rule table's reason vocabulary (589 distinct on the current file) | 5C reason-stratified audit | schema-declared |
| `counterfactual` | noul | "is this pair a minted twin (one attribute flipped)?" — training-data hygiene / leakage probe over the 7,875 counterfactual aug pairs | corpus QA, not pipeline | yes (by construction) |
| `same_brand_only` | noul | "do these share ONLY the brand?" — the exact FP mode of brand blocking (`src/pipeline.py:3913-3919`) | 5B contradiction explainer | schema-declared |
| `attribute_alignment` | choice | "is this ONE listing's attribute evidence internally consistent?" — node-quality census over the catalog | data-prep census row (not blocking) | yes (4,054 state cases) |
| `package_state` | noul | "does the state carry explicit package-quantity evidence?" — the pack-blindness detector (catalog carries no pack-count field, `scripts/laya_metrics_pairs.py` docstring) | 5A: pairs failing this are abstain candidates | yes (3,555 true / 499 false) |
| `field_same:volume/pack/package_type/sweetener/flavor/carbonation` | choice ×6 | "which SINGLE attribute decides this pair?" — per-dimension attribution, mirrors the gate's `dimension_census` (`src/pipeline.py:4024-4028`) | 5B/5C explanation columns | schema-declared |
| `pack_volume_equal` | noul | "are these pack volumes equal (unit-normalized)?" — the gate's volume-veto learning check | 5C on volume-reason hard_no | schema-declared |
| `pack_format_equivalent` | noul | "are these pack formats equivalent?" (can vs bottle…) | 5C on package-reason hard_no | schema-declared |
| `better_match` | choice | **"anchor vs TWO candidates — which is the better match?"** the group-vs-single question | 5E (parked) | **no — 3 cases** |

Reading: laya's answerable set covers the edge question, the abstain
precondition, gate attribution, and per-field explanation. It does NOT cover:
retrieval ("find me candidates"), embeddings, multi-item listwise ranking
(>2 candidates), or any attribute outside the six slice fields.

---

## 7. Prioritized recommendation (cheapest highest-value first)

| # | plug | cost (T4) | value | gating dependency |
|---|---|---|---|---|
| 0 | **rebuild laya data artifacts** (§3.1) | ~0 (local scripts) + 1 holdout-eval session | every number below is against the 2026-10-10 regeneration only after this | none — `scripts/laya_holdout.py` + `laya_build_dataset.py` + `holdout-eval` kernel |
| 1 | **5C** hard_no sample audit | ~20 min | measured contamination CI on the negative class BOTH tracks train from; prerequisite trust for everything else | step 0 |
| 2 | **5D** flip `eval_calibration.abstention: true` | 0 (config) | declared error budget on every laya verdict | none (validator already landed) |
| 3 | **5A** fallback adjudicator | 1.5–2.5 h / 18.6k pairs | largest honest label supply left (13.8% of the universe, quarantined today) | step 0 + 5C's contamination number + labelled-stratum holdout F1 + the fallback probe (open question 9) |
| 4 | **5B** cascade band verifier | ~15 min / 2k band | protects cluster formation where over-merges are transitive | band knob (config) + state composition in the worker path |
| 5 | **5E** better_match chooser | parked | contested-merge reranking | corpus growth decision (open question 1) |

Sequencing note (contract §7 merge ownership): the laya lane is mid-refactor
in `.worktrees/laya` (factories + HPO lane). 5A/5C/5E behavior belongs in
that DI structure (`laya_recipe`/`laya_runtime`/…), NOT grafted onto the
4,278-line `src/cli/laya_lane.py` monolith on main — and whoever builds it
branches from the integration head *after* that refactor lands, or lands the
merge itself.

---

## 8. Open questions for the owner

1. **`better_match` corpus:** invest in composing anchor+2-candidate states
   at scale (from cascade top-3s on the holdout?) to make 5E real, or drop
   the question from the schema until then? Today: 3 trained cases.
2. **Adjudicated-label status:** do 5A's laya-decided fallback pairs enter
   `labeled_pairs.csv` proper (with `label_source=laya`) or a parallel
   artifact only the graph tracks' scored pairs consume (`ScoredNegativeSampler`
   precedent)? The P0/final_validation truth must stay laya-free either way —
   confirm.
3. **Circularity budget:** laya was fine-tuned on gate `hard_no`/`proceed`
   outputs; 5A asks it to decide the tier the gate *excluded* from training.
   Accept with holdout discipline (component-disjoint, clustered CIs), or
   require a gate-blind re-fine-tune (drop hard_no/proceed-derived rows) for
   the adjudicator checkpoint?
4. **Latency budget:** is a 1.5–2.5 h Kaggle T4 session per data-prep
   regeneration acceptable for 5A, or must adjudication be an async lane run
   whose artifact the NEXT regeneration consumes (immutability makes this
   natural)?
5. **Replacement ambition:** §4.5/§5F recommend add-on and reject
   blocker/replacement with named costs. If the owner still wants a
   replacement path for the fallback *rule tier* specifically, say so — it
   needs the equivalence-pinning protocol the gate loop's own ruling demands
   (`src/pipeline.py:3946-3950`) before anything is removed
   (replace-before-remove, contract §7).
6. **Join-key consolidation:** route `scripts/laya_compare.py`'s
   `_pair_key` through `PairIdentity` when 5B lands (one spelling, contract
   SSOT), or keep the scripts-lane key as the accepted second surface?
7. **HPO champion:** the worktree HPO lane (`laya_hpo.py`, Optuna/TPE over
   `config/laya_hpo_space.yaml`) will produce champion checkpoints — should
   the 5A/5B adjudicator pin the HPO champion via the hosted `ckpt` dataset
   role, or stay on the current fine-tune receipt?
8. **Staleness trigger (§3.1):** the gate was regenerated 2026-10-10 and the
   laya corpus/holdout silently predate it. Should the data-prep regeneration
   (`src/training/data_prep.py`) emit a trace row / receipt marker that the
   laya lane reads and fails loud on ("holdout predates current gate"), so
   the rebuild in §7 step 0 can never be forgotten? (A config/receipt
   staleness *marker*, not a data-integrity gate — the artifacts stay
   immutable.)
9. **Fallback probe truth (5A validation step 2):** no labelled fallback
   pair exists anywhere in the repo today (measured: holdout labelled rows
   are exactly real_listing + p0). Who labels the ~100–200-pair probe that
   makes the adjudicator's error measurable *before* its labels touch
   training — the owner by hand, a P0-style evidence pass, or do we accept
   the downstream ablation (step 3) as the first measurement?

---

## Appendix — evidence index (all read-only verified on 2026-10-10)

- Pair key SSOT: `src/core/pair_identity.py:26-71`.
- Gate: `src/pipeline.py:2139-2156` (decision table), `:3910-3953` (brand
  blocking + scalar-loop/label-SSOT ruling), `:4034-4083` (pair_id +
  contract + atomic write); measured census on `data/gate_results.csv`:
  134,365 rows = hard_no 115,432 / fallback 18,592 / proceed 341.
- Labeled pairs: `src/training/labeled_pairs.py:1-27,80-92`; thresholds
  `config/training.yaml:1161,1166`.
- Calibration (ER): `src/training/folds.py:768-846`;
  `src/training/training.py:7047-7060,7169`; `config/training.yaml:694,713`;
  `src/core/ranking_metrics.py:81`; `src/core/hard_negatives.py:1857`.
- Graph tracks: `src/graph_tracks/model.py:1-5,193-230`;
  `src/graph_tracks/train.py:143,202`; `src/graph_tracks/setup.py:374-395,545-549`;
  `src/graph_tracks/infer.py:176`.
- Cascade: `src/model_tracks/cascade.py:1-20,64,123,152,189,219-240,335`;
  `src/model_tracks/worker.py:144-189,262-289`; `src/model_tracks/resume.py:24`.
- Clusters: `src/training/masking.py:507-546`;
  `src/training/robust_validation.py:147-157,287`;
  `src/training/cluster_quality_plot.py:25-28`.
- Laya lane: `src/cli/laya_lane.py:1381-1396` (agent), `:2494-2506`
  (calibration fit), `:2980-3045` (state composer + holdout staging),
  `:3201-3220` (pair score = softmax p[1]), `:3275-3350` (holdout kernel);
  `src/core/laya_config.py:91-135,153-277`; `src/core/eval_trace.py:315-335,400-411`.
- Laya data: `data/laya/holdout.receipt.json` (1,758 rows / 1,042 components /
  604 labelled = real_listing 576 + p0 28; gate strata label-less);
  `data/laya/receipt.json` (corpus census incl. `better_match_cases: 3`,
  `gate_hard_no_available: 6,947` — the PRE-regeneration gate);
  `scripts/laya_build_dataset.py:1-60,85,940-953,1146-1151`;
  `scripts/laya_holdout.py:1-40`; `scripts/laya_compare.py:1-40`;
  `scripts/laya_verify.py:1-30`; `scripts/laya_metrics_pairs.py:1-60`.
- Data freshness (§3.1), measured mtimes: `data/gate_results.csv` +
  `data/labeled_pairs.csv` + `data/final_validation.csv` +
  `data/track_setup/listing_pairs.csv` = 2026-10-10 10:38–10:47;
  `data/laya/receipt.json` + `data/laya/holdout.receipt.json` = 2026-10-09
  13:53. Current gate census re-measured directly from the CSV:
  134,365 rows = hard_no 115,432 / fallback 18,592 / proceed 341; all
  134,365 rows have both endpoints in `eligible_catalog.csv` (14,846 unique
  catalog gtins), so state composition is catalog-unbounded for 5A.
- Holdout label-less ruling: `src/cli/laya_lane.py:3012-3014,3025-3029`
  (only labelled rows travel); `scripts/laya_holdout.py:10-13` (gate verdicts
  are difficulty tags, never truth).
- Holdout metrics machinery: `src/core/holdout_eval.py:40-193`.
- Worktree (read-only peek, refactor in flight):
  `.worktrees/laya/src/cli/laya_lane.py:24-30` (factory DI),
  `.worktrees/laya/src/cli/laya_local_eval.py:24-50` (CPU eval twin),
  `.worktrees/laya/src/cli/laya_hpo.py:1-30` (HPO lane).
- Docs: `docs/laya-lane.md` (lane contract, decision kinds, holdout +
  comparison, caveats).
