# Flowchart findings — EuromonitoR training lane

Untracked notes kept while authoring the repo's first pipeline diagram.
No diagram existed anywhere in the repo before this (checked all branches on
remote `ER`, the stash, and git history — zero `.drawio`, `.mermaid`, `.svg`,
`.mmd`, `.puml`, `.dot`, or diagram-flavored `.md` files).

Final output (DELIVERED, spec frozen):
`archify/generated/euromonitor-lane.workflow.json` →
`archify/generated/euromonitor-lane.workflow.html`
(sha256 `8c08cb2b6ee2227efbd50141a17b45a30b1dc7a863f83635a449a76be0a2ca28`,
831,773 bytes). Validation: **ok, 0 errors, 0 warnings, 9/9 artifact checks**
(single_svg, finite_svg, orthogonal_arrows, label_route_clearance,
relationship_crossings, relationship_corridors, container_border_runs,
route_rhythm, legend_clearance). visual-check: exit 2 — no Chrome/Chromium on
host, honestly skipped, sidecars written
(`euromonitor-lane.workflow.visual-check.json`).

---

## Verified facts (scout pass over the `training` branch)

### Entry points
- `run_all.py` — orchestrates steps 1–4: **1 embeddings, 2 sweep-2k, 3 sweep-full, 4 ablation**; flags `--only`, `--from`, `--stop-on-fail`; resumable.
- `er-colab` (`src/cli/colab.py`) — Colab T4 lane; clones **ER branch `training`**; downloads frozen CSVs; **manifest-gated** downloads (sha256 re-hash before accepting artifacts/checkpoints); `--what train|hpo|sims`.
- `er-train` (`python -m training.train`).

### Config SSOT
- `config/paths.yaml` + `config/training.yaml`, pydantic models **DataConfig / TrainingConfig** (`src/core/common.py`).
- Every knob (gate thresholds to HPO spaces) lives in config; no inline literals.

### Pipeline stages (all in `src/training/`)
- `dedupe.py` — T1/T2/T3 passes → `dataset_deduped.csv`.
- `data_prep.py` — volume / pack / flavor → canonical forms.
- `three_way_gate` — decisions **hard_no · fallback · proceed**.
- `zero_shot_sims.py` — cosine sims per model.
- `labeled_pairs.py` — pos ∩ proceed, neg ∩ hard_no; sim ≥ 0.80.
- `evaluate_models.py` — Youden on DEV → TEST.
- `folds.py` — **component_folds 50/25/25** (q0+q1 / q2 / q3) — test never trains, early-stops, or tunes.
- `masking.py` — masked positives, U(0.20–0.30).
- `hpo.py` — grid + TPE, ranks on **best_dev_ap**; per-config test eval skipped (`test_eval=skipped_selection_mode`).
- `train.py` — two-tower contrastive, dev-AP early stop.
- `rerank.py` (07e) — **cross-encoder/ms-marco-MiniLM-L-6-v2**, ship band [0.50, 0.75].
- `report_plots.py` (07f) — PR curves, ablation, scaling.
- Base encoder: **paraphrase-multilingual-MiniLM-L12-v2**.

### Data
- Raw export: **71,623 rows**, sha256-pinned `dataset.csv`.
- `train_manifest.csv` — appended by `run_all.py`, never destroyed.

### Manifests & tracking
- `results/manifests/*.json` — StageManifest per stage (dedupe, data_prep, labeled_pairs, evaluate_models, zero_shot_sims), **atomic, written last, sha256** row-accounting.
- Metrics: **pr_auc, best_dev_ap, P/R/H@K (1,5,10)**.
- Trackers: **W&B (`e-r`) + MLflow (`artifacts/mlruns`)**.
- Ship rule: **ΔPR-AUC or ΔF1 > 0.005**.

## Ambiguities the scout flagged (shown as-is in the diagram, not resolved)
1. **Two rerank protocols coexist**: `run_all.py` STEP 4 (fixed L12, 07e) vs `er-colab --what hpo` (per-winner `--rerank`). Diagram keeps both as distinct edges.
2. **`era` suffix** appears in filenames; semantics undocumented.
3. `build_reference` docstring row-count drift: 1,742 vs 1,745.

---

## Graphify compatibility (run in this worktree, branch `graphify-compat`)

Toolchains: **fully compatible — zero interference, one silent-exclusion gotcha.**

1. **Isolated stacks.** archify is Node-only (no Python deps); graphify is
   Python 3.14 via `uv tool install graphifyy` (PyPI name has double-y —
   `graphify` on PyPI is an unrelated package; the GitHub org is Graphify-Labs).
   No shared files, no shared processes; the ER venv is untouched.
2. **graphify parses archify's code.** Pointed at `archify/` directly it
   indexed the whole compiler (4,730 nodes / 6,645 edges / 391 communities,
   `.mjs` handled via the `jsts` tree-sitter grammar — the README's
   "JS/TS" understates this).
3. **Gotcha — root-level ER graph silently excludes archify.** ER's
   `.gitignore` line 40 ignores `archify/`, and graphify honors ignore files:
   the ER root graph (854 nodes / 1,671 edges / 59 communities) never sees
   archify. Workaround: index `archify/` as its own target (what we did), or
   add `!archify/` negation in `.graphifyignore`.
4. **graphify writes `graphify-out/` into the target tree.** It self-excludes
   that dir from future scans (honorignore `_SKIP_DIRS`), but in archify it
   had to be `rm -rf`'d to keep the repo clean; in ER it is untracked noise.

## Layout rules learned (archify workflow compiler, schema v2)

Grid is (lane, column); **one node per lane×column cell**. Additional rules
discovered while reaching showcase quality:

1. **Same-lane labeled horizontal edges inflate column gaps.** The solver
   reserves `halfW(A) + max(28, labelWidth+8) + halfW(B)` between adjacent
   columns. Labels ≳132px (cost >112px detour budget) instead ride a
   **label channel** above the lane and stop inflating columns — but channels
   get forced top-side exits and can collide with each other, jogging edges
   to the canvas edge. Sweet spot: either drop the label (when adjacency
   implies it — main-path steps), or keep it short.
2. **Fan-in congestion beats label width.** Six edges entering `train` forced
   202px detours to x≈978–1,290. Fixes: re-anchor edges to their true
   producer (`run_all.py` appends `train_manifest.csv`, not train), move
   nodes across lanes (rerank → ship_lane) to break long vertical bridges,
   and only then trim labels.
3. **Node text fit is width-driven.** `fittedNodeFontSize` =
   `max(6, min(8, (width−8)/(units×0.6)))` for sublabels; desktop-readability
   needs `sourceFontPx × (930/viewBoxWidth) ≥ 6`. Sublabels that hit the 6px
   floor at large viewBoxs fail; fix by shortening the sublabel (telegraphic
   is fine when cards carry the verbatim facts) or widening the node.
4. **requiredPaths are reachability checks**, not edge lists — a transitive
   chain satisfies them, which frees edges to model the true producer.
5. **Cards are the verbatim-fact store.** Node sublabels can be telegraphic
   because the side cards carry exact identifiers, thresholds, and flags.

Final grid (18 nodes / 21 edges / 6 lanes / cols 0–5):
- data_lane: raw_export c0, dedupe c1, data_prep c2, three_way_gate c3
- pairs_lane: zero_shot_sims c2, labeled_pairs c3, evaluate_models c4
- finetune_lane: masking c2, component_split c3, train c4
- ship_lane: rerank c4, report_plots c5
- infra_lane: run_all c3, colab_lane c4, hpo c5
- state_lane: config_ssot c1, stage_manifests c4, tracking c5

Structural choices of note:
- `run_all → colab_lane → train` (Colab dispatch), with `python -m
  training.train` riding the colab→train edge — the verbatim command.
- `run_all → stage_manifests` ("appends train_manifest.csv") — the true
  producer; freed train's ports and cleared the last corridor.
- rerank in ship_lane c4 beside report_plots: the ship band [0.50, 0.75]
  and ΔPR-AUC>0.005 ship rule live on the rerank→report edge.
- Phases: p_source (0–1), p_pairs (2–3), p_finetune (4, emphasis),
  p_ship (5, dashed "Sweep, rerank & ship").
- semanticChecks pin the contract: roots
  raw_export/config_ssot/masking/component_split/run_all; terminals
  report_plots/tracking; requiredEdges config_ssot→dedupe,
  stage_manifests→colab_lane, hpo→train, train→tracking; requiredPaths
  (reachability) raw_export→report_plots, run_all→train,
  run_all→report_plots, labeled_pairs→train, colab_lane→train.
