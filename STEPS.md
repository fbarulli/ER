# STEPS — every step of the lane, fully defined

One standalone folder for GPU training. `00_config.yaml` is the SSOT: every
path, file name, threshold, model name, and split lives there; the numbered
scripts and `lib/` read it through `lib.common` — nothing is hardcoded.

This is the contract. Each step states exactly what enters, what happens,
what leaves. Nothing undefined is allowed to run.

## The core problem

Retailers sell the same physical product, but each describes it differently
(language, formatting, missing units). We need to determine which listings
refer to the same real-world product. The GTIN (barcode) is supposed to
uniquely identify a product, but:

- Many GTINs are missing, invalid, or reused incorrectly.
- The same GTIN can have inconsistent attributes (volume, pack size, flavor)
  across retailers, indicating data errors or barcode reuse.
- Different GTINs can describe the same product because retailers sometimes
  list the same item under different barcodes (private label vs branded,
  local variations).

We need to group listings into true product clusters, using GTIN where it's
valid, and semantic matching where GTIN is missing or unreliable.

## Our approach

1. **Validate GTINs** (length, check digit) to separate clean from noisy
   barcodes.
2. **Extract product attributes** (volume, pack count, flavor, type) from
   titles and structured fields.
3. **Use a deterministic three-way gate** on volume/pack/flavor to block
   impossible matches and route uncertain pairs to fallback.
4. **Fine-tune an embedding model** to score the remaining pairs based on
   cleaned text (no size numbers), so it learns product identity beyond
   just brand/type.
5. **Evaluate with component-fold cross-validation** to avoid leakage and
   get honest metrics (PR-AUC, F1).

## The three data sets (config `split:`)

50 / 25 / 25 over **connected components** of the positive-pair graph (two
products linked by any positive chain share one component — a component is
never split across boundaries):

| set | share | used for | model trains on it? |
|---|---|---|---|
| **train** | 50% (q0+q1) | gradient updates | **YES — the only one** |
| **dev** | 25% (q2) | early stopping / metric selection (dev AP) | weights chosen here, never gradient-updated |
| **test** | 25% (q3) | HOLDOUT — final reported metrics only | **NEVER — not in training, not in early stopping, not in any tuning** |

Config SSOT (`00_config.yaml` → `split:`): `train_fraction: 0.50`,
`dev_fraction: 0.25`, `test_fraction: 0.25` (asserted to sum to 1.0),
`fixed_threshold: 0.55` (the operating threshold for F1/P/R).

---

## 1 — Data the model sees: original + augmented, NO NUMBERS

**Sku side** (`clean_sku_text`): title + attributes → normalized, volume/pack
tokens stripped, minimal stopwords dropped, then the **number-token strip**
driven by `artifacts/data/number_tokens_reference.csv` (1,742 tokens,
95.16% occurrence coverage; verdicts: strip / keep_brand / keep_nutrient /
keep_name). Kept digits are semantic only (b12, o2, b6, alkaline88,
good2grow, 12shots). Pure-numeric brands are spelled out (1724 → seventeen,
1642 → sixteen forty two).

**Canonical side** (`canonical_model_text` — NEW): the model payload gets a
**number-free canonical**. The gate's `canonical_records.csv` KEEPS numbers
(volume/pack drive hard_no) — **the gate CSV and the model payload are not
the same file/purpose**:
- gate canonical: `isostar orange orange_12x500 ... pet_orange_12x500_juice...`
- model payload:  `isostar orange type_plastic_flavour_orange ... pet_orange_juice...`
Measured: 3,562 canonicals carried digits → 13 after the model-strip (12×
`o2`, 1× `9.5ph` — semantic whitelist), gate CSV unchanged.

**Augmented**: for 15% of positive pairs (`masking.frac`), the ANCHOR sku
text gets random token masking at an extent drawn per-copy from U(5%, 15%),
appended as an EXTRA positive (mask token "`", same barcode → same
component, no fold contamination). +4,015 pairs at current settings.

**Pair scheme**: positive = (sku_text, own-GTIN canonical) 26,767; negative =
(sku_text, other-GTIN canonical) 28,436 from gate hard_no ∩ sim≥0.80 both
directions (now includes flavor-mismatch hard-nos).

## 2 — First step of training: the objective function

**Loss: MultipleNegativesRankingLoss (MNRL)** — the standard bi-encoder
ranking objective. For each positive pair (anchor a, positive b) in a batch,
the score matrix S = cos(a_i, b_j) over all pairs in-batch is optimized with
cross-entropy: the diagonal (true pairs) maximized against every off-diagonal
(in-batch negatives, batch_size−1 per anchor). No explicit negative labels —
MNRL ignores them, which is exactly why negatives are only mined/hard-pairs
evaluated, never fed as `label=0` training rows.

- optimizer: AdamW, discriminative LRs (8 layer groups, decay 0.9 bottom→top)
- schedule: linear warmup (5% of steps) + linear decay
- max_grad_norm 1.0, weight_decay 0.01, bf16 on GPU
- early stopping: dev AP every eval interval, patience 3, best checkpoint restored

## 3 — Success metric

**Primary: PR-AUC (average precision) on the holdout test set.** Positives are
rare relative to the pair pool, so PR-AUC is the honest ranking metric.

Reported per fold/run (all on the 25% holdout, never on train/dev):
- **PR-AUC** (primary)
- Precision / Recall / **F1 at the fixed threshold 0.55** (config
  `split.fixed_threshold`)
- ROC-AUC (secondary sanity)
- accuracy at the Youden point + the Youden threshold itself

Fold line example:
```
fold 0: loss=1.35 acc@0.71=0.60 AUC=0.62 cross=0.62 PR-AUC=0.28 F1@0.55=0.29 P@0.55=0.17 R@0.55=0.94 | best_dev_ap=0.31
```

## 4 — Cross-encoder evaluation (stage 2) — the A/B protocol

The cross-encoder (rerank) must PROVE itself against the bi-encoder on the
SAME held-out component folds:

1. Same holdout: no barcode in validation was seen in training (component
   split guarantees this for both stages).
2. For every validation pair, compute:
   - **bi-encoder similarity** (stage 1, cosine)
   - **hybrid score** = cross-encoder score for pairs in the confusion band
     (0.35–0.90 cosine), else the bi-encoder score
3. Compare on the holdout: **PR-AUC** (primary), **Precision/Recall/F1 at a
   threshold chosen on validation**, ROC-AUC (secondary).
4. **Decision rule: the hybrid must clearly improve PR-AUC / F1 over the
   bi-encoder alone.** If not, the cross-encoder is not worth its latency —
   drop it.

## 5 — (reserved for numbering alignment)

## 6 — Flavor semantics (transparency contract)

- **Flavor STAYS in the embedding input**: canonical/sku texts keep flavor
  words (orange, apple, ginger) — discriminative signal the model needs.
- **Gate flavor check = hard block ONLY on exact extracted mismatches**:
  `if flavor1 and flavor2 and flavor1 != flavor2 → hard_no`. One side empty →
  NO block (unknown ≠ different). Currently 12,417 hard-no pairs from flavor
  mismatch; the remaining low-sim proceed tail (3,093 pairs) is a
  flavor-EXTRACTION coverage gap (Finnish/Dutch compounds), not gate logic.

## The three-way gate — decision table

| Decision | Meaning | Criteria |
|---|---|---|
| **hard_no** | Confidently different products | Any of these: |
| | | • Volume sets have no overlap within ±5% tolerance (e.g., 250ml vs 500ml) |
| | | • Pack sets have no common pack count (e.g., single vs 6‑pack) |
| | | • Both sides have non‑empty flavors and they are different (e.g., apple vs orange) |
| **fallback** | Not sure; needs semantic scoring | Any of these: |
| | | • Volume or pack confidence is below 0.85 |
| | | • Volume and pack overlap, but consistency is very low (<0.3) |
| **proceed** | Likely duplicates; send to embeddings | All of these: |
| | | • Volume confidence ≥0.85 on both sides |
| | | • Pack confidence ≥0.85 on both sides |
| | | • Volume sets overlap within tolerance |
| | | • Pack sets intersect |
| | | • Flavors are compatible (same, or one/both missing) |
| | | • Consistency ≥0.3 on both sides |

## Pipeline steps (numbered scripts)

- **TRAIN/01_data_prep.py** — within-brand pipeline: extract volume/pack/flavor
  per row → canonical per GTIN → three-way gate (with flavor check) every
  candidate pair → `canonical_records.csv` (14,997) + `gate_results.csv`
  (153,901 pairs: hard_no 104,522 / proceed 32,128 / fallback 17,251).
- **TRAIN/02_zero_shot_similarities.py** — encode canonical texts with each
  model, score gate pairs → `embedding_similarities.csv` (per-model sim
  columns, incremental per-model writes, resumable).
- **TRAIN/03_labeled_pairs.py** — gate decisions + sim≥0.8 → auditable
  `labeled_pairs.csv` (7,912 pos / 14,675 hard-neg).
- **TRAIN/04_evaluate_models.py** — per model: ROC-AUC + Youden + P/R/F1,
  per-model plots with absolute n → `model_evaluation_summary.csv`.
- **TRAIN/05_train.py** — the training entry (masking, holdout/cv folds,
  MNRL, early stopping, plots, mlflow, rerank).
- **TRAIN/report_plots.py** — the 07_report figure family, per model.
- **TRAIN/composition_plot.py** — training-data composition with absolute n.
- **06_run_all.py** — orchestrator: embeddings → 2k-sample sweep → full-data
  run → ablation suite.

## Transparency guarantees (every step)

- Every count printed at run time: pairs, canonicals, dropped endpoints,
  masked additions, per-fold n_pos/n_neg.
- Every file name and path from `00_config.yaml` only.
- Every plot carries absolute n (titles + per-bar annotations).
- `gate_results.csv` = GATE input (numbers kept). Model payload = NUMBER-FREE
  variant (derived at pair-construction time, never persisted as a second
  CSV — one canonical SSOT, no drift).
- Loss/acc/AUC/PR-AUC/F1 in console + `train_fold_metrics.csv` + mlflow
  (local sqlite backend, `artifacts/mlruns/`).

## MLflow (local backend + artifact store)

Every `05_train` invocation = one parent run + nested run per fold. Default
backend LOCAL: `sqlite:///artifacts/mlruns/mlflow.db`, artifacts under
`artifacts/mlruns/artifacts/`. Browse:
`mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db`.
`MLFLOW_TRACKING_URI` overrides; `=off` disables.

## Training loss plot

`training_loss_<split>_payload-<variant>.png` — per-fold loss curve with the
best-dev-AP step marked; history persisted in `train_fold_metrics.csv`.

## Report plots — per model

`TRAIN/report_plots.py` runs the 07_report family for EVERY config model:
`07_report_*_<model_key>.png`. `--models <keys>` selects a subset. Deberta
panels fill in on the GPU pass (CPU: ~2000× slower on this torch build).

## Docker (reproducibility)

`Dockerfile` builds `broadway-train-gpu` from the repo root:
```
docker build -t broadway-train-gpu \
  -f project/experiments/euromonitor/TRAIN_GPU/Dockerfile .
```
deps pinned via `uv export --extra nlp` from `uv.lock`; run the lane with
`--workdir /app/project/experiments/euromonitor/TRAIN_GPU`; mount
`artifacts/` to persist. Verified in-image: lint clean, config SSOT + data
pipe import, exact version pins (torch 2.13.0, transformers 5.16.1,
sentence-transformers 6.0.1, mlflow 3.15.1, optuna 4.4.0, sentencepiece
0.2.2, ruff 0.16.3). 10.6GB.

## Deberta CPU warning

deberta-v3 relative attention on this torch CPU build runs ~2000× slower
than MiniLM (3.9 s/text vs 2 ms/text measured). Zero-shot deberta scoring
and deberta training run on the GPU lane.
