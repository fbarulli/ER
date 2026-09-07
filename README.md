# EuromonitoR

Product-identity matching lane: group retailer listings into true product
clusters — GTIN where it's valid, semantic matching where it's missing or
unreliable. Clean break from the broadway monorepo: everything here is
self-contained and runs from `00_config.yaml` (the SSOT — every path, file
name, threshold, model name and split comes from it; nothing is hardcoded).

## Core problem

Retailers sell the same physical product, but each describes it differently
(language, formatting, missing units). The GTIN (barcode) is supposed to
uniquely identify a product, but many are missing/invalid/reused, one GTIN
can carry inconsistent attributes across retailers, and different GTINs can
describe the same product. We cluster listings into real-world products.

## Approach

1. **Validate GTINs** (length, check digit) — clean vs noisy barcodes.
2. **Extract attributes** (volume, pack, flavor, type) from titles + fields.
3. **Deterministic three-way gate** on volume/pack/flavor: block impossible
   matches (hard_no), route uncertain ones to fallback, send likely
   duplicates to embeddings (proceed).
4. **Fine-tune an embedding model** on cleaned text (NO numbers — sizes are
   the gate's job, never the model's) to learn product identity.
5. **Component-fold evaluation** — no barcode straddles a split boundary;
   metrics are honest (PR-AUC primary, F1 at a fixed threshold).

## The three data sets (50 / 25 / 25, component-aware)

| set | share | used for | model trains on it? |
|---|---|---|---|
| train | 50% | gradient updates | **YES — the only one** |
| dev | 25% | early stopping only | never gradient-updated |
| test | 25% | **HOLDOUT — final metrics only** | **NEVER — not in training, not in early stopping, not in tuning** |

## Layout

```
00_config.yaml        SSOT: paths, files, thresholds, models, split
data_pipe.py          the DATA_PIPE pipeline (extract → canonical → gate)
06_run_all.py         orchestrator (single caller of the whole lane)
TRAIN/                01_data_prep … 05_train, folds, hpo, rerank, plots
EDA/                  corpus analyses (funnel, sparsity, cross-country)
lib/                  common (config), text, blocking, hard_negatives, mlflow
STEPS.md              every step fully defined (the contract)
Dockerfile            reproducible image (uv-locked deps, lane as /app)
artifacts/data/       raw export + number_tokens_reference (committed inputs)
```

## Run

```bash
# 1. dedupe the raw export → dataset_deduped.csv
python TRAIN/06_dedupe.py
# 2. canonicals + gate (flavor check) → canonical_records.csv, gate_results.csv
python TRAIN/01_data_prep.py
# 3. zero-shot sims per model
python TRAIN/02_zero_shot_similarities.py
# 4. labeled pairs + model evaluation
python TRAIN/03_labeled_pairs.py && python TRAIN/04_evaluate_models.py
# 5. finetune (GPU lane; --sample 100 for CPU smoke)
python TRAIN/05_train.py --model artifacts/models/all-MiniLM-L6-v2
```

Docker: `docker build -t broadway-train-gpu -f Dockerfile .` from the repo
root (WORKDIR=/app = this folder, deps from uv.lock, artifacts/ is the
volume point).

Full step-by-step contract, gate decision table, cross-encoder A/B
protocol, and transparency guarantees: **STEPS.md**.
