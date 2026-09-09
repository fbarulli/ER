# EuromonitoR


# My Approach
 This dataset required more of a judgment call rather than anything else:
 - How to define ground truth?  
 - I need more ground truth

Which GTINs do I use? How do I define a GTIN?

The rest is fairly straightforward; regex and semantic transformer model:

I didnt want to use REGEX to extract as much as possible, since it's brittle and not an elegant solution, but as ive done before, I ended up back with a lookup dictionary as a starting point. For the first submission, I chose to touch as many bases as possible instead on focusing on just REGEX extraction. 
  
  
Since REGEX is deterministic, we get a clear picture of what the semantic sentence transformer can do and serves as a solid starting point. The additional experiments would be further finetuning the gates, or regex to extract enough but not too much.

Since the imbalance was around 1:17, I masked 1:1 to augment data, resulting in an almost 1:1. While we're talking about our data situtation, Ive already REGEX'ed as much as I could, the now hard negatives and positives serve for OnlineContrastiveLoss, Tripletloss, and MultipleNegativesRakingLoss. 

Of course theres a lot more room for some gains, but hopefully with all ive done, you guys have a clear picture of what i do. 


# In a Perfect World:
- TruncatedSVD
- Multi-vector representations -ColBERT-
- Better pooling strategies
- NMF
- WandB
- Topic modeling
- Fuzzy string matching
- Embedding-based similarity search
- Text classification
- More experiments loss function / model payload
- Custom semantic transformer model
- Minimize larger semantic transformer model
- Pure Embedding Clustering
- Graph‑Based Entity Resolution
- LLM as a judge 
- Hybrid Models
- Error Analysis
- Ablaition analysis
- Testing
- Operational Performance (Latency, Cost).
- Model API endpoint, Model card, Prom/Graf/ Pandera/ GE/ Pydantic.
- Stress test model, alert + retraining.
- Model calibration.
- Embedding versioning + reindex cost.
- Optimize model/data for cost/performance.
- Canary deployment



# Coding Conventions:
- SSOT (configs.*)
- Factory Pattern
- Deterministic Checks (idempotency/pure functions)
- Reproducibility (Docker)
- Data transparency / traceability (model payload)

# Random Findings
GTIN	58% missing; of the 42% populated, 43.1% are non-unique; 0.6% shared across different brands	  

Volume Generally reliable (92%+) — but a placeholder value (200) overwrites sizes from 2 to 96 fl oz flattened to "200." for at least 3,247 rows	Concentrated in a subset (coffee/cold brew).



  






[Entity Matching]

Business Case:  
Wrong metrics will invalidate all downstream economic studies. Economical inaccuracies on my watch? Think again.


## Core problem

Same physical product with GTIN (barcode/ Ground Truth) but many are missing/invalid/reused, one GTIN
can carry inconsistent attributes across retailers, and different GTINs can describe the same product. 

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
00_config.yaml        split-SSOT root: paths, files, column mapping, seed,
                      model registry (DataConfig, pydantic-validated)
TRAIN/training.yaml   training-lane config: loss, split, masking, knobs,
                      pair thresholds + eval-pair caps, bands, plots, audit
                      (TrainingConfig; absorbed the EDA config keys when
                      the EDA dir was deleted 2026-09-10)
data_pipe.py          the DATA_PIPE pipeline (extract → canonical → gate)
run_all.py            orchestrator (single caller of the whole lane)
TRAIN/                build_reference, data_prep, zero_shot_sims, labeled_pairs,
                      evaluate_models, train, dedupe, selftest, folds, hpo,
                      rerank, plots, masking (+ training.yaml)
lib/                  common (merged config + typed accessors), schemas
                      (pydantic contracts), gtin, text, blocking,
                      hard_negatives, mlflow, pipe_stopwords.json,
                      sklearn_stopwords.json
STEPS.md              every step fully defined (the contract)
Dockerfile            reproducible image (uv-locked deps, lane as /app)
artifacts/data/       raw export + number_tokens_reference (committed inputs)
```

Configs are read ONLY through `lib.common` (deep-merged view +
`data_cfg()/training_cfg()/resolve_model()`), each file
pydantic-validated at load. Every transform boundary in the pipeline
crosses a contract in `lib/schemas.py` (see STEPS.md — the first live win
was catching an impossible pack_qty=0 from "pack 0.5 l" title forms).

## Run

```bash
# 1. dedupe the raw export → dataset_deduped.csv
python TRAIN/dedupe.py
# 2. number-token reference census (committed CSV; --verify pins it)
python TRAIN/build_reference.py --verify
# 3. canonicals + gate (flavor check) → canonical_records.csv, gate_results.csv
python TRAIN/data_prep.py
# 4. zero-shot sims per model
python TRAIN/zero_shot_sims.py
# 5. labeled pairs + model evaluation
python TRAIN/labeled_pairs.py && python TRAIN/evaluate_models.py
# 6. oracle selftest (known-good entries; exit 0 = green)
python TRAIN/selftest.py
# 7. finetune — OnlineContrastiveLoss over hard pos/neg pairs
#    (GPU lane; --sample 100 for CPU smoke; also emits 07c/07d, and 07b
#     with --rerank)
python TRAIN/train.py --model artifacts/models/all-MiniLM-L6-v2
```

Docker: `docker build -t EuromonitoR -f Dockerfile .` from the repo
root (WORKDIR=/app = this folder, deps from uv.lock, artifacts/ is the
volume point).

Full step-by-step contract, gate decision table, cross-encoder A/B
protocol, and transparency guarantees: **STEPS.md**.
 