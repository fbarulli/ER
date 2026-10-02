# Data train flow — 2026-10-02

Full trace from pair creation to model payload. Offline prep only; no training launched.

## Stage 1: Offline prep (`src/training/data_prep.py`)

```
raw export (gtin/sku_name_eng/attribute)
  → run_within_brand_pipeline(df)
  → canonical_records.csv + gate_results.csv + trace
```

## Stage 2: `build_training_data()` (`src/pipeline.py:3124`)

Inputs: deduped dataset (`load_dataset_deduped`) + `canonical_records.csv` + `gate_results.csv`

```
df → exclude_reviewed_rows → reset_index
  ↓
payload variant: full = title+attrs | title_only = title blanked out
  ↓
build_sku_texts(model_frame) → sku_texts, sku_structured
  # core.model_input: [Brand] [Title] [Attributes], cleaned profile
  ↓
build_canonical_text(record, info) per GTIN (sorted gtin order) → canon_texts
  ↓
payload = sku_texts + canon_texts
row_bc  = [sku_barcodes... | GTINs...]
  ↓
pos = (sku_row_idx, canon_start + canon_idx) for every row with resolvable canonical
      empty-text pairs dropped (counted, not silent)
  ↓
neg = gate hard-no pairs, both directions, resolved to payload indices
      same-canonical excluded (true match), similarity >= hardneg_sim_threshold
  ↓
targeted_attribute_neg = mine_targeted_attribute_negatives (same brand/name, explicit conflict)
cross_brand_neg        = mine_cross_brand_negatives (brands differ, rest agrees)
  ↓
return {payload, structured_features, row_bc, pos, neg,
        targeted_attribute_neg, cross_brand_neg, gtin_to_row, stats}
```

Payload index layout: `[sku_0 .. sku_{N-1} | canon_0 .. canon_{M-1}]`
- anchor side of pos is always the SKU row
- neg both directions: (gtin1→row, gtin2→canon) + (gtin2→row, gtin1→canon)

## Stage 3: Masking augmentation (`src/training/train.py:720`)

Mutates pos/neg/payload/row_bc in place, in this order:

1. `augment_positives` — random token masks on anchor copies (label=1)
2. `augment_value_swaps` — agreed-field donor transplant on positives (label=1)
3. `augment_declaration_dropout` — drop 1..3 structured groups (label=1)
4. `augment_hard_negatives` — mask neg copies (label=0)
5. `augment_value_swaps` on negatives (label=0)
6. `mint_swap_counterpart_positives` — (copy, counterpart) new pos pairs
7. `augment_counterfactual_twins` — agreed-field flip (label=0)

`extend_augmented_features` re-syncs structured_features after each step.
Runtime guard: `len(structured_features) != len(payload)` → crash.

## Stage 4: Component-aware split (`src/training/train.py:1045`)

```
derive_holdout(pos, row_bc, split_cfg) or component_folds(pos, row_bc, k)
  → train_bc / dev_bc / test_bc
pairs_in_set(pos, row_bc, train_bc) → train_pos (component-safe)
same for neg → train_neg
```

## Stage 5: HuggingFace Dataset + batch sampler (`src/training/training.py`)

### Dataset construction (loss-dependent)

**contrastive** (`training.py:3777`):
```python
pair_populations = _training_pair_populations(train_all, tr_negs, ...)
train_ds = Dataset.from_dict({
    "sentence1": s1, "sentence2": s2, "label": lab,
    "pair_id": [...], "pair_population": pair_populations,
    "structured_features": [...],
})
```

**mnrl** (`training.py:3914`):
```python
triples = _build_mnrl_training_triples(train_all, tr_negs, ...)
triple_populations = _build_mnrl_triple_populations(...)
train_ds = Dataset.from_dict({
    "anchor": [...], "positive": [...], "negative": [...],
    "pair_id": [...], "population": triple_populations,
})
```

**triplet** (`training.py:3950`):
```python
examples = build_triplets(train_all, hard_train, payload, ...)
train_ds = Dataset.from_dict({
    "anchor": [...], "positive": [...], "negative": [...],
    "pair_population": ["triplet"] * len(examples),
})
```

### Batch sampler

`ControlledBatchSampler` (`src/training/sampler.py`) — deterministic per-population batch composition.

Config (`config/training.yaml` → `training.batch_sampler`):
```yaml
batch_sampler:
  enabled: false    # true = controlled sampler; false = default HF behavior
  composition:
    gate_positive: 4
    masked_positive: 4
    hard_negative: 8
  seed: 42
```

**Algorithm** (each epoch):
1. Shuffle each population's indices independently with `seed + epoch`
2. Cycle through the composition template, drawing one index per group
3. When any group is exhausted, epoch ends

**Why this exists** (rationale):
- MNRL is an in-batch loss: every other row is a negative for each anchor — batch composition *is* the negative pool
- Twin warmup needs twin rows in every batch to down-weight them; without twins present the warmup has no effect
- Per-population telemetry can only attribute loss for populations that appear in the batch
- OnlineContrastiveLoss mines hard pos/neg within the batch; both must be present
- NoDuplicatesBatchSampler (used by MNRL when uncontrolled) prevents duplicate texts colliding as in-batch negatives — the controlled sampler inherits that discipline by construction

### Trainer wiring (`training.py:4163`)
```python
args_hf = STArgs(
    ...
    batch_sampler=controlled_sampler if controlled_sampler is not None
                  else (BatchSamplers.NO_DUPLICATES if loss == "mnrl"
                        else BatchSamplers.BATCH_SAMPLER),
)
```

## Stage 6: Eval encode (`src/training/training.py:4908`)

```python
model.encode(payload[row])  # fused with structured_features via fuse_numpy()
cos(pos_emb, neg_emb) → AUC / PR-AUC / Youden
```
