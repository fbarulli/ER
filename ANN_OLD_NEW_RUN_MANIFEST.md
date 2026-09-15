# ANN old/new comparison manifest

## Shared data contract

Both versions must use the same immutable inputs:

- Holdout CSV: `artifacts/data/dataset_deduped_sample_3000.csv`
- Training complement: `artifacts/data/dataset_deduped_train_minus_3000.csv`
- Holdout rows: `3,000`
- Training rows: `58,529`
- Source rows: `61,529`
- Holdout SHA-256: `a8ae6d6f6f14c25de1df8269ee24f53fca82c810d81ddcfc1ec7967631678834`
- Training-complement SHA-256: `4091dce86173db72fad3f62dfdb37a08a8e300bf247025c0b306bb81c9cc562d`
- Seed: `42`
- Base model: `minilm_l6` / `all-MiniLM-L6-v2`
- Epochs: `10`
- Training loss: `mnrl`
- Inference threshold: `0.787`
- Inference batch size: `128`
- Hardware: `T4 GPU`

## Old baseline

- Git commit: `b38d2d6` (`Expose ANN candidate recall plot`)
- Gate: volume/pack/brand; no first-class `package_type`
- Structured model tokens: volume and pack only
- Run: `0915T063500554948Z`
- Remote root: `/content/EuromonitoR`
- Worker 1 (matcher) checkpoint:
  `/content/EuromonitoR/results/concurrent_train_0915T063500554948Z/worker_1/_checkpoints/all-MiniLM-L6-v2/r0915T063500554948Z-worker_1-matcher_f0/checkpoint-102`
- Worker 2 (ANN) checkpoint:
  `/content/EuromonitoR/results/concurrent_train_0915T063500554948Z/worker_2/_checkpoints/all-MiniLM-L6-v2/r0915T063500554948Z-worker_2-ann_embedding_f0/checkpoint-44`
- Worker 1 predictions:
  `training_results/0915T063500554948Z/worker_1/validation_inference/sku_predictions.csv`
- Worker 2 predictions:
  `training_results/0915T063500554948Z/worker_2/validation_inference/sku_predictions.csv`
- Worker 1 report:
  `training_results/0915T063500554948Z/worker_1/report/`
- Worker 2 report:
  `training_results/0915T063500554948Z/worker_2/report/`

Recovered local checkpoint directories:

- `training_results/0915T063500554948Z/worker_1/_checkpoints/all-MiniLM-L6-v2/r0915T063500554948Z-worker_1-matcher_f0/checkpoint-102/`
- `training_results/0915T063500554948Z/worker_2/_checkpoints/all-MiniLM-L6-v2/r0915T063500554948Z-worker_2-ann_embedding_f0/checkpoint-44/`

Both directories were pulled from their `.dvc` pointers and load successfully
as 384-dimensional SentenceTransformer models with local files only.

## New first-class-attribute version

- Git commit: `e3df5f6` (`Carry package type through ANN inference gates`)
- Gate: volume/pack/package-type/brand
- Structured model tokens: volume, pack, and `package_type_*`
- Remote run root: `results/concurrent_train_<new_run_id>/`
- Worker 1 expected root: `results/concurrent_train_<new_run_id>/worker_1/`
- Worker 2 expected root: `results/concurrent_train_<new_run_id>/worker_2/`
- Expected inference output per worker:
  `validation_inference/sku_predictions.csv`
- Expected ANN report per worker:
  `report/`

The new run ID and checkpoint numbers are assigned at launch and must be
recorded here immediately after launch. Do not overwrite the old baseline.

## Manual download targets

Download each completed worker bundle into:

- `training_results/<new_run_id>/worker_1/`
- `training_results/<new_run_id>/worker_2/`

Required files include `training.log`, `training.status`, `live_status.json`,
`validation_inference/`, `report/`, `dvc_manifest.json`, and the checkpoint
pointer files. Preserve the old `0915T063500554948Z` tree unchanged.
