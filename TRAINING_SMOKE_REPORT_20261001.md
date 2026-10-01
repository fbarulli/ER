# Training smoke verification — 2026-10-01

The current prepared generation is `data/track_setup_20261001/`.

| Stage | Verified evidence |
|---|---|
| Source catalog | 63,079 deduped rows; 152 reviewed rows excluded from text training |
| Text bundle | 62,927 source rows; 117,956 payload rows; 51,712 positive views; 31,619 negative views |
| Full bundle diet | Strict configuration drift check passed; negative augmentation 0.3136; effective positive/negative ratio 1.1664 |
| Shared graph catalog | 27,852 eligible listings: 13,927 train / 6,902 dev / 7,023 test |
| Graph supervision | train 6,055 positive / 1,927 negative; dev 3,061 / 592; test 3,052 / 466 |
| CPU smoke | 128 source listings; one epoch; all three workers in one CPU Colab VM; test evaluation disabled |
| Smoke diet | Negative augmentation 0.3359; effective positive/negative ratio 1.4351; strict preflight passed |
| Dashboard | `/training` displays plots and metrics directly from downloaded suite archives or local result folders |

The first 100-listing projection failed the diet gate. Its failure was kept visible; thresholds were not relaxed. The smoke sampler now preserves selected counterfactual/swap donor lineage, resolves dependent positive/negative copies to closure, supports an absent full-catalog text cache, and accepts relative setup paths.

Local checks: 33 focused training/control/package/diet tests passed. Six dashboard HTTP tests passed before the user's sandbox-only instruction; browser verification displayed seven metric tables and four existing report plots. Subsequent commands use the sandbox.

## Current runtime

First Colab run: `1001T050218144340Z`. All three training epochs completed; graph postprocessing completed. Text completion failed because the no-test branch classified intentionally unavailable sample calibration as a failed fold. The source fix accepts completed samples while retaining explicit calibration-unavailable metrics; full runs still raise on missing calibration. Failed runtime teardown was verified.

Retry: `1001T050826082308Z`, launched inside the sandbox after the user requested sandbox-only commands. **PASS**: text, GNN-only, and hybrid workers completed training, selected checkpoints, inference/index export, and report generation. Suite worker runtime: 187.10 seconds; all-track stage polling: 227.78 seconds. VM teardown verified.

Downloaded archive: `results/model_tracks/1001T050826082308Z.zip` (338,060,683 bytes). SHA-256: `96ac07aaef36fb1fc0d1d3205ce853c3d00ab762a5563996e7e07b82b13b00e9`. Portable inventory verification passed. Three completion markers, four checkpoint weight files, four valid PNG plots, and all three model evaluation summaries verified. Vector exports have 128 unique listing IDs with finite nonzero vectors (text 384 dimensions; graph and hybrid 128 dimensions), and each track retained catalog and dev retrieval HNSW indexes. Evaluation summaries contain only dev rows; text fold status is `ok`, with `test_eval=skipped_selection_mode` and explicitly unavailable native sample calibration. The text postprocessing report fits its operating threshold on the shared dev listing pairs.

Verification receipt: `results/model_tracks/1001T050826082308Z.verification.json`. Dashboard standalone preview: `results/model_tracks/1001T050826082308Z_dashboard.html`.

Dashboard startup inside the sandbox could not bind to port 8001. The dashboard's saved-archive render and all four plot responses were verified directly inside the sandbox; no outside-sandbox server was started after the user's instruction.

Full-catalog frozen MiniLM cache finished: 27,852 vectors, 384 dimensions. Full suite preflight passed and is saved at `data/track_setup_20261001/preflight.json`. `config/model_tracks.yaml` now points to this verified generation.

Legacy listing provenance and interrupted-suite recovery are now implemented; see `TRAINING_DATAFLOW_AUDIT_20261001.md`. GPU budgeting remains deferred. Smoke scores are lifecycle evidence, not quality benchmarks.

## Follow-up fixes before push

Legacy preflight passed with 27,852 eligible listings: 13,927 component training listings and 13,925 dev/test inference listings, with zero entity overlap. Strict bundle drift is enabled. Obsolete sampled legacy launches are rejected before provisioning.

Updated local CPU suite `cpu_resume_validation_20261001` completed all three tracks and archived reports. A subsequent resume correctly rejected source changes made during its initial run; this confirms fail-closed provenance checking. Focused recovery tests cover archive restoration, completed-track verification, changed-source rejection, missing state, and restarting only unfinished workers. Live remote interruption/recovery remains untested; the earlier three-track Colab CPU smoke passed.
