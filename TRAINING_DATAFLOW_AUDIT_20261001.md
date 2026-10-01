# Training dataflow audit — 2026-10-01

Read-only code/data audit during concurrent setup rebuild and Colab smoke work. No preparation or training was launched by this audit. Runtime results belong to the coordinating agent's smoke report.

## Current dataflow

1. `config/paths.yaml` binds raw export, deduped catalog, canonical records, gate results, labeled pairs, and identity reviews. `graph_tracks.setup.setup` loads the deduped catalog, builds text supervision, and derives train/dev/test through `training.folds.derive_holdout`.
2. `derive_holdout` merges training positives with labeled positive edges before assigning connected components. This prevents labeled equivalent identities from being split between training and validation. `data/final_validation.csv` is a labeled-pair population; it is not a listing catalog and cannot replace the retired listing sample as an inference CSV without an adapter.
3. Graph setup exports eligible listings, listing splits, same-split listing pairs, model attributes, hashes, and pair accounting. Invalid/unassigned identities are excluded; cross-split negatives are counted and excluded. Barcode and verified identity edges are excluded from graph model features. Initial graph features cover the declared relations/numeric subset, not every report identity dimension.
4. Prepared text bundles contain payloads, features, positive/negative supervision, augmentation lineage, optional baseline embeddings, and frozen labeled/canonical/gate CSV bytes. Loaders validate archive digest, required keys, row/pair counts, input composition, augmentation features, and counterfactual audit validity. Training materializes frozen CSVs inside its own output directory.
5. Suite preflight checks frozen text baseline, graph manifests/cache hashes, text CSV freshness, canonical payload ownership, reviewed exclusions, and each eligible listing's shared split. Smoke bundles preserve parent holdout populations rather than deriving a new split from a sample.
6. Packaging ships a source overlay plus frozen setup and configuration. Three isolated workers train concurrently, select checkpoints, postprocess, and emit completion markers. Reports use a dev-fitted threshold; test reporting is controlled by the suite setting. The hybrid worker uses the frozen baseline independently of simultaneous text fine-tuning.
7. Completion requires all three markers, a checksummed portable result archive, and archive verification after download. Publication is conditional on suite settings and has receipts. Completed result collection can retry without restarting training.

## Remaining concrete gaps

### Legacy sampler contract resolved during this audit

Retired 3k/5k CSV references and fixed population counts were removed. Full training now uses `data/dataset_deduped.csv`; a validated shared component setup supplies eligible source listings, component training listings, and dev/test inference listings for provenance. Materialization checks setup freshness, eligible catalog hash, exact listing coverage, reviewed exclusions, and normalized entity train/holdout overlap.

Actual prepared bundles are checked against the shared listing split before use, including configured full bundles before provisioning and locally rebuilt/cache bundles before return. Plain legacy smoke and sampled legacy training fail before provisioning with an actionable explicit sampled-suite command, because their old sampling does not preserve parent components. Use `--what tracks --tracks-config results/model_tracks/smoke_20261001_128/suite.yaml --gpu CPU` for the tested sampled flow.

### Interrupted suite resume implemented

Suite launch accepts `--resume-run` and reuses the original verified input package. Reachable failed runtimes save portable recovery archives before teardown. Restore validates input/source provenance, completed-track artifact hashes, and native checkpoints; only unfinished workers restart with a fresh barrier. Graph workers can finish postprocessing after their final epoch. Text workers restore optimizer, scheduler, RNG, and portable selected-checkpoint paths. Existing older interrupted outputs without resume provenance are rejected. Focused resume/control tests passed; a full remote interruption/recovery experiment has not been run.

### Source and shape gaps resolved during this audit

The coordinating agent added a direct comparison of current `F['dataset_deduped']` against `setup_manifest.source_catalog_sha256` in suite preflight; this now rejects catalog-only drift. Package inputs now include the matching deduped catalog, frozen supervision CSVs, pipeline source, and configuration snapshot, so local and remote checks see the same generation even when workspace inputs are uncommitted.

Preflight also now validates the text `DataTuple`, integer negative endpoint pair shapes/ranges, and source-label alignment. These changes were confirmed by reading the updated modules; their runtime validation belongs to the coordinating agent.

### GPU overlap has no mandatory measured memory budget

`config/model_tracks.yaml` defaults `memory_reservations_gb: {}`. `model_tracks.run` enforces summed worker peaks plus headroom only when that mapping is nonempty. CPU smoke cannot establish CUDA/MPS readiness. Populate measured peaks for all three workers and verify available target GPU memory before claiming full parallel GPU readiness.

### Suite strict drift and diet gates resolved during this audit

Suite preflight now invokes `scripts/diet_manifest.py` in a subprocess with `PREPARED_BUNDLE_DRIFT_STRICT=1`, rejects a nonzero return code, and records its output in preflight evidence. Packaging ships the script. The script resolves `masking_cfg(manifest.masking_profile)`, so thresholds follow the prepared profile rather than base masking defaults. These code changes were confirmed by reading the updated modules. The active training configuration now also enables strict bundle drift checking by default. An explicit caller override can opt out; suite preflight always forces strict checking.

The coordinating agent reports the fresh full bundle `data/track_setup_20261001/text_prepared.pkl.gz` finished with 62,927 source rows, 117,956 payload rows, 51,712 positive pairs, and 31,619 negative pairs. Strict diet passed: augmented negative fraction 0.3136 and effective positive/negative view ratio 1.1664. These figures are reported run evidence, not an independently repeated preparation or diet run by this audit.

## Readiness conclusion

The all-track architecture contains the essential train/dev/test isolation, frozen input transport, worker ownership, checkpoint selection, postprocessing, and result verification mechanisms. The regenerated setup, hybrid cache, and text bundle passed strict suite preflight. CPU Colab run `1001T050826082308Z` then passed all three training and reporting tracks; archive inventory, checkpoints, reports, withheld test metrics, and runtime teardown were verified. Detailed evidence is in `TRAINING_SMOKE_REPORT_20261001.md`. Confirm success using three completion markers, suite result status, verified downloaded archive, checkpoint files, and nonempty report CSV/plot artifacts.

GPU readiness remains conditional on measured GPU memory; GPU work is deferred by request. Plain legacy smoke is intentionally rejected before provisioning; explicit prepared suite smoke is the supported component-safe flow.
