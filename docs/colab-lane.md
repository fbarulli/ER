# Colab lane

`er-colab` (console script for `cli.colab:main`) drives both Colab lanes,
declared once in `config/training.yaml` `colab.lanes`: `bundle` (`--gpu CPU`,
the CPU data-bundle lane) and `training` (any other accelerator — the GPU lane
trains the installed bundle and runs no checks and no CPU prep on the GPU VM;
the only GPU-side check is data presence at the sparse checkout,
`cli.colab_runtime.validate_runtime_checkout`).

## The calls

```bash
# full GPU cohort
er-colab --what tracks --tracks-config config/model_tracks.yaml --gpu T4 --allow-gpu

# GPU smoke
er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu T4 --allow-gpu

# CPU rebundle (dataset default repo:dataset.csv, config/paths.yaml `dataset`;
# explicit archive: --dataset-csv <file>)
er-colab --what bundle --gpu CPU

# stop the session's VM
er-colab --what stop
```

Existing bundle: `--prepared-input-package <archive|bundle-dir>` overrides;
with none named the launch consumes the fixed slot
`results/kaggle_lane/full/bundle` (`cli.colab.default_prepared_input_package`:
staging root + `default_cohort` + `install_dir` from config). Every CPU
promote/rebundle OVERWRITES that slot, so the plain calls above always train
the newest full bundle.

`--preflight-only` prints the JSON checks (package verify / suite preflight /
lane contract) and returns — no VM is started.

## Boundaries

| crossing | contract |
|---|---|
| local → GPU VM | transport only: `Bundle.load(...).reader()` install; the single local presence check is `ColabLaneBase.require_launch_checkout` (the sparse-checkout contract the VM clones) |
| CPU bundle lane → bundle | seal runs here: `write_prepared_bundle` + `model_tracks.package` (`seal_prepared_bundle` → `verify_prepared_bundle`) before delivery; `prepare_all`'s `full_bundle` stage writes worker 1's bundle and the training-bundle stage writes every other entry `ColabGPULane.prepared_bundles` hands the GPU lane |
| bundle → GPU trainer | pure IO: manifest + payload read, the seal logged as provenance, never gating |

## The bundles the CPU lane builds, declared once

`config/training.yaml` `colab.prepared_bundles` is the ONE declaration of the
prepared training bundles: a `dir`, the `file_template` that renders each
entry's name, and the entries themselves (the WORKER is the key; an entry's
masking profile and model default to the trainer's own `masking.profile` /
`training.base_model`). Both lanes read it, so a launch can never ask for a
bundle the CPU lane does not build.

* **The CPU lane builds them.** `--what bundle --gpu CPU` runs
  `training.prepare_all` and then, in the SAME detached process (same log,
  same status file, same budget, same poll), the training-bundle stage
  (`cli.colab_lane_cpu_bundles`): `training.train --prepare-bundle` once per
  declared entry, each one diet-gated by the tracked `scripts/diet_manifest.py`
  and refused loud (non-zero rc, full traceback) if it misses the contract.
  Worker 1's bundle is the exception: `prepare_all`'s own `full_bundle` stage
  writes, seals and handoff-verifies it, so the stage never rebuilds it. The
  bundles land under `data/prepared/full/`, which the delivery already ships as
  one directory — no delivery change.
* **The GPU lane consumes them.** `ColabGPULane.prepared_bundles` resolves the
  launch's own `(workers, masking profiles, model)` against that declaration:
  `--what dual-train`, `--workers 2`, another masking profile or model all work
  again, CPU-free. A request the declaration has no entry for fails loud,
  naming the exact entry to declare — it never prepares one locally.

## Nothing prepares locally

The launcher never builds data, and the CPU bundle lane is the only machine that
does. The legacy local `training.train --prepare-bundle` build, its prewarm
thread, its diet-gate wiring and the validation-source upload it fed are deleted;
the remote worker resolves its own scored population and ignores the retired
`--validation-source` (`training/complete_colab_worker.py`). A request whose
inputs the CPU bundle lane does not own — a retired sampled run, a different
source CSV, a resume id or run label — is refused by the same boundary rather
than served.
