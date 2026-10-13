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
| local → GPU VM | transport only: `Bundle.load(...).reader()` install; the single presence check is the sparse-checkout validation |
| CPU bundle lane → bundle | seal runs here: `write_prepared_bundle` + `model_tracks.package` (`seal_prepared_bundle` → `verify_prepared_bundle`) before delivery |
| bundle → GPU trainer | pure IO: manifest + payload read, the seal logged as provenance, never gating |
