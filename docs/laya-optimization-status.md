# LAYA optimization status — 2026-10-11

The work was heading toward removing repeated work from each training trial,
then measuring throughput with trustworthy phase timing and per-epoch metrics.
This is a reconstruction from Git history, current source, fetched artifacts,
and read-only W&B queries. Dataset work is excluded.

## Profile evidence

The baseline family is `laya_hpo_1kbaseline_20261010T165809Z`:

- [Trial 0](https://wandb.ai/fbarulli-none/e-r/runs/9t2h3qh3): its last
  recorded phase summaries include backward 1548.885 s and forward 806.069 s.
- [Trial 1](https://wandb.ai/fbarulli-none/e-r/runs/wdz6c0ro): backward
  742.384 s and forward 385.428 s. The same dev work appears as dev_eval
  449.623 s, calibration 448.531 s, and calibration_records 448.520 s.
  These overlapping timings must not be added as three independent costs.
- Trial 1's console profiler table records 16.364 s self CPU time, 8.695 s
  self CUDA time, 43,752 `_to_copy` calls and 50,862 `copy_` calls in the
  captured window. These are profile-window totals, not full-epoch times or
  proof of an end-to-end speedup. The console is preserved under the lane's
  `plots/<baseline-tag>/profile_console.log` artifact directory.

The inference behind the campaign was that repeated CPU dispatch, weight
casting and evaluation overhead were starving useful GPU work. The existing
documentation's other profile-window numbers describe another capture; they
are not substituted for this inspected trial's values.

## Committed direction

| Commit | Change | Repeated work addressed |
|---|---|---|
| `0d5f913` | Shared W&B helpers; automatic global steps with an epoch coordinate | Epoch rows dropped after other sinks advanced the step |
| `e9a6b7a` | Idempotent calibration timing; nested work counted by its enclosing phase | Triple attribution and wrappers stacking across trials |
| `4b2cfa3` | Configured batched AMP evaluation for HPO | Small, fp32 dev forwards |
| `52a4746` | One process-wide encode memo, reset per trial | Train/eval repeatedly tokenizing the same items |
| `40a3384` | HPO skips redundant full-dev before/after evaluation | Scoring the same dev population in addition to per-epoch evaluation |
| `1d5666e` | Compilation after DDP wrapping; searchable compile mode | Unused compile control and large operator dispatch overhead |
| `0694529` | Pinned asynchronous whole-batch transfer; scoped dev-collate memo | Blocking copies and rebuilding unchanged dev batches |
| `c5e13e6` | Checkpointing default off; HPO micro-batches 8/16/32 | Backward recomputation and dispatch overhead per sample |
| `fd64e7f` | Three parallel trials; one sampler log per worker | Shared telemetry attribution between independent workers |
| `f7d485b` | NCCL heartbeat timeout derived from the DDP timeout | Teardown abort preventing checkpoint/receipt completion |

HPO currently specifies three trials, up to three parallel worker processes,
and at most three epochs per trial. It retains the memory/speed controls;
it does not silently reduce batches or parallelism to avoid OOM. Checkpointing
and compilation remain searchable. Compilation is off by default.

## Completed in this continuation

`c115a73` fixes the consolidated history reader introduced by `4d20855`:

- Read sparse W&B rows without requiring epoch, GPU and network keys together.
- Share typed history/point models across the network, reader and plot seams.
- Use epoch coordinates for training/dev metrics and global steps for GPU/network.
- Report invalid points with metric, source step and full traceback.
- Fail explicitly when no metric can be plotted; preserve the diagnostic JSON.

Verified against [the latest smoke run](https://wandb.ai/fbarulli-none/e-r/runs/ht50wil1):
the old query returned zero rows despite 1,111 stored rows. The fixed artifact
contains the epoch's loss, learning rate, dev accuracy, dev loss and recorded
epoch time, plus 1,086 points for each GPU series. Dev accuracy is
0.9742000102996826. `train/grad_norm` at step 1050 is `NaN`; the report now
names this gap instead of silently dropping it. It does not repair the
training value or reinterpret the run's crashed state as success.

## Completed optimization and boundary fixes

All changes stay on `laya-candidate`, the user's requested destination; no
merge into `main` is intended. Dataset drafts are excluded.

| Commit | Verified behavior |
|---|---|
| `f5bc3d9` | Equal-rate optimizer groups retain weight decay, betas and incoming options |
| `cff0bee` | Dynamic compilation keeps unrelated batch dimensions independent |
| `67dd84b` | DDP accumulation synchronizes at update boundaries, preserving adversarial semantics |
| `299f539` | AMP weight casts are reused until parameter mutation; loss/backward precision retained |
| `01b99fe` | Reinstalled evaluation policy updates without stacking wrappers |
| `1d3cefc` | Encode memo retains tokenizer/item identities until trial reset |
| `97e9b8e` | Post-training calibration reuses identical logits and releases retained models in finally |
| `62c9d19` | Typed dev artifacts surface lost confusion records with full tracebacks |
| `950462a` | Logged gradient norms retain the pre-clipping magnitude |
| `e6e8bb4` | Epoch timing includes evaluation and checkpoints |
| `a2d361f` | Host optimizer and dev evaluator import their canonical class dependencies |
| `2dfcabc` | Optional optimizer failures retain full tracebacks and share construction |
| `c7a7def` | Pydantic dev reports require scoring fields; sparse history extras and memo metadata are typed |

The host-boundary regressions were reproduced before fixing: extraction in
`0694529` left TrainingOptimizer dependent on injected ParamGroupBuilder;
`4b2cfa3` left DevEvaluator dependent on injected DevReport. Both now import
the canonical classes, with public host behavior and generated-kernel coverage.
DevReport is now a Pydantic boundary model: accuracy and loss are required;
missing fields fail validation instead of masquerading as zero. Its constructor
assembly and duplicated optional-field parsing were removed.
W&B sparse extras are explicitly typed on their Pydantic owner; function
memo metadata uses the standard function attribute dictionary.

## Validation and remaining limits

Checks use `uv`; scoped Ruff and Ty checks cover the changed optimizer,
evaluation, compile/cache and W&B history owners. The combined regression
suite covers CPU/Gloo DDP, accumulation, AMP cache parity, dynamic shapes,
optimizer updates, evaluator artifacts, epoch timing, W&B plotting/steps,
generated kernels, HPO SSOT parity and repository invariants.

CPU tests do not establish T4 speedup, GPU peak memory or NCCL health. The
latest inspected W&B smoke run remains crashed and its NaN gradient norm is
not repaired by correcting the norm measurement. No new remote run was
launched in this continuation; a Kaggle COMPLETE status alone is not evidence
of healthy training. No msgspec migration or benchmark was performed.

The final combined suite passed **128 tests with one pre-existing expected
failure** in 80.43 seconds, including all training-control tests. Three
warnings are pre-existing: two Pydantic instance-field deprecations and the
scheduler-order warning in a focused scheduler test.
Full-file Ruff on `src/core/laya_controls.py` reports 16 pre-existing findings,
with identical diagnostic counts/codes before and after this change. These
include silent malformed abstention records (`src/core/laya_controls.py:231`),
loop-variable closure warnings (`src/core/laya_controls.py:537`), and truncated profiler
errors (`src/core/laya_controls.py:893`, `:919`, `:928`, `:965`). No baseline or allowlist was widened. Other
changed Python files pass Ruff, and scoped Ty checks pass. Full-repository
type checking was not performed.

Other observed legacy gaps are not concealed by this delivery:

- `src/core/laya_train_parts.py:136`: malformed targets are skipped while
  computing class weights; this data-related path is outside the requested scope.
- `src/cli/laya_train_patch_loop.py:60`: malformed balancing targets become
  label -1; this data-related path is likewise excluded.

Dataset scripts and the pre-existing `docs/laya-lane.md` draft remain
uncommitted and untouched by this optimization delivery.
