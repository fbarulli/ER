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

Verification used `uv`: four focused W&B/plot tests passed, Ruff passed, and
the branch's invariant suite passed four tests with one pre-existing expected
failure. Mypy, Pyright and Ty are unavailable in the shared environment;
type checking was not run.

## Unfinished work, one unit at a time

1. Investigate the training-side gradient-norm value before trusting the
   optimization telemetry. The loop also clamps the reported norm to the
   clipping threshold; inspect that measurement contract.
2. Finish the existing autocast-cache draft: reuse weight casts between
   forwards, invalidate at optimizer/adversarial mutations, and preserve
   loss/backward precision and R-Drop behavior.
3. Finish the existing DDP accumulation draft: synchronize only on update
   boundaries while preserving the adversarial path's gradient semantics.
4. Finish the existing independent-dimension compile draft: equal dimensions
   in the first batch must not accidentally tie later variable shapes.
5. Finish the existing optimizer-group draft: group equal learning rates and
   preserve every incoming group option. The committed implementation drops
   options such as weight_decay/betas and creates one group per parameter.
6. Validate each completed optimization on a comparable real T4 workload,
   including numerical behavior, peak memory, phase timing, completion and
   artifact harvest. CPU tests do not establish NCCL health or GPU speedup.

The four performance drafts predate this continuation and remain uncommitted.
Their initial local checks passed 11 non-DDP tests; the CPU/Gloo DDP regression
run passed 18 tests after allowing localhost networking. Those checks do not
make the drafts landed or GPU-validated. Dataset edits remain separate.

## Integration gap

The W&B fix is committed on the existing `laya-candidate` integration branch.
It is not on `main`: that branch lacks the plot/reader implementation and the
candidate carries dataset and other-lane changes. Landing the whole candidate
would exceed the current LAYA-only, no-dataset scope. This is an outstanding
integration obligation, not a claim that a pushed branch completes delivery.
