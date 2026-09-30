# GPU optimization after lifecycle setup

Owner sequence: wire training, post-training inference/indexing, reports and
artifact collection first; then optimize GPU training. No full training run
or GPU provisioning is part of the current setup work.

The standalone graph workers already call post-training completion using the
dev-selected checkpoint. Local lifecycle tests cover inference, reports,
resume and portable result collection. The shared Colab entry point dispatches
all three prepared tracks through one supervisor, with isolated worker outputs
and a consolidated result archive.

Enable `profiling: true` in the suite YAML to collect PyTorch CPU/CUDA traces,
operator timing and memory summaries. Each worker writes its own profile;
profiles travel with the downloaded results. Profiling captures three training
steps (three epochs for graph tracks), or the shorter smoke run. It introduces
overhead and should be disabled for final throughput comparisons. Open
`training_trace.json` in Perfetto or Chrome tracing; `operator_summary.txt`
provides a readable timing summary.

## Measurement harness

`graph_tracks.benchmark` measures synchronized GPU training steps on prepared
inputs. It separates warmup/compile cost, reports peak allocation and loss,
and can export a profiler trace. It does not save a model or score dev/test.
Run only after setup preflight and when GPU benchmarking is authorized:

```bash
PYTHONPATH=src python -m graph_tracks.benchmark \
  --config data/graph_worker/gnn_only/worker.yaml \
  --output results/gnn_gpu_eager.json --profile results/gnn_gpu_eager_trace.json
PYTHONPATH=src python -m graph_tracks.benchmark \
  --config data/graph_worker/gnn_only/worker.yaml \
  --output results/gnn_gpu_compiled.json --compile
PYTHONPATH=src python -m graph_tracks.benchmark \
  --config data/graph_worker/gnn_only/worker.yaml \
  --output results/gnn_gpu_fused.json --fused-optimizer
```

Repeat for hybrid; compare one change at a time on the same device/input/seed.
The harness supports `--bf16` as an experiment, but graph reductions and
high-degree means require numeric parity checks before worker adoption.
No GPU is available in the current local host, so none of these options is
claimed to improve runtime yet. Benchmark options do not change worker defaults.

## Candidate work, in order

1. Profile relation aggregation, graph encoding, per-parameter gradient
   telemetry, dev encoding, checkpoint serialization and artifact publication
   separately. Include end-to-end epoch time in the final speed comparison.
2. Remove unnecessary work: cache static degree counts/edge masks, encode
   only supervised endpoints for the training loss, and encode only dev
   endpoints for selection while preserving all training graph support.
3. Measure TorchInductor compilation and fused AdamW independently. Account
   for compilation cost on the initial ten-epoch workload and preserve
   checkpoint/optimizer/resume compatibility.
4. Measure mixed precision with FP32 graph reductions, FP32 loss/normalization
   where needed, and explicit BF16 capability checks. T4 needs a separate
   FP16/GradScaler experiment; do not assume BF16 support.
5. Consider custom Triton only for a measured remaining aggregation or
   pair-scoring bottleneck. Test forward/backward parity, empty relations,
   unknown values, high-degree hubs and deterministic execution. A package
   installation alone does not accelerate these workers.

Adopt changes only after gradient/score parity, lifecycle/resume checks and
measured throughput/memory improvement. Keep test labels sealed. Review text
trainer GPU settings separately; graph benchmarks do not establish text speed.

Primary references: [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide),
[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile),
[AdamW implementations](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html),
and [Triton fusion tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

## Implemented static graph optimization

`graph_tracks.pooling` caches per-batch detached degree counts and non-unknown
support memberships. Repeated steps retain only topology, with no autograd
graphs or trainable values. Eager execution invalidates metadata when an edge
tensor is replaced or edited in place. Inference-mode and compiled inputs
are immutable; the benchmark primes topology before compilation. Graph-disabled
encoding also skips unused message projections. Existing checkpoint keys and
train-only support semantics are unchanged.

A synthetic local CPU diagnostic used 1,000 listings, 800 support listings,
eight relations, two memberships per relation, hidden dimension 64, output
dimension 128, two threads, five warmup steps and 30 measured forward/backward
steps per variant (PyTorch 2.14.0+cu130). Median baseline was 0.8633 seconds
and cached topology 0.5862 seconds (1.47x). Scores matched exactly and gradients
matched within rtol 1e-4 / atol 1e-7; the largest observed initial gradient
difference was 3.7e-9. Concurrent host tests and synthetic inputs limit this
result: it establishes neither production throughput nor GPU speedup.
The cached CPU profile remained dominated by matrix multiplication (44.5%
self time) and scatter addition (13.4%); custom Triton work is not justified
by this CPU profile. Reproduce from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmarks/graph_pooling_cpu.py
```

The diagnostic loads its original model from pinned Git history. It does not
change worker configuration, publish artifacts, or evaluate held-out labels.
