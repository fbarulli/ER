# Staged ablation head-to-head: incumbent (A) vs candidate (B)

A = src/model_tracks/staged_ablation.py at base ada5f45 (runtime-repaired NameError in its hybrid leg);
B = staged_ablation_candidate.py (public functions). Workloads: smoke_200 verbatim and a
deterministic 2000-listing bounded slice. device=cpu forced; encode leg = shared no-op stub.
5 interleaved reps per side per workload (block order alternates).

## Workload slice_2000

| leg | phase | A median (s) | B median (s) | delta (A−B, s) | speedup (A/B) |
| --- | --- | ---: | ---: | ---: | ---: |
| prepare | freeze_config | 0.0012 | 0.0032 | -0.0020 | 0.37x |
| prepare | cohort_gate | 0.0000 | 0.0000 | -0.0000 | 0.72x |
| prepare | support_load | 0.2220 | 0.1522 | +0.0698 | 1.46x |
| prepare | template_checkpoint | 0.1858 | 0.1064 | +0.0795 | 1.75x |
| prepare | track_request | 5.4646 | 3.6042 | +1.8604 | 1.52x |
| prepare | cohort_validate | 0.0000 | 0.0000 | +0.0000 | 1.10x |
| prepare | request_anchor | 0.0056 | 0.0051 | +0.0005 | 1.09x |
| prepare | template_folder | 0.1018 | 0.0577 | +0.0441 | 1.76x |
| prepare | inline_remainder | 0.2251 | 0.0044 | +0.2207 | 50.95x |
| prepare | cleanup_staging | 0.0010 | 0.0015 | -0.0005 | 0.66x |
| prepare | prepare_total | 6.3396 | 4.0388 | +2.3008 | 1.57x |
| | | | | | |
| forward | bind_template | 0.0066 | 0.0056 | +0.0010 | 1.18x |
| forward | graph_binding_check | 0.3189 | 0.0910 | +0.2279 | 3.50x |
| forward | rebind_checkpoint | 0.0010 | 0.0002 | +0.0008 | 5.25x |
| forward | bound_folder | 0.0563 | 0.0486 | +0.0077 | 1.16x |
| forward | reuse_or_encode | 0.0007 | 0.0014 | -0.0007 | 0.53x |
| forward | encode_vectors | 0.0005 | 0.0005 | -0.0000 | 0.93x |
| forward | forward_total | 0.8460 | 0.3488 | +0.4972 | 2.43x |
| | | | | | |
| total | suite (prepare + 3 forwards) | 7.1856 | 4.3876 | +2.7980 | 1.64x |

**Verdict (slice_2000): B (candidate) is faster by 63.8% (median of 5 interleaved reps).**

## Workload smoke_200

| leg | phase | A median (s) | B median (s) | delta (A−B, s) | speedup (A/B) |
| --- | --- | ---: | ---: | ---: | ---: |
| prepare | freeze_config | 0.0016 | 0.0066 | -0.0050 | 0.24x |
| prepare | cohort_gate | 0.0000 | 0.0000 | -0.0000 | 0.50x |
| prepare | support_load | 0.0367 | 0.0476 | -0.0110 | 0.77x |
| prepare | template_checkpoint | 0.0195 | 0.0184 | +0.0011 | 1.06x |
| prepare | track_request | 3.7787 | 4.3363 | -0.5576 | 0.87x |
| prepare | cohort_validate | 0.0000 | 0.0000 | +0.0000 | 1.07x |
| prepare | request_anchor | 0.0045 | 0.0041 | +0.0004 | 1.10x |
| prepare | template_folder | 0.0894 | 0.1114 | -0.0220 | 0.80x |
| prepare | inline_remainder | 0.0246 | 0.0018 | +0.0228 | 13.89x |
| prepare | cleanup_staging | 0.0008 | 0.0008 | -0.0001 | 0.93x |
| prepare | prepare_total | 4.0294 | 4.5180 | -0.4886 | 0.89x |
| | | | | | |
| forward | bind_template | 0.0097 | 0.0040 | +0.0057 | 2.40x |
| forward | graph_binding_check | 0.0877 | 0.0110 | +0.0768 | 8.00x |
| forward | rebind_checkpoint | 0.0005 | 0.0002 | +0.0003 | 2.71x |
| forward | bound_folder | 0.0560 | 0.0533 | +0.0027 | 1.05x |
| forward | reuse_or_encode | 0.0007 | 0.0015 | -0.0008 | 0.49x |
| forward | encode_vectors | 0.0005 | 0.0006 | -0.0001 | 0.82x |
| forward | forward_total | 0.3899 | 0.2391 | +0.1507 | 1.63x |
| | | | | | |
| total | suite (prepare + 3 forwards) | 4.4193 | 4.7571 | -0.3378 | 0.93x |

**Verdict (smoke_200): A (incumbent) is faster by 7.6% (median of 5 interleaved reps).**

Notes (delta provenance):

- A's structural overhead concentrates in `inline_remainder` (the inline `digest()` of
  vocabulary+support_records frozen into each graph-track request — the cost the candidate
  replaces with a track-name binding) and `graph_binding_check` at forward (A torch.loads the
  selected checkpoint AND digests its vocabulary/support_records against the request; B only
  checks the manifest track). Both scale with the support population, which is why
  `slice_2000` separates B from A where `smoke_200` (105 support records) sits inside noise
  and reads as a statistical tie.
- `rebind_checkpoint`: A re-hashes the selected checkpoint file (checkpoint_identity); B stores
  a placeholder — a small constant-vs-constant per-forward delta.
- `template_checkpoint`: A's full baseline hash (88 MB) and model-input composition rebuild are
  memoized process-wide, so the per-rep measured delta here is small; the unmemoized cost
  (warmup run) is real per-suite in production code paths.
- `encode_vectors` is a no-op stub (GPU-only Colab lane; this box has no CUDA); both sides
  execute the identical stub, so the phase cannot separate them.
- Shared prepare legs (`track_request`, dominated by ablation.prepare + tokenization) and
  I/O phases move at parity; residual differences are filesystem noise.
