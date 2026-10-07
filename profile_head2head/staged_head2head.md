# Staged ablation head-to-head: incumbent (A) vs candidate (B)

A = src/model_tracks/staged_ablation.py at base ada5f45 (runtime-repaired NameError in its hybrid leg);
B = staged_ablation_candidate.py (public functions). Workloads: smoke_200 verbatim and a
deterministic 2000-listing bounded slice. device=cpu forced; encode leg = shared no-op stub.
5 interleaved reps per side per workload (block order alternates).

## Workload slice_2000

| leg | phase | A median (s) | B median (s) | delta (A−B, s) | speedup (A/B) |
| --- | --- | ---: | ---: | ---: | ---: |
| prepare | freeze_config | 0.0011 | 0.0024 | -0.0013 | 0.46x |
| prepare | cohort_gate | 0.0000 | 0.0000 | -0.0000 | 0.63x |
| prepare | support_load | 0.0815 | 0.0764 | +0.0052 | 1.07x |
| prepare | template_checkpoint | 0.0415 | 0.0406 | +0.0009 | 1.02x |
| prepare | track_request | 1.5792 | 1.6029 | -0.0237 | 0.99x |
| prepare | cohort_validate | 0.0000 | 0.0000 | +0.0000 | 1.06x |
| prepare | request_anchor | 0.0029 | 0.0029 | -0.0000 | 0.99x |
| prepare | template_folder | 0.0259 | 0.0258 | +0.0001 | 1.01x |
| prepare | inline_remainder | 0.0717 | 0.0016 | +0.0701 | 45.48x |
| prepare | cleanup_staging | 0.0008 | 0.0010 | -0.0002 | 0.81x |
| prepare | prepare_total | 1.8123 | 1.7640 | +0.0483 | 1.03x |
| | | | | | |
| forward | bind_template | 0.0035 | 0.0037 | -0.0002 | 0.95x |
| forward | graph_binding_check | 0.1047 | 0.0354 | +0.0693 | 2.96x |
| forward | rebind_checkpoint | 0.0010 | 0.0002 | +0.0008 | 5.35x |
| forward | bound_folder | 0.0191 | 0.0188 | +0.0004 | 1.02x |
| forward | reuse_or_encode | 0.0006 | 0.0014 | -0.0007 | 0.47x |
| forward | encode_vectors | 0.0004 | 0.0005 | -0.0000 | 0.95x |
| forward | forward_total | 0.2696 | 0.1329 | +0.1367 | 2.03x |
| | | | | | |
| total | suite (prepare + 3 forwards) | 2.0819 | 1.8968 | +0.1850 | 1.10x |

**Verdict (slice_2000): B (candidate) is faster by 9.8% (median of 5 interleaved reps).**

## Workload smoke_200

| leg | phase | A median (s) | B median (s) | delta (A−B, s) | speedup (A/B) |
| --- | --- | ---: | ---: | ---: | ---: |
| prepare | freeze_config | 0.0011 | 0.0024 | -0.0013 | 0.46x |
| prepare | cohort_gate | 0.0000 | 0.0000 | -0.0000 | 0.74x |
| prepare | support_load | 0.0142 | 0.0135 | +0.0007 | 1.05x |
| prepare | template_checkpoint | 0.0061 | 0.0049 | +0.0012 | 1.24x |
| prepare | track_request | 1.2158 | 1.2078 | +0.0080 | 1.01x |
| prepare | cohort_validate | 0.0000 | 0.0000 | +0.0000 | 1.17x |
| prepare | request_anchor | 0.0028 | 0.0028 | +0.0000 | 1.01x |
| prepare | template_folder | 0.0244 | 0.0241 | +0.0003 | 1.01x |
| prepare | inline_remainder | 0.0084 | 0.0015 | +0.0069 | 5.62x |
| prepare | cleanup_staging | 0.0007 | 0.0008 | -0.0001 | 0.87x |
| prepare | prepare_total | 1.2779 | 1.2603 | +0.0176 | 1.01x |
| | | | | | |
| forward | bind_template | 0.0034 | 0.0035 | -0.0001 | 0.98x |
| forward | graph_binding_check | 0.0133 | 0.0058 | +0.0075 | 2.28x |
| forward | rebind_checkpoint | 0.0005 | 0.0001 | +0.0003 | 3.05x |
| forward | bound_folder | 0.0183 | 0.0183 | +0.0000 | 1.00x |
| forward | reuse_or_encode | 0.0007 | 0.0012 | -0.0005 | 0.57x |
| forward | encode_vectors | 0.0004 | 0.0004 | -0.0000 | 0.93x |
| forward | forward_total | 0.0846 | 0.0693 | +0.0153 | 1.22x |
| | | | | | |
| total | suite (prepare + 3 forwards) | 1.3625 | 1.3296 | +0.0329 | 1.02x |

**Verdict (smoke_200): B (candidate) is faster by 2.5% (median of 5 interleaved reps).**

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
