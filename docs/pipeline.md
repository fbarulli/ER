# Pipeline

## The shape of it

```
dataset.csv
    │
    ├─ RAW VIEW ─────────────────────────────────────────────┐
    │   extract attributes per GTIN                          │
    │   decide every candidate pair  ← the gate              │
    │                                                         ▼
    │                                        canonical_records.csv
    │                                        gate_results.csv
    │   rewrite the measured census into config              │
    │   turn decisions into labels                            │
    │                                                         ▼
    │                                                   labeled_pairs.csv
    │                                                         │
    ├─ TRAINING VIEW ────────────────────────────────────┐   │
    │   dedupe to one representative per product         │   │
    │                                                     ▼   ▼
    │                                              final_validation.csv
    │                                              (+ fold map)
    │                                                     │
    │   augment, build the fixed objective, freeze the epoch batches
    │                                                     │
    └──────────────────────────────►  all_tracks_inputs package
```

No optimizer step lives in preparation. It stops right before training.

## Dependency order

1. Source CSV + config
2. Dedupe + lineage
3. Cross-country positives + number reference
4. Canonical extraction + gates
5. Gate census + labeled pairs
6. Negative supply + discriminator
7. Shared base payload + component split
8. Validation population
9. Augmentation + final text objective
10. Graph projection **and** native token table + epoch batch plan
11. Input validation
12. Publish the immutable bundle
13. Verify

Any change to a source, config or input byte restarts the whole chain. There is no
invalidation matrix and no post-hoc patching.

## Config surfaces

Everything tunable is in config, never at a call site.

| Block | File | Owns |
|---|---|---|
| `gate` | `training.yaml` | tolerances, reason strings, pool families |
| `pairs` | `training.yaml` | similarity thresholds, negative draw |
| `split` | `training.yaml` | fractions, folds, negative fold policy |
| `masking` | `training.yaml` | masking, balanced augmentation, value swaps |
| `audit` | `training.yaml` | census pins, manifest stages |
| `negative_supply` | `training.yaml` | lane mode, GTIN handling |
| `rand_matching` | `training.yaml` | veto gates, penalties, gate census pin |
| `files` / `layouts` | `paths.yaml` | every artifact path |
| `seed` | `paths.yaml` | the one seed |
| — | `model_tracks.yaml` | setup dir, text bundle, epochs, device, parallelism |
| — | `attribute_ablation.yaml` | ablation cohort, channels, batch size |
| — | `vocabulary.json` | flavor, made-from, brand aliases |

Validation is fail-closed everywhere: a bad value or a missing key crashes at
import, never mid-run.

## Text composition

What the encoder actually sees is config-owned (`training.model_input`):

- `profile: cleaned` — `[Brand] [Title] [Attributes]`, underscore compounds split
  so they match source text, discriminative numbers kept, the
  description/breadcrumb channel excluded
- `profile: legacy` — the original committed byte-for-byte stream, kept for
  byte-stability audits

Two redundancy removals ship enabled, both measured as wins on paired ANN A/B:
`emit_field_markers: false` (`[FIELD_*]` markers were 26% of tokens) and
`emit_singleton_pack_token: false` (`pack_qty_1` sat in 76% of texts).

`keep_redundant_attribute_words` stays **true** on purpose. Dropping a plain word
whose structured twin is present measured better Youden but cost 3.15pp of
recall@1, which breaks the standing "hold recall at or above baseline" bar.

## Augmentation

Balanced augmentation mints four populations — `minted_negatives`,
`masked_minted_negatives`, `masked_positives`, `vendor_variation_positives` — plus
static pre-training value swaps, counterfactual twins, and declaration dropout.

Every population must be presented exactly once per epoch. The batch sampler
composition is config-owned (`training.batch_sampler`), scaled to the runtime batch
size; exhausted populations redistribute their slots.

Field quotas are set from **measured** same-GTIN duplicate disagreement rather than
intuition. The same-GTIN census found title-token dissimilarity at 0.703 mean, 91.4%
of duplicates cross-retailer, attribute cells differing in 100% of groups, and real
per-field disagreement: health claims 58.1%, juice features 63.6%, caffeine 33.3%,
sweetener 30.2%, carbonization 26.1%. That measured distribution is the headroom.

Anti-dominance caps keep the swap lanes honest: no one field takes more than ~⅓ of
the picks, and no single `(field, value)` transplant more than ~3%.

Counterfactual twins break exactly one previously-agreed field of a verified match
and are labeled 0 by construction — a one-field break of a real match cannot be the
same product.

Declaration dropout mints the shape retailers actually produce: attribute cells
differ in 100% of same-GTIN groups because each retailer declares a different
partial key subset.

The diet gate (`scripts/diet_manifest.py`) checks the **actual** frozen MNRL
presentations, not the configured counts: negative augmentation floor 0.30,
effective positive-to-negative view ratio ceiling 1.50. Do not relax the thresholds
— an earlier 100-listing projection failed the gate and the failure was left visible
on purpose.

## Ablation

`coverage: all` means every clean, minted-supply and frozen-objective pair, with no
cap — the cohort is exhaustive, which is why it is the memory-hungry step.

Each declared attribute is removed with title, brand and training graph context
held fixed. Baseline and trained-text results stay separate; each track ablates
from its own selected checkpoint.

## GPU execution knobs

| Setting | Value | Note |
|---|---|---|
| `schedule` / `max_parallel` | parallel / 3 | one runtime, MPS |
| `gpu_optimizer_backend` | `cuda_fused` | |
| `gpu_graph_aggregation_backend` | `cuda_segment` | |
| `memory_reservations_gb` | `{}` | **empty — overlap not enforced** |
| `gpu_headroom_gb` | 2.0 | only applied once reservations are populated |

Static graph topology is cached per batch — detached degree counts and support
memberships, no autograd graph, no trainable values.

## Related

- [data-prep.md](data-prep.md) — the stages in detail
- [training.md](training.md) — the three tracks and the Colab launch
