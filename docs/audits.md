# Gate and identity audits

What the review passes concluded, and what is still open. The gate's mechanics live
in [data-prep.md](data-prep.md#the-gate); this is the evidence behind its decisions.

## The measurement that shaped the rules

Same-GTIN duplicate groups — the same product listed by several retailers — were
measured to find out what retailers genuinely disagree about:

| Finding | Value |
|---|---|
| Title-token dissimilarity within a GTIN | 0.703 mean, 0.750 median |
| Duplicates that are cross-retailer | 91.4% |
| Groups where attribute cells differ | 100% |
| Health claims disagree | 58.1% |
| Juice features disagree | 63.6% |
| Caffeine disagree | 33.3% |
| Sweetener disagree | 30.2% |
| Carbonization disagree | 26.1% |

This distribution **is** the augmentation headroom, and it replaced an earlier
weakspot-quota guess. Two consequences:

- Attribute cells differing in 100% of groups is normal, not a defect. Each
  retailer declares a different partial key subset.
- Field quotas are weighted by measured conflict rate, so `sweetener` gets the
  largest share and `juice_content` the smallest.

## Why the descriptor bundle is not identity

Measured against GTIN-labeled ground truth, the descriptor bundle alone merges
**59.0%** of provably-different pairs. A missing descriptor reads as agreement, so
absence of conflict is not evidence.

Consequences, all enforced:

- Undecided descriptor groups are **not** merged. They escalate to an
  owner-adjudicated table.
- Descriptor compatibility is not transitive — every pair in a group is checked,
  not a sample.
- Category is deliberately not a veto. It is as noisy as the attribute cell on
  these rows.
- `made from` is excluded from veto dimensions: 21.2% same-GTIN noise. Water type
  14.9%, caffeine 3.2%.

## Veto dimension evidence

Measured per dimension on 6,898 labeled holdout pairs at the operating threshold:

| Dimension | False merges removed | **True matches lost** |
|---|---|---|
| volume | 60 | 0 |
| pack | 20 | 0 |
| package_type | 13 | 0 |
| flavor | 1 | 0 |
| carbonation | 0 | 0 |
| pulp | 3 | 0 |
| **sweetener** | 6 | **74** |

The full set removes 82 false merges but loses 74 true ones — recall 0.8670 → 0.8536
for +0.001 precision. Excluding sweetener keeps 81 of the 82 and loses none:
precision 0.9751 → 0.9915 at unchanged recall.

That historical evaluation argued for exclusion. **The current list still includes
sweetener.** The exclusion result is retained as review evidence, not as a
description of today's policy. Changing it requires a fresh measured decision.

`pack_material` is identity-safe by construction: the material set unions every
member row at canonical level, and both sides of a true match share the canonical —
so a set-intersection veto *structurally cannot* lose a true match. On the full
corpus, 67,899 hard_no pairs had material populated on both sides, 32,641 of them
genuinely disjoint.

## Brand differentiation

The open question per brand: does between-GTIN difference exceed within-GTIN noise?
If yes, brands can be told apart by comparing volume/pack/package against the
product's own observed variation. If no, their low-confidence pairs go to the
clarification lane instead.

Thresholds: gate-similarity floor 0.35, ≥30 similar pairs and ≥5 multi-listing
GTINs for a verdict, median between-delta ≥ 1.5× median within-noise.

## Mode-flavor evidence

216 of 523 falsified positives carried differing `mode_flavor` values; 69 more had
one side empty and were unreachable by the two-sided rule. That is why a `mode_flavor`
lane exists as **review** evidence rather than a veto — absence is never a veto.

## Packaging level

One-sided in practice: 217 of 13,250 records assert a level, and **zero** pairs have
it on both sides. It still must not become a silent proceed, because a case and a
retail pack are distinct GS1 items. Measured impact: 109 proceed → fallback, 0
hard_no.

Its position in the decision order is deliberate. Checking it earlier downgraded 79
genuine flavour conflicts from `hard_no` to `fallback`.

## Bundle scope

1,783 multi-feed families were censused across all 11 identity dimensions. 80
pack-collision families are held rather than merged. Full data: the machine-readable
census plus the promoted-repair provenance.

## Identity repairs

56 GTIN-scoped repairs are promoted, each with recorded provenance. Repaired matches
grew 19 → 28 across the adjudication rounds. Key reversals:

- ZenWTR and AQUAhydrate 1L plastics reversed five-feed metal majorities
- Alpro Caffe cups are plastic pots, superseding an earlier Asda Flexible → Metal
- Starbucks pair codes are unregistered pseudo-codes
- Weider flavours are registry-confirmed distinct
- Multipower is family-consistent

The largest untrusted-title groups resolve to **keep-distinct**.

## Pseudonymous GTINs

A census of side-codes flagged as suspicious found 8 of the most discriminative were
unregistered pseudo-codes, 1 was a misassigned real code (BioTech → Blackskull), and
1 was brand-correct (Holoslife). Keep-distinct verdicts stand.

## Open capture issues

All 13 source columns are retained in `canonical_records.source_rows` and every
registered attribute has a canonical evidence channel. That does **not** mean
capture or use is complete. Still misread:

- specific numeric roles
- negated ingredients
- nested quantities
- source contradictions

Two structural gaps: the decision engine evaluates more fields than the gate
currently lets decide, and early returns can prevent later clarification from
running at all.

Runtime config is the authority. Census rates and comments in this file are not
accuracy estimates.

## Reproduce

```bash
.venv/bin/python scripts/attribute_universe_census.py
.venv/bin/python scripts/census_bundle_scope.py
.venv/bin/python scripts/pseudo_gtin_census.py
.venv/bin/python scripts/feed_reliability.py
.venv/bin/python scripts/brand_differentiation_audit.py
.venv/bin/python scripts/identity_discovery_replay.py
```

Machine-readable outputs sit beside the findings and are the authoritative record;
the prose summaries here are the readable index into them.
