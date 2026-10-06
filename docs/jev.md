# JEV audits

JEV is an LLM used as an **independent second opinion** on gate decisions. Its
judgments are a signal to investigate, never ground truth.

## How it works

Each candidate pair is sent to the model twice — both orderings — so an
order-dependent judgment shows up as a discrepancy. Mean absolute order difference
across the rounds was 0.023–0.027, with a worst case of 0.50.

JEV only ever sees the **first source listing** for each GTIN, with the description
truncated to 300 characters. The gate sees canonical evidence merged across all
listings. So disagreement is expected and informative: it usually means the gate
over-trusts a merged view or JEV under-reads a truncated one.

## Rounds

| Round | Pairs | Calls | Focus |
|---|---|---|---|
| 1 | 80 | 160 | proceed split by mode-flavor evidence, plus fallback and negative-similarity strata |
| 2 | 55 | 110 | post-fix proceed survivors and newly minted negatives/fallbacks |
| 3 | 500 | 1,000 | broad: 84 pairs per positive stratum, 83 per negative stratum, across high/lower similarity |
| 4–8 | — | — | incremental rounds behind the final checkpoint |
| 9 | 50 | 50 | explicit repeat of old pairs to measure judgment stability |

All 635 pairs from rounds 1–3 are **reserved**. Any future sampler must read
`sample_ledger.json` or exclude the unordered pairs, otherwise swapping GTIN order
reintroduces a pair that was already tested.

Round 9's repeat found 48 judgment bands stable and 2 moving from *same* to
*uncertain*. No queued calls remain.

## What the scores mean

| Stratum | Mean score | Below 0.2 |
|---|---|---|
| proceed survivors | 0.136 | 31 / 40 |
| newly minted hard_no | 0.024 | 30 / 30 |
| newly minted fallback | 0.042 | 20 / 20 |
| retained hard_no | 0.017 | 20 / 20 |

The negative strata scoring near zero is the expected result and validates the
direction of the flavor fix.

The worrying part is the **first row**: proceeding pairs that JEV scores near zero.
Across both early rounds, 28 distinct low-score pairs still proceed. The fixes are
supported but do **not** resolve the remaining matching errors.

The reassuring part: **no rejected pair had consistently high JEV confidence** in
either audit. Zero. So the gate is not over-rejecting.

These samples are stratified, so they do not estimate population accuracy or
recall.

## What was changed because of JEV

Round 2 supported withdrawing two kinds of rescue:

- **partial-overlap** flavor rescues
- **semantic-family** flavor rescues

Generic-only containment stays inconclusive. Equal specific token bags match;
specific containment remains a subset; divergent specifics conflict. Both flavor
spellings follow the same rule, and exact/alias matches keep precedence.
Non-flavor family rescues are untouched.

Later rounds implemented, with offline replay rather than new live calls:

- category-derived flavors removed
- anise, cassis, blackcurrant and exotic flavor vocabulary recognized
- juice "With Bits" and smooth texture captured
- carbonation strength, cola/mate family and named variants preserved as **review**
  evidence

Missing declarations now require review rather than an invented rejection.
Configured vetoes keep priority. The shared review contract is wired into both the
three-way gate and the targeted SKU/canonical gate.

Replaying 900 round-7 pairs over 1,364 rebuilt canonicals moved 167 approvals to
review and 21 to rejection. Of 275 low-scoring prior approvals, 185 stopped being
approved and 90 remained. All six high-scoring approvals survived.

Golden model-input bytes changed for only 11 of 855 examples. Canonical golden
bytes are unchanged.

## Still open

- Source measurement disagreement
- Nested pack counts
- Named variants
- Configured carbonation policy
- Mode-flavor false review

The shipped CSV census and labels still need a controlled regeneration before
training on the updated features.

One historical ANN test remains a pre-existing failure: it assumes all committed
proceed pairs are conflict-free, while ~6,443 are rejected by the flavor policy
that predates these changes. It was deliberately not re-pinned to stale artifacts.

## Reproduce

```bash
.venv/bin/python jev/verify_audits.py
.venv/bin/python jev/stage_sample_3.py --pairs 500
.venv/bin/python jev/run_audit.py --staging jev/sample_doubled_3.json \
  --out jev/audit_results_3.jsonl
```

Completed ordered pairs are skipped on resume. A custom staging path requires an
explicit output path. Per-round detail stays in the raw artifacts:
`audit_results*.jsonl`, `verification_results.json`, `report_9.json`, and
`full_evidence/saved_jev_replay.json`.
