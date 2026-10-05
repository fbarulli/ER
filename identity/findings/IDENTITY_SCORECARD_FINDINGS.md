# Identity-evidence scorecard — findings and decision (2026-10-05)

Branch: `identity-evidence-union` (worktree study, repo untouched; this
document + the CSVs/JSON are the promoted record). Script:
`scripts/dedupe_predicate_scorecard.py` on branch
`identity-evidence-union` (the only new tracked-intent file).

## What was measured

Whether additional evidence surfaces can disambiguate the product-identity
cases where typed descriptors are ABSENT (the 59.0% false-merge residual:
absence of a conflict reads as agreement on provably-different pairs).

Evaluation set (deterministic, seed 42, dataset sha verified
`539c2472…312fab88c`, 71,623 rows / 26,209 gtin-trusted):
- label-0 (provably different products): 1,044 chain pairs — same
  (retailer, title) with different zero-filled-14 canonical gtins,
  691 groups.
- label-1 (same product): 4,000 pairs from 6,821 same-gtin families.

The baseline surface IS the cleaned typed one
(`core.sku_identity.row_identity`: gtin, title, attribute, description,
url slug tokens, image filename tokens, brand, category — call chain and
exact line ids in `DEDUPE_BASELINE.md`), so this study measured candidates
*on top of* regex cleaning, not instead of it.

## Results (false-merge on provably-different; true-merge on same-gtin)

| stack | FM | TM | recovery | X=0.05 | X=0.10 | X=0.20 | verdict |
|---|---|---|---|---|---|---|---|
| S0 typed-only (current) | 0.2500 | 0.8013 | — dims: volume 599, pkg_material 307, brand 188, pkg_type 139 | BASELINE | BASELINE | BASELINE | current residual |
| S1 + composed-row claims | 0.2414 | 0.7843 | 9 pairs (flavor 7, sweetener 2) | SUGGESTION | SUGGESTION | SUGGESTION | marginal; −1.7pp recall |
| S2 + exact image_url (same retailer) | 0.2414 | 0.7843 | 225/4000 label-1 decided (104 cross-retailer); 35 label-0 contaminated | SUGGESTION | SUGGESTION | SUGGESTION | ranker/positive lane only |
| S3 + GTIN-sibling closure (canonical_records) | 0.2414 | 0.7843 | 441/1000 absence-restorable (sweetener 257, carbonation 106, flavor 54, pulp 24) | SUGGESTION | SUGGESTION | SUGGESTION | strongest absence-repair; positive lane only |

S4 (embedding-band ordering) skipped: no local checkpoint in the study
worktree.

## Conclusion

1. No stack reaches VETO-CLASS at any promotion bar: false-merge stays in
   the 0.23–0.25 band. Keep-apart + owner escalation is CONFIRMED with
   data, and no merge-decision change is promoted.
2. The residual failure mode is ABSENCE (both sides declare nothing
   discriminative), not noise: regex cleaning is in the baseline and the
   residual persists.
3. S3 recovers 44% of missing-field pairs from canonical sibling
   declarations — an INPUT-REPAIR supply (positive lane), not a decision
   rule. S2 image-exactness decides positives but carries contamination
   (35 label-0 pairs with identical images across different gtins).

## Next step (owner-approved direction)

FALSE-MERGE AUTOPSY before any product-card construction: sample 30 of the
1,044 label-0 pairs (the false-merge band), print side-by-side product
cards, hand-classify:
- "same product, different retail semantics" (multipack / bundle / variety
  GTINs) — NOT a defect: keep-apart is correct by retail identity, and the
  59%-style framing was measuring intent, not leaks.
- "true confusions" (recoverable signal) — then and only then build the
  organized product-card representation per GTIN family (all columns,
  noise-cleaned, sibling-closed, image-locked) and test conflict
  COHERENCE per card as an identity signal.
S3's sibling-closure supply may be used as offline input repair in the
positive lane during any such card construction — decision rule stays
untouched.

## Files

- `identity/findings/dedupe_predicate_scorecard.results.csv` — the 13-row
  measured grid (stack × X).
- `identity/findings/dedupe_predicate_scorecard.summary.json` — machine
  summary.
- `identity/findings/DEDUPE_BASELINE.md` — the measured baseline surface
  with exact call-chain line ids.
