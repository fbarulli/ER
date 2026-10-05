# Investigation checkpoint and next steps

Tracking: [investigation #4](https://github.com/fbarulli/ER/issues/4),
[bundle scope #5](https://github.com/fbarulli/ER/issues/5).
Branch: `identity-residual-investigation`. Updated 2026-10-05.

## Results already obtained

- Frozen evidence cards for 1,056 residual pairs: 261 different-GTIN pairs
  without descriptor conflicts and 795 same-GTIN pairs with descriptor conflicts.
- 22 listing+GTIN scoped source repairs, with individual evidence and peers in
  `REVIEWED_REPAIRS.md`.
- 34 bundle-identifier families held for scope review, covering 364 retained
  source rows. The frozen cohort contains 56 affected positive pairs.
- Fixed title/URL pack precedence, volume-shaped pack counts, slug decimals,
  dilution/count separators and case/size fractions.
- Found 959 untrusted-title groups / 3,007 rows / 3,302 conflicting pairs at risk
  of title-only collapse; added a descriptor guard before T2/T3. This is a raw
  risk census, not a measured end-to-end loss count.
- First frozen replay: 13 negative residuals acquire discriminating descriptors;
  10 same-GTIN descriptor conflicts disappear; 56 apparent positives become
  review cases. Later quantity fixes still require replay.

## Active validation priorities, largest first

1. Packaging material: 332 same-GTIN pair conflicts, including 104 glass/plastic,
   78 metal/plastic and 32 paper/carton versus metal. Check source material and
   inner/outer packaging before promoting any normalization rule.
2. Carbonation: 184 carbonated/still conflicts. Compare explicit source prose,
   ingredient declarations and same-GTIN peers; do not vote by duplicated rows.
3. Apparent strong agreement: 33 of the 261 negative residuals share image URLs;
   12 share listing URLs. Validate variant-selector and shared-image failures.
4. Largest untrusted-title groups: Allyouneedfresh true fruits (22 rows, 160
   conflicting pairs), Chronodrive natural mineral water (14/83), Walmart DECAF
   Cold Brew (17/82). Retain distinct evidence until source-backed repair.
5. Replay the fixed, original cohort after each repair batch. Keep repaired,
   held and unresolved cases in the denominator and record any regressions.

## Remaining limitations

Case cards are exhaustive for the scorecard residual cohort; they are not a
claim that every linked page has been manually verified. Every unresolved case
remains explicitly unresolved. This does not yet measure inference match recall.
Prepared/training artifacts have not been rebuilt; future runs must use the
updated policy and extraction code. Existing unrelated working-tree edits are
not included in this branch.
