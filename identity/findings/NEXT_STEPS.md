# Investigation checkpoint and next steps

Tracking: [investigation #4](https://github.com/fbarulli/ER/issues/4),
[bundle scope #5](https://github.com/fbarulli/ER/issues/5).
Branches: `identity-residual-investigation`, then `identity-regex-repairs`
(corpus-wide bundle census + regex fixes). Updated 2026-10-05.

## Results already obtained

- Frozen evidence cards for 1,056 residual pairs: 261 different-GTIN pairs
  without descriptor conflicts and 795 same-GTIN pairs with descriptor conflicts.
- 22 listing+GTIN scoped source repairs, with individual evidence and peers in
  `REVIEWED_REPAIRS.md`; field repairs are bound to the expected GTIN and the
  exact source URL in `core.identity_policy`.
- 34 bundle-identifier families held for scope review from the frozen cohort,
  plus a corpus-wide census (`CORPUS_BUNDLE_SCOPE.md`) that found 80 MORE
  unheld families / 471 rows — the biggest same-GTIN merge families in the
  corpus. All 80 are held now; the census re-run finds 0 open families.
- Regex fixes: GDSN weight prose ("gross weight: 527 unit (specific)") is no
  longer read as a retail pack; `case of N` is a declared bundle in the
  blocking-key extractor.
- Fixed title/URL pack precedence, volume-shaped pack counts, slug decimals,
  dilution/count separators and case/size fractions.
- Found 959 untrusted-title groups / 3,007 rows / 3,302 conflicting pairs at risk
  of title-only collapse; added a descriptor guard before T2/T3. This is a raw
  risk census, not a measured end-to-end loss count.
- Frozen replay re-measured after the census holds: 86 scope holds (was 56),
  729→695 feed-conflict retentions, 10→15 repaired descriptor matches; no
  previously-resolved case regressed.

## Active validation priorities, largest first

1. Feed attribute errors on retained biggest merges (true merges, wrong
   attribute): NOS carbonation "still" (Harris Teeter), Lemon Perfect
   sweetener "sugar" on zero-sugar products, Harmless Harvest / Alpro Caffè
   plastic-vs-carton materials. Source-check the single dissenting feed and
   promote listing+GTIN scoped repairs in `REVIEWED_REPAIRS.md`.
2. Packaging material: 332 same-GTIN pair conflicts, including 104 glass/plastic,
   78 metal/plastic and 32 paper/carton versus metal. Check source material and
   inner/outer packaging before promoting any normalization rule.
3. Carbonation: 184 carbonated/still conflicts. Compare explicit source prose,
   ingredient declarations and same-GTIN peers; do not vote by duplicated rows.
4. Apparent strong agreement: 33 of the 261 negative residuals share image URLs;
   12 share listing URLs. Validate variant-selector and shared-image failures.
5. Largest untrusted-title groups: Allyouneedfresh true fruits (22 rows, 160
   conflicting pairs), Chronodrive natural mineral water (14/83), Walmart DECAF
   Cold Brew (17/82). Retain distinct evidence until source-backed repair.
6. Replay the fixed, original cohort after each repair batch. Keep repaired,
   held and unresolved cases in the denominator and record any regressions.

## Remaining limitations

Case cards are exhaustive for the scorecard residual cohort; they are not a
claim that every linked page has been manually verified. Every unresolved case
remains explicitly unresolved. This does not yet measure inference match recall.
Prepared/training artifacts have not been rebuilt; future runs must use the
updated policy and extraction code. Existing unrelated working-tree edits are
not included in this branch.
