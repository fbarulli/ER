# Remaining identity cases — investigation log

Started 2026-10-05. Status: reproducing the scorecard and enumerating cases.

Scope: investigate every residual pair from the existing identity scorecard,
including different-GTIN pairs without descriptor conflicts and same-GTIN
pairs blocked by descriptors. These are two different failure modes; the
scorecard does not identify an unmatched-inference output file.

For each case, retain both source rows, compare same-GTIN siblings and similar
products, record missing or contradictory dimensions, and specify a supported
disposition. Missing evidence is an unresolved case, never proof of a match.
Different checksum-valid GTINs are distinct catalog identifiers, but validity
alone does not prove that a retailer assigned the identifier correctly.

The user subsequently authorized evidence-backed fixes as well as investigation.
Existing unrelated working-tree changes are preserved.

GitHub tracking: https://github.com/fbarulli/ER/issues/4

## Progress

- Read the existing baseline and scorecard findings.
- Located the original reproducible scorecard in the identity-evidence-union worktree.
- Next: reproduce the exact pair supply, enumerate residuals, inspect each pair
  against source declarations and sibling evidence, and publish incremental results.

## Census checkpoint

- All 261 baseline different-GTIN residual pairs have individual evidence cards.
- All 795 baseline same-GTIN descriptor-conflict pairs also have evidence cards.
- Actual production decisions are 261 `different` and 795 `same`; the descriptor
  scorecard alone must not be called production matching recall.
- Same-GTIN closure suggests additional conflicts in 50 of the 261 negative
  residuals. These remain suggestions: sibling feeds also contain bad packaging,
  sweetener and volume declarations.
- First census included six extra blank-title negative pairs (none residual).
  Reproduction now explicitly excludes blank titles as the original groupby did.
- A serious additional finding: some same-GTIN "positives" explicitly advertise
  different bundle counts. A same identifier may describe an inner item, not the
  complete offer. Those labels require scope review rather than blanket matching.

## Source-confirmed fixes being prepared

- ESN Sour Cherry `4250519666563` and Lemon Ice Tea `4250519662558`:
  source pages explicitly bind flavor and **65 ml** to the EAN; source corpus
  rows 927354566 and 927640172 instead say 1000 ml and omit flavor.
  [Cherry](https://www.shop-apotheke.com/fitness/upmED42K2/esn-ultra-vitamin-syrup-sour-cherry.htm),
  [Lemon](https://www.shop-apotheke.com/fitness/upmX4WEXF/esn-ultra-vitamin-syrup-lemon-ice-tea.htm).
- Eska `671785201960`: Metro identifies **6 × 1.5 L**. The sparse row
  924310310 omits volume and count; same-GTIN rows 114732827 and 998232612
  corroborate 1500 ml. [Metro](https://www.metro.ca/en/online-grocery/aisles/beverages/water/bottled-water/natural-spring-water/p/671785201960).
- Sokos "add to wish list" is scraped interface text, not a product name.
  Its URLs and same-GTIN siblings identify four distinct Puhdistamo flavors.
- Different net contents require distinct retail identity; do not repair this
  by merging related formats. [GS1 net-content rule](https://www.gs1.org/1/gtinrules/en/rule/266/declared-net-content).
