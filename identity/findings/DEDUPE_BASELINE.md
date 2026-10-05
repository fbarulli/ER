# DEDUPE_BASELINE — exact measured evidence surface of the current predicate

Worktree: `/tmp/opencode/er-identity` @ `416b5ca` (branch identity-evidence-union).
Scope: which **columns** feed the product-identity decision in
`src/training/dedupe.py::_same_product_by_title` →
`src/core/sku_identity.py::{row_identity, evaluate_sku_identity, identity_conflict}`.

## Call chain (exact lines)

1. `src/training/dedupe.py:101-129` — `_same_product_by_title(sub, retailer, gtin)`
   - `dedupe.py:119-121` config adjudicated overrides win.
   - `dedupe.py:123` — `identities = [row_identity(row) for row in sub.to_dict("records")]`
   - `dedupe.py:127-129` — collapse is authorized iff EVERY pair of rows in the
     group returns `evaluate_sku_identity(left, right)["decision"] in {"same", "compatible_unverified"}`.

2. `src/core/sku_identity.py:378-450` — `row_identity(row)` consumes,
   via `get(...)` reads (Mapping access on a `dict` built from the CSV row):
   - `gtin`            → `sku_identity.py:417-418` → `_gtin_facts` (`:367-375`, trusted-gate w/ held keys)
   - `sku_name_eng`    → title (`:391`), fed to `sku_info` (`:402-407`), flavor title-tokens (`:409-414`)
   - `attribute`       → `:392`, fed to `sku_info`, `identity_tokens_set` (`:408`)
   - `description_short_eng` → `:393`, fed to `sku_info` (`:402-407`) — **already consumed
     by the CURRENT surface** (so S1's "description" is not new; the test is which columns
     `row_identity` DEPENDS on for conflict detection).
   - `sku_url`         → `:394`, fed to `sku_info` (URL slug tokens for volume/pack/sweetener)
   - `image_url`       → `:395`, fed to `sku_info` (image filename tokens, volume/pack/sweetener)
   - `category`, `breadcrumbs_eng` → `:396-398`, fed to `sku_info` (type/flavor hints)
   - `brand`           → `:423` (`normalize_brand`)
   - `sku_id`          → `:417` (`listing_review_reason`) — review holds only
   - `completeness` (`:446`) reads `DESCRIPTOR_COLUMNS` (config/paths.yaml:260-266) =
     `sku_name_eng, brand, category, breadcrumbs_eng, attribute, description_short_eng`.

## What the DECISION actually compares (`identity_conflict`, `:479-519`)

Compared dimensions (rule-4 gtin short-circuit first, `:493-494`):
`brand` (`:497`), `volume` (`:499-503`), `pack` (`:504-507`),
`flavor, carbonation, sweetener, sweetener_type, sweetening, pulp, package_type, package_material`
(`:508-512`), `diet_claim` x `sugar_claim` (`:514-518`).

So `row_identity` reads 12 columns, but the COMPARISONS that can fire a vet/
merge decision use only these descriptor dimensions extracted from:
`sku_name_eng, attribute, description_short_eng (+ url/image-url-derived volume only),
breadcrumbs_eng/category (flavor/type), brand`, plus the(gtin, sku_id) policy keys.

## The residual is ABSENCE, not noise (recorded in code comments)

- `src/training/dedupe.py:93-98`: "measured on gtin-labeled ground truth
  (2026-09-30) the descriptor predicate alone merges 59.0% of
  provably-different pairs, because **a missing descriptor reads as agreement**."
- `src/training/dedupe.py:110-113`: "the absence of a conflict is not evidence
  of identity ... 59.0% false-merge rate on gtin-labeled hard negatives".
- `src/core/sku_identity.py:26-31`: historic 29.3% compatible-rate on GT-NEG
  when one side's field was empty.

## Attribution note on the 59% figure (measurement fidelity)

The 59.0% number predates the current HEAD (`dedupe.py:93-98` mentions it as
already measured when GT-NEG was 13,517 pairs = *cross*-retailer same-gtin
negatives in `sku_identity.py:8-11`). The reconstruction here uses the
available pair supply under gtin-validity + duplicate-title constraints
(structured S0-S3 evaluation in `scripts/dedupe_predicate_scorecard.py`);
any reproduction of exactly 59% is luck — the figure is an owner-recorded
measurement, not a head-checkout label.

## Consumed columns summary table

| column          | evidence fate                | fires vetoes? | fires positive merges? |
|-----------------|------------------------------|---------------|------------------------|
| gtin            | trusted identity `:493-494`  | yes (rule 4)  | yes (same gtin)        |
| sku_name_eng    | title tokens → all dims      | yes           | no (no positive merge) |
| attribute       | attribute cell → all dims    | yes           | no                     |
| description_short_eng | `:402-407` sku_info     | yes (sweetener…) | no                  |
| sku_url         | `:394` volume/pack/sweetener | yes (volume/pack) | no                |
| image_url       | `:395` volume/pack/sweetener | yes (volume/pack) | no                |
| breadcrumbs_eng + category | `:396-398` flavor/type | yes     | no                    |
| brand           | `:423` brand                 | yes           | no                     |
| sku_id          | `:417` review holds only     | holds         | no                     |
| (NOT read) any other column | ignored by the decision | -    | -                      |

Note: T5 / vetoes / GT-NEG labeling uses this SAME surface, so the
"cleaned typed surface" is exactly what `row_identity` produces, and the
owner question ("was this measured on the cleaned surface?") is answered
YES in `dedupe.py:93-98` + `sku_identity.py:6-11`.

## Measured reconstruction (this worktree, S0 in the scorecard)

Protocol (scripts/dedupe_predicate_scorecard.py): labeled pairs are built ONLY
from gtin proofs, both sides passing `core.gtin.gtin_validity` minus the
per-sku reviewed-row mask (`core.identity_policy.reviewed_row_mask`) --
* label 0: same (retailer, title) carrying DIFFERENT canonical (zfill-14) gtins
  -> provably different products; 1,044 pairs from 691 multi-gtin title groups;
* label 1: duplicate rows of the SAME canonical gtin (within- and
  cross-retailer families) -> provably same product; 4,000 pairs from 6,821
  families (budget cap 4,000; deterministic `np.linspace` down-sampling,
  seed SSOT = core.common.SEED = 42).

The text surface is measured with the rule-4 gtin lane DISABLED
(`_drop_gtin_trust`) so the decision is the descriptor predicate alone,
exactly the surface the 59% comment describes.

Measured at HEAD:

| metric | S0 (dedupe.py surface) |
|---|---|
| label-0 false-merge (no conflict = would merge) | **0.2500** (25.0%) |
| label-1 true-merge | 0.8013 |
| conflict-dims firing on label-0 | volume 599, package_material 307, brand 188, package_type 139, carbonation 24, sweetener 9, pulp 3, pack 2, flavor 2 |

Reconciliation with the recorded 59.0%: the owner figure was measured on the
2026-09-30 GT-NEG structure (13,517 within-retailer different-gtin pairs =
`sku_identity.py:10`); at HEAD the exploitable gtin-labeled supply is smaller
(2026-10-05 corpus state: 1,273 title-dup groups -> 1,044 chain-pairs after
dedupe/held-key exclusions) and the extraction fixes flown since (brand alias
seeding, packaged claims) already cut some absence residual, so 25.0% is the
HEAD's honest residual — SAME CLASS of failure (false merges from descriptor
absence; no conflict when a field is empty on one side, `sku_identity.py:26-31`
rule 2), smaller magnitude. The 59% figure stays owner-recorded; this probe
does NOT re-measure or re-litigate it, it measures the current surface to
compare stacks against.
