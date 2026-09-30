# Product identity discovery, 2026-09-30

These findings concern identity and splitting. Runtime guards now exclude reviewed
identities from labeling and split inputs; source listings are preserved.
The dashboard at http://127.0.0.1:8001 is the browsable evidence surface.

## 01 — same barcode, inconsistent dimensions

In the capped cross-retailer sample of 19,123 same-valid-GTIN pairs, 5,657
(29.6%) have at least one disjoint raw dimension. These involve 2,343 GTINs
and 6,973 deduped listings. All 37 observed dimensions are registered; missing
values remain unknown. This is a sample of metadata disagreement, not a rate
of identity errors in the full catalog.

Source: `dashboard/evidence/identity/01_identity_discovery_*.json`.

## 02 — identical titles, different sellable units

Kroger repeats “Brew Dr. Kombucha Organic Clear Mind Kombucha” for
851107003001 (414 ml glass bottle) and 851107003872 (355 ml metal cans).
Target and Whole Foods explicitly name the latter as a four-pack. The original
columns support separating those sellable variants within one flavor family.
An exact brand/title match is insufficient to collapse them or treat them as
interchangeable positive pairs.

Source: `02_exact_title_variants_original_rows.json` in dashboard/evidence/identity.

## 03 — missing measurement and packaging context

A read-only audit of all 71,623 original listings found:

- 27,175 listings with a Caffeine attribute; the raw attribute schema has mg
  but no denominator field.
- 905 listings with a recognized numeric caffeine claim in title/description;
  304 have a nearby serving/container basis cue. These are heuristic selectors,
  not verified conversions.
- 15,812 titles with a recognized multipack cue; 15,429 of those lack
  Count per Unit. A cue does not establish a reliable extracted pack count.
- 121 listings combining Can with selected nonmetal material values. Some
  can legitimately describe an outer package or a powder canister.
- 71 checksum-valid GTIN groups (341 source listings) with multiple normalized,
  nonempty brand names. These require review before adding brand aliases.

For 3D blue energy, descriptions state 200 mg caffeine per 473 ml can;
200/473*100 ≈ 42.3 mg per 100 ml. A 25–50 mg attribute could therefore use a
per-100-ml basis. The feed does not state that basis, so this remains a possible
explanation rather than a correction.

Kevyt Olo's 12-pack image shows metal cans inside shrinkwrap. Inner container
and outer wrapping can coexist; mineral and flavoured describe different
properties of water. Olvi versus Kevyt Olo needs brand hierarchy evidence,
not a global fuzzy alias. Halfday's metal-can image instead gives specific
support against Paper/Carton as the inner container material.

Reproduce: `PYTHONPATH=src .venv/bin/python scripts/audit_identity_context.py`.

## Implications for identity and splits

Keep raw evidence and its source. A future resolved evidence record should
carry quantity, unit, measurement basis, packaging level, count, and provenance
separately. Unknown scope must remain unknown; disagreement alone must not
create a negative pair. Reviewed variants can remain distinct SKUs while sharing
a broader product family. Split construction should explicitly choose whether
that broader family must stay together to avoid variant leakage.

The shared evaluator remains `core.product_identity`; all raw-dimension parsing
and comparison remains `core.product_dimensions`. The dashboard uses those
shared dimension routines and `core.common.COLUMN_MAPPING`, not another matcher.


## Implemented resolutions

`core.product_context` stores linked caffeine quantities and their basis, unit
volume, pack count, inner packaging, outer packaging, and unscoped raw claims.
It reuses the existing title quantity parsers and shared unit conversion. A
serving without a size remains unknown. Raw caffeine ranges are never assigned
a denominator by guesswork. The 3D Harris Teeter and Hy-Vee descriptions both
resolve to 200 mg per 473 ml, so their contextual comparison agrees despite
100–150 versus 25–50 mg raw ranges. The raw discrepancy remains auditable.

Reviewed retailer image evidence corrects Halfday SKU 897369430's inner material
to metal and separates Kevyt Olo SKU 282677444's cans from outer shrinkwrap.
Overrides require the original listing ID and matching GTIN; they are not brand-
or barcode-wide inferred corrections.

`config/identity_reviews.json`, validated by `core.identity_policy`, holds 21
explicit GLNs misfiled as GTINs and the two unresolved Brew Dr groups. All 23
identifiers are blocked from identity trust and all labeling/split inputs. This
covers 109 original rows and 30 deduped listings. Source records remain intact.
The existing canonical artifact lost 23 rows; gate results lost 338 affected
pairs; the labeled census lost 5 affected pairs. Final validation was then
rebuilt from the filtered merged component graph: 6,322 pairs (596 positives
and 5,726 negatives), zero quarantined endpoints and zero straddling positives.
The split graph rejects stale quarantined entities, and graph preparation
rejects such catalogs before writing inputs. Unknown true item GTINs remain
unknown; no replacement barcodes have been invented.

Discovery can continue with open findings. The open Cool Best mixed-flavor
source cases in finding 06 have not been adjudicated by this resolution.

## 07–08 — repair old collapses and reattach reviewed duplicates

The original Mat Smart number `11982760` mixed Maxim BCAA and Löfbergs
Protein/Caffeine Boost. The Cortas number `735143004010` mixed flavors, unit
volumes and packs; UK Amazon descriptions repeated it as an item model number.
Both claims are now held. The source export remains unchanged.

Mat Smart source listings 142547188 and 143441192 match source reference
140880643 on exact product URL, retailer, brand, title, description, dimensions
and 12 × 230 ml pack. The reviewed policy reattaches those two listings to
7310050105482, with representative 140880643. This link requires original
listing ID, source barcode and exact URL; it is not a broad substitution of
11982760, nor a mapping from a 12-pack to a single drink.

A targeted catalog repair restores original listings previously merged on all
25 held identifiers, while preserving existing unrelated representatives.
The catalog grows from 62,963 to 63,053 records. All 71,623 original source IDs
still resolve to valid representative positions; 123 catalog rows remain held
and 62,930 remain eligible. Two reviewed duplicate aliases share their correct
representative. The 12 Cortas suspect source listings remain inspectable,
without invented replacement GTINs. Separately numbered 500 ml rose and orange
blossom references remain separate sellable identities.

Future dedupe applies reviewed links first and puts held records into distinct
listing partitions so they cannot be merged merely on their bad identifier or
same title. Reproduce the targeted repair with
`PYTHONPATH=src .venv/bin/python scripts/repair_reviewed_catalog.py --apply`.
Findings 07 and 08 show the before/after comparisons.
