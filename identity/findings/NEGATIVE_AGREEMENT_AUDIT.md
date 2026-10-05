# Audit of the "apparent strong agreement" negatives (different-GTIN keep-distinct pairs)

Cohort (NEXT_STEPS item 4, now validated): of the 261 different-GTIN residual
negatives, **37 pairs share the same title AND (33 pairs) the same image URL or
(14 pairs) the same listing URL** — pairs the production keeps distinct on GTIN
alone (`keep_distinct_gtins`). Machines: `scripts/negative_missing_probe.py` +
`negative_missing_probe.json`. Every flagged field was then verified by hand in all
13 original columns (`sku_id, retailer, country, sku_name_eng, description_short_eng,
breadcrumbs_eng, sku_url, image_url, sku_last_price, gtin, brand, category, attribute`).

## 1. Feature inventory — what the pairs have and do not have

- **340 descriptor cells are "missing" across the 37 pairs** (mostly the sweetener
  family, pulp, package_material/type).
- **282 of them are absent in EVERY original column** — there is nothing to extract.
  Not gaps. (Examples: Weider/peeroton/ironmaxx/ESN/got7 Sportnahrung+Vitalabo
  listings that simply never name the flavour anywhere; Carethy/NAYA pricing pages.)
- **58 carry raw-text signal**; hand-verdicted below — only 3 are true gaps.

## 2. Manual verdicts on the 58 signals

| Class | Cases | Verdict |
|---|---|---|
| **Real extraction gap — value in an evidence column, not extracted** | R0005 L+R, R0014 L+R, R0071 R (package_material/type) | `R0005` description: "pack of four 9.5 oz. **glass** bottles"; `R0014` title: "(12 **Glass** Bottles)"; `R0071` right description: "5 oz in **glass** bottle" |
| Weight masquerading as volume (probe hit, correctly NOT extracted) | R0003 ("17.6 OZ" powder tea mix), R0121 ("600 gr" powder), R0097 ("15kcal to 0.5 liters" = mixing ratio, not package size) | absent — right |
| Claim text present but typed to another dimension per design | R0018 (attribute "**Sweetener:** cane sugar, sucralose" → sweetener_type is populated; claim-class sweetener stays empty), R0065/R0176/R0177 (only "Health Claims: **no added sugar**" — no sugar TYPE declared), R0100-105 (attribute Sweetener: sucralose/fructose/acesulfame → types populated; no sugar claim → class empty), R0174 ("Sweetener: cane sugar"), R0175/R0129 (Pack **Material** field present, Pack **Type** field genuinely absent) | absent — right |
| Substring noise in the probe ("Me**xi**can", "carnitin**e**", bread crumbs "Bottled Beverages", category "Functional Bottled Water") | R0201, R0218-desc, R0030/R0038/R0055/R0192 breadcrumb/category hits | absent — right |
| Genuinely unnamed variant | R0210 (ESN Flavor Drops: no Flavour key anywhere in the listing) | absent — right |

## 3. Pair verdicts (keep-distinct correct or not?)

**A. Suspected duplicate listings with pseudo/incoherent GTINs — TRUE matches
production may be over-splitting (`keep_distinct` questionable):**

- Walmart `classType=VARIANT` pairs with IDENTICAL titles, descriptions, images and
  brand but different item IDs and different GTINs whose GS1 prefixes do not match a
  US brand: R0001 (Fruit2O, GTINs 646…/613…), R0003 (Cha Sen tea, 604…×2), R0011 (Fruit2O chain pair), and
  REGULAR-class twins R0005/R0008/R0012/R0014 (Starbucks 100…, Stumptown
  615…/673…, Blenheim 758…/747…). Prices differ — the two pages are live separately.
- Amazon duplicate-ASIN/retail pairs with the same structure: R0030 (De mi Pais,
  688…/896…), R0018 (Stirrings, ASINs B0F2MFDG7Q vs B0D3MXK73V), R0055 (Bare Nature
  iced tea, prefix 732…/743…), NAYA water R0211 (769… prefix), Carethy money-scale
  odd pair R0121 (prices 826 vs 7.04 for "600 gr" powder — same page duplicated).

  Local evidence can take these to "same listing scraped under two fake-looking
  GTINs" but cannot prove what the GS1 registry says; registry lookup per pair is
  the missing validation step.

**B. Genuine sibling variants that share a generic family image/title —
keep-distinct CORRECT (the extraction gaps only mask the difference):**

- R0218 Harmless Harvest: left description "8.75 fl oz **259 ml**" vs right
  "16 fl oz (**473 ml**)" — different sizes, provably distinct once description is
  consulted (today distinct by GTIN alone; if description were an identity surface,
  volumes_compatible would mint the conflict directly).
- R0038 Brecon Carreg still water (left 2 L vs right unknown size), R0127 alnavit
  smoothie (cherry-pear vs apple-cherry-grape-rhubarb — attribute-named), R0129
  Amecke (apple vs apple-grapefruit), R0176/R0177 ékolo juices (grape vs kiwi-apple),
  R0175 vs R0178 226ers isotonic (berries vs lemon), the Sportnahrung generic-title
  siblings R0096/R0097/R0098/R0099/R0100-105/R0102/R0110/R0210 (L-carnitine waters,
  peeroton, Power Kick, ironmaxx, GOT7 syrup, ESN drops — German/Austrian EANs,
  distinct SKUs of unnamed flavours).

## 4. Actions decided

1. Keep the 261 negatives distinct (unchanged) until per-pair registry checks.
2. The three material/type gaps above are worth a description-surface extension
   experiment in a future run; they do NOT change any current verdict.
3. Item 4 of NEXT_STEPS is validated as: no false negative lurking in this cohort —
   either wrong-keep pairs with pseudo GTINs (need registry), or real siblings.
