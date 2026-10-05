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

## 5. Local checks run (all of them; no network)

`scripts/negative_local_checks.py` → `negative_local_checks.json`. For every pair:
corpus-wide **brand-prefix family scan** (all other GTINs of the same brand cell →
2-3-digit GS1 prefix census, pair GTINs classified in/out of family), price/desc/
country forensics, and URL-slug archaeology.

**Pair prefix verdict (canonical-13 first-3):**

- **Outlier GTINs on both sides — 14 pseudo-suspicion pairs, confirmed locally,
  independent of any registry:**
  R0001+R0011 Fruit2O (pair 604/613/646 vs sparse family),
  R0003 Sen Cha (604×2 vs family 081), R0008+R0012 Stumptown (613/673 vs family
  085...), R0014 Blenheim (754/743 vs 084...), R0018 Stirrings (0810… vs family
  078...), R0030 De mi Pais (688 vs family 089...), R0055 Bare Nature
  (613/073 vs family 085...), R0071 Fee Brothers (613/079), R0121 BioTech USA
  (085…/400… vs family 599…, countries Belgium/UK, prices 826 vs 7.04 — same
  Carethy path p-471012 twice), R0192 Holoslife (063/064, UK/Netherlands, same
  slug p-407149 twice), R0201 Weider (085/599 vs family 404…, same slug twice),
  R0211 NAYA (076×2 vs family 063…).
- **Registry-consistent prefixes (keep-distinct stands, 23 pairs):** the entire
  German/Austrian/Spanish sibling blocks — Weider/Multipower/IronMaxx/GOT7/ESN/
  peeroton/Alnavit/Amecke (400/404/425/426/900 families), ékolo/Kombutxa/226ers
  (843), Brecon Carreg (503), plus Starbucks Frappuccino R0005 (010×2 both sides).

**URL archaeology:**
- R0030 + R0211 = the SAME Amazon search query (`keywords=` and `ts_id` identical)
  scraped at two result positions (`sr=1-411` vs `sr=1-588`) — one listing, two
  pseudo-GTINs. R0030's right GTIN matches its brand family; the left does not.
- R0121/R0192/R0201 = same Carethy page slug duplicated in two countries, each
  carrying a different chosen GTIN — cross-country duplicate listings.
- R0018/R0071 = two different Amazon ASINs with byte-identical marketing text.

**Net upgrade of §3 verdicts:** of the 37 identical-name pairs, the brand-prefix
scan classifies **14 three-channel pseudo-suspicion pairs** (Walmart
classType=VARIANT chain, Amazon ASIN twins, Carethy cross-country slug twins,
NAYA amazon search twins) backed locally by prefix-family outliers and
duplicate-page forensics; 23 pairs are registry-consistent real sibling blocks.
Borderline within the registry-consistent set: R0005 Starbucks (legit 010… UPCs
both sides, byte-identical descriptions but 7.47 vs 11.99 prices — a possible
legal twin-barcode/duplicate offer) and R0096/R0098/R0099 same-price German
sibling EANs — indistinguishable locally; registry only. This is the maximum
decidable LOCALLY; GS1 registry per-GTIN and live-page fetches remain the only
unresolved step for the 14 pseudo-suspicion pairs. Recommendation stands:
keep-distinct until registered.

## 4. Actions decided

1. Keep the 261 negatives distinct (unchanged) until per-pair registry checks.
2. The three material/type gaps in §2 are worth a description-surface extension
   experiment in a future run; they do NOT change any current verdict.
3. Local checks (§5) cannot decide the 14 pseudo-suspicion pairs; GS1-registry
   and live-page checks are next (network). Item 4 of NEXT_STEPS is validated
   as: no false negative can be minted locally — pairs are either real
   sibling variants (keep correct) or duplicate pseudo-GTIN channels
   (registry check pending).
