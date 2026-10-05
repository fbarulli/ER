# Corpus-wide bundle-scope census — 2026-10-05

Branch: `identity-regex-repairs`. Scripts (reproducible, in order):
`scripts/census_bundle_scope.py` → `scripts/apply_bundle_scope_holds.py` →
`scripts/replay_identity_residuals.py`.

## What changed and why

The previous bundle-scope inventory (34 GTINs) was derived from the FROZEN
residual cohort only — the 56 positive pairs held there. The biggest
same-GTIN merge families in the corpus were never observed by that cohort:
Cocofina coconut water 1 l (`05060118260203`, 44 rows, 652 pack-conflicting
pairs), Equinox Kombucha (`05060452360065`, 28/313), Actiph Water
(`05060477670002`, 25/202), Aqua Coco (`05060468010510`, 21/153), What A
Melon 330 ml (`05060167500169`, 19/94) and 1 l (`05060167500220`, 12/53),
duskin Golden Delicious (`05060128500405`, 13/46), Hi Ball
(`00897351000427`, 11/6), Cherry Bay (`00853955000904`, 10/3), and more.
Every one of them advertises different explicit retail bundle counts under
one GTIN — the same documented identifier-scope situation ("shared GTIN may
identify an inner item; it does not prove complete-offer identity") — but
their GTINs kept full identity authority and positive-label eligibility.

## Census result (before holds)

- eligible rows 25,845 / eligible GTINs 13,182 (gtin-validity minus reviewed
  rows, dataset sha `539c2472…312fab88c`);
- 80 unheld families / 471 rows with disjoint title-advertised pack counts
  (one pair each minimum, 652 pairs at the top);
- after applying the holds: **0 open families remain** — the census is
  exhausted against the current corpus.

The full per-family evidence (rows, retailers, titles, source URLs, sample
collision pairs, conflict dimensions) is frozen in
`corpus_bundle_scope.json` at census time and in `bundle_scope_holds.json`
+ `BUNDLE_SCOPE_FINDINGS.md` after the holds were applied.

## Regex defects found by the biggest-merge investigation

1. **GDSN weight prose read as a pack count.**
   `pipeline.extract_pack_evidence` parsed `gross weight: 527 unit
   (specific)` as a bundle of 527 (89 GDSN-style descriptions in the corpus;
   66 produced bogus pack quantities; 2 more were in-range: `gross weight:
   95/96 unit (specific) … grams`). Poisoned rows could false-veto real
   matches (Lete `05060118260203` sibling "water 500 ml 24PCS") and corrupt
   blocking keys. Fix: a count immediately preceded by a weight label is
   skipped (pipeline.py `extract_pack_evidence`); 0 misfires remain.
2. **`case of N` was not a pack count in `core.text.PACK_COUNT_RE`.**
   The identity parser and the NER already handled `cases? of N`; the
   blocking-key extractor did not (`16oz Bottle ( Case of 12)` blocked under
   NO_PACK). Fix: added the `cases? of N` alternative; pinned in
   `tests/test_pack_count_config_bounds.py`.
3. Deliberately NOT changed: shipping-scale quantities ("19 Pallets of 84
   cases each = 1.596 cases, 38.304 bottles") stay valid unit counts —
   pinned by `tests/test_pack_quantity_boundaries.py`. Plausibility
   bounding is a hold-policy decision, not an extraction decision.

## Measured effect on the frozen replay (no resampling, all 1,056 cases kept)

| action | before | after |
|---|---|---|
| keep_distinct_need_variant_evidence | 248 | 247 |
| keep_distinct_repaired_descriptors | 13 | 13 |
| retain_match_review_remaining_feed_conflicts | 729 | 695 |
| hold_identifier_scope | 56 | 86 |
| retain_match_descriptors_repaired | 10 | 15 |

Decisions: different 261→260, same 739→710, review 56→86. The 30 new scope
holds come from the 24 census families overlapping the frozen cohort (some
families contribute several cohort pairs); 5 pairs moved from
"retain_match_review_remaining_feed_conflicts" to
"retain_match_descriptors_repaired" because the regex fixes above repaired
their descriptors. No previously-resolved case regressed.

## True vs false merges among the biggest merges

Verified per family from local evidence (sibling consensus across retailers;
see family sections in BUNDLE_SCOPE_FINDINGS.md):

- **Bundle-count collisions (FALSE merges under one identity, now held):**
  the 80 census families above. Same inner product is repeatedly re-listed
  with different pack counts / case-vs-inner retail semantics under one
  GTIN; merging them into one offer is wrong until scope adjudication.
- **True merges with feed attribute noise (merge retained):** e.g. NOS
  Energy `00815154020008` (one Harris Teeter row claims "still" while 10
  siblings say carbonated), Jones Green Apple `00620221200128` (Harris
  Teeter "still"), Zing Zang `00616003708777` (Whole Foods claims a sugar
  sweetener on a product whose own title says the mix has sugar; volumes
  1750/1751 ml are the same 1.75 L within tolerance — the 59.2 fl oz
  conversion buckets to 1751), Alpro Caffè `05411188128021`/`05411188129257`
  (Metal vs Paper/Carton carton declarations), Harmless Harvest
  `00859078002153`/`00859078002627` (plastic-bottle vs carton feeds),
  A SHOC/Accelerator `00810014530178`/`00810014530482`, Lemon Perfect
  `00850003748788`/`00850003748771` (stevia vs "sugar" on a zero-sugar
  product). These are genuine single-feed attribute errors; production
  already retains them as matches with review notes.
