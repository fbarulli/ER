# Structural difficulty census

Difficulty is a label-specific, pre-training proxy over exact frozen encoder inputs. It is not measured model error. GTINs, vendor names, and generation provenance do not determine difficulty.

Definition version 1: prose token-set Jaccard; low overlap <= 0.20, high overlap >= 0.50; at least two attributes observed on both sides. Positive conflicts, low overlap, or >=50% asymmetric evidence are hard. High-overlap positives without those conditions are easy. High-overlap negatives with <=1 attribute conflict are hard; low-overlap negatives or >=3 conflicts are easy. Remaining comparable pairs are medium; insufficient evidence is unknown. Positive conflicts require review and never change the label.

Coverage uses the existing 11-field masking registry. This is encoder-evidence coverage, separate from the 37-key raw-attribute census. Difficulty of a singleton catalog listing is undefined; catalog rows receive attribute coverage only.

## Full stored bundle

Source listings: 62,939. SHA-256: `119e8356738fd350ebefef2c2df615f759d34debe1cadebeb1f14749e6b22209`.

| Population | Pairs | Easy | Medium | Hard | Unknown |
|---|---:|---:|---:|---:|---:|
| original_positive | 24,361 | 24.1% | 60.6% | 14.9% | 0.5% |
| original_negative | 8,092 | 55.6% | 41.4% | 2.9% | 0.1% |
| augmented_positive | 33,251 | 14.5% | 48.0% | 33.1% | 4.4% |
| augmented_negative | 6,464 | 40.1% | 48.4% | 10.8% | 0.8% |
| consumed_positive | 5,532 | 24.7% | 58.8% | 15.8% | 0.7% |
| consumed_negative | 5,532 | 46.1% | 29.0% | 24.0% | 0.9% |

Stored-pair populations are unique within each label. Consumed pairs are weighted by frozen objective coordinates, not epoch presentation counts. Supply and objective populations must not be conflated.

## Coverage gaps

Juice-content evidence: 87.7% of catalog listings, 0% both-observed original positives and negatives. Pack evidence: 27.8% of catalog listings, 17.5% both-observed positives and 7.7% negatives. Pulp is sparse throughout (2.0% catalog; 2.4% positive pairs and 1.6% negatives both-observed). Missing evidence is never counted as agreement or conflict.

## Threshold sensitivity

| Setting (low/high overlap) | Original negative hard | Augmented negative hard |
|---|---:|---:|
| 0.15 / 0.60 | 0.9% | 4.1% |
| 0.20 / 0.50 | 2.9% | 10.8% |
| 0.25 / 0.40 | 7.1% | 19.9% |

Augmented negatives have more hard examples under all three settings. Absolute difficulty shares are threshold-sensitive; thresholds are provisional.

## Fresh coherent 3,000-row bundle

SHA-256: `13d762bd725b5c0497d2bb3cc82a7723af1c3e0ad16eb4b28fd3a0c49dbb97a7`.

113 of 1,754 augmented negative supply pairs are hard (6.4%), versus 11 of 4,868 original negatives (0.2%). Only 6 of 1,785 consumed negatives are hard (0.3%). The next allocation step needs to address objective survival as well as generation volume.

The full and 3,000-row bundles have different source populations and objectives; their difference is not a controlled before/after estimate of the code change.

## Vendor and original-data accounting

Every original positive in these bundles pairs a listing with a canonical endpoint; vendor relation is therefore unknown. Vendor-stratified augmentation needs real listing-to-listing variation evidence, not an invented vendor for canonicals.

- full consumed_positive: 58.9% untouched original pairs (the rest have at least one augmented endpoint).
- full consumed_negative: 36.9% untouched original pairs (the rest have at least one augmented endpoint).
- coherent_3000 consumed_positive: 67.5% untouched original pairs (the rest have at least one augmented endpoint).
- coherent_3000 consumed_negative: 65.0% untouched original pairs (the rest have at least one augmented endpoint).

These are frozen-objective shares; runtime masking can further change presentations. A desired original-presentation share still needs enforcement at the sampler/masking boundary.

## Artifacts

Each report directory contains `report.json` (Pydantic-validated complete counts), `attribute_coverage.csv`, and `strata.csv` (population x split x generation x edit mode x vendor relation x difficulty). Per-pair evidence remains local in `pairs.csv`; it can be regenerated from the pinned bundle with `scripts/measure_pair_difficulty.py`.
