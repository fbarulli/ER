# Run findings — the 3k cohort is not a viable validated-tracks cohort

Evidence record for the attempted 3k all-tracks preparation (Colab CPU bundle
lane, `er-colab --what bundle --gpu CPU --dataset-csv dataset_3k.csv`). Every
number is read back from the run's own output; nothing is refitted.

## Verdict

**3k cannot produce a validated tracks suite. Do not re-decide the
`negative_fold_policy` config for it.** The full cohort is the runnable path.

## Cohort sizes

| cohort | file | bytes | wc-lines | note |
|---|---|---|---|---|
| 3k | `dataset_3k.csv` | 2,188,027 | 7,750 | official 3k export |
| full | `dataset.csv` | 54,787,809 | 198,870 | wc counts embedded newlines; runbook row count is 71,623 |

## What the 3k cohort assembles to

- positive pairs: **985** (981 train + 4 validation) over **1,105** entities.
- graph entities: **3,755**.
- negative supply: **hard = 8**, **targeted_attribute = 16**, **cross_brand = 2,738**.

The hard-negative pool of 8 is the cause: the scored dev/test halves contain
effectively no negatives.

## Where it fails

`training.prepare_all` completes stages `dedupe … discriminator`, then dies at
stage `validation`:

```
subprocess.CalledProcessError: ['python','-m','training.build_final_validation'] returned non-zero exit status 1
SystemExit: the pinned scored-half decision no longer holds: policy 'train_side'
does not score MORE negatives per fold than 'withhold_straddle' (...)
```

Location: `src/training/build_final_validation.py:1289` → `LeakGuards.assert_pinned_evidence`
(:942) → `assert_pinned_evidence_train_side` (:956/:984) raise at :994.
Config pin: `config/training.yaml:726` `split.negative_fold_policy: "train_side"`
(floor `min_test_negatives: 5`, `:809`).

This is a config/policy DECISION guard (re-measured at every emit, mandated by
AGENTS.md), not a banned data-integrity gate. It STAYS.

## The evidence, both policies (exact `NegativeFoldPolicy.evidence` output, n_folds=4)

| field | `withhold_straddle` | `train_side` |
|---|---|---|
| scored_dev_negatives | 0 | 0 |
| scored_test_negatives | 0 | 1 |
| scored_negatives_with_trained_on_endpoint | 0 | 0 |
| scored_dev_positives / scored_test_positives | 0 / 0 | 0 / 0 |
| negatives_withheld_from_scored_half | 3 | 2 |
| thin-cell share dev / test | nan / nan | nan / 1.0 |
| populated_cells dev | all 0 | all 0 |
| populated_cells test | all 0 | volume 1, pack 1, package_type 1, sweetener 1, flavor 2, carbonation 1 |

Failed criterion: **(a)** `train_side` must score MORE negatives per fold than
`withhold` — precisely the **dev** half (test 1 > 0 ok, dev 0 > 0 fails).
**(c)** trained-on-endpoint passes (0 both). **(b)** thin-share also fails if
reached (nan comparisons).

## Which policy the 3k evidence supports

**Neither.** `train_side` fails (a) and (b). `withhold_straddle` fails its own
guard `assert_pinned_evidence_withhold` (both halves must score ≥ 1 negative;
3k gives dev 0 and test 0 → `empty_halves=[dev,test]`). No policy passes
cleanly — 3k simply lacks scored-half negatives.
