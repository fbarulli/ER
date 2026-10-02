JEV gate audit, 2026-10-02

Round 1: sample_doubled.json / audit_results.jsonl.
Round 2: sample_doubled_2.json / audit_results_2.jsonl (55 pairs, both orders).

Round 2 completed 110 calls. Scores are JEV judgments, not ground-truth labels.

| Stratum | Calls | Mean same-product score | Scores below 0.2 |
| --- | ---: | ---: | ---: |
| proceed_survivor | 40 | 0.136 | 31 |
| hardno_newly_minted | 30 | 0.024 | 30 |
| fallback_newly_minted | 20 | 0.042 | 20 |
| hardno_kept | 20 | 0.017 | 20 |

The checkpoint supports withdrawing partial-overlap and semantic-family
rescues for flavor conflicts. Generic-only containment is inconclusive;
equal specific token bags match, specific containment remains a subset,
and divergent specifics conflict. Both flavor spellings use the same rule.
Exact/alias matches retain precedence. Non-flavor family rescues remain.

The low scores among surviving proceeds show remaining matching errors.
This stratified sample does not measure population accuracy or recall.
The additional changes after round 2 have offline regression coverage;
this checkpoint is not a live evaluation of those additional changes.

Resume round 2 without mixing checkpoints:

    python jev/run_audit.py --staging jev/sample_doubled_2.json --out jev/audit_results_2.jsonl

Completed ordered pairs are skipped. A custom staging path requires an
explicit output path. No new live JEV calls were made while applying the fix.

Validation: 113 passed, one failed across full attribute decisions, gate
logic/evidence precedence, replay, and targeted ANN gates. The failing
live-population regression assumes all 12,733 frozen proceed pairs remain
accepted and pins historical route counts. Current flavor policy rejects
6,443 of those historical pairs. Regenerate gate_results.csv and review
population changes before re-pinning that artifact-dependent test.
