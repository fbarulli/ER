JEV sample ledger — updated 2026-10-02

| Round | Status | Sample | Checkpoint | Pairs / calls |
| --- | --- | --- | --- | --- |
| 1 | Tested | sample_doubled.json | audit_results.jsonl | 80 / 160 |
| 2 | Tested | sample_doubled_2.json | audit_results_2.jsonl | 55 / 110 |
| 3 | Staged, not tested | sample_doubled_3.json | audit_results_3.jsonl (not created) | 120 / 240 |

Round 1: proceed split by mode-flavor evidence, fallback, and negative
similarity strata. Selection seed 29. Round 2: post-fix proceed survivors,
new negatives, new fallbacks, and retained negatives; its generation seed
was not recorded. Both rounds tested each pair in both directions.

Round 3: seed 43; 20 pairs in each positive/negative/uncertain × high/lower
similarity stratum. High similarity is >= 0.8. Polarity means the current
gate's decision, not an independent truth label. Replayed 900 candidates
and selected for diverse critical and full-universe attribute states.
Coverage includes volume, pack, package type, flavor, carbonation,
sweetener, pulp, and pack material. sample_3_summary.json records source
hashes, candidate pool sizes, allocations, and attribute coverage.

Round 3 excludes all 135 pairs staged/tested in rounds 1–2. All 255 pairs
across these three rounds are reserved: exclude unordered pairs, so
swapping GTIN order cannot reintroduce one. sample_ledger.json records
explicit unordered pairs and sample SHA-256 hashes for each round.
Future samplers must read this ledger or all sample_doubled*.json files.
After a run, update status and completed-call counts in this ledger.

Reproduce round 3: `.venv/bin/python jev/stage_sample_3.py`.
Run it when ready: `.venv/bin/python jev/run_audit.py --staging jev/sample_doubled_3.json --out jev/audit_results_3.jsonl`.
No round-3 live calls have been made.
