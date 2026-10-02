JEV sample ledger — updated 2026-10-02

| Round | Status | Sample | Checkpoint | Pairs / calls |
| --- | --- | --- | --- | --- |
| 1 | Tested | sample_doubled.json | audit_results.jsonl | 80 / 160 |
| 2 | Tested | sample_doubled_2.json | audit_results_2.jsonl | 55 / 110 |
| 3 | Tested | sample_doubled_3.json | audit_results_3.jsonl (not created) | 500 / 1000 |

Round 1: proceed split by mode-flavor evidence, fallback, and negative
similarity strata. Selection seed 29. Round 2: post-fix proceed survivors,
new negatives, new fallbacks, and retained negatives; its generation seed
was not recorded. Both rounds tested each pair in both directions.

Round 3: seed 43; 84 pairs in each positive stratum and 83 in each negative/uncertain stratum, across high/lower
similarity stratum. High similarity is >= 0.8. Polarity means the current
gate's decision, not an independent truth label. Retained the original 120 reserved pairs. Replayed 2,496 candidates
and selected for diverse critical and full-universe attribute states.
Coverage includes volume, pack, package type, flavor, carbonation,
sweetener, pulp, and pack material. sample_3_summary.json records source
hashes, candidate pool sizes, allocations, and attribute coverage.

Round 3 excludes all 135 pairs staged/tested in rounds 1–2. All 635 pairs
across these three rounds are reserved: exclude unordered pairs, so
swapping GTIN order cannot reintroduce one. sample_ledger.json records
explicit unordered pairs and sample SHA-256 hashes for each round.
Future samplers must read this ledger or all sample_doubled*.json files.
After a run, update status and completed-call counts in this ledger.

Reproduce round 3: `.venv/bin/python jev/stage_sample_3.py --pairs 500`.
Run it when ready: `.venv/bin/python jev/run_audit.py --staging jev/sample_doubled_3.json --out jev/audit_results_3.jsonl`.
Round 3 completed all 1,000 calls via OpenRouter (typesafe/jev-1.13).
The initial sandbox DNS errors are preserved separately in audit_errors_3.jsonl;
audit_results_3.jsonl contains the successful checkpoint. audit_run_3.json
records the adapter, model, question, completion time, and artifact hashes.
