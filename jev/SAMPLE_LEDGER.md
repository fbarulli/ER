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

Round 4: 500 fresh pairs, 1,000 completed OpenRouter calls. Excludes all
635 prior pairs. Stratified split seed 44 assigns 250 pairs to merged gate
inputs and 250 to all original source listings. Both halves have 84 proceed,
83 negative, and 83 fallback pairs, with balanced similarity bands. Gate
inputs omit source_rows and do not include gate decisions. Original inputs
retain all source fields and full descriptions. Independent cohorts do not
establish the causal effect of the input representation.

Round 5: deliberate controlled repeat of round-4 pair 810036262576 /
810036266772 (Bones Coffee S'morey Time versus High Voltage). One unique
pair, both input representations, both orders: four completed OpenRouter
calls. This repetition is intentional and must not be counted as four
unique pairs. sample_control_5.json and audit_results_5.jsonl preserve it.

Rounds 4–5 save frozen input_states files, request SHA-256 hashes, raw
responses, model/adapter metadata, and timestamps. Reproduce verification
with `.venv/bin/python jev/verify_frozen_runs.py`. The ledger now records
1,135 distinct reserved pairs; round 5 repeats one of those intentionally.

Round 6: paired comparison of 100 fresh pairs, excluding all 1,135 prior
reserved pairs. Selection seed 46; 34 proceed, 34 rejected, 32 fallback.
The same pairs are judged with merged processed gate evidence and all
original listings, each in both orders: 400 completed OpenRouter calls.
Each request contains both products using one evidence format, not the gate
verdict or pair-level attribute agreement summary. input_states_6.json
freezes both formats; audit_results_6.jsonl saves responses and request
hashes; paired_comparison_6.json saves matched results. All hashes and scores
verified. The ledger now reserves 1,235 distinct pairs, plus the intentional
round-5 repeat. Future sampling must exclude the ledger's unordered pairs.
