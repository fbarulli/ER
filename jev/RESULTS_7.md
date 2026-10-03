JEV round 7 — completed 2026-10-02

Completed and verified 1,000 OpenRouter calls to typesafe/jev-1.13: 900 fresh unique pairs and 100 reversed-order checks. All request hashes and raw response scores verified; no errors.

Question: Do record_a and record_b describe the exact same retail product (same brand, product type, flavor/variant, and pack size), accounting for typos, abbreviations, and wording differences?

Inputs: full original source listings. Primary results use only the first order. Score bands: different <=0.2; same >=0.8; uncertain between.

| Training label / stored → current gate | Pairs | Different | Uncertain | Same |
| --- | ---: | ---: | ---: | ---: |
| label_0|hard_no->fallback | 25 | 25 | 0 | 0 |
| label_0|hard_no->hard_no | 154 | 150 | 4 | 0 |
| label_0|hard_no->proceed | 1 | 1 | 0 | 0 |
| label_1|proceed->fallback | 233 | 213 | 19 | 1 |
| label_1|proceed->hard_no | 152 | 151 | 1 | 0 |
| label_1|proceed->proceed | 335 | 274 | 55 | 6 |

The current gate still approves 335 sampled stored positives; 274 receive low same-product scores. Rejections of stale positives mostly align with JEV (151/152 low scores), but surviving approvals still warrant direct source inspection. JEV judgments are evidence for review, not independent human ground truth.

Weighted estimates over the fresh eligible training population: stored positives score different in 93.6%, uncertain in 5.8%, and same in 0.6%; stored negatives score different in 97.4% and uncertain in 2.6%. These estimates use recorded stratum inclusion probabilities. They exclude previously reserved pairs and do not describe the entire candidate universe.

Order checks: 7/100 score-band changes; 2/100 absolute score gaps >=0.2. The check sample is stratified, so these are diagnostic counts rather than prevalence estimates.

Artifacts: report_7.json, audit_run_7.json, audit_results_7.jsonl, input_states_7.json, sample_7_summary.json. The ledger marks round 7 tested.
