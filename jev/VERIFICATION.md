Both JEV checkpoints verified offline against their staged samples and the
current gate on 2026-10-02. Reproduce with `.venv/bin/python jev/verify_audits.py`.
Detailed ordered scores, replay verdicts, and suspect pairs are in
verification_results.json.

| Check | Round 1 | Round 2 |
| --- | ---: | ---: |
| Completed / staged calls | 160 / 160 | 110 / 110 |
| Unique pairs, each in both orders | 80 | 55 |
| Integrity errors | 0 | 0 |
| Current hard_no / fallback / proceed pairs | 53 / 6 / 21 | 24 / 11 / 20 |
| Proceed pairs with both JEV scores below 0.2 | 13 | 15 |
| Rejected pairs with both scores above 0.8 | 0 | 0 |
| Gate decision differences when swapped | 0 | 0 |
| Mean absolute JEV order difference | 0.0265 | 0.0227 |
| Largest absolute JEV order difference | 0.50 | 0.32 |

All result statuses are ok, scores are finite and in [0, 1], ordered pairs
are unique, and staged metadata matches checkpoint metadata. Reverse-order
reason text differs for 2 and 11 pairs, respectively, solely by the order
of the reported left/right mode-flavor values; decisions are unchanged.

Round 1 originally staged 50 proceed pairs. Current policy rejects 27,
routes 2 to fallback, and still proceeds on 21. Round 2's 20 proceed
survivors all still proceed, including 15 pairs with consistently low JEV
scores. Across both rounds, 28 distinct low-score pairs still proceed.
The fixes are supported but do not resolve the remaining matching errors.

Round 2's newly rejected stratum has 30 calls and all scores below 0.2.
That validates the direction of the flavor fix within this sample. No
rejected pair has consistently high JEV confidence in either audit.
These samples are stratified and do not estimate population accuracy.

JEV evaluates only the first source listing for each GTIN, with description
truncated to 300 characters; the gate evaluates canonical evidence merged
across listings. Their inputs therefore differ. The checkpoints retain
scores and staging metadata, but not exact request payloads, raw responses,
model/adapter metadata, or timestamps. This verification establishes saved
checkpoint integrity and current replay behavior; it cannot independently
prove historical payload fidelity or reproduce the live model responses.
No new live calls were made.
