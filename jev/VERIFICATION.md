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

Round 3 — live OpenRouter audit

All 1,000 calls completed successfully for 500 fresh pairs using
OpenRouter typesafe/jev-1.13. Sample/checkpoint metadata, score ranges,
uniqueness, and doubled ordering passed verification. Current gate decisions
agree in both orders for every pair.

Current routes: 168 proceed, 166 hard_no, 166 fallback. Of the proceed
pairs, 127 score below 0.2 in both orders. One hard_no pair scores above
0.8 in both orders. These are review candidates, not established truth
labels. Mean absolute JEV order difference is 0.03752; maximum is 0.61.
Detailed pairs and attribute strata are in verification_results.json.

audit_run_3.json records run provenance. audit_results_3.jsonl contains
only the 1,000 successes; the 1,000 sandbox DNS failures from the initial
attempt are retained separately in audit_errors_3.jsonl. The first-source
versus merged-input limitation remains. No gate changes were made from
these new judgments.

Round 4 — 50/50 evidence split

All 1,000 calls succeeded. All frozen request hashes and raw response scores
verified; there are no gate decision asymmetries or checkpoint integrity
errors. Each cohort has 250 pairs: 84 proceed, 83 rejected, 83 fallback.
Gate-data proceeds include 60/84 pairs with both JEV scores below 0.2;
original-data proceeds include 68/84. Neither cohort has a rejected pair
with both scores above 0.8. This independent-cohort comparison does not
isolate the effect of the evidence representation.

Round 5 — controlled same-pair repeat

Bones Coffee S'morey Time (810036262576) versus High Voltage (810036266772)
is a gate-approved pair whose original listings explicitly name distinct
variants, while the captured flavor values share latte/coffee. OpenRouter
returned 0.24 / 0.06 on merged gate inputs and 0.04 / 0.04 on original
listings, in original / swapped order respectively. All four responses and
request hashes verified. Both representations lean against matching;
merged inputs are less stable in this case. This selected example is not
a population estimate. control_comparison_5.json records the comparison.

Round 6 — paired original versus processed comparison

All 400 calls completed for the same 100 new pairs under both formats and
both orders. Frozen request hashes, raw response scores, uniqueness, and
sample coverage passed verification. Processed versus original evidence:
26/34 versus 30/34 approved pairs scored below 0.2 in both orders. Among
34 rejected pairs, 34 versus 33 were consistently low; neither format had a
consistently high rejection. Among 32 fallback pairs, 26 versus 29 were
consistently low. Ten of 100 pairs changed category (low / uncertain / high)
between formats; none flipped between consistently low and consistently
high. Average original-minus-processed score was -0.0296. Two approved
pairs scored consistently high with original evidence; none did with
processed evidence. These judgments are diagnostic rather than ground
truth, and this balanced sample does not measure population accuracy.
