# Gate and regex evidence review — 2026-10-02

## Conclusion

All 13 source columns are retained in canonical `source_rows`, and registered
attribute values have a canonical evidence channel. That does not establish
complete semantic capture or complete use of helpful evidence. Specific numeric
roles, negated ingredients, nested quantities, and source contradictions are
still misread. The decision engine evaluates more fields than the gate lets
decide, and early returns can prevent later clarification from running.

This review uses the current 71,623-row source export and 135,246 committed
candidate pairs. Counterfactuals are rule comparisons, not independent truth
labels or measured accuracy improvements. The full gate replay was stopped
because other replay processes were already running; sampled replay and full
predicate censuses are explicitly distinguished in the evidence reports.

## Evidence to inspect

- `results/gate_logic_eval/pack_missing_evidence.json`: original listing cards,
  confidence, extracted sets, current and counterfactual verdicts.
- `results/regex_logic_eval/training_regex_findings.md`: exact text, extracted
  values, expected interpretation, confidence, downstream masking/exposure.
- `results/regex_logic_eval/training_regex_evidence.json`: full source fields
  and extraction ledgers for regex failures.
- `results/regex_logic_eval/training_regex_fusion.json`: source-disagreement
  examples, including confidence omitted by the current fusion calculation.
- `scripts/evaluate_gate_logic.py`: repeatable sampled decision review with
  source fingerprints, capture/use inventory, committed census, current actual
  verdicts, and separately labeled attribute-engine diagnostics.

## Findings

1. **Unknown pack is treated as a mismatch.** 25,739 pairs have one-sided pack
   evidence; 1,040 have known-side confidence below the configured 0.85 bar.
   Removing only the one-sided pack blocker sends 7,875 to fallback and leaves
   17,864 hard-negative. None proceeds. However, 3,948 of those 7,875 have
   another configured structured contradiction (2,672 flavor, 1,830 material,
   300 volume; counts overlap), which the earlier confidence return can hide.
   These are shared-predicate contradictions, not the complete fuzzy engine.
   Do not blanket downgrade this population.

   Example: key-lime listings `1047256705` and `1046775686`, GTINs
   `100140709075` and `100140791131`, both carry 1000 ml at confidence 0.995.
   One explicitly says pack of two; the other has no pack evidence. This is
   a missing-pack question, not evidence that the second listing is single.
   It is also not proof that the two trade items match.

2. **Captured disagreement can remain highly confident.** Among 31,509 raw
   rows selected for numeric/unit URL or image surfaces, 49 have fused volume
   confidence above an ignored weaker reader despite source disagreement.
   `fuse_confidence` examines only the first two readers when any reader
   disagrees. Title and copied listing URLs also are not independent sources.

3. **Regexes confuse measurement roles.** Source `1052557924` has
   "1 kcal per 100 ml ... 24 x 330 ml": title parsing returns 100 ml at 0.95.
   Source `79028934` has "1 / 2 gal": title parsing returns 7571 ml instead
   of about 1893 ml. Attributes mask both numeric errors, leaving fused
   confidence 0.90. Source `74185738` has "Makes 22 Quarts ... 72-Ounce
   Canister (Pack of 2)": prepared yield becomes 20820 ml at 0.95, and the
   multipack exemption prevents an ambiguity flag.

4. **Pack count loses nesting and packaging level.** `71073123` has
   "6X4 / 12 Oz" and produces count six instead of 24. `674512045` has
   "84 Cases, 2016 Bottles" and produces count 84; one scalar cannot express
   which level that count describes.

5. **Ingredient negation is not fully represented.** `371312669` says
   "No Stevia" but declares "Sweetener: stevia". The positive ingredient
   remains without a contradiction flag. Absence of an ingredient name is
   different from an explicit negative assertion.

6. **Displayed evidence can differ from deciding evidence.** The dashboard's
   first-listing attribute comparison is not the aggregate canonical evidence
   used by `three_way_gate`. The new report includes all persisted source rows
   and labels independent engine diagnostics separately from the actual fired
   gate reason. Recorded evidence is not proof it was consumed by that branch.

7. **Configuration does not steer every veto.** Carbonation is absent from
   configured `veto_dimensions`, but the earlier hardcoded `pack_gate` branch
   still blocks it. There are 16,763 carbonation-conflict candidates, including
   2,220 that pass its other predicates. In eight eligible counterfactual
   samples, disabling only that check leaves five hard-negative, sends one to
   review, and allows two proceeds. Those eight are not a population accuracy
   estimate; this demonstrates a policy-wiring discrepancy, not a reason to
   disable a useful contradiction. Details are in
   `results/gate_logic_eval/carbonation_policy_bypass.json`.

## Prioritized improvements

The committed snapshot has 13,216 canonical records. Volume confidence is
below 0.85 for 1,236: 875 have no observed volume and 361 have weak observed
volume. Pack confidence is below 0.85 for 9,921: 9,861 have no observed pack
and only 60 have weak observed pack. These are distinct problems; absent
evidence cannot be repaired by adjusting a confidence threshold. Volume
source-disagreement flags appear on 1,252 canonicals and pack disagreement on
37. The candidate census is 112,372 hard-negative, 16,611 proceed, and 6,263
fallback. Twenty-two deterministic samples across the 11 observed
decision/reason buckets reproduce the committed outcomes; this is sampled
consistency, not full replay fidelity or independently labeled accuracy.

1. Separate known contradictions from uncertain evidence in gate order. A
   trusted two-sided contradiction should remain hard-negative; a missing
   pack count should remain unknown. Measure combined rule changes together
   rather than changing the missing-pack veto in isolation.
2. Preserve numeric spans and roles before punctuation normalization: fraction,
   nutrition denominator, dilution yield, net weight, per-unit volume, total
   pack volume, outer quantity, and inner quantity. Record every candidate and
   why one wins; ambiguous role stays reviewable.
3. Make disagreement explicit in confidence calculation. Use all contradicting
   readers, record which value each reader supports, and distinguish copied
   surfaces from independent corroboration. Validate confidence calibration
   using independently reviewed samples, not gate-generated labels.
4. Let low-confidence cases reach targeted original-column clarification before
   returning review, while preserving hard conflicts and explicit uncertainty.
   Use remaining fields as supporting evidence only after measuring their
   within-product noise and their effect on true-match retention.
5. Separate missing evidence from weak observed evidence in review priorities.
   Break counts down by retailer, source column, parse status, and contradiction
   type. Review high-similarity uncertain pairs first for labeling value; do not
   lower global confidence thresholds simply to increase proceeds.
6. Keep extraction sanity bounds and confidence policy in typed YAML. Retain
   regex grammar in shared parsing code and avoid adding another parser per
   consumer. A multipack should not exempt an arbitrary per-unit outlier, and
   legitimate bulk formats need explicit roles rather than a larger global cap.

## Changes made during evaluation

- Fixed undefined pack bounds in `core.text.extract_pack_counts` using the
  existing configured bounds. This fixes the blocking-audit caller, not the
  separate canonical pack parser's semantic errors.
- Removed an unused duplicate critical evaluation from `three_way_gate`; its
  result did not participate in any decision.
- Added the repeatable evidence report and focused tests.

No gate policy, global confidence threshold, regex semantics, generated data,
or census pin was changed to suppress the findings.
