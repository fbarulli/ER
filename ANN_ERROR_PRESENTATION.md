# ANN Error Analysis: Current Run → Next Run

Validation set: 3,000 SKUs. The next run should be compared against this same
split (or a deliberately versioned replacement) so changes are attributable.

Each row is the same plot from the current run beside the corresponding
plot from the next run. Replace `NEXT_RUN_REPORT` with the new ANN report
directory after the next inference completes.

<table>
<tr><th>Plot</th><th>Current run</th><th>Next run</th></tr>
<tr><td>Overall ANN quality</td><td><img width="100%" src="training_results/0915T063500554948Z/worker_2/report/holdout_operating_metrics.png"></td><td><img width="100%" src="NEXT_RUN_REPORT/holdout_operating_metrics.png"></td></tr>
<tr><td>Error rate by attribute</td><td><img width="100%" src="training_results/0915T063500554948Z/worker_2/report/attribute_error_breakdown.png"></td><td><img width="100%" src="NEXT_RUN_REPORT/attribute_error_breakdown.png"></td></tr>
<tr><td>ANN candidate recall</td><td><img width="100%" src="training_results/0915T063500554948Z/worker_2/report/ann_ranking_hits_corrected.png"></td><td><img width="100%" src="NEXT_RUN_REPORT/ann_ranking_hits_corrected.png"></td></tr>
<tr><td>Score separation/calibration</td><td><img width="100%" src="training_results/0915T063500554948Z/worker_2/report/holdout_score_distributions.png"></td><td><img width="100%" src="NEXT_RUN_REPORT/holdout_score_distributions.png"></td></tr>
</table>

Baseline: adjusted Rand `0.9900`, pair recall `0.9803`, pair precision `1.0000`,
over-merge `0%`, under-merge `1.97%`.

Baseline attribute error rates: `pack/0` `41.9%`, `none/0` `34.2%`,
`flavor/1` `14.6%`, and `volume/0` `8.1%`.

Baseline retrieval recall is `99.84% @1`, `100% @5`, and `100% @10`. This
indicates that the primary opportunity is post-retrieval scoring and gating,
not ANN candidate generation.

Use this plot to verify that unit canonicalization and the missing-attribute
confidence penalty separate true and false pairs without lowering the global
decision threshold.

## Next-run acceptance criteria

- Improve `pack/0`, `none/0`, and `volume/0` error rates.
- Preserve `over-merge rate = 0%`.
- Do not lower the global threshold to obtain recall.
- Keep retrieval recall at or above the current baseline.
