All items: old and rebuilt partner sets

The comparison holds every canonical item fixed and obtains its partners from each complete labeled-pair snapshot. Items with no eligible partners remain in the census.

| Measure | Before | After |
| --- | ---: | ---: |
| Canonical items | 13,216 | 13,216 |
| Candidate pairs | 135,246 | 135,246 |
| Positive training pairs | 8,071 | 864 |
| Negative training pairs | 7,766 | 7,814 |
| Items with positive partners | 5,333 | 1,229 |
| Items with no eligible partners | 5,417 | 7,937 |

5,775 items change their labeled partner sets; 2,732 lose all eligible partners. 7,747 pair IDs remain, 8,090 disappear, and 931 are added. 68 retained IDs change label.

These labels come from extraction and gating, not human truth. A smaller positive population can remove incorrect links or lose valid links; both require inspection. Final JEV validation includes current positives, current negatives and lost-positive diagnostics.

all_item_pair_changes.csv has one row per item. all_item_partner_sets.jsonl lists its actual old/new positive and negative partner IDs. same_items_summary.json includes cohorts of items from each previous JEV run; no old score is transferred to a different pairing. Previous CSVs remain under before/.
