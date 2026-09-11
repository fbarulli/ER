i made several changes. and we have a new mission:
For silent drops specifically, it does not yet guarantee:

  - every script has a row-level input→output lineage table
  - every output file is written atomically with a completion marker
  - an interrupted run cannot leave partial or stale result files
  - a remote Colab result download is verified against a manifest of expected files and hashes
  - a changed source export automatically fails when row counts drift beyond an approved threshold
  - every deduplication removal is individually reviewable without opening its mapping/audit artifact
  - manual deletion or overwriting of artifacts is prevented
  - every external-library operation is checked for unexpected row loss

  Current controls make many intended drops counted and visible, and the self-test pins important census counts. To close the remaining gap, we should add a per-stage manifest: input/output row counts, IDs, hashes, expected artifacts, completion status, and explicit
  drop-reason totals.

give one item to an agent, have them research what has to be done, take that list and delegate with short single tasks to other agents




PR AUC + Recall@K + Precision@K + H@1