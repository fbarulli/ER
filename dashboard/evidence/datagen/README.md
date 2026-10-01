# Gate-decisions evidence

Serves the `/gate` route of `dashboard/app.py`.

- `gate_decision_sample.json` — deterministic sample written at render time by
  the `/gate` route: first 5 pairs in candidate (file) order from each gate
  decision bucket (`proceed` / `hard_no` / `fallback`) of `data/gate_results.csv`,
  both pair endpoints joined back to the RAW export (`dataset.csv` via
  `core.common.load_dataset`, SSOT column mapping) and shown with the
  original columns, as exported.
- `gate_reason` is the deciding clause; for `fallback` pairs it is also
  surfaced as `fallback_reason` on the page.
- Frames are cached by file modification time (same discipline as
  `dashboard/catalog.py`): re-rendering picks up artifact/data edits
  automatically; the JSON snapshot is byte-identical unless the artifacts
  change.
