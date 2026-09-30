# ER discovery dashboard

Copied from `../broad-way/src/broadway/reports/experiments_dashboard.py` on
2026-09-30. The original dashboard is preserved in `vendor/`; `app.py` adds
an ER discovery view and configures local experiment/results/observation paths.
Experiment pages, observation verdicts, artifact previews, and canvas tools
remain available under `/experiments`, `/series`, and `/canvas`.

From the ER root:

```sh
uv pip install --python .venv/bin/python -r dashboard/requirements.txt
.venv/bin/python dashboard/app.py
```

Open http://127.0.0.1:8001. Findings start at 01, the identity dimension audit.
The initial evidence is a snapshot of `results/identity_dimensions/`; refresh
those JSON files into `dashboard/evidence/identity/` after rerunning
the audit. Images are verified downloads of retailer URLs, not historical
proof that the feed showed the same image at collection time. Missing images
remain links to their source. Observations are local generated files.

Counts describe a capped sample (20 pairs per GTIN). Disjoint metadata is a
review signal, not an established identity difference. No training is run by
this dashboard.

Finding 02 records the Clear Mind bottle versus four-can-pack example.
Finding 03 records a reproducible full-original-dataset audit of missing
measurement/packaging context:

```sh
PYTHONPATH=src .venv/bin/python scripts/audit_identity_context.py
```

The GTIN lookup at `/catalog?gtin=868784000346` loads the configured original
source via `core.common.DATA_PATH`, preserves all source columns, and compares
all 37 dimensions using `core.product_dimensions`. `/api/catalog` returns the
same evidence as JSON. Cross-retailer pair previews are capped at 20; UPC-12
and its leading-zero GTIN-13 representation find the same source group.
Source changes invalidate the in-memory catalog cache by file modification time.

Each experiment presents an HTML comparison with original values, resolved
values, and an outcome badge. Raw JSON/CSV evidence is retained under
`dashboard/evidence/identity/`, separately from dashboard result selectors.
Finding 04 documents unresolved contradictions in two Brew Dr GTIN groups.

Policy: `config/identity_reviews.json`. Identity trust, split inputs, and graph
preparation enforce the holds. Existing frozen artifacts were repaired with
`PYTHONPATH=src .venv/bin/python scripts/apply_identity_review_exclusions.py --apply`.
Re-render comparisons with `PYTHONPATH=src .venv/bin/python scripts/render_identity_fixes.py`.
