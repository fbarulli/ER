# ER discovery dashboard

`/decisions` adds attribute-first decision tracking; `/api/decisions` exports
the same evidence as JSON. Select an attribute, optionally a GTIN pair, saved
gate decision, JEV scope or round. Gate pairs with JEV evidence appear first;
pagination defaults to 50 pairs and is bounded at 100. Attribute comparison
counts describe the displayed page, while saved report metrics retain their
original population and source.

The view reuses the shared attribute registry, decision engine, veto-policy
ledger, GTIN normalizer, embedding provenance validator, graph preflight and
the dashboard's safe training-run/archive readers. Current engine comparisons
are labeled separately from saved gate decisions. Historical model scores
retain their run, artifact, source line, frozen SKU→GTIN mapping and saved
threshold/checkpoint manifest. JEV rounds, scopes and both orders remain
separate observations. Gate-derived labels are not presented as human truth.

The attribute gate is leaving the decision path (owner ruling 2026-10-03): it
is attribute-driven, so it can never be the label source nor a feature; it runs
in shadow mode only, to compare "model alone" against the gate on real pairs.
The `/gate` page states the ruling and the active negative-supply mode
(`training.negative_supply.mode`; default `gate`), whose replacement is the
real-partner-first lane (`src/training/negative_supply.py`: real partners
first, minted only to top-up, minted rows training-only). The default stays
`gate` until the real-vs-minted discriminator and the stratified eval clear.

The attribute × difficulty × masking × generated-data table reads existing
visibility CSVs and frozen bundle headers. It preserves field hits, modes,
donor/source payload indices, folds, epochs and presentation records when
emitted. Missing fields remain unknown; a population label is distinguished
from a measured difficulty grade. Entity-only lineage is not asserted to
identify an exact pair. Bundle checksum validation is distinguished from
current training readiness. Payload-variant ablation results are shown when
available; they do not establish individual attribute weights.

Source locations use `core.common.F` and the configured model-track suite.
Additional masking-field aliases live in `decision_attribute_aliases` in
`config/paths.yaml`; shared critical-dimension aliases come from the existing
attribute-conflict registry. No GPU jobs or models execute in this view.
Evidence is reread on each request, concurrent file changes reject the
snapshot, and stale vectors cannot contribute cosine scores. Configuration
changes require restarting the dashboard and are rejected until then.

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

Training reports are available at `/training`. The run selector discovers
downloaded suite ZIPs and local run directories directly under
`results/model_tracks/`, `results/graph_tracks/`, and `results/training_results/`.
It displays saved PNG plots and model evaluation, retrieval, and fold metric
CSV summaries (the first 20 data rows). Click a plot to open its original image.
Suite metadata shows the device, epochs, test-reporting setting, and whether a
suite result artifact is present. Smoke scores verify the workflow and should
not be treated as model quality benchmarks. Input bundles are excluded from
the selector. Incomplete runs can be inspected as reports arrive; unreadable
archives or metrics show a message while other report artifacts remain usable.

Each run also exposes durable training logs: `suite_events.jsonl`, each track's
`worker_events.jsonl`, raw `<track>__worker.log` output, and saved epoch events.
Runs containing logs can be selected before plots exist. Structured event rows
record UTC timestamps, attempts, run/track identifiers, phases, outcomes, and
available stage details. Failed stages retain their error details in the logs.
The dashboard preserves the original JSONL and raw text, escapes HTML, and
previews at most 65,536 bytes per log with an explicit truncation notice.
Use **Download full log** for the complete byte-for-byte file; downloads stream
without the preview limit. `/training/log?run=...&artifact=...` opens a text
preview; add `&download=true` to download the original file.

For downloaded ZIPs, an adjacent `<run>.events.jsonl` appears as
`__collected__/suite_events.jsonl`. This final supervisor snapshot can include
archive/publication events recorded after the ZIP was written. The archived
`suite_events.jsonl` remains separately available. Refresh local runs to see
newly saved events. Older runs without event logs explicitly show that no saved
training logs are available; the dashboard does not reconstruct missing events.
Traversal, symlink files/directories, and unrelated artifacts are rejected by
the log route.

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

JEV audit samples and results are available at `/jev`. The page reads the
`jev/` sample ledger, current checkpoints, round-3 attribute coverage, and
saved verification results on each request. Round 3 contains 500 unique
pairs (1,000 ordered calls), tested via OpenRouter (all 1,000 calls completed).

The JEV page defaults to the highest-numbered fresh-sample round in the
ledger, with 250 gate-data pairs and 250 original-data pairs for round 4. It
displays per-cohort judgments and the deliberately repeated same-pair
comparison (round 5). Downloads include frozen inputs, successful results, and
run provenance.

Round 6 compares both original and processed evidence on the same 100 fresh
pairs (400 completed calls). Selecting that round shows matched score-category
changes, per-format judgments, and a downloadable paired comparison. Ledger
call counts distinguish input format and order.

Frozen-checkpoint attribute influence is read from the configured
`decision_ablation_report` binding. The report preserves attribute, intervention
channel, difficulty, masking and generation lineage alongside gate/JEV fields
when supplied by the pair artifact. Missing lineage stays unknown. Request,
result, checkpoint, source and composer hashes must still match; stale scores
are withheld. Influence describes this intervention on this sample, rather than
an intrinsic model weight. Retrieval ranks use the fixed sampled endpoint
catalog, not the full production ANN index.
