"""Export graph inputs from shared identity extraction and explicit listing splits.

No splits are invented and no gtin/label edges enter the model. The
listing schema is DERIVED from the shared extractor contract
(core.sku_identity.graph_schema) and its manifest records what was
derived at prepare time; a loader that sees a different schema refuses the
inputs as stale.

RESPONSIBILITY MAP (single-responsibility decomposition; behaviour pinned)
-------------------------------------------------------------------------
- :class:`ListingSplits` — the sku_id -> split contract over the retained
  catalog (the split map must cover exactly the retained catalog).
- :class:`ListingScraper` — one catalog row -> one graph listing record +
  one report-attributes row, through the shared extractor.
- :class:`PairSourceBinding` — the lineage file that binds the prepared
  pair source (byte copy + trace columns).
- :class:`PreparedManifest` — the er-graph-inputs-v1 manifest.
- :func:`prepare` — the stage orchestrator that threads these owners.

TRACE ROWS (core.tracing, the ONE consolidated trace)
-----------------------------------------------------
Stage ``graph_prepare``. Emitted:
  run   inputs.read              retained catalog rows -> split assignments
  run   local_inputs.read        the prepared package's own catalog/split files
  run   listing_scrape.scraped   catalog rows -> graph listing records, with the
                                 per-split census and the sampled listings
  group listing_scrape.reason_census  the EXACT per-split census
  ent   listing_scrape.*         the sampled listings behind each split
  run   catalog.quarantined_identity  EXCEPTION: the stage refuses a catalog
                                 carrying quarantined identity groups and names
                                 the offending sku_ids before it raises
  run   pairs.source_bound       the pair-source lineage binding (or 'unknown')
  run   manifest.published       the er-graph-inputs-v1 publication
  run   training_tensors.prepared  the tensor stage, when it runs
  run   output.published         listings.json written
Sampling caps are core.tracing's (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP) and
appear in the sample_budget row's detail; nothing here is unbounded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd

from core.run_log import RunLogger
from core.tracing import ENTITY_ROW_CAP, ENTITY_SAMPLE_PER_REASON, TraceRun
from training.prepare_all_trace import timed
from graph_tracks.data import RELATIONS, NUMERIC, file_hash, load_records

_LOG = RunLogger(__name__)

#: The pipeline stage these rows belong to (core.tracing ``stage`` column).
STAGE = "graph_prepare"

#: How many quarantined sku_ids the exception row names (the census carries the
#: exact count; the sample is what makes the failure actionable).
QUARANTINE_SAMPLE = 5


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


class ListingSplits:
    """The sku_id -> split assignment contract.

    No splits are invented here: the caller supplies the map, and the whole
    retained catalog must be covered exactly once.
    """

    def __init__(self, catalog: pd.DataFrame, assignment: pd.DataFrame):
        if 'sku_id' not in catalog or set(assignment.columns) != {'sku_id', 'split'}:
            raise ValueError('catalog needs sku_id; split CSV needs exactly sku_id,split')
        if catalog.sku_id.duplicated().any() or assignment.sku_id.duplicated().any():
            raise ValueError('listing IDs and split assignments must be unique')
        if set(catalog.sku_id) != set(assignment.sku_id):
            raise ValueError('split map must cover exactly the retained catalog')
        self.map: dict[str, str] = assignment.set_index('sku_id').split.to_dict()

    def split_of(self, sku_id: str) -> str:
        return self.map[sku_id]

    def listing_record(self, sku_id: str, identity, relations, numeric) -> dict:
        """One graph listing record for a scraped identity."""
        return {'sku_id': sku_id, 'split': self.split_of(sku_id),
                'attribute': {key: sorted(getattr(identity, key)) for key in relations},
                'numeric': {key: sorted(getattr(identity, key)) for key in numeric}}


class ListingScraper:
    """Catalog rows -> graph listings + report-attribute rows (shared extractor).

    The scrape is a pure per-row map (row_identity reads the row dict +
    config constants only), so the map is computed fork-parallel over the
    CPU budget and consumed ORDERED: child results arrive in catalog order,
    which keeps listings.json byte-identical to the serial scrape.
    """

    # Below this population the pool's fork + IPC overhead exceeds the
    # serial scrape cost; stay serial.
    PARALLEL_MINIMUM = 256

    def __init__(self, splits: ListingSplits):
        self._splits = splits

    def scrape(self, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        from core.sku_identity import row_identity  # bind pre-fork imports
        from graph_tracks.report_attributes import identity_attributes
        payloads = [
            (row.to_dict(), self._splits.split_of(row.sku_id))
            for _, row in frame.iterrows()
        ]
        if (os.cpu_count() or 1) > 1 and len(payloads) >= self.PARALLEL_MINIMUM:
            return self._parallel(payloads)
        return self._serial(payloads)

    def _serial(self, payloads: list[tuple[dict, str]]) -> tuple[list[dict], list[dict]]:
        from core.sku_identity import row_identity
        from graph_tracks.report_attributes import identity_attributes
        records, report_rows = [], []
        for payload, split in _LOG.progress(
            payloads, desc="listing_identity_scrape", unit="listing",
            total=len(payloads),
        ):
            identity = row_identity(payload)
            records.append({'sku_id': payload['sku_id'], 'split': split,
                            'attribute': {key: sorted(getattr(identity, key)) for key in RELATIONS},
                            'numeric': {key: sorted(getattr(identity, key)) for key in NUMERIC}})
            report_rows.append({'sku_id': payload['sku_id'],
                                'attribute': identity_attributes(identity)})
        return records, report_rows

    def _parallel(self, payloads: list[tuple[dict, str]]) -> tuple[list[dict], list[dict]]:
        import multiprocessing
        pool = multiprocessing.get_context('fork').Pool(
            os.cpu_count(), initializer=_warm_worker
        )
        try:
            results = list(pool.imap(_scrape_row, payloads, chunksize=32))
        finally:
            pool.close()
            pool.join()
        consumed = []
        for record, report_row in _LOG.progress(
            results, desc="listing_identity_scrape", unit="listing", total=len(results)
        ):
            consumed.append((record, report_row))
        records = [record for record, _ in consumed]
        report_rows = [report_row for _, report_row in consumed]
        return records, report_rows


def _warm_worker() -> None:
    """Bind the fork child's lazily-loaded config caches once, pre-scan."""
    try:
        from core.sku_identity import flavor_vocabulary  # noqa: F401
        from core.identity_policy import review_policy
        review_policy()
    except Exception:
        pass


def _scrape_row(scraped: tuple[dict, str]) -> tuple[dict, dict]:
    """One ordered fork-worker unit: identity -> (record, report row)."""
    from core.sku_identity import row_identity
    from graph_tracks.report_attributes import identity_attributes
    from graph_tracks.data import RELATIONS, NUMERIC
    payload, split = scraped
    identity = row_identity(payload)
    return (
        {'sku_id': payload['sku_id'], 'split': split,
         'attribute': {key: sorted(getattr(identity, key)) for key in RELATIONS},
         'numeric': {key: sorted(getattr(identity, key)) for key in NUMERIC}},
        {'sku_id': payload['sku_id'], 'attribute': identity_attributes(identity)},
    )


class PairSourceBinding:
    """The lineage file that binds the prepared pair source (or 'unknown')."""

    SCHEMA = 'er-graph-pair-lineage-v1'

    def __init__(self, pairs: Path, output: Path):
        self._pairs = pairs
        self._output = output
        self._source = pairs.parent / _setup_layout().pair_lineage
        self.lineage: dict | None = self._load()

    def _load(self) -> dict | None:
        if not self._source.is_file():
            return None
        lineage = json.loads(self._source.read_text())
        if (lineage.get('schema') != self.SCHEMA
                or lineage.get('listing_pairs_sha256') != file_hash(self._pairs)):
            raise ValueError('pair lineage does not bind the prepared pair source')
        return lineage

    def available(self) -> bool:
        return self.lineage is not None

    def publish(self) -> None:
        """Byte-copy the validated lineage into the prepared package."""
        if self.lineage is None:
            return
        (self._output / _setup_layout().pair_lineage).write_bytes(self._source.read_bytes())

    def trace(self) -> dict:
        """The manifest's pair_trace block (byte-identical key set)."""
        lineage = self.lineage
        return {'status': 'source_bound' if lineage
                        else 'unknown: no source lineage supplied',
                'source_trace_columns': lineage.get('source_trace_columns', [])
                                        if lineage else [],
                'missing_axes': lineage.get('missing_axes', []) if lineage
                                else ['difficulty', 'gate_evidence', 'gendata',
                                      'masking']}


class PreparedManifest:
    """The er-graph-inputs-v1 manifest over the prepared package's hashes."""

    def __init__(self, *, catalog: Path, metric_splits: Path, metric_pairs: Path,
                 policy_path: Path, identity_dimensions: Path, listing_path: Path):
        self._catalog, self._metric_splits, self._metric_pairs = (
            catalog, metric_splits, metric_pairs)
        self._policy_path, self._identity_dimensions = policy_path, identity_dimensions
        self._listing_path = listing_path

    def write(self, binding: PairSourceBinding) -> None:
        from graph_tracks.report_attributes import FILENAME
        from graph_tracks.train import write_json
        from graph_tracks.data import RELATIONS, NUMERIC
        write_json(self._listing_path.parent / 'input_manifest.json', {
            'schema': 'er-graph-inputs-v1', 'catalog_sha256': file_hash(self._catalog),
            'identity_policy_sha256': file_hash(self._policy_path),
            'identity_dimensions_sha256': file_hash(self._identity_dimensions),
            'splits_sha256': file_hash(self._metric_splits),
            'pairs_sha256': file_hash(self._metric_pairs),
            'pair_lineage_sha256': file_hash(
                self._listing_path.parent / _setup_layout().pair_lineage
            ) if (self._listing_path.parent / _setup_layout().pair_lineage).is_file() else None,
            'pair_trace': binding.trace(),
            'augmentation': 'not_applicable: fixed graph pairs; no masking/gendata pipeline',
            'listings_sha256': file_hash(self._listing_path),
            'identity_extractor': 'core.sku_identity.row_identity',
            'report_attributes_sha256': file_hash(
                self._listing_path.parent / FILENAME),
            'relations': list(RELATIONS), 'numeric': list(NUMERIC),
            'feature_scope': 'derived from core.sku_identity.graph_schema; every extractor descriptor is a model input',
            'excluded_model_inputs': ['gtin', 'verified identity edges', 'raw text'],
        })


@timed
def prepare(catalog: Path, splits: Path, pairs: Path, output: Path, *, training_tensors: bool = True) -> Path:
    from graph_tracks.train import load_pairs, write_json
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH, reviewed_row_mask
    # ONE consolidated-trace writer for the stage; committed once at the end (and
    # once, before raising, on the quarantine exception, so the failure survives).
    trace = TraceRun(STAGE)
    with _LOG.section("graph_prepare.read_inputs"):
        frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
        quarantined = reviewed_row_mask(frame)
        if quarantined.any():
            _record_quarantine(trace, frame, quarantined, catalog)
            trace.write()
            raise ValueError("catalog contains quarantined identity groups/listings; apply reviewed exclusions before preparing splits")
        assignment = pd.read_csv(splits, dtype=str, keep_default_na=False)
        trace.add(
            "inputs",
            "read",
            in_count=int(len(frame)),
            out_count=int(len(assignment)),
            reason=(
                "the retained catalog and its split assignment are read under the "
                "string-dtype contract; the split map must cover exactly the "
                "retained catalog"
            ),
            detail={"catalog": str(catalog), "splits": str(splits)},
            source=str(catalog),
        )
    with _LOG.section("graph_prepare.listing_scrape"):
        contract = ListingSplits(frame, assignment)
        records, report_rows = ListingScraper(contract).scrape(frame)
    _record_scrape(trace, frame, records, report_rows, catalog)
    with _LOG.section("graph_prepare.publish"):
        output.mkdir(parents=True, exist_ok=False)
        listing_path = output / 'listings.json'
        write_json(listing_path, {'schema': 'er-graph-listings-v1', 'listings': records})
        from graph_tracks.report_attributes import write_inputs
        write_inputs(output, report_rows)
        records = load_records(listing_path, trace=trace)
        load_pairs(pairs, records)
        (output / 'pairs.csv').write_bytes(pairs.read_bytes())
        binding = PairSourceBinding(pairs, output)
        binding.publish()
        trace.add(
            "pairs",
            "source_bound",
            detail=binding.trace(),
            reason=(
                "the prepared pair source is bound to its lineage file, or "
                "explicitly 'unknown: no source lineage supplied'"
            ),
            source=str(pairs),
        )
        manifest = PreparedManifest(
            catalog=catalog, metric_splits=splits, metric_pairs=pairs,
            policy_path=POLICY_PATH,
            identity_dimensions=TRAIN_ROOT / 'config' / 'identity_dimensions.yaml',
            listing_path=listing_path,
        )
        manifest.write(binding)
        trace.add(
            "manifest",
            "published",
            in_count=int(len(records)),
            out_count=int(len(records)),
            reason="the er-graph-inputs-v1 manifest is written over the package's hashes",
            detail={
                "path": str(listing_path.parent / 'input_manifest.json'),
                "listings": str(listing_path),
                "report_attributes_rows": int(len(report_rows)),
            },
            source=str(listing_path.parent / 'input_manifest.json'),
        )
        trace.add(
            "output",
            "published",
            in_count=int(len(records)),
            out_count=int(len(records)),
            reason="the graph listings the training tracks consume are published",
            detail={
                "listings": str(listing_path),
                "pairs": str(output / 'pairs.csv'),
                "records": int(len(records)),
                "split_census": {
                    split: sum(1 for record in records if record['split'] == split)
                    for split in sorted({record['split'] for record in records})
                },
            },
            source=str(listing_path),
        )
    from graph_tracks.prepared_inputs import prepare_training
    if training_tensors:
        with _LOG.section("graph_prepare.training_tensors"):
            prepare_training(listing_path, output / 'pairs.csv')
        from graph_tracks.prepared_inputs import ARRAYS, PLAN
        trace.add(
            "training_tensors",
            "prepared",
            in_count=int(len(records)),
            out_count=int(len(records)),
            reason=(
                "the fixed tensors (plan + arrays) are built once after the "
                "supervision projection, so the tracks never re-tensorize"
            ),
            detail={
                "plan": str(output / PLAN),
                "arrays": str(output / ARRAYS),
                "listings": int(len(records)),
            },
            source=str(output),
        )
    trace.write()
    return listing_path


# ── the stage's trace rows (real counts, named reasons) ─────────────────────
def _record_scrape(
    trace: TraceRun,
    frame: pd.DataFrame,
    records: list[dict],
    report_rows: list[dict],
    catalog: Path,
) -> None:
    """Catalog rows -> listing records, with the per-split census and its sample.

    The scrape is a pure 1:1 ordered map (row -> record), so a BATCH grain would
    only restate the row count; the split census is the grain that carries
    information, and it names both a bucket's exact population and the sample of
    listings behind it. The parallel path's chunk size is recorded so a reader
    knows how the work was partitioned.
    """
    trace.add(
        "listing_scrape",
        "scraped",
        in_count=int(len(frame)),
        out_count=int(len(records)),
        reason=(
            "one graph listing per retained catalog row, through the shared "
            "identity extractor; a row cannot be silently skipped (the map is 1:1 "
            "and ordered)"
        ),
        detail={
            "catalog_rows": int(len(frame)),
            "listing_records": int(len(records)),
            "report_attribute_rows": int(len(report_rows)),
            "parallel_chunk_size": 32,
            "parallel_minimum": ListingScraper.PARALLEL_MINIMUM,
            "batch_grain": (
                "the scrape is a 1:1 ordered map, so batches would only restate "
                "this row; the per-split census below is the informative grain"
            ),
        },
        source=str(catalog),
    )
    trace.add_entities(
        "listing_scrape",
        records,
        key_of=lambda record: record["sku_id"],
        reason_of=lambda record: record["split"],
        detail_of=lambda record: {
            "relations": sorted(record["attribute"]),
            "numeric_fields": sorted(record["numeric"]),
        },
        source=str(catalog),
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )


def _record_quarantine(
    trace: TraceRun, frame: pd.DataFrame, quarantined, catalog: Path
) -> None:
    """The EXCEPTION row: which identity groups are quarantined, and why.

    The stage refuses the catalog rather than preparing splits over poisoned
    identity. The row names the offending sku_ids (bounded sample) and the exact
    reason, so the trace answers "which rows, and why" without re-running the
    policy mask by hand.
    """
    offenders = frame.loc[quarantined.to_numpy(), "sku_id"] if "sku_id" in frame else []
    trace.add(
        "catalog",
        "quarantined_identity",
        in_count=int(len(frame)),
        out_count=int(len(frame)) - int(quarantined.sum()),
        reason=(
            "the catalog carries identity-quarantined groups/listings (identity "
            "review policy), so the stage REFUSES to prepare splits; apply the "
            "reviewed exclusions first"
        ),
        detail={
            "quarantined_rows": int(quarantined.sum()),
            "sample_sku_ids": [str(value) for value in list(offenders)[:QUARANTINE_SAMPLE]],
            "catalog": str(catalog),
        },
        source=str(catalog),
    )


def main():
    RunLogger.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'splits', 'pairs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.splits, args.pairs, args.output)

if __name__ == '__main__':
    main()
