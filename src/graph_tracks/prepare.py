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
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from core.run_log import RunLogger
from training.prepare_all_trace import timed
from graph_tracks.data import RELATIONS, NUMERIC, file_hash, load_records

_LOG = RunLogger(__name__)


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
    """Catalog rows -> graph listings + report-attribute rows (shared extractor)."""

    def __init__(self, splits: ListingSplits):
        self._splits = splits

    def scrape(self, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        from core.sku_identity import row_identity
        from graph_tracks.report_attributes import identity_attributes
        records: list[dict] = []
        report_rows: list[dict] = []
        for _, row in _LOG.progress(
            frame.iterrows(), desc="listing_identity_scrape", unit="listing",
            total=len(frame),
        ):
            identity = row_identity(row)
            report_rows.append({'sku_id': row.sku_id,
                                'attribute': identity_attributes(identity)})
            records.append(self._splits.listing_record(
                row.sku_id, identity, RELATIONS, NUMERIC))
        return records, report_rows


class PairSourceBinding:
    """The lineage file that binds the prepared pair source (or 'unknown')."""

    SCHEMA = 'er-graph-pair-lineage-v1'

    def __init__(self, pairs: Path, output: Path):
        self._pairs = pairs
        self._output = output
        self._source = pairs.parent / 'pair_lineage.json'
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
        (self._output / 'pair_lineage.json').write_bytes(self._source.read_bytes())

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
                self._listing_path.parent / 'pair_lineage.json'
            ) if (self._listing_path.parent / 'pair_lineage.json').is_file() else None,
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
    with _LOG.section("graph_prepare.read_inputs"):
        frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
        if reviewed_row_mask(frame).any():
            raise ValueError("catalog contains quarantined identity groups/listings; apply reviewed exclusions before preparing splits")
        assignment = pd.read_csv(splits, dtype=str, keep_default_na=False)
    with _LOG.section("graph_prepare.listing_scrape"):
        contract = ListingSplits(frame, assignment)
        records, report_rows = ListingScraper(contract).scrape(frame)
    with _LOG.section("graph_prepare.publish"):
        output.mkdir(parents=True, exist_ok=False)
        listing_path = output / 'listings.json'
        write_json(listing_path, {'schema': 'er-graph-listings-v1', 'listings': records})
        from graph_tracks.report_attributes import write_inputs
        write_inputs(output, report_rows)
        records = load_records(listing_path)
        load_pairs(pairs, records)
        (output / 'pairs.csv').write_bytes(pairs.read_bytes())
        binding = PairSourceBinding(pairs, output)
        binding.publish()
        PreparedManifest(
            catalog=catalog, metric_splits=splits, metric_pairs=pairs,
            policy_path=POLICY_PATH,
            identity_dimensions=TRAIN_ROOT / 'config' / 'identity_dimensions.yaml',
            listing_path=listing_path,
        ).write(binding)
    from graph_tracks.prepared_inputs import prepare_training
    if training_tensors:
        with _LOG.section("graph_prepare.training_tensors"):
            prepare_training(listing_path, output / 'pairs.csv')
    return listing_path


def main():
    RunLogger.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'splits', 'pairs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.splits, args.pairs, args.output)

if __name__ == '__main__':
    main()
