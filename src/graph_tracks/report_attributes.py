"""Report the existing ER attribute classes using shared extraction and metrics."""
import json
from pathlib import Path

import pandas as pd

from training.attribute_separation import ATTRIBUTE_SOURCES, attribute_separation, separation_spec

FILENAME = 'report_attributes.json'


def identity_attributes(identity):
    return {attribute: sorted(getattr(identity, 'volume_ml' if attribute == 'volume' else attribute))
            for attribute in ATTRIBUTE_SOURCES}


def write_inputs(output: Path, rows: list[dict]):
    (output / FILENAME).write_text(json.dumps({
        'schema': 'er-report-attributes-v1', 'attributes': list(ATTRIBUTE_SOURCES), 'listings': rows,
        'extractor': 'core.product_identity.row_identity',
        'class_registry': 'training.attribute_separation.ATTRIBUTE_SOURCES',
    }, sort_keys=True) + '\n')


def load_inputs(listings: Path, records: list[dict]):
    companion = listings.parent / FILENAME
    if companion.is_file():
        data = json.loads(companion.read_text())
        if data.get('schema') != 'er-report-attributes-v1' or data.get('attributes') != list(ATTRIBUTE_SOURCES):
            raise ValueError('report attribute class registry mismatch')
        rows = data['listings']
        values = {row['product_id']: row['attributes'] for row in rows}
        if len(values) != len(rows) or set(values) != {r['product_id'] for r in records}:
            raise ValueError('report attribute listing population mismatch')
        if any(set(row) != set(ATTRIBUTE_SOURCES) for row in values.values()):
            raise ValueError('report attribute classes missing')
    else:
        # Legacy/synthetic graph inputs omit some existing classes. Preserve
        # those as unobservable instead of inventing values or new classes.
        values = {r['product_id']: {a: r['numeric'].get('volume_ml' if a == 'volume' else a,
                    r['attributes'].get(a, [])) for a in ATTRIBUTE_SOURCES} for r in records}
    return {key: {a: frozenset(f'{v:g}' if isinstance(v, (int, float)) else str(v)
                              for v in row[a]) for a in ATTRIBUTE_SOURCES}
            for key, row in values.items()}


def write_reports(listings: Path, records, pairs, splits, output: Path, track: str):
    from graph_tracks.artifacts import name
    values = load_inputs(listings, records)
    summaries, per_values = [], []
    for split in splits:
        indices, labels = pairs[split]
        population = pd.DataFrame({'true_label': labels.astype(int)})
        for attribute in ATTRIBUTE_SOURCES:
            for side in (0, 1):
                population[f'{attribute}__{side + 1}'] = [values[records[i]['product_id']][attribute]
                                                          for i in indices[:, side]]
        summary, by_value = attribute_separation(population, spec=separation_spec())
        summaries.append(summary.assign(split=split, model=track))
        per_values.append(by_value.assign(split=split, model=track))
    pd.concat(summaries, ignore_index=True).to_csv(output / name(track, 'attribute_separation_summary.csv'), index=False)
    pd.concat(per_values, ignore_index=True).to_csv(output / name(track, 'attribute_separation_values.csv'), index=False)
