"""Account for rows excluded from gtin grouping, without assigning identity.

Run: PYTHONPATH=src python -m training.audit_identity_retention
"""
from pathlib import Path
import json

import pandas as pd

from core.common import F, load_raw_export, load_dataset_deduped
from core.gtin import gtin_validity
from core.identity_policy import apply_identity_links, reviewed_row_mask


def main():
    raw = apply_identity_links(load_raw_export()).reset_index(drop=True)
    gtin = raw.gtin.fillna('').astype(str).str.strip()
    missing = gtin.str.lower().isin(['', 'nan', 'none', 'null'])
    reviewed = reviewed_row_mask(raw)
    valid = gtin_validity(gtin)
    excluded = missing | ~valid | reviewed
    retained = pd.read_csv(F['dataset_deduped'], dtype=str, keep_default_na=False)
    mapping = pd.read_csv(F['sku_to_rep'], dtype=str, keep_default_na=False)
    if mapping.sku_id.duplicated().any():
        raise ValueError('duplicate source IDs in representative map')
    # rep_id is a CSV row position, not a sku_id.
    positions = pd.to_numeric(mapping.rep_id, errors='raise').astype(int)
    if ((positions < 0) | (positions >= len(retained))).any():
        raise ValueError('representative map points outside the retained dataset')
    representatives = dict(zip(
        mapping.sku_id, retained.sku_id.iloc[positions].tolist(), strict=True
    ))
    retained_ids = set(retained.sku_id)
    eligible_ids = set(load_dataset_deduped().sku_id.astype(str))
    rows = raw.loc[excluded].copy()
    rows['exclusion_reason'] = [
        'identity_review' if reviewed.at[i] else
        'missing_gtin' if missing.at[i] else 'invalid_gtin'
        for i in rows.index
    ]
    rows['representative_id'] = rows.sku_id.astype(str).map(
        lambda value: representatives.get(value, value)
    )
    rows['retention_status'] = rows.representative_id.map(
        lambda value: 'matching_eligible' if value in eligible_ids else
        'preserved_review_hold' if value in retained_ids else 'unaccounted'
    )
    output = Path('results/identity_retention')
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / 'excluded_rows.csv', index=False)
    counts = rows.groupby(['exclusion_reason', 'retention_status']).size()
    report = {'excluded_rows': len(rows), 'counts': [
        {'reason': reason, 'status': status, 'rows': int(count)}
        for (reason, status), count in counts.items()
    ], 'note': 'Eligibility is not proof that a matching run completed or found a match.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if rows.retention_status.eq('unaccounted').any():
        raise RuntimeError('Source rows lack a retained representative; inspect excluded_rows.csv')


if __name__ == '__main__':
    main()
