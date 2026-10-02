"""Preserve production source capture when reporting freshly rebuilt canonicals.

The canonical factory does not add the caller's original-source columns.
Those columns are functional: the decision engine reparses them at stage 7.
"""
from __future__ import annotations

import pandas as pd


def canonical_source_capture(source_group: pd.DataFrame) -> dict:
    """Return the exact evidence fields added by run_within_brand_pipeline."""
    from pipeline import _source_rows_for

    def evidence(column: str) -> list[str]:
        return sorted({str(value).strip() for value in source_group[column]
                       if pd.notna(value) and str(value).strip()})

    return {
        'source_rows': _source_rows_for(source_group),
        'description_evidence': evidence('description_short_eng'),
        'breadcrumb_evidence': evidence('breadcrumbs_eng'),
    }


def attach_canonical_source_capture(record: dict, source_group: pd.DataFrame) -> dict:
    """Attach original sources before a report invokes the actual pair gate."""
    return {**record, **canonical_source_capture(source_group)}
