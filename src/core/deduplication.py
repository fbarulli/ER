"""Shared representative selection with complete source-row lineage.

Callers supply identity keys and ranking. This helper only implements the
stable first-representative reduction; it does not infer product identity.
"""

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def collapse_representatives(
    frame: pd.DataFrame,
    groups: Sequence[str],
    sort_cols: Sequence[str],
    ascending: Sequence[bool],
    *,
    parent: MutableMapping[Any, Any],
) -> tuple[pd.DataFrame, pd.Index]:
    """Keep the preferred row per group and map every row to its survivor.

    Missing group values group together, matching pandas keep-first duplicate
    removal. Ranking uses a stable sort with missing values last. Source
    indices must identify rows uniquely because lineage is keyed by index.
    Parent pointers are local to this tier; callers resolve their transitive
    closure after subsequent tiers finish.
    """
    if not frame.index.is_unique:
        raise ValueError("representative collapse requires unique source row indices")
    ordered = frame.sort_values(
        list(sort_cols), ascending=list(ascending), na_position="last", kind="stable"
    )
    positions = pd.Series(np.arange(len(ordered)), index=ordered.index)
    representative_positions = positions.groupby(
        [ordered[column] for column in groups], dropna=False
    ).transform("min")
    survivors = ordered.loc[representative_positions == positions]
    parent.update(dict(zip(
        ordered.index,
        ordered.index.to_numpy()[representative_positions.to_numpy()],
        strict=True,
    )))
    return survivors, frame.index.difference(survivors.index)
