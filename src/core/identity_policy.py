"""Reviewed identity holds and scoped listing evidence; the only policy reader."""
from functools import lru_cache
import json
import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
POLICY_PATH = Path(__file__).resolve().parents[2] / 'config/identity_reviews.json'

class Hold(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reason: str = Field(min_length=1)
    evidence_sku_ids: list[str]

class ListingContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    gtin: str
    inner_type: str | None = None
    inner_material: str | None = None
    outer_type: str | None = None
    outer_material: str | None = None
    pack_count: int | None = Field(default=None, gt=0)
    source: str = Field(min_length=1)

class ReviewPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int
    quarantined_gtins: dict[str, Hold]
    listing_context: dict[str, ListingContext]

@lru_cache(maxsize=1)
def review_policy() -> ReviewPolicy:
    policy = ReviewPolicy.model_validate(json.loads(POLICY_PATH.read_text()))
    if policy.schema_version != 1:
        raise ValueError('unsupported identity review schema')
    from core.gtin import is_valid_gtin_checksum
    if any(not is_valid_gtin_checksum(key) for key in policy.quarantined_gtins):
        raise ValueError('identity review keys must be structurally valid GTINs')
    return policy


def held_keys() -> frozenset[str]:
    return frozenset(key.zfill(14) for key in review_policy().quarantined_gtins)


def review_mask(barcodes: pd.Series) -> pd.Series:
    from core.gtin import normalize_and_validate_gtin
    facts = normalize_and_validate_gtin(barcodes)
    return facts.gtin_clean.astype('string').str.zfill(14).isin(held_keys())


def review_reason(barcode: object) -> str:
    from core.gtin import normalize_and_validate_gtin
    key = normalize_and_validate_gtin(pd.Series([barcode])).gtin_clean.iat[0]
    if pd.isna(key):
        return ''
    return next((hold.reason for raw, hold in review_policy().quarantined_gtins.items()
                 if raw.zfill(14) == str(key).zfill(14)), '')


def exclude_reviewed_rows(frame: pd.DataFrame, *, column: str | None = None) -> pd.DataFrame:
    column = column or ('barcode' if 'barcode' in frame else 'gtin' if 'gtin' in frame else None)
    if column is None:
        return frame.copy()
    held = review_mask(frame[column])
    if held.any():
        logger.warning('identity review: excluded %s rows from labeling/splits; source listings retained', int(held.sum()))
    return frame.loc[~held].copy()
