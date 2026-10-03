"""Reviewed identity holds and scoped listing evidence; the only policy reader."""
from functools import lru_cache
import json
import logging
from pathlib import Path

import pandas as pd

from core.columns import raw_of
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
POLICY_PATH = Path(__file__).resolve().parents[2] / 'config/identity_reviews.json'

class Hold(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reason: str = Field(min_length=1)
    evidence_sku_ids: list[str]

class ListingHold(Hold):
    expected_gtin: str

class ListingContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    gtin: str
    inner_type: str | None = None
    inner_material: str | None = None
    outer_type: str | None = None
    outer_material: str | None = None
    pack_count: int | None = Field(default=None, gt=0)
    unit_volume_ml: float | None = Field(default=None, gt=0)
    prepared_volume_ml: float | None = Field(default=None, gt=0)
    source: str = Field(min_length=1)

class ListingIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_gtin: str
    target_gtin: str
    reference_sku_id: str
    expected_url: str = Field(min_length=1)
    reason: str = Field(min_length=1)

class ListingFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_gtin: str
    fields: dict[str, str]
    reason: str = Field(min_length=1)

class ReviewPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int
    quarantined_gtins: dict[str, Hold]
    listing_context: dict[str, ListingContext]
    listing_identity: dict[str, ListingIdentity] = Field(default_factory=dict)
    quarantined_listings: dict[str, ListingHold] = Field(default_factory=dict)
    listing_fields: dict[str, ListingFields] = Field(default_factory=dict)

@lru_cache(maxsize=1)
def review_policy() -> ReviewPolicy:
    policy = ReviewPolicy.model_validate(json.loads(POLICY_PATH.read_text()))
    if policy.schema_version != 1:
        raise ValueError('unsupported identity review schema')
    from core.gtin import is_valid_gtin_checksum
    if any(not is_valid_gtin_checksum(key) for key in policy.quarantined_gtins):
        raise ValueError('identity review keys must be structurally valid GTINs')
    held = {key.zfill(14) for key in policy.quarantined_gtins}
    for link in policy.listing_identity.values():
        if not is_valid_gtin_checksum(link.target_gtin) or link.target_gtin.zfill(14) in held:
            raise ValueError('reviewed listing identity target must be valid and eligible')
        if link.source_gtin.zfill(14) not in held:
            raise ValueError('reviewed identity corrections require a held source identifier')
    for fix in policy.listing_fields.values():
        if not set(fix.fields).issubset({"sku_name_eng", "attribute"}):
            raise ValueError("reviewed field corrections may only change title or attributes")
    return policy


def held_keys() -> frozenset[str]:
    return frozenset(key.zfill(14) for key in review_policy().quarantined_gtins)


def review_mask(gtins: pd.Series) -> pd.Series:
    from core.gtin import normalize_and_validate_gtin
    facts = normalize_and_validate_gtin(gtins)
    return facts.gtin_clean.astype('string').str.zfill(14).isin(held_keys())


def review_reason(gtin: object) -> str:
    from core.gtin import normalize_gtin_value
    key, _ = normalize_gtin_value(gtin)
    if key is None:
        return ''
    return next((hold.reason for raw, hold in review_policy().quarantined_gtins.items()
                 if raw.zfill(14) == str(key).zfill(14)), '')


def listing_review_reason(sku_id: object, gtin: object) -> str:
    hold = review_policy().quarantined_listings.get(str(sku_id))
    if hold is None:
        return ''
    from core.gtin import normalize_gtin_value
    key, _ = normalize_gtin_value(gtin)
    return hold.reason if key is not None and key.zfill(14) == hold.expected_gtin.zfill(14) else ''


def reviewed_row_mask(frame: pd.DataFrame, *, column: str | None = None) -> pd.Series:
    """Block reviewed source listings without rejecting their consistent GTIN peers."""
    column = column or ('gtin' if 'gtin' in frame else 'gtin' if 'gtin' in frame else None)
    if column is None:
        return pd.Series(False, index=frame.index)
    result = review_mask(frame[column])
    id_column = 'sku_id' if 'sku_id' in frame else 'sku_id' if 'sku_id' in frame else None
    if id_column:
        from core.gtin import normalize_and_validate_gtin
        keys = normalize_and_validate_gtin(frame[column]).gtin_clean.astype('string').str.zfill(14)
        for sku, hold in review_policy().quarantined_listings.items():
            result |= frame[id_column].astype(str).eq(sku) & keys.eq(hold.expected_gtin.zfill(14)).fillna(False)
    return result


def exclude_reviewed_rows(frame: pd.DataFrame, *, column: str | None = None) -> pd.DataFrame:
    column = column or ('gtin' if 'gtin' in frame else 'gtin' if 'gtin' in frame else None)
    if column is None:
        return frame.copy()
    frame = apply_identity_links(frame)
    held = reviewed_row_mask(frame, column=column)
    if held.any():
        logger.warning('identity review: excluded %s rows from labeling/splits; source listings retained', int(held.sum()))
    return frame.loc[~held].copy()


def apply_identity_links(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit reviewed links to a derived view; raw export stays intact."""
    result = frame.copy()
    id_column = 'sku_id' if 'sku_id' in frame else None
    gtin_column = 'gtin' if 'gtin' in frame else None
    url_column = 'sku_url' if 'sku_url' in frame else None
    if not all((id_column, gtin_column)):
        return result
    for sku, link in review_policy().listing_identity.items():
        if url_column is None:
            continue
        candidates = result[id_column].astype(str).eq(sku) & result[url_column].eq(link.expected_url)
        if candidates.any():
            from core.gtin import normalize_and_validate_gtin
            actual = normalize_and_validate_gtin(result.loc[candidates,gtin_column]).gtin_clean.astype('string').str.zfill(14)
            indices = actual.index[actual.eq(link.source_gtin.zfill(14))]
            result.loc[indices,gtin_column] = link.target_gtin
    for sku, fix in review_policy().listing_fields.items():
        candidates = result[id_column].astype(str).eq(sku)
        if candidates.any():
            from core.gtin import normalize_and_validate_gtin
            actual = normalize_and_validate_gtin(result.loc[candidates, gtin_column]).gtin_clean.astype("string").str.zfill(14)
            indices = actual.index[actual.eq(fix.expected_gtin.zfill(14))]
            for field, value in fix.fields.items():
                # raw_of resolves the destination through column_mapping
                # instead of re-declaring {"sku_name_eng": "sku_name_eng", …} here.
                destination = (
                    field
                    if field in result
                    else raw_of(field) or field
                )
                if destination in result:
                    result.loc[indices, destination] = value
    return result


def resolve_listing_row(row: dict) -> dict:
    """Scalar adapter to the same reviewed-link conditions."""
    sku = str(row.get('sku_id', '') or '')
    fix = review_policy().listing_fields.get(sku)
    if fix:
        from core.gtin import normalize_and_validate_gtin
        key = normalize_and_validate_gtin(pd.Series([row.get("gtin", "")])).gtin_clean.iat[0]
        if pd.notna(key) and str(key).zfill(14) == fix.expected_gtin.zfill(14):
            row = {**row, **fix.fields}
    link = review_policy().listing_identity.get(sku)
    if not link or row.get('sku_url') != link.expected_url:
        return row
    from core.gtin import normalize_and_validate_gtin
    actual = normalize_and_validate_gtin(pd.Series([row.get('gtin', '')])).gtin_clean.iat[0]
    if pd.notna(actual) and str(actual).zfill(14) == link.source_gtin.zfill(14):
        return {**row, 'gtin': link.target_gtin}
    return row
