#!/usr/bin/env python3
"""Restore original records merged on held identifiers; collapse only reviewed links."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from core.common import DATA_PATH, COLUMN_MAPPING, F, TRAIN_ROOT
from core.identity_policy import apply_identity_links, reviewed_row_mask, review_policy


def repair(source, catalog, mapping):
    held = source.loc[reviewed_row_mask(source)].copy()
    affected = set(held.product_id)
    old_ids = catalog.product_id.tolist()
    representatives = {}
    for row in mapping.itertuples(index=False):
        representatives[str(row.product_id)] = old_ids[int(row.rep_id)]
    restored = apply_identity_links(held)
    aliases = {}
    for sku, link in review_policy().listing_identity.items():
        candidates = restored.loc[restored.product_id.eq(sku)]
        reference = source.loc[source.product_id.eq(link.reference_product_id)]
        if candidates.empty:
            continue
        a, b = candidates.iloc[0], reference.iloc[0] if len(reference)==1 else None
        if b is None or a.barcode != link.target_gtin or b.barcode != link.target_gtin:
            raise ValueError('reviewed identity link is not supported by the reference source')
        # Exact source URL, retailer, brand, title and description are mandatory.
        for field in ('url','retailer','brand','title','description'):
            if str(a[field]).strip() != str(b[field]).strip():
                raise ValueError(f'reviewed duplicate source mismatch: {field}')
        from core.product_dimensions import row_dimensions, evaluate_dimensions
        differences = evaluate_dimensions(row_dimensions(a.to_dict()), row_dimensions(b.to_dict()))
        if any(v['review'] for v in differences.values()):
            raise ValueError('reviewed duplicate has unresolved raw dimension differences')
        aliases[sku] = link.reference_product_id
    retained = catalog.loc[~catalog.product_id.isin(affected)].copy()
    repaired = pd.concat([retained, restored.loc[~restored.product_id.isin(aliases)]],ignore_index=True)
    if repaired.product_id.duplicated().any():
        raise ValueError('repair produced duplicate source listing IDs')
    positions = {sku:i for i,sku in enumerate(repaired.product_id)}
    for sku in affected:
        representatives[sku] = aliases.get(sku,sku)
    if any(rep not in positions for rep in representatives.values()):
        raise ValueError('repair leaves a dangling representative mapping')
    remapped = pd.DataFrame({'product_id':mapping.product_id, 'rep_id':[positions[representatives[str(s)]] for s in mapping.product_id]})
    return repaired, remapped, {'catalog_before_rows':len(catalog),'catalog_after_rows':len(repaired),
        'original_affected_listings':len(held),'restored_listing_ids':sorted(set(repaired.product_id)-set(old_ids)),
        'validated_duplicate_aliases':aliases, 'held_rows_after_repair':int(reviewed_row_mask(repaired).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--output',type=Path,default=TRAIN_ROOT/'dashboard/evidence/identity/07_08_catalog_repair.json')
    args=parser.parse_args()
    source=pd.read_csv(DATA_PATH,dtype=str,keep_default_na=False).rename(columns=COLUMN_MAPPING)
    catalog=pd.read_csv(F['dataset_deduped'],dtype=str,keep_default_na=False)
    mapping=pd.read_csv(F['sku_to_rep'],dtype=str,keep_default_na=False)
    repaired,remapped,report=repair(source,catalog,mapping)
    report['applied']=args.apply
    report['source_sha256']=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if args.apply:
        for frame, key in ((repaired,'dataset_deduped'),(remapped,'sku_to_rep')):
            path=F[key]
            temporary=path.with_suffix(path.suffix+'.repair-tmp')
            frame.to_csv(temporary,index=False)
            temporary.replace(path)
    output=args.output
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
