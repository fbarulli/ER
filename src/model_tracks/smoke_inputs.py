"""Project exactly 100 real listings and prepared text views for a CPU smoke.

Retains parent split assignments and materialized augmentation lineage. This
is a lifecycle smoke, not a quality experiment or a new training diet.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml


def prepare_smoke(setup: Path, output: Path, *, sample: int = 100):
    from core.common import SEED, training_cfg
    from graph_tracks.data import file_hash, load_text_cache
    from graph_tracks.prepare import prepare
    from graph_tracks.train import write_json
    from training.folds import normalize_gtin
    from training.prepared_bundle import load_prepared_bundle, prepared_holdout, write_prepared_bundle
    if output.exists():
        raise FileExistsError(output)
    catalog=pd.read_csv(setup/'eligible_catalog.csv',dtype=str,keep_default_na=False,low_memory=False)
    splits=pd.read_csv(setup/'listing_splits.csv',dtype=str)
    pairs=pd.read_csv(setup/'listing_pairs.csv',dtype={'product_id1':str,'product_id2':str})
    _,bundle=load_prepared_bundle(setup/'text_prepared.pkl.gz')
    populations=prepared_holdout(bundle,dict(training_cfg().split),seed=SEED)
    roles={normalize_gtin(key):role for role,values in enumerate(populations) for key in values}
    ids=set(catalog.product_id)
    representatives={}
    for row in catalog.itertuples(index=False):
        representatives.setdefault(normalize_gtin(row.barcode),row.product_id)
    chosen=set()
    for split in ('train','dev','test'):
        for label in (0,1):
            for row in pairs[(pairs.split==split)&(pairs.label==label)].head(4).itertuples(index=False):
                chosen.update((row.product_id1,row.product_id2))
    ndf=len(bundle['df'])
    for role in range(3):
        count=0
        for a,b in bundle['train_neg']:
            if a>=ndf:
                continue
            ka,kb=normalize_gtin(bundle['row_bc'][a]),normalize_gtin(bundle['row_bc'][b])
            source=str(bundle['df'].iloc[a].product_id)
            target=representatives.get(kb)
            if roles.get(ka)==roles.get(kb)==role and source in ids and target:
                chosen.update((source,target))
                count+=1
                if count==4:
                    break
    if len(chosen)>sample:
        raise ValueError('sample too small for selected smoke supervision')
    for listing in catalog.product_id:
        if len(chosen)==sample:
            break
        chosen.add(listing)
    if len(chosen)!=sample:
        raise ValueError('not enough listings for requested sample')
    frame=catalog[catalog.product_id.isin(chosen)].copy()
    assignment=splits[splits.product_id.isin(chosen)].copy()
    pair_frame=pairs[pairs.product_id1.isin(chosen)&pairs.product_id2.isin(chosen)].copy()
    output.mkdir(parents=True)
    frame.to_csv(output/'eligible_catalog.csv',index=False)
    assignment.to_csv(output/'listing_splits.csv',index=False)
    pair_frame.to_csv(output/'listing_pairs.csv',index=False)
    listings=prepare(output/'eligible_catalog.csv',output/'listing_splits.csv',output/'listing_pairs.csv',output/'prepared')
    selected_keys={normalize_gtin(v) for v in frame.barcode}
    source_indices=[i for i,p in enumerate(bundle['df'].product_id.astype(str)) if p in chosen]
    first_copy=min(int(r['copy_payload_idx']) for r in bundle['mask_audit']+bundle['hard_negative_mask_audit'])
    # Native retrieval reads the complete frozen canonical corpus immediately
    # after source rows. Retain that layout, but supervise only sampled pairs.
    supervised=set(source_indices+[i for i in range(ndf,first_copy) if normalize_gtin(bundle['row_bc'][i]) in selected_keys])
    indices=source_indices+list(range(ndf,first_copy))
    retained=set(indices)
    audits={}
    for field in ('mask_audit','hard_negative_mask_audit'):
        rows=[]
        for row in bundle[field]:
            sources=[int(v) for k,v in row.items() if k.endswith('_payload_idx') and not k.startswith('copy_') and v is not None]
            if not all(i in supervised for i in sources):
                continue
            rows.append(dict(row))
            retained.update(int(v) for k,v in row.items() if k.startswith('copy_') and k.endswith('_payload_idx') and v is not None)
            supervised.update(int(v) for k,v in row.items() if k.startswith('copy_') and k.endswith('_payload_idx') and v is not None)
        audits[field]=rows
    indices=source_indices+sorted(retained-set(source_indices))
    remap={old:new for new,old in enumerate(indices)}
    for rows in audits.values():
        for row in rows:
            for key,value in row.items():
                if key.endswith('_payload_idx') and value is not None:
                    row[key]=remap[int(value)]
    arrays={}
    masks={}
    for field in ('pos','hp_pairs','neg','train_neg'):
        source=np.asarray(bundle[field],dtype=int).reshape(-1,2)
        mask=np.array([int(a) in supervised and int(b) in supervised for a,b in source],dtype=bool)
        arrays[field]=np.array([[remap[int(a)],remap[int(b)]] for a,b in source[mask]],dtype=int).reshape(-1,2)
        masks[field]=mask
    write_prepared_bundle(output/'text_prepared.pkl.gz',df=bundle['df'].iloc[source_indices].reset_index(drop=True),
        payload=[bundle['payload'][i] for i in indices],structured_features=np.asarray(bundle['structured_features'])[indices],
        row_bc=np.asarray(bundle['row_bc'])[indices],country=np.asarray(bundle['country'])[indices],
        emb0=bundle['emb0'],**arrays,neg_sources=np.asarray(bundle['neg_sources'])[masks['neg']],
        train_neg_sources=np.asarray(bundle['train_neg_sources'])[masks['train_neg']],**audits,
        **{key:bundle[key] for key in ('labeled_pairs_csv','canonical_records_csv','gate_results_csv','payload_variant','masking_profile')},
        holdout_populations={s:sorted(values) for s,values in zip(('train','dev','test'),populations)})
    vectors,metadata=load_text_cache(setup/'shared_minilm__embeddings.npz',frame.product_id.tolist())
    metadata.update(parent_cache_sha256=file_hash(setup/'shared_minilm__embeddings.npz'),
                    catalog_sha256=file_hash(output/'eligible_catalog.csv'))
    np.savez_compressed(output/'shared_minilm__embeddings.npz',ids=frame.product_id.to_numpy(dtype=str),embeddings=vectors,metadata=json.dumps(metadata))
    manifest=json.loads((setup/'setup_manifest.json').read_text())
    manifest.update(smoke=True,source_listing_count=sample,parent_setup_sha256=file_hash(setup/'setup_manifest.json'))
    write_json(output/'setup_manifest.json',manifest)
    for track in ('gnn_only','hybrid'):
        settings=yaml.safe_load((setup/f'{track}.yaml').read_text())
        for key in ('listings','pairs','input_manifest','text_cache'):
            if settings.get(key):
                settings[key]=str(output/Path(settings[key]).relative_to(setup))
        settings.update(device='cpu',epochs=1,report_test=False)
        (output/f'{track}.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    cfg={'setup_dir':str(output),'text_bundle':str(output/'text_prepared.pkl.gz'),'text_model':'minilm_l6',
         'epochs':1,'device':'cpu','report_test':False,'publish_git':False,'profiling':True}
    (output/'suite.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    return output/'suite.yaml'
