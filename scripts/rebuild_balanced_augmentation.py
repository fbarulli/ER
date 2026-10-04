"""Rebuild augmentation from a verified bundle's original frozen endpoints.

Reuse immutable base text/identity inputs, discard the old generated suffix,
and prepare fresh tokens, objectives and epoch batches under current config.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

from core.common import training_cfg, resolve_model, SEED
from training.balanced_augmentation import augment_balanced
from training.prepared_bundle import load_prepared_bundle, write_prepared_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix+'.json').exists():
        raise FileExistsError('Frozen output already exists; choose a new versioned output path')
    _,base=load_prepared_bundle(args.source)
    canon=pd.read_csv(io.BytesIO(base['canonical_records_csv']),dtype=str,keep_default_na=False)
    end=len(base['df'])+len(canon)
    pos=base['pos'][np.asarray(base['pos']).max(axis=1)<end]
    keep=np.asarray(base['neg']).max(axis=1)<end
    neg, sources=base['neg'][keep],base['neg_sources'][keep]
    frozen=base.get('holdout_populations') or base['training_plan']['holdout']
    train=set(map(str,frozen['train']))
    indices={i for i,value in enumerate(base['row_bc'][:end]) if str(value) in train}
    spec=training_cfg().masking.balanced_augmentation
    if args.sample and spec.sample_counts is not None: spec=spec.model_copy(update={'counts':spec.sample_counts})
    pos,neg,payload,row_bc,features,pos_audit,neg_audit,coverage=augment_balanced(
        pos=pos,neg=neg,payload=base['payload'][:end],row_bc=base['row_bc'][:end],
        features=base['structured_features'][:end],df=base['df'],train_indices=indices,
        canonical_indices=set(range(len(base['df']),end)),spec=spec,seed=SEED)
    sources=np.concatenate([sources,np.full(len(neg)-len(sources),'counterfactual',dtype=object)])
    parents={int(row['copy_payload_idx']):int(row.get('copy_source_payload_idx')
        if row.get('copy_source_payload_idx') is not None else row['anchor_payload_idx'])
        for row in pos_audit+neg_audit}
    country=np.concatenate([base['country'][:end],np.asarray([base['country'][parents[i]]
        for i in range(end,len(payload))],dtype=object)])
    print(coverage.model_dump_json(indent=2),flush=True)
    manifest=write_prepared_bundle(args.output,df=base['df'],payload=payload,
        structured_features=features,row_bc=row_bc,country=country,pos=pos,hp_pairs=base['hp_pairs'],
        emb0=np.empty((0,0),dtype=np.float32),neg=neg,train_neg=neg,neg_sources=sources,
        train_neg_sources=sources.copy(),mask_audit=pos_audit,hard_negative_mask_audit=neg_audit,
        labeled_pairs_csv=base['labeled_pairs_csv'],canonical_records_csv=base['canonical_records_csv'],
        gate_results_csv=base['gate_results_csv'],payload_variant=base['payload_variant'],
        masking_profile=base['masking_profile'],holdout_populations=frozen,
        augmentation_coverage=coverage.model_dump(mode='json'),
        token_checkpoint=resolve_model(training_cfg().training.base_model),plan_sample=args.sample)
    args.output.with_suffix('.coverage.json').write_text(coverage.model_dump_json(indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':manifest.sha256}),flush=True)


if __name__=='__main__': main()
