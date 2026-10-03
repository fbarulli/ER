import importlib.util
from pathlib import Path
import pandas as pd
from core.identity_policy import apply_identity_links, exclude_reviewed_rows, review_policy
from core.sku_identity import row_identity, evaluate_sku_identity


def listing(sku,gtin='11982760',url=None):
    link=review_policy().listing_identity['142547188']
    return {'sku_id':sku,'gtin':gtin,'sku_url':link.expected_url if url is None else url,
            'retailer':'Mat Smart','brand':'Löfbergs','sku_name_eng':'Caffeine Boost 12 x 230ml',
            'description_short_eng':'Caffeine Boost coffee 12 drinks','attribute':'Volume: 230; Roast Type: dark'}


def test_reviewed_identity_link_restores_only_explicit_matching_listing():
    frame=pd.DataFrame([listing('142547188'),listing('143441192'),listing('unreviewed'),listing('142547188',url='different')])
    corrected=apply_identity_links(frame)
    assert corrected.gtin.tolist()==['7310050105482','7310050105482','11982760','11982760']
    assert frame.gtin.eq('11982760').all()
    eligible=exclude_reviewed_rows(frame)
    assert eligible.sku_id.tolist()==['142547188','143441192']
    decision=evaluate_sku_identity(row_identity(listing('142547188')),row_identity(listing('140880643','7310050105482')))
    assert decision['decision']=='same'


def test_catalog_restoration_preserves_variants_and_remaps_validated_duplicates():
    spec=importlib.util.spec_from_file_location('repair_catalog',Path(__file__).resolve().parents[1]/'scripts/repair_reviewed_catalog.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=pd.DataFrame([listing('117435982'),listing('140354705'),listing('142547188'),listing('143441192'),listing('140880643','7310050105482')])
    source.loc[0,['brand','sku_name_eng']]=['Maxim','Lemon ginger 24 x 330ml']
    source.loc[1,'sku_name_eng']='Protein Boost 12 x 230ml'
    catalog=source.iloc[[0,4]].reset_index(drop=True)
    mapping=pd.DataFrame({'sku_id':source.sku_id,'rep_id':[0,0,0,0,1]})
    repaired,remapped,report=module.repair(source,catalog,mapping)
    assert set(repaired.sku_id)=={'117435982','140354705','140880643'}
    assert report['validated_duplicate_aliases']=={'142547188':'140880643','143441192':'140880643'}
    reps=remapped.set_index('sku_id').rep_id
    assert reps['142547188']==reps['143441192']==reps['140880643']
    assert reps['117435982']!=reps['140354705']
