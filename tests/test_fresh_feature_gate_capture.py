"""A report must expose original sources to the actual stage-7 gate consumer."""
import pandas as pd
from pipeline import generate_canonical,NgramIDF,three_way_gate
from scripts.fresh_feature_gate_report import attach_canonical_source_capture


def test_source_capture_changes_missing_flavor_into_original_column_conflict():
    groups={
        '1234567890123':[('Water 330ml','')],
        '1234567890130':[('Orange water 330ml','')],
    }
    idf=NgramIDF(groups)
    left=generate_canonical('1234567890123','Example',groups['1234567890123'],idf,None,
                            urls=['https://shop.example/vanilla-water-330ml'])
    right=generate_canonical('1234567890130','Example',groups['1234567890130'],idf,None)
    assert not left['flavor_set']
    assert three_way_gate(left,right)['decision']=='proceed'
    def source(title,url):
        return pd.DataFrame([dict(sku_name_eng=title,attribute='',description_short_eng='',
                                  breadcrumbs_eng='',sku_url=url)])
    left=attach_canonical_source_capture(left,source('Water 330ml','https://shop.example/vanilla-water-330ml'))
    right=attach_canonical_source_capture(right,source('Orange water 330ml',''))
    assert three_way_gate(left,right)['decision']=='hard_no'
