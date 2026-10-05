import pandas as pd
from training.dedupe import _protect_untrusted_title_conflicts
from core.deduplication import collapse_representatives


def test_title_tiers_preserve_untrusted_flavor_splits_and_review_dimensions():
    rows = [
        ('A','Flavour: lemon',''), ('A','Flavour: lime',''),
        ('B','Tea Type: green',''), ('B','Tea Type: black',''),
        ('C','Flavour: lemon',''), ('C','Flavour: lemon',''),
    ]
    frame = pd.DataFrame(rows, columns=['sku_name_eng','attribute','_ident'])
    frame['retailer'] = 'shop'
    frame['brand'] = 'Example'
    frame['gtin'] = ''
    frame['rank'] = 1
    protected = _protect_untrusted_title_conflicts(frame)
    parent = {}
    kept, dropped = collapse_representatives(protected,
        ['retailer','sku_name_eng','_ident'], ['rank'], [False], parent=parent)
    assert len(kept) == 5
    assert set(kept[kept.sku_name_eng.eq('A')].attribute) == {'Flavour: lemon','Flavour: lime'}
    assert len(kept[kept.sku_name_eng.eq('B')]) == 2
    assert frame._ident.eq('').all()


def test_unknown_anchor_does_not_bridge_conflicting_variants():
    frame = pd.DataFrame({'sku_name_eng':['Drink']*3, 'retailer':['Shop']*3,
        'brand':['Example']*3,'gtin':['']*3,'_ident':['']*3,
        'attribute':['','Flavour: lemon','Flavour: lime']}, index=[10,20,30])
    assert _protect_untrusted_title_conflicts(frame)._ident.nunique() == 3


def test_existing_trusted_and_review_partitions_are_preserved():
    frame = pd.DataFrame({'_ident':['8715600246377','review:42'],
        'retailer':['Shop']*2,'sku_name_eng':['Drink']*2})
    assert _protect_untrusted_title_conflicts(frame) is frame
