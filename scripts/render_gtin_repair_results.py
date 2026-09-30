#!/usr/bin/env python3
"""Show original GTIN collisions, validated reattachments, and restored variants."""
import json
from pathlib import Path
import pandas as pd
from core.common import DATA_PATH, COLUMN_MAPPING, TRAIN_ROOT
from core.identity_policy import apply_identity_links, review_mask
from render_identity_fixes import table, write, text


def main():
    original=pd.read_csv(DATA_PATH,dtype=str,keep_default_na=False)
    records=original.rename(columns=COLUMN_MAPPING).set_index('product_id',drop=False)
    repair=json.loads((TRAIN_ROOT/'dashboard/evidence/identity/07_08_catalog_repair.json').read_text())
    selected=original[original.gtin.isin(['11982760','735143004010','7310050105482','7310050005492','7310050005485','735143004300','735143004355','735143004119'])]
    (TRAIN_ROOT/'dashboard/evidence/identity/07_08_original_rows.json').write_text(json.dumps({'columns':list(original.columns),'rows':selected.to_dict('records')},indent=2)+'\n')
    a,b=records.loc['117435982'],records.loc['140354705']
    collision='<section><h2>Same filed number · distinct products</h2>'+table([
        ('Original SKU',a.product_id,b.product_id),('Filed GTIN',a.barcode,b.barcode),('Brand',a.brand,b.brand),
        ('Title',a.title,b.title),('Pack','24 × 330 ml','12 × 230 ml'),('Product','Lemon/ginger BCAA drink','Protein coffee drink'),
        ('Before dedupe','Shared representative 117435982','Shared representative 117435982'),
        ('After repair','Separate listing restored','Separate listing restored'),('Label/split eligibility','Blocked pending true GTIN','Blocked pending true GTIN')],('Field','Maxim','Löfbergs'))+'</section>'
    source=records.loc['142547188'];ref=records.loc['140880643']
    corrected=apply_identity_links(pd.DataFrame([source.to_dict()])).iloc[0]
    link='<section><h2>Validated duplicate · same Caffeine Boost product</h2>'+table([
        ('SKU','142547188 + 143441192',ref.product_id),('Title',source.title,ref.title),('Product URL',source.url,ref.url),
        ('Description',source.description,ref.description),('Pack','12 × 230 ml','12 × 230 ml'),
        ('Source barcode','11982760',ref.barcode),('Resolved identity',corrected.barcode,ref.barcode),
        ('Representative',ref.product_id,ref.product_id),('Outcome','Reattached + duplicate collapsed','Reference item retained')],('Field','Two duplicate records','Validated source reference'))+'<span class="ok">Fixed · two reviewed duplicates rejoin item 7310050105482. Maxim and Protein Boost remain separate.</span></section>'
    write('07_matsmart_identity_repair','07 · Mat Smart · separate products + reattach duplicates',collision+link)
    rose,orange=records.loc['936055182'],records.loc['937665038']
    ambiguous='<section><h2>Shared model number · different sellable variants</h2>'+table([
        ('SKU',rose.product_id,orange.product_id),('Filed number',rose.barcode,orange.barcode),('Source title',rose.title,orange.title),
        ('Flavor','Rose','Orange blossom'),('Title unit size','500 ml','300 ml'),('Pack','Single size listing','2 bottles'),
        ('Description number label','Item model number','Item model number'),('Description Units','500 ml','500 ml — conflicts with 300 ml title'),
        ('Before dedupe','Amazon variants collapsed to SKU 872366880','Amazon variants collapsed to SKU 872366880'),
        ('After repair','Original listing retained separately','Original listing retained separately'),('Label/split eligibility','Blocked','Blocked')],('Field','Rose listing','Orange blossom listing'))+'</section>'
    goodrose,goodorange=records.loc['1008060154'],records.loc['1008893486']
    compare='<section><h2>Separately numbered comparison products</h2>'+table([
        ('SKU',goodrose.product_id,goodorange.product_id),('Title',goodrose.title,goodorange.title),
        ('GTIN',goodrose.barcode,goodorange.barcode),('Unit size','500 ml / 17 fl oz','500 ml / 17 fl oz'),
        ('Variant','Rose','Orange blossom'),('Can absorb unknown packs?','No · pack evidence required','No · flavor/volume/pack evidence required'),
        ('Outcome','Separate eligible item','Separate eligible item')],('Field','Rose reference','Orange blossom reference'))+'<span class="ok">Fixed · all 12 suspect source listings preserved; model-number grouping blocked. Replacement GTINs remain unknown.</span></section>'
    write('08_cortas_variant_repair','08 · Cortas · restore flavors, volumes and pack variants',ambiguous+compare)
    print({'catalog_before':repair['catalog_before_rows'],'catalog_after':repair['catalog_after_rows'],'validated_duplicate_aliases':repair['validated_duplicate_aliases']})

if __name__=='__main__':
    main()
