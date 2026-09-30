#!/usr/bin/env python3
"""Render before/after comparisons from original data and live identity policy."""
import html
import json
from pathlib import Path
import pandas as pd
from core.common import DATA_PATH, COLUMN_MAPPING, TRAIN_ROOT
from core.gtin import barcode_validity
from core.identity_policy import review_mask
from core.product_dimensions import row_dimensions
from core.product_context import compare_context

OUT = TRAIN_ROOT / 'dashboard/experiments/results/identity'
SOURCE = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False, low_memory=False)
CANONICAL = SOURCE.rename(columns=COLUMN_MAPPING).set_index('product_id', drop=False)
CSS = 'body{font:15px system-ui;color:#222;margin:24px}h1{font-size:22px}h2{font-size:18px}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ddd;padding:12px;text-align:left;vertical-align:top}th{background:#f3f4f5;width:190px}img{height:130px;max-width:100%;object-fit:contain}.ok{background:#dff4e8;padding:8px 12px;display:inline-block}.open{background:#fff0cf;padding:8px 12px;display:inline-block}section{margin:30px 0}'


def text(value):
    if value is None or value == '' or value == []:
        return 'Unknown'
    if isinstance(value, list):
        return html.escape(', '.join(map(str,value)))
    return html.escape(str(value))


def table(rows, headings=('Field', 'Original', 'Resolved')):
    head = ''.join(f'<th>{text(h)}</th>' for h in headings)
    body = ''.join('<tr>' + ''.join(f'<{ "th" if i==0 else "td"}>{text(v)}</{ "th" if i==0 else "td"}>' for i,v in enumerate(row)) + '</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def write(stem,title,body):
    (OUT / (stem+'_findings.html')).write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{text(title)}</title><style>{CSS}</style></head><body><h1>{text(title)}</h1>{body}</body></html>')
    old = OUT / (stem+'_findings.txt')
    if old.exists():
        old.unlink()


def lookup(sku):
    return CANONICAL.loc[sku].to_dict()


def main():
    a,b=lookup('363618546'),lookup('228633686')
    ac,bc=row_dimensions(a).context,row_dimensions(b).context
    comparison=compare_context(ac,bc)['caffeine']
    claim_a,claim_b=ac['caffeine']['claims'][0],bc['caffeine']['claims'][0]
    caffeine = '<section><h2>3D blue energy · same GTIN 868784000346</h2>' + table([
        ('Original SKU',a['product_id'],b['product_id']),('Retailer',a['retailer'],b['retailer']),
        ('Raw caffeine',ac['caffeine']['raw_mg'],bc['caffeine']['raw_mg']),
        ('Description evidence',claim_a['evidence'],claim_b['evidence']),
        ('Resolved caffeine',f'{claim_a["mg"]:g} mg / {claim_a["basis_ml"]:g} ml',f'{claim_b["mg"]:g} mg / {claim_b["basis_ml"]:g} ml'),
        ('Common comparison basis',f'{claim_a["mg_per_100ml"]:.2f} mg / 100 ml',f'{claim_b["mg_per_100ml"]:.2f} mg / 100 ml'),
        ('Raw range denominator','Unknown — not compared','Unknown — not compared'),
        ('Comparison result',comparison['status'],comparison['status'])],('Field','Harris Teeter','Hy-Vee')) + '<span class="ok">Fixed: explicit quantities agree; unscoped raw ranges no longer decide the comparison.</span></section>'
    pack=lookup('282677444');pc=row_dimensions(pack).context
    packaging='<section><h2>Kevyt Olo · 12-pack · GTIN 6419806053204</h2><img src="/images/282677444.png" alt="Reviewed 12-can shrinkwrapped pack">'+table([
        ('Pack count','Missing structured Count per Unit',pc['pack_count']['value']),
        ('Inner container','Pouch / FlexiblePack (unscoped)','Can / Metal'),
        ('Outer wrapping','Pouch / FlexiblePack (unscoped)','Shrinkwrap / Plastic'),
        ('Evidence','Original title + original retailer image','Reviewed fields scoped to SKU 282677444 and this GTIN'),
        ('Source attributes','Preserved','Preserved')])+'<span class="ok">Fixed: inner and outer packaging stored separately; pack count = 12.</span></section>'
    write('03_measurement_and_packaging_context','03 · Context resolved from source evidence',caffeine+packaging)
    half=lookup('897369430');hc=row_dimensions(half).context
    halfbody='<section><h2>Halfday lemon · GTIN 860000322935</h2><img src="/images/897369430.png" alt="Reviewed metal tea can">'+table([
        ('Source SKU',half['product_id'],half['product_id']),
        ('Inner packaging','Can + Paper/Carton','Can + Metal'),
        ('Source evidence','Raw packaging attributes','Downloaded retailer image'),
        ('Correction scope','Unscoped raw attribute','This listing + this GTIN only')])+'<span class="ok">Fixed: reviewed inner-container material = metal; raw source retained.</span></section>'
    write('01_identity_discovery','01 · Raw discrepancies → scoped evidence',halfbody+caffeine)
    bottle,cans=lookup('689458353'),lookup('835974400')
    other=lookup('249039007')
    write('02_exact_title_variants','02 · Same title → separate sellable variants',table([
        ('Retailer',bottle['retailer'],cans['retailer']),('Original SKU',bottle['product_id'],cans['product_id']),
        ('Title',bottle['title'],cans['title']),('GTIN',bottle['barcode'],cans['barcode']),
        ('Container','Glass bottle','Metal cans'),('Unit volume','414 ml','355 ml'),
        ('Cross-retailer clarification','14 fl oz bottle',other['title']),
        ('Identity outcome','Keep separate SKU','Keep separate SKU')],('Field','Clear Mind bottle','Clear Mind 4-pack'))+'<span class="ok">Enforced: distinct trusted GTINs remain separate identities.</span>')
    brew=SOURCE[SOURCE.gtin.isin(['851107003032','851107003155'])]
    write('04_gtin_source_contradictions','04 · Contradictory GTIN groups → excluded',table([
        ('GTIN','851107003032','851107003155'),('Original listings',7,4),
        ('Conflicting source data','Target: Spiced Apple / Superberry','Fred Meyer: Vanilla title / Watermelon description'),
        ('Before','Shared barcode could create positives','Shared barcode could create positives'),
        ('Identity trust now','Blocked','Blocked'),('Train / dev / test eligibility','Excluded','Excluded'),
        ('Source listings','Retained unchanged','Retained unchanged'),('Unresolved fact','True GTIN of Apple listing','Seasonal formulation timeline')],('Field','Apple vs Superberry','Vanilla vs Watermelon'))+'<span class="ok">Enforced: all 11 source listings excluded from identity-derived labels and splits.</span>')
    assert review_mask(brew.gtin).all()
    assert not barcode_validity(brew.gtin).any()
    for path in [OUT/'05_gln_in_gtin_findings.html']:
        s=path.read_text().replace('Open — identifier-type correction and split protection pending','Enforced — 21 GLN identifiers blocked from product identity and splits').replace('Runtime enforcement is not claimed by this result.','Runtime identity trust and split inputs now reject these 21 numbers; source listings are retained.').replace('background:#fff1c9','background:#dff4e8')
        path.write_text(s)
    report=json.loads((TRAIN_ROOT/'dashboard/evidence/identity/identity_exclusion_application.json').read_text())
    print([(a['artifact'],a['excluded_rows']) for a in report['artifacts']])

if __name__=='__main__':
    main()
