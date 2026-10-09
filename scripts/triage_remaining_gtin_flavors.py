#!/usr/bin/env python3
"""Read-only, complete review census of disjoint raw Flavour GTIN groups.

Classifications are curated source-evidence reviews, never replacement GTINs.
Source records and source anchors are retained outside the visible dashboard.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pandas as pd
from core.gtin import normalize_and_validate_gtin
from core.product_dimensions import row_dimensions
from core.text import normalized_attribute_text

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
# Every classification is grounded in the original fields retained below.
REVIEWS = {
 '20868784000326': ('mixed_identifier_variants', 'Carethy Gold SKU477584544 has query r=850003560295; White Grapefruit477949984 has r=20868784000326. Walmart485699652 is White with ginseng/guarana ingredient enum. Gold/White variation remains unresolved; ingredient terms alone are compatible.'),
 '3265266075026': ('compatible_naming', 'All three original URLs name menthe verte; green mint/spearmint naming does not establish different products.'),
 '3274490970212': ('confirmed_source_product_contradiction', 'Naturalia20442990 title, description and fraise URL say strawberry; Carrefour339334884 title, mint ingredients and menthe URL say mint; Biocoop959300328 says mint. True assignment unresolved.'),
 '3292482220015': ('compatible_naming', 'K Ruoka733706765 citron title has description explicitly stating taste of lemon; lemon/citrus enum difference is scope variation.'),
 '5404017402744': ('unresolved_flavor_and_pack_scope', 'Carrefour373484076 citron-vert URL means lime; Farmaline488935554 title/URL says lemon; Amazon1035247595 guarana enum names functional ingredient. Twelve-bottle versus single-bottle sellable scope also differs. No direct correcting source anchor.'),
 '5411188128021': ('compatible_product_style_enum', 'All names/descriptions agree caramel soy coffee; coffee versus latte enum is style. Repeated Co-op URL anchors seven listing versions; bracketed3/6 prefixes have no verified pack meaning.'),
 '5601607074866': ('confirmed_source_product_contradiction', 'Carethy653623196 blueberry and653993349 mango-passion have distinct product paths and flavor-named image paths; titles and raw flavors disagree under one number.'),
 '5999076228973': ('confirmed_source_product_contradiction', 'El Corte185319729 apple/pear title and URL with explicit EAN5999076228973; Carethy658173610 raspberry/lime title with number in query. Different flavor/pack assertions; no verified correction.'),
 '6417802389198': ('compatible_naming', 'Tangerine/mandarin terms map to the same original mandariini URL paths and concentrate product naming; no source flavor contradiction demonstrated.'),
 '7310050005171': ('compatible_product_style_enum', 'Every title is latte/macchiato iced coffee; coffee versus latte is ingredient/style overlap, not contradictory products.'),
 '7310070004819': ('compatible_ingredient_vs_flavor', 'ICA425667159 description explicitly has ginger and green tea extracts and lemon flavor; MatHem427770458 describes same extracts. Raw ginger/tea versus lemon separates ingredients from flavor.'),
 '7310070004826': ('compatible_ingredient_vs_flavor', 'Descriptions explicitly combine green coffee/ginseng extracts with melon flavor. Raw ingredient terms versus melon are compatible.'),
 '7310860007396': ('mixed_attribute_translation', 'MatHem55971544 title blood grape but description says red grapefruit, and grapefrukt URL; ICA96220854 same red grapefruit description. Raw grape came from translated blodgrape naming.'),
 '7314720722344': ('mixed_attribute_brand_leakage_and_reconstitution', 'MatHem730520510 title translated Kiviks as kiwi and raw Flavour kiwi, but both retailers have identical lemon descriptions. Title is2dl concentrate; description dilution1+4 can explain1000ml prepared volume, not source container volume.'),
 '7314720739038': ('compatible_generic_specific_enum_with_metadata_gaps', 'Blackcurrant versus generic berry is compatible; descriptions explicitly state6% berries while raw Juice Content25-50% is unsupported. ICA85588637 title1.5dl versus peers1.5l is unresolved unit transcription.'),
 '7350089431268': ('unresolved_compound_flavor', 'Hemkop983459721 title combines strawb Cherry Sunset, Coop988395862 says Strawberry Sunset; raw cherry/strawberry differ. No complete ingredients or label in snapshot to establish compound flavor versus mistaken title.'),
 '7610463011975': ('compatible_ingredient_vs_flavor_with_translation', 'Vitalabo language paths pfirsich and pesca both denote peach; descriptions name black tea, rose hips, hibiscus and peach/fishing. Tea enum is base versus peach flavor. First description says without carbonic acid while raw carbonization says sparkling: metadata contradiction.'),
 '7612100031650': ('compatible_naming', 'mcdrogerie480236139 Citron title description explicitly says Lemon500ml. Lemon/citrus enum represents varying specificity.'),
 '8003170086197': ('compatible_compound_flavor_translation', 'Conad638041767 original URL names mirtillo nero selvatico e rosso con uva: berry mixture with grapes; shorter blueberry row and richer berry/grape enum are compatible, not verified identical formulations.'),
 '8006290804917': ('compatible_ingredient_vs_flavor_with_translation', 'All source paths/labels describe peach tea with rosehip and lemon balm. Rosehip versus tea enum identifies different aspects of same named formula; fishing is translated pesca.'),
 '8006290804948': ('confirmed_source_product_contradiction', 'Carrefour424981349 andIper448919674 title/URL say agrumi/citrus; Coop468042285 title/URL/image filename say mela-mandorla/apple-almond. Source flavor variant assignment unresolved.'),
 '810013080315': ('mixed_source_fields_and_pack_scope', 'Meijer959752789 Strawberry Hibiscus title/attributes contradict its black-cherry/lime description. Target946857301 title3x8pack and description24units/288fl oz differ from8packs elsewhere. Same item-family number does not settle sellable pack.'),
 '8414192311974': ('confirmed_source_product_contradiction', 'Carethy690199361 BerryRush,691629885 coffee protein shake, andDecathlon896646718 strawberry Fruit&Fiber syrup have different type/flavor titles and distinct URLs; Decathlon description confirms syrup.'),
 '8437018830145': ('confirmed_source_product_contradiction', 'Auchan561072864 lemon/lime250ml title and URL versusElCorte758133718 orange/mango1.5L title,description and URL: distinct flavor and container.'),
 '850003797267': ('compatible_ingredient_vs_flavor_with_pack_scope', 'Safeway title abbreviates watermelon mint and names tea enum; Amazon ingredient block watermelon and mint flavor in yerba mate. Amazon5pack versus Safeway16oz listing scope not resolved by same number.'),
 '851220003124': ('compatible_product_style_enum', 'Wholefoods andHEB Chocolate Peanut Butter cold brew; Amazon title GirlScout PeanutButter latte. Shared ASINB0CHMYMMLS appears Wholefoods URL and Amazon dpURL: strong scoped source anchor for coffee/latte naming.'),
 '851220003223': ('compatible_product_style_enum', 'All titles ThinMint cold brew/latte; coffee versus mint enum is base/flavor. Shared ASINB0CV5PVLSX in Wholefoods andAmazon URLs corroborates named variant.'),
 '857161008457': ('confirmed_source_product_contradiction', 'Target318562504 andHyVee363370407 WildberryGinger with green-tea ingredient description; Amazon1000887803 Superberry title and oolong/raspberry ingredients. This is a formulation contradiction despite shared berry words.'),
 '860000709644': ('compatible_product_style_enum', 'All titles/descriptions oat-milk vanilla cold brew; coffee versus latte is style. No variant contradiction shown.'),
 '860002364100': ('compatible_product_style_enum', 'Thrive macadamia latte description explicitly coffee; Vitacost macadamia coffee same named variant. Style difference compatible.'),
 '860002364124': ('compatible_product_style_enum', 'Thrive black coffee andVitacost BlackLatte description explicitly black coffee with no milk. Latte naming not ingredient proof.'),
 '868784000346': ('compatible_generic_specific_enum', 'All listings name3DBlue/BerryBlue/BlueRaspberry473ml. Generic berry versus raspberry compatible; enum does not establish distinct formula. Caffeine basis issue separately documented.'),
 '8713300049748': ('root_in_progress_mixed_fields', 'JanLinders512665846 Passionfruit title conflicts with Orange URL; root owns Finding06 scoped repair.'),
 '8713300049779': ('root_in_progress_mixed_fields', 'JanLinders512465527 Strawberry title conflicts with MangoDream URL and supplier-description; root owns Finding06 scoped repair.'),
 '8717399840552': ('compatible_compound_flavor', 'Coop andDirk explicit lime-lychee compound title; Vomar abbreviated lime title; individual lime/lychee enums compatible subset evidence.'),
 '8719214811167': ('mixed_attribute_generic_word', 'Dirk520739801 title URL and description say orange raspberry; Flavour lemon appears unsupported, likely lemonade-word extraction. No verified lemon variant.'),
 '8720157465379': ('compatible_compound_flavor', 'Dirk841533100 explicit green-tea peach-hibiscus title; AH hibiscus/Coop peach+tea enums are compatible subsets. No contradictory formula shown.'),
}

def main():
    frame = pd.read_csv(ROOT/'dataset.csv', dtype=str, keep_default_na=False)
    original = frame.to_dict('records')
    facts = normalize_and_validate_gtin(frame.gtin)
    valid = facts.gtin_structurally_valid.tolist()
    dims = [row_dimensions({'attribute':r['attribute']}).attributes for r in original]
    gt_indices = defaultdict(list)
    url_indices = defaultdict(list)
    brand_indices = defaultdict(list)
    for i,r in enumerate(original):
        if valid[i]:gt_indices[r['gtin']].append(i)
        if r['sku_url']:url_indices[r['sku_url']].append(i)
        if valid[i]:brand_indices[normalized_attribute_text(r['brand'])].append(i)
    holds = json.loads((ROOT/'config/identity_reviews.json').read_text())['quarantined_gtins']
    entries=[]
    for gt, indices in sorted(gt_indices.items()):
        values = [dims[i].get('Flavour',frozenset()) for i in indices]
        if not any(a and b and a.isdisjoint(b) for a in values for b in values):continue
        if gt in REVIEWS: classification, reason=REVIEWS[gt]
        elif gt in holds: classification, reason='previously_held_identifier',holds[gt]['reason']
        else: raise AssertionError('Unreviewed flavor group '+gt)
        rows=[original[i] for i in indices]
        brands={normalized_attribute_text(r['brand']) for r in rows}
        terms={v for valueset in values for v in valueset}
        alternates=[]
        for j in sorted({j for brand in brands for j in brand_indices[brand]}):
            r = original[j]
            if r['gtin']==gt:continue
            if dims[j].get('Flavour',frozenset()) & terms:
                alternates.append({**r,'parsed_dimensions':{k:sorted(v) for k,v in dims[j].items()}})
        anchors=[]
        for i in indices:
            peers=[original[j] for j in url_indices[original[i]['sku_url']] if j!=i]
            if peers:anchors.append({'sku_id':original[i]['sku_id'],'sku_url':original[i]['sku_url'],'peer_rows':peers,
                                    'note':'Same source URL anchors listing continuity, not formulation/pack equality without other fields.'})
        entries.append({'gtin':gt,'classification':classification,'rationale':reason,
                        'already_held':gt in holds,'source_sku_ids':[r['sku_id'] for r in rows],
                        'row_count':len(rows),'source_rows':rows,
                        'parsed_dimensions':{original[i]['sku_id']:{k:sorted(v) for k,v in dims[i].items()} for i in indices},
                        'same_url_source_anchors':anchors,'valid_same_brand_flavor_alternatives':alternates,
                        'safe_action':'Preserve original evidence; investigate or apply explicitly adjudicated scoped metadata fixes. Never infer replacement GTIN from similarity.'})
    assert len(entries)==48, len(entries)
    # Additional explicit identifier-label matches outside the48raw-flavor signal.
    pattern=re.compile(r'\b(gln|item model number|model number|product code|product number|item code)\s*[:#-]?\s*[\u200e\u200f]*\s*(\d{8,14})\b',re.I)
    labelled=defaultdict(list)
    for i,r in enumerate(original):
        for match in pattern.finditer(r['description_short_eng']):
            if match[2]==r['gtin'] and valid[i]:
                labelled[r['gtin']].append({'sku_id':r['sku_id'],'label':match[1],'matched_text':match[0]})
    additional=[]
    flavor_keys={e['gtin'] for e in entries}
    for gt,matches in sorted(labelled.items()):
        if gt in flavor_keys or gt in holds:continue
        indices=gt_indices[gt]
        if len(indices)<2:continue
        additional.append({'gtin':gt,'classification':'no_additional_misuse_demonstrated_by_source_names',
                           'review_outcome':'Original variant names are consistent or abbreviated; model/product code may legitimately echo UPC. This label alone does not warrant identifier quarantine. Missing pack/volume context remains unknown.',
                           'label_matches':matches,'source_rows':[original[i] for i in indices],
                           'parsed_dimensions':{original[i]['sku_id']:{k:sorted(v) for k,v in dims[i].items()} for i in indices}})
    volume_candidates = []
    volume_reviews = {
        '5703828023392': 'Two titles and URLs explicitly125cl, raw125ml; peers1250ml. Unit conversion contradiction is grounded; true item label remains original.',
        '5060338920567': 'SKU44267256 title1Litre but raw100ml; peer1366430931000ml. Grounded raw/title dimension conflict.',
        '7310090189534': 'SKU56075909 title2dl and description dilution1+4 gives1liter; raw1000ml versus peers200ml. Container versus prepared volume context gap.',
        '7310090167532': 'SKU56163815 title2dl, URL200ml, description dilute1+4; raw1000ml versus peers200ml. Container versus prepared volume context gap.'}
    for gt, indices in sorted(gt_indices.items()):
        if gt in flavor_keys or gt in holds:continue
        claims=[]
        for i in indices:
            for value in dims[i].get('Volume',()):
                if value.replace('.','',1).isdigit() and float(value)>0:
                    claims.append({'sku_id':original[i]['sku_id'],'volume_raw':value,'numeric_ml_claim':float(value)})
        if len(claims)<2:continue
        lo=min(c['numeric_ml_claim'] for c in claims);hi=max(c['numeric_ml_claim'] for c in claims)
        if hi/lo<3:continue
        volume_candidates.append({'gtin':gt,'raw_volume_ratio':hi/lo,'claims':claims,
                                  'classification':'source_dimension_context_gap' if gt in volume_reviews else 'unadjudicated_volume_review_signal',
                                  'rationale':volume_reviews.get(gt,'Raw volume differs by3x or more. This signal alone cannot distinguish unit error, concentrate yield, variant, or offer pack.'),
                                  'source_rows':[original[i] for i in indices]})
    ambiguous_urls = []
    for url, indices in sorted(url_indices.items()):
        valid_indices = [i for i in indices if valid[i]]
        by_retailer = defaultdict(list)
        for i in valid_indices: by_retailer[original[i]['retailer']].append(i)
        for retailer, peers in by_retailer.items():
            gtins = sorted({original[i]['gtin'] for i in peers})
            if len(gtins) > 1:
                ambiguous_urls.append({'retailer':retailer,'sku_url':url,'distinct_valid_raw_gtins':gtins,
                                       'source_sku_ids':[original[i]['sku_id'] for i in peers],
                                       'titles':[original[i]['sku_name_eng'] for i in peers],
                                       'outcome':'URL is shared by multiple valid-numbered listings; require item-specific variant evidence before scoped reattachment or collapse.'})
    output={'source':'dataset.csv','original_rows':len(frame),'reviewed_group_count':len(entries),
            'reviewed_rows':sum(e['row_count'] for e in entries),'classifications':dict(Counter(e['classification'] for e in entries)),
            'entries':entries,'additional_explicit_identifier_label_candidates':additional,
            'same_retailer_exact_url_multiple_valid_gtin_groups':ambiguous_urls,
            'additional_high_ratio_raw_volume_candidates':volume_candidates,
            'scope':'All 48 disjoint raw Flavour groups exhaustively reviewed. Alternative records are evidence candidates, not adjudicated corrections. Additional numeric label candidates are not verified identifier misuse.'}
    target=ROOT/'dashboard/evidence/identity/remaining_gtin_flavor_triage.json'
    target.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['entries','additional_explicit_identifier_label_candidates','same_retailer_exact_url_multiple_valid_gtin_groups','additional_high_ratio_raw_volume_candidates']},indent=2))
    print('Additional explicit-number candidates:',len(additional),[e['gtin'] for e in additional])

if __name__=='__main__':main()
