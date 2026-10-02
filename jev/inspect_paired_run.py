"""Inspect the paired round-6 payloads and reproduce category-flavor contamination."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import extract_all

FINDINGS={
('8693354001223','8693354004231'):('Named variant missing from structured comparison','Original titles distinguish spicy from plain turnip juice; both flavor sets are empty. Ingredient sets overlap, and variant wording survives mainly as residual text.','Prioritize a source-grounded spicy/plain variant field.','strong source evidence, noisy translated titles'),
('352154336161','811130031631'):('Likely wording/typo equivalence','Original titles describe Twiix/Twix iced coffee, both 8 fl oz ×12. Original scores are high; processed scores remain uncertain. Structured data agrees but readable title equivalence is less obvious.','Preserve original product names; verify potential duplicate identity before treating as a regression.','plausible match, not verified ground truth'),
('4104450005571','4104450005588'):('Carbonation strength collapsed','Titles distinguish Classic from Medium mineral water; both structured carbonation sets are carbonated. Pack material has contradictory Metal/PET evidence, which requires source review.','Capture product-specific carbonation strength separately from carbonated/still; retain material contradictions.','strong variant evidence; material evidence inconsistent'),
('856472002055','856472002086'):('Category contamination plus flavor subset approval','Original aloe drink versus pomegranate aloe drink. Coconut appears from the category Coconut and Other Plant Waters, not a declared product flavor. The added pomegranate is retained but accepted as subset-compatible.','Stop broad categories creating flavor agreement; review added declared flavors before approving containment.','confirmed category origin; explicit title distinction'),
('7310867561402','7310867562706'):('Captured pulp contradiction, correctly routed to review','Original titles distinguish orange juice versus with-pulp juice. With-pulp record contains both no_pulp and with_pulp and categorical_source_conflict:pulp. Gate already falls back.','Review the contradictory source rows; retain fallback rather than fabricate a confident veto.','explicit contradiction, existing review route appropriate'),
('5705010079224','5705010080176'):('Multiword flavor qualifier lost','Orange juice versus orange/blood-orange juice; both structured flavor sets contain only orange. Blood is not retained as part of a distinct blood-orange flavor.','Preserve blood orange as a phrase-level flavor and retain declared variant evidence.','explicit titles and descriptions'),
('868235000451','868235000475'):('Measurement/source contradictions, not a clear false rejection','Original titles say 7 fl oz while captured attributes say Volume:355. Both records have volume_sources_disagree and no_added_sugar_with_cane_sugar flags. Original judgments are still uncertain, not high.','Resolve contradictory measurements and claims before changing rejection policy.','conflicting sources; identity uncertain'),
('8711900018874','8711900018935'):('Ingredient/category bag obscures declared variant','Fruit drink versus strawberry 0% sugar. Both share many ingredients; coffee is introduced from category paths mentioning juice, coffee, tea. Strawberry is retained, but dilution by shared terms obscures it. Gate already falls back.','Separate declared flavor from ingredient inventory and broad category context.','confirmed category origin; explicit titles'),
('7313619000181','7313619001201'):('Recipe evidence differs; reporting incomplete','Both titles mention blueberry/blackcurrant, but one record retains only blueberry in the flavor set. One source ingredient set includes grape; the other carries 100% juice. Gate already falls back.','Preserve title plurals and recipe/juice-content evidence; review source reliability before deciding identity.','ambiguous product distinction; do not call a proven false match'),
('752697964546','758918255318'):('Threshold crossing, little evidence of extraction failure','Original titles are identical Proud Source 24×12 oz water. Processed scores 0.87/0.79 versus original 0.89/0.88 cross the strict both>0.8 threshold.','Treat as likely wording equivalence and threshold sensitivity, not a large semantic reversal.','strong surface match; no human ground truth'),
}

def main():
    paired=json.loads((ROOT/'jev/paired_comparison_6.json').read_text())
    states=json.loads((ROOT/'jev/input_states_6.json').read_text())
    changed=[]
    for pair in paired['pairs']:
        if pair['gate_data']['bucket']==pair['original_data']['bucket']:continue
        key=(pair['gtin1'],pair['gtin2']);category,observation,next_step,confidence=FINDINGS[key]
        records=[]
        for gtin in key:
            processed=states['gate_data'][gtin]
            records.append({'gtin':gtin,'titles':[x.get('title','') for x in states['original_data'][gtin]['listings']], 'processed':{k:processed.get(k) for k in ('canonical','mode_type','flavor_set','volume_set','pack_set','package_type_set','package_material_set','carbonation_set','pulp_set','sweetener_set','attribute_consistency_flags','universe_evidence')}})
        changed.append({**pair,'finding':category,'observation':observation,'next_step':next_step,'confidence':confidence,'records':records})
    contamination=[]
    for gtin in ('856472002055','8711900018874','860002364179'):
        for row in states['original_data'][gtin]['listings']:
            kwargs={k:row.get(column,'') for k,column in [('sku_name','title'),('attribute','attributes'),('description','description'),('url','url'),('image_url','image_url'),('category_path','category_path'),('category','category')]}
            full=extract_all(**kwargs)
            without=extract_all(**{**kwargs,'category_path':'','category':''})
            added=sorted(set(full['flavor_set'])-set(without['flavor_set']))
            if added:contamination.append({'gtin':gtin,'title':row.get('title',''),'category':row.get('category',''),'category_path':row.get('category_path',''),'full_flavors':sorted(full['flavor_set']),'without_category_flavors':sorted(without['flavor_set']),'added_by_category':added})
    both_low=[p for p in paired['pairs'] if p['gate']=='proceed' and all(p[scope]['bucket']=='low' for scope in ('gate_data','original_data'))]
    count=Counter()
    for p in both_low:
        a,b=[set(states['gate_data'][p[g]]['flavor_set']) for g in ('gtin1','gtin2')]
        count['empty_flavor_on_at_least_one_side' if not a or not b else 'equal_flavor_sets' if a==b else 'flavor_subset' if a<=b or b<=a else 'other_flavor_relationship']+=1
    result={'round':6,'changed_pair_count':len(changed),'changed_pairs':changed,'category_ablation':contamination,'both_low_approved_pair_count':len(both_low),'both_low_approved_flavor_patterns':dict(count),'both_low_approved_pairs':both_low,'limitations':'Source-based review and offline extraction ablation, not human-verified truth. No gate policy changed and no additional JEV calls made.'}
    (ROOT/'jev/inspection_6.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['Paired JEV round-6 inspection, 2026-10-02','', 'Inspected all 10 category changes and all approved pairs consistently low under both formats. No new live calls or gate changes.','', '| Pair | Gate | Processed scores | Original scores | Finding |','| --- | --- | --- | --- | --- |']
    for p in changed:
        fmt=lambda scope: ' / '.join(str(p[scope]['scores'][k]) for k in ('a_order','b_swapped'))
        lines.append(f"| {p['gtin1']} / {p['gtin2']} | {p['gate']} | {fmt('gate_data')} | {fmt('original_data')} | {p['finding']} |")
    lines+=['','The ten changes comprise six approved pairs, three fallback pairs, and one rejected pair. Two approved pairs look like wording-equivalent matches (Twix/Twiix and identical Proud Source titles), rather than extraction failures. The other four approvals expose spicy/plain variants, carbonation strength, blood-orange specificity, and category-contaminated flavor containment. Three fallback cases are already held for review; the rejected Rise case contains contradictory volume and sugar claims.','',f"Both-format low approvals: {len(both_low)}. Flavor relationships: {dict(count)}. These are not all lexical misses; missing evidence and permissive containment both contribute.",'','Offline category ablation reproduced category-created flavor tokens. Savia Original loses coconut when category inputs are removed; Wicky loses coffee. Taika Matcha remains classified as coffee in the saved processed snapshot, but this ablation did not isolate its source. This confirms the broad-category promotion path rather than merely assuming it from the title. Detailed before/after records are in inspection_6.json.','','Priority candidates for a future fix:','', '1. Stop broad category/breadcrumb terms becoming specific flavor identity evidence.','2. Separate declared product flavor/variant from ingredients and keep phrase-level distinctions (blood orange, matcha, plain/spicy, carbonation strength).','3. Route added specific flavor and unresolved measurement contradictions to review instead of approving on any subset/intersection.','4. Preserve readable source titles and source provenance so typo-equivalent matches remain possible.','','The evidence supports targeted investigation. It does not justify declaring every JEV disagreement a gate error or turning every missing field into a conflict.']
    (ROOT/'jev/INSPECTION_6.md').write_text('\n'.join(lines)+'\n')
    print('Inspected',len(changed),'changed pairs;',len(both_low),'both-low approvals:',dict(count))
    print('Category ablation added tokens:',[(x['gtin'],x['added_by_category']) for x in contamination])
if __name__=='__main__':main()
