"""Rank and exhaust local evidence for the frozen residual cohort; no web calls."""
from __future__ import annotations
import json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
from core.project_root import find_project_root

ROOT=find_project_root(Path(__file__))
sys.path.insert(0,str(ROOT/'src'))
from core.common import AUDIT_FINDINGS_DIR
from core.critical_attributes import extract_critical_claims
from core.identity_policy import resolve_listing_row
from core.text import attribute_fields


def main():
 out=AUDIT_FINDINGS_DIR
 cases=json.loads((out/'residual_cases.json').read_text())
 df=pd.read_csv(ROOT/'dataset.csv',dtype=str,keep_default_na=False)
 rows={r['sku_id']:r for r in df.to_dict('records')}
 target=set()
 row_cases=defaultdict(list)
 for c in cases:
  for side in ['left','right']:
   target.add(c[side]['sku_id']);row_cases[c[side]['sku_id']].append(c['id'])
   target.update(r['sku_id'] for r in c['siblings'][side])
 groups=defaultdict(list)
 for c in cases:
  if c['kind']=='same_gtin':
   for d in c['descriptor_conflicts']:
    vals=tuple(sorted((tuple(c['left_dimensions'].get(d,[])),tuple(c['right_dimensions'].get(d,[])))))
    groups[(d,vals)].append(c)
 ranks=[]
 for (dim,values),cc in sorted(groups.items(),key=lambda item:-len(item[1])):
  ranks.append({'dimension':dim,'values':values,'pairs':len(cc),'gtins':len({c['left']['gtin'] for c in cc}),
                'case_ids':[c['id'] for c in cc],'retailer_pairs':dict(Counter(' / '.join(sorted([c['left']['retailer'],c['right']['retailer']])) for c in cc))})
 agreement=[]
 for field in ['sku_url','image_url','description_short_eng']:
  for c in cases:
   a,b=c['left'],c['right']
   if c['kind']=='different_gtin' and a[field] and a[field]==b[field]:
    agreement.append({'case_id':c['id'],'field':field,'value':a[field],
                      'sku_ids':[a['sku_id'],b['sku_id']],'gtins':[a['gtin'],b['gtin']],
                      'sibling_candidate_conflicts':c['sibling_candidate_conflicts'],
                      'action':'shared_content_is_not_identity'})
 candidates=[]; direct=[]
 material_patterns={'Glass':r'\bglass\s+bottles?\b', 'Plastic':r'\bplastic\s+bottles?\b|\bPET\b',
                    'Metal':r'\b(?:alumin(?:um|ium)|metal)\s+(?:bottles?|cans?)\b'}
 for sku in sorted(target):
  raw=rows[sku];r=resolve_listing_row(raw)
  attrs=dict(attribute_fields(r['attribute']))
  title=extract_critical_claims(r['sku_name_eng'])['carbonation']
  desc=extract_critical_claims(r['description_short_eng'])['carbonation']
  declared=extract_critical_claims('',r['attribute'])['carbonation']
  evidence={'sku_id':sku,'gtin':r['gtin'],'title':r['sku_name_eng'],'retailer':r['retailer'],
            'title_carbonation':sorted(title),'description_carbonation':sorted(desc),
            'attribute_carbonation':sorted(declared),'cases':row_cases[sku]}
  if title or desc:direct.append(evidence)
  if len(title)==1 and len(declared)==1 and title.isdisjoint(declared):
   candidates.append({**evidence,'field':'Carbonization','proposed_value':next(iter(title)),
      'source':'sku_name_eng','evidence':r['sku_name_eng'],'description_conflicts':bool(desc and title.isdisjoint(desc))})
  material=[]
  for value,pattern in material_patterns.items():
   m=re.search(pattern,r['sku_name_eng'],re.I if value!='Plastic' else 0)
   if m:material.append((value,m[0]))
  raw_material=next((v for k,v in attrs.items() if k=='pack material type'),'')
  if len(material)==1 and raw_material and material[0][0].casefold() not in str(raw_material).casefold():
   candidates.append({**evidence,'field':'Pack Material Type','proposed_value':material[0][0],
      'source':'sku_name_eng','evidence':material[0][1],'raw_value':raw_material,
      'action':'review_inner_outer_scope_before_repair'})
 inventory=[]
 for p in sorted((ROOT/'dashboard/evidence/identity').glob('*.json')):
  data=p.read_text(); refs=sorted(target & set(re.findall(r'(?<!\d)\d+(?!\d)',data)))
  info=p.stat()
  # structural census, never a byte length as the identity (owner directive
  # 2026-10-08): two different files of equal size would alias
  inventory.append({'path':str(p.relative_to(ROOT)),'size':info.st_size,'mtime_ns':info.st_mtime_ns,
                    'referenced_target_skus':refs})
 result={'dataset_size':ByteCount((ROOT/'dataset.csv').read_bytes()).total,
   'residual_cases':len(cases),'distinct_target_and_sibling_rows':len(target),
   'disagreement_ranking':ranks,'false_agreements':agreement,'direct_source_claims':direct,
   'repair_candidates':candidates,'prior_evidence_inventory':inventory}
 (out/'local_evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Local evidence audit — ranked agreements and disagreements','','Local-only investigation of every frozen residual and its original same-GTIN siblings. These rankings identify evidence to review; repeated rows are not independent votes.','','## Disagreements','','| Dimension | Values | Pairs | Distinct GTINs |','|---|---|---:|---:|']
 for x in ranks:lines.append(f"| {x['dimension']} | {x['values']} | {x['pairs']} | {x['gtins']} |")
 lines+=['','## Apparent agreements that cannot authorize merges','']
 for field,n in Counter(x['field'] for x in agreement).items():lines.append(f'- Identical {field}: {n} different-GTIN residual pairs.')
 lines+=['','## Direct source contradiction candidates','']
 for c in candidates:lines += [f"- SKU {c['sku_id']}, GTIN {c['gtin']}: {c['field']} → {c['proposed_value']}; {c['source']}: {c['evidence']}. Cases: {', '.join(c['cases']) or 'sibling only'}. Status: requires review."]
 lines+=['','## Existing local investigation coverage','']
 for x in inventory:lines.append(f"- `{x['path']}`: {len(x['referenced_target_skus'])} relevant SKU references.")
 (out/'LOCAL_EVIDENCE_AUDIT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'target_rows':len(target),'candidates':len(candidates),'by_field':dict(Counter(x['field'] for x in candidates)),
  'agreements':dict(Counter(x['field'] for x in agreement))},indent=2),flush=True)

if __name__=='__main__': main()
