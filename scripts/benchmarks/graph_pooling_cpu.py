"""Synthetic CPU diagnostic against pre-cache model; run from ER repository root.

PYTHONPATH=src .venv/bin/python scripts/benchmarks/graph_pooling_cpu.py
Requires repository history at the pinned baseline; never measures GPU speed.
"""
import subprocess,time,json,torch
from graph_tracks.data import RELATIONS,fit_vocabulary,tensorize
from graph_tracks.model import AttributeGNN
scope={};exec(subprocess.check_output(['git','show','5001027893ad8598e69bd2ab37693a2a981bf04e:src/graph_tracks/model.py'],text=True),scope)
torch.set_num_threads(2);torch.manual_seed(42)
records=[dict(product_id=str(i),split='train',numeric={'volume_ml':[330],'pack':[1]},attributes={r:[f'v{i%100}',f'v{(i+1)%100}'] for r in RELATIONS}) for i in range(1000)]
v=fit_vocabulary(records);batch=tensorize(records,v,'cpu');support=tensorize(records[:800],v,'cpu')
old=scope['AttributeGNN'](v);new=AttributeGNN(v);new.load_state_dict(old.state_dict())
def step(m):
 m.zero_grad(set_to_none=True); out=m.encode(batch,m.context(support));loss=out[:,0].sum();loss.backward();return out.detach()
a,b=step(old),step(new);torch.testing.assert_close(a,b,rtol=0,atol=0)
for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-7)
results={}
for name,m in [('baseline',old),('cached',new)]:
 for _ in range(5):step(m)
 times=[]
 for _ in range(30):
  t=time.perf_counter();step(m);times.append(time.perf_counter()-t)
 results[name]=sorted(times)[len(times)//2]
print(json.dumps({'median_step_seconds':results,'speedup':results['baseline']/results['cached'],'listings':1000,'support':800,'threads':2,'steps':30,'score_parity':'bit exact','gradient_parity':'rtol 1e-4 atol 1e-7','torch':str(torch.__version__)}))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:step(new)
print(p.key_averages().table(sort_by='self_cpu_time_total',row_limit=8))
