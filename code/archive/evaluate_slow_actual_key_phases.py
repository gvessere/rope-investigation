"""Actual slow K phases: force to zero or round; preserve amplitudes; no bias edit."""
from pathlib import Path
import json,time
import numpy as np
import torch
from balanced_rope import loadmodel
R=Path(__file__).resolve().parents[1];P=R/'outputs/slow_actual_key_phases';P.mkdir(exist_ok=True)
rows=json.loads((R/'outputs/query_prototype_key_index_long/passages.json').read_text())
tok,model=loadmodel();state={'name':'native'};hooks=[]
configs={'native':None,'identity':('identity',False,0),'zero_all':('zero',False,0),'zero_small':('zero',True,0),'pi2_all':('pi2',False,0),'pi4_all':('pi4',False,0),'zero_all_prefix14':('zero',False,14),'zero_small_prefix14':('zero',True,14),'pi2_all_prefix14':('pi2',False,14),'pi4_all_prefix14':('pi4',False,14)}
configs.update(pi2_small=('pi2',True,0),pi4_small=('pi4',True,0),pi2_small_prefix14=('pi2',True,14),pi4_small_prefix14=('pi4',True,14))
stats={k:{'selected':0,'eligible':0,'max_relative_radius_squared_error':0.,'max_unselected_error':0.} for k in configs}

if (P/'results.json').exists():
 stats.update(json.loads((P/'results.json').read_text())['checks'])

def hook(mod,args,output):
 name=state['name']
 if name=='native':return output
 kind,small,prefix=configs[name];z=output.reshape(output.shape[0],output.shape[1],2,128)
 x=z[...,49:64];y=z[...,113:128];r=torch.sqrt(x*x+y*y);a=torch.atan2(y,x)
 selected=torch.ones_like(r,dtype=torch.bool)
 if small:selected &= r<2.5
 selected[:,:prefix]=False
 if kind=='zero':nx=r;ny=torch.zeros_like(r)
 else:
  step=np.pi/2 if kind=='pi2' else np.pi/4
  th=a if kind=='identity' else torch.round(a/step)*step
  nx=r*th.cos();ny=r*th.sin()
 xx=torch.where(selected,nx,x);yy=torch.where(selected,ny,y)
 out=z.clone();out[...,49:64]=xx;out[...,113:128]=yy
 s=stats[name];s['selected']+=int(selected.sum().cpu());s['eligible']+=r.numel()
 s['max_relative_radius_squared_error']=max(s['max_relative_radius_squared_error'],float(((xx*xx+yy*yy-x*x-y*y).abs()/(1+x*x+y*y)).max().cpu()))
 s['max_unselected_error']=max(s['max_unselected_error'],float(torch.where(~selected,(xx-x).abs()+(yy-y).abs(),0).max().cpu()))
 assert torch.equal(out[...,:49],z[...,:49]);assert torch.equal(out[...,64:113],z[...,64:113])
 return out.reshape_as(output)
for layer in model.model.layers:hooks.append(layer.self_attn.k_proj.register_forward_hook(hook))
try:
 with torch.inference_mode():
  for name in configs:
   state['name']=name
   for row in rows:
    dest=P/f'{name}_doc{row["id"]}.npz'
    if dest.exists():continue
    ids=np.array(row['input_ids']);target=np.array(row['body'])[256:];x=torch.tensor(ids[None],device='mps');ix=torch.tensor(target-1,device='mps');gold=torch.tensor(ids[target],device='mps');t=time.perf_counter()
    out=model(input_ids=x,use_cache=False);logits=out.logits[0,ix].float();loss=torch.nn.functional.cross_entropy(logits,gold,reduction='none').cpu().numpy();pred=logits.argmax(-1).cpu().numpy()
    np.savez_compressed(dest,loss=loss,pred=pred,gold=ids[target]);del out,logits;torch.mps.empty_cache()
    print(json.dumps(dict(condition=name,document=row['id'],seconds=time.perf_counter()-t)),flush=True)
finally:
 for h in hooks:h.remove()
results={}
base=[np.load(P/f'native_doc{r["id"]}.npz') for r in rows]
for name in configs:
 aa=[np.load(P/f'{name}_doc{r["id"]}.npz') for r in rows];diff=np.concatenate([a['loss']-b['loss'] for a,b in zip(aa,base)]);changed=np.concatenate([a['pred']!=b['pred'] for a,b in zip(aa,base)])
 results[name]=dict(tokens=len(diff),loss_increase=float(diff.mean()),changed_prediction_fraction=float(changed.mean()),per_document_loss_increase=[float((a['loss']-b['loss']).mean()) for a,b in zip(aa,base)],selected_fraction=stats[name]['selected']/max(1,stats[name]['eligible']))
for i,row in enumerate(rows):
 ref=np.load(R/'outputs/key_quantization_protected_prefix'/f'validation_native_doc{row["id"]}.npz');assert np.array_equal(base[i]['pred'],ref['pred']);assert np.max(np.abs(base[i]['loss']-ref['loss']))<1e-4
assert results['identity']['changed_prediction_fraction']==0;assert abs(results['identity']['loss_increase'])<1e-4
for s in stats.values():assert s['max_unselected_error']==0;assert s['max_relative_radius_squared_error']<2e-6
payload=dict(results=results,checks=stats,protocol=dict(model='Qwen/Qwen2.5-1.5B-Instruct',band='pairs49..63 (15 of64); Qwen split-half pair=(j,j+64)',point='actual k_proj activations including learned bias, before RoPE; no centering or bias subtraction',zero='(x,y)->(sqrt(x*x+y*y),0); amplitude preserved, phase set to angle0',small='select current activation radius<2.5, not bias radius',prefix='unsuffixed: all positions; suffix prefix14: first14 native keys protected',scope='all28 layers, bothKVheads; Q/V unmodified; full downstream recomputation',data='4 previously used long book excerpts,2062inputtokens each,7168targets total; not an untouched final test',checks='Native replay matches previous native run; identity polar roundtrip changes no predictions; outside-band/unselected components unchanged; radii preserved',limitation='No packed storage or speed test; phase intervention before RoPE does not remove the positional rotation.'))
(P/'results.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2),flush=True)
