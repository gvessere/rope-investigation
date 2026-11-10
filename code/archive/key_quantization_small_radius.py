"""Variable phase bits gated by per-token pair amplitude; all-layer K quality + accounted storage."""
from pathlib import Path
import json,time
import numpy as np
import torch
from balanced_rope import loadmodel
R=Path(__file__).resolve().parents[1];P=R/'outputs/key_quantization_small_radius';P.mkdir(exist_ok=True)
cal=json.loads((R/'outputs/key_index_fresh_validation_captures/passages.json').read_text());val=json.loads((R/'outputs/query_prototype_key_index_long/passages.json').read_text())
# Freeze all codec specifications before evaluation. No fitting on validation keys.
configs={
 'raw_uniform_p8r6':dict(center='raw',lo=8,hi=8,rb=6,band='all',gate='raw'),
 'raw_small3_large8_r6':dict(center='raw',lo=3,hi=8,rb=6,band='all',gate='raw'),
 'raw_small3_large8_r8':dict(center='raw',lo=3,hi=8,rb=8,band='all',gate='raw'),
 'raw_slow_small3_large8_r6':dict(center='raw',lo=3,hi=8,rb=6,band='slow',gate='raw'),
 'mean_small3_large6_r6':dict(center='mean',lo=3,hi=6,rb=6,band='all',gate='residual'),
 'mean_small3_large6_r4':dict(center='mean',lo=3,hi=6,rb=4,band='all',gate='residual'),
 'mean_small4_large6_r6':dict(center='mean',lo=4,hi=6,rb=6,band='all',gate='residual'),
 'mean_small4_large6_r4':dict(center='mean',lo=4,hi=6,rb=4,band='all',gate='residual'),
 'mean_rawgate_small3_large6_r6':dict(center='mean',lo=3,hi=6,rb=6,band='all',gate='raw'),
}
for lp in [3,4]:
 for lr in [4,6]:configs[f'hybrid_smallp{lp}r{lr}_largep6r6']=dict(center='mean',lo=lp,hi=6,rb=6,lowrb=lr,band='all',gate='raw',hybrid=True)
(P/'locked_configs.json').write_text(json.dumps(configs,indent=2))
tok,model=loadmodel();state={'mode':'native','capture':False};hooks=[];collected={l:[] for l in range(28)}
cb=np.load(R/'outputs/key_quantization_protected_prefix/codebooks.npz')
centers={'mean':torch.tensor(cb['mean_center'],device='mps'),'raw':torch.zeros((28,2,128),device='mps')}
medians={'mean':torch.tensor(cb['mean_median'],device='mps')};maxima={'mean':torch.tensor(cb['mean_maximum'],device='mps')}
stats={name:dict(body_pairs=0,small_pairs=0,total_key_records=0,packed_body_bytes=0,prefix_bytes=0,clipped_radius_pairs=0,eligible_gate_pairs=0) for name in configs};examples={};checks={}
previous=json.loads((P/'results.json').read_text()) if (P/'results.json').exists() else {'results':{}}
for name,v in previous['results'].items():stats[name]=v['stats']

def quant(z,l,c,name):
 center=centers[c['center']][l][None,None];x=z[...,:64]-center[...,:64];y=z[...,64:]-center[...,64:];r=torch.sqrt(x*x+y*y);a=torch.atan2(y,x)
 gate=torch.sqrt(z[...,:64]**2+z[...,64:]**2) if c['gate']=='raw' else r
 eligible=torch.ones(64,dtype=torch.bool,device=z.device)
 if c['band']=='slow':eligible[:49]=False
 variable=c['lo']!=c['hi'];small=(gate<2.5)&eligible[None,None,None]
 if c.get('hybrid'):a=torch.where(small,torch.atan2(z[...,64:],z[...,:64]),a)
 pb=torch.where(small,c['lo'],c['hi']);levels=2.**pb;thcode=torch.round(a*levels/(2*np.pi)).to(torch.int64)%levels.to(torch.int64);aq=thcode.float()*(2*np.pi/levels)
 m=medians[c['center']][l][None,None];mx=maxima[c['center']][l][None,None];lm=torch.log1p(mx/m);nr=2**c['rb']-1
 rc=torch.round(torch.log1p(r/m)/lm*nr).clamp(0,nr).to(torch.int64);rq=m*torch.expm1(rc.float()/nr*lm)
 if c.get('hybrid'):
  lr=c['lowrb'];rawr=gate;lowcode=torch.round(rawr/2.5*(2**lr-1)).clamp(0,2**lr-1).to(torch.int64);rc=torch.where(small,lowcode,rc);rq=torch.where(small,lowcode.float()/(2**lr-1)*2.5,rq)
  cx=torch.where(small,0.,center[...,:64]);cy=torch.where(small,0.,center[...,64:])
 else:cx=center[...,:64];cy=center[...,64:]
 out=torch.cat((rq*aq.cos()+cx,rq*aq.sin()+cy),-1);out[:,:14]=z[:,:14]
 bodypb=pb[:,14:];nbits=bodypb.sum(-1)+(torch.where(small[:,14:],c.get('lowrb',c['rb']),c['rb']).sum(-1) if c.get('hybrid') else 64*c['rb'])+(int(eligible.sum().cpu()) if variable else 0)
 st=stats[name];st['packed_body_bytes']+=int(((nbits+7)//8).sum().cpu());st['total_key_records']+=bodypb.shape[0]*bodypb.shape[1]*bodypb.shape[2]
 st['prefix_bytes']+=z.shape[0]*min(14,z.shape[1])*2*512;st['body_pairs']+=bodypb.numel();st['small_pairs']+=int(small[:,14:].sum().cpu());st['eligible_gate_pairs']+=bodypb.shape[0]*bodypb.shape[1]*bodypb.shape[2]*int(eligible.sum().cpu());st['clipped_radius_pairs']+=int((r[:,14:]>mx).sum().cpu())
 if l==0 and state['row']==0:
  examples[name]=dict(phase=thcode[0,14:,0].cpu().numpy(),radius=rc[0,14:,0].cpu().numpy(),small=small[0,14:,0].cpu().numpy(),eligible=eligible.cpu().numpy(),pbits=pb[0,14:,0].cpu().numpy(),prefix=z[0,:14,0].cpu().numpy(),center=centers[c['center']][0,0].cpu().numpy(),median=medians[c['center']][0,0].cpu().numpy(),maximum=maxima[c['center']][0,0].cpu().numpy())
 return out

for l,layer in enumerate(model.model.layers):
 def hook(mod,args,out,l=l):
  z=out.reshape(out.shape[0],out.shape[1],2,128)
  if state['capture']:collected[l].append(z[0,14:].detach().cpu().numpy())
  if state['mode']=='native':return out
  return quant(z,l,configs[state['mode']],state['mode']).reshape_as(out)
 hooks.append(layer.self_attn.k_proj.register_forward_hook(hook))

def evaluate(row,name):
 ids=np.array(row['input_ids']);target=np.array(row['body'])[256:];x=torch.tensor(ids[None],device='mps');ix=torch.tensor(target-1,device='mps');gold=torch.tensor(ids[target],device='mps');t=time.perf_counter();out=model(input_ids=x,use_cache=False);logits=out.logits[0,ix].float();loss=torch.nn.functional.cross_entropy(logits,gold,reduction='none').cpu().numpy();pred=logits.argmax(-1).cpu().numpy()
 np.savez_compressed(P/f'{name}_doc{row["id"]}.npz',loss=loss,pred=pred,gold=ids[target]);del out,logits;torch.mps.empty_cache();print(json.dumps(dict(condition=name,document=row['id'],seconds=time.perf_counter()-t)),flush=True)

try:
 with torch.inference_mode():
  cp=P/'raw_body_codebooks.npz'
  if not cp.exists():
   state.update(capture=True,mode='native')
   for i,row in enumerate(cal):state['row']=i;evaluate(row,'calibration_native')
   rms=[];rx=[]
   for l in range(28):
    z=np.concatenate(collected[l]);r=np.hypot(z[...,:64],z[...,64:]);rms.append(np.maximum(np.median(r,axis=0),.01));rx.append(np.maximum(r.max(0)*1.05,.02))
   np.savez_compressed(cp,median=np.array(rms),maximum=np.array(rx));collected.clear()
  rawcb=np.load(cp);medians['raw']=torch.tensor(rawcb['median'],device='mps');maxima['raw']=torch.tensor(rawcb['maximum'],device='mps');state.update(capture=False,mode='native')
  for i,row in enumerate(val):state['row']=i;evaluate(row,'native')
  for name in configs:
   if name in previous['results']:continue
   state['mode']=name
   for i,row in enumerate(val):state['row']=i;evaluate(row,name)
finally:
 for h in hooks:h.remove()
# Pack a real layer0/KVhead0 cache per codec, including explicit flags, offsets, codebooks and FP32 prefix.
def pack_example(name,e):
 c=configs[name];buf=bytearray();offsets=[0];variable=c['lo']!=c['hi']
 for pc,rc,sm in zip(e['phase'],e['radius'],e['small']):
  accum=0;shift=0
  for j in range(64):
   if variable and e['eligible'][j]:accum|=int(sm[j])<<shift;shift+=1
   b=c['lo'] if sm[j] else c['hi'];accum|=int(pc[j])<<shift;shift+=b;accum|=int(rc[j])<<shift;shift+=(c.get('lowrb',c['rb']) if sm[j] and c.get('hybrid') else c['rb'])
  buf.extend(accum.to_bytes((shift+7)//8,'little'));offsets.append(len(buf))
 offsets=np.array(offsets,dtype='<u4');code=np.frombuffer(bytes(buf),dtype='uint8')
 # Decode codes and flags independently from the packed bytes, checking no hidden phase array is needed.
 for t in range(len(offsets)-1):
  accum=int.from_bytes(buf[offsets[t]:offsets[t+1]],'little');shift=0
  for j in range(64):
   small=False
   if variable and e['eligible'][j]:small=bool((accum>>shift)&1);shift+=1
   b=c['lo'] if small else c['hi'];pc=(accum>>shift)&((1<<b)-1);shift+=b;rb=(c.get('lowrb',c['rb']) if small and c.get('hybrid') else c['rb']);rc=(accum>>shift)&((1<<rb)-1);shift+=rb
   assert pc==e['phase'][t,j] and rc==e['radius'][t,j]
 codebook=np.concatenate([e['median'].ravel(),e['maximum'].ravel()]+([e['center'].ravel()] if c['center']=='mean' else [])).astype('<f4')
 rawprefix=e['prefix'].astype('<f4');dest=P/f'packed_example_{name}.bin'
 with dest.open('wb') as f:
  for arr in [rawprefix,codebook,offsets,code]:f.write(arr.tobytes())
 expected=rawprefix.nbytes+codebook.nbytes+offsets.nbytes+code.nbytes;assert dest.stat().st_size==expected
 return dict(body_keys=len(offsets)-1,bytes=expected,bytes_per_key=expected/(len(offsets)-1+14),codes_verified=True,layout='14 nativeFP32 keys; codebook; uint32 offsets; byte-aligned variable phase/radius records; global codec metadata in locked_configs.json')
packed={k:v['packed_example'] for k,v in previous['results'].items()};packed.update({k:pack_example(k,e) for k,e in examples.items()})
base=[np.load(P/f'native_doc{row["id"]}.npz') for row in val];results={}
for i,row in enumerate(val):
 old=np.load(R/'outputs/key_quantization_protected_prefix'/f'validation_native_doc{row["id"]}.npz');assert np.array_equal(base[i]['pred'],old['pred']);assert np.max(np.abs(base[i]['loss']-old['loss']))<1e-4
for name,c in configs.items():
 aa=[np.load(P/f'{name}_doc{row["id"]}.npz') for row in val];delta=np.concatenate([a['loss']-b['loss'] for a,b in zip(aa,base)]);changed=np.concatenate([a['pred']!=b['pred'] for a,b in zip(aa,base)])
 s=stats[name];# Four documents,56 caches each; offset array has Nbody+1 entries. Codebooks shared across requests.
 totalkeys=sum(len(row['input_ids'])*56 for row in val);offsetbytes=4*(s['total_key_records']+56*len(val));codebookbytes=(1024 if c['center']=='mean' else 512)*56*len(val)
 stored=s['packed_body_bytes']+s['prefix_bytes']+offsetbytes+codebookbytes
 results[name]=dict(tokens=len(delta),loss_increase=float(delta.mean()),changed_prediction_fraction=float(changed.mean()),per_document_loss_increase=[float((a['loss']-b['loss']).mean()) for a,b in zip(aa,base)],stored_K_bytes_per_key=stored/totalkeys,key_compression_vs_fp16=256/(stored/totalkeys),full_KV_compression_if_V_fp16=512/(256+stored/totalkeys),small_fraction=s['small_pairs']/s['body_pairs'],stats=s,packed_example=packed[name])
payload=dict(configs=configs,results=results,protocol=dict(threshold=2.5,prefix=14,phase='uniform circular quantization',radius='per-pair log-companded levels; separate frozen raw/centered codebooks fitted on4 calibration passages only',comparison='4 previously used validation excerpts,2062tokens each,7168next-token targets',storage='bit-aligned fields; each key record padded to byte; flag per eligible pair if variable phase bits; uint32 offset per key; 14FP32 prefix keys; codebooks charged once per full context/KVhead',scope='all28layers,2KVheads,all64pairs unless slowband specified; Q/V unquantized; all downstream states recomputed',caveat='Quality simulation plus real packed/decode-verified layer0KVhead0 examples, not a packed attention kernel or runtime benchmark. No entropy coding. Full KV ratio assumes VFP16.',checks='Native replay matches old reference; all packed examples decode back to their integer phase/radius codes exactly; no phase arrays retained inside .bin files.'))
(P/'results.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2),flush=True)
