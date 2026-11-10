"""Measure inverse-distance decay at matched near-end query positions, native and uniform99 runs."""
from pathlib import Path
import json,time,shutil,csv
import numpy as np
import torch
from balanced_rope import loadmodel
from transformers.models.qwen2 import modeling_qwen2 as mq
R=Path(__file__).resolve().parents[1];P=R/'outputs/inverse_distance_attention';P.mkdir(exist_ok=True);rows=json.loads((R/'outputs/query_prototype_key_index_long/passages.json').read_text());reference=json.loads((R/'outputs/adaptive_head_mass_cutoffs/results.json').read_text());BINS=[(2**j,2**(j+1)-1) for j in range(3,10)];modes=['dense','uniform99'];todo=[(mode,ri) for mode in modes for ri in range(4) if not (P/f'{mode}_{ri}.npz').exists()]
if todo:
 _,model=loadmodel();original=mq.eager_attention_forward;state={}
 def capture(l,raw,kept,start):
  qs=state['positions'];ii=np.flatnonzero((qs>=start)&(qs<start+raw.shape[-2]))
  if not len(ii):return
  raw=raw[0,:,torch.tensor(qs[ii]-start,device=raw.device),:].cpu().numpy().astype(float)
  chosen=raw if kept is None else kept[0,:,torch.tensor(qs[ii]-start,device=kept.device),:].cpu().numpy().astype(float)
  qpos=qs[ii];qidx=np.arange(len(ii))[:,None]
  for bi,(lo,hi) in enumerate(BINS):
   keys=qpos[:,None]-np.arange(lo,hi+1)[None];state['raw_bins'][l,:,ii,bi]=raw[:,qidx,keys].sum(-1).T;state['kept_bins'][l,:,ii,bi]=chosen[:,qidx,keys].sum(-1).T
  state['raw_self'][l,:,ii]=raw[:,np.arange(len(ii)),qpos].T;state['kept_self'][l,:,ii]=chosen[:,np.arange(len(ii)),qpos].T;state['retained_mass'][l,:,ii]=chosen.sum(-1).T
  assert np.allclose(raw.sum(-1),1,atol=1e-6)
 def wrap(module,q,k,v,attention_mask,scaling,**kw):
  out,w=original(module,q,k,v,attention_mask,scaling,**kw);l=module.layer_idx
  if state['mode']=='dense':capture(l,w,None,0);return out,w
  vv=mq.repeat_kv(v,module.num_key_value_groups);chunks=[]
  for start in range(0,q.shape[-2],128):
   ww=w[:,:,start:start+128];vals,ix=ww.sort(-1,descending=True);used=(vals.cumsum(-1)-vals<.99)&(vals>0);kept=torch.zeros_like(ww).scatter(-1,ix,vals*used);mass=kept.sum(-1,keepdim=True);capture(l,ww,kept,start);chunks.append(torch.matmul(kept/mass,vv))
  return torch.cat(chunks,dim=2).transpose(1,2).contiguous(),None
 mq.eager_attention_forward=wrap
 try:
  with torch.inference_mode():
   for mode,ri in todo:
    row=rows[ri];ids=np.array(row['input_ids']);N=len(ids);positions=np.arange(N-129,N-1);state.update(mode=mode,positions=positions,raw_bins=np.zeros((28,12,128,7)),kept_bins=np.zeros((28,12,128,7)),raw_self=np.zeros((28,12,128)),kept_self=np.zeros((28,12,128)),retained_mass=np.zeros((28,12,128)));t=time.perf_counter();x=torch.tensor(ids[None],device='mps');out=model(input_ids=x,use_cache=False);target=np.array(row['body'])[256:];ix=torch.tensor(target-1,device='mps');gold=torch.tensor(ids[target],device='mps');z=out.logits[0,ix].float();pred=z.argmax(-1).cpu().numpy();nll=float(torch.nn.functional.cross_entropy(z,gold));ref=next(r for r in reference if r['split']=='validation' and r['passage']==ri and r['policy']==('dense' if mode=='dense' else 'uniform_99'));assert np.array_equal(pred,ref['predictions']);assert abs(nll-ref['mean_nll'])<1e-5
    assert (state['retained_mass']>= (.99-1e-6 if mode=='uniform99' else 1-1e-6)).all();assert (state['kept_bins'].sum(-1)+state['kept_self']<=1+1e-6).all();np.savez_compressed(P/f'{mode}_{ri}.npz',**{k:state[k] for k in ['positions','raw_bins','kept_bins','raw_self','kept_self','retained_mass']});print(json.dumps({'mode':mode,'passage':ri,'seconds':time.perf_counter()-t,'baseline_predictions_match':True,'nll':nll}),flush=True);del out,z
 finally:mq.eager_attention_forward=original
# Fit mean per-key weights in seven complete distance octaves. x = geometric mean age.
width=np.array([hi-lo+1 for lo,hi in BINS]);x=np.array([np.exp(np.log(np.arange(lo,hi+1)).mean()) for lo,hi in BINS]);lx=np.log(x);xc=lx-lx.mean();harmonic=np.array([(1/np.arange(lo,hi+1)).sum() for lo,hi in BINS]);data={};fitrecords=[]
def fit(masses):
 weights=masses/width;y=np.log(np.maximum(weights,1e-300));slope=(y*xc).sum(-1)/(xc*xc).sum();intercept=y.mean(-1)-slope*lx.mean();pred=intercept[...,None]+slope[...,None]*lx;ss=((y-y.mean(-1,keepdims=True))**2).sum(-1);r2=1-((y-pred)**2).sum(-1)/np.maximum(ss,1e-30);alpha=-slope;eligible=(masses.sum(-1)>=.01)&(masses>0).all(-1);approx=eligible&(alpha>=.8)&(alpha<=1.2)&(r2>=.8);return alpha,r2,eligible,approx,intercept
summary={}
for mode in modes:
 caps=[np.load(P/f'{mode}_{ri}.npz') for ri in range(4)];M=np.stack([c['kept_bins'] for c in caps]);mean=M.mean((0,3));alpha,r2,eligible,approx,intercept=fit(mean);pa,pr,pe,pg,_=fit(M.mean(3));qa,qr,qe,qg,_=fit(M);data[mode]={'M':M,'mean':mean,'alpha':alpha,'r2':r2,'eligible':eligible,'approx':approx,'per_text_alpha':pa,'per_text_r2':pr,'intercept':intercept};eligible_alpha=alpha[eligible]
 # Amount of attention in octaves must be flat, up to the exact harmonic sum, under1/d.
 normalized=mean/harmonic;cv=normalized.std(-1)/np.maximum(normalized.mean(-1),1e-300)
 for l in range(28):
  for h in range(12):fitrecords.append({'mode':mode,'layer':l,'head':h,'mass_in_fit_range':float(mean[l,h].sum()),'exponent_alpha':float(alpha[l,h]),'log_log_r_squared':float(r2[l,h]),'eligible_at_least1percent_mass':bool(eligible[l,h]),'approximately_inverse_distance':bool(approx[l,h]),'harmonic_corrected_octave_mass_cv':float(cv[l,h]),'per_text_alpha':pa[:,l,h].tolist(),'per_text_r_squared':pr[:,l,h].tolist(),'texts_matching_inverse_distance':int(pg[:,l,h].sum())})
 summary[mode]={'eligible_heads':int(eligible.sum()),'excluded_low_mass_or_zero_bin_heads':int((~eligible).sum()),'approximately_inverse_distance_heads':int(approx.sum()),'fraction_of_eligible_approximately_inverse_distance':float(approx.sum()/eligible.sum()),'eligible_alpha_percentiles':{str(p):float(np.percentile(eligible_alpha,p)) for p in [0,10,25,50,75,90,100]},'eligible_median_r_squared':float(np.median(r2[eligible])),'eligible_heads_alpha_below0p8':int((eligible&(alpha<.8)).sum()),'eligible_heads_alpha_above1p2':int((eligible&(alpha>1.2)).sum()),'matching_all_four_texts':int((approx&(pg.sum(0)==4)).sum()),'matching_at_least3_texts':int((approx&(pg.sum(0)>=3)).sum()),'per_query_eligible_cases':int(qe.sum()),'per_query_matching_cases':int(qg.sum()),'per_query_matching_fraction':float(qg.sum()/qe.sum())}
 print('SUMMARY',mode,json.dumps(summary[mode]),flush=True)
# Correct the old all-query pooled plots for distance availability, using exactly the same bins.
oldcaps=[np.load(R/f'outputs/retained_mass99_head_locations/passage_{i}.npz') for i in range(4)];oldmass=sum(c['age_kept_mass'][0] for c in oldcaps);avail=sum(c['age_available'][0] for c in oldcaps);old_binmass=np.stack([oldmass[...,lo:hi+1].sum(-1) for lo,hi in BINS],-1);binavail=np.array([avail[lo:hi+1].sum() for lo,hi in BINS]);old_conditional_perkey=old_binmass/binavail;old_unadjusted_perkey=old_binmass/(sum(int(c['nqueries'][0]) for c in oldcaps)*width);olda,oldr,_,_,_=fit(old_conditional_perkey*width);unadj,_,_,_,_=fit(old_unadjusted_perkey*width);summary['old_plot_exposure_effect']={'median_exponent_before_availability_correction':float(np.median(unadj[data['uniform99']['eligible']])),'median_exponent_after_availability_correction':float(np.median(olda[data['uniform99']['eligible']]))}
with (P/'head_decay_fits.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(fitrecords[0]));w.writeheader();w.writerows(fitrecords)
np.savez_compressed(P/'fit_arrays.npz',bin_lo=np.array([b[0] for b in BINS]),bin_hi=np.array([b[1] for b in BINS]),geometric_mean_age=x,**{f'{mode}_{key}':d[key] for mode,d in data.items() for key in ['mean','alpha','r2','eligible','approx','per_text_alpha','per_text_r2']})
# Select illustrative fits by a declared reproducible rule, alongside the previously discussed L11H6.
d=data['dense'];candidate=[r for r in fitrecords if r['mode']=='dense' and r['approximately_inverse_distance']];candidate.sort(key=lambda r:(-r['texts_matching_inverse_distance'],abs(r['exponent_alpha']-1),-r['log_log_r_squared']));example=(candidate[0]['layer'],candidate[0]['head']) if candidate else tuple(np.unravel_index(np.argmin(np.where(d['eligible'],abs(d['alpha']-1),np.inf)),(28,12)));summary['example_rule']='Choose a pooled eligible dense head matching inverse-distance criteria; prioritize number of matching individual texts, then closeness of alpha to1. Exploratory illustration, not independent confirmation.';summary['example_head']=list(example)
protocol={'model':'Qwen2.5-1.5B-Instruct, all336 query heads','data':'Same four separate ~2k-token prose excerpts used in uniform99 gallery. Every model run reevaluated; predictions and loss match prior dense/uniform99 baselines.','query_control':'Last128 scored queries per text, positionsN−129..N−2. Every query has every tested age available; no prefix key lies inside the fitted age range.','range':'Seven complete bins8–15,16–31,32–63,64–127,128–255,256–511,512–1023. Self and very nearest tokens excluded; far edge beyond1023 excluded to keep full octaves and equal exposure.','fit':'OLS of log(mean attention per available key) vs log geometric mean token age in each bin. Mean per-key weight=bin attention mass/(queries * number of keys per bin). w(d) proportional d^(−alpha). alpha1 corresponds approximately equal mass per doubling of distance; alpha0 is flat weight per key. R² measures fit in log space, not causal variance explained.','criteria':'Descriptive approximate1/d: at least1% of head attention mass lies in8..1023, all7bins positive, alpha in[.8,1.2], log-spaceR²>=.8. Thresholds chosen before inspecting fits; not a formal statistical power-law test.','modes':'dense: native model attention. uniform99: all heads/layers keep99% mass and propagate modified outputs; measured selected weights BEFORE final restricted-softmax renormalization, matching previous plot. Dense is the primary test to avoid trimming creating an apparent law.','limits':'Averages over four texts and128late queries per text. Head means can conceal query-dependent patterns; per-query fit counts supplied. No general1/d law inferred from pooled fit alone.','old_plot_check':'All-query pooled histograms additionally fitted after division by available query-key counts in each bin; this corrects unequal exposure but does not match query context lengths, hence secondary only.'}
summary['protocol']=protocol;(P/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4.6));cols={'dense':'#367caf','uniform99':'#c9843e'}
for mode in modes:
 z=data[mode];axs[0].hist(z['alpha'][z['eligible']],bins=np.linspace(-1,3,33),histtype='step',lw=1.8,label=f"{mode} ({z['eligible'].sum()} eligible heads)",color=cols[mode])
axs[0].axvspan(.8,1.2,color='gray',alpha=.12);axs[0].axvline(1,color='black',ls='--',lw=1);axs[0].set_xlabel('Decay exponent α (1 means1/d)');axs[0].set_ylabel('Number of heads');axs[0].legend(fontsize=8);axs[0].set_title('Distribution across heads\nHistogram shown over −1≤α≤3')
for ax,(l,h),title in zip(axs[1:],[example,(11,6)],['Exploratory example nearest1/d','Previously discussed head']):
 vals=data['dense']['mean'][l,h]/width
 for ri in range(4):ax.plot(x,data['dense']['M'][ri,l,h].mean(0)/width,color='#9db9d0',alpha=.65,lw=.9)
 ax.plot(x,vals,'o-',label='Native dense: four-text mean',color=cols['dense']);ax.plot(x,data['uniform99']['mean'][l,h]/width,'s--',label='99% retained',color=cols['uniform99']);constant=np.exp(np.mean(np.log(np.maximum(vals,1e-300))+np.log(x)));ax.plot(x,constant/x,':',color='black',label='1/d reference, fitted scale');ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Distance into past (tokens)');ax.set_ylabel('Mean attention weight per key');ax.set_title(f"{title}: L{l} H{h}\nα={d['alpha'][l,h]:.2f}, R²={d['r2'][l,h]:.2f}");ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Does attention fall approximately as1/d?\nMatched late queries; all tested distances available to every query',fontsize=13);fig.tight_layout(rect=[0,.01,1,.89]);fig.savefig(P/'inverse_distance_measurement.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,9),sharey=True)
for ax,mode in zip(axs,modes):
 z=data[mode];a=np.ma.masked_where(~z['eligible'],z['alpha']);im=ax.imshow(a,vmin=0,vmax=2,cmap='coolwarm',aspect='auto',interpolation='nearest');ax.set_xticks(range(12));ax.set_yticks(range(28));ax.set_xlabel('Head');ax.set_title(mode+' decay exponent');fig.colorbar(im,ax=ax,label='α: 0 flat; 1 inverse-distance; 2 steeper')
axs[0].set_ylabel('Layer');fig.suptitle('Per-head distance decay; white = <1% mass in tested range or zero bin');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(P/'exponent_head_maps.png',dpi=170);plt.close(fig)
page='''<!doctype html><html><meta charset="utf-8"><title>Measured inverse-distance attention decay</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px}img{width:100%}p{line-height:1.5}pre{white-space:pre-wrap}</style><h1>Does attention decay as1/d?</h1><p>Every tested distance is available to every sampled query. Seven complete distance octaves from8 through1023, last128scored queries of each of four2k-token excerpts, all336heads. Native dense attention is compared with the99%-retained model.</p><img src="inverse_distance_measurement.png"><img src="exponent_head_maps.png"><p>For each head, divide the attention mass in a distance bin by its number of keys. Fit log(weight per key) against log(distance): slope−1 corresponds to1/d. A near1exponent alone is insufficient; the curve must fit well and carry meaningful mass.</p><p><a href="head_decay_fits.csv">All head fits</a> · <a href="summary.json">Full protocol and results</a> · <a href="../retained_mass99_head_locations/log_distance_mass/index.html">Original log-distance gallery</a></p><h2>Summary</h2><pre>'''+json.dumps(summary,indent=2)+'</pre></html>';(P/'index.html').write_text(page)
D=Path('/Users/geryvessere/Documents/Codex/2026-10-20/i/outputs/inverse_distance_attention');D.mkdir(exist_ok=True)
for f in P.iterdir():shutil.copy2(f,D/f.name)
shutil.copy2(__file__,D/Path(__file__).name);print('COMPLETE',str(P/'index.html'),flush=True)
