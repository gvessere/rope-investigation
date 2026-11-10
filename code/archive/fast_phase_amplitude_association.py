"""Phase concentration by amplitude, with within-head/token permutation controls."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];P=R/'outputs/fast_phase_amplitude_association';P.mkdir(exist_ok=True);edges=np.linspace(0,2*np.pi,73);centers=(edges[:-1]+edges[1:])/2;rng=np.random.default_rng(20261027);summary={}
colors=['#5488bb','#999999','#d77d3c'];labels=['r < 1','1 ≤ r ≤ 5','r > 5']
for kind,source in [('queries',R/'outputs/query_amplitude_phase_scatter/points.npz'),('keys',R/'outputs/query_key_amplitude_phase_y40/key_points.npz')]:
 with np.load(source) as f:a=f['amplitude'][...,0:16];ph=f['phase'][...,0:16]
 H=a.shape[0]*a.shape[1];T=a.shape[2];hist=np.zeros((16,3,72),int);pairrows=[];totals={}
 # Concentration measured within each layer/head/pair, never across arbitrary head phase origins.
 resultants={b:[] for b in range(3)};permuted=[];observed=[];qualifying=[]
 for jj,j in enumerate(range(0,16)):
  aa=a[...,jj];pp=ph[...,jj];masks=[aa<1,(aa>=1)&(aa<=5),aa>5]
  for b,m in enumerate(masks):hist[jj,b]=np.histogram(pp[m],bins=edges)[0]
  row={'pair':j,'bands':[]}
  for b,m in enumerate(masks):
   counts=hist[jj,b];prob=counts/counts.sum();row['bands'].append({'band':labels[b],'count':int(counts.sum()),'largest_phase_bin_percent':float(prob.max()*100),'circular_resultant_pooled':float(abs(np.exp(1j*pp[m]).mean()))})
  pairrows.append(row)
  aa=aa.reshape(H,T);pp=pp.reshape(H,T);zz=np.exp(1j*pp)
  for b,m in enumerate(masks):
   mm=m.reshape(H,T);n=mm.sum(1);res=abs((zz*mm).sum(1))/np.maximum(n,1);resultants[b].extend(res[n>=20].tolist())
  # For each head/pair with >=20 large and >=20 nonlarge observations,
  # compare large-vector phase concentration with phase-shuffled amplitude labels.
  mm=aa>5;n=mm.sum(1);valid=(n>=20)&((T-n)>=20)
  if valid.any():
   z=zz[valid];m=mm[valid];nn=n[valid];obs=abs((z*m).sum(1))/nn
   null=np.stack([abs((np.take_along_axis(z,np.argsort(rng.random(z.shape),axis=1),axis=1)*m).sum(1))/nn for _ in range(40)])
   observed.extend(obs.tolist());permuted.extend(null.mean(0).tolist());qualifying.extend([[j,int(h)] for h in np.flatnonzero(valid)])
 pct=hist/np.maximum(hist.sum(2,keepdims=True),1)*100
 fig,axs=plt.subplots(4,4,figsize=(16,13))
 for jj,ax in enumerate(axs.flat):
  for b in [1,0,2]:ax.stairs(pct[jj,b],edges,color=colors[b],lw=1.4,label=labels[b])
  ax.set_xlim(0,2*np.pi);ax.set_xticks([0,np.pi,2*np.pi],['0','π','2π']);ax.set_title(f'Pair {jj} · large n={hist[jj,2].sum():,}',fontsize=10);ax.set_xlabel('Phase before RoPE (rad)');ax.set_ylabel('% within amplitude group');ax.tick_params(labelsize=7)
 handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.5,.952),ncol=3)
 fig.suptitle(f'Qwen {kind} · do fast-pair phase peaks come from large amplitudes?',fontsize=14);fig.text(.5,.015,'Each curve separately normalized to 100%; counts pooled across heads/layers. 72 bins from 0 to 2π.\nOrange = large amplitude; blue = small amplitude; gray = intermediate. Y-axis fitted per panel.',ha='center',fontsize=9);fig.subplots_adjust(top=.88,bottom=.10,left=.055,right=.99,wspace=.4,hspace=.65);fig.savefig(P/f'{kind}_phase_by_amplitude.png',dpi=160);fig.savefig(P/f'{kind}_phase_by_amplitude.pdf');plt.close(fig)
 obs=np.array(observed);nul=np.array(permuted);diff=obs-nul
 summary[kind]={'pairs':pairrows,'within_head_pair_large_phase_concentration':{'eligible_head_pairs':len(obs),'eligibility':'at least 20 tokens with r>5 and 20 with r<=5','mean_observed_resultant':float(obs.mean()),'mean_permuted_resultant':float(nul.mean()),'mean_excess_concentration':float(diff.mean()),'fraction_with_positive_excess':float((diff>0).mean()),'null':'40 shuffles of phase-to-amplitude correspondence within each same layer/head/pair; preserves amplitude and phase marginals','resultant_definition':'abs(mean(exp(i*phase))); 0=spread around circle, 1=same angle'}}
 np.savez_compressed(P/f'{kind}_histograms.npz',counts=hist,percent=pct,edges=edges,observed_resultants=obs,permuted_resultants=nul,eligible_head_pairs=qualifying)
(P/'summary.json').write_text(json.dumps(summary,indent=2));D=Path('/Users/geryvessere/Documents/Codex/2026-10-20/i/outputs/fast_phase_amplitude_association');D.mkdir(parents=True,exist_ok=True)
for f in P.iterdir():shutil.copy2(f,D/f.name)
shutil.copy2(__file__,D/Path(__file__).name)
print(json.dumps({k:v['within_head_pair_large_phase_concentration'] for k,v in summary.items()},indent=2))
