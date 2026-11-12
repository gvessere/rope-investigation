"""Regenerate per-prompt phase/radius contours from original captured losses.
python code/phase_sensitivity.py
No model inference. Each panel compares a one-pair activation intervention with
its own native prompt baseline; axes are relative to that prompt's native pair.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from figures import PAPER
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/sensitivity';OUT=ROOT/'figures/analysis'
def main():
 for f,v in json.loads((D/'provenance.json').read_text()).items():assert hashlib.sha256((D/f).read_bytes()).hexdigest()==v['sha256']
 docs=[0,4];meta={doc:json.loads((D/f'doc{doc}-summary.json').read_text()) for doc in docs};summary={}
 for key in ['Q_high2','K_low1','K_high1']:
  points=[meta[doc]['points'][key] for doc in docs];assert len({(p['kind'],p['layer'],p['head'],p['pair']) for p in points})==1
  grid=meta[0]['protocol']['grid'];x=np.array(grid['phase_offsets']);y=np.array(grid['radius_multipliers']);fields=[]
  for doc in docs:
   assert meta[doc]['protocol']['grid']==grid
   with np.load(D/f'doc{doc}-{key}.npz') as z:
    np.testing.assert_allclose(z['delta'].mean(-1),z['next16_mean'],atol=1e-7)
    a=z['delta'].astype(float).mean(-1);fields.append(a)
  mean=(fields[0]+fields[1])/2;limit=max(float(np.abs(fields).max()),1e-8)
  fig,axes=plt.subplots(1,3,figsize=(10,3.7),layout='constrained',sharex=True,sharey=True)
  titles=['Alice in Wonderland','Sherlock Holmes','Mean of these two prompts']
  for ax,z,title in zip(axes,[*fields,mean],titles):
   im=ax.pcolormesh(x,y,z,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='nearest',rasterized=True)
   ax.contour(x,y,z,levels=np.linspace(-limit,limit,11),colors='#555555',linewidths=.45,alpha=.55)
   if z.min()<0<z.max():ax.contour(x,y,z,levels=[0],colors='black',linewidths=1)
   ax.plot(0,1,'o',mfc='white',mec='black',ms=5)
   ax.set(title=title,xlabel='Phase change from native',xticks=[-np.pi/4,0,np.pi/4],xticklabels=['−π/4','0','π/4'],ylim=(0,2));ax.tick_params(labelsize=9)
  axes[0].set_ylabel('Radius / native radius')
  p=points[0];fig.suptitle(f'{p["kind"]} · layer {p["layer"]}, head {p["head"]}, pair {p["pair"]} · same intervention, different prompts',fontsize=12)
  cb=fig.colorbar(im,ax=list(axes),shrink=.82,pad=.02);cb.set_label('Mean next-16-token loss change\n(nats/token)',fontsize=9);cb.ax.tick_params(labelsize=8);cb.formatter.set_powerlimits((-2,2));cb.update_ticks()
  out=OUT/f'phase-sensitivity-{key}.png';fig.savefig(out,dpi=160);plt.close(fig)
  opposed=fields[0]*fields[1]<0
  summary[key]=dict(kind=p['kind'],layer=p['layer'],head=p['head'],pair=p['pair'],native_radii=[q['radius'] for q in points],grid_points=int(mean.size),opposite_sign_grid_points=int(opposed.sum()),range_by_prompt=[[float(z.min()),float(z.max())] for z in fields],maximum_absolute_mean_delta=float(abs(mean).max()))
  # Find an illustrative cancellation in the original grid, not a fresh evaluation.
  candidates=opposed&(abs(fields[0])>1e-4)&(abs(fields[1])>1e-4)
  if candidates.any():
   ratio=np.where(candidates,abs(fields[0]+fields[1])/(abs(fields[0])+abs(fields[1])+1e-30),np.inf);i,j=np.unravel_index(ratio.argmin(),ratio.shape)
   summary[key]['illustrative_cancellation']=dict(phase_offset=float(x[j]),radius_multiplier=float(y[i]),prompt_deltas=[float(z[i,j]) for z in fields],mean_delta=float(mean[i,j]),selection='Post-hoc example: smallest relative two-prompt mean among opposite-sign effects each exceeding 1e-4 nats/token.')
 (D/'recomputed-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
