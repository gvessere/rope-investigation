"""Recreate the article's observations and quantization figures from posted data.

Run: python code/observations.py
No inference or model download is required. Quantization plots summarize recorded
counterfactual results; they do not constitute a fresh model evaluation.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/observations';A=ROOT/'data/analysis';OUT=ROOT/'figures/analysis'
PAPER='#f6f5ef';INK='#172e2b';GREEN='#176452'
plt.rcParams.update({'font.family':['DejaVu Sans','Arial','sans-serif'],'font.size':10,'figure.facecolor':PAPER,'axes.facecolor':PAPER,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'rope-observations'})
angles=sorted(set(Fraction(j,4) for j in range(9))|set(Fraction(j,3) for j in range(7)))
labels=['0' if f==0 else ('π' if f.numerator==1 else f'{f.numerator}π')+('' if f.denominator==1 else f'/{f.denominator}') for f in angles]

def load(kind,layer):
 with np.load(DATA/f'{kind}_layer_{layer:02d}.npz') as z:return z['amplitude'].astype('float64'),z['phase'].astype('float64')

def guides(ax):
 for f in angles:ax.axvline(float(f)*np.pi,color='.60',ls=':',lw=.7,zorder=3,alpha=.55)
 ax.set_xticks([float(f)*np.pi for f in angles],labels,rotation=45,ha='right',fontsize=8)
 ax.set_xlim(0,2*np.pi)

def save(fig,name):
 fig.savefig(OUT/name,dpi=170,bbox_inches='tight',pad_inches=.2,metadata={'Date':None} if name.endswith('.svg') else None)
 plt.close(fig)

def main():
 OUT.mkdir(exist_ok=True)
 stats={};bias=np.load(A/'projection-biases.npz');rows=json.loads((DATA/'passages.json').read_text());trainids={0,1,4,5,8,9,12,13}
 for kind in ['queries','keys']:
  counts=np.zeros(3,dtype='int64');total=0;energies=[];constant_errors=[];bias_errors=[];trainamps=[]
  train=[];test=[];offset=0
  for row in rows:
   n=len(row['queries'] if kind=='queries' else row['body']);(train if row['id'] in trainids else test).extend(range(offset,offset+n));offset+=n
  for l in range(28):
   a,p=load(kind,l);total+=a.size
   counts+=np.array([np.count_nonzero(a<t) for t in (1,2.5,5)])
   z=a*np.exp(1j*p);tr=z[:,train];te=z[:,test];mean=tr.mean(1)
   trainamps.append(np.abs(tr).mean(1));energies.append(np.mean(abs(te)**2,axis=1))
   constant_errors.append(np.mean(abs(te-mean[:,None,:])**2,axis=1))
   bias_errors.append(np.mean(abs(te-bias[kind][l,:,None,:])**2,axis=1))
  e=np.stack(energies);ce=np.stack(constant_errors);be=np.stack(bias_errors);amp=np.stack(trainamps);large=amp>5
  stats[kind]={'points':int(total),'fractions_below':{str(t):int(n)/total for t,n in zip((1,2.5,5),counts)},'counts_below':{str(t):int(n) for t,n in zip((1,2.5,5),counts)},'large_mean_pairs':int(large.sum()),'large_bias_energy_fit':float(1-be[large].sum()/e[large].sum()),'large_constant_energy_fit':float(1-ce[large].sum()/e[large].sum())}
  ref=json.loads((A/'original-static-summary.json').read_text())[kind]['bands'][-1]
  np.testing.assert_allclose(stats[kind]['large_bias_energy_fit'],ref['share_heldout_energy_explained_by_projection_bias_alone'],rtol=1e-7)

 # A specific observed head, restricted transparently to the low-r region.
 a,p=load('keys',2);r=a[0].ravel();phi=p[0].ravel();mask=r<2.5
 stats['low_r_example']={'layer':2,'head':0,'points':int(r.size),'retained':int(mask.sum()),'fraction':float(mask.mean()),'r_window':[0,2.5]}
 fig,axs=plt.subplots(2,1,figsize=(9,7.5),sharex=True)
 pair=np.tile(np.arange(64),a.shape[1]);axs[0].scatter(phi[mask],r[mask],c=pair[mask],cmap='turbo',vmin=0,vmax=63,s=.55,alpha=.22,rasterized=True,linewidths=0)
 axs[0].set_title('L2 KV0 → query heads 0–5 · saved pre-RoPE pairs',loc='left')
 bins=[np.linspace(0,2*np.pi,145),np.linspace(0,2.5,81)]
 counts,xe,ye=np.histogram2d(phi[mask],r[mask],bins=bins)
 mesh=axs[1].pcolormesh(xe,ye,np.ma.masked_equal(counts.T,0),norm=LogNorm(vmin=1,vmax=max(2,counts.max())),cmap='YlGnBu',shading='auto',rasterized=True)
 axs[1].set_title('The same points as density counts · logarithmic color scale',loc='left')
 fig.subplots_adjust(left=.10,right=.84,bottom=.12,top=.94,hspace=.25)
 cb=fig.colorbar(mesh,cax=fig.add_axes([.87,.12,.025,.364]));cb.set_label('Points per bin')
 for ax in axs:guides(ax);ax.set_ylim(0,2.5);ax.set_ylabel('Radius r')
 axs[-1].set_xlabel('Pre-RoPE phase φ (radians)')
 save(fig,'low-r-L2-KV0.png')
 np.savez_compressed(A/'L2-KV0-density.npz',counts=counts,phase_edges=xe,radius_edges=ye)

 # One strongly bias-dominated cloud and a counterexample: do not generalize to all.
 fig,axs=plt.subplots(1,2,figsize=(10,5))
 examples=[]
 for ax,(l,h,j) in zip(axs,[(13,4,52),(15,7,63)]):
  a,p=load('queries',l);z=a[h,:,j]*np.exp(1j*p[h,:,j]);b=complex(bias['queries'][l,h,j]);fit=1-float(np.mean(abs(z-b)**2)/np.mean(abs(z)**2))
  ax.scatter(z.real,z.imag,s=6,alpha=.25,color=GREEN,label='768 query observations')
  ax.scatter([b.real],[b.imag],marker='x',s=100,lw=2,color='#be6338',label='Learned bias bQ,j')
  ax.scatter([0],[0],marker='+',s=35,color='.5');ax.axhline(0,lw=.5,color='.8');ax.axvline(0,lw=.5,color='.8')
  ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('x before RoPE');ax.set_ylabel('y before RoPE')
  ax.set_title(f'L{l} H{h}, pair {j}\n'+('Cloud near its bias' if fit>0 else 'Large r, but bias alone is insufficient'))
  examples.append({'layer':l,'head':h,'pair':j,'bias_real':b.real,'bias_imag':b.imag,'mean_real':float(z.real.mean()),'mean_imag':float(z.imag.mean()),'bias_energy_fit_all_tokens':fit})
 axs[0].legend(loc='best',fontsize=8);fig.tight_layout()
 save(fig,'bias-clouds.png');stats['bias_examples']=examples

 # Measured counterfactual runs, not a simulated quality estimate.
 results=json.loads((A/'slow-phase-results.json').read_text())['results'];names=['zero_small','pi2_small','pi4_small']
 fig,axs=plt.subplots(1,2,figsize=(9,4.2))
 for ax,field,factor,title in [(axs[0],'loss_increase',1,'Mean loss increase (nats/token)'),(axs[1],'changed_prediction_fraction',100,'Next-token choices changed (%)')]:
  vals=[results[n][field]*factor for n in names];ax.bar(range(3),vals,color=['#be6338','#8d9e78',GREEN]);ax.set_xticks(range(3),['0 bits\nphase → 0','2 bits\nnearest π/2','3 bits\nnearest π/4']);ax.set_title(title,loc='left',fontsize=11);ax.set_ylim(0,max(vals)*1.22)
  for i,v in enumerate(vals):ax.text(i,v+max(vals)*.04,f'{v:.4f}' if field=='loss_increase' else f'{v:.2f}%',ha='center',fontsize=9)
 fig.tight_layout();save(fig,'phase-rounding.png')
 (A/'observations-summary.json').write_text(json.dumps(stats,indent=2)+'\n')
 print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
