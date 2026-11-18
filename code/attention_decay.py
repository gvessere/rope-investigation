"""Replot original binned attention measurements and verify inverse-distance fits.
Run: python code/attention_decay.py. No model inference or downloads.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures import PAPER
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/decay'
def main():
 plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
 for name,sha in json.loads((D/'provenance.json').read_text())['files'].items():
  assert hashlib.sha256((D/name).read_bytes()).hexdigest()==sha
 z=np.load(D/'summary_arrays.npz');age=z['geometric_mean_age'];width=z['bin_width'];lx=np.log(age);xc=lx-lx.mean();out={}
 for split in ['discovery','heldout']:
  mass=z[split+'_mass'];logw=np.log(np.maximum(mass/width,1e-300));slope=(logw*xc).sum(-1)/(xc@xc);pred=logw.mean(-1,keepdims=True)+slope[:,None]*xc
  r2=1-((logw-pred)**2).sum(-1)/np.maximum(((logw-logw.mean(-1,keepdims=True))**2).sum(-1),1e-30)
  eligible=(mass.sum(-1)>=.01)&(mass>0).all(-1);near=eligible&(-slope>=.8)&(-slope<=1.2)&(r2>=.8)
  np.testing.assert_allclose(-slope,z[split+'_alpha'],atol=1e-10);np.testing.assert_array_equal(near,z[split+'_near1d'])
  out[split]={'eligible_heads':int(eligible.sum()),'near_inverse_distance_heads':int(near.sum())}
 out['replicated_heads']=int((z['discovery_near1d']&z['heldout_near1d']).sum())
 fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');colors=['#126552','#c26632','#7788a6'];selected=[]
 for (layer,head),color in zip([(2,5),(7,6),(15,6)],colors):
  i=layer*12+head;mass=z['heldout_mass'][i];conditional=mass/mass.sum();alpha=float(z['heldout_alpha'][i]);label=f'L{layer} H{head} · λ={alpha:.2f}'
  ax.loglog(age,conditional/width,'o-',color=color,label=label)
  selected.append({'layer':layer,'head':head,'heldout_exponent':alpha,'heldout_r_squared':float(z['heldout_r2'][i])})
 ref=(1/age);ref*=np.exp(np.log((z['heldout_mass'][29]/z['heldout_mass'][29].sum()/width)/ref).mean())
 ax.loglog(age,ref,'--',color='#555555',label='1/n reference')
 ax.set(xlabel='Distance n (tokens into the past)',ylabel='Mean weight per key (conditional)',title='Mean weight per key')
 ax.grid(alpha=.15);ax.legend(fontsize=8)
 dest=ROOT/'figures/analysis/attention-decay.png';fig.savefig(dest,dpi=160);plt.close(fig)

 # Counterexamples use the same held-out per-key means, without fitting a decay law.
 fig,axes=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True,layout='constrained')
 counterexamples=[]
 for ax,(layer,head),color,title in zip(axes,[(6,7),(6,9)],['#126552','#c26632'],['Broadly flat across most of the range','Higher at the far end than the near end']):
  i=layer*12+head
  per_key=z['heldout_mass'][i]/width
  relative=per_key/per_key[0]
  reference=age[0]/age
  ax.loglog(age,relative,'o-',color=color,lw=2,label='Measured per-key average')
  ax.loglog(age,reference,'--',color='#666666',label='1/d reference')
  ax.set(title=f'L{layer} H{head}\n{title}',xlim=(8,1024),ylim=(.01,4),xlabel='Look-back distance d (tokens)')
  ax.set_xticks([8,32,128,512],[str(v) for v in [8,32,128,512]])
  ax.set_yticks([.01,.1,1],[str(v) for v in [.01,.1,1]])
  ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower left')
  counterexamples.append({'layer':layer,'head':head,'mean_weight_per_key':per_key.tolist(),'relative_to_first_bin':relative.tolist(),'far_to_near_ratio':float(relative[-1])})
 axes[0].set_ylabel('Relative mean weight per key\n(first distance interval = 1)')
 fig.savefig(ROOT/'figures/analysis/attention-nondecay.png',dpi=160);plt.close(fig)
 out['counterexample_heads']=counterexamples

 out['plotted_heads']=selected;(D/'recomputed-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
