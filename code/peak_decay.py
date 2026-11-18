"""Recompute peak-frequency/height statistics and redraw the recorded L7 H6 plot.
Run: python code/peak_decay.py. Uses exact posted per-query bin summaries.
The full-softmax attention values were measured previously, not rerun here.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data/decay'
meta=json.loads((D/'peak-decay-provenance.json').read_text())
assert hashlib.sha256((D/'peak-decay-L7H6.npz').read_bytes()).hexdigest()==meta['published_summary_sha256']
z=np.load(D/'peak-decay-L7H6.npz')
c=z['counts'];m=z['peak_mass'];t=z['total_mass'];bins=[tuple(map(int,row)) for row in z['bins']]
assert c.shape==m.shape==t.shape==(2560,6)
assert len(np.unique(z['document_ids']))==20
assert np.all(c>0) and np.all(m>=0) and np.all(t>=m)
x=np.array([np.exp(np.log(np.arange(s,t)).mean()) for s,t in bins])
width=np.array([t-s for s,t in bins]);nq=len(c)
density=c.sum(0)/(nq*width);height=m.sum(0)/c.sum(0)
contrib=m.sum(0)/(nq*width);mean=t.sum(0)/(nq*width);share=m.sum(0)/t.sum(0)
np.testing.assert_allclose(density*height,contrib)
dr=density[-1]/density[0];hr=height[0]/height[-1];mr=mean[0]/mean[-1]
reference=json.loads((D/'peak-decay-reference.json').read_text())
for name,actual in [('peak_density',density),('mean_peak_height',height),('mean_attention_all_tokens',mean),('fraction_of_mass_at_peak_apexes',share)]:
 np.testing.assert_allclose(actual,reference[name],rtol=1e-12,atol=1e-15)
# Keep the original 20-document, 128-query grouping for the bootstrap.
assert np.all(z['document_ids'].reshape(20,128)==z['document_ids'][::128,None])
rng=np.random.default_rng(2026111718);bc=np.stack([np.bincount(d,minlength=20) for d in rng.integers(0,20,(2000,20))])
dc=c.reshape(20,128,-1).sum(1);dm=m.reshape(20,128,-1).sum(1);dt=t.reshape(20,128,-1).sum(1)
cb=np.einsum('bd,dk->bk',bc,dc,optimize=False);mb=np.einsum('bd,dk->bk',bc,dm,optimize=False);tb=np.einsum('bd,dk->bk',bc,dt,optimize=False)
db=cb/(nq*width);hb=mb/cb;ab=tb/(nq*width)
fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
for vals,label,color,style in [(density,'How often a token is a local peak','#217daf','-'),(height,'Mean height of those peaks','#db7c23','-'),(mean,'Mean attention across ALL tokens','#273447','--')]:
    ax.plot(x,vals/vals[0],style+'o',ms=6,lw=2.5,color=color,label=label)
ax.axhline(1,color='#cbd1d7',lw=.8)
ax.text(x[-1]*1.1,dr,f'{dr:.2f}×\nas frequent',va='center',color='#217daf',fontsize=11)
ax.annotate(f'Peaks: {hr:.0f}× weaker\nAll-token mean: {mr:.0f}× smaller',xy=(x[-1],height[-1]/height[0]),xytext=(x[2],.04),arrowprops=dict(arrowstyle='->',color='#737c86'),fontsize=11,color='#273447')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(18,1300);ax.set_ylim(.008,1.6)
ax.set_xticks([24,48,96,192,384,768]);ax.set_xticklabels(['24','48','96','192','384','768'])
ax.set_yticks([1,.1,.01]);ax.set_yticklabels(['1 (reference)','1/10','1/100'])
ax.set_xlabel('Distance before the query (tokens; log scale)',fontsize=11)
ax.set_ylabel('Relative to the 16–31-token interval',fontsize=11)
ax.grid(alpha=.16);ax.legend(loc='lower left',fontsize=10)
ax.set_title('L7 H6: local peaks stay about as frequent, but become smaller',fontsize=15,pad=17)
fig.text(.5,-.035,'Every local maximum is counted, including small ones. Same 20 texts and 2,560 queries.\nEach point averages one doubling-distance bin; no attention is renormalized within bins.',ha='center',fontsize=10,color='#52616d')
fig.savefig(ROOT/'figures/analysis/peak-decay-L7H6.png',dpi=180,bbox_inches='tight');plt.close(fig)
result=dict(head='L7 H6',bins_inclusive=[[s,t-1] for s,t in bins],distance_centers=x.tolist(),peak_density=density.tolist(),mean_peak_height=height.tolist(),peak_token_contribution=contrib.tolist(),mean_attention_all_tokens=mean.tolist(),fraction_of_mass_at_peak_apexes=share.tolist(),far_to_near_peak_density_ratio=float(dr),near_to_far_mean_peak_height_ratio=float(hr),near_to_far_all_token_mean_ratio=float(mr),bootstrap_95CI=dict(density_ratio=np.quantile(db[:,-1]/db[:,0],[.025,.975]).tolist(),height_drop_factor=np.quantile(hb[:,0]/hb[:,-1],[.025,.975]).tolist(),all_token_mean_drop_factor=np.quantile(ab[:,0]/ab[:,-1],[.025,.975]).tolist()),definition='All local maxima, no height/rank selection: p[d]>p[d-1] and p[d]>=p[d+1]. Peak density is peak count/(queries*bin width). Mean height is sum of weights at peak apex tokens/peak count. Their product is the mean contribution at peak tokens. Alltoken mean includes peak flanks and troughs too. Native full-softmax probabilities; same20heldoutdocs,128latequeries/doc. Bins16..1022; age1023 used as neighbour. Each plotted series divided by its own value in the16..31reference bin only, to compare proportional changes. Original attention is never locally renormalized.',interpretation='For all local maxima, peak density is nearly constant across near/far bins; mean peak height drops similarly to total mean attention. Peak apex mass share near55.3%,far56.1%, so this is a substantial, not exhaustive, mass accounting. This does not contradict rarity of top8 or >=1% peaks, which is a different selection; no causalRoPEmechanism proven.')
(D/'peak-decay-recomputed.json').write_text(json.dumps(result,indent=2))
