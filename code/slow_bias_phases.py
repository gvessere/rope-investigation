"""Regenerate the original slow-pair Q/K bias-phase view, before any rounding.
Run: python code/slow_bias_phases.py. Uses the posted checkpoint bias values.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from observations import PAPER,OUT,save
ROOT=Path(__file__).resolve().parents[1]
def main():
    OUT.mkdir(exist_ok=True)
    source=np.load(ROOT/'data/analysis/projection-biases.npz')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    norm=Normalize(49,63);stats={}
    ticks=np.arange(9)*np.pi/4
    labels=['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π']
    for ax,kind in zip(axes,['queries','keys']):
        b=source[kind][...,49:64].astype(np.complex128)
        phase=np.angle(b)%(2*np.pi);radius=abs(b);pairs=np.broadcast_to(np.arange(49,64),b.shape)
        points=ax.scatter(phase.ravel(),radius.ravel(),c=pairs.ravel(),cmap='turbo',norm=norm,s=9,alpha=.65,linewidths=0)
        for i,t in enumerate(ticks):ax.axvline(t,color='.45' if i%2==0 else '.7',ls=':',lw=1 if i%2==0 else .7,zorder=0)
        ax.set(xticks=ticks,xticklabels=labels,xlim=(-.06,2*np.pi+.06),ylim=(0,float(radius.max())*1.06),xlabel='Pre-RoPE bias phase (radians)',ylabel='Bias radius r',title=('Query biases' if kind=='queries' else 'Key biases')+f' · {b.size:,} pairs')
        ax.tick_params(axis='x',rotation=45,labelsize=8)
        stats[kind]=dict(bias_pairs=int(b.size),inclusive_pair_indices=[49,63],maximum_radius=float(radius.max()))
        for spacing,name in [(np.pi/2,'pi_over_2'),(np.pi/4,'pi_over_4')]:
            distance=abs(np.angle(np.exp(1j*(phase-spacing*np.round(phase/spacing)))))
            stats[kind][name]=dict(within_five_degrees_fraction=float(np.mean(distance<np.deg2rad(5))))
    fig.colorbar(points,ax=list(axes),label='Frequency-pair index',ticks=[49,52,56,60,63],shrink=.78,pad=.03)
    fig.suptitle('Learned slow-pair bias phases · before rounding',fontsize=13)
    save(fig,'slow-bias-phases.png')
    (ROOT/'data/analysis/slow-bias-phases.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
