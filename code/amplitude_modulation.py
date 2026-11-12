"""Construct a fixed-phase RoPE example whose score peak moves with amplitude.
Run: python code/amplitude_modulation.py
No model data or fitting is used. All plotted values are saved alongside the SVG.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from figures import frequencies, PAPER, GREEN, ORANGE
ROOT=Path(__file__).resolve().parents[1]
def main():
    delta=np.linspace(0,128,2049)
    pairs=np.array([27,28]);omega=frequencies()[pairs]
    centers=np.array([40.,100.]);phase=-omega*centers
    waves=np.cos(delta[:,None]*omega+phase)
    gains=np.array([0.,.5,1.]);amplitudes=np.column_stack([1-gains,gains])
    scores=amplitudes[:,0,None]*waves[None,:,0]+amplitudes[:,1,None]*waves[None,:,1]
    peaks=delta[scores.argmax(axis=1)]
    assert np.all(np.diff(peaks)>0)
    assert abs(peaks[0]-40)<.1 and abs(peaks[-1]-100)<.1
    # Only nonnegative pair amplitudes vary; both coefficient phases stay fixed.
    np.testing.assert_allclose(scores[0],waves[:,0])
    np.testing.assert_allclose(scores[-1],waves[:,1])
    fig,(ax,bars)=plt.subplots(2,1,figsize=(8.2,5.4),gridspec_kw={'height_ratios':[2.4,1]},layout='constrained')
    colors=[ORANGE,'#7586a0',GREEN]
    for i,(g,c) in enumerate(zip(gains,colors)):
        ax.plot(delta,scores[i],color=c,lw=2,label=f'g = {g:g} · peak at {peaks[i]:.1f}')
        ax.plot(peaks[i],scores[i].max(),'o',color=c,ms=5)
    ax.set(xlim=(0,128),xlabel='Displacement Δ (tokens into the past)',ylabel='Pair-sum score',title='Fixed phases, changing amplitudes')
    ax.legend(fontsize=9,loc='lower center',ncol=1);ax.grid(alpha=.12)
    x=np.arange(3)
    bars.bar(x-.16,amplitudes[:,0],width=.3,color=ORANGE,label='Pair 27 · peak at 40')
    bars.bar(x+.16,amplitudes[:,1],width=.3,color=GREEN,label='Pair 28 · peak at 100')
    bars.set(xticks=x,xticklabels=['g = 0','g = 0.5','g = 1'],ylabel='Amplitude',ylim=(0,1.3))
    bars.legend(fontsize=8,ncol=2,loc='upper center')
    fig.savefig(ROOT/'figures/amplitude-modulation.svg',metadata={'Date':None});plt.close(fig)
    np.savez_compressed(ROOT/'figures/amplitude-modulation.npz',delta=delta,pairs=pairs,omega=omega,phase=phase,gains=gains,amplitudes=amplitudes,scores=scores)
    result=dict(description='Constructed fixed-phase two-pair score example, not measured model behavior',base=10000,head_dimension=128,pairs=pairs.tolist(),omega=omega.tolist(),fixed_phase=phase.tolist(),gains=gains.tolist(),peak_displacements=peaks.tolist())
    (ROOT/'figures/amplitude-modulation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
