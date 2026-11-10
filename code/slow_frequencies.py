"""Quantify Qwen's slow RoPE frequencies and regenerate the explanatory figure.

python code/slow_frequencies.py
Uses the posted model configuration; no model inference. The figure shows
unit-amplitude basis functions, not measured attention or model activations.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'data/analysis/qwen-config.json').read_text())
    d=cfg['hidden_size']//cfg['num_attention_heads'];base=cfg['rope_theta'];L=cfg['max_position_embeddings']
    omega=base**(-2*np.arange(d//2)/d);period=2*np.pi/omega;extent=(L-1)*omega
    assert (d,base,L)==(128,1_000_000,32768)
    assert np.flatnonzero(extent<.1).tolist()==list(range(59,64))
    np.testing.assert_allclose(omega[1:]/omega[:-1],base**(-2/d))
    # Lipschitz bound is uniform in the learned coefficient phase.
    delta=np.linspace(0,L-1,1024);phase=np.linspace(-np.pi,np.pi,19)
    for j in range(64):
        change=abs(np.cos(omega[j]*delta[:,None]+phase)-np.cos(phase))
        assert np.all(change<=omega[j]*delta[:,None]+1e-12)
    out={'model':'Qwen/Qwen2.5-1.5B-Instruct','config_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
         'config_source':'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json',
         'base':base,'head_dimension':d,'context_tokens':L,'max_displacement':L-1,
         'pairs':[{'pair':j,'omega':float(w),'period_tokens':float(period[j]),'rotation_over_window_radians':float(extent[j]),'rotation_over_256_tokens':float(256*w)} for j,w in enumerate(omega)],
         'pairs_under_one_cycle':int((extent<2*np.pi).sum()),'pairs_under_point_one_radian':int((extent<.1).sum())}
    (ROOT/'data/analysis/slow-frequencies.json').write_text(json.dumps(out,indent=2)+'\n')
    paper='#f6f5ef';plt.rcParams.update({'font.size':10,'figure.facecolor':paper,'axes.facecolor':paper,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(9,7),layout='constrained');grid=fig.add_gridspec(2,2)
    ax=fig.add_subplot(grid[0,:]);ax.semilogy(np.arange(64),period,color='#176452');ax.axhline(L,color='#bd682b',ls='--',label='32,768-token context')
    ax.set(xlabel='Pair index (0 = fastest)',ylabel='Period (tokens)',title='Qwen2.5-1.5B: B = 1,000,000; 64 RoPE pairs');ax.legend();ax.grid(alpha=.15)
    colors=plt.cm.viridis(np.linspace(.1,.85,5))
    for column,(fn,label) in enumerate([(np.cos,'cos(ωⱼΔ)'),(np.sin,'sin(ωⱼΔ)')]):
        ax=fig.add_subplot(grid[1,column])
        for j,c in zip(range(59,64),colors):ax.plot(delta,fn(omega[j]*delta),color=c,label=f'Pair {j}')
        ax.set(xlabel='Relative displacement Δ (tokens)',ylabel=label,title='Slowest five · unit-amplitude basis')
        ax.set_xticks([0,16384,32767],['0','16k','32k']);ax.grid(alpha=.15)
        if column:ax.legend(fontsize=8,ncol=2)
        else:ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.savefig(ROOT/'figures/analysis/slow-frequencies.png',dpi=170,bbox_inches='tight',pad_inches=.18);plt.close(fig)
    print(json.dumps({'last_pair':out['pairs'][-1],'pairs_under_point_one_radian':out['pairs_under_point_one_radian']}))
if __name__=='__main__':main()
