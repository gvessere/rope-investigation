"""Regenerate EVERY displayed blog figure from the distributed numeric data.

Original PNGs remain untouched. Replots go to generated_figures/ at matching
relative paths. Plot data and parameters are the archived ones; raster bytes
may vary with Matplotlib, fonts, and operating system.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated_figures'
MANIFEST=[]


def load(case,name):
    p=ROOT/'data'/case/name
    return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p)


def save(fig,case,name,inputs,recipe):
    p=OUT/case/name;p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=160)
    MANIFEST.append(dict(original='assets/figures/'+case+'/'+name,
                         regenerated=str(p.relative_to(ROOT)),inputs=inputs,
                         code='code/figures.py',function=recipe))


def frequency_schedule():
    case='rope_frequency_schedule'
    a=np.loadtxt(ROOT/'data'/case/'frequency_schedule.csv',delimiter=',',skiprows=1)
    j,omega,gap=a.T
    np.testing.assert_allclose(omega,1e6**(-j/64),rtol=1e-12)
    fig,ax=plt.subplots(1,2,figsize=(14,5.6),layout='constrained')
    ax[0].plot(j,omega,'o-',ms=3,label='Qwen: base 1,000,000',color='#2967a5')
    ax[0].plot(j,1e4**(-j/64),'--',label='For comparison: base 10,000',color='gray')
    ax[0].set(title='Frequency decreases geometrically',xlabel='Rotary pair index (zero-based)',ylabel='Angular frequency (radians/token)',yscale='log')
    ax[0].legend(fontsize=9)
    ax[1].plot(j[:-1],gap[:-1],'o-',ms=3,label='Gap between pair j and pair j+1',color='#2967a5')
    ax[1].axhline(2*np.pi/32768,ls='--',color='#be672f',label='DFT spacing: 2π / 32,768')
    cross=32
    ax[1].axvline(cross,ls=':',color='gray')
    ax[1].scatter([cross],[gap[cross]],color='#be672f',s=45,zorder=3)
    ax[1].set(title='High frequencies have wider absolute gaps',xlabel='First pair index j',ylabel='Neighbor frequency gap (radians/token)',yscale='log')
    ax[1].legend(fontsize=9)
    for aa in ax:aa.grid(alpha=.2)
    fig.suptitle('Qwen2.5-1.5B-Instruct: 64 RoPE frequencies, ωⱼ = 1,000,000⁻ʲ/⁶⁴\nVertical axes use logarithmic scales')
    save(fig,case,'frequency_schedule.png',[f'data/{case}/frequency_schedule.csv'],'frequency_schedule');plt.close(fig)


def range_constructions():
    case='rope_post_examples';a=load(case,'vectors_and_profiles.npz');meta=load(case,'constructions.json')
    w,d=a['omega'],a['distances'];q,k=a['native_query'],a['native_keys'];target=meta['target'];foils=[i for i in range(4) if i!=target]
    basis=np.exp(1j*w[:,None]*d);scale=1/np.sqrt(128)
    synth=lambda c,b:np.einsum('kj,jd->kd',c,b,optimize=False).real
    native_coeff=q[None]*k.conj()*scale
    native_fast=synth(native_coeff[:,:32],basis[:32]);native_slow=synth(native_coeff[:,32:],basis[32:])
    dw=np.zeros(64);dw[0]=(w[0]-w[1])/2;dw[-1]=(w[-2]-w[-1])/2;dw[1:-1]=(w[:-2]-w[2:])/2
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');full_profiles=[]
    for i,(definition,ax) in enumerate(zip(meta['examples'],axes.flat)):
        sigma=definition['fwhm']/(2*np.sqrt(2*np.log(2)))
        B=dw*np.exp(-.5*(sigma*w)**2)*sum(np.exp(-1j*w*c) for c in definition['centers'])/len(definition['centers']);B[32:]=0
        if definition['kind']=='negative':B=-B
        qe=q.copy();qe[:32]=np.sqrt(128)*B[:32]/k[target,:32].conj();qe[:32]*=np.linalg.norm(q[:32])/np.linalg.norm(qe[:32])
        np.testing.assert_allclose(qe,a[f'query_{i}'],rtol=1e-12,atol=1e-12)
        coeff=qe[None]*k.conj()*scale;fast=synth(coeff[:,:32],basis[:32]);full=synth(coeff,basis);full_profiles.append(full)
        np.testing.assert_allclose(full,a[f'full_{i}'],atol=1e-12)
        for left,right in definition['ranges']:ax.axvspan(left,right,color='red' if definition['kind']=='negative' else 'orange',alpha=.1)
        ax.plot(d,native_fast[target],ls=':',color='gray',alpha=.75,label='Unmodified query, range-band term')
        ax.plot(d,fast[foils].mean(0),color='darkorange',ls='--',label='Constructed query, mean other-key term')
        ax.plot(d,fast[target],color='navy',lw=2,label='Constructed query, matching-key term');ax.axhline(0,color='gray',lw=.6)
        ax.set(title=definition['name'],xlabel='Hypothetical upstream distance (tokens)',ylabel='Pairs 0–31 score contribution');ax.grid(alpha=.15)
        if i==0:ax.legend(fontsize=7,loc='upper right')
    fig.suptitle('Four range functions built against actual Qwen key vectors\nEngineered queries; native query norm and slow content component retained; linear axes',fontsize=12)
    inputs=[f'data/{case}/vectors_and_profiles.npz',f'data/{case}/constructions.json']
    save(fig,case,'range_constructions.png',inputs,'range_constructions');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for i,label in enumerate(meta['labels']):
        color='navy' if i==target else ['darkorange','green','purple','brown'][i];style='-' if i==target else '--'
        ax[0].plot(d,native_slow[i],color=color,ls=style,label=label+(' (red round target)' if i==target else ''))
        ax[1].plot(d,full_profiles[0][i],color=color,ls=style,label=label+(' (target)' if i==target else ''))
    ax[0].set(title='Unmodified slow component, pairs 32–63',xlabel='Hypothetical upstream distance',ylabel='Score contribution')
    ax[1].axvspan(10,20,color='orange',alpha=.12);ax[1].set(title='Slow content component + constructed local window',xlabel='Hypothetical upstream distance',ylabel='Full 64-pair attention logit')
    for aa in ax:aa.legend(fontsize=8);aa.grid(alpha=.15)
    fig.suptitle('Actual keys from a correctly answered red-round retrieval prompt • L23 H4',fontsize=12)
    save(fig,case,'content_plus_range.png',inputs,'range_constructions');plt.close(fig)


def phase_histograms():
    labels=['r < 1','1 ≤ r ≤ 5','r > 5'];colors=['#5488bb','#999999','#d77d3c']
    for case,start in [('fast_phase_amplitude_association',0),('slow_phase_amplitude_association',49)]:
        a=load(case,'keys_histograms.npz');hist=a['counts'];edges=a['edges']
        pct=hist/np.maximum(hist.sum(2,keepdims=True),1)*100
        np.testing.assert_allclose(pct,a['percent'],atol=1e-12)
        fig,axs=plt.subplots(4,4,figsize=(16,13))
        for jj,ax in enumerate(axs.flat):
            if jj>=len(hist):ax.axis('off');continue
            for b in [1,0,2]:ax.stairs(pct[jj,b],edges,color=colors[b],lw=1.4,label=labels[b])
            ax.set_xlim(0,2*np.pi);ax.set_xticks([0,np.pi,2*np.pi],['0','π','2π']);ax.set_title(f'Pair {start+jj} · large n={hist[jj,2].sum():,}',fontsize=10)
            ax.set_xlabel('Phase before RoPE (rad)');ax.set_ylabel('% within amplitude group');ax.tick_params(labelsize=7)
        handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.5,.952),ncol=3)
        band='fast' if start==0 else 'slow'
        fig.suptitle(f'Qwen keys · do {band}-pair phase peaks come from large amplitudes?',fontsize=14)
        fig.text(.5,.015,'Each curve separately normalized to 100%; counts pooled across heads/layers. 72 bins from 0 to 2π.\nOrange = large amplitude; blue = small amplitude; gray = intermediate. Y-axis fitted per panel.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.88,bottom=.10,left=.055,right=.99,wspace=.4,hspace=.65)
        save(fig,case,'keys_phase_by_amplitude.png',[f'data/{case}/keys_histograms.npz'],'phase_histograms');plt.close(fig)


def margin_density():
    case='native_margin_angle_density';pool=load(case,'pooled_histograms.npz')['pool'];weighted=pool[:,1]/pool[:,1].sum((1,2),keepdims=True)*100
    bands=['All64pairs','High frequencies: pairs0–15','Low frequencies: pairs48–63']
    lines=[(0,'#2ca02c','-'),(np.pi,'#c62828','--'),(-np.pi,'#c62828','--'),(np.pi/2,'#bbbbbb',':'),(-np.pi/2,'#bbbbbb',':'),(3*np.pi/2,'#bbbbbb',':'),(-3*np.pi/2,'#bbbbbb',':')]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(3,2,figsize=(11,13.3));countmax=pool[:,0].max();weightmax=weighted.max()
        for b,name in enumerate(bands):
            for col in range(2):
                ax=axs[b,col];v=pool[b,0] if col==0 else weighted[b]
                norm=LogNorm(vmin=1,vmax=countmax) if col==0 else LogNorm(vmin=1e-5,vmax=weightmax)
                im=ax.imshow(np.ma.masked_where(v<=0,v),origin='lower',extent=[-1,1,-1,1],cmap='magma',norm=norm,interpolation='nearest',aspect='equal')
                for s,color,style in lines:
                    xx=np.linspace(-1,1,401);yy=s/np.pi-xx;valid=(yy>=-1)&(yy<=1);ax.plot(xx[valid],yy[valid],style,color=color,lw=.8,alpha=.8)
                ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-1,-.5,0,.5,1]);ax.set_xlabel('(query phase − key-contrast angle) / π');ax.set_ylabel('RoPE distance angle / π')
                ax.set_title(name+(' · raw counts' if col==0 else ' · amplitude-weighted'),fontsize=11)
                fig.colorbar(im,ax=ax,shrink=.85,pad=.02).set_label('Observations / bin' if col==0 else '% of total amplitude / bin')
        handles=[Line2D([0],[0],color='#2ca02c',label='Sum0: maximally positive'),Line2D([0],[0],color='#c62828',ls='--',label='Sum±π: maximally negative'),Line2D([0],[0],color='#aaaaaa',ls=':',label='Sum±π/2: zero')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.047),ncol=3,frameon=False,fontsize=9);fig.suptitle('Native attention winners versus runners-up: angle density',fontsize=15,y=.99)
        fig.text(.5,.012,'All336 heads · 768 sampled text-query positions · 16 random prose excerpts. Linear angle axes, logarithmic color scales.\nAmplitude weight = |query pair| × |position-adjusted key difference| / √128. These are descriptive densities, not mechanism usage fractions.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.075,1,.975))
        save(fig,case,'angle_density.png',[f'data/{case}/pooled_histograms.npz'],'margin_density');plt.close(fig)


def slow_phase_effect():
    case='slow_actual_key_phases';rr=load(case,'results.json')['results']
    labels={'zero_all':'All slow phases → 0','zero_small':'Only slow pairs r<2.5 → 0','pi2_all':'All slow phases → nearest π/2','pi4_all':'All slow phases → nearest π/4','pi2_small':'Slow r<2.5 → nearest π/2','pi4_small':'Slow r<2.5 → nearest π/4'}
    fig,axs=plt.subplots(1,2,figsize=(13,7),layout='constrained');base=list(labels);xx=np.arange(len(base));width=.36
    for shift,suffix,label,color in [(-width/2,'','All key positions','#bb6860'),(width/2,'_prefix14','First 14 keys protected','#437fb0')]:
        axs[0].barh(xx+shift,[rr[k+suffix]['loss_increase'] for k in base],width,label=label,color=color)
        axs[1].barh(xx+shift,[100*rr[k+suffix]['changed_prediction_fraction'] for k in base],width,label=label,color=color)
    for ax in axs:ax.set_yticks(xx,[labels[k] for k in base]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axs[0].set_xlabel('Next-token loss increase (nats/token)');axs[1].set_xlabel('Next-token choices changed (%)');axs[0].legend(fontsize=9)
    fig.suptitle('Actual slow-key phases: pairs 49–63, all layers\nAmplitudes preserved; before RoPE; 7,168 next-token predictions',fontsize=14)
    save(fig,case,'slow_phase_effect.png',[f'data/{case}/results.json'],'slow_phase_effect');plt.close(fig)


def codec_frontier():
    case='key_quantization_small_radius';rows=load(case,'comparison.json')
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    colors={'Origin':'#c67545','Centered':'#3b86b2','Previous uniform':'#666666','Hybrid':'#348a62'}
    for family,color in colors.items():
        group=[r for r in rows if r['family']==family]
        for ax,key,scale in [(axs[0],'loss_increase',1),(axs[1],'changed_prediction_fraction',100)]:
            ax.scatter([r['stored_K_bytes_per_key'] for r in group],[r[key]*scale for r in group],label=family,color=color,marker='s' if family=='Previous uniform' else 'o',s=55,alpha=.8)
    for ax in axs:ax.set_xlabel('K bytes/key at 2,062 tokens, including overhead');ax.grid(alpha=.2);ax.legend(fontsize=9)
    axs[0].set_ylabel('Loss increase (nats/token)');axs[1].set_ylabel('Next-token choices changed (%)');axs[0].axhline(0,lw=.7,color='black')
    axs[0].set_ylim(-.002,.03);axs[1].set_ylim(0,12)
    fig.suptitle('Small-radius adaptive phase compression: zoom on lower-distortion codecs\nPoints outside the displayed ranges remain in the full plot and table',fontsize=14)
    save(fig,case,'quality_vs_bytes_zoom.png',[f'data/{case}/comparison.json'],'codec_frontier');plt.close(fig)


def decay():
    case='inverse_distance_attention';f=load(case,'fit_arrays.npz');l,h=2,5;lo,hi=f['bin_lo'],f['bin_hi'];width=hi-lo+1;x=f['geometric_mean_age']
    M=load(case,'dense_query_means.npz')['means'][:,l,h];mean=M.mean(0);weights=mean/width
    colors=['#397ab2','#d68239','#4a9675','#a161a7'];names=['Alice','Sherlock','Pride','Darwin'];fig,ax=plt.subplots(1,3,figsize=(14,4.8))
    for i in range(4):ax[0].plot(x,M[i]*100,'o-',color=colors[i],lw=1.2,label=names[i],markersize=3)
    ax[0].plot(x,mean*100,'k--',lw=1.6,label='Four-text mean');ax[0].set_xscale('log',base=2);ax[0].set_xticks(x,[f'{a}–{b}' for a,b in zip(lo,hi)],rotation=40,ha='right');ax[0].set_ylim(0,max(M.max()*110,12))
    ax[0].set(ylabel='Attention mass per distance bin (%)',xlabel='Distance bin (tokens; logarithmic spacing)',title='L2 H5: similar mass per doubling of distance');ax[0].legend(fontsize=7,ncol=2);ax[0].grid(alpha=.2)
    for i in range(4):ax[1].plot(x,M[i]/width,lw=.8,color=colors[i],alpha=.45)
    ax[1].plot(x,weights,'o-',color='#397ab2',label='Native dense mean');ax[1].plot(x,f['uniform99_mean'][l,h]/width,'s--',color='#d68239',label='99%-retained mean');scale=np.exp(np.mean(np.log(weights)+np.log(x)));ax[1].plot(x,scale/x,'k:',label='1/d reference')
    ax[1].set_xscale('log',base=2);ax[1].set_yscale('log');ax[1].set(xlabel='Distance into past (tokens)',ylabel='Mean attention weight per available key',title=f"L2 H5: exponent α={f['dense_alpha'][l,h]:.3f}, R²={f['dense_r2'][l,h]:.3f}");ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    for mode,col in [('dense','#397ab2'),('uniform99','#d68239')]:ax[2].hist(f[mode+'_alpha'][f[mode+'_eligible']],bins=np.linspace(-1,3,33),histtype='step',lw=1.5,color=col,label=mode)
    ax[2].axvline(1,color='black',ls='--',lw=.9);ax[2].axvspan(.8,1.2,color='gray',alpha=.12);ax[2].set(xlabel='Decay exponent α (1 corresponds to1/d)',ylabel='Number of heads',title='All eligible heads: a broad range of decay rates');ax[2].legend(fontsize=8)
    fig.suptitle('Measured inverse-distance attention: an illustrative head and the full population\nFour texts ×128late queries; every tested distance available to every query',fontsize=13);fig.tight_layout(rect=[0,.03,1,.88])
    inputs=[f'data/{case}/fit_arrays.npz',f'data/{case}/dense_query_means.npz']
    save(fig,case,'inverse_distance_substantial_example.png',inputs,'decay');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,9),sharey=True)
    for ax,mode in zip(axs,['dense','uniform99']):
        im=ax.imshow(np.ma.masked_where(~f[mode+'_eligible'],f[mode+'_alpha']),vmin=0,vmax=2,cmap='coolwarm',aspect='auto',interpolation='nearest')
        ax.set_xticks(range(12));ax.set_yticks(range(28));ax.set_xlabel('Head');ax.set_title(mode+' decay exponent');fig.colorbar(im,ax=ax,label='α: 0 flat; 1 inverse-distance; 2 steeper')
    axs[0].set_ylabel('Layer');fig.suptitle('Per-head distance decay; white = <1% mass in tested range or zero bin');fig.tight_layout(rect=[0,0,1,.95])
    save(fig,case,'exponent_head_maps.png',inputs,'decay');plt.close(fig)


def main():
    for fn in [frequency_schedule,range_constructions,phase_histograms,margin_density,slow_phase_effect,codec_frontier,decay]:fn()
    from reproduce import main as theory
    theory()
    source=ROOT/'assets/illustrations/dft_rope.png';dest=OUT/'illustrations/dft_rope.png';dest.parent.mkdir(exist_ok=True);dest.write_bytes(source.read_bytes())
    MANIFEST.append(dict(original='assets/illustrations/dft_rope.png',regenerated=str(dest.relative_to(ROOT)),inputs=['code/reproduce.py'],code='code/reproduce.py',function='main'))
    (ROOT/'data/figure_recipes.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
    print('Regenerated',len(MANIFEST),'displayed figures from posted data and code.')


if __name__=='__main__':main()
