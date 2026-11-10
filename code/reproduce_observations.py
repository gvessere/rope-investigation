"""Regenerate the original 70 Q/K plot pages from losslessly saved activations.

Run: python code/reproduce_observations.py --kind queries --view xy --page 0
Omit filters to regenerate all pages. Add --pdf to also regenerate all four PDFs.
The published PNGs/PDFs are the original investigation files, not replacements.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/observations'
PLOTS=ROOT/'figures/observations'
COLORS=plt.get_cmap('turbo')(np.arange(64)/63)
plt.rcParams.update({'font.size':9})
SM=plt.cm.ScalarMappable(norm=Normalize(0,63),cmap='turbo')


def load_layer(kind,layer):
    with np.load(DATA/f'{kind}_layer_{layer:02d}.npz') as z:
        return z['amplitude'],z['phase']


def page_figure(kind,view,page):
    """Same data, scales, scatter style and panel layout as the original scripts."""
    lpp=1 if kind=='queries' else 4
    start=page*lpp;end=start+lpp-1
    heads=12 if kind=='queries' else 2
    if view=='xy':
        figsize=(17,13) if kind=='queries' else (11,16)
    else:
        figsize=(16,11) if kind=='queries' else (12,14)
    fig,axs=plt.subplots(*( (3,4) if kind=='queries' else (4,2) ),figsize=figsize)
    for li in range(lpp):
        l=start+li;a,phase=load_layer(kind,l)
        x=a*np.cos(phase);y=a*np.sin(phase)
        for h in range(heads):
            ax=axs.flat[li*heads+h]
            title=f'L{l} H{h}' if kind=='queries' else f'L{l} KV{h} → Q heads {h*6}–{h*6+5}'
            if view=='xy':
                ax.scatter(x[h].ravel(),y[h].ravel(),c=np.tile(COLORS,(a.shape[1],1)),s=.55,alpha=.30,linewidths=0,rasterized=True)
                lim=max(np.abs(x[h]).max(),np.abs(y[h]).max())*1.04
                ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_aspect('equal',adjustable='box')
                ax.axhline(0,color='0.75',lw=.5,zorder=0);ax.axvline(0,color='0.75',lw=.5,zorder=0)
                ax.set_xlabel('x = coordinate j (before RoPE)');ax.set_ylabel('y = coordinate j + 64 (before RoPE)')
                ax.set_title(title,fontsize=10)
            else:
                ax.scatter(phase[h].ravel(),a[h].ravel(),c=np.tile(COLORS,(a.shape[1],1)),s=.55,alpha=.28,linewidths=0,rasterized=True)
                ax.set_xlim(0,2*np.pi);ax.set_ylim(0,40)
                ax.set_xticks([0,np.pi,2*np.pi],['0','π','2π']);ax.set_yticks([0,10,20,30,40])
                ax.set_xlabel('Phase φ before RoPE (rad)');ax.set_ylabel('Pair amplitude a = √(x² + y²)')
                ax.set_title(title,fontsize=11)
            ax.tick_params(labelsize=8)
    layer_label=f'layer {start}'+(f'–{end}' if end>start else '')
    tokens='768 sampled query tokens' if kind=='queries' else '2,379 prose key tokens; chat-prefix keys excluded'
    if view=='xy':
        fig.suptitle(f'Qwen {kind} · {layer_label} · axes fitted separately for each head',fontsize=14)
        fig.subplots_adjust(left=.07,right=.90,bottom=.10,top=.93,wspace=.40,hspace=.55)
        fig.colorbar(SM,cax=fig.add_axes([.93,.25,.012,.5]),label='Pair index: 0 = highest frequency; 63 = lowest')
        fig.text(.5,.025,f'Each point = one token × one RoPE pair. {tokens}; 64 pairs per head.\nKnown positional rotation removed. Linear axes; no clipping; full range fitted per head.\nWithin each panel, x and y have the same scale. Axis limits differ between panels; compare tick values.',ha='center',fontsize=9)
        name=f'{kind}_xy_layers_{start:02d}_{end:02d}.png'
    else:
        tokens='768 sampled query tokens' if kind=='queries' else '2,379 prose key tokens (chat-prefix keys excluded)'
        fig.suptitle(f'Qwen {kind} · {layer_label} · amplitude axis 0–40',fontsize=14)
        fig.subplots_adjust(left=.07,right=.9,bottom=.10,top=.94,wspace=.35,hspace=.55)
        fig.colorbar(SM,cax=fig.add_axes([.93,.25,.015,.5]),label='Pair index: 0 = highest frequency; 63 = lowest')
        fig.text(.5,.025,f'Every point = one token × one RoPE pair. {tokens}, 64 pairs per head.\nKnown positional rotation removed. Linear axes; points above amplitude 40 are clipped.\n'+('Two shared key heads per layer: KV0 serves Q heads 0–5; KV1 serves Q heads 6–11.' if kind=='keys' else 'Native query vectors only; no key or margin enters this plot.'),ha='center',fontsize=9)
        name=f'{kind}_layers_{start:02d}_{end:02d}_y40.png'
    return fig,name


def verify_data():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for kind,record in manifest['data'].items():
        points=above=0
        for layer,file in enumerate(record['layers']):
            p=ROOT/file['file']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==file['sha256'],p
            a,phase=load_layer(kind,layer)
            assert list(a.shape)==record['shape'][1:]
            assert a.dtype==np.float32 and phase.dtype==np.float32
            assert np.isfinite(a).all() and np.isfinite(phase).all()
            assert (a>=0).all() and (phase>=0).all() and (phase<=2*np.pi).all()
            points+=a.size;above+=int((a>40).sum())
        assert points==record['points'] and above==record['above_40']
    print('All 56 data chunks, point counts and amplitude-cap counts verified.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['queries','keys'])
    p.add_argument('--view',choices=['xy','phase'])
    p.add_argument('--page',type=int,help='Zero-based page index: queries 0–27, keys 0–6')
    p.add_argument('--pdf',action='store_true',help='Regenerate PDFs too (requires all pages)')
    p.add_argument('--verify',action='store_true',help='Check saved data without plotting')
    args=p.parse_args()
    if args.verify:return verify_data()
    if args.pdf and args.page is not None:p.error('--pdf requires all pages')
    PLOTS.mkdir(exist_ok=True,parents=True)
    for kind in ([args.kind] if args.kind else ['queries','keys']):
        pages=28 if kind=='queries' else 7
        if args.page is not None and not 0<=args.page<pages:p.error('Page out of range')
        for view in ([args.view] if args.view else ['xy','phase']):
            pdf=None
            if args.pdf:
                name=f'{kind}_xy_dynamic.pdf' if view=='xy' else f'{kind}_all_layers_y40.pdf'
                pdf=PdfPages(ROOT/'documents'/name)
            try:
                for page in ([args.page] if args.page is not None else range(pages)):
                    fig,name=page_figure(kind,view,page)
                    fig.savefig(PLOTS/name,dpi=120)
                    if pdf:pdf.savefig(fig,dpi=110)
                    plt.close(fig)
                    print(name,flush=True)
            finally:
                if pdf:pdf.close()


if __name__=='__main__':main()
