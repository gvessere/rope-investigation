"""Regenerate the default query-pair atlas from the posted saved activations.
python code/head_pair_atlas.py --layer 14 --start 48
The interactive canvas uses the same arrays and per-column maximum rule.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--layer',type=int,default=14);p.add_argument('--start',type=int,choices=[0,16,32,48],default=48);args=p.parse_args()
 if not 0<=args.layer<28:p.error('Layer must be 0–27')
 with np.load(ROOT/f'data/observations/queries_layer_{args.layer:02d}.npz') as z:a=z['amplitude'];phase=z['phase']
 assert a.shape==phase.shape==(12,768,64)
 colors=__import__('json').loads((ROOT/'data/observations/palette.json').read_text())
 maxima=a[:,:,args.start:args.start+16].astype(float).max(axis=(0,1))*1.035
 fig,axes=plt.subplots(12,16,figsize=(26,19),sharex=True)
 for h in range(12):
  for c,j in enumerate(range(args.start,args.start+16)):
   ax=axes[h,c];ax.scatter(phase[h,:,j],a[h,:,j],s=.7,alpha=.4,color=colors[j],linewidths=0,rasterized=True)
   ax.set(xlim=(0,2*np.pi),ylim=(0,maxima[c]),yticks=[],xticks=[0,np.pi,2*np.pi],xticklabels=['0','π','2π'])
   ax.tick_params(labelsize=6);ax.spines[['top','right']].set_visible(False)
   ax.set_facecolor('#f2f7f2' if h<6 else '#f5f3fa')
   if c==0:ax.set_ylabel(f'H{h} · KV{h//6}',rotation=0,labelpad=28,fontsize=9)
   if h==0:ax.set_title(f'Pair {j}\nr: 0–{maxima[c]:.3g}',fontsize=8)
 fig.suptitle(f'Query r versus phase · layer {args.layer} · pairs {args.start}–{args.start+15}',fontsize=16)
 fig.subplots_adjust(left=.065,right=.99,top=.94,bottom=.045,hspace=.20,wspace=.13)
 fig.text(.5,.012,'One column = one query pair, with the same radius scale across all heads. H0–H5 share KV0; H6–H11 share KV1. All 768 saved points per cell.',ha='center',fontsize=10)
 out=ROOT/'figures/analysis'/f'layer_{args.layer:02d}_pairs_{args.start:02d}_{args.start+15:02d}.png';out.parent.mkdir(exist_ok=True);fig.savefig(out,dpi=100);plt.close(fig);print(out)
if __name__=='__main__':main()
