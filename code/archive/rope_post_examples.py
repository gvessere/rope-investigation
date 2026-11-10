"""Construct range terms against captured Qwen keys, retaining native slow content term."""
import json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'outputs/implicit_semantic_translation/confirmation';P=ROOT/'outputs/rope_post_examples';P.mkdir(exist_ok=True)
def synth(coeff,basis):
 out=np.einsum('kj,jd->kd',coeff,basis,optimize=False).real
 assert np.isfinite(out).all()
 return out
def main():
 r=json.loads((S/'examples.json').read_text())[0];a=np.load(S/'capture_000.npz');omega=np.load(S/'frequencies.npy').astype(float);l,h=23,4;m=r['query_positions'][1];kp=np.array(r['label_positions']);qraw=a['q_23'][0,1,h].astype(float);kraw=a['k_23'][0,h].astype(float);q=(qraw[:64]+1j*qraw[64:])*np.exp(-1j*m*omega);k=(kraw[:,:64]+1j*kraw[:,64:])*np.exp(-1j*kp[:,None]*omega);target=r['target'];foils=[i for i in range(4) if i!=target];d=np.linspace(0,80,1601);basis=np.exp(1j*omega[:,None]*d);scale=1/np.sqrt(128);dw=np.zeros(64);dw[0]=(omega[0]-omega[1])/2;dw[-1]=(omega[-2]-omega[-1])/2;dw[1:-1]=(omega[:-2]-omega[2:])/2
 defs=[dict(name='Local window',kind='positive',centers=[15],fwhm=10,ranges=[[10,20]]),dict(name='Broad window',kind='positive',centers=[40],fwhm=40,ranges=[[20,60]]),dict(name='Two preferred neighborhoods',kind='positive',centers=[12,36],fwhm=8,ranges=[[8,16],[32,40]]),dict(name='Local suppression',kind='negative',centers=[15],fwhm=10,ranges=[[10,20]])]
 native_coeff=q[None,:]*k.conj()*scale;native_profiles=synth(native_coeff,basis);native_slow=synth(native_coeff[:,32:],basis[32:]);native_fast=synth(native_coeff[:,:32],basis[:32]);fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');meta=dict(model='Qwen/Qwen2.5-1.5B-Instruct',layer=l,head=h,query_anchor='Final assistant-prefix token before first answer',prompt=r['rendered'],answer=r['answer'],target_label=r['expected'],target=target,labels=[x['label'] for x in r['records']],native_q_norm=float(np.linalg.norm(q)),native_fast_q_norm=float(np.linalg.norm(q[:32])),native_key_norms=np.linalg.norm(k,axis=1).tolist(),construction='Keep all four actual captured pre-RoPE keys fixed. Retain native query pairs32–63 as a slow content component. Construct pairs0–31 to approximate Gaussian-window Fourier coefficients against the target key. Scale each constructed fast subvector to the native fast-query norm. Both total and fast-subvector query norms therefore equal the native values. Individual pair radii and phases may change. These are engineered queries, not unmodified native queries or phase-only interventions.',distance='Sweep the same frozen key content over hypothetical upstream distances0–80; all four alternative key vectors evaluated at the same distance. Actual record positions and surrounding context are not regenerated. Plotted terms are raw pre-softmax score contributions, not attention probabilities.',examples=[])
 arrays=dict(omega=omega,distances=d,native_query=q,native_keys=k,native_profiles=native_profiles,native_slow=native_slow,native_fast=native_fast);max_norm=0
 for ei,(definition,ax) in enumerate(zip(defs,axes.flat)):
  sigma=definition['fwhm']/(2*np.sqrt(2*np.log(2)));B=dw*np.exp(-.5*(sigma*omega)**2)*sum(np.exp(-1j*omega*c) for c in definition['centers'])/len(definition['centers']);B[32:]=0
  if definition['kind']=='negative':B=-B
  qe=q.copy();qe[:32]=np.sqrt(128)*B[:32]/k[target,:32].conj();factor=np.linalg.norm(q[:32])/np.linalg.norm(qe[:32]);qe[:32]*=factor;coeff=qe[None,:]*k.conj()*scale;fast=synth(coeff[:,:32],basis[:32]);full=synth(coeff,basis);max_norm=max(max_norm,abs(np.linalg.norm(qe)-np.linalg.norm(q)));assert np.isfinite(qe).all();assert np.allclose(qe[32:],q[32:]);ideal=sum(np.exp(-.5*((d-c)/sigma)**2) for c in definition['centers'])/len(definition['centers']);ideal*=(-1 if definition['kind']=='negative' else 1);fit_scale=np.dot(ideal,fast[target])/np.dot(ideal,ideal);rmse=float(np.sqrt(np.mean((fast[target]-fit_scale*ideal)**2)));fitcorr=float(np.corrcoef(fast[target],ideal)[0,1]);windows=np.zeros(len(d),bool)
  for left,right in definition['ranges']:windows|=(d>=left)&(d<=right);ax.axvspan(left,right,color=('red' if definition['kind']=='negative' else 'orange'),alpha=.10)
  ax.plot(d,native_fast[target],ls=':',color='gray',alpha=.75,label='Unmodified query, range-band term');ax.plot(d,fast[foils].mean(0),color='darkorange',ls='--',label='Constructed query, mean other-key term');ax.plot(d,fast[target],color='navy',lw=2,label='Constructed query, matching-key term');ax.axhline(0,color='gray',lw=.6);ax.set(title=definition['name'],xlabel='Hypothetical upstream distance (tokens)',ylabel='Pairs 0–31 score contribution');ax.grid(alpha=.15)
  if ei==0:ax.legend(fontsize=7,loc='upper right')
  selected=[]
  for x in sorted(set(sum(([a,b,(a+b)/2] for a,b in definition['ranges']),[]))):
   ix=np.argmin(abs(d-x));scores=full[:,ix];selected.append(dict(distance=float(d[ix]),range_term=float(fast[target,ix]),full_target_score=float(scores[target]),full_mean_foil_score=float(scores[foils].mean()),full_target_minus_best_foil=float(scores[target]-scores[foils].max()),target_rank=int(1+(scores>scores[target]).sum())))
  result=dict(**definition,q_norm=float(np.linalg.norm(qe)),fast_q_norm=float(np.linalg.norm(qe[:32])),query_norm_ratio=float(np.linalg.norm(qe)/np.linalg.norm(q)),range_shape_correlation=fitcorr,range_shape_RMSE=rmse,peak_absolute_range_score=float(abs(fast[target]).max()),target_rank_first_fraction_inside_window=float((full[target,windows]>=full[foils][:,windows].max(0)).mean()),sample_scores=selected,phase_changes_wrapped=np.angle(qe*np.conj(q)).tolist(),amplitude_ratios=(abs(qe)/abs(q)).tolist());meta['examples'].append(result);arrays[f'query_{ei}']=qe;arrays[f'fast_{ei}']=fast;arrays[f'full_{ei}']=full;print(definition['name'],fitcorr,'rankinside',result['target_rank_first_fraction_inside_window'],'normratio',result['query_norm_ratio'],flush=True)
 meta['max_query_norm_error']=float(max_norm);assert max_norm<1e-10
 fig.suptitle('Four range functions built against actual Qwen key vectors\nEngineered queries; native query norm and slow content component retained; linear axes',fontsize=12);fig.savefig(P/'range_constructions.png',dpi=180);fig.savefig(P/'range_constructions.pdf');plt.close(fig)
 # Full-score illustration: the same constructed query combines range and content.
 fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
 for ki,label in enumerate(meta['labels']):
  color='navy' if ki==target else ['darkorange','green','purple','brown'][ki];style='-' if ki==target else '--';ax[0].plot(d,native_slow[ki],color=color,ls=style,label=f'{label}'+(' (red round target)' if ki==target else ''))
  ax[1].plot(d,arrays['full_0'][ki],color=color,ls=style,label=f'{label}'+(' (target)' if ki==target else ''))
 ax[0].set(title='Unmodified slow component, pairs 32–63',xlabel='Hypothetical upstream distance',ylabel='Score contribution');ax[1].axvspan(10,20,color='orange',alpha=.12);ax[1].set(title='Slow content component + constructed local window',xlabel='Hypothetical upstream distance',ylabel='Full 64-pair attention logit')
 for aa in ax:aa.legend(fontsize=8);aa.grid(alpha=.15)
 fig.suptitle('Actual keys from a correctly answered red-round retrieval prompt • L23 H4',fontsize=12);fig.savefig(P/'content_plus_range.png',dpi=180);fig.savefig(P/'content_plus_range.pdf');plt.close(fig)
 # Unmodified Q and K at their actual captured positions provide a separate reality check.
 actual=(native_coeff*np.exp(1j*omega*(m-kp)[:,None])).sum(-1).real;saved=a['scores_23'][0,1,h];meta['native_score_reconstruction_error']=float(abs(actual-saved).max());meta['native_target_rank']=int(1+(saved>saved[target]).sum());assert meta['native_score_reconstruction_error']<1e-4
 (P/'constructions.json').write_text(json.dumps(meta,indent=2));np.savez_compressed(P/'vectors_and_profiles.npz',**arrays)
 dest=Path('/Users/geryvessere/Documents/Codex/2026-10-20/i/outputs/rope_post_examples');dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(P/'reproduce.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  z.write(Path(__file__),'rope_post_examples.py');z.write(P/'vectors_and_profiles.npz','vectors_and_profiles.npz');z.write(P/'constructions.json','constructions.json')
 for n in ['range_constructions.png','range_constructions.pdf','content_plus_range.png','content_plus_range.pdf','constructions.json','vectors_and_profiles.npz','reproduce.zip']:shutil.copy2(P/n,dest/n)
 print(dest)
if __name__=='__main__':main()
