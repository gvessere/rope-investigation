"""Reanalyse recorded adapter runs and regenerate Part 2 tables and figures.
No model evaluation or training is performed. Source arrays and hashes are posted.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from figures import PAPER,GREEN,ORANGE
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/adapters';O=ROOT/'figures/adapters'
def read(n):return json.loads((D/n).read_text())
def metric(base,records,parameters):
    losses=np.array([r['test']['sequence_nll'] for r in records]);delta=losses-np.array(base)[None]
    return dict(parameters=parameters,seeds=[r['seed'] for r in records],delta_nll=float(delta.mean()),perplexity_reduction_percent=float(-100*np.expm1(delta.mean())),per_seed_perplexity_reduction_percent=(-100*np.expm1(delta.mean(1))).tolist())
def table(rows,caption):
    h='<div class="table-scroll"><table><caption>'+caption+'</caption><thead><tr><th>Method</th><th>Trainable weights</th><th>Loss change (nats/token)</th><th>Perplexity reduction</th></tr></thead><tbody>'
    for name,v in rows:h+=f'<tr><td>{name}</td><td>{v["parameters"]:,}</td><td>{v["delta_nll"]:+.5f}</td><td>{v["perplexity_reduction_percent"]:.2f}%</td></tr>'
    return h+'</tbody></table></div>'
def plot(rows,name,title):
    fig,ax=plt.subplots(figsize=(8.2,4.1),layout='constrained')
    for y,(label,v) in enumerate(rows):
        vals=v['per_seed_perplexity_reduction_percent'];ax.plot([min(vals),max(vals)],[y,y],color=GREEN,lw=2)
        ax.scatter(vals,np.linspace(y-.08,y+.08,len(vals)),s=22,color=GREEN,alpha=.6)
        ax.plot(v['perplexity_reduction_percent'],y,'D',color=ORANGE,ms=5)
    ax.set(yticks=range(len(rows)),yticklabels=[n for n,v in rows],xlabel='Test perplexity reduction (%)',title=title)
    ax.invert_yaxis();ax.axvline(0,color='.6',lw=.7);ax.grid(axis='x',alpha=.15)
    fig.savefig(O/name,metadata={'Date':None});plt.close(fig)
def main():
    O.mkdir(exist_ok=True)
    for name,p in read('provenance.json').items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==p['sha256'],name
    phase=read('phase-results.json');dynamic=read('dynamic-results.json');base=phase['test_baseline']['sequence_nll']
    np.testing.assert_allclose(base,dynamic['test_baseline']['sequence_nll'],atol=1e-5)
    phase_rows=[];lookup={}
    for key,label in [('coordinated','Coordinated phase'),('permuted','Permuted-frequency phase'),('q_lora','Q-only LoRA, rank 6')]:
        rec=[r for r in phase['test_results'] if r['arm']==key];assert len(rec)==5
        v=metric(base,rec,516096);phase_rows.append((label,v));lookup[key]=v
    dynamic_rows=[]
    base=dynamic['test_baseline']['sequence_nll']
    for key,label in [('coordinated','Coordinated phase'),('q_lora','Q-only LoRA, rank 6')]:
        rec=[r for r in phase['test_results'] if r['arm']==key and r['seed'] in [1101,1102]]
        v=metric(base,rec,516096);lookup[key+'_two']=v;dynamic_rows.append((label,v))
    for key,label in [('pair_magnitude','Pair amplitudes'),('coordinated_pair_magnitude','Phase + pair amplitudes'),('q_lora12','Q-only LoRA, rank 12')]:
        rec=[r for r in dynamic['test_results'] if r['arm']==key];assert len(rec)==2
        v=metric(base,rec,rec[0]['parameters']);lookup[key]=v;dynamic_rows.append((label,v))
    native=dict(parameters=0,delta_nll=0,perplexity_reduction_percent=0)
    control_rows=[('Original model, no training',native)]
    for name,label in [('wq-results.json','Original WQ matrices trained'),('full-model-results.json','All original weights trained')]:
        x=read(name);rec=x['test_results'];v=metric(x['test_baseline']['sequence_nll'],rec,rec[0]['parameters']);control_rows.append((label,v))
    captures=np.load(D/'peak-captures.npz');peak_results={};b=captures['loss_native'].astype(float);full=captures['loss_full'].astype(float)
    for key in ['native','full','full_native_peak','full_local_peak','native_full_peak']:
        loss=captures['loss_'+key].astype(float);delta=float((loss-b).mean())
        peak_results[key]=dict(delta_nll=delta,perplexity_reduction_percent=float(-100*np.expm1(delta)),loss_gain_retained=float((b-loss).mean()/(b-full).mean()))
        np.testing.assert_allclose(delta,read('peak-interventions.json')['records'][key]['nll']-read('peak-interventions.json')['records']['native']['nll'],atol=1e-6)
    rows=changed=direct=small=0
    for l in range(28):
        n=captures[f'native_peak_layer{l}'][:,:,1:].astype(int);f=captures[f'full_peak_layer{l}'][:,:,1:].astype(int);local=captures[f'local_peak_layer{l}'][:,:,1:].astype(int)
        rows+=n.size;changed+=int((n!=f).sum());direct+=int((f!=local).sum());small+=int(((n!=f)&(abs(n-f)<=3)).sum())
    peak_stats=dict(rows=rows,changed_fraction=changed/rows,direct_changed_fraction=direct/rows,changed_peaks_within_three_tokens=small/changed)
    share=lookup['pair_magnitude']['delta_nll']/lookup['q_lora_two']['delta_nll']
    summary=dict(phase_five_seed=phase_rows,dynamic_two_seed=dynamic_rows,controls_two_seed=control_rows,amplitude_share_of_lora_loss_gain=share,peak_interventions=peak_results,peak_statistics=peak_stats)
    (D/'recomputed-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(phase_rows,'phase-comparison.svg','Same 516,096 weights · five optimization seeds')
    plot(dynamic_rows,'amplitude-comparison.svg','Two shared seeds · compare parameter budgets in table')
    p=ROOT/'appendix-translation.html';s=p.read_text()
    for tag,html in [('PHASE',table(phase_rows,'Phase-only adaptation · five seeds · same parameter budget')),('AMPLITUDE',table(dynamic_rows,'Same two seeds · 64 updates · 10,240 test predictions per seed')),('CONTROL',table(control_rows,'Unchanged-architecture controls · two seeds · parameter budgets differ'))]:
        a,b='<!-- '+tag+' TABLE START -->','<!-- '+tag+' TABLE END -->'
        before,rest=s.split(a);_,after=rest.split(b);s=before+a+'\n'+html+'\n'+b+after
    p.write_text(s);print(json.dumps(dict(amplitude_share=share,peaks=peak_stats,controls=control_rows),indent=2))
if __name__=='__main__':main()
