"""Two-seed tests of content-dependent head and pair magnitudes.

No additional tuning seed: every LR candidate uses only seeds1101,1102.
Choose LR by their mean step64 validation loss; open test after all fitting.
"""
from scalar_rope_train import ROOT
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

P=ROOT/'outputs/scalar_rope_dynamic_magnitude'
P.mkdir(exist_ok=True)
OLD=ROOT/'outputs/scalar_rope_multiseed'
START=time.monotonic()
SEEDS=[1101,1102]
ARMS=['head_magnitude','pair_magnitude','coordinated_head_magnitude',
      'coordinated_pair_magnitude','permuted_pair_magnitude',
      'coordinated_pair_magnitude_qk','q_lora12']
GRID={arm:([3e-5,1e-4,3e-4] if 'head_magnitude' in arm else [3e-4,1e-3,3e-3]) for arm in ARMS}
POINTS=[8,16,32,64]

def emit(**x):
    print(json.dumps(dict(seconds=round(time.monotonic()-START,2),**x)),flush=True)

def dump(name,x):
    dest=P/name;dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(dest)

class Adapter(nn.Module):
    def __init__(self,model):
        super().__init__()
        c=model.config
        self.L,self.H,self.D=c.num_hidden_layers,c.num_attention_heads,c.hidden_size
        self.K=c.num_key_value_heads;self.group=self.H//self.K;self.dh=self.D//self.H
        self.phase=nn.Parameter(torch.zeros((self.L,self.H,self.D),device='mps'))
        self.head=nn.Parameter(torch.zeros((self.L,self.H,self.D),device='mps'))
        self.magA=nn.Parameter(torch.zeros((self.L,8,self.D),device='mps'))
        self.magB=nn.Parameter(torch.zeros((self.L,self.H*self.dh//2,8),device='mps'))
        self.qA=nn.Parameter(torch.zeros((self.L,12,self.D),device='mps'))
        self.qB=nn.Parameter(torch.zeros((self.L,self.D,12),device='mps'))
        self.register_buffer('omega',model.model.rotary_emb.inv_freq.detach().clone())
        self.register_buffer('freq',self.omega.clone())
        self.mode='off';self.last_s={};self.collect=False;self.gate_sum=None;self.gate_count=0;self.gate_stats=[]
        self.mean_s=None
        self.handles=[]
        for l,layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(self.q_hook(l)))
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self.k_hook(l)))
        self.counts={m:sum(p.numel() for p in self.selected(m)) for m in ARMS}
        assert self.counts['head_magnitude']==self.counts['pair_magnitude']==516096
        assert all(self.counts[m]==1032192 for m in ARMS[2:])

    def is_phase(self,mode=None):
        return (mode or self.mode).startswith(('coordinated','permuted'))

    def selected(self,mode):
        if mode=='q_lora12':return [self.qA,self.qB]
        mag=[self.head] if 'head_magnitude' in mode else [self.magA,self.magB]
        return ([self.phase] if self.is_phase(mode) else [])+mag

    def magnitude_params(self,mode):
        if mode=='q_lora12':return [self.qA,self.qB]
        return [self.head] if 'head_magnitude' in mode else [self.magA,self.magB]

    def reset(self,mode,seed):
        self.zero_grad(set_to_none=True);self.last_s={};self.mean_s=None
        gen=torch.Generator().manual_seed(seed)
        # Separate local RNGs reproduce the standard LoRA initialization convention.
        with torch.no_grad():
            self.phase.zero_();self.head.zero_();self.magB.zero_();self.qB.zero_()
            self.magA.copy_((torch.rand(self.magA.shape,generator=gen)*2-1).to('mps')/math.sqrt(self.D))
            genq=torch.Generator().manual_seed(seed)
            self.qA.copy_((torch.rand(self.qA.shape,generator=genq)*2-1).to('mps')/math.sqrt(self.D))
            perm=torch.randperm(len(self.omega),generator=torch.Generator().manual_seed(seed+700000))
            self.freq.copy_(self.omega[perm.to('mps')])
        for p in self.parameters():p.requires_grad_(False)
        for p in self.selected(mode):p.requires_grad_(True)
        self.mode=mode
        return perm.tolist()

    def log_magnitude(self,l,x):
        if self.mean_s is not None:
            return self.mean_s[l].view(1,1,self.H,self.dh//2).expand(*x.shape[:-1],self.H,self.dh//2)
        if 'head_magnitude' in self.mode:
            z=F.linear(x,self.head[l]).unsqueeze(-1).expand(*x.shape[:-1],self.H,self.dh//2)
        else:
            z=F.linear(F.linear(x,self.magA[l]),self.magB[l]).view(*x.shape[:-1],self.H,self.dh//2)
        return 2*torch.tanh(z/2)

    def q_hook(self,l):
        def apply(module,inputs,out):
            if self.mode=='off':return out
            x=inputs[0]
            if self.mode=='q_lora12':return out+F.linear(F.linear(x,self.qA[l]),self.qB[l])
            q=out.view(*x.shape[:-1],self.H,self.dh)
            if self.is_phase():
                delta=2*torch.tanh(F.linear(x,self.phase[l])/2)
                omega=self.freq if self.mode.startswith('permuted') else self.omega
                angle=-delta.unsqueeze(-1)*omega
                a,b=q.chunk(2,dim=-1);cs,sn=angle.cos(),angle.sin()
                q=torch.cat((a*cs-b*sn,a*sn+b*cs),dim=-1)
            s=self.log_magnitude(l,x)
            if self.collect:
                self.gate_sum[l].add_(s.detach().sum((0,1)))
                if l==0:self.gate_count+=s.shape[0]*s.shape[1]
                self.gate_stats.append(torch.stack((s.detach().square().sum(),s.detach().abs().max(),
                    (s.detach().abs()>1.9).float().sum(),s.new_tensor(s.numel()))))
            if self.mode.endswith('_qk'):
                self.last_s[l]=s
                s=s/2
            g=s.exp();return (q*torch.cat((g,g),dim=-1)).reshape_as(out)
        return apply

    def k_hook(self,l):
        def apply(module,inputs,out):
            if not self.mode.endswith('_qk'):return out
            # Shared predictor at each KEY token; group-mean log gate for GQA keys.
            s=self.last_s.pop(l)
            sk=s.view(*s.shape[:-2],self.K,self.group,self.dh//2).mean(-2)/2
            g=sk.exp();k=out.view(*out.shape[:-1],self.K,self.dh)
            return (k*torch.cat((g,g),dim=-1)).reshape_as(out)
        return apply

    def collect_begin(self):
        self.collect=True;self.gate_count=0;self.gate_stats=[]
        self.gate_sum=torch.zeros((self.L,self.H,self.dh//2),device='mps')

    def collect_end(self):
        self.collect=False
        mean=self.gate_sum/self.gate_count
        v=torch.stack(self.gate_stats).detach().cpu().numpy()
        stats=dict(rms_log_gain=float(np.sqrt(v[:,0].sum()/v[:,3].sum())),
            max_abs_log_gain=float(v[:,1].max()),saturation_fraction=float(v[:,2].sum()/v[:,3].sum()),
            rms_mean_log_gain=float(mean.square().mean().sqrt()))
        self.gate_stats=[];self.gate_sum=None
        return mean,stats

def loss_batch(model,rows):
    ids=torch.tensor([r['input_ids'] for r in rows],device='mps')
    logits=model(input_ids=ids[:,:-1],use_cache=False).logits
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').view(len(rows),128).mean(-1)

def evaluate(model,rows,per_sequence=False):
    vals=[]
    with torch.no_grad():
        for start in range(0,len(rows),4):vals.extend(loss_batch(model,rows[start:start+4]).detach().cpu().tolist())
    v=dict(nll=float(np.mean(vals)),sequences=len(rows),tokens=128*len(rows))
    if per_sequence:v['sequence_nll']=vals
    return v

def fit(model,a,data,mode,seed,lr,baseline):
    stem=f'fits/{mode}_seed{seed}_lr{lr:g}';dest=P/f'{stem}.json'
    if dest.exists():return json.loads(dest.read_text())
    perm=a.reset(mode,seed)
    groups=[dict(params=a.magnitude_params(mode),lr=lr)]
    if a.is_phase():groups.append(dict(params=[a.phase],lr=3e-4))
    opt=torch.optim.Adam(groups)
    order=np.random.default_rng(seed).permutation(128)
    history=[dict(step=0,validation=baseline)];losses=[]
    emit(stage='fit_start',arm=mode,seed=seed,lr=lr)
    for step in range(1,65):
        batch=[data['train'][int(i)] for i in order[(step-1)*2:step*2]]
        opt.zero_grad(set_to_none=True);loss=loss_batch(model,batch).mean()
        if not torch.isfinite(loss):raise RuntimeError((mode,seed,lr,step,'nonfinite'))
        loss.backward();torch.nn.utils.clip_grad_norm_(a.selected(mode),1.);opt.step()
        losses.append(float(loss.detach()))
        if step in POINTS:
            val=evaluate(model,data['validation']);history.append(dict(step=step,validation=val))
            emit(stage='validation',arm=mode,seed=seed,lr=lr,step=step,nll=val['nll'])
    ck=P/f'{stem}.pt';ck.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(mode=mode,seed=seed,lr=lr,values=[p.detach().cpu().clone() for p in a.selected(mode)]),ck)
    v=dict(arm=mode,seed=seed,lr=lr,phase_lr=3e-4 if a.is_phase() else None,steps=64,
        parameters=a.counts[mode],history=history,online_train_losses=losses,
        final_validation_nll=history[-1]['validation']['nll'],checkpoint=str(ck),
        permutation=perm if mode.startswith('permuted') else None)
    dump(f'{stem}.json',v);a.zero_grad(set_to_none=True);del opt
    return v

def load(a,run):
    a.reset(run['arm'],run['seed']);ck=torch.load(run['checkpoint'],weights_only=True)
    with torch.no_grad():
        for p,v in zip(a.selected(run['arm']),ck['values']):p.copy_(v.to('mps'))

def main():
    torch.set_num_threads(4);torch.manual_seed(1101)
    rows=json.loads((OLD/'dataset.json').read_text());dump('dataset.json',rows)
    data={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    protocol=dict(model='Qwen/Qwen2.5-1.5B-Instruct',original_weights_frozen=True,dtype='float32',
        seeds=SEEDS,no_additional_pilot_seed=True,steps=64,batch_size=2,training_tokens=16384,
        splits={k:len(v) for k,v in data.items()},lr_grid=GRID,
        selection='Every LR configuration uses exactly the same TWO seeds. Select by mean step64 validation NLL across these two seeds. Test only after all fits/choices. Retain phase LR3e-4 from preceding validation search; no joint phase/magnitude LR retuning.',
        head_magnitude='One bias-free row/head predicts s=2*tanh((a@normalized_hidden)/2); multiply all64Qpairs by exp(s).516096parameters.',
        pair_magnitude='Rank8 predictor s=2*tanh((BA@normalized_hidden)/2), A8x1536,B768x8 per layer; reshape to12headsx64pairs.516096parameters, same as rank6 Q-LoRA.',
        phase='One zero-initialized displacement row/head, delta=2*tanh((w@normalized_hidden)/2); extra Q rotation -delta*omega (or permutedomega).516096additionalparameters.',
        qk='Shared pair predictor: Q at token m scaled by exp(s_m,h,j/2); GQA K at token n scaled by exp(mean_over_6_sharing_heads(s_n,h,j)/2). Tied Q/K gates, not independently learned key gates. Same parameter budget as Q-only combined. Unlike static gains this changes key-content selection.',
        q_lora12='Standard rank12Q-onlyLoRA, alpha/r1, A uniform±1/sqrt1536, Bzero;1032192parameters, exactly matches phase+magnitude.',
        initialization='Head rows and phase rows zero; lowrank predictor/LoRA A random,Bzero. Every variant starts as original model. Gains bounded exp(-2)..exp(2); QK factors use half log gains.',
        optimizer='Adamdefaults, no weightdecay; global gradnormclip1 on selectedparams; model eval/dropoutoff.',
        controls='Previous static gain and rank6QLoRA results restricted to the SAME TWO seeds are references, not new runs.',
        ablation='Selected magnitude models: compute mean predicted log-gain per layer/head/pair on TRAIN with adapted model active, replace token-dependent predictions with these constants on TEST. No refitting; diagnostic only.',
        limitations='Two seeds only, shared fixed classic-prose corpus and reused test set. Exploratory extension. Pair predictor is lowrank, QK is tied, amplitude ranges bounded; no unrestricted Q/K magnitude fine-tuning.')
    dump('protocol.json',protocol);emit(stage='loading')
    model=AutoModelForCausalLM.from_pretrained(protocol['model'],local_files_only=True,
        dtype=torch.float32,attn_implementation='sdpa').to('mps').eval();model.requires_grad_(False)
    a=Adapter(model);a.mode='off';baseline=evaluate(model,data['validation'])
    off=evaluate(model,data['validation'][:2])['nll'];checks={}
    for mode in ARMS:
        a.reset(mode,SEEDS[0]);zero=evaluate(model,data['validation'][:2])['nll']
        checks[mode]=abs(zero-off);assert checks[mode]<2e-6,(mode,zero,off)
    dump('zero_checks.json',checks)
    # Verify a genuinely content-dependent KEY path and magnitude gradient numerically.
    a.reset('coordinated_pair_magnitude_qk',SEEDS[0])
    loss=loss_batch(model,data['train'][:2]).mean();loss.backward()
    grad=a.magB.grad.detach().clone();norm=float(grad.norm());unit=grad/norm;eps=.005;values=[]
    with torch.no_grad():
        for sign in [-1,1]:
            a.magB.copy_(sign*eps*unit);values.append(float(loss_batch(model,data['train'][:2]).mean()))
    finite=(values[1]-values[0])/(2*eps)
    check=dict(analytic=norm,finite_difference=finite,relative_error=abs(norm-finite)/norm)
    assert check['relative_error']<.02,check
    dump('qk_magnitude_gradient_check.json',check);emit(stage='checks',zero=checks,gradient=check)
    fits=[];choices={}
    for mode in ARMS:
        runs=[]
        for lr in GRID[mode]:
            for seed in SEEDS:
                v=fit(model,a,data,mode,seed,lr,baseline);runs.append(v);fits.append(v)
                dump('fit_progress.json',fits)
        means={str(lr):float(np.mean([v['final_validation_nll'] for v in runs if v['lr']==lr])) for lr in GRID[mode]}
        best=min(GRID[mode],key=lambda lr:means[str(lr)])
        choices[mode]=dict(lr=best,validation_by_lr=means)
        dump('locked_choices.json',choices);emit(stage='lr_selected',arm=mode,**choices[mode])
    selected=[v for v in fits if v['lr']==choices[v['arm']]['lr']]
    a.mode='off';base_test=evaluate(model,data['test'],True);records=[]
    for run in selected:
        mode,seed=run['arm'],run['seed'];dest=P/f'test/{mode}_seed{seed}.json'
        if dest.exists():
            cached=json.loads(dest.read_text())
            if cached['lr']==run['lr']:
                records.append(cached);continue
        load(a,run);test=evaluate(model,data['test'],True)
        delta=np.array(test['sequence_nll'])-np.array(base_test['sequence_nll'])
        rec=dict(arm=mode,seed=seed,lr=run['lr'],parameters=run['parameters'],test=test,
            sequence_delta_nll=delta.tolist(),delta_nll=float(delta.mean()),
            perplexity_change_percent=float(100*np.expm1(delta.mean())))
        if mode!='q_lora12':
            a.collect_begin();evaluate(model,data['train']);mean,stats=a.collect_end()
            a.mean_s=mean;ab=evaluate(model,data['test'],True);a.mean_s=None
            ab_delta=np.array(ab['sequence_nll'])-np.array(base_test['sequence_nll'])
            rec.update(gate_statistics=stats,train_mean_gate_test=ab,
                train_mean_gate_perplexity_change_percent=float(100*np.expm1(ab_delta.mean())))
            torch.save(mean.detach().cpu(),P/f'train_mean_gate_{mode}_seed{seed}.pt')
        records.append(rec);dump(str(dest.relative_to(P)),rec)
        emit(stage='test',arm=mode,seed=seed,perplexity_change_percent=rec['perplexity_change_percent'],
             mean_gate_perplexity_change_percent=rec.get('train_mean_gate_perplexity_change_percent'))
    dump('results.json',dict(protocol=protocol,validation_baseline=baseline,test_baseline=base_test,
        choices=choices,fits=fits,selected_runs=selected,test_results=records))
    emit(stage='complete')

if __name__=='__main__':main()
