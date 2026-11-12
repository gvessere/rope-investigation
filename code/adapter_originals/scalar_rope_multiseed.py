"""Matched-budget scalar rotations vs standard Q-only LoRA, paired seeds.

Uses the frozen local Qwen checkpoint. Pilot validation chooses one LR/method;
all main runs complete 64 steps, and test is opened after all runs finish.
"""
from scalar_rope_train import ROOT, OLD
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

P = ROOT/'outputs/scalar_rope_multiseed'
P.mkdir(exist_ok=True)
START = time.monotonic()
SEEDS = [1101,1102,1103,1104,1105]
PILOT = 9101
ARMS = ['coordinated','permuted','q_lora','constant']
GRID = {'coordinated':[1e-5,3e-5,1e-4,3e-4,1e-3],
        'permuted':[1e-5,3e-5,1e-4,3e-4,1e-3],
        'q_lora':[1e-4,3e-4,1e-3,3e-3,1e-2]}
POINTS = [8,16,32,64]

def dump(name,x):
    dest=P/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2))
    tmp.replace(dest)

def emit(**x):
    print(json.dumps(dict(seconds=round(time.monotonic()-START,2),**x)),flush=True)

def prepare(tok):
    old=json.loads((ROOT/'outputs/scalar_rope_train/dataset.json').read_text())
    rng=np.random.default_rng(2026111001)
    rows=[r for r in old if r['split']=='train']
    for bid in [11,1661,1342,2009]:
        text=(ROOT/f'outputs/pair_usage_prose/book_{bid}.txt').read_text(encoding='utf-8-sig')
        text=re.split(r'\*\*\* START OF .*?\*\*\*',text,maxsplit=1,flags=re.S)[1]
        text=re.split(r'\*\*\* END OF ',text,maxsplit=1)[0]
        paras=re.split(r'\n\s*\n',text)
        for split,lo,hi,n in [('validation',.60,.72,12),('test',.77,.90,20)]:
            start,end=int(len(paras)*lo),int(len(paras)*hi)
            ids=tok('\n\n'.join(paras[start:end]),add_special_tokens=False,verbose=False)['input_ids']
            used={r['token_offset']//129 for r in old if r['book']==bid and r['split']==split}
            eligible=[i for i in range(len(ids)//129) if i not in used]
            assert len(eligible)>=n,(bid,split,len(eligible))
            for block in rng.choice(eligible,n,replace=False):
                offset=int(block)*129
                window=ids[offset:offset+129]
                rows.append(dict(split=split,book=bid,paragraph_band=[start,end],token_offset=offset,
                                 input_ids=window,text=tok.decode(window)))
    assert len({tuple(r['input_ids']) for r in rows})==len(rows)
    assert not ({tuple(r['input_ids']) for r in rows if r['split']!='train'} &
                {tuple(r['input_ids']) for r in old})
    # Preserve the pilot training set exactly; shuffle held-out rows once.
    train=[r for r in rows if r['split']=='train']
    held=[r for r in rows if r['split']!='train']
    rng.shuffle(held)
    rows=train+held
    dump('dataset.json',rows)
    return {s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}

class Adapter(nn.Module):
    def __init__(self,model):
        super().__init__()
        c=model.config
        self.L=c.num_hidden_layers
        self.H=c.num_attention_heads
        self.D=c.hidden_size
        self.dh=self.D//self.H
        self.rank=6
        self.w=nn.Parameter(torch.zeros((self.L,self.H,self.D),device='mps'))
        self.A=nn.Parameter(torch.zeros((self.L,self.rank,self.D),device='mps'))
        self.B=nn.Parameter(torch.zeros((self.L,self.D,self.rank),device='mps'))
        self.bias=nn.Parameter(torch.zeros((self.L,self.H),device='mps'))
        self.register_buffer('omega',model.model.rotary_emb.inv_freq.detach().clone())
        self.register_buffer('freq',self.omega.clone())
        self.mode='off'
        self.collect=False
        self.stats=[]
        self.handles=[layer.self_attn.q_proj.register_forward_hook(self.hook(l))
                      for l,layer in enumerate(model.model.layers)]
        self.counts={mode:sum(p.numel() for p in self.selected(mode)) for mode in ARMS}
        assert self.counts['coordinated']==self.counts['permuted']==self.counts['q_lora']==516096

    def selected(self,mode):
        return {'coordinated':[self.w],'permuted':[self.w],
                'q_lora':[self.A,self.B],'constant':[self.bias]}[mode]

    def reset(self,mode,seed):
        self.zero_grad(set_to_none=True)
        gen=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.w.zero_();self.B.zero_();self.bias.zero_()
            # Standard LoRA initialization: A Kaiming-uniform, B zero; alpha/r=1.
            self.A.copy_((torch.rand(self.A.shape,generator=gen)*2-1).to('mps')/math.sqrt(self.D))
            perm=torch.randperm(len(self.omega),generator=torch.Generator().manual_seed(seed+700000))
            self.freq.copy_(self.omega[perm.to('mps')])
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.selected(mode):
            p.requires_grad_(True)
        self.mode=mode
        return perm.tolist()

    def hook(self,l):
        def apply(module,inputs,out):
            if self.mode=='off':
                return out
            x=inputs[0]
            if self.mode=='q_lora':
                diff=F.linear(F.linear(x,self.A[l]),self.B[l])
                result=out+diff
                scalar=None
            else:
                if self.mode=='constant':
                    scalar=2*torch.tanh(self.bias[l]/2).view(1,1,-1).expand(*x.shape[:-1],-1)
                else:
                    scalar=2*torch.tanh(F.linear(x,self.w[l])/2)
                omega=self.freq if self.mode=='permuted' else self.omega
                angle=-scalar.unsqueeze(-1)*omega
                q=out.view(*x.shape[:-1],self.H,self.dh)
                a,b=q.chunk(2,dim=-1)
                cs,sn=angle.cos(),angle.sin()
                result=torch.cat((a*cs-b*sn,a*sn+b*cs),dim=-1).reshape_as(out)
                diff=result-out
            if self.collect:
                value=0 if scalar is None else scalar.detach().square().sum()
                num=0 if scalar is None else scalar.numel()
                self.stats.append(torch.stack((diff.detach().square().sum(),out.detach().square().sum(),
                    out.new_tensor(diff.numel()),out.new_tensor(value) if isinstance(value,int) else value,
                    out.new_tensor(num))))
            return result
        return apply

    def get_stats(self):
        v=torch.stack(self.stats).detach().cpu().numpy().sum(0)
        self.stats=[]
        return dict(query_change_rms=float(np.sqrt(v[0]/v[2])),
                    query_relative_change_rms=float(np.sqrt(v[0]/v[1])),
                    scalar_rms=None if v[4]==0 else float(np.sqrt(v[3]/v[4])))

def loss_batch(model,rows,per_sequence=False):
    ids=torch.tensor([r['input_ids'] for r in rows],device='mps')
    logits=model(input_ids=ids[:,:-1],use_cache=False).logits
    if per_sequence:
        losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none')
        return losses.reshape(len(rows),128).mean(-1)
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))

def evaluate(model,a,rows,collect=False,per_sequence=False):
    a.collect=collect
    a.stats=[]
    vals=[]
    with torch.no_grad():
        for start in range(0,len(rows),2):
            batch=rows[start:start+2]
            loss=loss_batch(model,batch,per_sequence=True)
            vals.extend(loss.detach().cpu().tolist())
    a.collect=False
    result=dict(nll=float(np.mean(vals)),sequences=len(rows),tokens=len(rows)*128)
    if collect:
        result.update(a.get_stats())
    if per_sequence:
        result['sequence_nll']=vals
    return result

def run_train(model,a,data,mode,seed,lr,baseline,phase):
    stem=f'{phase}/{mode}_seed{seed}_lr{lr:g}'
    path=P/f'{stem}.json'
    if path.exists():
        return json.loads(path.read_text())
    perm=a.reset(mode,seed)
    order=np.random.default_rng(seed).permutation(len(data['train']))
    assert len(order)==128
    opt=torch.optim.Adam(a.selected(mode),lr=lr)
    checkpoints=[]
    history=[dict(step=0,validation=baseline)]
    losses=[]
    t0=time.monotonic()
    emit(stage='train_start',phase=phase,arm=mode,seed=seed,lr=lr)
    for step in range(1,65):
        batch=[data['train'][int(i)] for i in order[(step-1)*2:step*2]]
        opt.zero_grad(set_to_none=True)
        loss=loss_batch(model,batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a.selected(mode),1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (phase=='pilot' and step==64) or (phase=='main' and step in POINTS):
            val=evaluate(model,a,data['validation'],collect=True)
            history.append(dict(step=step,validation=val))
            emit(stage='validation',phase=phase,arm=mode,seed=seed,lr=lr,step=step,**val)
    state=[p.detach().cpu().clone() for p in a.selected(mode)]
    checkpoint=P/f'{stem}.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(mode=mode,seed=seed,lr=lr,values=state),checkpoint)
    result=dict(phase=phase,arm=mode,seed=seed,lr=lr,steps=64,
                trainable_parameters=a.counts[mode],history=history,online_train_losses=losses,
                final_validation_nll=history[-1]['validation']['nll'],
                permutation=perm if mode=='permuted' else None,
                checkpoint=str(checkpoint),elapsed_seconds=time.monotonic()-t0)
    dump(f'{stem}.json',result)
    a.zero_grad(set_to_none=True)
    del opt
    return result

def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026111000)
    name='Qwen/Qwen2.5-1.5B-Instruct'
    tok=AutoTokenizer.from_pretrained(name,local_files_only=True)
    data=prepare(tok)
    protocol=dict(model=name,original_weights_frozen=True,dtype='float32',attention='sdpa',
        main_seeds=SEEDS,pilot_seed=PILOT,steps=64,batch_size=2,predictions_per_sequence=128,
        training_tokens_per_run=16384,splits={k:len(v) for k,v in data.items()},
        train_set='Same 128 windows as original pilot; held-out validation/test are new nonoverlapping windows from the same disjoint book regions, excluding all original pilot windows.',
        seed_variation='Paired training-order shuffle; LoRA A initialization and permuted-frequency control use seed-derived random draws. Training data, validation and test are fixed across seeds.',
        primary_parameter_counts=dict(coordinated=516096,permuted=516096,q_lora=516096),
        constant_parameters=336,
        coordinated='One zero-initialized bias-free row per head: delta=2*tanh((w@input_layernorm_hidden)/2); rotate q pair j by -delta*omega_j before standard RoPE.',
        permuted='Same as coordinated with one seed-specific fixed permutation of omega; not a true token displacement.',
        q_lora='Q projection update BAx; rank6 full-layer LoRA, A shape6x1536 Kaiming uniform, B shape1536x6 zero; alpha/r=1; no learned additional biases. Original Q bias remains frozen.',
        exact_budget='28*12*1536 = 28*6*(1536+1536) = 516096. Extra predictor biases removed for exact equality; original pilot had336additionalbiases.',
        optimizer='Adam defaults, no weight decay; clip selected parameter global grad norm at1; model eval/dropout off.',
        learning_rate_grid=GRID,
        lr_selection='Separate pilot training-order/init seed, 64steps per LR, same five-trial validation effort per primary arm; select final validation NLL, then lock LR for all five main seeds.',
        constant_lr=.01,
        main_evaluation='Validation curves at steps0,8,16,32,64; final step64 test only, no early checkpoint selection. Test opened after all main training finishes.',
        limitations='Five optimization seeds on fixed small classic-prose dataset; 128-token contexts and one frozen checkpoint. Equal parameter counts/updates do not imply equal function families or FLOPs. Constant baseline intentionally has fewer parameters.')
    dump('protocol.json',protocol)
    emit(stage='loading',splits=protocol['splits'])
    model=AutoModelForCausalLM.from_pretrained(name,local_files_only=True,dtype=torch.float32,
        attn_implementation='sdpa').to('mps').eval()
    model.requires_grad_(False)
    a=Adapter(model)
    a.mode='off'
    baseline=evaluate(model,a,data['validation'])
    # All adapters must preserve the exact initial prediction loss.
    checks={}
    for mode in ARMS:
        a.reset(mode,PILOT)
        z=evaluate(model,a,data['validation'][:2])['nll']
        a.mode='off'
        off=evaluate(model,a,data['validation'][:2])['nll']
        checks[mode]=abs(z-off)
        assert checks[mode]<2e-6,(mode,z,off)
    dump('zero_checks.json',checks)
    pilots={}
    selected={}
    for mode in GRID:
        pilots[mode]=[run_train(model,a,data,mode,PILOT,lr,baseline,'pilot') for lr in GRID[mode]]
        best=min(pilots[mode],key=lambda r:r['final_validation_nll'])
        selected[mode]=best['lr']
        emit(stage='lr_selected',arm=mode,lr=best['lr'],validation_nll=best['final_validation_nll'])
        dump('pilot_results.json',dict(pilots=pilots,selected_lrs=selected))
    selected['constant']=.01
    dump('locked_learning_rates.json',selected)
    main_runs=[]
    for seed in SEEDS:
        for mode in ARMS:
            main_runs.append(run_train(model,a,data,mode,seed,selected[mode],baseline,'main'))
            dump('main_progress.json',main_runs)
    # All model/hyperparameter choices are complete before touching test loss.
    a.mode='off'
    baseline_test=evaluate(model,a,data['test'],per_sequence=True)
    test_results=[]
    for run in main_runs:
        dest=Path(run['checkpoint']).with_suffix('.test.json')
        if dest.exists():
            test_results.append(json.loads(dest.read_text()))
            continue
        a.reset(run['arm'],run['seed'])
        ck=torch.load(run['checkpoint'],weights_only=True)
        with torch.no_grad():
            for p,v in zip(a.selected(run['arm']),ck['values']):
                p.copy_(v.to('mps'))
        val=evaluate(model,a,data['test'],collect=True,per_sequence=True)
        delta=np.array(val['sequence_nll'])-np.array(baseline_test['sequence_nll'])
        result=dict(arm=run['arm'],seed=run['seed'],lr=run['lr'],test=val,
                    delta_nll=float(delta.mean()),perplexity_change_percent=float(100*np.expm1(delta.mean())),
                    sequence_delta_nll=delta.tolist())
        dump(str(dest.relative_to(P)),result)
        test_results.append(result)
        emit(stage='test',arm=run['arm'],seed=run['seed'],delta_nll=result['delta_nll'],
             perplexity_change_percent=result['perplexity_change_percent'])
    dump('results.json',dict(protocol=protocol,validation_baseline=baseline,test_baseline=baseline_test,
                             selected_lrs=selected,main_runs=main_runs,test_results=test_results))
    emit(stage='complete')

if __name__=='__main__':
    main()
