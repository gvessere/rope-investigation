"""Frozen seed-1101 gain adapter: attention peak diagnostics and interventions.

No fitting. Fixed native/full reference peaks are recorded separately per batch.
Peak swaps preserve the complete probability multiset (and hence entropy).
Local ungated controls use the SAME incoming hidden states and keys.
"""
from scalar_rope_train import ROOT
import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

P=ROOT/'outputs/amplitude_attention_peaks'; P.mkdir(exist_ok=True)
START=time.monotonic()
MODES=['native','full','full_native_peak','full_local_peak','native_full_peak',
       'full_random_peak','drop_shift_gains','drop_stay_gains','drop_random_gains']
def emit(**kw): print(json.dumps(dict(seconds=round(time.monotonic()-START,1),**kw)),flush=True)
def dump(name,x):
    f=P/name; tmp=f.with_suffix(f.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2)); tmp.replace(f)

def swap_peak(p,target):
    current=p.argmax(-1,keepdim=True); target=target.unsqueeze(-1)
    a=p.gather(-1,current); b=p.gather(-1,target)
    return p.scatter(-1,current,b).scatter(-1,target,a)

class Probe(nn.Module):
    def __init__(self,model,checkpoint):
        super().__init__(); ck=torch.load(checkpoint,weights_only=True)
        self.A=ck['A'].to('mps'); self.B=ck['B'].to('mps')
        self.mode='native'; self.raw={}; self.pos={}; self.native={};self.full={}
        self.local={}; self.random_mask={}; self.random_target={};self.diag={}
        self.native_p={}; self.operations=[]; self.audit=[]
        self.handles=[]
        for l,layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self.pre(l),with_kwargs=True))
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(self.hook(l)))
    def pre(self,l):
        def f(module,args,kwargs): self.pos[l]=kwargs['position_embeddings']
        return f
    def hook(self,l):
        def f(module,inputs,out):
            self.raw[l]=out.view(*out.shape[:-1],12,128).transpose(1,2)
            z=F.linear(F.linear(inputs[0],self.A[l]),self.B[l]).view(*out.shape[:-1],12,64)
            s=2*torch.tanh(z/2)
            if self.mode in ['drop_shift_gains','drop_stay_gains','drop_random_gains']:
                changed=self.native[l]!=self.full[l]
                drop={'drop_shift_gains':changed,'drop_stay_gains':~changed,
                      'drop_random_gains':self.random_mask[l]}[self.mode]
                s=s*(~drop).transpose(1,2).unsqueeze(-1)
            g=s.exp(); q=out.view(*out.shape[:-1],12,128)
            return (q*torch.cat((g,g),-1)).reshape_as(out)
        return f
    def attention(self,module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        assert dropout==0
        l=module.layer_idx
        k=repeat_kv(key,module.num_key_value_groups); v=repeat_kv(value,module.num_key_value_groups)
        cos,sin=self.pos.pop(l); raw=self.raw.pop(l)
        q0=raw*cos.unsqueeze(1)+rotate_half(raw)*sin.unsqueeze(1)
        mask=attention_mask[:,:,:,:k.shape[-2]] if attention_mask is not None else None
        s0=torch.matmul(q0,k.transpose(-1,-2))*scaling
        sg=torch.matmul(query,k.transpose(-1,-2))*scaling
        if mask is not None: s0=s0+mask;sg=sg+mask
        else:
            causal=torch.arange(k.shape[-2],device=query.device)[None,:]>torch.arange(query.shape[-2],device=query.device)[:,None]
            s0=s0.masked_fill(causal,float('-inf'));sg=sg.masked_fill(causal,float('-inf'))
        p0=s0.softmax(-1);pg=sg.softmax(-1)
        if self.mode=='native':
            p=p0; self.native[l]=p.argmax(-1);self.native_p[l]=p
        elif self.mode=='full':
            p=pg; self.full[l]=p.argmax(-1);self.local[l]=p0.argmax(-1)
            self.diagnostics(l,p0,pg)
            changed=self.native[l]!=self.full[l]
            # Random ablation preserves the changed-row count within each layer,
            # head and query position by permuting the batch axis independently.
            # Randomization is a control, not another trained-model seed.
            rng=np.random.default_rng(1101+1009*self.batch_offset+l)
            arr=changed.cpu().numpy(); rand=np.empty_like(arr)
            for h in range(12):
                for t in range(arr.shape[-1]): rand[:,h,t]=arr[rng.permutation(arr.shape[0]),h,t]
            self.random_mask[l]=torch.tensor(rand,device='mps')
            # Random competitor is causal and differs from the full reference winner.
            t=np.arange(arr.shape[-1])[None,None,:]
            u=torch.tensor(rng.random(arr.shape),device='mps',dtype=torch.float32)
            n=torch.arange(arr.shape[-1],device='mps').view(1,1,-1)
            candidate=(u*n).long()
            candidate=candidate+(candidate>=self.full[l]).long()
            self.random_target[l]=torch.where(changed,candidate,self.full[l])
        elif self.mode=='full_native_peak': p=swap_peak(pg,self.native[l])
        elif self.mode=='full_local_peak': p=swap_peak(pg,p0.argmax(-1))
        elif self.mode=='native_full_peak': p=swap_peak(p0,self.full[l])
        elif self.mode=='full_random_peak': p=swap_peak(pg,self.random_target[l])
        else: p=pg
        if self.mode in ['full_native_peak','full_local_peak','native_full_peak','full_random_peak']:
            source=p0 if self.mode=='native_full_peak' else pg
            entropy=lambda a: -(a*a.clamp_min(1e-30).log()).sum(-1)
            valid=torch.arange(p.shape[-2],device='mps')>0
            vals=torch.stack(((p.argmax(-1)!=source.argmax(-1))[...,valid].float().mean(),
                (p-source).abs().sum(-1)[...,valid].mean(),
                (entropy(p)-entropy(source)).abs().max(),(p.sum(-1)-1).abs().max()))
            self.operations.append(vals.cpu().numpy())
            target={'full_native_peak':self.native[l],'full_local_peak':p0.argmax(-1),
                    'native_full_peak':self.full[l],'full_random_peak':self.random_target[l]}[self.mode]
            # Floating-point softmax can tie several maxima. Swapping ensures
            # the target is a maximum; argmax's first-index tie break can differ.
            assert torch.equal(p.gather(-1,target.unsqueeze(-1)).squeeze(-1),p.max(-1).values), (self.mode,l)
        return torch.matmul(p,v).transpose(1,2).contiguous(), None
    def diagnostics(self,l,p0,pg):
        pn=self.native_p.pop(l); idx=torch.arange(pg.shape[-1],device='mps')
        ent=lambda p: -(p*p.clamp_min(1e-30).log()).sum(-1)
        vals=dict(native_peak=self.native[l],full_peak=self.full[l],local_peak=self.local[l],
            native_max=pn.max(-1).values,full_max=pg.max(-1).values,
            native_entropy=ent(pn),full_entropy=ent(pg),
            native_center=(pn*idx).sum(-1),full_center=(pg*idx).sum(-1),
            total_variation=.5*(pg-pn).abs().sum(-1),
            local_total_variation=.5*(pg-p0).abs().sum(-1))
        self.diag.setdefault(l,[]).append({k:v.cpu().numpy() for k,v in vals.items()})
    def close(self):
        for h in self.handles:h.remove()

def forward_losses(model,rows):
    ids=torch.tensor([r['input_ids'] for r in rows],device='mps')
    logits=model(input_ids=ids[:,:-1],use_cache=False).logits.float()
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').view(len(rows),-1)

def main():
    torch.set_num_threads(4);torch.manual_seed(1101)
    old=json.loads((ROOT/'outputs/banded_query_amplitude/results.json').read_text())
    run=next(r for r in old['runs'] if r['kind']=='pair' and r['band']=='all_0_63' and r['lr']==.001)
    rows=json.loads((ROOT/'outputs/scalar_rope_multiseed/dataset.json').read_text())
    test=[r for r in rows if r['split']=='test']
    dump('protocol.json',dict(seed=1101,checkpoint=run['checkpoint'],training='No new training or model seeds.',
        dataset='Existing 80 held-out windows, 128 next-token predictions each. Reused exploratory test.',
        reference='Native and full-adapter peaks from separate unmodified forward passes; local ungated peaks use identical incoming hidden states/keys to gated attention.',
        modes=MODES,swap='Swap probabilities of current maximum and requested target. Preserves full probability multiset and entropy exactly (up to floating point). Changes relative value routing.',
        masks='Gain-removal masks fixed from native/full reference peak differences, not recomputed under intervention. Random masks shuffle sequences within each layer/head/query-position; same row count, but batch-limited randomization.',
        caveats='Intervention effects interact across layers; necessity/sufficiency probes, not an additive mediation decomposition. Peak relocation does not test continuous fixed-content RoPE waveform displacement.'))
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True,
        dtype=torch.float32,attn_implementation='sdpa').to('mps').eval();model.requires_grad_(False)
    # Independent native SDPA reference verifies custom eager attention numerics.
    with torch.no_grad(): sdpa=forward_losses(model,test[:4]).cpu().numpy()
    probe=Probe(model,run['checkpoint'])
    ALL_ATTENTION_FUNCTIONS.register('amplitude_peak_probe',probe.attention)
    model.config._attn_implementation='amplitude_peak_probe'
    losses={m:[] for m in MODES};operations={m:[] for m in MODES}
    with torch.no_grad():
        for off in range(0,len(test),4):
            batch=test[off:off+4];probe.batch_offset=off
            for mode in MODES:
                probe.mode=mode;probe.operations=[]
                values=forward_losses(model,batch).cpu().numpy();losses[mode].append(values)
                operations[mode].extend(probe.operations)
                if off==0 and mode=='native':
                    err=float(np.max(np.abs(sdpa-values)));assert err<1e-4,err
                    dump('implementation_check.json',dict(max_token_nll_difference_eager_vs_native_sdpa=err))
            emit(stage='batch',sequences=off+len(batch),total=len(test),
                 running_nll={m:float(np.concatenate(losses[m]).mean()) for m in MODES})
            dump('progress.json',dict(sequences=off+len(batch),total=len(test)))
    arrays={f'loss_{m}':np.concatenate(losses[m]) for m in MODES}
    for l,batches in probe.diag.items():
        for k in batches[0]: arrays[f'{k}_layer{l}']=np.concatenate([b[k] for b in batches])
    np.savez_compressed(P/'diagnostics.npz',**arrays)
    base=arrays['loss_native'].mean();full=arrays['loss_full'].mean(); result={}
    for mode in MODES:
        a=arrays[f'loss_{mode}']; seq=a.mean(-1)
        result[mode]=dict(nll=float(a.mean()),perplexity_improvement_percent=float(-100*np.expm1(a.mean()-base)),
            retained_nll_improvement_fraction=float((base-a.mean())/(base-full)),sequence_nll=seq.tolist())
        if operations[mode]:
            z=np.stack(operations[mode]);result[mode]['swap_audit']=dict(mean_changed_row_fraction=float(z[:,0].mean()),
                mean_L1_probability_change=float(z[:,1].mean()),max_entropy_difference=float(z[:,2].max()),max_probability_sum_error=float(z[:,3].max()))
    dump('results.json',dict(seed=1101,records=result));probe.close();emit(stage='complete',records={k:{a:b for a,b in v.items() if a!='sequence_nll'} for k,v in result.items()})

if __name__=='__main__':main()
