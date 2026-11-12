"""Rerun one adapter or unchanged-architecture control on the posted dataset.

Install requirements-model.txt. Example:
python code/run_adapter_comparison.py --arm amplitude --device mps
Use --allow-download for uncached weights. This reruns a selected configuration,
not the original full LR search. Recorded tables are regenerated separately.
"""
from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True,choices=['baseline','phase','permuted','amplitude','combined','q_lora6','q_lora12','wq','full'])
    p.add_argument('--device',default='mps',choices=['cpu','mps','cuda'])
    p.add_argument('--seed',type=int,default=1101);p.add_argument('--steps',type=int,default=64)
    p.add_argument('--lr',type=float);p.add_argument('--allow-download',action='store_true')
    p.add_argument('--output',type=Path,default=Path('adapter-rerun.json'))
    args=p.parse_args()
    if not 1<=args.steps<=64:p.error('steps must be between 1 and 64')
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM
    from query_adapters import QueryAdapter
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_files_only=not args.allow_download,dtype=torch.float32,attn_implementation='sdpa').to(args.device).eval()
    model.requires_grad_(False);hooks=[];adapter=None
    lr=args.lr or {'phase':3e-4,'permuted':3e-4,'wq':1e-4,'full':3e-6}.get(args.arm,1e-3)
    if args.arm=='full':model.requires_grad_(True);params=list(model.parameters());groups=[dict(params=params,lr=lr)]
    elif args.arm=='wq':
        params=[l.self_attn.q_proj.weight for l in model.model.layers]
        for v in params:v.requires_grad_(True)
        groups=[dict(params=params,lr=lr)]
    elif args.arm!='baseline':
        cfg=model.config;adapter=QueryAdapter(cfg.num_hidden_layers,cfg.num_attention_heads,cfg.hidden_size,model.model.rotary_emb.inv_freq,args.arm,args.seed).to(args.device)
        params=list(adapter.parameters());groups=[dict(params=params,lr=lr)]
        if args.arm=='combined':groups=[dict(params=[adapter.A,adapter.B],lr=lr),dict(params=[adapter.phase],lr=3e-4)]
        for i,layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(lambda mod,inputs,out,i=i:adapter(i,inputs[0],out)))
    else:params=[];groups=[]
    rows=json.loads((ROOT/'data/adapters/dataset.json').read_text());data={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    assert len(data['train'])==128 and len(data['validation'])==48 and len(data['test'])==80
    def losses(batch):
        ids=torch.tensor([r['input_ids'] for r in batch],device=args.device)
        logits=model(input_ids=ids[:,:-1],use_cache=False).logits
        return F.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),reduction='none').reshape(len(batch),128).mean(-1)
    def evaluate(split):
        out=[]
        with torch.no_grad():
            for i in range(0,len(data[split]),2):out.extend(losses(data[split][i:i+2]).cpu().tolist())
        return dict(nll=float(np.mean(out)),sequence_nll=out)
    try:
        before=evaluate('validation');history=[]
        if params:
            opt=torch.optim.Adam(groups,foreach=False);order=np.random.default_rng(args.seed).permutation(128)
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True);loss=losses([data['train'][int(j)] for j in order[2*step:2*step+2]]).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();history.append(float(loss.detach()))
        result=dict(arm=args.arm,seed=args.seed,steps=0 if not params else args.steps,parameters=sum(x.numel() for x in params),lr=lr,validation_before=before,validation_after=evaluate('validation'),test=evaluate('test'),training_losses=history)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['training_losses','test','validation_before','validation_after']})
    finally:
        for h in hooks:h.remove()
if __name__=='__main__':main()
