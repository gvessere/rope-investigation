"""Rerun the recorded slow-key phase experiment using the exact saved token IDs.

Install requirements-model.txt first. Example, with already cached weights:
python code/run_phase_quantization.py --device mps --output rerun-results.json
Add --allow-download for first model download; --protect-prefix 14 for the control.
Reference runs used float32, eager attention, MPS, and the pinned model revision.
Hardware/library differences can alter numeric results. This script does not pack
codes into a compressed cache or benchmark inference speed.
"""
from pathlib import Path
import argparse,json
import numpy as np
from phase_quantization import round_keys_torch
ROOT=Path(__file__).resolve().parents[1]
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--protect-prefix',type=int,default=0)
    p.add_argument('--output',type=Path,default=Path('rerun-results.json'))
    args=p.parse_args()
    if args.protect_prefix<0:p.error('Prefix count must be nonnegative')
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,local_files_only=not args.allow_download,dtype=torch.float32,attn_implementation='eager').to(args.device).eval()
    cfg=model.config
    assert cfg.num_hidden_layers==28 and cfg.num_key_value_heads==2 and cfg.hidden_size//cfg.num_attention_heads==128
    saved=np.load(ROOT/'data/analysis/projection-biases.npz')
    for l,layer in enumerate(model.model.layers):
        for kind,projection in [('queries',layer.self_attn.q_proj),('keys',layer.self_attn.k_proj)]:
            b=projection.bias.detach().float().cpu().numpy().reshape(-1,128)
            np.testing.assert_array_equal(b[:,:64]+1j*b[:,64:],saved[kind][l])
    rows=json.loads((ROOT/'data/analysis/quantization-tokens.json').read_text())
    state={'bits':None,'selected':0,'eligible':0}
    def hook(module,inputs,output):
        if state['bits'] is None:return output
        changed,n,total=round_keys_torch(output,state['bits'],args.protect_prefix)
        state['selected']+=n;state['eligible']+=total
        return changed
    hooks=[layer.self_attn.k_proj.register_forward_hook(hook) for layer in model.model.layers]
    baseline=[];results={}
    try:
        with torch.inference_mode():
            for name,bits in [('native',None),('identity','identity'),('zero_small',0),('pi2_small',2),('pi4_small',3)]:
                state.update(bits=bits,selected=0,eligible=0);diffs=[];changes=[];perdoc=[]
                for i,row in enumerate(rows):
                    ids=np.array(row['input_ids']);target=np.array(row['body'])[256:]
                    x=torch.tensor(ids[None],device=args.device);ix=torch.tensor(target-1,device=args.device);gold=torch.tensor(ids[target],device=args.device)
                    out=model(input_ids=x,use_cache=False);logits=out.logits[0,ix].float()
                    loss=torch.nn.functional.cross_entropy(logits,gold,reduction='none').cpu().numpy();pred=logits.argmax(-1).cpu().numpy()
                    if bits is None:baseline.append((loss,pred))
                    d=loss-baseline[i][0];diffs.append(d);changes.append(pred!=baseline[i][1]);perdoc.append(float(d.mean()))
                    del out,logits
                    if args.device=='mps':torch.mps.empty_cache()
                    if args.device=='cuda':torch.cuda.empty_cache()
                results[name]={'tokens':sum(len(d) for d in diffs),'loss_increase':float(np.concatenate(diffs).mean()),'changed_prediction_fraction':float(np.concatenate(changes).mean()),'per_document_loss_increase':perdoc,'selected_fraction':state['selected']/max(1,state['eligible'])}
                print(name,results[name],flush=True)
    finally:
        for hook in hooks:hook.remove()
    assert results['identity']['changed_prediction_fraction']==0 and abs(results['identity']['loss_increase'])<1e-4
    args.output.write_text(json.dumps({'results':results,'protocol':{'model':args.model,'revision':args.revision,'dtype':'float32','attention':'eager','device':args.device,'protect_prefix':args.protect_prefix,'torch':torch.__version__,'description':'Actual pre-RoPE slow key pairs 49–63, small radius<2.5; full downstream recomputation. Not a packed-cache benchmark.'}},indent=2)+'\n')

if __name__=='__main__':main()
