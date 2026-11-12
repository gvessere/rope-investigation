"""Rerun learned query-bias phase rounding on saved token IDs.

Install requirements-model.txt. Use --allow-download for uncached pinned weights.
Rounds selected slow query-bias phases to nearest pi/2, preserving radius.
"""
from pathlib import Path
import argparse,json
import numpy as np
from bias_phase_results import snap_query_biases
ROOT=Path(__file__).resolve().parents[1]
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--condition',required=True,choices=['large','small'])
    p.add_argument('--output',type=Path,default=Path('query-bias-rounding-rerun.json'))
    args=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision=REVISION,local_files_only=not args.allow_download,dtype=torch.float32,attn_implementation='eager').to(args.device).eval()
    projections=[l.self_attn.q_proj for l in model.model.layers]
    originals=[x.bias.detach().clone() for x in projections]
    z=np.stack([b.cpu().numpy().reshape(12,128)[:,:64]+1j*b.cpu().numpy().reshape(12,128)[:,64:] for b in originals])
    np.testing.assert_array_equal(z,np.load(ROOT/'data/analysis/projection-biases.npz')['queries'])
    edited,mask=snap_query_biases(z,args.condition);assert mask.sum()==(922 if args.condition=='large' else 4118)
    rows=json.loads((ROOT/'data/analysis/bias-removal-tokens.json').read_text());results={}
    try:
        with torch.inference_mode():
            for mode in ['baseline','slow']:
                if mode=='slow':
                    for layer,projection in enumerate(projections):
                        packed=np.concatenate([edited[layer].real,edited[layer].imag],axis=-1).reshape(-1)
                        projection.bias.copy_(torch.as_tensor(packed,device=args.device))
                outrows=[]
                for row in rows:
                    ids=np.array(row['input_ids']);target=np.array(row['body'])[8:]
                    out=model(input_ids=torch.tensor(ids[None],device=args.device),use_cache=False)
                    logits=out.logits[0,torch.tensor(target-1,device=args.device)].float()
                    gold=torch.tensor(ids[target],device=args.device)
                    loss=torch.nn.functional.cross_entropy(logits,gold,reduction='none').cpu().numpy()
                    pred=logits.argmax(-1).cpu().numpy()
                    outrows.append(dict(passage_id=row['id'],n=len(target),target_positions=target.tolist(),token_nll=loss.tolist(),token_predictions=pred.tolist()))
                    del out,logits
                results[mode]=outrows
    finally:
        with torch.no_grad():
            for projection,original in zip(projections,originals):projection.bias.copy_(original)
    args.output.write_text(json.dumps(dict(protocol={'revision':REVISION,'device':args.device,'all_token_positions':True,'condition':args.condition,'selected_bias_pairs':int(mask.sum())},results=results),indent=2)+'\n')
if __name__=='__main__':main()
