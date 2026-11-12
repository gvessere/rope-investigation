"""Recompute large/small learned-query-bias phase rounding from recorded tokens."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/analysis'
def snap_query_biases(z,condition):
    z=np.asarray(z,dtype=np.complex128);radius=abs(z);phase=np.angle(z)
    mask=np.zeros(z.shape,dtype=bool)
    mask[...,49:64]=(radius[...,49:64]>2.5) if condition=='large' else (radius[...,49:64]<=2.5)
    snapped=radius*np.exp(1j*(np.pi/2)*np.round(phase/(np.pi/2)))
    return np.where(mask,snapped,z),mask

def main():
    provenance=json.loads((D/'query-bias-phase-provenance.json').read_text())
    for f,v in provenance.items():assert hashlib.sha256((D/f).read_bytes()).hexdigest()==v['sha256']
    z=np.load(D/'projection-biases.npz')['queries'].astype(np.complex128);rows=json.loads((D/'bias-removal-tokens.json').read_text());tokens={r['id']:r['input_ids'] for r in rows}
    result={}
    for name in ['large','small']:
        raw=json.loads((D/f'query-bias-{name}-results.json').read_text());ref=json.loads((D/f'query-bias-{name}-protocol.json').read_text());delta=[];changed=[];accuracy=[]
        edited,mask=snap_query_biases(z,name)
        assert mask.sum()==ref['selected_query_bias_pairs']
        np.testing.assert_array_equal(edited[~mask],z[~mask]);np.testing.assert_allclose(abs(edited),abs(z),atol=1e-12)
        for a,b in zip(raw['baseline'],raw['q']):
            assert a['passage_id']==b['passage_id'] and a['target_positions']==b['target_positions']
            delta.extend(np.array(b['token_nll'])-a['token_nll'])
            pa=np.array(a['token_predictions']);pb=np.array(b['token_predictions'])
            gold=np.array(tokens[a['passage_id']])[a['target_positions']]
            changed.extend(pb!=pa);accuracy.extend((pb==gold).astype(int)-(pa==gold).astype(int))
        v=dict(selected_bias_pairs=int(mask.sum()),scored_tokens=len(delta),loss_change=float(np.mean(delta)),changed_choices=int(sum(changed)),changed_fraction=float(np.mean(changed)),accuracy_change_pp=float(100*np.mean(accuracy)))
        np.testing.assert_allclose(v['loss_change'],ref['conditions']['q']['loss_increase'],atol=2e-7)
        np.testing.assert_allclose(v['accuracy_change_pp'],ref['conditions']['q']['accuracy_change_pp'],atol=1e-10)
        assert v['changed_fraction']==ref['conditions']['q']['changed_top1_fraction'];result[name]=v
    (D/'query-bias-phase-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='<div class="table-scroll"><table><caption>Query biases only · pairs 49–63 · phase rounded to nearest π/2</caption><thead><tr><th>Selected biases</th><th>Pairs edited</th><th>Loss change (nats/token)</th><th>Accuracy change (percentage points)</th><th>Choices changed</th></tr></thead><tbody>'
    for name,label in [('large','r &gt; 2.5'),('small','r ≤ 2.5')]:
        v=result[name];table+=f'<tr><td>{label}</td><td>{v["selected_bias_pairs"]:,} / 5,040</td><td>{v["loss_change"]:+.5f}</td><td>{v["accuracy_change_pp"]:+.2f}</td><td>{100*v["changed_fraction"]:.2f}%</td></tr>'
    table+='</tbody></table></div>'
    p=ROOT/'rope-in-practice.html';s=p.read_text();a,b='<!-- BIAS PHASE TABLE START -->','<!-- BIAS PHASE TABLE END -->';before,rest=s.split(a);_,after=rest.split(b);p.write_text(before+a+'\n'+table+'\n'+b+after)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
