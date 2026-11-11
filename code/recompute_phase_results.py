"""Reanalyse original all-position model outputs; no new inference is performed."""
from pathlib import Path
import json
import numpy as np
from small_bias_removal import remove_small_slow_biases
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/analysis'
def main():
    x=np.load(D/'all-position-phase-evaluation.npz')
    protocol=json.loads((D/'phase-evaluation-protocol.json').read_text())
    results={}
    for name in ['pi2_small','pi4_small']:
        np.testing.assert_array_equal(x[name+'_gold'],x['native_gold'])
        delta=x[name+'_loss'].astype(float)-x['native_loss'].astype(float)
        changed=x[name+'_pred']!=x['native_pred']
        r=dict(tokens=delta.size,loss_increase=float(delta.mean()),changed_predictions=int(changed.sum()),changed_prediction_fraction=float(changed.mean()),per_document_loss_increase=delta.mean(axis=1).tolist())
        ref=protocol['reference_results'][name]
        np.testing.assert_allclose(r['loss_increase'],ref['loss_increase'],atol=1e-7)
        assert r['changed_prediction_fraction']==ref['changed_prediction_fraction']
        results[name]=r
    (D/'all-position-phase-summary.json').write_text(json.dumps(dict(all_token_positions=True,results=results),indent=2)+'\n')
    raw=json.loads((D/'small-slow-bias-evaluation.json').read_text());p=json.loads((D/'small-slow-bias-protocol.json').read_text())
    delta=[];changed=[]
    for a,b in zip(raw['baseline'],raw['slow']):
        assert a['passage_id']==b['passage_id'] and a['target_positions']==b['target_positions']
        delta.extend(np.array(b['token_nll'])-a['token_nll'])
        changed.extend(np.array(b['token_predictions'])!=a['token_predictions'])
    biases=np.load(D/'projection-biases.npz')['keys'];edited,mask=remove_small_slow_biases(biases)
    assert mask.sum()==p['selected_bias_pairs']==707
    np.testing.assert_array_equal(edited[~mask],biases[~mask])
    r=dict(intervention=p['intervention'],scored_tokens=len(delta),loss_increase=float(np.mean(delta)),changed_predictions=int(sum(changed)),changed_prediction_fraction=float(np.mean(changed)),selected_bias_pairs=int(mask.sum()),eligible_bias_pairs=28*2*15)
    assert r['scored_tokens']==2251 and r['changed_predictions']==1
    np.testing.assert_allclose(r['loss_increase'],p['original_summary']['loss_increase'],atol=1e-7)
    (D/'small-slow-bias-summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(phase=results,bias_removal=r),indent=2))
if __name__=='__main__':main()
