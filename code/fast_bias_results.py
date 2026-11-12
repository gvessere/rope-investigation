"""Recompute fast-key-bias effects from recorded per-token outputs; no inference."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 out={}
 for file,condition in [('fast-bias-rounding-results.json','k'),('fast-bias-removal-results.json','fast')]:
  raw=json.loads((ROOT/'data/analysis'/file).read_text());delta=[];changed=[]
  for a,b in zip(raw['baseline'],raw[condition]):
   assert a['passage_id']==b['passage_id'] and a['target_positions']==b['target_positions']
   delta.extend(np.array(b['token_nll'])-a['token_nll'])
   changed.extend(np.array(b['token_predictions'])!=a['token_predictions'])
  out[condition]=dict(tokens=len(delta),loss_increase=float(np.mean(delta)),changed_predictions=int(sum(changed)),changed_fraction=float(np.mean(changed)))
 assert out['k']['tokens']==out['fast']['tokens']==2251
 np.testing.assert_allclose(out['k']['loss_increase'],.002713323645674351,atol=1e-7)
 np.testing.assert_allclose(out['fast']['loss_increase'],1.0276046569270063,atol=1e-7)
 (ROOT/'data/analysis/fast-bias-summary.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
