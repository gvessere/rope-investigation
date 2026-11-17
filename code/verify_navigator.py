"""Verify every static navigator capture, including causal masks and pair scores."""
from pathlib import Path
import json,gzip,base64,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/navigator'
def main():
 m=json.loads((D/'manifest.json').read_text());assert len(m['files'])==len(m['heads'])*len(m['documents'])
 for r in m['files']:
  b=(D/r['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];x=json.loads(gzip.decompress(b))
  def dec(k,shape):return np.frombuffer(base64.b64decode(x[k]),dtype='<f4').reshape(shape)
  q=dec('q',(x['nq'],128));k=dec('k',(x['n'],128));s=dec('scores',(x['nq'],x['n']));p=dec('probs',s.shape)
  pos=np.array(x['positions']);mask=np.arange(x['n'])[None,:]>pos[:,None]
  assert not p[mask].any();assert (s[mask]==-1e9).all();np.testing.assert_allclose(p.sum(1),1,atol=2e-5)
  # Captured scores may differ slightly from reconstructed float64 pair sums.
  for i in [0,63,127]:
   for age in [0,3,511,1023]:
    j=int(pos[i])-age;theta=np.array(x['freq'])*age
    a=(q[i,:64].astype(float)*k[j,:64]+q[i,64:].astype(float)*k[j,64:])/np.sqrt(128)
    b=(q[i,:64].astype(float)*k[j,64:]-q[i,64:].astype(float)*k[j,:64])/np.sqrt(128)
    score=(a*np.cos(theta)+b*np.sin(theta)).sum()
    np.testing.assert_allclose(score,s[i,j],atol=x['audit']['max_reconstruction_error_recent']+2e-4,rtol=1e-5)
 print(f'Verified {len(m["files"])} navigator captures: hashes, array shapes, causal masks, row probabilities and pair-score reconstruction.')
if __name__=='__main__':main()
