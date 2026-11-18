"""Export all 336 native distance profiles for the article's interactive viewer.

Run: python code/all_head_distance.py
Native full-softmax attention mass, without conditioning on the plotted range.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data/decay'

def main():
    source = D / 'summary_arrays.npz'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == json.loads((D / 'provenance.json').read_text())['files'][source.name]
    with np.load(source) as z:
        mass = z['heldout_mass']
        widths = z['bin_width']
        age = z['geometric_mean_age']
        assert mass.shape == (336, 7) and np.all(mass > 0)
        total = mass.sum(-1)
        conditional = mass / total[:, None]
        per_key = mass / widths
        np.testing.assert_allclose(conditional.sum(-1), 1)
        assert np.all((total > 0) & (total <= 1))
        reference = 1 / age
        reference *= np.exp(np.log(per_key[2 * 12 + 5] / reference).mean())
        heads = [dict(layer=i // 12, head=i % 12,
                      range_mass=float(total[i]),
                      mass=mass[i].tolist(),
                      per_key=per_key[i].tolist(),
                      fit_lambda=float(z['heldout_alpha'][i]),
                      fit_r2=float(z['heldout_r2'][i])) for i in range(336)]
    out = dict(source_sha256=sha, documents=20, queries_per_document=128,
               bins=[[2**j, 2**(j+1)-1] for j in range(3, 10)],
               widths=widths.tolist(), centers=age.tolist(), reference=reference.tolist(),
               description='Native held-out full-softmax measurements. Pool mean interval mass with the original full causal softmax denominator, then divide by bin width for per-key view. No plotted-range renormalization. Reference is a 1/d line anchored to L2 H5 in these full-attention units. No historical/pruned-run data.',
               heads=heads)
    (D / 'all-head-distance.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print('Exported and checked all 28 × 12 heads from the native held-out measurements.')

if __name__ == '__main__':
    main()
