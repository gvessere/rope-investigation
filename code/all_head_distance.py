"""Export all 336 full-history native profiles, without attention renormalization.

Run: python code/all_head_distance.py
Reads the posted summary of the original verified, frozen-Q/K reconstruction.
No model inference is needed to reproduce the viewer.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data/decay'

def main():
    source = D / 'full-history-summary.npz'
    provenance = json.loads((D / 'full-history-provenance.json').read_text())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == provenance['posted_summary_sha256']
    with np.load(source) as z:
        mass = z['mean_mass'].reshape(336, -1)
        per_key = z['mean_weight_per_available_key'].reshape(336, -1)
        bins = z['bins']  # Half-open intervals, with singleton self and ages 1–7.
        positions = z['query_positions'].ravel()
        opportunities = z['opportunities']
        count = len(positions)
        availability = (np.arange(bins[-1, 1])[None] <= positions[:, None]).sum(0)
        expected_opp = np.array([availability[a:b].sum() for a, b in bins])
        np.testing.assert_array_equal(opportunities, expected_opp)
        np.testing.assert_allclose(mass.sum(-1), 1, rtol=0, atol=1e-10)
        np.testing.assert_allclose(per_key * opportunities / count, mass, rtol=1e-12, atol=1e-14)
        assert mass.shape == (336, 17) and np.all(mass > 0)
        # The old seven-bin view must be reproduced exactly, before adding the missing mass.
        old = np.load(D / 'summary_arrays.npz')['heldout_mass']
        np.testing.assert_allclose(mass[:, 8:15], old, rtol=1e-10, atol=1e-12)
        centers = [float(a) if b == a+1 else float(np.expm1(np.log1p(np.arange(a,b)).mean())) for a,b in bins]
        unit_mass = [None if a == 0 else float((availability[a:b] / np.arange(a,b)).sum() / count) for a,b in bins]
        unit_per_key = [None if v is None else v * count / opportunities[j] for j,v in enumerate(unit_mass)]
        anchor = next(j for j,(a,b) in enumerate(bins) if (a,b) == (16,32))
        heads = []
        for i in range(336):
            amplitude = per_key[i,anchor] / unit_per_key[anchor]
            heads.append(dict(layer=i//12,head=i%12,mass=mass[i].tolist(),per_key=per_key[i].tolist(),
                              reference_mass=[None if v is None else amplitude*v for v in unit_mass],
                              reference_per_key=[None if v is None else amplitude*v for v in unit_per_key]))
    out = dict(source_sha256=sha,documents=20,queries_per_document=128,query_count=count,
               bins=[[int(a),int(b)-1] for a,b in bins],centers=centers,
               opportunities=opportunities.tolist(),effective_widths=(opportunities/count).tolist(),
               description='Full native causal attention, ages 0–2060, including self and every available key. No range renormalization. Per-key weights use actual available query-key opportunities; self has no 1/d guide. Guides match each head at ages 16–31.',heads=heads)
    (D / 'all-head-distance.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print('All 336 heads: full-history mass totals 100%, available-key counts checked, original seven bins reproduced.')

if __name__ == '__main__':
    main()
