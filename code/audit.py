"""Recompute published numbers from archived numeric measurements, without a model."""
from pathlib import Path
import hashlib
import json
import numpy as np
from rope import fit_decay

ROOT = Path(__file__).resolve().parents[1]


def audit():
    manifest = json.loads((ROOT/'data/provenance.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    checked = {}
    for case in ['slow_actual_key_phases','key_quantization_small_radius']:
        folder = ROOT/'data'/case
        saved = json.loads((folder/'results.json').read_text())['results']
        baseline = [np.load(folder/f'native_doc{i}.npz') for i in range(4)]
        for name, expected in saved.items():
            paths = [folder/f'{name}_doc{i}.npz' for i in range(4)]
            if not all(p.exists() for p in paths):
                continue
            arrays = [np.load(p) for p in paths]
            delta = np.concatenate([a['loss']-b['loss'] for a,b in zip(arrays,baseline)])
            changed = np.concatenate([a['pred']!=b['pred'] for a,b in zip(arrays,baseline)])
            assert len(delta) == expected['tokens'] == 7168
            assert abs(float(delta.mean())-expected['loss_increase']) < 1e-7, name
            assert abs(float(changed.mean())-expected['changed_prediction_fraction']) < 1e-12, name
            checked[f'{case}/{name}'] = dict(loss_increase=float(delta.mean()),changed_fraction=float(changed.mean()))
    folder = ROOT/'data/inverse_distance_attention'
    arrays = np.load(folder/'fit_arrays.npz')
    means = np.load(folder/'dense_query_means.npz')['means'].mean(0)
    np.testing.assert_allclose(means, arrays['dense_mean'], atol=1e-12)
    alpha, r2 = fit_decay(means, arrays['bin_lo'], arrays['bin_hi'])
    np.testing.assert_allclose(alpha, arrays['dense_alpha'], atol=1e-10)
    np.testing.assert_allclose(r2, arrays['dense_r2'], atol=1e-10)
    eligible=(means.sum(-1)>=.01)&(means>0).all(-1)
    match=eligible&(alpha>=.8)&(alpha<=1.2)&(r2>=.8)
    assert int(eligible.sum())==333 and int(match.sum())==101
    # Reconstruct every engineered profile from the frozen vectors.
    a=np.load(ROOT/'data/rope_post_examples/vectors_and_profiles.npz')
    basis=np.exp(1j*a['omega'][:,None]*a['distances'][None])
    for i in range(4):
        coeff=a[f'query_{i}'][None]*a['native_keys'].conj()/np.sqrt(128)
        profile=np.einsum('kj,jd->kd',coeff,basis,optimize=False).real
        np.testing.assert_allclose(profile,a[f'full_{i}'],atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(a[f'query_{i}']),np.linalg.norm(a['native_query']),atol=1e-12)
    return dict(verified_original_files=len(manifest['files']),replayed_conditions=checked,
                decay=dict(eligible_heads=int(eligible.sum()),approximately_inverse_distance=int(match.sum()),
                           example_L2H5_alpha=float(alpha[2,5])),
                scope='Numerical replay and provenance checks; no new full-model inference.')


if __name__ == '__main__':
    report=audit()
    print(json.dumps(report,indent=2))
