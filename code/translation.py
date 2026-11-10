"""Reanalyse the original L23 H4 confirmation captures and output interventions.

python code/translation.py — NumPy/Matplotlib only; no model download or new run.
All figures derive from the posted captured Q/K, prompts, and output measurements.
Pair indices are zero-based, split-half coordinates (j, j+64).
"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/translation'
OUT = ROOT / 'figures/analysis'
PAPER, GREEN, GRAY, ORANGE = '#f6f5ef', '#176452', '#7d8981', '#bd682b'
plt.rcParams.update({'font.size': 10, 'figure.facecolor': PAPER,
    'axes.facecolor': PAPER, 'axes.spines.top': False, 'axes.spines.right': False})

def read(name):
    return json.loads((DATA / name).read_text())

def bootstrap(x, seed=2026102218):
    """Keep all four questions (and their transfers) clustered within each table."""
    tables = np.asarray(x).reshape(32, 4, -1).mean((1, 2))
    draws = np.random.default_rng(seed).integers(0, 32, (3000, 32))
    return {'mean': float(tables.mean()),
            'table_bootstrap95': np.quantile(tables[draws].mean(1), [.025, .975]).tolist()}

def margin(x, target):
    foil = np.delete(x, target, axis=-1)
    mx = foil.max(-1)
    return x[..., target] - mx - np.log(np.exp(foil-mx[..., None]).sum(-1))

def compute():
    provenance = read('provenance.json')
    for name, digest in provenance['files_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    rows = read('prompts.json')
    with np.load(DATA/'L23-H4-captures.npz') as data:
        q, k, scores, omega = (data[n].astype(float) for n in ('q', 'k', 'scores', 'frequencies'))
    z, keys = q[..., :64]+1j*q[..., 64:], k[..., :64]+1j*k[..., 64:]
    band = slice(16, 32)
    t = np.arange(-64, 65)
    basis = np.exp(-1j*omega[band, None]*t)
    grid = np.linspace(-2, 2, 801)
    numer, denom = np.zeros((32, len(grid))), np.zeros(32)
    metrics = {n: [] for n in ('corr_translated', 'corr_unchanged', 'profile_advantage', 'rescue', 'wrong_rescue')}
    examples, angles, weights = [], [], []
    for g in range(128):
        rr = rows[3*g:3*g+3]
        assert len({tuple(r['query_positions']) for r in rr}) == 1
        target = rr[0]['target']
        by_transfer = {n: [] for n in metrics}
        for s, d in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1)):
            delta = rr[d]['translation']-rr[s]['translation']
            np.testing.assert_array_equal(np.array(rr[d]['label_positions'])-rr[s]['label_positions'], delta)
            zs, zd, ks, sc = z[g,s], z[g,d], keys[g,s], scores[g,s]
            patch = abs(zs)*zd/np.maximum(abs(zd), 1e-30)
            np.testing.assert_allclose(abs(patch), abs(zs), atol=3e-14)
            dk = ks[target]-np.delete(ks, target, axis=0).mean(0)
            source = zs[band]*np.conj(dk[band])
            actual = patch[band]*np.conj(dk[band])
            rotation = np.exp(1j*omega*delta)
            curves = np.array([np.einsum('j,jt->t',source,basis).real, np.einsum('j,jt->t',actual,basis).real,
                               np.einsum('j,jt->t',source*rotation[band],basis).real])/np.sqrt(128)
            direct = np.einsum('j,jt->t',source,np.exp(-1j*omega[band,None]*(t-delta))).real/np.sqrt(128)
            np.testing.assert_allclose(curves[2], direct, atol=3e-14)
            c0 = np.corrcoef(curves[1], curves[0])[0,1]
            ct = np.corrcoef(curves[1], curves[2])[0,1]
            by_transfer['corr_unchanged'].append(c0)
            by_transfer['corr_translated'].append(ct)
            by_transfer['profile_advantage'].append(ct-c0)
            dp = patch[band]-zs[band]
            def score(a,b): return (a*np.conj(b)).sum(-1).real/np.sqrt(128)
            base = margin(sc, target)
            donor = margin(sc+score(dp,ks[:,band]), target)-base
            for name, rot in [('rescue',rotation),('wrong_rescue',np.conj(rotation))]:
                moved = ks*rot
                st = sc+score(zs,moved-ks)
                benefit = margin(st+score(dp,moved[:,band]), target)-margin(st,target)
                by_transfer[name].append(benefit-donor)
            ang = np.angle(zd[band]*np.conj(zs[band]))
            w = abs(source)
            numer[g//4] += ((1-np.cos(ang-grid[:,None]*delta*omega[None,band]))*w).sum(-1)
            denom[g//4] += w.sum()
            if (s,d)==(0,2): angles.append(ang); weights.append(w)
            if g==0 and s==0: examples.append((delta, curves))
        for name in metrics: metrics[name].append(np.mean(by_transfer[name]))
    # Recompute independently from Q/K and check the original stored analysis.
    with np.load(DATA/'candidate-metrics.npz') as original:
        for name, values in metrics.items():
            np.testing.assert_allclose(values, original[name], atol=1e-10)
    summary = {n: bootstrap(v) for n,v in metrics.items()}
    reference = read('original-summary.json')
    for name, result in summary.items():
        np.testing.assert_allclose(list(result.values())[0], reference['confirmation_metrics'][name]['mean'], atol=1e-10)
        np.testing.assert_allclose(result['table_bootstrap95'], reference['confirmation_metrics'][name]['table_bootstrap95'], atol=1e-10)
    loss = numer/denom[:,None]
    fit = float(grid[loss.mean(0).argmin()])
    np.testing.assert_allclose(fit, reference['phase_scale_fit']['confirmation_scale_fit'])
    np.testing.assert_allclose(loss.mean(0), reference['phase_scale_fit']['meanloss'], atol=1e-12)
    draws = np.random.default_rng(2026102218).integers(0,32,(3000,32))
    fit_ci = np.quantile(grid[loss[draws].mean(1).argmin(1)], [.025,.975])
    np.testing.assert_allclose(fit_ci, reference['phase_scale_fit']['table_bootstrap95'])
    output = read('output-interventions.json')
    modes = ['native','source_phase','translated_phase','wrong_direction']
    vals = {m: np.array([r['target_logodds'] for r in output if r['mode']==m]) for m in modes}
    effects = {}
    for label, mode in [('source','source_phase'),('wrong','wrong_direction'),('native','native')]:
        effects[label] = bootstrap(vals['translated_phase']-vals[mode], 2026102217)
        for statistic in ('mean','table_bootstrap95'):
            np.testing.assert_allclose(effects[label][statistic], read('output-summary.json')['target_logodds']['corrected_minus_'+label][statistic])
    accuracy = {m: sum(r['first_token_correct'] for r in output if r['mode']==m) for m in modes}
    assert all(v==230 for v in accuracy.values())
    return t, omega, examples, np.array(angles), np.array(weights), summary, effects, fit

def save(fig,name):
    fig.savefig(OUT/name, dpi=170, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    t, omega, examples, angles, weights, summary, effects, fit = compute()
    fig, axes = plt.subplots(2,1,figsize=(8,6.3),layout='constrained')
    for ax,(delta,curves) in zip(axes,examples):
        for values,color,ls,label in zip(curves,[GRAY,GREEN,ORANGE],['-','-','--'],
                ['Original query phase','Actual moved-context query phase','Ideal translated source']):
            ax.plot(t, values, color=color, ls=ls, label=label)
        ax.set(title=f'List moved +{delta} tokens · first confirmation example',
               xlabel='Synthetic offset applied to source keys (tokens)',
               ylabel='Band score contrast')
        ax.axhline(0,color=GRAY,lw=.6); ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    save(fig,'translation-profiles.png')
    fig,axes = plt.subplots(2,1,figsize=(8,6.5),layout='constrained')
    names=['corr_unchanged','corr_translated']
    means=np.array([summary[n]['mean'] for n in names]); ci=np.array([summary[n]['table_bootstrap95'] for n in names])
    axes[0].barh([0,1], means,color=[GRAY,GREEN]);axes[0].errorbar(means,[0,1],xerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',color='black',capsize=4)
    axes[0].set(yticks=[0,1],yticklabels=['Unchanged source','Translated source'],xlim=(0,1),xlabel='Correlation with the actual phase-updated band profile',title='A translation-like component on 32 new tables')
    for i,m in enumerate(means):axes[0].text(.025,i,f'{m:.3f}',color='white',va='center')
    means=np.array([effects[n]['mean'] for n in ['source','wrong','native']]);ci=np.array([effects[n]['table_bootstrap95'] for n in ['source','wrong','native']])
    axes[1].barh([0,1,2],means,color=[GREEN,GREEN,ORANGE]);axes[1].errorbar(means,[0,1,2],xerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',color='black',capsize=4)
    axes[1].axvline(0,color=GRAY,lw=.8)
    axes[1].set(yticks=[0,1,2],yticklabels=['vs frozen source phase','vs wrong-sign correction','vs native phase'],xlabel='Change in output target-vs-other-label log odds',title='Geometric correction: small output effects; accuracy unchanged')
    save(fig,'translation-results.png')
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    observed=np.angle((np.exp(1j*angles)*weights).sum(0))
    ax.plot(np.arange(16,32),omega[16:32]*40,'--',color=ORANGE,label='Rigid +40-token prediction')
    ax.plot(np.arange(16,32),observed,'o-',color=GREEN,label='Observed weighted circular mean')
    ax.set(xlabel='Pair index (zero-based)',ylabel='Query phase change (radians)',title='The individual pairs depart from a rigid translation')
    ax.legend(fontsize=9);ax.grid(alpha=.15)
    save(fig,'translation-phases.png')
    print(json.dumps({'profile_advantage':summary['profile_advantage'],'phase_scale':fit,'output_effects':effects,'first_token_accuracy':'230/256 for all four conditions'},indent=2))

if __name__=='__main__':main()
