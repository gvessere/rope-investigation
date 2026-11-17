"""Redraw the earlier logarithmic-distance heatmap and example curves.

Run: python code/distance_heatmaps.py [--layer 14]
Uses recorded bin totals only; no model inference, smoothing or fitted curves.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures import PAPER

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, choices=range(28), default=14)
    layer = parser.parse_args().layer
    path = ROOT / 'data/decay/retained-distance-bins.csv'
    provenance = json.loads(path.with_name('retained-distance-provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['export_sha256']
    rows = list(csv.DictReader(path.open()))
    bins = sorted({(int(r['age_min_inclusive']), int(r['age_max_inclusive']))
                   for r in rows if r['view'] == 'logbins'})
    mass = np.full((28, 12, len(bins)), np.nan)
    own = np.full((28, 12), np.nan)
    seen = set()
    for r in rows:
        l, h = int(r['layer']), int(r['head'])
        v = float(r['mean_retained_attention_mass_percent'])
        key = (r['view'], l, h, int(r['age_min_inclusive']))
        assert key not in seen
        seen.add(key)
        if r['view'] == 'self':
            own[l, h] = v
        else:
            mass[l, h, bins.index((int(r['age_min_inclusive']), int(r['age_max_inclusive'])))] = v
    assert np.isfinite(mass).all() and np.isfinite(own).all()
    assert bins[0][0] == 1 and bins[-1][1] == 2061
    assert all(bins[i][1] + 1 == bins[i + 1][0] for i in range(len(bins) - 1))
    totals = mass.sum(-1) + own
    assert (totals >= 99 - 1e-5).all() and (totals <= 100 + 1e-5).all()

    edges = np.array([.5] + [hi + .5 for lo, hi in bins])
    centers = np.sqrt([lo * hi for lo, hi in bins])
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    out = ROOT / 'figures/analysis'
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
    fig, (ax, self_ax) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True,
                                    gridspec_kw={'width_ratios': [20, 1]}, layout='constrained')
    y = np.arange(13) - .5
    im = ax.pcolormesh(edges, y, mass[layer], cmap='magma', vmin=0, vmax=100,
                       shading='flat', rasterized=True)
    self_ax.pcolormesh([-.5, .5], y, own[layer, :, None], cmap='magma', vmin=0, vmax=100)
    ax.set(xscale='log', xlim=(edges[-1], edges[0]), ylim=(11.5, -.5),
           yticks=range(12), yticklabels=[f'H{h}' for h in range(12)],
           ylabel='Query head', xlabel='Look-back distance (tokens; older ← → recent)')
    ax.set_xticks(ticks, [str(t) for t in ticks], rotation=45)
    ax.minorticks_off()
    self_ax.set_xticks([0], ['Self\n0'])
    self_ax.tick_params(axis='y', left=False, labelleft=False)
    fig.colorbar(im, ax=[ax, self_ax], shrink=.8, label='Mean attention mass per interval (%)')
    fig.suptitle(f'Layer {layer}: different heads look at different distances')
    fig.savefig(out / f'distance-heatmap-L{layer}.png', dpi=160)
    plt.close(fig)

    # Same heads as the original layer-14 sheet: three older-history examples,
    # a recent-history example, and two with a middle-distance hump.
    heads = [0, 4, 5, 8, 10, 11]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True, layout='constrained')
    ymax = mass[layer, heads].max() * 1.12
    for ax, h in zip(axes.flat, heads):
        ax.plot(centers, mass[layer, h], 'o-', color='#126552', lw=1.7, ms=4)
        ax.set(xscale='log', xlim=(2062, .8), ylim=(0, ymax),
               title=f'L{layer} H{h} · self {own[layer, h]:.1f}%')
        ax.set_xticks([1, 4, 16, 64, 256, 2048], ['1', '4', '16', '64', '256', '2048'])
        ax.minorticks_off()
        ax.grid(alpha=.18)
    fig.supylabel('Mean attention mass per interval (%)')
    fig.supxlabel('Look-back distance (tokens, log scale; older ← → recent)')
    fig.suptitle('The same recorded intervals, shown as curves')
    fig.savefig(out / f'distance-curves-L{layer}.png', dpi=160)
    plt.close(fig)
    print(f'Checked all 336 heads; regenerated layer {layer} heatmap and six curves.')

if __name__ == '__main__':
    main()
