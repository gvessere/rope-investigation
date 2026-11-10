"""Recompute article speed tables from recorded hardware measurements.

python code/speed_results.py
No inference is performed. The tables report throughput-style attention timings
and whole-model cached decode timings, not prompt-to-completion speedups.
"""
from pathlib import Path
import hashlib, json, statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/speed'
def read(name):return json.loads((DATA/name).read_text())
def compute():
    for name,digest in read('provenance.json')['sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    micro=read('attention-benchmark.json')['results']
    attention=[]
    for batch in [1,3,8,32]:
        rows={r['method']:r for r in micro if r['batch']==batch}
        base=rows['dense_grouped_matmul']['median_ms']
        for method in ['dense_grouped_matmul','compact_all_values','shortlist64_plus22','shortlist128_plus22']:
            r=rows[method]
            np.testing.assert_allclose(r['median_ms'],np.median(r['trial_ms']))
            attention.append(dict(batch=batch,method=method,ms=r['median_ms'],speedup=base/r['median_ms'],
                speedup_vs_strided_sdpa=rows['dense_sdpa']['median_ms']/r['median_ms']))
    raw=read('decode-timings.json')['results'];quality=read('decode-quality.json')['results']
    configs=[('dense',0),('all_values',0),('shortlist',64),('shortlist',128)]
    base=statistics.median(r['mean_ms_per_token_after8warmup'] for r in raw if r['mode']=='dense')
    native={r['document']:r for r in quality if r['mode']=='dense'}
    full=[]
    for mode,c in configs:
        rows=[r for r in raw if (r['mode'],r['C'])==(mode,c)]
        assert len(rows)==4
        means=[]
        for r in rows:
            mean=float(np.mean(r['step_ms'][8:]));means.append(mean)
            np.testing.assert_allclose(mean,r['mean_ms_per_token_after8warmup'])
        ms=statistics.median(means)
        qr=[r for r in quality if (r['mode'],r['C'])==(mode,c)]
        assert len(qr)==3
        delta=np.concatenate([np.array(r['losses'])-native[r['document']]['losses'] for r in qr])
        changed=np.mean([np.mean(np.array(r['predictions'])!=native[r['document']]['predictions']) for r in qr])
        full.append(dict(method=mode,candidates=c,ms=ms,speedup=base/ms,run_mean_range_ms=[min(means),max(means)],
            loss_increase=float(delta.mean()),changed_prediction_fraction=float(changed)))
    ref=read('original-summary.json')['summary']
    for a,b in zip(full,ref):
        np.testing.assert_allclose(a['ms'],b['counterbalanced_runtime_ms'])
        np.testing.assert_allclose(a['loss_increase'],b['loss_increase_vs_dense'])
        np.testing.assert_allclose(a['changed_prediction_fraction'],b['changed_prediction_fraction'])
    return dict(attention=attention,decode=full)

def tables(x):
    labels={'dense':'Dense grouped attention','all_values':'Compact scores, all values','shortlist64':'Compact shortlist, top 64 + 22 anchors','shortlist128':'Compact shortlist, top 128 + 22 anchors'}
    out=['<div class="table-scroll"><table><caption>Attention only · 32,000 keys · milliseconds per layer</caption><thead><tr><th>Batch</th><th>Dense grouped</th><th>Compact / all values</th><th>Top 64 + anchors</th><th>Top 128 + anchors</th></tr></thead><tbody>']
    for batch in [1,3,8,32]:
        rows=[r for r in x['attention'] if r['batch']==batch]
        out.append('<tr><td>'+str(batch)+'</td>'+''.join(f'<td>{r["ms"]:.3f} ms'+(f' · {r["speedup"]:.2f}×' if i else '')+'</td>' for i,r in enumerate(rows))+'</tr>')
    out.append('</tbody></table></div>')
    out.append('<div class="table-scroll"><table><caption>Whole-model cached decode · batch 1 · excludes prefill and index setup</caption><thead><tr><th>Method</th><th>ms/token</th><th>Speed-up</th><th>Loss increase (nats/token)</th><th>Choices changed</th></tr></thead><tbody>')
    for r in x['decode']:
        key=r['method']+(str(r['candidates']) if r['method']=='shortlist' else '')
        out.append(f'<tr><td>{labels[key]}</td><td>{r["ms"]:.2f}</td><td>{r["speedup"]:.2f}×</td><td>{r["loss_increase"]:+.4f}</td><td>{100*r["changed_prediction_fraction"]:.2f}%</td></tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)

if __name__=='__main__':
    result=compute()
    (DATA/'recomputed-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    start,end='<!-- SPEED TABLES START -->','<!-- SPEED TABLES END -->'
    p=ROOT/'index.html';s=p.read_text();assert s.count(start)==s.count(end)==1
    before,rest=s.split(start);_,after=rest.split(end)
    p.write_text(before+start+'\n'+tables(result)+'\n'+end+after)
    print('Recomputed attention/decode ratios, trial medians, and quality deltas; article tables updated.')
