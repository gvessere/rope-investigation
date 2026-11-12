"""Frozen-Qwen scalar-displacement gradient screen and short adapter training.

Run with the existing research venv. No network or original checkpoint writes.
The q_proj hook rotates BEFORE standard RoPE; the rotations commute.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/Users/geryvessere/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work')
os.environ['HF_HOME'] = str(OLD / 'hf')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import copy
import hashlib
import json
import math
import re
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

P = ROOT / 'outputs/scalar_rope_train'
P.mkdir(exist_ok=True)
START = time.monotonic()

def emit(**x):
    print(json.dumps(dict(seconds=round(time.monotonic()-START, 2), **x)), flush=True)

def dump(name, x):
    (P / name).write_text(json.dumps(x, indent=2))

def dataset(tok):
    """Unwrapped prose; paragraph-level disjoint splits inside each book."""
    rng = np.random.default_rng(202611091)
    rows = []
    specs = [('train', .15, .55, 32), ('validation', .60, .72, 4), ('test', .77, .90, 8)]
    for bid in [11, 1661, 1342, 2009]:
        f = ROOT / f'outputs/pair_usage_prose/book_{bid}.txt'
        text = f.read_text(encoding='utf-8-sig')
        text = re.split(r'\*\*\* START OF .*?\*\*\*', text, maxsplit=1, flags=re.S)[1]
        text = re.split(r'\*\*\* END OF ', text, maxsplit=1)[0]
        paras = re.split(r'\n\s*\n', text)
        for split, lo, hi, count in specs:
            pstart, pend = int(len(paras)*lo), int(len(paras)*hi)
            text = '\n\n'.join(paras[pstart:pend])
            ids = tok(text, add_special_tokens=False)['input_ids']
            blocks = len(ids)//129
            assert blocks >= count, (bid, split, blocks)
            for ix in rng.choice(blocks, count, replace=False):
                offset = int(ix)*129
                window = ids[offset:offset+129]
                rows.append(dict(split=split, book=bid, paragraph_band=[pstart,pend], token_offset=offset,
                                 input_ids=window, text=tok.decode(window)))
    assert len({(r['book'], r['split'], r['token_offset']) for r in rows}) == len(rows)
    assert len({tuple(r['input_ids']) for r in rows}) == len(rows)
    rng.shuffle(rows)
    dump('dataset.json', rows)
    return {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation', 'test']}

class Shifts(nn.Module):
    def __init__(self, model):
        super().__init__()
        c = model.config
        self.layers, self.heads, self.dim = c.num_hidden_layers, c.num_attention_heads, c.hidden_size
        self.head_dim = c.hidden_size // c.num_attention_heads
        shape = (self.layers, self.heads, self.dim)
        self.w = nn.Parameter(torch.zeros(shape, device='mps'))
        self.b = nn.Parameter(torch.zeros(shape[:2], device='mps'))
        self.pw = nn.Parameter(torch.zeros(shape, device='mps'))
        self.pb = nn.Parameter(torch.zeros(shape[:2], device='mps'))
        self.constant = nn.Parameter(torch.zeros(shape[:2], device='mps'))
        omega = model.model.rotary_emb.inv_freq.detach().clone()
        permutation = torch.randperm(len(omega), generator=torch.Generator().manual_seed(202611092))
        self.register_buffer('omega', omega)
        self.register_buffer('permuted', omega[permutation.to(omega.device)])
        self.mode = 'off'
        self.collect = False
        self.stats = []
        self.handles = [layer.self_attn.q_proj.register_forward_hook(self.hook(l))
                        for l, layer in enumerate(model.model.layers)]
        dump('permutation.json', permutation.tolist())

    def zero(self):
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

    def selected(self, mode):
        return {'constant': [self.constant], 'coordinated': [self.w, self.b],
                'permuted': [self.pw, self.pb]}[mode]

    def hook(self, l):
        def apply(module, inputs, output):
            if self.mode == 'off':
                return output
            x = inputs[0]
            # Bound the single scalar; derivative at zero is exactly one.
            bound = lambda z: 2 * torch.tanh(z / 2)
            angle = 0
            delta = None
            if self.mode in ['joint', 'coordinated']:
                delta = bound(F.linear(x, self.w[l], self.b[l]))
                angle = angle - delta.unsqueeze(-1)*self.omega
            if self.mode in ['joint', 'permuted']:
                delta = bound(F.linear(x, self.pw[l], self.pb[l]))
                angle = angle - delta.unsqueeze(-1)*self.permuted
            if self.mode in ['joint', 'constant']:
                delta = bound(self.constant[l]).view(1, 1, -1).expand(*x.shape[:-1], -1)
                angle = angle - delta.unsqueeze(-1)*self.omega
            if self.collect:
                self.stats.append(torch.stack((delta.detach().square().sum(),
                                               delta.detach().abs().max(),
                                               (delta.detach().abs() > 1.9).float().sum(),
                                               delta.new_tensor(delta.numel()))))
            q = output.view(*x.shape[:-1], self.heads, self.head_dim)
            a, b = q.chunk(2, dim=-1)
            cs, sn = angle.cos(), angle.sin()
            return torch.cat((a*cs-b*sn, a*sn+b*cs), dim=-1).reshape_as(output)
        return apply

    def statistics(self):
        a = torch.stack(self.stats).detach().cpu().numpy()
        self.stats = []
        return dict(rms_tokens=float(np.sqrt(a[:, 0].sum()/a[:, 3].sum())),
                    max_abs_tokens=float(a[:, 1].max()),
                    saturation_fraction=float(a[:, 2].sum()/a[:, 3].sum()))

def batches(rows, size=2):
    for start in range(0, len(rows), size):
        yield rows[start:start+size]

def loss_batch(model, rows):
    ids = torch.tensor([r['input_ids'] for r in rows], device='mps')
    # 128 inputs and 128 supervised targets; no padding, wrappers or cached keys.
    logits = model(input_ids=ids[:, :-1], use_cache=False).logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))

def evaluate(model, adapter, rows, collect=False):
    adapter.collect = collect
    adapter.stats = []
    losses = []
    with torch.no_grad():
        for batch in batches(rows):
            losses.append((float(loss_batch(model, batch)), len(batch)))
    adapter.collect = False
    out = dict(nll=sum(v*n for v,n in losses)/sum(n for _,n in losses),
               sequences=sum(n for _,n in losses), tokens=128*sum(n for _,n in losses))
    if collect:
        out.update(adapter.statistics())
    return out

def screen(model, a, data):
    a.zero()
    a.mode = 'off'
    baseline = evaluate(model, a, data['validation'])
    a.mode = 'coordinated'
    zero = evaluate(model, a, data['validation'][:2])
    a.mode = 'off'
    off = evaluate(model, a, data['validation'][:2])
    assert abs(zero['nll']-off['nll']) < 2e-6, (zero, off)
    dump('zero_check.json', dict(zero=zero, off=off, absolute_error=abs(zero['nll']-off['nll'])))
    emit(stage='baseline', validation=baseline, zero_error=abs(zero['nll']-off['nll']))
    a.mode = 'joint'
    a.zero_grad(set_to_none=True)
    train = data['train'][:16]
    grad_losses = []
    for i, batch in enumerate(batches(train)):
        loss = loss_batch(model, batch)
        (loss/8).backward()
        grad_losses.append(float(loss.detach()))
        emit(stage='gradient', batch=i+1, nll=grad_losses[-1])
    directions = {mode: [-p.grad.detach().clone() for p in a.selected(mode)]
                  for mode in ['constant', 'coordinated', 'permuted']}
    gradient_info = {mode: dict(norm=float(torch.sqrt(sum((d*d).sum() for d in ds))))
                     for mode, ds in directions.items()}
    torch.save({m: [v.cpu() for v in vs] for m,vs in directions.items()}, P/'screen_directions.pt')
    a.zero_grad(set_to_none=True)
    result = dict(baseline_validation=baseline, gradient_train_nll=float(np.mean(grad_losses)),
                  gradient_info=gradient_info, arms={})
    for mode, ds in directions.items():
        a.zero()
        a.mode = mode
        # Measure linear-predictor RMS using a tiny step to avoid tanh saturation.
        with torch.no_grad():
            for p,d in zip(a.selected(mode), ds):
                p.copy_(d * .001)
        calibration = evaluate(model, a, train[:4], collect=True)
        unit_rms = calibration['rms_tokens']/.001
        assert unit_rms > 0
        arm = []
        for target in [.01, .03, .1, .3]:
            with torch.no_grad():
                for p,d in zip(a.selected(mode), ds):
                    p.copy_(d * (target/unit_rms))
            val = evaluate(model, a, data['validation'], collect=True)
            rec = dict(target_rms_tokens=target, **val, delta_nll=val['nll']-baseline['nll'])
            arm.append(rec)
            emit(stage='screen', arm=mode, **rec)
        result['arms'][mode] = arm
        dump('screen.json', result)
    a.zero()
    return result

def train(model, a, data, steps):
    summary = {}
    # Same training examples/order across arms; checkpoint choice uses validation only.
    rng = np.random.default_rng(202611093)
    order = np.concatenate([rng.permutation(len(data['train'])) for _ in range(math.ceil(2*steps/len(data['train'])))])
    for mode in ['constant', 'coordinated', 'permuted']:
        a.zero()
        a.mode = mode
        a.zero_grad(set_to_none=True)
        lr = .01 if mode == 'constant' else .0001
        opt = torch.optim.Adam(a.selected(mode), lr=lr)
        initial = evaluate(model, a, data['validation'], collect=True)
        best = initial['nll']
        best_step = 0
        best_state = [p.detach().cpu().clone() for p in a.selected(mode)]
        history = [dict(step=0, validation=initial)]
        for step in range(1, steps+1):
            batch = [data['train'][int(i)] for i in order[(step-1)*2:step*2]]
            opt.zero_grad(set_to_none=True)
            loss = loss_batch(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(a.selected(mode), 1.0)
            opt.step()
            if step % 8 == 0:
                emit(stage='train', arm=mode, step=step, nll=float(loss.detach()))
            if step % 16 == 0 or step == steps:
                val = evaluate(model, a, data['validation'], collect=True)
                history.append(dict(step=step, validation=val))
                emit(stage='validation', arm=mode, step=step, **val)
                if val['nll'] < best:
                    best, best_step = val['nll'], step
                    best_state = [p.detach().cpu().clone() for p in a.selected(mode)]
                dump(f'train_{mode}.json', dict(lr=lr, steps=steps, best_step=best_step, history=history))
        torch.save(dict(mode=mode, best_step=best_step, values=best_state), P/f'adapter_{mode}.pt')
        summary[mode] = dict(lr=lr, steps=steps, best_step=best_step, best_validation_nll=best, history=history)
        del opt
        a.zero_grad(set_to_none=True)
    # Test is read only after all checkpoint choices have been made.
    a.zero()
    a.mode = 'off'
    test_baseline = evaluate(model, a, data['test'])
    per_sequence_baseline = []
    with torch.no_grad():
        for row in data['test']:
            per_sequence_baseline.append(float(loss_batch(model, [row])))
    for mode, s in summary.items():
        a.zero()
        a.mode = mode
        ck = torch.load(P/f'adapter_{mode}.pt', weights_only=True)
        with torch.no_grad():
            for p,v in zip(a.selected(mode), ck['values']):
                p.copy_(v.to(p.device))
        evaluation = evaluate(model, a, data['test'], collect=True)
        with torch.no_grad():
            losses = [float(loss_batch(model, [row])) for row in data['test']]
        differences = np.array(losses)-np.array(per_sequence_baseline)
        rng = np.random.default_rng(202611094)
        # Paired sequence bootstrap, stratified by book.
        indices = [np.array([i for i,r in enumerate(data['test']) if r['book']==bid]) for bid in [11,1661,1342,2009]]
        bs = np.stack([differences[rng.choice(ix, (5000,len(ix)), replace=True)].mean(1) for ix in indices]).mean(0)
        s.update(test=evaluation, test_delta_nll=float(differences.mean()),
                 test_delta_nll_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),
                 test_perplexity_relative_change=float(np.expm1(differences.mean())),
                 test_sequence_delta_nll=differences.tolist(),
                 test_by_book={str(bid):float(differences[ix].mean()) for bid,ix in zip([11,1661,1342,2009],indices)})
        emit(stage='test', arm=mode, best_step=s['best_step'], **evaluation, delta_nll=s['test_delta_nll'])
    result = dict(baseline_test=test_baseline, arms=summary,
                  limitations='Single seed, small classic-prose sample, 128-token contexts, frozen model. Bootstrap reflects sequence sampling within four fixed books, not corpus/seed uncertainty. Learning rates differ by parameterization; no claim of optimal tuning or pretraining acceleration.')
    dump('training_results.json', result)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--screen-only', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(202611090)
    torch.set_num_threads(4)
    name = 'Qwen/Qwen2.5-1.5B-Instruct'
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    data = dataset(tok)
    dump('protocol.json', dict(model=name, dtype='float32', attention='sdpa', context_inputs=128,
        predictions_per_sequence=128, seed=202611090, training_steps=args.steps,
        splits={k:len(v) for k,v in data.items()}, original_weights_frozen=True,
        trainable_layers='all 28', trainable_query_heads='all 12 per layer',
        displacement='2*tanh((w@input_layernorm_hidden+b)/2); extra query rotation -delta*omega',
        dataset='Four cached public-domain Gutenberg books; disjoint paragraph-index bands train15–55%, validation60–72%, test77–90%; 129-token contiguous windows; raw text, no chat template.',
        controls=['constant per head','dynamic correct frequencies','dynamic one fixed random permutation of frequencies'],
        screen='Gradients on 16 train sequences; steps normalized by predicted RMS displacement on 4 train sequences; validation only for step-size sweep.',
        training='Adam, 64 minibatches of 2 sequences by default; same order all arms; validate every16steps; choose best including step0; test only after all choices.',
        parameter_counts=dict(constant=336, coordinated=336*(1536+1), permuted=336*(1536+1))))
    emit(stage='loading', sequences={k:len(v) for k,v in data.items()})
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float32,
                                                attn_implementation='sdpa').to('mps').eval()
    model.requires_grad_(False)
    a = Shifts(model)
    emit(stage='loaded', memory_gb=torch.mps.current_allocated_memory()/1e9)
    screen(model, a, data)
    if not args.screen_only:
        train(model, a, data, args.steps)
    emit(stage='complete')

if __name__ == '__main__':
    main()
