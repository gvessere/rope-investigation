# RoPE investigation

[Read the research site](https://gvessere.github.io/rope-investigation/) · [About Gery Vessere](https://gvessere.github.io/)

A five-part research blog on positional encodings, RoPE’s Fourier-like expressivity, empirical phase/radius structure and compression, and attention decay.

The primary source is the Codex chat **ROPE investigation**, thread `01a15f24-929d-77a3-9610-0068124672c0`, and its local research archive. The ChatGPT **ROPE Blog Outline**, conversation `6af25ae1-7528-83ea-8486-c00651f73142`, supplied the requested structure. Earlier assistant statements were treated as claims to verify, not evidence by themselves.

Original figures are reused byte for byte. `data/provenance.json` maps them and the measurements to their original relative paths and SHA-256 hashes. The Gaussian DFT/RoPE comparison is a **new deterministic illustration**, clearly labeled separately from model measurements.

## Reproduce

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/audit.py
python code/figures.py
npm ci
npm run build
npm run preview
```

The tests and audit need NumPy; the theory figure also needs Matplotlib. The published pages use native MathML generated at build time by KaTeX. No external scripts or fonts are required by the deployed site. The package lock pins the website build dependency.

`content/` contains editable article fragments. `scripts/build.mjs` renders equations, inserts code directly from the tested implementation, and builds tables from saved JSON. `dist/` is the static deployment output. The Pages workflow tests, regenerates the theory illustration, builds, checks local links, and deploys it.

Every figure displayed in the blog can be regenerated with `python code/figures.py`. `data/figure_recipes.json` maps each original image to its code and distributed numeric inputs. Replots go to `generated_figures/` without modifying recovered PNGs. They reuse the archived measurements; they do not rerun the model. Visual styling and rasterization can differ with fonts/library versions.

## What was verified

- Rotation identity, displacement convention, and sine sign.
- Phase-rounding radii, selection, and circular seam.
- DFT reconstruction and restricted-basis limitations.
- Softmax shift invariance and output differential.
- Known-curve decay fits with bin exposure correction.
- Original-file checksums, per-token loss/prediction aggregates, dense decay fits, and engineered profile reconstruction.

This is numerical replay of saved experiments, not a new full-model replication. `code/archive/` preserves five original scripts with their original workspace dependencies; these are archival references rather than standalone entry points. Model weights and source book passages are not distributed. Exact model/tokenizer revisions and a complete immutable environment were not recovered. The source summaries and processed measurements document the experiment scope.

## Findings that qualified the outline

- π/2 is zero contribution at a reference point, not global positional neutrality. π suppresses a finite score contribution, not a hard mask.
- Negative pair contributions are common in the saved winner/runner-up margin decomposition. This does not count ordinary query/key suppression mechanisms.
- Actual small slow-key phases tolerate three-bit rounding much better than zero phase bits. Nearly free simplification was a narrower result about learned slow key biases.
- The tested full three-bit hybrid codec did not beat the centered 6/4 codec at matched storage.
- 101/333 eligible dense heads approximately matched 1/m under the archived criterion; geometry alone does not prove this pattern or fix its exponent.

Citations and experiment limitations appear beside the corresponding claims in each article. The existing personal site lives separately in `gvessere/gvessere.github.io`.
