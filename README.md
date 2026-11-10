# ROPE exploration

A personal exploration of RoPE through its connection to Fourier transforms.

[Read the blog](https://gvessere.github.io/rope-investigation/).

## Edit and preview

Edit `index.html` for prose and equations; `style.css` controls presentation.
Equations use native MathML, with their LaTeX source in each annotation. The site
has no JavaScript runtime, remote fonts, or external rendering dependencies.

Run `python3 -m http.server 8771` from this directory to preview locally.

## Regenerate every figure

Requires Python 3.9 or newer. From this repository's root:

```sh
python -m pip install -r requirements.txt
python code/figures.py
python -m unittest discover -s code -p 'test_*.py'
```

The pinned packages are NumPy 2.0.2 and Matplotlib 3.9.4.
`code/figures.py` contains every target function, fitting operation, and plot.
It generates `figures/dft-reconstruction.svg`, `figures/frequency-grids.svg`,
`figures/rope-shapes.svg`, `figures/results.json`, and `figures/examples.npz`.
No external data, model weights, or randomness are used to generate figures.
SVG dates are suppressed and IDs are stable. The array archive contains the
samples, predictions, frequencies, and fitted coefficients behind the plots.

The RoPE examples fit 256 scores using 64 geometric frequencies (128 real
coefficients), base 10000, no extra intercept, and ridge penalty 0.001. They
illustrate representational possibilities, not trained-model findings or
long-context performance. Displacement is query position minus key position;
positive values refer to earlier keys.

Tests check arbitrary real DFT reconstruction, the sine/cosine expansion against
explicit query/key rotations, the geometric ratio, and the fit's optimality
condition. GitHub Actions runs these checks and regenerates the figures before
deploying the static site.

## References

- [RoFormer (Su et al.)](https://arxiv.org/abs/2104.09864)
- [Fourier Position Embedding (Hua et al., ICML 2025)](https://proceedings.mlr.press/v267/hua25b.html), particularly Section 2.2
- [NumPy DFT conventions and real-transform symmetry](https://numpy.org/doc/stable/reference/routines.fft.html)

Earlier articles and experiments remain available in Git history.
