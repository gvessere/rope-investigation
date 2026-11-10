"""Regenerate every figure in ROPE exploration. No model downloads or input data.

Run from the repository root: python code/figures.py
Outputs: figures/*.svg and figures/results.json, with all plotted samples and fits
also saved in figures/examples.npz. These are constructed examples, not model data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER, INK, GREEN, ORANGE = "#f6f5ef", "#172e2b", "#176452", "#bf6536"
plt.rcParams.update({"font.family": ["DejaVu Sans", "Arial", "sans-serif"], "font.size": 11,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "svg.hashsalt": "rope-exploration"})


def frequencies(d=128, base=10000.0):
    """Angular frequencies in radians/token, one per coordinate pair."""
    return base ** (-2 * np.arange(d // 2) / d)


def basis(positions, omega):
    phase = np.asarray(positions)[:, None] * omega[None, :]
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)


def fit_rope(positions, target, d=128, base=10000.0, ridge=1e-3):
    """Minimize ||A c - target||^2 + ridge ||c||^2 via stable SVD.

    Coefficients are [a_0,...,a_J-1,b_0,...,b_J-1]. There is no added
    intercept or learned frequency. Ridge limits large cancelling coefficients.
    """
    design = basis(positions, frequencies(d, base))
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    projection = np.einsum("ij,i->j", u, target, optimize=False)
    coeff = np.einsum("ji,j->i", vh,
                      singular / (singular**2 + ridge) * projection, optimize=False)
    prediction = np.einsum("ij,j->i", design, coeff, optimize=False)
    return prediction, coeff


def targets(x):
    return {
        "A peak eighty tokens away": np.exp(-0.5 * ((x - 80) / 8)**2),
        "A broad neighborhood": 1 / (1 + np.exp(-(x - 64) / 6))
                                 - 1 / (1 + np.exp(-(x - 128) / 6)),
        "Two separate regions": np.exp(-0.5 * ((x - 48) / 9)**2)
                                + .8 * np.exp(-0.5 * ((x - 176) / 12)**2),
        "A one-token spike": (x == 80).astype(float),
    }


def tidy(ax):
    ax.grid(axis="y", alpha=.12)
    ax.spines["bottom"].set_color("#b9c5ba")
    ax.spines["left"].set_color("#b9c5ba")
    ax.tick_params(length=3)


def save(fig, name):
    fig.savefig(ROOT / "figures" / name, metadata={"Date": None},
                bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    arrays, results = {}, {"d": 128, "base": 10000, "ridge": .001,
                           "fit_interval": [0, 255]}

    # An arbitrary finite sampled shape: full DFT recovers even its abrupt edge.
    x = np.arange(128)
    y = np.exp(-.5 * ((x - 36) / 8)**2) + .65 * ((x >= 76) & (x < 104))
    spectrum = np.fft.rfft(y)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.8), sharex=True)
    for ax, count in zip(axes, (4, 16, 65)):
        truncated = spectrum.copy()
        truncated[count:] = 0
        reconstructed = np.fft.irfft(truncated, n=len(x))
        ax.plot(x, y, ".", color=INK, ms=3.5, label="Original samples")
        ax.plot(x, reconstructed, color=ORANGE, lw=1.8, label="DFT reconstruction")
        ax.set_title(f"{count} of 65 stored frequency bins", loc="left", fontsize=12)
        ax.set_ylim(-.2, 1.22)
        tidy(ax)
        arrays[f"dft_{count}"] = reconstructed
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Sample index n")
    fig.tight_layout(h_pad=1.2)
    save(fig, "dft-reconstruction.svg")
    arrays.update(dft_x=x, dft_target=y, dft_spectrum=spectrum)
    results["dft_full_max_error"] = float(np.max(np.abs(arrays["dft_65"] - y)))

    # See how the two grids distribute their angular frequencies.
    omega = frequencies()
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 4.7))
    grids = (2 * np.pi * np.arange(65) / 128, omega, omega)
    titles = ("DFT · equal spacing on a linear frequency axis",
              "RoPE · geometric spacing on a linear frequency axis",
              "The same RoPE frequencies · equal spacing on a log axis")
    for i, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
        ax.plot(grid, np.zeros_like(grid), "|", ms=15, mew=1.3,
                color=ORANGE if i == 0 else GREEN)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks([])
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#b9c5ba")
        ax.set_ylim(-1, 1)
        if i == 2:
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1.2)
        else:
            ax.set_xlim(-.04, np.pi + .04)
    axes[-1].set_xlabel("Angular frequency ω (radians per token)")
    fig.tight_layout(h_pad=1.4)
    save(fig, "frequency-grids.svg")
    arrays["rope_omega"] = omega

    x = np.arange(256)
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 9), sharex=True)
    results["rope_examples"] = {}
    for i, (ax, (name, y)) in enumerate(zip(axes, targets(x).items())):
        prediction, coeff = fit_rope(x, y)
        rmse = float(np.sqrt(np.mean((prediction - y)**2)))
        ax.plot(x, y, color=INK, lw=1.4, ls="--", label="Desired score")
        ax.plot(x, prediction, color=GREEN, lw=1.9, label="RoPE-basis fit")
        ax.set_title(name, loc="left", fontsize=12)
        ax.text(.99, .86, f"RMSE {rmse:.4f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=GREEN)
        ax.set_ylim(-.18, 1.23)
        tidy(ax)
        arrays.update({f"rope_target_{i}": y, f"rope_fit_{i}": prediction,
                       f"rope_coeff_{i}": coeff})
        results["rope_examples"][name] = {"rmse": rmse,
            "max_error": float(np.max(np.abs(prediction - y))),
            "coefficient_norm": float(np.linalg.norm(coeff))}
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, .9))
    axes[-1].set_xlabel("Relative displacement Δ (tokens)")
    arrays["rope_x"] = x
    fig.tight_layout(h_pad=1.2)
    save(fig, "rope-shapes.svg")
    shift_examples = {"omega": omega.tolist(), "examples": [
        {"name": name, "coefficients": arrays[f"rope_coeff_{i}"].tolist()}
        for i, name in enumerate(results["rope_examples"])]}
    (ROOT / "figures" / "shift-examples.json").write_text(
        json.dumps(shift_examples, separators=(",", ":")) + "\n")
    np.savez(ROOT / "figures" / "examples.npz", **arrays)
    (ROOT / "figures" / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
