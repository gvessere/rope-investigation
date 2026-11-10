"""Small, explicit RoPE tools. Pairs occupy the final axis (..., pairs, 2)."""
import numpy as np


def frequencies(dimension=128, base=10000.0):
    if dimension <= 0 or dimension % 2 or base <= 1:
        raise ValueError("Use a positive even dimension and base > 1")
    return base ** (-2 * np.arange(dimension // 2) / dimension)


def rotate(pairs, position, omega):
    pairs = np.asarray(pairs, dtype=float)
    angle = np.asarray(position)[..., None] * omega
    c, s = np.cos(angle), np.sin(angle)
    x, y = pairs[..., 0], pairs[..., 1]
    return np.stack((c*x-s*y, s*x+c*y), axis=-1)


def coefficients(q, k):
    """For delta=key_position-query_position: score=A cos + B sin."""
    q, k = np.asarray(q), np.asarray(k)
    a = (q*k).sum(-1)
    b = q[..., 1]*k[..., 0] - q[..., 0]*k[..., 1]
    return a, b


def score(q, k, delta, omega):
    a, b = coefficients(q, k)
    angle = np.asarray(delta)[..., None] * omega
    return (a*np.cos(angle)+b*np.sin(angle)).sum(-1)/np.sqrt(q.size)


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = np.exp(z-z.max(axis=-1, keepdims=True))
    return z/z.sum(axis=-1, keepdims=True)


def quantize_phase(pairs, bits, threshold=np.inf):
    """Origin-based phase rounding; zero bits maps selected phases to zero.

    Radius is retained in full precision. This is NOT a packed KV codec.
    """
    if not isinstance(bits, int) or bits < 0:
        raise ValueError("bits must be a nonnegative integer")
    pairs = np.asarray(pairs, dtype=float)
    radius = np.linalg.norm(pairs, axis=-1)
    phase = np.arctan2(pairs[..., 1], pairs[..., 0])
    levels = 2**bits
    step = 2*np.pi/levels
    rounded = np.zeros_like(phase) if bits == 0 else np.round(phase/step)*step
    out = radius[..., None]*np.stack((np.cos(rounded), np.sin(rounded)), -1)
    return np.where((radius < threshold)[..., None], out, pairs)


def design(distances, omega):
    a = np.asarray(distances)[:, None]*omega[None]
    return np.concatenate((np.cos(a), np.sin(a)), axis=1)


def fit_curve(distances, values, omega, rcond=1e-10):
    matrix = design(distances, omega)
    coef, _, rank, singular = np.linalg.lstsq(matrix, values, rcond=rcond)
    return coef, int(rank), singular


def fit_decay(bin_mass, lower, upper):
    """OLS on log mean weight per key; input mass has final axis=bin."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    width = upper-lower+1
    centers = np.array([np.exp(np.log(np.arange(a,b+1)).mean())
                        for a,b in zip(lower,upper)])
    y = np.log(np.asarray(bin_mass)/width)
    x = np.log(centers)
    xc = x-x.mean()
    slope = ((y-y.mean(-1, keepdims=True))*xc).sum(-1)/(xc*xc).sum()
    fitted = y.mean(-1, keepdims=True)+slope[..., None]*xc
    total = ((y-y.mean(-1, keepdims=True))**2).sum(-1)
    r2 = 1-((y-fitted)**2).sum(-1)/np.maximum(total, 1e-30)
    return -slope, r2
