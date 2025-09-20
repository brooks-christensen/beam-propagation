import numpy as np


def supergaussian_mask_2d(X, Y, x_span, y_span, width_frac=0.1, order=4):
    """
    Build a separable super-Gaussian edge mask M(x)M(y) that decays to ~0 near boundaries.
    width_frac: fraction of half-span over which the mask rolls off.
    """
    Lx = x_span / 2
    Ly = y_span / 2
    wx = width_frac * Lx
    wy = width_frac * Ly

    def edge(v, L, w):
        return np.exp(-((np.maximum(0, np.abs(v) - (L - w)) / (w + 1e-12)) ** order))

    Mx = edge(X, Lx, wx)
    My = edge(Y, Ly, wy)
    return Mx * My


def absorbing_field_1d(N: int, w: int, gamma: float) -> np.ndarray:
    c1 = (1 - gamma) / 2
    c2 = (1 + gamma) / 2

    def abf(i):
        return c1 * np.cos(np.pi * i / w) + c2

    rl = np.array(range(w, -1, -1))
    rr = np.array(range(N - 1, N - w - 2, -1))

    al = abf(1 - rl)
    am = [1] * (N - 2 * w - 2)
    ar = abf(N - 1 - rr)

    return np.concatenate([al, am, ar])
