import numpy as np

def supergaussian_mask_2d(X, Y, x_span, y_span, width_frac=0.1, order=4):
    """
    Build a separable super-Gaussian edge mask M(x)M(y) that decays to ~0 near boundaries.
    width_frac: fraction of half-span over which the mask rolls off.
    """
    Lx = x_span/2
    Ly = y_span/2
    wx = width_frac * Lx
    wy = width_frac * Ly

    def edge(v, L, w):
        return np.exp(-((np.maximum(0, np.abs(v)- (L - w)) / (w + 1e-12))**order))

    Mx = edge(X, Lx, wx)
    My = edge(Y, Ly, wy)
    return Mx*My
