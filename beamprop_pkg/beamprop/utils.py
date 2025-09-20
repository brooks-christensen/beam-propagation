import numpy as np


def k0_from_lambda(wavelength):
    """Return free-space wavenumber k0 = 2π/λ."""
    return 2.0 * np.pi / wavelength


def fftfreq_2d(nx, ny, dx, dy):
    """Return angular spatial frequencies (kx, ky) on the FFT grid."""
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    return np.meshgrid(kx, ky, indexing="ij")


def fft2c(u):
    """Centered 2D FFT (unitary norm)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm="ortho"))


def ifft2c(U):
    """Centered 2D IFFT (unitary norm)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U), norm="ortho"))


def fft1c(u):
    """Centered 1D FFT (unitary)."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(u), norm="ortho"))


def ifft1c(U):
    """Centered 1D IFFT (unitary)."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(U), norm="ortho"))


def robust_limits(arr, lo=1.0, hi=99.5):
    """Percentile-based vmin/vmax so a hot spot doesn't crush the scale."""
    vmin = np.percentile(arr, lo)
    vmax = np.percentile(arr, hi)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def auto_x_roi(Ixz, x, frac=0.92, pad_pixels=6):
    """
    Given I(x,z) and x array, find the smallest x-range containing `frac`
    of total power (integrated over z). Returns (x_min, x_max) and index slice.
    """
    Ix = Ixz.sum(axis=1)  # integrate over z
    cdf = np.cumsum(Ix)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-20)
    lo_idx = int(np.searchsorted(cdf, (1 - frac) / 2))
    hi_idx = int(np.searchsorted(cdf, 1 - (1 - frac) / 2))
    lo_idx = max(0, lo_idx - pad_pixels)
    hi_idx = min(Ixz.shape[0], hi_idx + pad_pixels)
    x_min, x_max = x[lo_idx], x[hi_idx - 1]
    return (x_min, x_max), slice(lo_idx, hi_idx)


def safe_log10(In, floor_rel=1e-6):
    m = np.max(In)
    eps = m * floor_rel + 1e-30
    return np.log10(np.maximum(In, eps))


def raised_cosine_gate(z, z0, w, ramp):
    """
    Smooth 0→1→0 gate centered at z0 with full width w and cosine ramps of width `ramp`.
    Returns g(z) in [0,1]. When ramp→0, this tends to a rect.
    """
    z1 = z0 - w / 2.0
    z2 = z0 + w / 2.0
    g = np.zeros_like(z, dtype=float)

    if isinstance(z, np.ndarray) or isinstance(z, list):
        # up-ramp on [z1, z1+ramp]
        up = (z >= z1) & (z < z1 + ramp)
        g[up] = 0.5 * (1 - np.cos(np.pi * (z[up] - z1) / ramp))

        # flat top on [z1+ramp, z2-ramp]
        flat = (z >= z1 + ramp) & (z <= z2 - ramp)
        g[flat] = 1.0

        # down-ramp on (z2-ramp, z2]
        down = (z > z2 - ramp) & (z <= z2)
        g[down] = 0.5 * (1 + np.cos(np.pi * (z[down] - (z2 - ramp)) / ramp))

    if isinstance(z, float):
        if z <= z1 or z >= z2:
            return 0.0
        elif z >= z1 + ramp or z <= z2 - ramp:
            return 1.0
        elif z1 < z < z1 + ramp:
            return 0.5 * (1 - np.cos(np.pi * (z - z1) / ramp))
        elif z2 - ramp < z < z2:
            return 0.5 * (1 + np.cos(np.pi * (z - (z2 - ramp)) / ramp))

    return g
