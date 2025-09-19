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


def safe_log10(I, floor_rel=1e-6):
    m = np.max(I)
    eps = m * floor_rel + 1e-30
    return np.log10(np.maximum(I, eps))
