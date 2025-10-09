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


def kappa_from_fwhm(fwhm, k):
    alpha = 2.0 * np.arccosh(np.sqrt(2.0)) / fwhm  # ≈ 1.76274 / fwhm
    return (alpha * alpha) / k


# def ln_safe(A):
#     return np.log(np.maximum(A, 1e-20))


def ln_safe(Ix, rel_floor=1e-6):
    m = float(np.nanmax(Ix))
    eps = m * rel_floor + 1e-30
    return np.log(np.maximum(Ix, eps))


def propagate(E0, bpm_obj, n2):
    Eout, _ = bpm_obj.propagate(E0, n2, store_every=0)
    return Eout, None


def kx_grid(Nx, dx):
    return 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))  # [1/m]


def order_kx(k, Lambda, theta_inc_rad, m):
    # thin transmission grating: kx_m = k*sin(theta_inc) + m*K
    K = 2 * np.pi / Lambda
    return k * np.sin(theta_inc_rad) + m * K


def power_in_order(Eout, dx, k, Lambda, theta_inc_rad, m, dkx_bin=0.6):
    """
    Integrate |Ê(kx)|^2 in a small window around the m-th order peak.
    dkx_bin is in units of K (i.e., 0.6 means ±0.6*K/2 half-width).
    """
    Nx = Eout.size
    kx = kx_grid(Nx, dx)
    Ef = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Eout), norm="ortho"))
    S = np.abs(Ef) ** 2

    K = 2 * np.pi / Lambda
    kxm = order_kx(k, Lambda, theta_inc_rad, m)

    w = dkx_bin * K / 2
    sel = (kx >= kxm - w) & (kx <= kxm + w)
    return S[sel].sum(), S.sum()  # (power in order, total power)


def run_one(
    angle_deg,
    build_input,
    propagate,
    dx,
    k,
    Lambda,
    x,
    Lz,
    bpm_obj,
    n2,
    orders=(-1, 0, +1),
):
    """
    angle_deg -> E0 via build_input(angle_deg),
    Eout via propagate(E0),
    returns dict of power fractions per order and a suggested score.
    """
    theta = np.deg2rad(angle_deg)
    E0 = build_input(angle_deg, x, k, Lz / 2, 0.0)
    Eout, _ = propagate(E0, bpm_obj, n2)

    frac = {}
    # Ptot = None
    for m in orders:
        Pm, Pt = power_in_order(Eout, dx, k, Lambda, theta, m)
        frac[m] = Pm / Pt
        # Ptot = Pt
    score_single = frac.get(+1, 0.0) - 0.5 * (frac.get(0, 0.0) + frac.get(-1, 0.0))
    score_bragg_min = -frac.get(+1, 0.0)  # minimize +1 efficiency
    return frac, score_single, score_bragg_min


def sdoOpt(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2, lambda0):
    # coarse grid then refine
    cand = np.linspace(-15, +15, 121)  # degrees; set your expected range
    best = None
    for a in cand:
        frac, score, _ = run_one(
            a, build_input, propagate, dx, k, Lambda, x, Lz, bpm_obj, n2
        )
        if (best is None) or (score > best[0]):
            best = (score, a, frac)

    best_score, best_angle, best_frac = best
    print(
        "Single-order optimum:", f"angle ≈ {best_angle:.2f}°", f"fractions {best_frac}"
    )

    # refine around best_angle
    fine = np.linspace(best_angle - 1.0, best_angle + 1.0, 41)
    for a in fine:
        frac, score, _ = run_one(
            a, build_input, propagate, dx, k, Lambda, x, Lz, bpm_obj, n2
        )
        if score > best_score:
            best_score, best_angle, best_frac = score, a, frac
    print(f"Refined: {best_angle}° {best_frac}")
    theta_theory = np.rad2deg(np.arcsin(lambda0 / (2 * Lambda)))  # ~4.30 deg
    print("External Littrow (theory):", theta_theory, "deg")
    return best_angle


def braggMismatchOptA(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2):
    # minimize +1 mode
    cand = np.linspace(-15, +15, 121)  # degrees; set your expected range
    best = None
    for a in cand:
        frac, *_ = run_one(a, build_input, propagate, dx, k, Lambda, x, Lz, bpm_obj, n2)
        eta1 = frac.get(+1, 0.0)
        if (best is None) or (eta1 < best[0]):  # more negative is better
            best = (eta1, a, frac)
    print(f"Bragg mismatch angle (min η+1): {best[1]}° {best[2]}")
    return best[1]


def braggMismatchOptB(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2):
    # Target efficiency η_{+1} = η* (e.g., 10%): pick the angle whose frac[+1] is closest to η*
    cand = np.linspace(-15, +15, 121)  # degrees; set your expected range
    target = 0.10
    best = None
    for a in cand:
        frac, *_ = run_one(a, build_input, propagate, dx, k, Lambda, x, Lz, bpm_obj, n2)
        err = abs(frac.get(+1, 0.0) - target)
        if (best is None) or (err < best[0]):
            best = (err, a, frac)
    print(f"Angle for η+1≈10%: {best[1]}° {best[2]}")
    return best[1]
