import numpy as np
from dataclasses import dataclass
from .utils import k0_from_lambda, fft1c

def bragg_angle(n, wavelength, period, order=1, transmission=True):
    """
    First-order Bragg condition (transmission): 2 n Λ sin θ_B = m λ  -> θ_B = arcsin(m λ / (2 n Λ)).
    For reflection gratings, use 2 n Λ cos θ_B = m λ (not used here).
    Returns θ_B in radians.
    """
    if transmission:
        arg = order * wavelength / (2.0 * n * period)
        arg = np.clip(arg, -1.0, 1.0)
        return np.arcsin(arg)
    else:
        arg = order * wavelength / (2.0 * n * period)
        arg = np.clip(arg, -1.0, 1.0)
        return np.arccos(arg)

def kogelnik_efficiency(n, wavelength, period, slab_thickness, dn, theta_in, order=1):
    """
    Very simple Kogelnik-style prediction for a lossless, unslanted, phase transmission grating:
      κ = π Δn / (λ cos θ_B),   η(Δ=0) = sin^2(κ L)
    Off-Bragg detuning reduces efficiency; approximate detuning by angular deviation from Bragg.
    """
    theta_B = bragg_angle(n, wavelength, period, order=order, transmission=True)
    kappa = np.pi * dn / (wavelength * np.cos(theta_B))
    detune = (2.0*np.pi*n/wavelength) * (np.sin(theta_in) - np.sin(theta_B)) * period
    Om = np.sqrt(kappa**2 + (0.5*detune)**2)
    eta = (np.sin(Om*slab_thickness)**2) * (kappa**2 / (Om**2 + 1e-16))
    return eta

def sweep_coupling_angles(n, wavelength, period, slab_thickness, dn, order=1, angle_range_deg=5.0, samples=2001):
    """
    Sweep angles around θ_B and return angles (deg) that minimize and maximize coupling (0th vs 1st order).
    Returns: theta_deg_array, eta_array, theta_minmax = (theta_min_eta_deg, theta_max_eta_deg)
    """
    theta_B = bragg_angle(n, wavelength, period, order=order, transmission=True)
    dth = np.deg2rad(angle_range_deg)
    thetas = np.linspace(theta_B - dth, theta_B + dth, samples)
    etas = kogelnik_efficiency(n, wavelength, period, slab_thickness, dn, thetas, order=order)
    idx_max = int(np.argmax(etas))
    idx_min = int(np.argmin(etas))
    th_min = np.rad2deg(thetas[idx_min])
    th_max = np.rad2deg(thetas[idx_max])
    return np.rad2deg(thetas), etas, (th_min, th_max)
