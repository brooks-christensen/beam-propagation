import numpy as np


def gaussian_beam_2d(X, Z, w0, x0=0.0, z0=0.0, kx=0.0, kz=0.0):
    """
    2D Gaussian envelope with optional tilt (kx, ky) [rad/m].
    E(x,y) = exp(-((x-x0)^2+(y-y0)^2)/w0^2) * exp(i*(kx*x + ky*y))
    """
    return np.exp(-(((X - x0) ** 2 + (Z - z0) ** 2) / (w0**2))) * np.exp(
        1j * (kx * X + kz * Z)
    )


def gaussian_beam_1d(x, w0, x0=0.0, k=0.0, angle_deg=0.0) -> np.ndarray:
    """1D Gaussian with optional tilt."""
    return np.exp(-((x - x0) ** 2) / (w0**2)) * np.exp(
        -1j * k * np.sin(np.deg2rad(angle_deg)) * x
    )


def rect(u):
    """1 inside |u| <= 1/2, else 0 (vectorized)."""
    return (np.abs(u) <= 0.5).astype(float)


def grin_prism_dn_xz(X, Z, Lx, amp=0.07, z_center=1050e-6, z_width=1900e-6):
    """
    Δn(x,z) = amp * (x/Lx) * rect((z - z_center)/z_width)
    X, Z: meshgrids in meters (shape [nx, nz])
    Lx: full x-extent (e.g., 200e-6 for x ∈ [-100e-6, +100e-6])
    """
    ramp = amp * (X / Lx)  # linear in x, uniform in y
    window = rect((Z - z_center) / z_width)  # finite-z slab
    return ramp * window


def grin_prism_n_xz(X, Z, n0=1.5, **kw):
    """Absolute index map n(x,z) = n0 + Δn(x,z)."""
    return n0 + grin_prism_dn_xz(X, Z, **kw)


def grin_lens_dn_xz_cosine(X, Z, z_center, z_width, n_glass, dn0, x_center):
    # compensates for dispersive effects
    g = np.pi / (2.0 * z_width)
    a = g * np.sqrt(n_glass / dn0)
    rod = (np.abs((Z - z_center) / z_width) <= 0.5).astype(float)
    dn = dn0 * np.cos(a * (X - x_center)) * rod  # Δn relative to n_ref
    return dn


def nref_grid_from_rod(X, Z, z_center, z_width, n_glass, n_air=1.0):
    """
    Returns an n_ref grid with shape == X == Z:
      n_ref = n_air outside the rod,
      n_ref = n_glass inside the rod (z_center ± z_width/2).
    """
    rod_mask = rect((Z - z_center) / z_width)  # 1 inside, 0 outside
    return n_air + (n_glass - n_air) * rod_mask  # broadcast over X


def soliton_profile(x, k, k0, n2, kappa):
    """
    Bright soliton of the physical 1D Kerr-NLS:
    i A_z + (1/(2k)) A_xx + k0*n2*|A|^2 A = 0
    """
    return np.sqrt(kappa / (n2 * k0)) / np.cosh(x * np.sqrt(kappa * k))


def sech_1d(x, a=1.0, A=1.0):
    """Amplitude-scaled sech: A * sech(a x)."""
    return A / np.cosh(a * x)


# def rect_2d(X, Z, x0, x_width, z0, z_width):
#     """Rectangular aperture of given width centered at 0."""
#     return ((np.abs(X - x0) <= x_width / 2) and (np.abs(Z - z0) <= z_width / 2)).astype(
#         float
#     )


# def thick_hologram(X, Z, x0, x_width, z0, z_width, wavelength):
#     rect_mask = rect_2d(X, Z, x0, x_width, z0, z_width)
#     sin_field = np.sin(2 * np.pi / wavelength * X)
#     return rect_mask * sin_field


def rect_2d(X, Z, x0, x_width, z0, z_width):
    """
    Rectangle mask equal to 1 inside:
        |X - x0| <= x_width/2  AND  |Z - z0| <= z_width/2
    X, Z are same-shaped meshgrids (indexing='ij' is fine).
    """
    in_x = np.abs(X - x0) <= (x_width / 2.0)
    in_z = np.abs(Z - z0) <= (z_width / 2.0)
    return (in_x & in_z).astype(float)


def thick_hologram(X, Z, x0, x_width, z0, z_width, period, phase=0.0, amplitude=0.07):
    """
    Cosine index modulation confined to a rectangular slab.
      period: grating period Λ along +x (in meters)
      phase : optional phase offset of the grating
    Returns Δn(x,z) = mask(x,z) * cos(2π x / Λ + phase)
    """
    mask = rect_2d(X, Z, x0, x_width, z0, z_width)
    grat = np.cos(2.0 * np.pi * X / period + phase)
    return amplitude * mask * grat


def build_input(angle_deg, x, k, z_g, x_g):
    theta = np.deg2rad(angle_deg)
    x0 = -(z_g) * np.tan(theta) + x_g  # make the beam hit (x_g,z_g)
    return gaussian_beam_1d(x, w0=30e-6, x0=x0, k=k, angle_deg=angle_deg)
