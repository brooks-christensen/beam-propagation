import numpy as np


def gaussian_beam_2d(X, Z, w0, x0=0.0, z0=0.0, kx=0.0, kz=0.0):
    """
    2D Gaussian envelope with optional tilt (kx, ky) [rad/m].
    E(x,y) = exp(-((x-x0)^2+(y-y0)^2)/w0^2) * exp(i*(kx*x + ky*y))
    """
    return np.exp(-(((X - x0) ** 2 + (Z - z0) ** 2) / (w0**2))) * np.exp(
        1j * (kx * X + kz * Z)
    )


def gaussian_beam_1d(x, w0, x0=0.0, kx=0.0) -> np.ndarray:
    """1D Gaussian with optional tilt."""
    return np.exp(-((x - x0) ** 2) / (w0**2)) * np.exp(1j * kx * x)


def rect_1d(x, width):
    """Rectangular aperture of given width centered at 0."""
    return (np.abs(x) <= width / 2).astype(float)


def sech_1d(x, a=1.0):
    """sech profile for fundamental bright soliton in normalized NLS: eta*sech(eta x)."""
    return 1.0 / np.cosh(a * x)


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


# def grin_lens_dn_xz(
#     X, Z, Lx, amp=0.07, z_center=1300e-6, z_width=2400e-6, x_center=0.0
# ):
#     """
#     Δn(x,z) = amp * cos(2*pi*x/Lx) * rect((z - z_center)/z_width)
#     X, Z: meshgrids in meters (shape [nx, nz])
#     Lx: full x-extent (e.g., 200e-6 for x ∈ [-100e-6, +100e-6])
#     """
#     cos_ramp = amp * np.cos(2 * np.pi * (X - x_center) / Lx)
#     window = rect((Z - z_center) / z_width)
#     return cos_ramp * window


def grin_lens_dn_xz_cosine(X, Z, z_center, z_width, n_glass, dn0, x_center):
    # compensates for dispersive effects
    g = np.pi / (2.0 * z_width)
    a = g * np.sqrt(n_glass / dn0)
    rod = (np.abs((Z - z_center) / z_width) <= 0.5).astype(float)
    dn = dn0 * np.cos(a * (X - x_center)) * rod  # Δn relative to n_ref
    return dn


# def grin_lens_dn_xz(
#     X, Z, Lx, amp=0.07, z_center=1300e-6, z_width=2400e-6, x_center=0.0
# ):
#     # choose 'a' so that the rod length z_width is a quarter-pitch
#     n0 = 1.0
#     g = np.pi / (2.0 * z_width)  # quarter-pitch focus at exit
#     a = g * np.sqrt(n0 / amp)  # link cosine to parabolic GRIN
#     cos_ramp = amp * np.cos(a * (X - x_center))  # <-- use 'a', NOT 2π/Lx
#     window = rect((Z - z_center) / z_width)
#     return cos_ramp * window


def nref_grid_from_rod(X, Z, z_center, z_width, n_glass, n_air=1.0):
    """
    Returns an n_ref grid with shape == X == Z:
      n_ref = n_air outside the rod,
      n_ref = n_glass inside the rod (z_center ± z_width/2).
    """
    rod_mask = rect((Z - z_center) / z_width)  # 1 inside, 0 outside
    n_ref = n_air + (n_glass - n_air) * rod_mask  # broadcast over X
    return n_ref
