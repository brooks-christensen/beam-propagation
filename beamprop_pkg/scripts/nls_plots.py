import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from beamprop_pkg.beamprop.utils import ln_safe  # kappa_from_fwhm,
from beamprop_pkg.beamprop.fields import (
    gaussian_beam_1d,
    soliton_profile,
)
from beamprop_pkg.beamprop.grids import Grid2D
from beamprop_pkg.beamprop.propagators import BPM2D
from beamprop_pkg.beamprop.absorbers import absorbing_field_1d


# define output path to store figures
base_path = Path.cwd() / "beamprop_pkg" / "out"
out_png = base_path / "nls.png"
log_flag = False
xConf = "B"


# define physical constants
lambda0 = 1e-6
k0 = 2 * np.pi / lambda0
if xConf == "A":
    Nx = 512 + 1
    dx = 2e-6
elif xConf == "B":
    Nx = 1024 + 1
    dx = 1e-6
Nz = 300 + 1
dz = 10e-6
Lx = dx * (Nx - 1)
Lz = dz * (Nz - 1)
wAbs: int = 40  # number of points to include at the edges of absorber
gamma = 1.0  # strength of absorber at the boundary [0.0, 1.0]
n2 = 1e-3  # strength of nonlinearity
w_gauss = 30e-6  # width of Gaussian input beam
n_glass = 1.5
k = n_glass * k0
Efactor = 1.105  #  np.sqrt(2.5 / 8.5) # factor matches node spacing from homework, physical behavior is different???
# n2 /= Efactor * Efactor


# create grids
grid_obj = Grid2D(Lx, Lz, Nx, Nz, center=True)
X, Z, dx_, dz_ = grid_obj.mesh()
x, z, dx__, dz__ = grid_obj.arrays()
assert np.isclose(dx, dx_)
assert np.isclose(dz, dz_)
assert np.isclose(dx, dx__)
assert np.isclose(dz, dz__)


# create 2D refractive index inhomogeneity mesh grid
# same dimensions as X, Z
nIN = np.zeros(X.shape)
nRef = np.ones(X.shape)  # * n_glass


# create input fields
kappa = 1 / 50 * 1e6
E0_gauss = gaussian_beam_1d(
    x,
    w_gauss,
    0.0,  # Lx / 2,
)
E0_soliton = soliton_profile(x, k, k0, n2, kappa)
# Pg = np.trapezoid(np.abs(E0_gauss) ** 2, x)
# Ps = np.trapezoid(np.abs(E0_soliton) ** 2, x)
# E0_gauss *= np.sqrt(Ps / Pg)
# E0_gauss *= np.sqrt(2.5 / 5.7)
E0_gauss *= Efactor


# perform unstable propagation
abs_mask = absorbing_field_1d(Nx, wAbs, gamma)
bpm_obj = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask, nIN, nRef)
Eout_gauss, snapshots = bpm_obj.propagate(E0_gauss, n2, store_every=1)


# perform stable propagation
Eout_sol, snapshots_stable = bpm_obj.propagate(E0_soliton, n2, store_every=1)

Ig = snapshots if snapshots.shape == nIN.shape else snapshots.T
Is = snapshots_stable if snapshots_stable.shape == nIN.shape else snapshots_stable.T

if log_flag:
    Ig = ln_safe(Ig)
    Is = ln_safe(Is)

# common extent: horizontal = z [µm], vertical = x [µm]
extent = (
    float(Z.min() * 1e6),
    float(Z.max() * 1e6),  # z-axis (x of plot)
    float(X.min() * 1e6),
    float(X.max() * 1e6),  # x-axis (y of plot)
)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)

# --- Left: |E(x,z)|² (unstable propagation) ---
im1 = axs[0].imshow(
    Ig,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
if log_flag:
    title0 = "Gaussian (Unstable) ln(|E(x,z)|²)"
else:
    title0 = "Gaussian (Unstable) |E(x,z)|²"
axs[0].set_title(title0)
axs[0].set_xlabel("z [µm]")
axs[0].set_ylabel("x [µm]")
fig.colorbar(im1, ax=axs[0], label="intensity")

# --- Right: |E(x,z)|² (stable propagation) ---
im1 = axs[1].imshow(
    Is,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
if log_flag:
    title1 = "Soliton (Stable) ln(|E(x,z)|²)"
else:
    title1 = "Soliton (Stable) |E(x,z)|²"
axs[1].set_title(title1)
axs[1].set_xlabel("z [µm]")
fig.colorbar(im1, ax=axs[1], label="intensity")

fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
