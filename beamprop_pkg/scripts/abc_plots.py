import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from beamprop_pkg.beamprop.utils import ln_safe  # kappa_from_fwhm,
from beamprop_pkg.beamprop.fields import (
    #     grin_lens_dn_xz_cosine,
    gaussian_beam_1d,
    #     nref_grid_from_rod,
    # soliton_profile,
)
from beamprop_pkg.beamprop.grids import Grid2D
from beamprop_pkg.beamprop.propagators import BPM2D
from beamprop_pkg.beamprop.absorbers import absorbing_field_1d


# define output path to store figures
base_path = Path.cwd() / "beamprop_pkg" / "out"
out_png = base_path / "abc.png"
log_flag = False


# define physical constants
lambda0 = 1e-6
k0 = 2 * np.pi / lambda0
Nx = 256 + 1
Nz = 500 + 1
dx = 0.5e-6
dz = 2e-6
Lx = dx * (Nx - 1)
Lz = dz * (Nz - 1)
wAbs1: int = 5  # number of points to include at the edges of absorber
wAbs2: int = 40
gamma = 0.0  # strength of absorber at the boundary [0.0, 1.0]
n2 = 0.0  # strength of nonlinearity
k = k0


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
nRef = np.ones(X.shape)  #  * n_glass


# create input field
w_gauss = 30e-6
tilt_deg = -10
E0_gt = gaussian_beam_1d(
    x,
    w_gauss,
    0.0,
    k,
    tilt_deg,
)


# create absorbing boundary masks
abs_mask1 = absorbing_field_1d(Nx, wAbs1, gamma)
abs_mask2 = absorbing_field_1d(Nx, wAbs2, gamma)


# create simulation objects
bpm_obj1 = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask1, nIN, nRef)
bpm_obj2 = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask2, nIN, nRef)


# perform propagation
Eout1, snapshots1 = bpm_obj1.propagate(E0_gt, n2, store_every=1)
Eout2, snapshots2 = bpm_obj2.propagate(E0_gt, n2, store_every=1)


# transpose if necessary
Ig = snapshots1 if snapshots1.shape == nIN.shape else snapshots1.T
Is = snapshots2 if snapshots2.shape == nIN.shape else snapshots2.T


# take log of propagation intensity, if indicated
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
    title0 = "Poor Absorber\nln(|E(x,z)|²)"
else:
    title0 = "Poor Absorber\n|E(x,z)|²"
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
    title1 = "Good Absorber\nln(|E(x,z)|²)"
else:
    title1 = "Good Absorber\n|E(x,z)|²"
axs[1].set_title(title1)
axs[1].set_xlabel("z [µm]")
fig.colorbar(im1, ax=axs[1], label="intensity")

fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
