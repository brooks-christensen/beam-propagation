import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from beamprop_pkg.beamprop.fields import grin_prism_dn_xz, gaussian_beam_1d
from beamprop_pkg.beamprop.grids import Grid2D
from beamprop_pkg.beamprop.propagators import BPM2D
from beamprop_pkg.beamprop.absorbers import absorbing_field_1d


# define output path to store figures
base_path = Path.cwd() / "beamprop_pkg" / "out"
out1 = base_path / "grism_geometry.png"
out2 = base_path / "grism_propagation.png"


# define physical constants
lambda0 = 1e-6
Nx = 512 + 1
Nz = 300 + 1
dx = 2e-6
dz = 10e-6
Lx = dx * (Nx - 1)
Lz = dz * (Nz - 1)
wAbs: int = 50  # number of points to include at the edges of absorber
gamma = 1.0  # strength of absorber [0.0, 1.0]
n2 = 0.0
w0 = 60e-6  # width of Gaussian input beam
z0 = 1050e-6
zW = 1900e-6


# create grids
grid_obj = Grid2D(Lx, Lz, Nx, Nz)
X, Z, dx_, dz_ = grid_obj.mesh()
x, z, dx__, dz__ = grid_obj.arrays()
assert np.isclose(dx, dx_)
assert np.isclose(dz, dz_)
assert np.isclose(dx, dx__)
assert np.isclose(dz, dz__)


# create 2D refractive index inhomogeneity mesh grid
# same dimensions as X, Z
# nIN = np.zeros(X.shape)
nIN = grin_prism_dn_xz(X, Z, Lx, z_center=z0, z_width=zW)


# generate and store figure of 2D grism refractive index inhomogeneity
extent = (
    float(Z.min() * 10**6),
    float(Z.max() * 10**6),
    float(X.min() * 10**6),
    float(X.max() * 10**6),
)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(
    nIN,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
cbar = fig.colorbar(im, ax=ax, label="intensity")
ax.set_xlabel("z [um]")
ax.set_ylabel("x [um]")
ax.set_title("|E(x)|²")
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)


# perform propagation
E0 = gaussian_beam_1d(
    x,
    w0,
    0.0,
)
abs_mask = absorbing_field_1d(Nx, wAbs, gamma)
bpm_obj = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask, nIN)
Eout, snapshots = bpm_obj.propagate(E0, n2, store_every=1)

# generate and store 2D figure of beam propagation intensity
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(
    snapshots,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
cbar = fig.colorbar(im, ax=ax, label="intensity")
ax.set_xlabel("z [um]")
ax.set_ylabel("x [um]")
ax.set_title("|E(x)|²")
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
