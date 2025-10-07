import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from beamprop_pkg.beamprop.fields import (
    grin_lens_dn_xz_cosine,
    gaussian_beam_1d,
    nref_grid_from_rod,
)
from beamprop_pkg.beamprop.grids import Grid2D
from beamprop_pkg.beamprop.propagators import BPM2D
from beamprop_pkg.beamprop.absorbers import absorbing_field_1d


# define output path to store figures
base_path = Path.cwd() / "beamprop_pkg" / "out"
out_png = base_path / "grin_lens.png"


# define physical constants
lambda0 = 1e-6
Nx = 512 + 1
Nz = 300 + 1
dx = 1e-6
dz = 10e-6
Lx = dx * (Nx - 1)
Lz = dz * (Nz - 1)
wAbs: int = 50  # number of points to include at the edges of absorber
gamma = 1.0  # strength of absorber at the boundary [0.0, 1.0]
n2 = 0.0
w0 = 300e-6  # width of Gaussian input beam
z0 = 1300e-6
zW = 2400e-6


# create grids
grid_obj = Grid2D(Lx, Lz, Nx, Nz, center=False)
X, Z, dx_, dz_ = grid_obj.mesh()
x, z, dx__, dz__ = grid_obj.arrays()
assert np.isclose(dx, dx_)
assert np.isclose(dz, dz_)
assert np.isclose(dx, dx__)
assert np.isclose(dz, dz__)


# create 2D refractive index inhomogeneity mesh grid
# same dimensions as X, Z
nIN = grin_lens_dn_xz_cosine(
    X, Z, z_center=z0, z_width=zW, n_glass=1.5, dn0=0.07, x_center=Lx / 2
)
nRef = nref_grid_from_rod(X, Z, z0, zW, n_glass=1.5)


# perform propagation
E0 = gaussian_beam_1d(
    x,
    w0,
    Lx / 2,
)
abs_mask = absorbing_field_1d(Nx, wAbs, gamma)
bpm_obj = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask, nIN, nRef)
Eout, snapshots = bpm_obj.propagate(E0, n2, store_every=1)

# If snapshots is (Nz, Nx), transpose so shape matches nIN (Nx, Nz)
snap = snapshots if snapshots.shape == nIN.shape else snapshots.T

# common extent: horizontal = z [µm], vertical = x [µm]
extent = (
    float(Z.min() * 1e6),
    float(Z.max() * 1e6),  # z-axis (x of plot)
    float(X.min() * 1e6),
    float(X.max() * 1e6),  # x-axis (y of plot)
)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)

# --- Left: Δn(x,z) (grism) ---
im0 = axs[0].imshow(
    nIN,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
axs[0].set_title("Δn(x,z)")
axs[0].set_xlabel("z [µm]")
axs[0].set_ylabel("x [µm]")
fig.colorbar(im0, ax=axs[0], label="Δn")

# --- Right: |E(x,z)|² (propagation) ---
im1 = axs[1].imshow(
    np.log(snap),
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
axs[1].set_title("ln(|E(x,z)|²)")
axs[1].set_xlabel("z [µm]")
fig.colorbar(im1, ax=axs[1], label="intensity")

# === Prism/grism boundaries (z0 center [m], zW width [m]) ===
z_start_um = (z0 - zW / 2.0) * 1e6
z_end_um = (z0 + zW / 2.0) * 1e6

# dashed guide lines
axs[1].axvline(z_start_um, ls="--", lw=1.8, color="w", alpha=0.9)
axs[1].axvline(z_end_um, ls="--", lw=1.8, color="w", alpha=0.9)

# optional: lightly shade the active slab
axs[1].axvspan(z_start_um, z_end_um, facecolor="w", alpha=0.08)

fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
