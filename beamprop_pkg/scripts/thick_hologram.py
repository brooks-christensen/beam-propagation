import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from beamprop_pkg.beamprop.utils import (
    ln_safe,
    sdoOpt,
    braggMismatchOptA,
    braggMismatchOptB,
)  # , propagate  # kappa_from_fwhm,
from beamprop_pkg.beamprop.fields import (
    # gaussian_beam_1d,
    thick_hologram,
    build_input,
)  # , rect_2d
from beamprop_pkg.beamprop.grids import Grid2D
from beamprop_pkg.beamprop.propagators import BPM2D
from beamprop_pkg.beamprop.absorbers import absorbing_field_1d


# define output path to store figures
base_path = Path.cwd() / "beamprop_pkg" / "out"
out_png = base_path / "thick_hologram.png"
log_flag = False


# define physical constants
lambda0 = 0.1e-6
k0 = 2 * np.pi / lambda0
Nx = 1024 + 1
Nz = 1024 + 1
dx = 0.05e-6
dz = 0.05e-6
Lx = dx * (Nx - 1)
Lz = dz * (Nz - 1)
wAbs: int = 30  # number of points to include at the edges of absorber
gamma = 0.0  # strength of absorber at the boundary [0.0, 1.0]
n2 = 0.0  # strength of nonlinearity
k = k0
n_glass = 1.5
w_gauss = 5e-6
Lambda: float = (1.0 / 1.5) * 10**-6  # wavelength of grating
amplitude = 1e-2

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
nIN = thick_hologram(X, Z, 0.0, 20e-6, Lz / 2, 2e-6, Lambda, amplitude)
nRef = np.ones(X.shape)  # * n_glass
# nRef = rect_2d(X, Z, 0.0, 20e-6, Lz / 2, 2e-6)


# create simulation object
abs_mask = absorbing_field_1d(Nx, wAbs, gamma)
bpm_obj = BPM2D(lambda0, dx, Nx, Nz, dz, abs_mask, nIN, nRef)


# # create input field for single diffracted order
# tilt_deg_sdo = 0
# x0_sdo = -Lz / 2 * np.tan(tilt_deg_sdo * np.pi / 180)
# E0_sdo = gaussian_beam_1d(
#     x,
#     w_gauss,
#     x0_sdo,
#     k,
#     tilt_deg_sdo,
# )
# Eout_sdo, snapshots_sdo = bpm_obj.propagate(E0_sdo, n2, store_every=1)
best_angle_sdo = sdoOpt(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2, lambda0)
Eout_sdo, snapshots_sdo = bpm_obj.propagate(
    build_input(best_angle_sdo, x, k, Lz / 2, 0.0), n2, store_every=1
)

# create input field for Bragg mismatch
# tilt_deg_bm = -10
# x0_bm = -Lz / 2 * np.tan(tilt_deg_bm * np.pi / 180)
# E0_bm = gaussian_beam_1d(x, w_gauss, x0_bm, k, tilt_deg_bm)
best_angle_bmA = braggMismatchOptA(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2)
best_angle_bmB = braggMismatchOptB(build_input, dx, k, Lambda, x, Lz, bpm_obj, n2)

# perform propagation
Eout_bmA, snapshots_bmA = bpm_obj.propagate(
    build_input(best_angle_bmA, x, k, Lz / 2, 0.0), n2, store_every=1
)
Eout_bmB, snapshots_bmB = bpm_obj.propagate(
    build_input(best_angle_bmB, x, k, Lz / 2, 0.0), n2, store_every=1
)


# transpose if necessary
Is = snapshots_sdo if snapshots_sdo.shape == nIN.shape else snapshots_sdo.T
IbA = snapshots_bmA if snapshots_bmA.shape == nIN.shape else snapshots_bmA.T
IbB = snapshots_bmB if snapshots_bmB.shape == nIN.shape else snapshots_bmB.T


# take log of propagation intensity, if indicated
if log_flag:
    Is = ln_safe(Is)
    IbA = ln_safe(IbA)
    IbB = ln_safe(IbB)


# common extent: horizontal = z [µm], vertical = x [µm]
extent = (
    float(Z.min() * 1e6),
    float(Z.max() * 1e6),  # z-axis (x of plot)
    float(X.min() * 1e6),
    float(X.max() * 1e6),  # x-axis (y of plot)
)

fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True, sharey=True)


# --- Top Left: |E(x,z)|² (Single Diffracted Order) ---
im1 = axs[0, 0].imshow(
    Is,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
if log_flag:
    title0 = "Single Diffracted Order\nln(|E(x,z)|²)"
else:
    title0 = "Single Diffracted Order\n|E(x,z)|²"
axs[0, 0].set_title(title0)
axs[0, 0].set_ylabel("x [µm]")
fig.colorbar(im1, ax=axs[0, 0], label="intensity")


# --- Top Right: FFT(|E(x,z)|²) (Single Diffracted Order) ---
Ef = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Eout_sdo), norm="ortho"))

if log_flag:
    y = 20 * np.log10(np.maximum(np.abs(Ef), 1e-12))  # dB magnitude, safe floor
    axs[0, 1].set_title("FFT of Output Field\n|Ê(kx)| (dB)")
    axs[0, 1].set_ylabel("magnitude [dB]")
else:
    y = np.abs(Ef)  # linear magnitude
    axs[0, 1].set_title("FFT of Output Field\n|Ê(kx)|")
    axs[0, 1].set_ylabel("magnitude")

# frequency axis (optional but recommended)
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Eout_sdo.size, d=dx))  # [1/m]
axs[0, 1].plot(kx * 1e-6, y)  # kx in 1/µm
axs[0, 1].set_xlabel(r"$k_x$ [$\mathrm{\mu m^{-1}}$]")


# --- Middle Left: |E(x,z)|² (Bragg Mismatch, technique A) ---
im1 = axs[1, 0].imshow(
    IbA,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
if log_flag:
    title0 = "Bragg Mismatch (Minimize +1 Mode)\nln(|E(x,z)|²)"
else:
    title0 = "Bragg Mismatch (Minimize +1 Mode)\n|E(x,z)|²"
axs[1, 0].set_title(title0)
axs[1, 0].set_ylabel("x [µm]")
fig.colorbar(im1, ax=axs[1, 0], label="intensity")


# --- Middle Right: FFT(|E(x,z)|²) (Bragg Mismatch, technique A) ---
Ef_bm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Eout_bmA), norm="ortho"))

if log_flag:
    y = 20 * np.log10(np.maximum(np.abs(Ef_bm), 1e-12))  # dB magnitude, safe floor
    axs[1, 1].set_title("FFT of Output Field\n|Ê(kx)| (dB)")
    axs[1, 1].set_ylabel("magnitude [dB]")
else:
    y = np.abs(Ef_bm)  # linear magnitude
    axs[1, 1].set_title("FFT of Output Field\n|Ê(kx)|")
    axs[1, 1].set_ylabel("magnitude")

# frequency axis (optional but recommended)
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Eout_sdo.size, d=dx))  # [1/m]
axs[1, 1].plot(kx * 1e-6, y)  # kx in 1/µm
axs[1, 1].set_xlabel(r"$k_x$ [$\mathrm{\mu m^{-1}}$]")


# --- Bottom Left: |E(x,z)|² (Bragg Mismatch, technique B) ---
im1 = axs[2, 0].imshow(
    IbB,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
)
if log_flag:
    title0 = "Bragg Mismatch (Balance +1,0,-1 Modes)\nln(|E(x,z)|²)"
else:
    title0 = "Bragg Mismatch (Balance +1,0,-1 Modes)\n|E(x,z)|²"
axs[2, 0].set_title(title0)
axs[2, 0].set_ylabel("x [µm]")
axs[2, 0].set_xlabel("z [µm]")
fig.colorbar(im1, ax=axs[2, 0], label="intensity")


# --- Bottom Right: FFT(|E(x,z)|²) (Bragg Mismatch, technique B) ---
Ef_bm = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Eout_bmB), norm="ortho"))

if log_flag:
    y = 20 * np.log10(np.maximum(np.abs(Ef_bm), 1e-12))  # dB magnitude, safe floor
    axs[2, 1].set_title("FFT of Output Field\n|Ê(kx)| (dB)")
    axs[2, 1].set_ylabel("magnitude [dB]")
else:
    y = np.abs(Ef_bm)  # linear magnitude
    axs[2, 1].set_title("FFT of Output Field\n|Ê(kx)|")
    axs[2, 1].set_ylabel("magnitude")

# frequency axis (optional but recommended)
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Eout_sdo.size, d=dx))  # [1/m]
axs[2, 1].plot(kx * 1e-6, y)  # kx in 1/µm
axs[2, 1].set_xlabel(r"$k_x$ [$\mathrm{\mu m^{-1}}$]")


fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
