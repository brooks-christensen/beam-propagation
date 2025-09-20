import argparse
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm


from .grids import Grid2D  # , Grid1D
from .fields import (
    gaussian_beam_1d,
    # grin_prism_n_xz,
)  # , gaussian_beam_2d, rect_1d, sech_1d

# from .index_profiles import grin_lens  # , hologram_grating, grin_prism
from .absorbers import absorbing_field_1d  # supergaussian_mask_2d,
from .propagators import BPM2D  # , NLS1D
# from .hologram import sweep_coupling_angles  # , bragg_angle

# from .utils import k0_from_lambda, fft2c, fft1c,
# from .utils import robust_limits, auto_x_roi  # , safe_log10
from .utils import raised_cosine_gate


def fig_a(out, seed=0):
    np.random.seed(seed)

    nx = 512
    nz = 300
    x_span = 1.024e-3
    z_span = 3e-3
    dx = x_span / (nx - 1)
    dz = z_span / (nz - 1)

    # ----- Geometry & sampling from the handout -----
    g = Grid2D(x_span=x_span, z_span=z_span, nx=nx, nz=nz)
    xs = (np.array(range(nx)) - nx // 2) * dx
    zs = np.array(range(nz)) * dz
    X, Z, dx, dz = g.mesh()
    # wavelength = 0.532e-6
    wavelength = 1e-6
    n0 = 1.5
    dz = 10e-6  # 10 µm
    nsteps = 300  # ~3 mm total
    # z = (np.arange(nsteps)) * dz  # + 0.5

    absorber_width = 50
    absorber_gamma = 0

    # Prism slab (Eq. 3): δn(x) = 0.07 * x/Lx, applied only in rect((z-1050 µm)/1900 µm)
    slope = 0.07 / x_span
    dn_x = slope * X  # uniform in y by construction

    z0, w = 1.05e-3, 1.90e-3  # center and thickness of the slab (meters)

    # nIN = grin_prism_n_xz(X, Z, Lx=x_span) - n0

    # Input (Eq. 4): Gaussian with 60 µm width; no initial tilt
    E = gaussian_beam_1d(xs, w0=60.0e-6, x0=0.0, kx=0.0)
    print(E.shape)

    # Absorber (gentle, symmetric)
    # M = supergaussian_mask_2d(X, Z, g.x_span, g.z_span, width_frac=0.30, order=6)
    M = absorbing_field_1d(nx, absorber_width, absorber_gamma)

    z0 = 1.05e-3
    w = 1.90e-3
    ramp = 5 * dz

    bpm = BPM2D(
        wavelength,
        n0,
        dx,
        g.nx,
        g.nz,
        dz,
        nsteps - 1,
        M,
        dn_x,
        raised_cosine_gate,
        z0=z0,
        w=w,
        ramp=ramp,
    )
    # k0 = 2 * np.pi / wavelength

    # ----- Propagate with the GRIN phase only inside the slab -----
    # center_y = g.nz // 2
    # rows = []
    # centroid = []
    Eout, Ixz = bpm.propagate(E, 0.0, store_every=1)
    print(f"Eout: {Eout.shape}")
    print(f"Ixz: {Ixz.shape}")

    # Ixz = np.array(rows).T  # shape [nx, nsteps]
    # x = (np.arange(g.nx) - g.nx // 2) * dx
    z_mm = (np.arange(nsteps)) * dz * 1e3  # for plotting

    # ----- Plot: left = Δn(x) (uniform in y), right = |E(x,z)|^2 with slab & trajectory ----

    # Auto-crop x to energetic region (keeps axes in µm)
    # (x_min, x_max), xsl = auto_x_roi(snapshots, x, frac=0.98, pad_pixels=8)
    # Ixz_c = snapshots[xsl, :]
    # print(f"Ixz_c: {Ixz_c.shape}")
    # print(Ixz_c[:5, :5])
    x_um = xs * 1e6
    z_len_mm = Ixz.shape[1] * dz * 1e3

    # vmin, vmax = robust_limits(
    #     Ixz_c, lo=1.0, hi=99.5
    # )  # linear scaling, robust to hot spots

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # gate_z: shape (nsteps,)
    # Make a 2D map Δn(x,z) = (slope * x * mask_x) * gate(z)
    # gate = raised_cosine_gate(zs, z0, w, ramp=ramp)
    # dn_xz = (slope * X)[:, None] * gate[None, :]  # shape (nx, nsteps)

    # LEFT: 1D Δn(x) profile (clarifies "uniform in y")
    # axs[0].plot(
    #     (x * 1e6), (dn_x[:, center_y])
    # )  # slice at y-center; same everywhere in y
    # axs[0].set_title("Δn(x) (uniform in y); prism slab only in z")
    # axs[0].set_xlabel("x [µm]")
    # axs[0].set_ylabel("Δn")
    # axs[0].grid(True, alpha=0.25)
    # Build an x–z index map: vacuum (1.0) outside slab/aperture; prism inside
    X1d, Z1d = np.meshgrid(xs, zs, indexing="ij")
    inside = np.abs(Z1d - z0) <= w / 2  # & (np.abs(X1d) <= prism_width_x / 2)
    n_xz = np.ones_like(X1d)  # vacuum by default
    n_xz[inside] = n0 + slope * X1d[inside]  # prism interior

    n_center = n0  # index at x=0 inside prism
    dn_plot = n_xz - n_center  # what to display

    _ = axs[0].imshow(
        dn_plot,
        origin="lower",
        extent=[z_mm[0], z_mm[-1], x_um[0], x_um[-1]],
        aspect="auto",
        cmap="coolwarm",
        vmin=-np.max(np.abs(dn_plot)),
        vmax=np.max(np.abs(dn_plot)),
    )
    axs[0].set_title("n(x,z) − n_center")
    axs[0].set_xlabel("z [mm]")
    axs[0].set_ylabel("x [µm]")

    # RIGHT: intensity map of central-y slice with slab limits and centroid overlay
    extent = [0.0, z_len_mm, x_um[0], x_um[-1]]  # z [mm], x [µm]
    im = axs[1].imshow(
        Ixz,
        origin="lower",
        extent=extent,
        aspect="auto",  # , vmin=vmin, vmax=vmax
    )
    axs[1].set_title("|E(x,z)|² (central y-slice)")
    axs[1].set_xlabel("z [mm]")
    axs[1].set_ylabel("x [µm]")
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="intensity")

    # Draw slab start/end
    z_start_mm = (z0 - w / 2) * 1e3
    z_end_mm = (z0 + w / 2) * 1e3
    axs[1].axvline(z_start_mm, ls="--", lw=1.5, color="w", alpha=0.9)
    axs[1].axvline(z_end_mm, ls="--", lw=1.5, color="w", alpha=0.9)

    # Overlay the centroid trajectory
    # cx_um = np.array(centroid) * 1e6
    # axs[1].plot(z_mm, cx_um, color="w", lw=2, alpha=0.9)

    # Optional: fit a straight line to the post-prism part to emphasize "straight after exit"
    # post = z_mm >= (z_end_mm + 0.1)  # ignore 0.1 mm right after exit
    # if np.any(post):
    #     p = np.polyfit(z_mm[post], cx_um[post], 1)
    #     axs[1].plot(z_mm[post], np.polyval(p, z_mm[post]), color="0.9", lw=2, ls=":")

    fig.savefig(out, dpi=200)
    plt.close(fig)


# def fig_b(out):
#     g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
#     X, Y, dx, dy = g.mesh()
#     wavelength = 0.6328e-6
#     n0 = 1.5
#     dz = 0.20e-3
#     nsteps = 250

#     n_map = grin_lens(X, Y, n0=n0, g_lens=1e7)
#     dn = n_map - n0
#     M = supergaussian_mask_2d(X, Y, g.x_span, g.y_span, width_frac=0.2, order=6)
#     E0 = gaussian_beam_2d(X, Y, w0=35e-6)

#     bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)
#     center = g.ny // 2
#     E = E0.copy()
#     row_slices = []
#     for it in range(nsteps):
#         E = np.fft.ifft2(
#             np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#             norm="ortho",
#         )
#         E *= np.exp(1j * (2 * np.pi / wavelength) * (dn / n0) * dz)
#         E = np.fft.ifft2(
#             np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#             norm="ortho",
#         )
#         E *= M
#         row_slices.append(np.abs(E[:, center]) ** 2)
#     Ixz = np.array(row_slices).T
#     extent = [0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6]

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     im0 = axs[0].imshow(
#         (n_map - n0).T,
#         origin="lower",
#         extent=[
#             -g.x_span / 2 * 1e6,
#             g.x_span / 2 * 1e6,
#             -g.y_span / 2 * 1e6,
#             g.y_span / 2 * 1e6,
#         ],
#         aspect="equal",
#     )
#     axs[0].set_title("Δn(x,y) (GRIN lens)")
#     axs[0].set_xlabel("x [µm]")
#     axs[0].set_ylabel("y [µm]")
#     fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Δn")

#     im1 = axs[1].imshow(Ixz, origin="lower", extent=extent, aspect="auto")
#     axs[1].set_title("Focusing |E(x,z)|²")
#     axs[1].set_xlabel("z [mm]")
#     axs[1].set_ylabel("x [µm]")
#     fig.colorbar(im1, ax=axs[1], shrink=0.8, label="intensity")
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


# def fig_c(out):
#     g = Grid2D(x_span=300e-6, y_span=300e-6, nx=256, ny=256)
#     X, Y, dx, dy = g.mesh()
#     wavelength = 1.064e-6
#     n0 = 1.45
#     dz = 0.25e-3
#     nsteps = 300
#     n2 = 3e-20

#     M = supergaussian_mask_2d(X, Y, g.x_span, g.y_span, width_frac=0.25, order=6)
#     E0 = gaussian_beam_2d(X, Y, w0=40e-6) * 20.0

#     bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)

#     center = g.ny // 2
#     E = E0.copy()
#     row_slices = []
#     for it in range(nsteps):
#         E = np.fft.ifft2(
#             np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#             norm="ortho",
#         )
#         E *= np.exp(1j * (2 * np.pi / wavelength) * n2 * (np.abs(E) ** 2) * dz)
#         E = np.fft.ifft2(
#             np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#             norm="ortho",
#         )
#         E *= M
#         row_slices.append(np.abs(E[:, center]) ** 2)
#     Ixz = np.array(row_slices).T
#     extent = (0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6)

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
#     im = ax.imshow(Ixz, origin="lower", extent=extent, aspect="auto")
#     ax.set_title("High‑intensity Kerr: |E(x,z)|² (central y)")
#     ax.set_xlabel("z [mm]")
#     ax.set_ylabel("x [µm]")
#     fig.colorbar(im, ax=ax, shrink=0.8, label="intensity")
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


# def fig_d(out):
#     nx = 1024
#     x_span = 40.0
#     dx = x_span / nx
#     dz = 0.005
#     nsteps = 3000

#     x = (np.arange(nx) - nx // 2) * dx
#     eta = 1.0
#     psi0 = eta * 1 / np.cosh(eta * x)

#     nls = NLS1D(dx=dx, nx=nx, dz=dz, n_steps=nsteps)
#     psi, snaps = nls.propagate(psi0, store_every=30)
#     snaps = np.array(snaps)
#     Ixz = np.abs(snaps.T) ** 2
#     extent = (0, Ixz.shape[1] * 30 * dz, x[0], x[-1])

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
#     im = ax.imshow(Ixz, origin="lower", extent=extent, aspect="auto")
#     ax.set_title("Focusing NLS bright soliton |ψ(x,z)|²")
#     ax.set_xlabel("z")
#     ax.set_ylabel("x")
#     fig.colorbar(im, ax=ax, shrink=0.8)
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


# def fig_e(out):
#     g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
#     X, Y, dx, dy = g.mesh()
#     wavelength = 0.532e-6
#     n0 = 1.5
#     dz = 0.2e-3
#     nsteps = 250

#     kx_tilt = 1.5e4
#     E0 = gaussian_beam_2d(X, Y, w0=30e-6, kx=kx_tilt)

#     def run(width_frac):
#         M = supergaussian_mask_2d(
#             X, Y, g.x_span, g.y_span, width_frac=width_frac, order=6
#         )
#         bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)
#         center = g.ny // 2
#         E = E0.copy()
#         rows = []
#         for it in range(nsteps):
#             E = np.fft.ifft2(
#                 np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#                 norm="ortho",
#             )
#             E = np.fft.ifft2(
#                 np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
#                 norm="ortho",
#             )
#             E *= M
#             rows.append(np.abs(E[:, center]) ** 2)
#         Ixz = np.array(rows).T
#         return Ixz

#     I_bad = run(0.05)
#     I_good = run(0.25)

#     extent = [0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6]
#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     axs[0].imshow(I_bad, origin="lower", extent=extent, aspect="auto")
#     axs[0].set_title("Insufficient absorber: wrap‑around/aliasing")
#     axs[0].set_xlabel("z [mm]")
#     axs[0].set_ylabel("x [µm]")
#     axs[1].imshow(I_good, origin="lower", extent=extent, aspect="auto")
#     axs[1].set_title("Sufficient absorber")
#     axs[1].set_xlabel("z [mm]")
#     axs[1].set_ylabel("x [µm]")
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


# def fig_f(out):
#     nx = 2048
#     x_span = 400e-6
#     x = (np.arange(nx) - nx // 2) * (x_span / nx)
#     dx = x[1] - x[0]

#     wavelength = 0.633e-6
#     n = 1.5
#     period = 12e-6
#     dn = 2e-4
#     L = 5e-3

#     kx_in = 0.0
#     E0 = np.exp(-(x**2) / (40e-6) ** 2) * np.exp(1j * kx_in * x)

#     k0 = 2 * np.pi / wavelength
#     phi = k0 * (dn * L / n) * np.cos(2 * np.pi * x / period)
#     E_out = E0 * np.exp(1j * phi)

#     F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_out), norm="ortho"))
#     kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
#     I_x = np.abs(E_out) ** 2
#     I_k = np.abs(F) ** 2

#     thetas_deg, etas, (th_min, th_max) = sweep_coupling_angles(
#         n, wavelength, period, L, dn, order=1, angle_range_deg=3.0, samples=1501
#     )

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     axs[0].plot(x * 1e6, I_x)
#     axs[0].set_title("Output near‑field |E_out(x)|²")
#     axs[0].set_xlabel("x [µm]")
#     axs[0].set_ylabel("intensity")

#     axs[1].plot(kx, I_k)
#     axs[1].set_title("Far‑field |FFT[E_out]|² (peaks at 0, ±K)")
#     axs[1].set_xlabel("kx [rad/m]")
#     axs[1].set_ylabel("spectral power")
#     axs[1].axvline(2 * np.pi / period, linestyle="--")
#     axs[1].axvline(-2 * np.pi / period, linestyle="--")

#     fig.suptitle(
#         f"Approx min/max coupling angles (deg): min≈{th_min:.3f}, max≈{th_max:.3f}"
#     )
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        prog="beamprop-figs", description="Reproduce optics figures (a)-(f)."
    )
    p.add_argument("which", choices=list("abcdef"), help="figure to generate")
    p.add_argument("--out", required=True, help="output image path (png)")
    args = p.parse_args()

    if args.which == "a":
        fig_a(args.out)
    # elif args.which == "b":
    #     fig_b(args.out)
    # elif args.which == "c":
    #     fig_c(args.out)
    # elif args.which == "d":
    #     fig_d(args.out)
    # elif args.which == "e":
    #     fig_e(args.out)
    # elif args.which == "f":
    #     fig_f(args.out)
