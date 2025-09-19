import argparse
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm


from .grids import Grid2D  # , Grid1D
from .fields import gaussian_beam_2d  # , gaussian_beam_1d, rect_1d, sech_1d
from .index_profiles import grin_prism, grin_lens  # , hologram_grating
from .absorbers import supergaussian_mask_2d
from .propagators import BPM2D, NLS1D
from .hologram import sweep_coupling_angles  # , bragg_angle

# from .utils import k0_from_lambda, fft2c, fft1c,
from .utils import robust_limits, auto_x_roi, safe_log10


# def fig_a(out, seed=0):
#     np.random.seed(seed)
#     g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
#     X, Y, dx, dy = g.mesh()
#     wavelength = 0.532e-6
#     n0 = 1.5
#     dz = 0.25e-3
#     nsteps = 300

#     n_map = grin_prism(X, Y, n0=n0, g_prism=2e4, g_lens=5e6)
#     dn = n_map - n0
#     M = supergaussian_mask_2d(X, Y, g.x_span, g.y_span, width_frac=0.2, order=6)

#     # k0 = 2 * np.pi / wavelength
#     tilt = 2e3
#     E0 = gaussian_beam_2d(X, Y, w0=30e-6, kx=tilt, ky=0.0)

#     bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)

#     def dn_at_z(z):
#         return dn

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
#     # Build x array for the crop:
#     # For 2D grids where you know x_span and nx:
#     x = (np.arange(Ixz.shape[0]) - Ixz.shape[0] // 2) * (g.x_span / g.nx)

#     # Auto-crop to where most of the energy lives (e.g. 92%)
#     (x_min, x_max), xsl = auto_x_roi(Ixz, x, frac=0.92, pad_pixels=6)
#     Ixz_c = Ixz[xsl, :]

#     # Robust color scaling (percentiles):
#     vmin, vmax = robust_limits(Ixz_c, lo=1.0, hi=99.5)

#     # If you prefer logarithmic dynamic range, replace imshow's vmin/vmax with:
#     # im = ax.imshow(safe_log10(Ixz_c), origin="lower", extent=..., aspect="auto")

#     # extent = [0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6]
#     extent = [0, nsteps * dz * 1e3, x_min * 1e6, x_max * 1e6]  # z in mm, x in µm

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     vmin_dn, vmax_dn = robust_limits((n_map - n0))
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
#         vmin=vmin_dn,
#         vmax=vmax_dn,
#     )

#     # im0 = axs[0].imshow(
#     #     (n_map - n0).T,
#     #     origin="lower",
#     #     extent=[
#     #         -g.x_span / 2 * 1e6,
#     #         g.x_span / 2 * 1e6,
#     #         -g.y_span / 2 * 1e6,
#     #         g.y_span / 2 * 1e6,
#     #     ],
#     #     aspect="equal",
#     # )
#     axs[0].set_title("Δn(x,y) (GRIN prism+lens)")
#     axs[0].set_xlabel("x [µm]")
#     axs[0].set_ylabel("y [µm]")
#     fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Δn")

#     # im1 = axs[1].imshow(Ixz, origin="lower", extent=extent, aspect="auto")
#     im1 = axs[1].imshow(
#         Ixz_c, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax
#     )
#     axs[1].set_title("|E(x,z)|² (central y-slice)")
#     axs[1].set_xlabel("z [mm]")
#     axs[1].set_ylabel("x [µm]")
#     fig.colorbar(im1, ax=axs[1], shrink=0.8, label="intensity")
#     fig.savefig(out, dpi=200)
#     plt.close(fig)


def fig_a(out, seed=0):
    np.random.seed(seed)

    # --- Grid & optics (slightly gentler so the action fills the panel) ---
    g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
    X, Y, dx, dy = g.mesh()
    wavelength = 0.532e-6
    n0 = 1.5
    dz = 0.15e-3  # was 0.25e-3
    nsteps = 300  # total length ~45 mm (enough to see the steering)

    # Prism weaker, lens weaker (less violent phase right at the entrance)
    n_map = grin_prism(X, Y, n0=n0, g_prism=5e3, g_lens=2e6)  # was 2e4, 5e6
    dn = n_map - n0

    # Absorber a bit wider so early loss is reduced
    M = supergaussian_mask_2d(
        X, Y, g.x_span, g.y_span, width_frac=0.30, order=6
    )  # was 0.20

    # Slightly larger input tilt so the prism effect is obvious, but not extreme
    tilt = 6e3  # was 2e3
    E0 = gaussian_beam_2d(X, Y, w0=30e-6, kx=tilt, ky=0.0)

    bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)

    center = g.ny // 2
    E = E0.copy()
    row_slices = []
    power_z = []  # diagnostic: total power vs z
    for it in range(nsteps):
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= np.exp(1j * (2 * np.pi / wavelength) * (dn / n0) * dz)
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= M
        row = np.abs(E[:, center]) ** 2
        row_slices.append(row)
        power_z.append(np.sum(np.abs(E) ** 2))  # keep an eye on absorber losses

    Ixz = np.array(row_slices).T  # [nx, nsteps]

    # --- Auto-crop + LOG view so the whole evolution is visible ---
    x = (np.arange(Ixz.shape[0]) - Ixz.shape[0] // 2) * (g.x_span / g.nx)
    (x_min, x_max), xsl = auto_x_roi(
        Ixz, x, frac=0.98, pad_pixels=8
    )  # a little tighter crop
    Ixz_c = Ixz[xsl, :]

    # Log-intensity (much friendlier when the entrance slice is hot)
    Ilog = safe_log10(Ixz_c)  # log10(|E|^2) with a safe floor
    vmin, vmax = robust_limits(Ilog, lo=1.0, hi=99.5)
    extent = [0, nsteps * dz * 1e3, x_min * 1e6, x_max * 1e6]  # z in mm, x in µm

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Left: Δn map with robust scaling
    vmin_dn, vmax_dn = robust_limits((n_map - n0))
    im0 = axs[0].imshow(
        (n_map - n0).T,
        origin="lower",
        extent=[
            -g.x_span / 2 * 1e6,
            g.x_span / 2 * 1e6,
            -g.y_span / 2 * 1e6,
            g.y_span / 2 * 1e6,
        ],
        aspect="equal",
        vmin=vmin_dn,
        vmax=vmax_dn,
    )
    axs[0].set_title("Δn(x,y) (GRIN prism+lens)")
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel("y [µm]")

    # Right: log-intensity x–z slice (no more crushed dynamic range)

    im1 = axs[1].imshow(
        Ilog, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax
    )
    axs[1].set_title("log10 |E(x,z)|² (central y-slice)")
    axs[1].set_xlabel("z [mm]")
    axs[1].set_ylabel("x [µm]")

    # Optional: print end-power so you can gauge absorber impact
    print(f"Power loss: {power_z[-1] / power_z[0]:.3f} of initial")

    fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Δn")
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label="log10 intensity")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_b(out):
    g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
    X, Y, dx, dy = g.mesh()
    wavelength = 0.6328e-6
    n0 = 1.5
    dz = 0.20e-3
    nsteps = 250

    n_map = grin_lens(X, Y, n0=n0, g_lens=1e7)
    dn = n_map - n0
    M = supergaussian_mask_2d(X, Y, g.x_span, g.y_span, width_frac=0.2, order=6)
    E0 = gaussian_beam_2d(X, Y, w0=35e-6)

    bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)
    center = g.ny // 2
    E = E0.copy()
    row_slices = []
    for it in range(nsteps):
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= np.exp(1j * (2 * np.pi / wavelength) * (dn / n0) * dz)
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= M
        row_slices.append(np.abs(E[:, center]) ** 2)
    Ixz = np.array(row_slices).T
    extent = [0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axs[0].imshow(
        (n_map - n0).T,
        origin="lower",
        extent=[
            -g.x_span / 2 * 1e6,
            g.x_span / 2 * 1e6,
            -g.y_span / 2 * 1e6,
            g.y_span / 2 * 1e6,
        ],
        aspect="equal",
    )
    axs[0].set_title("Δn(x,y) (GRIN lens)")
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel("y [µm]")
    fig.colorbar(im0, ax=axs[0], shrink=0.8, label="Δn")

    im1 = axs[1].imshow(Ixz, origin="lower", extent=extent, aspect="auto")
    axs[1].set_title("Focusing |E(x,z)|²")
    axs[1].set_xlabel("z [mm]")
    axs[1].set_ylabel("x [µm]")
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label="intensity")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_c(out):
    g = Grid2D(x_span=300e-6, y_span=300e-6, nx=256, ny=256)
    X, Y, dx, dy = g.mesh()
    wavelength = 1.064e-6
    n0 = 1.45
    dz = 0.25e-3
    nsteps = 300
    n2 = 3e-20

    M = supergaussian_mask_2d(X, Y, g.x_span, g.y_span, width_frac=0.25, order=6)
    E0 = gaussian_beam_2d(X, Y, w0=40e-6) * 20.0

    bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)

    center = g.ny // 2
    E = E0.copy()
    row_slices = []
    for it in range(nsteps):
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= np.exp(1j * (2 * np.pi / wavelength) * n2 * (np.abs(E) ** 2) * dz)
        E = np.fft.ifft2(
            np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
            norm="ortho",
        )
        E *= M
        row_slices.append(np.abs(E[:, center]) ** 2)
    Ixz = np.array(row_slices).T
    extent = (0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(Ixz, origin="lower", extent=extent, aspect="auto")
    ax.set_title("High‑intensity Kerr: |E(x,z)|² (central y)")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [µm]")
    fig.colorbar(im, ax=ax, shrink=0.8, label="intensity")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_d(out):
    nx = 1024
    x_span = 40.0
    dx = x_span / nx
    dz = 0.005
    nsteps = 3000

    x = (np.arange(nx) - nx // 2) * dx
    eta = 1.0
    psi0 = eta * 1 / np.cosh(eta * x)

    nls = NLS1D(dx=dx, nx=nx, dz=dz, n_steps=nsteps)
    psi, snaps = nls.propagate(psi0, store_every=30)
    snaps = np.array(snaps)
    Ixz = np.abs(snaps.T) ** 2
    extent = (0, Ixz.shape[1] * 30 * dz, x[0], x[-1])

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(Ixz, origin="lower", extent=extent, aspect="auto")
    ax.set_title("Focusing NLS bright soliton |ψ(x,z)|²")
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_e(out):
    g = Grid2D(x_span=200e-6, y_span=200e-6, nx=256, ny=256)
    X, Y, dx, dy = g.mesh()
    wavelength = 0.532e-6
    n0 = 1.5
    dz = 0.2e-3
    nsteps = 250

    kx_tilt = 1.5e4
    E0 = gaussian_beam_2d(X, Y, w0=30e-6, kx=kx_tilt)

    def run(width_frac):
        M = supergaussian_mask_2d(
            X, Y, g.x_span, g.y_span, width_frac=width_frac, order=6
        )
        bpm = BPM2D(wavelength, n0, dx, dy, g.nx, g.ny, dz, nsteps, M)
        center = g.ny // 2
        E = E0.copy()
        rows = []
        for it in range(nsteps):
            E = np.fft.ifft2(
                np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
                norm="ortho",
            )
            E = np.fft.ifft2(
                np.fft.fft2(E, norm="ortho") * np.fft.fftshift(bpm.lin_halfstep),
                norm="ortho",
            )
            E *= M
            rows.append(np.abs(E[:, center]) ** 2)
        Ixz = np.array(rows).T
        return Ixz

    I_bad = run(0.05)
    I_good = run(0.25)

    extent = [0, nsteps * dz * 1e3, -g.x_span / 2 * 1e6, g.x_span / 2 * 1e6]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].imshow(I_bad, origin="lower", extent=extent, aspect="auto")
    axs[0].set_title("Insufficient absorber: wrap‑around/aliasing")
    axs[0].set_xlabel("z [mm]")
    axs[0].set_ylabel("x [µm]")
    axs[1].imshow(I_good, origin="lower", extent=extent, aspect="auto")
    axs[1].set_title("Sufficient absorber")
    axs[1].set_xlabel("z [mm]")
    axs[1].set_ylabel("x [µm]")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_f(out):
    nx = 2048
    x_span = 400e-6
    x = (np.arange(nx) - nx // 2) * (x_span / nx)
    dx = x[1] - x[0]

    wavelength = 0.633e-6
    n = 1.5
    period = 12e-6
    dn = 2e-4
    L = 5e-3

    kx_in = 0.0
    E0 = np.exp(-(x**2) / (40e-6) ** 2) * np.exp(1j * kx_in * x)

    k0 = 2 * np.pi / wavelength
    phi = k0 * (dn * L / n) * np.cos(2 * np.pi * x / period)
    E_out = E0 * np.exp(1j * phi)

    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_out), norm="ortho"))
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    I_x = np.abs(E_out) ** 2
    I_k = np.abs(F) ** 2

    thetas_deg, etas, (th_min, th_max) = sweep_coupling_angles(
        n, wavelength, period, L, dn, order=1, angle_range_deg=3.0, samples=1501
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].plot(x * 1e6, I_x)
    axs[0].set_title("Output near‑field |E_out(x)|²")
    axs[0].set_xlabel("x [µm]")
    axs[0].set_ylabel("intensity")

    axs[1].plot(kx, I_k)
    axs[1].set_title("Far‑field |FFT[E_out]|² (peaks at 0, ±K)")
    axs[1].set_xlabel("kx [rad/m]")
    axs[1].set_ylabel("spectral power")
    axs[1].axvline(2 * np.pi / period, linestyle="--")
    axs[1].axvline(-2 * np.pi / period, linestyle="--")

    fig.suptitle(
        f"Approx min/max coupling angles (deg): min≈{th_min:.3f}, max≈{th_max:.3f}"
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        prog="beamprop-figs", description="Reproduce optics figures (a)-(f)."
    )
    p.add_argument("which", choices=list("abcdef"), help="figure to generate")
    p.add_argument("--out", required=True, help="output image path (png)")
    args = p.parse_args()

    if args.which == "a":
        fig_a(args.out)
    elif args.which == "b":
        fig_b(args.out)
    elif args.which == "c":
        fig_c(args.out)
    elif args.which == "d":
        fig_d(args.out)
    elif args.which == "e":
        fig_e(args.out)
    elif args.which == "f":
        fig_f(args.out)
