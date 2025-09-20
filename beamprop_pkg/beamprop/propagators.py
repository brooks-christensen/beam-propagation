from dataclasses import dataclass
import numpy as np
from .utils import k0_from_lambda, fft1c, ifft1c  # , fft2c, ifft2c, fftfreq_2d

# from .absorbers import supergaussian_mask_2d
from typing import Tuple, Callable


@dataclass
class BPM2D:
    wavelength: float  # [m]
    n0: float  # background index
    dx: float  # [m]
    nx: int
    nz: int
    dz: float  # step [m]
    n_steps: int  # number of steps
    absorber_mask: np.ndarray  # [nx] mask (<=1)
    n_field: np.ndarray  # [nx, nz] refractive index (>=1)
    gate_func: Callable
    z0: float
    w: float
    ramp: float

    # def __post_init__(self):
    #     self.k0 = k0_from_lambda(self.wavelength)
    #     # self.kx, self.kz = fftfreq_2d(self.nx, self.nz, self.dx, self.dz)
    #     kt = np.linspace(-self.k0, self.k0, self.nx)
    #     self.kz = np.sqrt(self.k0**2 - kt**2)
    #     # k = self.n0 * self.k0
    #     # self.lin_halfstep = np.exp(
    #     #     -1j * (self.kx**2 + self.kz**2) * self.dz / (4.0 * k)
    #     # )
    #     self.lin_halfstep = np.exp(-1j * self.kz * self.dz)

    def __post_init__(self):
        self.k0 = k0_from_lambda(self.wavelength)
        # 1D transverse kx for the 1-D FFT
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        # Uniform background longitudinal wavenumber kz = sqrt((n0*k0)^2 - kx^2)
        k = self.n0 * self.k0
        self.kz = np.sqrt(np.maximum(0.0, k**2 - self.kx**2))
        self.lin_halfstep = np.exp(-1j * self.kz * self.dz)  # free-space/host medium

    def propagate(
        self,
        E0,
        n2=0.0,
        store_every=0,  # , z0=0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split-step propagation for the paraxial envelope A(x,y,z):
          ∂A/∂z = (i/2k) ∇_⊥^2 A + i k Δn(x,y,z)/n0 * A + i k n2 |A|^2 A
        n_of_xyz: callable returning Δn(x,y) (or scalar) at given z.
        n2: Kerr coefficient [m^2/W]-like in envelope units (scaled here).
        store_every: if >0, return snapshots every N steps.
        """
        E = E0.copy()
        snapshots = []
        # z = z0

        for it in range(self.n_steps):
            # # Linear half-step in k-space (diffraction)
            # E = ifft1c(self.lin_halfstep * fft1c(E))

            # # Index / nonlinear phase (full step in real space)
            # if n_of_xyz is not None:
            #     dn = n_of_xyz(z)
            # else:
            #     dn = 0.0

            # phase = self.k0 * (dn / self.n0) * self.dz
            # E *= np.exp(1j * phase)

            # if n2 != 0.0:
            #     E *= np.exp(1j * self.k0 * n2 * np.abs(E) ** 2 * self.dz)

            # # Linear half-step again
            # E = ifft2c(self.lin_halfstep * fft2c(E))

            # from the paper....
            # get inhomogeneous refractive index at the half step through averaging
            j2 = min(it + 1, self.n_field.shape[1] - 1)
            nIN = (self.n_field[:, it] + self.n_field[:, j2]) / 2 / self.n0

            # refract
            if self.gate_func(it, self.z0, self.w, self.ramp) > 0.0:
                Er = fft1c(
                    E
                    * np.exp(
                        -1j
                        * self.k0
                        * (nIN + n2 * np.abs(E) ** 2)
                        * self.dz
                        * self.gate_func(it, self.z0, self.w, self.ramp)
                    )
                )
            else:
                # free-space propagation
                Er = fft1c(E * np.exp(-1j * self.kz * self.dz))

            # diffract
            E = ifft1c(Er * self.lin_halfstep)

            # Apply absorber
            E *= self.absorber_mask

            # z += self.dz
            if store_every and ((it + 1) % store_every == 0):
                snapshots.append(np.abs(E) ** 2)

        Ixz = np.stack(snapshots, axis=1) if snapshots else np.empty((self.nx, 0))

        return E, Ixz


@dataclass
class NLS1D:
    """
    Normalized focusing NLS (1D):
      i ∂ψ/∂z + (1/2) ∂^2ψ/∂x^2 + |ψ|^2 ψ = 0
    Fundamental bright soliton: ψ(x, z) = η sech(η x) exp(i η^2 z/2).
    """

    dx: float
    nx: int
    dz: float
    n_steps: int

    def __post_init__(self):
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.linear = np.exp(-1j * 0.5 * (kx**2) * self.dz)

    def propagate(self, psi0, store_every=0):
        psi = psi0.copy()
        snaps = []
        for it in range(self.n_steps):
            PSI = np.fft.fft(psi, norm="ortho")
            PSI = self.linear * PSI
            psi = np.fft.ifft(PSI, norm="ortho")

            psi *= np.exp(1j * np.abs(psi) ** 2 * self.dz)

            PSI = np.fft.fft(psi, norm="ortho")
            PSI = self.linear * PSI
            psi = np.fft.ifft(PSI, norm="ortho")

            if store_every and ((it + 1) % store_every == 0):
                snaps.append(psi.copy())
        return psi, snaps
