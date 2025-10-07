from dataclasses import dataclass
import numpy as np
from .utils import k0_from_lambda, fft1c, ifft1c  # , fft2c, ifft2c, fftfreq_2d

# from .absorbers import supergaussian_mask_2d
from typing import Tuple  # , Callable
# import matplotlib.pyplot as plt
# from pathlib import Path


@dataclass
class BPM2D:
    wavelength: float  # [m]
    # n0: float  # background index
    dx: float  # [m]
    nx: int
    nz: int
    dz: float  # step [m]
    absorber_mask: np.ndarray  # [nx] mask (<=1)
    n_field: np.ndarray  # [nx, nz] refractive index (>=1)
    # gate_func: Callable
    # z0: float
    # w: float
    # ramp: float

    def __post_init__(self):
        self.k0 = k0_from_lambda(self.wavelength)
        # 1D transverse kx for the 1-D FFT
        self.kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.nx, d=self.dx))
        self.kz = np.sqrt(np.maximum(0.0, self.k0**2 - self.kx**2))

    def propagate(self, E0, n2=0.0, store_every=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split-step propagation for the paraxial envelope A(x,y,z):
          ∂A/∂z = (i/2k) ∇_⊥^2 A + i k Δn(x,y,z)/n0 * A + i k n2 |A|^2 A
        n_of_xyz: callable returning Δn(x,y) (or scalar) at given z.
        n2: Kerr coefficient [m^2/W]-like in envelope units (scaled here).
        store_every: if >0, return snapshots every N steps.
        """
        E = E0.copy()
        snapshots = []

        for it in range(self.nz):
            # from the paper....
            # get inhomogeneous refractive index at the half step through averaging
            j2 = min(it + 1, self.n_field.shape[1] - 1)
            nIN = (self.n_field[:, it] + self.n_field[:, j2]) / 2

            # refract
            Er = fft1c(
                E * np.exp(-1j * self.k0 * (nIN + n2 * np.abs(E) ** 2) * self.dz)
            )

            # diffract
            E = ifft1c(Er * np.exp(-1j * self.kz * self.dz))

            # Apply absorber
            E *= self.absorber_mask

            # store snapshot
            if store_every and ((it + 1) % store_every == 0):
                snapshots.append(np.abs(E) ** 2)

        Ixz = np.stack(snapshots, axis=1) if snapshots else np.empty((self.nx, 0))

        return E, Ixz
