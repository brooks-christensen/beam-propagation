from dataclasses import dataclass
import numpy as np
from .utils import k0_from_lambda, fft2c, ifft2c, fftfreq_2d, fft1c, ifft1c
from .absorbers import supergaussian_mask_2d

@dataclass
class BPM2D:
    wavelength: float           # [m]
    n0: float                   # background index
    dx: float                   # [m]
    dy: float                   # [m]
    nx: int
    ny: int
    dz: float                   # step [m]
    n_steps: int                # number of steps
    absorber_mask: np.ndarray   # [nx,ny] mask (<=1)

    def __post_init__(self):
        self.k0 = k0_from_lambda(self.wavelength)
        self.kx, self.ky = fftfreq_2d(self.nx, self.ny, self.dx, self.dy)
        k = self.n0 * self.k0
        self.lin_halfstep = np.exp(-1j * (self.kx**2 + self.ky**2) * self.dz / (4.0 * k))

    def propagate(self, E0, n_of_xyz=None, n2=0.0, store_every=0, z0=0.0):
        """
        Split-step propagation for the paraxial envelope A(x,y,z):
          ∂A/∂z = (i/2k) ∇_⊥^2 A + i k Δn(x,y,z)/n0 * A + i k n2 |A|^2 A
        n_of_xyz: callable returning Δn(x,y) (or scalar) at given z.
        n2: Kerr coefficient [m^2/W]-like in envelope units (scaled here).
        store_every: if >0, return snapshots every N steps.
        """
        E = E0.copy()
        snapshots = []
        z = z0

        for it in range(self.n_steps):
            # Linear half-step in k-space (diffraction)
            E = ifft2c(self.lin_halfstep * fft2c(E))

            # Index / nonlinear phase (full step in real space)
            if n_of_xyz is not None:
                dn = n_of_xyz(z)
            else:
                dn = 0.0

            phase = self.k0 * (dn / self.n0) * self.dz
            E *= np.exp(1j * phase)

            if n2 != 0.0:
                E *= np.exp(1j * self.k0 * n2 * np.abs(E)**2 * self.dz)

            # Linear half-step again
            E = ifft2c(self.lin_halfstep * fft2c(E))

            # Apply absorber
            E *= self.absorber_mask

            z += self.dz
            if store_every and ((it+1) % store_every == 0):
                snapshots.append(E.copy())

        return E, snapshots

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
        kx = 2*np.pi*np.fft.fftfreq(self.nx, d=self.dx)
        self.linear = np.exp(-1j * 0.5 * (kx**2) * self.dz)

    def propagate(self, psi0, store_every=0):
        psi = psi0.copy()
        snaps = []
        for it in range(self.n_steps):
            PSI = np.fft.fft(psi, norm="ortho")
            PSI = self.linear * PSI
            psi = np.fft.ifft(PSI, norm="ortho")

            psi *= np.exp(1j * np.abs(psi)**2 * self.dz)

            PSI = np.fft.fft(psi, norm="ortho")
            PSI = self.linear * PSI
            psi = np.fft.ifft(PSI, norm="ortho")

            if store_every and ((it+1) % store_every == 0):
                snaps.append(psi.copy())
        return psi, snaps
