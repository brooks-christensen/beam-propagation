from dataclasses import dataclass
import numpy as np


@dataclass
class Grid1D:
    x_span: float  # physical width [m]
    nx: int  # number of samples

    def arrays(self):
        dx = self.x_span / self.nx
        x = (np.arange(self.nx) - self.nx // 2) * dx
        return x, dx


@dataclass
class Grid2D:
    x_span: float  # width [m]
    z_span: float  # height [m]
    nx: int
    nz: int
    center: bool

    def arrays(self):
        if self.center:
            x = np.linspace(-self.x_span / 2, +self.x_span / 2, self.nx)
        else:
            x = np.linspace(0, self.x_span, self.nx)
        z = np.linspace(0.0, self.z_span, self.nz)
        dx = x[1] - x[0]
        dz = z[1] - z[0]
        return x, z, dx, dz

    def mesh(self):
        x, z, dx, dz = self.arrays()
        X, Z = np.meshgrid(x, z, indexing="ij")
        return X, Z, dx, dz
