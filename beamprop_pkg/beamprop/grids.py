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

    def arrays(self):
        dx = self.x_span / (self.nx - 1)
        dz = self.z_span / (self.nz - 1)
        x = (np.arange(self.nx) - self.nx // 2) * dx
        z = (np.arange(self.nz)) * dz  # - self.nz // 2
        return x, z, dx, dz

    def mesh(self):
        x, z, dx, dz = self.arrays()
        X, Z = np.meshgrid(x, z, indexing="ij")
        return X, Z, dx, dz
