from dataclasses import dataclass
import numpy as np

@dataclass
class Grid1D:
    x_span: float   # physical width [m]
    nx: int         # number of samples

    def arrays(self):
        dx = self.x_span / self.nx
        x = (np.arange(self.nx) - self.nx//2) * dx
        return x, dx

@dataclass
class Grid2D:
    x_span: float   # width [m]
    y_span: float   # height [m]
    nx: int
    ny: int

    def arrays(self):
        dx = self.x_span / self.nx
        dy = self.y_span / self.ny
        x = (np.arange(self.nx) - self.nx//2) * dx
        y = (np.arange(self.ny) - self.ny//2) * dy
        return x, y, dx, dy

    def mesh(self):
        x, y, dx, dy = self.arrays()
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y, dx, dy
