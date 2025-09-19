import numpy as np

def gaussian_beam_2d(X, Y, w0, x0=0.0, y0=0.0, kx=0.0, ky=0.0):
    """
    2D Gaussian envelope with optional tilt (kx, ky) [rad/m].
    E(x,y) = exp(-((x-x0)^2+(y-y0)^2)/w0^2) * exp(i*(kx*x + ky*y))
    """
    return np.exp(-(((X-x0)**2 + (Y-y0)**2) / (w0**2))) * np.exp(1j*(kx*X + ky*Y))

def gaussian_beam_1d(x, w0, x0=0.0, kx=0.0):
    """1D Gaussian with optional tilt."""
    return np.exp(-((x-x0)**2) / (w0**2)) * np.exp(1j*kx*x)

def rect_1d(x, width):
    """Rectangular aperture of given width centered at 0."""
    return ((np.abs(x) <= width/2).astype(float))

def sech_1d(x, a=1.0):
    """sech profile for fundamental bright soliton in normalized NLS: eta*sech(eta x)."""
    return 1.0/np.cosh(a*x)
