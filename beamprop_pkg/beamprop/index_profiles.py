import numpy as np

def grin_prism(X, Y, n0=1.5, g_prism=50.0, g_lens=0.0):
    """
    Graded-index profile that acts like a prism (linear ramp in x) and optionally lens (quadratic).
    n(x,y) = n0 + g_prism*x + g_lens*(x^2 + y^2)
    g_prism   [1/m]   small linear gradient -> beam steering
    g_lens    [1/m^2] quadratic -> focusing/defocusing
    """
    return n0 + g_prism*X + g_lens*(X**2 + Y**2)

def grin_lens(X, Y, n0=1.5, g_lens=1e3):
    """Pure quadratic index profile acting like a lens: n = n0 + g_lens*(x^2+y^2)."""
    return n0 + g_lens*(X**2 + Y**2)

def hologram_grating(X, Z, n0=1.5, dn=1e-4, period=10e-6, slant_angle=0.0):
    """
    Sinusoidal phase grating (volume hologram) index modulation in a slab 0<=z<=L:
      n(x,z) = n0 + dn * cos(K·r) with |K| = 2π/Λ, optionally slanted by angle in x-z plane.
    X: [nx,] transverse grid; Z: scalar z position or array matching X; slant in radians.
    """
    K = 2.0*np.pi/period
    Kx = K*np.sin(slant_angle)
    Kz = K*np.cos(slant_angle)
    phase = Kx*X + Kz*Z
    return n0 + dn*np.cos(phase)
