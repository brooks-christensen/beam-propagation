# beamprop

A compact, well‑commented Python package for reproducing several optics simulations and figures:

- (a) GRIN prism/lens index map + beam propagation
- (b) GRIN lensing case
- (c) High‑intensity Kerr self‑action
- (d) 1D bright soliton of the focusing nonlinear Schrödinger equation (NLS)
- (e) Angled Gaussian with insufficient vs sufficient absorber width
- (f) Fourier spectrum of the field after a thick (volume) hologram; angles that minimize/maximize coupling between 0th and 1st order

> Physics core uses split‑step Fourier (SSFM) BPM for the paraxial Helmholtz equation and the normalized NLS (1D).

## Install (editable)

```bash
cd beamprop_pkg
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Quick start

```bash
beamprop-figs a --out out/a.png
beamprop-figs b --out out/b.png
beamprop-figs c --out out/c.png
beamprop-figs d --out out/d.png
beamprop-figs e --out out/e.png
beamprop-figs f --out out/f.png
```

You can also import the library and build your own setups:

```python
from beamprop import grids, fields, propagators, index_profiles, absorbers
```

## Notes

- The BPM assumes the **paraxial** approximation with slowly varying envelope A(x,y,z).
- Absorbing boundary masks are applied each step to suppress wrap‑around in FFT propagation.
- The hologram module includes a simple Kogelnik‑style efficiency sweep to locate min/max angles numerically.
