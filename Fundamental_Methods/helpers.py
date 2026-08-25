import os
from typing import Dict, Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

def save(state: Dict[str, Any], file_name: str) -> None:
    """
    Saves the full simulation state dictionary to disk.

    Every key in the dictionary is saved automatically.
    Works for numpy arrays and scalars.
    """

    os.makedirs("saved_states", exist_ok=True)
    path = os.path.join("saved_states", f"{file_name}.npz")

    # Save everything in dictionary
    np.savez_compressed(path, **state)

    print(f"[Saved] {path}")


def load(file_name: str) -> Dict[str, Any]:
    path = os.path.join("saved_states", f"{file_name}.npz")
    data = np.load(path, allow_pickle=True)

    return {key: data[key] for key in data.files}


def interp_stretched_grid(R, Z, F, Rn, Zn, kx=1, ky=1, bounds_error=False):
    """
    Interpolate a (possibly complex-valued) field F defined on a structured,
    non-uniformly stretched (R, Z) grid onto arbitrary query points (Rn, Zn).

    Improvements over a plain RectBivariateSpline wrapper:
      - Complex F is supported by interpolating real and imaginary parts
        separately (RectBivariateSpline only accepts real data).
      - Row/column ordering is fixed by sorting, not by assuming the only
        possible disorder is a simple reversal.
      - Out-of-domain query points are clamped to the grid bounds by default
        instead of being silently extrapolated with a high-order polynomial
        (which can produce large, unphysical values). Set bounds_error=True
        to raise instead.
      - The requested spline degree is reduced automatically if the grid is
        too small to support it, instead of raising a cryptic scipy error.

    Parameters
    ----------
    R, Z : 2D arrays, shape (Nz, Nr)
        Meshgrid-style coordinate arrays (as produced by np.meshgrid with
        indexing='xy' or similar) — R varies along axis 1, Z along axis 0.
    F : 2D array, shape (Nz, Nr)
        Field values on the (R, Z) grid. May be real or complex.
    Rn, Zn : array-like, any shape
        Query point coordinates (same shape, broadcastable).
    kx, ky : int
        Spline degree in r and z (1 = linear, 3 = cubic, etc.).
    bounds_error : bool
        If True, raise ValueError when query points fall outside the grid.
        If False (default), clamp query points to the grid bounds.

    Returns
    -------
    out : ndarray, same shape as Rn/Zn
        Interpolated values, complex if F is complex.
    """
    r = np.asarray(R[0, :], dtype=float)
    z = np.asarray(Z[:, 0], dtype=float)
    F = np.asarray(F)

    if F.shape != (z.size, r.size):
        raise ValueError(
            f"F.shape={F.shape} does not match grid shape (Nz,Nr)=({z.size},{r.size})"
        )

    # Sort by value rather than assuming the only disorder is a reversal.
    r_order = np.argsort(r)
    z_order = np.argsort(z)
    r_s = r[r_order]
    z_s = z[z_order]
    F_s = F[np.ix_(z_order, r_order)]

    if r_s[0] == r_s[-1] or z_s[0] == z_s[-1]:
        raise ValueError("Degenerate grid: r or z coordinates are all identical.")

    # Spline degree must be < number of points along that axis.
    kx_eff = int(min(kx, r_s.size - 1))
    ky_eff = int(min(ky, z_s.size - 1))

    Rn_arr = np.asarray(Rn, dtype=float)
    Zn_arr = np.asarray(Zn, dtype=float)
    if Rn_arr.shape != Zn_arr.shape:
        raise ValueError(
            f"Rn.shape={Rn_arr.shape} must match Zn.shape={Zn_arr.shape} "
            f"(they describe paired query-point coordinates; that shape is "
            f"independent of the source grid's shape)."
        )
    out_shape = Rn_arr.shape

    r_flat = Rn_arr.ravel()
    z_flat = Zn_arr.ravel()

    out_of_bounds = (
        (r_flat < r_s[0]) | (r_flat > r_s[-1]) |
        (z_flat < z_s[0]) | (z_flat > z_s[-1])
    )
    if out_of_bounds.any():
        if bounds_error:
            raise ValueError(
                f"{out_of_bounds.sum()} query point(s) fall outside the grid domain "
                f"r∈[{r_s[0]}, {r_s[-1]}], z∈[{z_s[0]}, {z_s[-1]}]."
            )
        r_flat = np.clip(r_flat, r_s[0], r_s[-1])
        z_flat = np.clip(z_flat, z_s[0], z_s[-1])

    if np.iscomplexobj(F_s):
        spl_re = RectBivariateSpline(z_s, r_s, F_s.real, kx=kx_eff, ky=ky_eff)
        spl_im = RectBivariateSpline(z_s, r_s, F_s.imag, kx=kx_eff, ky=ky_eff)
        out = spl_re.ev(z_flat, r_flat) + 1j * spl_im.ev(z_flat, r_flat)
    else:
        spl = RectBivariateSpline(z_s, r_s, F_s, kx=kx_eff, ky=ky_eff)
        out = spl.ev(z_flat, r_flat)

    return out.reshape(out_shape)


class StretchedGridInterpolator:
    """
    Cached interpolator for repeated calls on a fixed (R, Z) grid where only
    the field F changes between calls — e.g. inside an iterative CFD/EM
    solver loop. Precomputes the sort order and validated grid coordinates
    once at construction, instead of redoing that work on every call.

    Usage
    -----
    interp = StretchedGridInterpolator(R, Z, kx=1, ky=1)
    A_new  = interp(F=A_field, Rn=Rn, Zn=Zn)        # call every iteration
    """

    def __init__(self, R, Z, kx=1, ky=1, bounds_error=False):
        r = np.asarray(R[0, :], dtype=float)
        z = np.asarray(Z[:, 0], dtype=float)

        self.r_order = np.argsort(r)
        self.z_order = np.argsort(z)
        self.r_s = r[self.r_order]
        self.z_s = z[self.z_order]

        if self.r_s[0] == self.r_s[-1] or self.z_s[0] == self.z_s[-1]:
            raise ValueError("Degenerate grid: r or z coordinates are all identical.")

        self.Nz, self.Nr = z.size, r.size
        self.kx = int(min(kx, self.r_s.size - 1))
        self.ky = int(min(ky, self.z_s.size - 1))
        self.bounds_error = bounds_error

    def __call__(self, F, Rn, Zn):
        """
        Parameters
        ----------
        F : 2D array, shape (Nz, Nr) — must match the (R, Z) grid this
            interpolator was built with.
        Rn, Zn : array-like, any matching shape
            Query point coordinates. Rn and Zn must have the same shape as
            each other, but that shape has nothing to do with (Nz, Nr) —
            it can be a totally different grid resolution, a flattened
            list of scattered points, a 1D line, etc.

        Returns
        -------
        out : ndarray, same shape as Rn/Zn
        """
        F = np.asarray(F)
        if F.shape != (self.Nz, self.Nr):
            raise ValueError(
                f"F.shape={F.shape} does not match the source grid shape "
                f"(Nz,Nr)=({self.Nz},{self.Nr})"
            )
        F_s = F[np.ix_(self.z_order, self.r_order)]

        Rn_arr = np.asarray(Rn, dtype=float)
        Zn_arr = np.asarray(Zn, dtype=float)
        if Rn_arr.shape != Zn_arr.shape:
            raise ValueError(
                f"Rn.shape={Rn_arr.shape} must match Zn.shape={Zn_arr.shape} "
                f"(they describe paired query-point coordinates; the shape "
                f"itself is independent of the source grid shape)."
            )
        out_shape = Rn_arr.shape

        r_flat = Rn_arr.ravel()
        z_flat = Zn_arr.ravel()

        out_of_bounds = (
            (r_flat < self.r_s[0]) | (r_flat > self.r_s[-1]) |
            (z_flat < self.z_s[0]) | (z_flat > self.z_s[-1])
        )
        if out_of_bounds.any():
            if self.bounds_error:
                raise ValueError(
                    f"{out_of_bounds.sum()} query point(s) fall outside the grid domain "
                    f"r∈[{self.r_s[0]}, {self.r_s[-1]}], z∈[{self.z_s[0]}, {self.z_s[-1]}]."
                )
            r_flat = np.clip(r_flat, self.r_s[0], self.r_s[-1])
            z_flat = np.clip(z_flat, self.z_s[0], self.z_s[-1])

        if np.iscomplexobj(F_s):
            spl_re = RectBivariateSpline(self.z_s, self.r_s, F_s.real, kx=self.kx, ky=self.ky)
            spl_im = RectBivariateSpline(self.z_s, self.r_s, F_s.imag, kx=self.kx, ky=self.ky)
            out = spl_re.ev(z_flat, r_flat) + 1j * spl_im.ev(z_flat, r_flat)
        else:
            spl = RectBivariateSpline(self.z_s, self.r_s, F_s, kx=self.kx, ky=self.ky)
            out = spl.ev(z_flat, r_flat)

        return out.reshape(out_shape)
    