from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# LOADING / PREPARATION
# =============================================================================

def load_saved_state(file_name: str | Path, saved_states_dir: str | Path = "saved_states") -> Dict[str, Any]:
    """
    Load one of the .npz files written by ICPState.as_dict()/helpers.save().

    Examples
    --------
    load_saved_state("161_steady_half")
    load_saved_state("saved_states/161_steady_half.npz")
    """
    p = Path(file_name)

    if p.suffix != ".npz":
        if p.parent == Path("."):
            p = Path(saved_states_dir) / f"{p.name}.npz"
        else:
            p = p.with_suffix(".npz")

    if not p.exists():
        raise FileNotFoundError(f"Saved state not found: {p}")

    with np.load(p, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _scalar(value, default=np.nan):
    if value is None:
        return default
    a = np.asarray(value)
    if a.size == 0:
        return default
    return a.reshape(-1)[0].item()


def prepare_post_state(raw: Dict[str, Any], mu0: float = 4.0 * np.pi * 1e-7) -> Dict[str, Any]:
    """
    Convert the CURRENT ICP solver's saved arrays to one common physical,
    cell-centred (z,r) grid for post-processing.

    Current storage convention
    --------------------------
    T, p, rho, R, Z : scalar grid including ghost cells, shape (Nz+2, Nr+2)
    uz               : axial faces, shape (Nz+1, Nr+2)
    ur               : radial faces, shape (Nz+2, Nr+1)
    A,E,Hr,Hz,P,Fr,Fz: already on physical scalar interior, shape (Nz, Nr)
    volume           : physical cell volumes, shape (Nz, Nr)
    """
    s = dict(raw)

    required = ["R", "Z", "T", "p", "rho", "uz", "ur"]
    missing = [k for k in required if k not in s]
    if missing:
        raise KeyError(f"Saved state is missing required keys: {missing}")

    Rfull = np.asarray(s["R"])
    Zfull = np.asarray(s["Z"])

    # Physical scalar grid.
    s["R_c"] = Rfull[1:-1, 1:-1]
    s["Z_c"] = Zfull[1:-1, 1:-1]
    s["T_c"] = np.asarray(s["T"])[1:-1, 1:-1]
    s["p_c"] = np.asarray(s["p"])[1:-1, 1:-1]
    s["rho_c"] = np.asarray(s["rho"])[1:-1, 1:-1]

    # Staggered -> cell centres.
    uz = np.asarray(s["uz"])
    ur = np.asarray(s["ur"])

    s["uz_c"] = 0.5 * (uz[1:, 1:-1] + uz[:-1, 1:-1])
    s["ur_c"] = 0.5 * (ur[1:-1, 1:] + ur[1:-1, :-1])
    s["U_c"] = np.sqrt(s["uz_c"] ** 2 + s["ur_c"] ** 2)

    physical_shape = s["R_c"].shape

    # Fields already stored on the physical interior grid.
    for key in ["P", "Fr", "Fz", "A", "E", "Hr", "Hz", "div", "volume"]:
        if key in s:
            arr = np.asarray(s[key])
            if arr.shape == physical_shape:
                s[f"{key}_c"] = arr

    # Electromagnetic derived quantities.
    if "Hz_c" in s:
        s["Hz_real"] = np.real(s["Hz_c"])
        s["Hz_imag"] = np.imag(s["Hz_c"])
        s["Bz_c"] = mu0 * s["Hz_c"]
        s["Bz_real"] = np.real(s["Bz_c"])
        s["Bz_imag"] = np.imag(s["Bz_c"])

    if "Hr_c" in s:
        s["Hr_real"] = np.real(s["Hr_c"])
        s["Hr_imag"] = np.imag(s["Hr_c"])
        s["Br_c"] = mu0 * s["Hr_c"]
        s["Br_real"] = np.real(s["Br_c"])
        s["Br_imag"] = np.imag(s["Br_c"])

    if "Hz_c" in s and "Hr_c" in s:
        s["Hmag_c"] = np.sqrt(np.abs(s["Hz_c"]) ** 2 + np.abs(s["Hr_c"]) ** 2)
        s["Bmag_c"] = mu0 * s["Hmag_c"]

    if "E_c" in s:
        s["E_real"] = np.real(s["E_c"])
        s["E_imag"] = np.imag(s["E_c"])
        s["Emag_c"] = np.abs(s["E_c"])

    if "A_c" in s:
        s["A_real"] = np.real(s["A_c"])
        s["A_imag"] = np.imag(s["A_c"])
        s["Amag_c"] = np.abs(s["A_c"])

    if "Fr_c" in s and "Fz_c" in s:
        s["Fmag_c"] = np.sqrt(s["Fr_c"] ** 2 + s["Fz_c"] ** 2)

    return s


# =============================================================================
# POST-PROCESSOR
# =============================================================================

class ICPPost:
    """
    Post-processing helper for the CURRENT RF-ICP solver.

    Internal array convention
    -------------------------
    Arrays are indexed [z, r].

    Plotting convention
    -------------------
    x-axis = radial coordinate r
    y-axis = axial coordinate z

    Notes
    -----
    * Scalars are stripped of ghost cells.
    * uz and ur are converted from the staggered grid to scalar-cell centres.
    * Electromagnetic fields are left on their saved physical interior grid.
    * Geometry is taken from the supplied torch object.
    * Coils are expected in your input.py convention: [z, r].
    """

    FIELD_MAP = {
        # convenient public key : prepared-state key
        "T": "T_c",
        "p": "p_c",
        "rho": "rho_c",
        "uz": "uz_c",
        "ur": "ur_c",
        "U": "U_c",
        "speed": "U_c",
        "P": "P_c",
        "Fr": "Fr_c",
        "Fz": "Fz_c",
        "Fmag": "Fmag_c",
        "div": "div_c",
        "A": "A_c",
        "A_real": "A_real",
        "A_imag": "A_imag",
        "Amag": "Amag_c",
        "E": "E_c",
        "E_real": "E_real",
        "E_imag": "E_imag",
        "Emag": "Emag_c",
        "Hr": "Hr_c",
        "Hz": "Hz_c",
        "Hr_real": "Hr_real",
        "Hr_imag": "Hr_imag",
        "Hz_real": "Hz_real",
        "Hz_imag": "Hz_imag",
        "Hmag": "Hmag_c",
        "Br": "Br_c",
        "Bz": "Bz_c",
        "Br_real": "Br_real",
        "Br_imag": "Br_imag",
        "Bz_real": "Bz_real",
        "Bz_imag": "Bz_imag",
        "Bmag": "Bmag_c",
    }

    FIELD_LABELS = {
        "T": "Temperature [K]",
        "p": "Pressure [Pa]",
        "rho": r"Density [kg m$^{-3}$]",
        "uz": r"Axial velocity $u_z$ [m s$^{-1}$]",
        "ur": r"Radial velocity $u_r$ [m s$^{-1}$]",
        "U": r"Velocity magnitude [m s$^{-1}$]",
        "speed": r"Velocity magnitude [m s$^{-1}$]",
        "P": r"Joule power density [W m$^{-3}$]",
        "Fr": r"Radial Lorentz force [N m$^{-3}$]",
        "Fz": r"Axial Lorentz force [N m$^{-3}$]",
        "Fmag": r"Lorentz-force magnitude [N m$^{-3}$]",
        "div": r"Mass divergence [kg m$^{-3}$ s$^{-1}$]",
        "A_real": r"Re$(A_\theta)$ [Wb m$^{-1}$]",
        "A_imag": r"Im$(A_\theta)$ [Wb m$^{-1}$]",
        "Amag": r"$|A_\theta|$ [Wb m$^{-1}$]",
        "E_real": r"Re$(E_\theta)$ [V m$^{-1}$]",
        "E_imag": r"Im$(E_\theta)$ [V m$^{-1}$]",
        "Emag": r"$|E_\theta|$ [V m$^{-1}$]",
        "Hr_real": r"Re$(H_r)$ [A m$^{-1}$]",
        "Hr_imag": r"Im$(H_r)$ [A m$^{-1}$]",
        "Hz_real": r"Re$(H_z)$ [A m$^{-1}$]",
        "Hz_imag": r"Im$(H_z)$ [A m$^{-1}$]",
        "Hmag": r"$|\mathbf{H}|$ [A m$^{-1}$]",
        "Br_real": r"Re$(B_r)$ [T]",
        "Br_imag": r"Im$(B_r)$ [T]",
        "Bz_real": r"Re$(B_z)$ [T]",
        "Bz_imag": r"Im$(B_z)$ [T]",
        "Bmag": r"$|\mathbf{B}|$ [T]",
    }

    FIELD_TITLES = {
        "T": "Temperature field",
        "p": "Pressure field",
        "rho": "Density field",
        "uz": "Axial velocity field",
        "ur": "Radial velocity field",
        "U": "Velocity-magnitude field",
        "speed": "Velocity-magnitude field",
        "P": "Joule power-density field",
        "Fr": "Radial Lorentz-force field",
        "Fz": "Axial Lorentz-force field",
        "Fmag": "Lorentz-force magnitude",
        "div": "Mass-divergence field",
        "Hmag": "Magnetic-field strength magnitude",
        "Bmag": "Magnetic-flux-density magnitude",
        "Emag": "Electric-field magnitude",
        "Amag": "Magnetic vector-potential magnitude",
    }

    def __init__(
        self,
        state: Dict[str, Any],
        torch,
        coils: Optional[np.ndarray] = None,
        mu0: float = 4.0 * np.pi * 1e-7,
        omega: Optional[float] = None,
        coil_current: Optional[float] = None,
    ):
        self.raw = dict(state)
        self.torch = torch
        self.mu0 = float(mu0)
        self.omega = None if omega is None else float(omega)
        self.coil_current = None if coil_current is None else float(coil_current)

        self.state = prepare_post_state(self.raw, mu0=self.mu0)
        self.R = self.state["R_c"]
        self.Z = self.state["Z_c"]
        self.r_vec = np.asarray(self.R[0, :], dtype=float)
        self.z_vec = np.asarray(self.Z[:, 0], dtype=float)

        if coils is None:
            self.coils = np.empty((0, 2), dtype=float)
        else:
            c = np.asarray(coils, dtype=float)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError("coils must have shape (N,2) with columns [z, r].")
            self.coils = c

        self._validate_shapes()

    @classmethod
    def from_file(
        cls,
        file_name: str | Path,
        torch,
        coils: Optional[np.ndarray] = None,
        mu0: float = 4.0 * np.pi * 1e-7,
        omega: Optional[float] = None,
        coil_current: Optional[float] = None,
        saved_states_dir: str | Path = "saved_states",
    ):
        raw = load_saved_state(file_name, saved_states_dir=saved_states_dir)
        return cls(raw, torch=torch, coils=coils, mu0=mu0,
                   omega=omega, coil_current=coil_current)

    # -------------------------------------------------------------------------
    # BASIC HELPERS
    # -------------------------------------------------------------------------

    def _validate_shapes(self):
        target = self.R.shape
        for key in ["T_c", "p_c", "rho_c", "uz_c", "ur_c", "U_c"]:
            if self.state[key].shape != target:
                raise ValueError(f"{key}.shape={self.state[key].shape}, expected {target}")

    def keys(self):
        return [k for k, sk in self.FIELD_MAP.items() if sk in self.state]

    def field(self, key: str, real_if_complex: bool = False):
        state_key = self.FIELD_MAP.get(key, key)
        if state_key not in self.state:
            raise KeyError(
                f"Field '{key}' is unavailable. Available convenient keys: {self.keys()}"
            )
        F = np.asarray(self.state[state_key])
        if np.iscomplexobj(F):
            if real_if_complex:
                return np.real(F)
            raise ValueError(
                f"Field '{key}' is complex. Plot a component such as '{key}_real'/'{key}_imag' "
                "or a magnitude key."
            )
        return F

    def field_label(self, key: str):
        return self.FIELD_LABELS.get(key, key)

    def field_title(self, key: str):
        return self.FIELD_TITLES.get(key, f"{key} field")

    @property
    def time(self):
        return float(_scalar(self.raw.get("time"), np.nan))

    @property
    def step(self):
        return int(_scalar(self.raw.get("step"), -1))

    @property
    def dt(self):
        return float(_scalar(self.raw.get("dt"), np.nan))

    # -------------------------------------------------------------------------
    # INTEGRALS / QUANTITIES OF INTEREST
    # -------------------------------------------------------------------------

    def absorbed_power(self, kW: bool = True):
        """Integrated plasma Joule power sum(P * volume), matching current ICPState.power_coil()."""
        if "P_c" not in self.state or "volume_c" not in self.state:
            raise KeyError("P and volume must both be present in the saved state.")
        value = float(np.nansum(self.state["P_c"] * self.state["volume_c"]))
        return value / 1000.0 if kW else value

    # Backward-compatible name, but explicit about what it means.
    power_diss = absorbed_power

    def hot_plasma_volume(self, temperature_limit: float = 8000.0):
        if "volume_c" not in self.state:
            raise KeyError("volume is not present in the saved state.")
        return float(np.nansum(self.state["volume_c"][self.state["T_c"] >= temperature_limit]))

    def hotspot(self):
        T = self.state["T_c"]
        i, j = np.unravel_index(np.nanargmax(T), T.shape)
        return {
            "Tmax": float(T[i, j]),
            "z": float(self.Z[i, j]),
            "r": float(self.R[i, j]),
            "i": int(i),
            "j": int(j),
        }

    def field_extrema(self, key: str):
        F = self.field(key)
        imin = np.unravel_index(np.nanargmin(F), F.shape)
        imax = np.unravel_index(np.nanargmax(F), F.shape)
        return {
            "min": float(F[imin]),
            "min_z": float(self.Z[imin]),
            "min_r": float(self.R[imin]),
            "max": float(F[imax]),
            "max_z": float(self.Z[imax]),
            "max_r": float(self.R[imax]),
        }

    def print_summary(self, hot_temperature: float = 8000.0):
        h = self.hotspot()
        print("\n================ ICP POST-PROCESSING SUMMARY ================")
        print(f"step                 : {self.step}")
        print(f"simulation time      : {self.time:.6e} s")
        print(f"last dt              : {self.dt:.6e} s")
        print(f"physical grid        : {self.R.shape[0]} x {self.R.shape[1]} = {self.R.size} cells")
        print(f"Tmax                 : {h['Tmax']:.3f} K")
        print(f"P max                : {np.nanmax(self.state['P']):.6g} W/m^3")
        print(f"Tmax location        : z={h['z']*1e3:.3f} mm, r={h['r']*1e3:.3f} mm")
        print(f"Umax                 : {np.nanmax(self.state['U_c']):.6g} m/s")
        if "P_c" in self.state and "volume_c" in self.state:
            print(f"absorbed Joule power : {self.absorbed_power(kW=True):.6f} kW")
            print(f"V(T>{hot_temperature:g} K)      : {self.hot_plasma_volume(hot_temperature):.6e} m^3")
        print("=============================================================\n")

    # -------------------------------------------------------------------------
    # PROFILE EXTRACTION -- INTERPOLATED, NOT NEAREST-CELL
    # -------------------------------------------------------------------------

    @staticmethod
    def _interp1_complex(x_new, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if np.iscomplexobj(y):
            return np.interp(x_new, x, y.real) + 1j * np.interp(x_new, x, y.imag)
        return np.interp(x_new, x, y)

    def axial_profile(self, r: float, key: str):
        F = np.asarray(self.state[self.FIELD_MAP.get(key, key)])
        if F.shape != self.R.shape:
            raise ValueError(f"Field {key} does not live on the scalar physical grid.")

        r_use = float(np.clip(r, self.r_vec[0], self.r_vec[-1]))
        values = np.empty(len(self.z_vec), dtype=complex if np.iscomplexobj(F) else float)
        for i in range(len(self.z_vec)):
            values[i] = self._interp1_complex(r_use, self.r_vec, F[i, :])
        return self.z_vec.copy(), values, r_use

    def radial_profile(self, z: float, key: str):
        F = np.asarray(self.state[self.FIELD_MAP.get(key, key)])
        if F.shape != self.R.shape:
            raise ValueError(f"Field {key} does not live on the scalar physical grid.")

        z_use = float(np.clip(z, self.z_vec[0], self.z_vec[-1]))
        values = np.empty(len(self.r_vec), dtype=complex if np.iscomplexobj(F) else float)
        for j in range(len(self.r_vec)):
            values[j] = self._interp1_complex(z_use, self.z_vec, F[:, j])
        return self.r_vec.copy(), values, z_use

    # Familiar aliases from your old Post class.
    def vary_axial(self, r, key):
        z, f, _ = self.axial_profile(r, key)
        return z, f

    def vary_radial(self, z, key):
        r, f, _ = self.radial_profile(z, key)
        return r, f

    # -------------------------------------------------------------------------
    # SOLID MASK / GEOMETRY
    # -------------------------------------------------------------------------

    def solid_mask(self, R=None, Z=None):
        R = self.R if R is None else np.asarray(R)
        Z = self.Z if Z is None else np.asarray(Z)
        t = self.torch

        mask = np.zeros_like(R, dtype=bool)

        if all(hasattr(t, x) for x in ["Lr_carrier", "t_carrier", "Lz_carrier"]):
            mask |= (
                (R >= t.Lr_carrier - t.t_carrier)
                & (R <= t.Lr_carrier)
                & (Z <= t.Lz_carrier)
            )

        if all(hasattr(t, x) for x in ["Lr_sheath", "t_sheath", "Lz_sheath"]):
            mask |= (
                (R >= t.Lr_sheath)
                & (R <= t.Lr_sheath + t.t_sheath)
                & (Z <= t.Lz_sheath)
            )

        return mask

    def masked_field(self, key: str):
        F = np.array(self.field(key), dtype=float, copy=True)
        F[self.solid_mask()] = np.nan
        return F

    def _draw_geometry(self, ax, r_scale=1.0, z_scale=1.0,
                       color="white", lw=1.0, fill=False,
                       fill_color="black", alpha=1.0,
                       show_outer_wall=True, zorder=10):
        t = self.torch

        def add_wall(r0, width, z0, height):
            if fill:
                ax.add_patch(Rectangle(
                    (r0 * r_scale, z0 * z_scale),
                    width * r_scale, height * z_scale,
                    facecolor=fill_color, edgecolor="none", alpha=alpha, zorder=zorder,
                ))
            else:
                ax.add_patch(Rectangle(
                    (r0 * r_scale, z0 * z_scale),
                    width * r_scale, height * z_scale,
                    facecolor="none", edgecolor=color, linewidth=lw, zorder=zorder,
                ))

        if all(hasattr(t, x) for x in ["Lr_carrier", "t_carrier", "Lz_carrier"]):
            add_wall(t.Lr_carrier - t.t_carrier, t.t_carrier, 0.0, t.Lz_carrier)

        if all(hasattr(t, x) for x in ["Lr_sheath", "t_sheath", "Lz_sheath"]):
            add_wall(t.Lr_sheath, t.t_sheath, 0.0, t.Lz_sheath)

        if show_outer_wall and hasattr(t, "Lr") and hasattr(t, "Lz"):
            wall_thickness = float(getattr(t, "t_wall", 0.0))
            if wall_thickness > 0:
                add_wall(t.Lr, wall_thickness, 0.0, t.Lz)
            else:
                ax.plot([t.Lr*r_scale, t.Lr*r_scale], [0, t.Lz*z_scale],
                        color=color, lw=lw, zorder=zorder)

    def _draw_coils(self, ax, r_scale=1.0, z_scale=1.0,
                    size=9, color="black", zorder=20):
        if self.coils.size == 0:
            return
        ax.plot(self.coils[:, 1] * r_scale, self.coils[:, 0] * z_scale,
                "o", color=color, markersize=size, zorder=zorder)

    def _limits(self, include_outer_wall=True, include_coils=True, pad=0.003):
        rmin = 0.0 if hasattr(self.torch, "Lr") else float(np.nanmin(self.r_vec))
        rmax = float(getattr(self.torch, "Lr", np.nanmax(self.r_vec)))
        zmin = 0.0
        zmax = float(getattr(self.torch, "Lz", np.nanmax(self.z_vec)))

        if include_outer_wall:
            rmax += float(getattr(self.torch, "t_wall", 0.0))

        if include_coils and self.coils.size:
            rmin = min(rmin, float(np.min(self.coils[:, 1])) - pad)
            rmax = max(rmax, float(np.max(self.coils[:, 1])) + pad)
            zmin = min(zmin, float(np.min(self.coils[:, 0])) - pad)
            zmax = max(zmax, float(np.max(self.coils[:, 0])) + pad)

        return (rmin, rmax), (zmin, zmax)

    # -------------------------------------------------------------------------
    # REGULAR GRID FOR MATPLOTLIB STREAMPLOT
    # -------------------------------------------------------------------------

    def _regular_grid(self, nr=180, nz=400):
        r = np.linspace(self.r_vec.min(), self.r_vec.max(), int(nr))
        z = np.linspace(self.z_vec.min(), self.z_vec.max(), int(nz))
        Zg, Rg = np.meshgrid(z, r, indexing="ij")
        return r, z, Rg, Zg

    def _interpolate_regular(self, F, nr=180, nz=400):
        F = np.asarray(F)
        r, z, Rg, Zg = self._regular_grid(nr=nr, nz=nz)

        # RegularGridInterpolator supports complex values too.
        interp = RegularGridInterpolator(
            (self.z_vec, self.r_vec), F,
            bounds_error=False, fill_value=np.nan,
        )
        pts = np.column_stack([Zg.ravel(), Rg.ravel()])
        Fg = interp(pts).reshape(Zg.shape)
        return r, z, Rg, Zg, Fg

    def regular_velocity(self, nr=180, nz=400, mask_solids=True):
        r, z, Rg, Zg, uz = self._interpolate_regular(self.state["uz_c"], nr, nz)
        _, _, _, _, ur = self._interpolate_regular(self.state["ur_c"], nr, nz)
        if mask_solids:
            mask = self.solid_mask(Rg, Zg)
            uz = np.array(uz, dtype=float, copy=True)
            ur = np.array(ur, dtype=float, copy=True)
            uz[mask] = np.nan
            ur[mask] = np.nan
        return r, z, ur, uz

    # -------------------------------------------------------------------------
    # 1-D PLOTS
    # -------------------------------------------------------------------------

    def plot_axial(self, r, key, figsize=(6.0, 4.2), z_scale=1e3,
                   lw=1.8, label=None, title=False, grid=True):
        z, F, r_used = self.axial_profile(r, key)
        if np.iscomplexobj(F):
            raise ValueError("Choose a real/imaginary/magnitude key for plotting.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(z * z_scale, F, lw=lw, label=label)
        ax.set_xlabel("Axial position, z [mm]" if np.isclose(z_scale, 1e3) else "Axial position, z [m]")
        ax.set_ylabel(self.field_label(key))
        if title:
            ax.set_title(f"{self.field_title(key)} at r = {r_used*1e3:.3f} mm")
        if grid:
            ax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        if label:
            ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

    def plot_radial(self, z, key, figsize=(6.0, 4.2), r_scale=1e3,
                    lw=1.8, label=None, title=False, grid=True):
        r, F, z_used = self.radial_profile(z, key)
        if np.iscomplexobj(F):
            raise ValueError("Choose a real/imaginary/magnitude key for plotting.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(r * r_scale, F, lw=lw, label=label)
        ax.set_xlabel("Radial position, r [mm]" if np.isclose(r_scale, 1e3) else "Radial position, r [m]")
        ax.set_ylabel(self.field_label(key))
        if title:
            ax.set_title(f"{self.field_title(key)} at z = {z_used*1e3:.3f} mm")
        if grid:
            ax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        if label:
            ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------------
    # 2-D FIELD PLOTTING
    # -------------------------------------------------------------------------

    def _plot_field_on_ax(self, ax, key, levels=40, cmap="inferno",
                          vmin=None, vmax=None, log_scale=False,
                          mask_solids=True, r_scale=1e3, z_scale=1e3,
                          show_geometry=True, fill_walls=False, 
                          geometry_color="white", wall_color="black",
                          show_outer_wall=True, show_coils=True, coil_size=9,
                          add_streamlines=False, streamline_density=0.8,
                          streamline_color="white", streamline_lw=0.55,
                          stream_nr=180, stream_nz=400):
        F = self.masked_field(key) if mask_solids else np.asarray(self.field(key))
        X = self.R * r_scale
        Y = self.Z * z_scale

        kwargs = {}
        if log_scale:
            positive = F[np.isfinite(F) & (F > 0)]
            if positive.size == 0:
                raise ValueError(f"Field '{key}' has no positive finite values for log scale.")
            lo = positive.min() if vmin is None else vmin
            hi = positive.max() if vmax is None else vmax
            kwargs["norm"] = LogNorm(vmin=lo, vmax=hi)
        else:
            if vmin is not None:
                kwargs["vmin"] = vmin
            if vmax is not None:
                kwargs["vmax"] = vmax

        m = ax.contourf(X, Y, F, levels=levels, cmap=cmap, **kwargs)

        if add_streamlines:
            rreg, zreg, urreg, uzreg = self.regular_velocity(
                nr=stream_nr, nz=stream_nz, mask_solids=True
            )
            ax.streamplot(
                rreg * r_scale, zreg * z_scale,
                urreg, uzreg,
                density=streamline_density,
                color=streamline_color,
                linewidth=streamline_lw,
                arrowsize=0.7,
                zorder=7,
            )

        if fill_walls:
            self._draw_geometry(ax, r_scale, z_scale, fill=True,
                                fill_color=wall_color, show_outer_wall=show_outer_wall)
        if show_geometry:
            self._draw_geometry(ax, r_scale, z_scale, color=geometry_color,
                                fill=False, show_outer_wall=show_outer_wall)
        if show_coils:
            self._draw_coils(ax, r_scale, z_scale, size=coil_size)

        return m

    def plot_2d(self, key, figsize=(4.8, 7.8), levels=40, cmap="inferno",
                vmin=None, vmax=None, log_scale=False, mask_solids=True,
                colorbar=True, cbar_label=None, title=False,
                r_scale=1e3, z_scale=1e3, ylabel=True, xlabel=True,
                show_geometry=True, fill_walls=False,
                geometry_color="white", wall_color="black",
                show_outer_wall=True, show_coils=True, coil_size=9,
                box_aspect=3.2, fontsize=8):
        fig, ax = plt.subplots(figsize=figsize)
        m = self._plot_field_on_ax(
            ax, key, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
            log_scale=log_scale, mask_solids=mask_solids,
            r_scale=r_scale, z_scale=z_scale,
            show_geometry=show_geometry, fill_walls=fill_walls,
            geometry_color=geometry_color, wall_color=wall_color,
            show_outer_wall=show_outer_wall, show_coils=show_coils,
            coil_size=coil_size,
        )

        if colorbar:
            cb = fig.colorbar(m, ax=ax, pad=0.02, fraction=0.035, aspect=35, )
            cb.set_label(cbar_label or self.field_label(key), fontsize=fontsize)
            cb.ax.tick_params(labelsize=fontsize)

        (rmin, rmax), (zmin, zmax) = self._limits(
            include_outer_wall=show_outer_wall, include_coils=show_coils
        )
        
        ax.set_xlim(rmin * r_scale, rmax * r_scale)
        ax.set_ylim(zmin * z_scale, zmax * z_scale)
        if xlabel:
            ax.set_xlabel("Radial position, r [mm]", fontsize=fontsize)
        if ylabel:
            ax.set_ylabel("Axial position, z [mm]", fontsize=fontsize)
        if title:
            ax.set_title(title or self.field_title(key), fontsize=fontsize)
        ax.set_box_aspect(box_aspect)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        fig.tight_layout()
        return fig, ax

    def plot_2d_with_streamlines(self, key="T", figsize=(4.8, 7.8), levels=40,
                                 cmap="inferno", vmin=None, vmax=None,
                                 mask_solids=True, colorbar=True, cbar_label=None,
                                 title=False, r_scale=1e3, z_scale=1e3, ylabel=True, xlabel=True,
                                 show_geometry=True, fill_walls=False,
                                 geometry_color="white", wall_color="black",
                                 show_outer_wall=True, show_coils=True, coil_size=9,
                                 stream_nr=180, stream_nz=400,
                                 density=0.8, streamline_color="white",
                                 streamline_lw=0.55, box_aspect=3.2, fontsize=8):
        fig, ax = plt.subplots(figsize=figsize)
        m = self._plot_field_on_ax(
            ax, key, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
            mask_solids=mask_solids, r_scale=r_scale, z_scale=z_scale,
            show_geometry=show_geometry, fill_walls=fill_walls,
            geometry_color=geometry_color, wall_color=wall_color,
            show_outer_wall=show_outer_wall, show_coils=show_coils,
            coil_size=coil_size,
            add_streamlines=True, streamline_density=density,
            streamline_color=streamline_color, streamline_lw=streamline_lw,
            stream_nr=stream_nr, stream_nz=stream_nz,
        )

        if colorbar:
            # cb = fig.colorbar(m, ax=ax, pad=0.02)
            cb = fig.colorbar(m, ax=ax, pad=0.02, fraction=0.035, aspect=35, )
            cb.set_label(cbar_label or self.field_label(key), fontsize=fontsize)
            cb.ax.tick_params(labelsize=fontsize)
            
        (rmin, rmax), (zmin, zmax) = self._limits(
            include_outer_wall=show_outer_wall, include_coils=show_coils
        )
        ax.set_xlim(rmin * r_scale, rmax * r_scale)
        ax.set_ylim(zmin * z_scale, zmax * z_scale)
        if xlabel:
            ax.set_xlabel("Radial position, r [mm]", fontsize=fontsize)
        if ylabel:
            ax.set_ylabel("Axial position, z [mm]", fontsize=fontsize)
        if title:
            ax.set_title(title or f"{self.field_title(key)} with flow streamlines")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        fig.tight_layout()
        return fig, ax

    def plot_streamlines(self, figsize=(4.8, 7.8), density=0.8,
                         stream_nr=180, stream_nz=400,
                         r_scale=1e3, z_scale=1e3,
                         show_geometry=True, fill_walls=True,
                         wall_color="0.75", geometry_color="black",
                         show_outer_wall=True, show_coils=True, coil_size=9,
                         box_aspect=3.2, fontsize=8):
        rreg, zreg, urreg, uzreg = self.regular_velocity(
            nr=stream_nr, nz=stream_nz, mask_solids=True
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.streamplot(rreg*r_scale, zreg*z_scale, urreg, uzreg,
                      density=density, linewidth=0.65, arrowsize=0.75)

        if fill_walls:
            self._draw_geometry(ax, r_scale, z_scale, fill=True,
                                fill_color=wall_color, show_outer_wall=show_outer_wall)
        if show_geometry:
            self._draw_geometry(ax, r_scale, z_scale, color=geometry_color,
                                fill=False, show_outer_wall=show_outer_wall)
        if show_coils:
            self._draw_coils(ax, r_scale, z_scale, size=coil_size)

        (rmin, rmax), (zmin, zmax) = self._limits(show_outer_wall, show_coils)
        ax.set_xlim(rmin*r_scale, rmax*r_scale)
        ax.set_ylim(zmin*z_scale, zmax*z_scale)
        ax.set_xlabel("Radial position, r [mm]")
        ax.set_ylabel("Axial position, z [mm]")
        ax.set_title("Flow streamlines")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------------
    # GRID VISUALISATION -- USE THE ACTUAL SAVED STAGGERED FACE COORDINATES
    # -------------------------------------------------------------------------

    def plot_grid(self, figsize=(5.0, 7.2), r_scale=1e3, z_scale=1e3,
                  radial_stride=1, axial_stride=1,
                  grid_color="0.35", grid_lw=0.35, grid_alpha=0.7,
                  show_geometry=True, geometry_color="black",
                  show_coils=True, coil_size=8, box_aspect=3.2):
        fig, ax = plt.subplots(figsize=figsize)

        if all(k in self.raw for k in ["Rur", "Zur", "Ruz", "Zuz"]):
            Rur = np.asarray(self.raw["Rur"])[1:-1, :]
            Zur = np.asarray(self.raw["Zur"])[1:-1, :]
            Ruz = np.asarray(self.raw["Ruz"])[:, 1:-1]
            Zuz = np.asarray(self.raw["Zuz"])[:, 1:-1]

            # Radial cell faces: approximately vertical lines.
            for j in range(0, Rur.shape[1], max(1, radial_stride)):
                ax.plot(Rur[:, j]*r_scale, Zur[:, j]*z_scale,
                        color=grid_color, lw=grid_lw, alpha=grid_alpha)

            # Axial cell faces: approximately horizontal lines.
            for i in range(0, Ruz.shape[0], max(1, axial_stride)):
                ax.plot(Ruz[i, :]*r_scale, Zuz[i, :]*z_scale,
                        color=grid_color, lw=grid_lw, alpha=grid_alpha)
        else:
            # Safe fallback: connect scalar centres.
            for j in range(0, self.R.shape[1], max(1, radial_stride)):
                ax.plot(self.R[:, j]*r_scale, self.Z[:, j]*z_scale,
                        color=grid_color, lw=grid_lw, alpha=grid_alpha)
            for i in range(0, self.R.shape[0], max(1, axial_stride)):
                ax.plot(self.R[i, :]*r_scale, self.Z[i, :]*z_scale,
                        color=grid_color, lw=grid_lw, alpha=grid_alpha)

        if show_geometry:
            self._draw_geometry(ax, r_scale, z_scale, color=geometry_color,
                                fill=False, show_outer_wall=True)
        if show_coils:
            self._draw_coils(ax, r_scale, z_scale, size=coil_size)

        (rmin, rmax), (zmin, zmax) = self._limits(True, show_coils)
        ax.set_xlim(rmin*r_scale, rmax*r_scale)
        ax.set_ylim(zmin*z_scale, zmax*z_scale)
        ax.set_xlabel("Radial position, r [mm]")
        ax.set_ylabel("Axial position, z [mm]")
        ax.set_title("Computational grid")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax

    def plot_sheath_mesh_zoom(
        self,
        figsize=(5.5, 4.5),
        r_scale=1e3,
        z_scale=1e3,
        radial_stride=1,
        axial_stride=1,
        grid_color="0.35",
        grid_lw=0.55,
        grid_alpha=0.8,
        geometry_color="black",
        z_max=0.060,
        r_min=0.014,
        r_max=0.026,
    ):
        """
        Zoomed computational-grid view around the sheath/inlet region.

        Default view:
            z = 0–60 mm
            r = 14–26 mm
        """

        fig, ax = plt.subplots(figsize=figsize)

        if all(k in self.raw for k in ["Rur", "Zur", "Ruz", "Zuz"]):

            Rur = np.asarray(self.raw["Rur"])[1:-1, :]
            Zur = np.asarray(self.raw["Zur"])[1:-1, :]

            Ruz = np.asarray(self.raw["Ruz"])[:, 1:-1]
            Zuz = np.asarray(self.raw["Zuz"])[:, 1:-1]

            # Radial faces
            for j in range(0, Rur.shape[1], max(1, radial_stride)):
                ax.plot(
                    Rur[:, j] * r_scale,
                    Zur[:, j] * z_scale,
                    color=grid_color,
                    lw=grid_lw,
                    alpha=grid_alpha,
                )

            # Axial faces
            for i in range(0, Ruz.shape[0], max(1, axial_stride)):
                ax.plot(
                    Ruz[i, :] * r_scale,
                    Zuz[i, :] * z_scale,
                    color=grid_color,
                    lw=grid_lw,
                    alpha=grid_alpha,
                )

        else:
            # Fallback using scalar grid
            for j in range(0, self.R.shape[1], max(1, radial_stride)):
                ax.plot(
                    self.R[:, j] * r_scale,
                    self.Z[:, j] * z_scale,
                    color=grid_color,
                    lw=grid_lw,
                    alpha=grid_alpha,
                )

            for i in range(0, self.R.shape[0], max(1, axial_stride)):
                ax.plot(
                    self.R[i, :] * r_scale,
                    self.Z[i, :] * z_scale,
                    color=grid_color,
                    lw=grid_lw,
                    alpha=grid_alpha,
                )

        # Torch geometry
        self._draw_geometry(
            ax,
            r_scale=r_scale,
            z_scale=z_scale,
            color=geometry_color,
            lw=1.4,
            fill=False,
            show_outer_wall=True,
        )

        ax.set_xlim(
            r_min * r_scale,
            r_max * r_scale,
        )

        ax.set_ylim(
            0,
            z_max * z_scale,
        )

        ax.set_xlabel("Radial position, r [mm]")
        ax.set_ylabel("Axial position, z [mm]")
        ax.set_title("Mesh refinement near the sheath inlet")

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        fig.tight_layout()

        return fig, ax

    def plot_staggered_grid(self, z=None, r=None, half_window_cells=3,
                            figsize=(6.0, 5.0), scale=1e3):
        required = ["Ruz", "Zuz", "Rur", "Zur"]
        if any(k not in self.raw for k in required):
            raise KeyError(f"Saved state must contain {required} for staggered-grid plotting.")

        if z is None:
            z = float(np.mean(self.z_vec))
        if r is None:
            r = float(np.mean(self.r_vec))

        i0 = int(np.argmin(np.abs(self.z_vec - z)))
        j0 = int(np.argmin(np.abs(self.r_vec - r)))
        n = int(half_window_cells)

        i1, i2 = max(0, i0-n), min(len(self.z_vec)-1, i0+n)
        j1, j2 = max(0, j0-n), min(len(self.r_vec)-1, j0+n)
        zmin, zmax = self.z_vec[i1], self.z_vec[i2]
        rmin, rmax = self.r_vec[j1], self.r_vec[j2]

        fig, ax = plt.subplots(figsize=figsize)

        centre_mask = ((self.R >= rmin) & (self.R <= rmax)
                       & (self.Z >= zmin) & (self.Z <= zmax))
        ax.scatter(self.R[centre_mask]*scale, self.Z[centre_mask]*scale,
                   marker="o", label=r"$p,T,\rho$")

        Ruz, Zuz = np.asarray(self.raw["Ruz"]), np.asarray(self.raw["Zuz"])
        m = ((Ruz >= rmin) & (Ruz <= rmax) & (Zuz >= zmin) & (Zuz <= zmax))
        ax.scatter(Ruz[m]*scale, Zuz[m]*scale, marker="s", label=r"$u_z$")

        Rur, Zur = np.asarray(self.raw["Rur"]), np.asarray(self.raw["Zur"])
        m = ((Rur >= rmin) & (Rur <= rmax) & (Zur >= zmin) & (Zur <= zmax))
        ax.scatter(Rur[m]*scale, Zur[m]*scale, marker="^", label=r"$u_r$")

        ax.set_xlabel("Radial position, r [mm]")
        ax.set_ylabel("Axial position, z [mm]")
        ax.set_title("Staggered variable arrangement")
        ax.legend(frameon=False)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------------
    # MAGNETIC VALIDATION: COIL IN AIR
    # -------------------------------------------------------------------------

    def analytical_Hz_axis(self, z, current: Optional[float] = None):
        """
        Exact on-axis H_z for the set of circular current loops in self.coils.
        Each coil uses its own radius.
        """
        if self.coils.size == 0:
            raise ValueError("No coils were supplied.")
        I = self.coil_current if current is None else float(current)
        if I is None:
            raise ValueError("Supply coil_current when constructing ICPPost or pass current=...")

        z = np.asarray(z, dtype=float)
        Hz = np.zeros_like(z)
        for zc, Rc in self.coils:
            Hz += 0.5 * I * Rc**2 / (Rc**2 + (z-zc)**2)**1.5
        return Hz

    def plot_Hz_axis_validation(self, r_target=0.0, start_at_middle_coil=False,
                                use_abs=False, figsize=(6.2, 4.3), z_scale=1e3,
                                title="Coil-in-air axial magnetic-field validation"):
        if "Hz_real" not in self.state:
            raise KeyError("Hz is not present in this saved state.")
        z, Hnum, r_used = self.axial_profile(r_target, "Hz_real")
        Hana = self.analytical_Hz_axis(z)

        if start_at_middle_coil:
            zmid = np.sort(self.coils[:, 0])[len(self.coils)//2]
            m = z >= zmid
            x = (z[m] - zmid) * z_scale
            Hnum, Hana = Hnum[m], Hana[m]
            xlabel = r"$z-z_{mid}$ [mm]" if np.isclose(z_scale, 1e3) else r"$z-z_{mid}$"
        else:
            x = z * z_scale
            xlabel = "Axial position, z [mm]" if np.isclose(z_scale, 1e3) else "Axial position, z [m]"

        if use_abs:
            Hnum, Hana = np.abs(Hnum), np.abs(Hana)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, Hnum, label=f"Numerical, r={r_used*1e3:.3f} mm")
        ax.plot(x, Hana, "--", label="Analytical circular coils")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$H_z$ [A m$^{-1}$]")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax

    def plot_Hz_axis_relative_difference(self, r_target=0.0,
                                         start_at_middle_coil=False,
                                         figsize=(6.2, 4.3), z_scale=1e3,
                                         eps=1e-30):
        z, Hnum, r_used = self.axial_profile(r_target, "Hz_real")
        Hana = self.analytical_Hz_axis(z)
        RD = 100.0 * np.abs(Hana-Hnum) / (np.abs(Hana)+eps)

        if start_at_middle_coil:
            zmid = np.sort(self.coils[:, 0])[len(self.coils)//2]
            m = z >= zmid
            x, RD = (z[m]-zmid)*z_scale, RD[m]
            xlabel = r"$z-z_{mid}$ [mm]"
        else:
            x = z*z_scale
            xlabel = "Axial position, z [mm]"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, RD)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative difference [%]")
        ax.set_title(f"Coil-in-air field relative difference (r={r_used*1e3:.3f} mm)")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax, RD

    def plot_skin_depth_validation(self, sigma=2500.0, z=None,
                                   figsize=(6.2, 4.3), r_scale=1e3,
                                   normalize_at_wall=True):
        """
        Compare |H| to H0 exp(-x/delta), with x measured inward from r=Lr.
        This is an approximate skin-depth check, not an exact finite-cylinder solution.
        """
        if self.omega is None:
            raise ValueError("omega is required for skin-depth validation.")
        if "Hmag_c" not in self.state:
            raise KeyError("Hr/Hz are required for skin-depth validation.")

        if z is None:
            if self.coils.size:
                z = np.sort(self.coils[:, 0])[len(self.coils)//2]
            else:
                z = float(np.mean(self.z_vec))

        r, H, z_used = self.radial_profile(z, "Hmag")
        r_wall = float(getattr(self.torch, "Lr", np.max(r)))
        m = r <= r_wall
        r, H = r[m], H[m]
        x = r_wall - r
        order = np.argsort(x)
        x, H = x[order], H[order]

        delta = np.sqrt(2.0 / (self.omega * self.mu0 * sigma))
        H0 = H[0] if normalize_at_wall else np.nanmax(H)
        Hana = H0 * np.exp(-x/delta)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x*r_scale, H, label="Numerical |H|")
        ax.plot(x*r_scale, Hana, "--", label=r"$H_0e^{-x/\delta}$")
        ax.axvline(delta*r_scale, ls=":", label=rf"$\delta$={delta*1e3:.2f} mm")
        ax.set_xlabel("Distance inward from torch wall [mm]")
        ax.set_ylabel(r"$|\mathbf{H}|$ [A m$^{-1}$]")
        ax.set_title(rf"Skin-depth check at z={z_used*1e3:.2f} mm, $\sigma$={sigma:g} S/m")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, ax, delta


    # -------------------------------------------------------------------------
    # HORIZONTAL THREE-PANEL COMPARISON WITH ONE SHARED COLORBAR
    # -------------------------------------------------------------------------

    @staticmethod
    def plot_horizontal_comparison(
        posts,
        key="T",
        titles=None,
        panel_labels=("(a)", "(b)", "(c)"),
        figsize=(8.2, 6.0),
        levels=40,
        cmap="inferno",
        vmin=None,
        vmax=None,
        log_scale=False,
        mask_solids=True,
        cbar_label=None,
        r_scale=1e3,
        z_scale=1e3,
        show_geometry=True,
        fill_walls=False,
        geometry_color="white",
        wall_color="black",
        show_outer_wall=True,
        show_coils=True,
        coil_size=9,
        streamlines=False,
        stream_nr=180,
        stream_nz=400,
        density=0.8,
        streamline_color="white",
        streamline_lw=0.55,
        box_aspect=3.2,
        fontsize=8,
        wspace=0.08,
        cbar_width=0.085,
        cbar_pad_fraction=0.18,
        cbar_shrink=0.72,
    ):
        """
        Plot exactly three ICPPost cases side-by-side with one shared colorbar.

        Parameters
        ----------
        posts : list/tuple of 3 ICPPost objects
        key : str
            Field to plot, e.g. "T", "U", "P".
        titles : list/tuple of 3 strings, optional
        figsize : tuple
            Overall figure size.
        box_aspect : float
            Height/width ratio of each panel.
        cbar_width : float
            Relative width of the colorbar column in the GridSpec.
        cbar_shrink : float
            Fraction of the colorbar-axis height to actually use.
        """

        posts = list(posts)
        if len(posts) != 3:
            raise ValueError("plot_horizontal_comparison requires exactly three ICPPost objects.")
        if not all(isinstance(p, ICPPost) for p in posts):
            raise TypeError("Every item in posts must be an ICPPost object.")

        if titles is None:
            titles = [None, None, None]
        if len(titles) != 3:
            raise ValueError("titles must contain exactly three entries.")
        if panel_labels is not None and len(panel_labels) != 3:
            raise ValueError("panel_labels must contain exactly three entries or be None.")

        # ------------------------------------------------------------------
        # Determine common colour range across all three cases
        # ------------------------------------------------------------------
        fields = []
        for post in posts:
            F = post.masked_field(key) if mask_solids else np.asarray(post.field(key), dtype=float)
            fields.append(F)

        finite = [F[np.isfinite(F)] for F in fields if np.any(np.isfinite(F))]
        if not finite:
            raise ValueError(f"Field '{key}' contains no finite values in the supplied cases.")

        all_values = np.concatenate(finite)

        if log_scale:
            positive = all_values[all_values > 0]
            if positive.size == 0:
                raise ValueError(f"Field '{key}' has no positive finite values for log scale.")
            vmin_use = float(np.min(positive)) if vmin is None else float(vmin)
            vmax_use = float(np.max(positive)) if vmax is None else float(vmax)
        else:
            vmin_use = float(np.min(all_values)) if vmin is None else float(vmin)
            vmax_use = float(np.max(all_values)) if vmax is None else float(vmax)

        if not np.isfinite(vmin_use) or not np.isfinite(vmax_use) or vmax_use <= vmin_use:
            raise ValueError("Shared colour limits must satisfy finite vmax > vmin.")

        # Shared contour levels
        if np.ndim(levels) == 0:
            nlev = int(levels)
            if nlev < 2:
                raise ValueError("levels must be at least 2.")
            contour_levels = (
                np.geomspace(vmin_use, vmax_use, nlev + 1)
                if log_scale else
                np.linspace(vmin_use, vmax_use, nlev + 1)
            )
        else:
            contour_levels = np.asarray(levels, dtype=float)

        # ------------------------------------------------------------------
        # Create figure with dedicated colorbar axis
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[1, 1, 1, cbar_width],
            wspace=wspace,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[0, 2], sharex=ax0, sharey=ax0)
        cax = fig.add_subplot(gs[0, 3])

        axs = [ax0, ax1, ax2]

        # ------------------------------------------------------------------
        # Common axis limits across all three cases
        # ------------------------------------------------------------------
        limits = [p._limits(show_outer_wall, show_coils) for p in posts]
        rmin = min(lim[0][0] for lim in limits)
        rmax = max(lim[0][1] for lim in limits)
        zmin = min(lim[1][0] for lim in limits)
        zmax = max(lim[1][1] for lim in limits)

        # ------------------------------------------------------------------
        # Plot each panel
        # ------------------------------------------------------------------
        m = None
        for i, (post, ax) in enumerate(zip(posts, axs)):
            m = post._plot_field_on_ax(
                ax,
                key,
                levels=contour_levels,
                cmap=cmap,
                vmin=vmin_use,
                vmax=vmax_use,
                log_scale=log_scale,
                mask_solids=mask_solids,
                r_scale=r_scale,
                z_scale=z_scale,
                show_geometry=show_geometry,
                fill_walls=fill_walls,
                geometry_color=geometry_color,
                wall_color=wall_color,
                show_outer_wall=show_outer_wall,
                show_coils=show_coils,
                coil_size=coil_size,
                add_streamlines=streamlines,
                streamline_density=density,
                streamline_color=streamline_color,
                streamline_lw=streamline_lw,
                stream_nr=stream_nr,
                stream_nz=stream_nz,
            )

            ax.set_xlim(rmin * r_scale, rmax * r_scale)
            ax.set_ylim(zmin * z_scale, zmax * z_scale)
            ax.set_box_aspect(box_aspect)

            ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
            ax.set_xlabel("Radial position, r [mm]", fontsize=fontsize)

            if i == 0:
                ax.set_ylabel("Axial position, z [mm]", fontsize=fontsize)
            else:
                ax.tick_params(labelleft=False)

            if titles[i]:
                ax.set_title(titles[i], fontsize=fontsize)

            if panel_labels is not None:
                ax.text(
                    -0.14,
                    1.01,
                    panel_labels[i],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    clip_on=False,
                )

        # ------------------------------------------------------------------
        # Shared colourbar in dedicated axis
        # ------------------------------------------------------------------
        cb = fig.colorbar(m, cax=cax)
        cb.set_label(cbar_label or posts[0].field_label(key), fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)

        # Optional shrinking of the colourbar inside its own axis
        if cbar_shrink < 1.0:
            pos = cax.get_position()
            new_h = pos.height * cbar_shrink
            new_y = pos.y0 + 0.5 * (pos.height - new_h)
            new_x = pos.x0 + cbar_pad_fraction * pos.width
            new_w = pos.width * (1.0 - cbar_pad_fraction)
            cax.set_position([new_x, new_y, new_w, new_h])

        return fig, axs, cb



    @staticmethod
    def plot_horizontal_stacked_comparison(
        posts,
        key="T",
        titles=None,
        panel_labels=("(a)", "(b)", "(c)"),
        figsize=(8.4, 6.8),
        levels=40,
        cmap="inferno",
        vmin=None,
        vmax=None,
        log_scale=False,
        mask_solids=True,
        cbar_label=None,
        r_scale=1e3,
        z_scale=1e3,
        show_geometry=True,
        fill_walls=True,
        geometry_color="black",
        wall_color="white",
        geometry_lw=1.0,
        show_outer_wall=True,
        show_coils=True,
        coil_size=9,
        streamlines=True,
        stream_nr=180,
        stream_nz=400,
        density=0.8,
        streamline_color="black",
        streamline_lw=0.55,
        panel_aspect=0.18,
        fontsize=8,
        hspace=0.08,
        cbar_height=0.10,
    ):
        """
        Plot exactly three ICPPost cases stacked vertically, with each torch
        rotated so that the axial direction is horizontal.

        New plotting convention in this figure:
            x-axis = axial coordinate z
            y-axis = radial coordinate r

        Parameters
        ----------
        posts : sequence of 3 ICPPost objects
        key : str
            Field to plot, e.g. "T", "U", "P".
        titles : sequence of 3 str, optional
            Titles for the three panels.
        panel_aspect : float
            Height/width ratio of each panel. Since the torch is horizontal,
            this should usually be < 1, e.g. 0.15–0.25.
        """

        posts = list(posts)
        if len(posts) != 3:
            raise ValueError("plot_horizontal_stacked_comparison requires exactly three ICPPost objects.")

        if titles is None:
            titles = [None, None, None]
        if len(titles) != 3:
            raise ValueError("titles must contain exactly three entries.")
        if panel_labels is not None and len(panel_labels) != 3:
            raise ValueError("panel_labels must contain exactly three entries or be None.")

        # ------------------------------------------------------------------
        # Determine one common colour range from all three cases
        # ------------------------------------------------------------------
        fields = []
        for post in posts:
            F = post.masked_field(key) if mask_solids else np.asarray(post.field(key), dtype=float)
            fields.append(F)

        finite = [F[np.isfinite(F)] for F in fields if np.any(np.isfinite(F))]
        if not finite:
            raise ValueError(f"Field '{key}' contains no finite values in the supplied cases.")

        all_values = np.concatenate(finite)

        if log_scale:
            positive = all_values[all_values > 0]
            if positive.size == 0:
                raise ValueError(f"Field '{key}' has no positive finite values for log scale.")
            vmin_use = float(np.min(positive)) if vmin is None else float(vmin)
            vmax_use = float(np.max(positive)) if vmax is None else float(vmax)
        else:
            vmin_use = float(np.min(all_values)) if vmin is None else float(vmin)
            vmax_use = float(np.max(all_values)) if vmax is None else float(vmax)

        if not np.isfinite(vmin_use) or not np.isfinite(vmax_use) or vmax_use <= vmin_use:
            raise ValueError("Shared colour limits must satisfy finite vmax > vmin.")

        # Shared contour levels
        if np.ndim(levels) == 0:
            nlev = int(levels)
            if nlev < 2:
                raise ValueError("levels must be at least 2.")
            contour_levels = (
                np.geomspace(vmin_use, vmax_use, nlev + 1)
                if log_scale else
                np.linspace(vmin_use, vmax_use, nlev + 1)
            )
        else:
            contour_levels = np.asarray(levels, dtype=float)

        # ------------------------------------------------------------------
        # Figure layout: 3 stacked panels + 1 shared horizontal colourbar
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            4, 1,
            height_ratios=[1.0, 1.0, 1.0, cbar_height],
            hspace=hspace,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
        cax = fig.add_subplot(gs[3, 0])

        axs = [ax0, ax1, ax2]

        # ------------------------------------------------------------------
        # Common limits across all three cases
        # _limits returns ((rmin,rmax),(zmin,zmax)) in the normal orientation
        # For the rotated plots, x=z and y=r.
        # ------------------------------------------------------------------
        limits = [p._limits(show_outer_wall, show_coils) for p in posts]
        rmin = min(lim[0][0] for lim in limits)
        rmax = max(lim[0][1] for lim in limits)
        zmin = min(lim[1][0] for lim in limits)
        zmax = max(lim[1][1] for lim in limits)

        # ------------------------------------------------------------------
        # Local helpers for rotated geometry
        # ------------------------------------------------------------------
        def draw_rotated_geometry(post, ax, fill=False):
            t = post.torch

            def add_wall(z0, zlen, r0, rlen, fill=False):
                if fill:
                    ax.add_patch(Rectangle(
                        (z0 * z_scale, r0 * r_scale),
                        zlen * z_scale,
                        rlen * r_scale,
                        facecolor=wall_color,
                        edgecolor="none",
                        zorder=10,
                    ))
                else:
                    ax.add_patch(Rectangle(
                        (z0 * z_scale, r0 * r_scale),
                        zlen * z_scale,
                        rlen * r_scale,
                        facecolor="none",
                        edgecolor=geometry_color,
                        linewidth=geometry_lw,
                        zorder=11,
                    ))

            # carrier wall
            if all(hasattr(t, x) for x in ["Lr_carrier", "t_carrier", "Lz_carrier"]):
                add_wall(
                    0.0,
                    t.Lz_carrier,
                    t.Lr_carrier - t.t_carrier,
                    t.t_carrier,
                    fill=fill,
                )

            # sheath wall
            if all(hasattr(t, x) for x in ["Lr_sheath", "t_sheath", "Lz_sheath"]):
                add_wall(
                    0.0,
                    t.Lz_sheath,
                    t.Lr_sheath,
                    t.t_sheath,
                    fill=fill,
                )

            # outer wall
            if show_outer_wall and hasattr(t, "Lr") and hasattr(t, "Lz"):
                wall_thickness = float(getattr(t, "t_wall", 0.0))
                if wall_thickness > 0:
                    add_wall(
                        0.0,
                        t.Lz,
                        t.Lr,
                        wall_thickness,
                        fill=fill,
                    )
                elif not fill:
                    ax.plot(
                        [0.0, t.Lz * z_scale],
                        [t.Lr * r_scale, t.Lr * r_scale],
                        color=geometry_color,
                        lw=geometry_lw,
                        zorder=11,
                    )

        def draw_rotated_coils(post, ax):
            if post.coils.size == 0:
                return
            ax.plot(
                post.coils[:, 0] * z_scale,   # x = z
                post.coils[:, 1] * r_scale,   # y = r
                "o",
                color="black",
                markersize=coil_size,
                zorder=20,
            )

        # ------------------------------------------------------------------
        # Plot each case
        # ------------------------------------------------------------------
        m = None
        for i, (post, ax) in enumerate(zip(posts, axs)):
            F = post.masked_field(key) if mask_solids else np.asarray(post.field(key), dtype=float)

            X = post.Z * z_scale   # horizontal axis = axial position z
            Y = post.R * r_scale   # vertical axis   = radial position r

            kwargs = {}

            kwargs["vmin"] = vmin_use
            kwargs["vmax"] = vmax_use

            m = ax.contourf(
                X, Y, F,
                levels=contour_levels,
                cmap=cmap,
                **kwargs,
            )

            if streamlines:
                rreg, zreg, urreg, uzreg = post.regular_velocity(
                    nr=stream_nr,
                    nz=stream_nz,
                    mask_solids=True,
                )
                ax.streamplot(
                    zreg * z_scale,         # x-grid
                    rreg * r_scale,         # y-grid
                    uzreg.T,                # horizontal velocity component
                    urreg.T,                # vertical velocity component
                    density=density,
                    color=streamline_color,
                    linewidth=streamline_lw,
                    arrowsize=0.7,
                    zorder=7,
                )

            if fill_walls:
                draw_rotated_geometry(post, ax, fill=True)
            if show_geometry:
                draw_rotated_geometry(post, ax, fill=False)
            if show_coils:
                draw_rotated_coils(post, ax)

            ax.set_xlim(zmin * z_scale, zmax * z_scale)
            ax.set_ylim(rmin * r_scale, rmax * r_scale)
            ax.set_box_aspect(panel_aspect)

            ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

            if i < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Axial position, z [mm]", fontsize=fontsize)

            ax.set_ylabel("Radial position, r [mm]", fontsize=fontsize)

            if titles[i]:
                ax.set_title(titles[i], fontsize=fontsize)

            if panel_labels is not None:
                ax.text(
                    -0.02,
                    1.03,
                    panel_labels[i],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    clip_on=False,
                )

        # ------------------------------------------------------------------
        # Shared horizontal colourbar
        # ------------------------------------------------------------------
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label or posts[0].field_label(key), fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.10)

        return fig, axs, cb

    @staticmethod
    def plot_horizontal_stacked_comparison(
        posts,
        key="T",
        titles=None,
        panel_labels=("(a)", "(b)", "(c)"),
        figsize=(8.8, 5.8),
        levels=40,
        cmap="inferno",
        vmin=None,
        vmax=None,
        log_scale=False,
        mask_solids=True,
        cbar_label=None,
        r_scale=1e3,
        z_scale=1e3,
        show_geometry=True,
        fill_walls=True,
        geometry_color="black",
        wall_color="white",
        geometry_lw=1.0,
        show_outer_wall=True,
        show_coils=True,
        coil_size=9,
        streamlines=True,
        stream_nr=180,
        stream_nz=400,
        density=0.8,
        streamline_color="black",
        streamline_lw=0.55,
        panel_aspect=0.18,
        fontsize=8,
        hspace=0.10,
        cbar_width=0.045,
        cbar_space=0.08,
        cbar_shrink=0.82,
    ):
        """
        Plot exactly three ICPPost cases stacked vertically with each torch
        rotated so that the axial direction is horizontal.

        Plotting convention
        -------------------
        x-axis = axial position, z
        y-axis = radial position, r

        Parameters
        ----------
        posts : sequence of ICPPost
            Exactly three ICPPost objects.

        key : str
            Field to plot, e.g. "T", "U", "P".

        titles : sequence of str, optional
            Titles for the panels, e.g. ["2 kW", "8 kW", "15 kW"].

        panel_aspect : float
            Height / width ratio of each individual panel.
            Typical range: 0.15–0.25.

        cbar_width : float
            Relative width of the colorbar column.

        cbar_space : float
            Horizontal spacing between plots and colorbar.

        cbar_shrink : float
            Fraction of the full stacked-panel height occupied by the colorbar.
        """

        posts = list(posts)

        if len(posts) != 3:
            raise ValueError(
                "plot_horizontal_stacked_comparison requires exactly "
                "three ICPPost objects."
            )

        if not all(isinstance(p, ICPPost) for p in posts):
            raise TypeError(
                "Every item in posts must be an ICPPost object."
            )

        if titles is None:
            titles = [None, None, None]

        if len(titles) != 3:
            raise ValueError(
                "titles must contain exactly three entries."
            )

        if panel_labels is not None and len(panel_labels) != 3:
            raise ValueError(
                "panel_labels must contain exactly three entries or be None."
            )

        # ==============================================================
        # SHARED FIELD RANGE
        # ==============================================================
        fields = []

        for post in posts:
            if mask_solids:
                F = post.masked_field(key)
            else:
                F = np.asarray(
                    post.field(key),
                    dtype=float,
                )

            fields.append(F)

        finite = [
            F[np.isfinite(F)]
            for F in fields
            if np.any(np.isfinite(F))
        ]

        if not finite:
            raise ValueError(
                f"Field '{key}' contains no finite values."
            )

        all_values = np.concatenate(finite)

        if log_scale:
            positive = all_values[
                all_values > 0
            ]

            if positive.size == 0:
                raise ValueError(
                    f"Field '{key}' has no positive values "
                    "for logarithmic plotting."
                )

            if vmin is None:
                vmin_use = float(
                    np.min(positive)
                )
            else:
                vmin_use = float(vmin)

            if vmax is None:
                vmax_use = float(
                    np.max(positive)
                )
            else:
                vmax_use = float(vmax)

        else:
            if vmin is None:
                vmin_use = float(
                    np.min(all_values)
                )
            else:
                vmin_use = float(vmin)

            if vmax is None:
                vmax_use = float(
                    np.max(all_values)
                )
            else:
                vmax_use = float(vmax)

        if (
            not np.isfinite(vmin_use)
            or not np.isfinite(vmax_use)
            or vmax_use <= vmin_use
        ):
            raise ValueError(
                "Shared colour limits must satisfy "
                "finite vmax > vmin."
            )

        # ==============================================================
        # SHARED CONTOUR LEVELS
        # ==============================================================
        if np.ndim(levels) == 0:

            nlev = int(levels)

            if nlev < 2:
                raise ValueError(
                    "levels must be at least 2."
                )

            if log_scale:
                contour_levels = np.geomspace(
                    vmin_use,
                    vmax_use,
                    nlev + 1,
                )

            else:
                contour_levels = np.linspace(
                    vmin_use,
                    vmax_use,
                    nlev + 1,
                )

        else:
            contour_levels = np.asarray(
                levels,
                dtype=float,
            )

        # ==============================================================
        # FIGURE LAYOUT
        # ==============================================================
        fig = plt.figure(
            figsize=figsize,
        )

        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[
                1.0,
                cbar_width,
            ],
            height_ratios=[
                1.0,
                1.0,
                1.0,
            ],
            hspace=hspace,
            wspace=cbar_space,
        )

        ax0 = fig.add_subplot(
            gs[0, 0],
        )

        ax1 = fig.add_subplot(
            gs[1, 0],
            sharex=ax0,
            sharey=ax0,
        )

        ax2 = fig.add_subplot(
            gs[2, 0],
            sharex=ax0,
            sharey=ax0,
        )

        cax = fig.add_subplot(
            gs[:, 1],
        )

        axs = [
            ax0,
            ax1,
            ax2,
        ]

        # ==============================================================
        # COMMON PHYSICAL LIMITS
        # ==============================================================
        limits = [
            p._limits(
                show_outer_wall,
                show_coils,
            )
            for p in posts
        ]

        rmin = min(
            lim[0][0]
            for lim in limits
        )

        rmax = max(
            lim[0][1]
            for lim in limits
        )

        zmin = min(
            lim[1][0]
            for lim in limits
        )

        zmax = max(
            lim[1][1]
            for lim in limits
        )

        # ==============================================================
        # ROTATED GEOMETRY
        # ==============================================================
        def draw_rotated_geometry(
            post,
            ax,
            fill=False,
        ):
            t = post.torch

            def add_wall(
                z0,
                zlen,
                r0,
                rlen,
                fill=False,
            ):
                if fill:

                    ax.add_patch(
                        Rectangle(
                            (
                                z0 * z_scale,
                                r0 * r_scale,
                            ),
                            zlen * z_scale,
                            rlen * r_scale,
                            facecolor=wall_color,
                            edgecolor="none",
                            zorder=10,
                        )
                    )

                else:

                    ax.add_patch(
                        Rectangle(
                            (
                                z0 * z_scale,
                                r0 * r_scale,
                            ),
                            zlen * z_scale,
                            rlen * r_scale,
                            facecolor="none",
                            edgecolor=geometry_color,
                            linewidth=geometry_lw,
                            zorder=11,
                        )
                    )

            # ----------------------------------------------------------
            # Carrier tube
            # ----------------------------------------------------------
            if all(
                hasattr(t, x)
                for x in [
                    "Lr_carrier",
                    "t_carrier",
                    "Lz_carrier",
                ]
            ):
                add_wall(
                    0.0,
                    t.Lz_carrier,
                    t.Lr_carrier
                    - t.t_carrier,
                    t.t_carrier,
                    fill=fill,
                )

            # ----------------------------------------------------------
            # Sheath tube
            # ----------------------------------------------------------
            if all(
                hasattr(t, x)
                for x in [
                    "Lr_sheath",
                    "t_sheath",
                    "Lz_sheath",
                ]
            ):
                add_wall(
                    0.0,
                    t.Lz_sheath,
                    t.Lr_sheath,
                    t.t_sheath,
                    fill=fill,
                )

            # ----------------------------------------------------------
            # Outer torch wall
            # ----------------------------------------------------------
            if (
                show_outer_wall
                and hasattr(t, "Lr")
                and hasattr(t, "Lz")
            ):
                wall_thickness = float(
                    getattr(
                        t,
                        "t_wall",
                        0.0,
                    )
                )

                if wall_thickness > 0:

                    add_wall(
                        0.0,
                        t.Lz,
                        t.Lr,
                        wall_thickness,
                        fill=fill,
                    )

                elif not fill:

                    ax.plot(
                        [
                            0.0,
                            t.Lz * z_scale,
                        ],
                        [
                            t.Lr * r_scale,
                            t.Lr * r_scale,
                        ],
                        color=geometry_color,
                        lw=geometry_lw,
                        zorder=11,
                    )

        # ==============================================================
        # ROTATED COILS
        # ==============================================================
        def draw_rotated_coils(
            post,
            ax,
        ):
            if post.coils.size == 0:
                return

            ax.plot(
                post.coils[:, 0]
                * z_scale,
                post.coils[:, 1]
                * r_scale,
                "o",
                color="black",
                markersize=coil_size,
                zorder=20,
            )

        # ==============================================================
        # PLOT THREE CASES
        # ==============================================================
        m = None

        for i, (
            post,
            ax,
        ) in enumerate(
            zip(
                posts,
                axs,
            )
        ):

            if mask_solids:
                F = post.masked_field(
                    key
                )

            else:
                F = np.asarray(
                    post.field(key),
                    dtype=float,
                )

            # ----------------------------------------------------------
            # Rotate plotting convention
            #
            # Normal:
            # x = r
            # y = z
            #
            # Here:
            # x = z
            # y = r
            # ----------------------------------------------------------
            X = (
                post.Z
                * z_scale
            )

            Y = (
                post.R
                * r_scale
            )

            kwargs = {}

            if log_scale:

                kwargs["norm"] = LogNorm(
                    vmin=vmin_use,
                    vmax=vmax_use,
                )

            else:

                kwargs["vmin"] = vmin_use
                kwargs["vmax"] = vmax_use

            m = ax.contourf(
                X,
                Y,
                F,
                levels=contour_levels,
                cmap=cmap,
                **kwargs,
            )

            # ==========================================================
            # STREAMLINES
            # ==========================================================
            if streamlines:

                (
                    rreg,
                    zreg,
                    urreg,
                    uzreg,
                ) = post.regular_velocity(
                    nr=stream_nr,
                    nz=stream_nz,
                    mask_solids=True,
                )

                ax.streamplot(
                    zreg * z_scale,
                    rreg * r_scale,
                    uzreg.T,
                    urreg.T,
                    density=density,
                    color=streamline_color,
                    linewidth=streamline_lw,
                    arrowsize=0.7,
                    zorder=7,
                )

            # ==========================================================
            # GEOMETRY
            # ==========================================================
            if fill_walls:
                draw_rotated_geometry(
                    post,
                    ax,
                    fill=True,
                )

            if show_geometry:
                draw_rotated_geometry(
                    post,
                    ax,
                    fill=False,
                )

            if show_coils:
                draw_rotated_coils(
                    post,
                    ax,
                )

            # ==========================================================
            # AXIS FORMAT
            # ==========================================================
            ax.set_xlim(
                zmin * z_scale,
                zmax * z_scale,
            )

            ax.set_ylim(
                rmin * r_scale,
                rmax * r_scale,
            )

            ax.set_box_aspect(
                panel_aspect
            )

            ax.tick_params(
                direction="in",
                top=True,
                right=True,
                labelsize=fontsize,
            )

            ax.set_ylabel(
                "r [mm]",
                fontsize=fontsize,
            )

            # Only bottom panel shows x labels
            if i < 2:

                ax.tick_params(
                    labelbottom=False,
                )

            else:

                ax.set_xlabel(
                    "z [mm]",
                    fontsize=fontsize,
                )

            # Panel title
            if titles[i]:

                ax.set_title(
                    titles[i],
                    fontsize=fontsize,
                )

            # Panel label
            if panel_labels is not None:

                ax.text(
                    -0.02,
                    1.03,
                    panel_labels[i],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    clip_on=False,
                )

        # ==============================================================
        # SHARED VERTICAL COLORBAR
        # ==============================================================
        cb = fig.colorbar(
            m,
            cax=cax,
            orientation="vertical",
        )

        cb.set_label(
            cbar_label
            or posts[0].field_label(key),
            fontsize=fontsize,
        )

        cb.ax.tick_params(
            labelsize=fontsize,
        )

        # --------------------------------------------------------------
        # Shorten colourbar while keeping it vertically centred
        # --------------------------------------------------------------
        if cbar_shrink < 1.0:

            pos = cax.get_position()

            new_height = (
                pos.height
                * cbar_shrink
            )

            new_y = (
                pos.y0
                + 0.5
                * (
                    pos.height
                    - new_height
                )
            )

            cax.set_position(
                [
                    pos.x0,
                    new_y,
                    pos.width,
                    new_height,
                ]
            )

        return fig, axs, cb

    # -------------------------------------------------------------------------
    # REAL MULTI-PANEL PUBLICATION FIGURE (same axes, no nested figures)
    # -------------------------------------------------------------------------

    def plot_publication_figure(self, figsize=(9.2, 9.0), levels=35,
                                stream_density=0.75, show_coils=True):
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        specs = [
            (axs[0,0], "T", "inferno", "(a) Temperature", False),
            (axs[0,1], "P", "magma", "(b) Joule power density", False),
            (axs[1,0], "U", "viridis", "(c) Velocity magnitude", False),
            (axs[1,1], "T", "inferno", "(d) Temperature with streamlines", True),
        ]

        for ax, key, cmap, title, streams in specs:
            if self.FIELD_MAP.get(key, key) not in self.state:
                ax.set_visible(False)
                continue
            m = self._plot_field_on_ax(
                ax, key, levels=levels, cmap=cmap,
                mask_solids=True, r_scale=1e3, z_scale=1e3,
                show_geometry=True, fill_walls=False,
                geometry_color="white", show_outer_wall=True,
                show_coils=show_coils,
                add_streamlines=streams,
                streamline_density=stream_density,
            )
            cb = fig.colorbar(m, ax=ax, pad=0.02)
            cb.set_label(self.field_label(key))
            (rmin, rmax), (zmin, zmax) = self._limits(True, show_coils)
            ax.set_xlim(rmin*1e3, rmax*1e3)
            ax.set_ylim(zmin*1e3, zmax*1e3)
            ax.set_xlabel("r [mm]")
            ax.set_ylabel("z [mm]")
            ax.set_title(title)
            ax.set_box_aspect(3.0)
            ax.tick_params(direction="in", top=True, right=True)

        fig.tight_layout()
        return fig, axs


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
#




