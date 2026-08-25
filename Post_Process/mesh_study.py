"""Journal-quality mesh-independence and operating-condition analysis.

Designed for the RF-ICP torch saved-state format used by this project.
The script produces publication-ready PDF + 600 dpi PNG figures and CSV tables.

Run from the project root so that ``saved_states/`` is available.
"""

from __future__ import annotations

from pathlib import Path
from itertools import cycle
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTPUT_DIR = Path("journal_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI_RASTER = 600
SHOW_FIGURES = True
SAVE_SVG = False

# Selected powers for profile comparisons. All cases are used in scalar trends.
PROFILE_POWERS = (2, 5, 8, 11, 15)

# Profile locations used consistently throughout the paper.
AXIAL_PROFILE_R = 0.0       # m; first physical radial cell approximates the axis
RADIAL_PROFILE_Z = 0.102    # m

# Optional derived metric.
HOT_TEMPERATURE = 8000.0    # K

# Current-power calibration used by the solver setup.
POWER_CASES = {
    120.0: 2,
    134.0: 3,
    147.0: 4,
    161.0: 5,
    173.0: 6,
    183.0: 7,
    194.0: 8,
    204.0: 9,
    214.0: 10,
    223.0: 11,
    231.5: 12,
    240.0: 13,
    250.0: 14,
    260.0: 15,
}

POWER_FILES = {
    120.0: "saved_states/120_steady.npz",
    134.0: "saved_states/134_steady.npz",
    147.0: "saved_states/147_steady.npz",
    161.0: "saved_states/161_steady.npz",
    173.0: "saved_states/173_steady.npz",
    183.0: "saved_states/183_steady.npz",
    194.0: "saved_states/194_steady.npz",
    204.0: "saved_states/204_steady.npz",
    214.0: "saved_states/214_steady.npz",
    223.0: "saved_states/223_steady.npz",
    231.5: "saved_states/231_5_steady.npz",
    240.0: "saved_states/240_steady.npz",
    250.0: "saved_states/250_steady.npz",
    260.0: "saved_states/260_steady.npz",
}

# Full mesh sequence for the 161 A / approximately 5 kW mesh study.
MESH_FILES = {
    "Coarse": "saved_states/161_steady_quatre.npz",
    "Base": "saved_states/161_steady_half.npz",
    "Fine": "saved_states/161_steady.npz",
    # "Double": "saved_states/161_steady_double.npz",
    # "Triple": "saved_states/161_steady_triple.npz",
}

# Current solver configuration uses the half mesh as the production mesh.
PRODUCTION_MESH = "Base"

# Solver clipping limits, used only to warn about potentially artificial peaks.
UZ_CLIP_LIMIT = 100.0  # m/s
UR_CLIP_LIMIT = 6.0   # m/s
CLIP_WARNING_FRACTION = 1e-4


# =============================================================================
# JOURNAL FIGURE STYLE
# =============================================================================

def set_journal_style() -> None:
    """Apply a compact, journal-friendly Matplotlib style."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI_RASTER,
        "font.family": "serif",
        "font.size": 9.0,
        "mathtext.fontset": "stix",
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.fontsize": 7.6,
        "legend.frameon": False,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    """Consistent finishing touches for every publication axis."""
    ax.minorticks_on()
    ax.tick_params(which="both", width=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, which="major", linewidth=0.35, alpha=0.22)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save vector PDF and high-resolution raster PNG."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=DPI_RASTER, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def line_styles(n: int):
    """Colour + linestyle + marker combinations that survive grayscale printing."""
    colours = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, max(n, 2)))
    linestyles = cycle(["-", "--", "-.", ":", (0, (5, 1.5))])
    markers = cycle(["o", "s", "^", "D", "v"])
    return [(colours[i], next(linestyles), next(markers)) for i in range(n)]


# =============================================================================
# SAVED-STATE HANDLING
# =============================================================================

def load_state(filename: str | Path) -> dict[str, np.ndarray]:
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"State file not found: {filename}")

    with np.load(filename, allow_pickle=True) as data:
        state = {key: np.asarray(data[key]) for key in data.files}

    return prepare_state(state)


def prepare_state(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create consistently cell-centred physical fields from a saved state."""
    required = ("T", "p", "rho", "R", "Z", "uz", "ur")
    missing = [key for key in required if key not in s]
    if missing:
        raise KeyError(f"Saved state is missing required arrays: {missing}")

    s["T_c"] = s["T"][1:-1, 1:-1]
    s["p_c"] = s["p"][1:-1, 1:-1]
    s["rho_c"] = s["rho"][1:-1, 1:-1]
    s["R_c"] = s["R"][1:-1, 1:-1]
    s["Z_c"] = s["Z"][1:-1, 1:-1]

    # Staggered velocities -> scalar cell centres.
    s["uz_c"] = 0.5 * (s["uz"][1:, 1:-1] + s["uz"][:-1, 1:-1])
    s["ur_c"] = 0.5 * (s["ur"][1:-1, 1:] + s["ur"][1:-1, :-1])
    s["U_c"] = np.hypot(s["uz_c"], s["ur_c"])

    target_shape = s["T_c"].shape
    for key in ("uz_c", "ur_c", "U_c", "R_c", "Z_c"):
        if s[key].shape != target_shape:
            raise ValueError(
                f"Cell-centred shape mismatch: T_c{target_shape}, {key}{s[key].shape}"
            )

    # Electromagnetic quantities are already stored on the physical scalar grid.
    for key in ("P", "Fr", "Fz", "E", "A", "Hr", "Hz"):
        if key in s:
            s[f"{key}_c"] = s[key]

    if "Hr" in s and "Hz" in s:
        s["Hmag_c"] = np.hypot(np.abs(s["Hr"]), np.abs(s["Hz"]))

    return s


def coordinate_axes(state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return structured 1D z and r coordinates of scalar cell centres."""
    z = np.asarray(state["Z_c"][:, 0], dtype=float)
    r = np.asarray(state["R_c"][0, :], dtype=float)

    if np.any(np.diff(z) <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("Expected monotonically increasing structured z and r coordinates.")

    return z, r


def axial_profile(
    state: dict[str, np.ndarray],
    field: np.ndarray,
    r_target: float = AXIAL_PROFILE_R,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a field versus z at a specified radial location."""
    z, r = coordinate_axes(state)

    if r_target <= r[0]:
        return z, np.asarray(field[:, 0]), float(r[0])
    if r_target >= r[-1]:
        return z, np.asarray(field[:, -1]), float(r[-1])

    out = np.array([
        np.interp(r_target, r, np.real(row)) for row in field
    ], dtype=float)
    return z, out, float(r_target)


def radial_profile(
    state: dict[str, np.ndarray],
    field: np.ndarray,
    z_target: float = RADIAL_PROFILE_Z,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a field versus r at a specified axial location."""
    z, r = coordinate_axes(state)

    if z_target <= z[0]:
        return r, np.asarray(field[0, :]), float(z[0])
    if z_target >= z[-1]:
        return r, np.asarray(field[-1, :]), float(z[-1])

    out = np.array([
        np.interp(z_target, z, np.real(field[:, j]))
        for j in range(len(r))
    ], dtype=float)
    return r, out, float(z_target)


def profile_l2_error_percent(
    x: np.ndarray,
    y: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
) -> float:
    """Relative L2 error after interpolating the candidate onto the reference grid."""
    mask = (x_ref >= np.min(x)) & (x_ref <= np.max(x))
    if np.count_nonzero(mask) < 2:
        return np.nan

    xr = x_ref[mask]
    yr = y_ref[mask]
    yi = np.interp(xr, x, y)
    denominator = np.linalg.norm(yr)
    if denominator <= 1e-30:
        return np.nan
    return 100.0 * np.linalg.norm(yi - yr) / denominator


def integrated_joule_power(state: dict[str, np.ndarray]) -> float:
    """Integrate volumetric Joule heating over axisymmetric cell volumes [W]."""
    if "P_c" not in state or "volume" not in state:
        return np.nan

    if state["P_c"].shape != state["volume"].shape:
        warnings.warn(
            f"Cannot integrate Joule power: P{state['P_c'].shape} and "
            f"volume{state['volume'].shape} differ."
        )
        return np.nan

    return float(np.nansum(np.real(state["P_c"]) * state["volume"]))


def hot_plasma_volume(state: dict[str, np.ndarray], threshold: float = HOT_TEMPERATURE) -> float:
    if "volume" not in state or state["volume"].shape != state["T_c"].shape:
        return np.nan
    return float(np.nansum(state["volume"][state["T_c"] >= threshold]))


def velocity_clipping_fraction(state: dict[str, np.ndarray]) -> tuple[float, float]:
    """Fraction of raw staggered velocities close to solver clipping limits."""
    uz = np.asarray(state["uz"], dtype=float)
    ur = np.asarray(state["ur"], dtype=float)
    fz = float(np.mean(np.abs(uz) >= 0.995 * UZ_CLIP_LIMIT))
    fr = float(np.mean(np.abs(ur) >= 0.995 * UR_CLIP_LIMIT))
    return fz, fr


# =============================================================================
# TABLES AND DIAGNOSTICS
# =============================================================================

def build_power_table() -> pd.DataFrame:
    rows = []

    for current, target_power in POWER_CASES.items():
        filename = POWER_FILES[current]
        try:
            s = load_state(filename)
        except Exception as exc:
            warnings.warn(f"Skipping {current:g} A / {target_power:g} kW: {exc}")
            continue

        T = s["T_c"]
        U = s["U_c"]
        iT, jT = np.unravel_index(np.nanargmax(T), T.shape)
        iu, ju = np.unravel_index(np.nanargmax(U), U.shape)
        q_joule_kw = integrated_joule_power(s) / 1000.0
        f_uz, f_ur = velocity_clipping_fraction(s)

        rows.append({
            "Current (A)": current,
            "Target power (kW)": target_power,
            "Integrated Joule power (kW)": q_joule_kw,
            "Tmax (K)": float(np.nanmax(T)),
            "Umax (m/s)": float(np.nanmax(U)),
            "Tmax z (mm)": float(s["Z_c"][iT, jT] * 1000.0),
            "Tmax r (mm)": float(s["R_c"][iT, jT] * 1000.0),
            "Umax z (mm)": float(s["Z_c"][iu, ju] * 1000.0),
            "Umax r (mm)": float(s["R_c"][iu, ju] * 1000.0),
            f"Volume T>{HOT_TEMPERATURE:.0f} K (m3)": hot_plasma_volume(s),
            "uz clipped fraction": f_uz,
            "ur clipped fraction": f_ur,
        })

    table = pd.DataFrame(rows).sort_values("Current (A)").reset_index(drop=True)
    if table.empty:
        return table

    # Use directly integrated absorbed power where available; otherwise fall back
    # to the calibrated target power so profile plotting still works.
    q = table["Integrated Joule power (kW)"].to_numpy(float)
    target = table["Target power (kW)"].to_numpy(float)
    table["Power used for plots (kW)"] = np.where(np.isfinite(q) & (q > 0), q, target)

    return table


def build_mesh_table() -> pd.DataFrame:
    states: dict[str, dict[str, np.ndarray]] = {}
    for name, filename in MESH_FILES.items():
        try:
            states[name] = load_state(filename)
        except Exception as exc:
            warnings.warn(f"Skipping mesh {name}: {exc}")

    if not states:
        return pd.DataFrame()

    # Finest available mesh = largest number of physical scalar cells.
    finest_name = max(states, key=lambda n: states[n]["T_c"].size)
    ref = states[finest_name]

    zT_ref, Tz_ref, _ = axial_profile(ref, ref["T_c"])
    rT_ref, Tr_ref, _ = radial_profile(ref, ref["T_c"])
    zU_ref, Uz_ref, _ = axial_profile(ref, ref["uz_c"])
    rU_ref, Ur_ref, _ = radial_profile(ref, ref["uz_c"])

    Tmax_ref = float(np.nanmax(ref["T_c"]))
    Umax_ref = float(np.nanmax(ref["U_c"]))

    rows = []
    for name, s in states.items():
        nz, nr = s["T_c"].shape
        zT, Tz, _ = axial_profile(s, s["T_c"])
        rT, Tr, _ = radial_profile(s, s["T_c"])
        zU, Uz, _ = axial_profile(s, s["uz_c"])
        rU, Ur, _ = radial_profile(s, s["uz_c"])

        Tmax = float(np.nanmax(s["T_c"]))
        Umax = float(np.nanmax(s["U_c"]))

        rows.append({
            "Mesh": name,
            "Nz": nz,
            "Nr": nr,
            "Cells": int(nz * nr),
            "Production mesh": name == PRODUCTION_MESH,
            "Tmax (K)": Tmax,
            "Tmax error vs finest (%)": 100.0 * abs(Tmax - Tmax_ref) / abs(Tmax_ref),
            "Umax (m/s)": Umax,
            "Umax error vs finest (%)": 100.0 * abs(Umax - Umax_ref) / max(abs(Umax_ref), 1e-30),
            "Axial T L2 error (%)": profile_l2_error_percent(zT, Tz, zT_ref, Tz_ref),
            "Radial T L2 error (%)": profile_l2_error_percent(rT, Tr, rT_ref, Tr_ref),
            "Axial uz L2 error (%)": profile_l2_error_percent(zU, Uz, zU_ref, Uz_ref),
            "Radial uz L2 error (%)": profile_l2_error_percent(rU, Ur, rU_ref, Ur_ref),
            "Reference mesh": finest_name,
        })

    return pd.DataFrame(rows).sort_values("Cells").reset_index(drop=True)


def print_diagnostics(power_table: pd.DataFrame, mesh_table: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("JOURNAL ANALYSIS DIAGNOSTICS")
    print("=" * 78)

    if not mesh_table.empty:
        print("\nMesh-independence summary")
        display_cols = [
            "Mesh", "Nz", "Nr", "Cells", "Tmax (K)",
            "Tmax error vs finest (%)", "Umax (m/s)",
            "Umax error vs finest (%)",
        ]
        print(mesh_table[display_cols].round(4).to_string(index=False))

    if not power_table.empty:
        print("\nOperating-condition summary")
        display_cols = [
            "Current (A)", "Target power (kW)", "Integrated Joule power (kW)",
            "Tmax (K)", "Umax (m/s)",
        ]
        print(power_table[display_cols].round(4).to_string(index=False))

        clipped = power_table[
            (power_table["uz clipped fraction"] > CLIP_WARNING_FRACTION)
            | (power_table["ur clipped fraction"] > CLIP_WARNING_FRACTION)
        ]
        if not clipped.empty:
            print("\nWARNING: one or more cases contain velocities close to solver clipping limits.")
            print("Do not interpret Umax as a physical trend until those cases are checked:")
            print(
                clipped[["Current (A)", "Target power (kW)",
                         "uz clipped fraction", "ur clipped fraction"]]
                .to_string(index=False)
            )


# =============================================================================
# MESH-INDEPENDENCE FIGURES
# =============================================================================

def _load_available_meshes() -> list[tuple[str, dict[str, np.ndarray]]]:
    out = []
    for name, filename in MESH_FILES.items():
        try:
            out.append((name, load_state(filename)))
        except Exception as exc:
            warnings.warn(f"Skipping mesh {name}: {exc}")
    out.sort(key=lambda pair: pair[1]["T_c"].size)
    return out


def plot_mesh_convergence(mesh_table: pd.DataFrame) -> None:
    if mesh_table.empty:
        return

    x = mesh_table["Cells"].to_numpy(float)
    styles = line_styles(2)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))

    ax = axes[0]
    c, ls, mk = styles[0]
    ax.plot(x, mesh_table["Tmax (K)"], color=c, linestyle=ls, marker=mk)
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel(r"Maximum temperature, $T_{\max}$ (K)")
    style_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    c, ls, mk = styles[1]
    ax.plot(x, mesh_table["Umax (m/s)"], color=c, linestyle=ls, marker=mk)
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel(r"Maximum velocity, $U_{\max}$ (m s$^{-1}$)")
    style_axes(ax)
    panel_label(ax, "(b)")

    # Mark the selected production mesh without cluttering the legend.
    prod = mesh_table[mesh_table["Mesh"] == PRODUCTION_MESH]
    if not prod.empty:
        for ax, ycol in zip(axes, ("Tmax (K)", "Umax (m/s)")):
            ax.scatter(
                prod["Cells"], prod[ycol],
                marker="*", s=75, facecolors="none", edgecolors="black",
                linewidths=0.9, zorder=5,
            )

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "01_mesh_convergence")


def plot_mesh_temperature_profiles() -> None:
    meshes = _load_available_meshes()
    if not meshes:
        return

    styles = line_styles(len(meshes))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    for (name, s), (colour, ls, _) in zip(meshes, styles):
        z, Tz, actual_r = axial_profile(s, s["T_c"])
        r, Tr, actual_z = radial_profile(s, s["T_c"])
        axes[0].plot(z * 1000, Tz, color=colour, linestyle=ls, label=name)
        axes[1].plot(r * 1000, Tr, color=colour, linestyle=ls, label=name)

    axes[0].set_xlabel(r"Axial position, $z$ (mm)")
    axes[0].set_ylabel(r"Temperature, $T$ (K)")
    style_axes(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel(r"Radial position, $r$ (mm)")
    axes[1].set_ylabel(r"Temperature, $T$ (K)")
    style_axes(axes[1])
    panel_label(axes[1], "(b)")

    axes[1].legend(loc="best", ncol=1, handlelength=2.6)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "02_mesh_temperature_profiles")


def plot_mesh_velocity_profiles() -> None:
    meshes = _load_available_meshes()
    if not meshes:
        return

    styles = line_styles(len(meshes))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    for (name, s), (colour, ls, _) in zip(meshes, styles):
        z, uz_z, actual_r = axial_profile(s, s["uz_c"])
        r, uz_r, actual_z = radial_profile(s, s["uz_c"])
        axes[0].plot(z * 1000, uz_z, color=colour, linestyle=ls, label=name)
        axes[1].plot(r * 1000, uz_r, color=colour, linestyle=ls, label=name)

    axes[0].axhline(0.0, linewidth=0.6, color="0.45")
    axes[1].axhline(0.0, linewidth=0.6, color="0.45")
    axes[0].set_xlabel(r"Axial position, $z$ (mm)")
    axes[0].set_ylabel(r"Axial velocity, $u_z$ (m s$^{-1}$)")
    style_axes(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel(r"Radial position, $r$ (mm)")
    axes[1].set_ylabel(r"Axial velocity, $u_z$ (m s$^{-1}$)")
    style_axes(axes[1])
    panel_label(axes[1], "(b)")

    axes[1].legend(loc="best", ncol=1, handlelength=2.6)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "03_mesh_velocity_profiles")


# =============================================================================
# CURRENT-POWER-TEMPERATURE-VELOCITY FIGURES
# =============================================================================

# def plot_operating_relationships(power_table: pd.DataFrame) -> None:
#     """The core current-power-temperature-velocity publication figure."""
#     if power_table.empty:
#         return

#     current = power_table["Current (A)"].to_numpy(float)
#     power = power_table["Power used for plots (kW)"].to_numpy(float)
#     Tmax = power_table["Tmax (K)"].to_numpy(float)
#     Umax = power_table["Umax (m/s)"].to_numpy(float)

#     styles = line_styles(3)
#     fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))

#     datasets = [
#         (power, r"Absorbed plasma power, $P_{\mathrm{abs}}$ (kW)"),
#         (Tmax, r"Maximum temperature, $T_{\max}$ (K)"),
#         (Umax, r"Maximum velocity, $U_{\max}$ (m s$^{-1}$)"),
#     ]

#     for k, (ax, (y, ylabel), (colour, ls, mk)) in enumerate(zip(axes, datasets, styles)):
#         ax.plot(current, y, color=colour, linestyle=ls, marker=mk)
#         ax.set_xlabel(r"Coil current amplitude, $I_c$ (A)")
#         ax.set_ylabel(ylabel)
#         style_axes(ax)
#         panel_label(ax, f"({chr(97 + k)})")

#     fig.tight_layout(w_pad=1.65)
#     save_figure(fig, "04_current_power_temperature_velocity")


def plot_operating_relationships(power_table: pd.DataFrame) -> None:
    """Current-power-temperature-velocity relationships."""
    if power_table.empty:
        return

    current = power_table["Current (A)"].to_numpy(float)
    power = power_table["Integrated Joule power (kW)"].to_numpy(float)
    Tmax = power_table["Tmax (K)"].to_numpy(float)
    Umax = power_table["Umax (m/s)"].to_numpy(float)

    styles = line_styles(3)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))

    # ============================================================
    # (a) CURRENT VS ABSORBED POWER + QUADRATIC FIT
    # ============================================================

    ax = axes[0]
    colour, _, marker = styles[0]

    # Numerical results
    ax.plot(
        current,
        power,
        linestyle="none",
        marker=marker,
        color=colour,
        label="Numerical results",
        zorder=3,
    )

    # Only use valid integrated-power values for the regression
    mask = np.isfinite(current) & np.isfinite(power)

    I_fit = current[mask]
    P_fit = power[mask]

    if len(I_fit) >= 3:
        # Quadratic fit: P = a I^2 + b I + c
        coeff = np.polyfit(I_fit, P_fit, 2)
        a, b, c = coeff

        I_smooth = np.linspace(
            I_fit.min(),
            I_fit.max(),
            300,
        )

        P_smooth = np.polyval(
            coeff,
            I_smooth,
        )

        # Coefficient of determination
        P_pred = np.polyval(coeff, I_fit)

        ss_res = np.sum((P_fit - P_pred)**2)
        ss_tot = np.sum((P_fit - np.mean(P_fit))**2)

        r_squared = 1.0 - ss_res / ss_tot

        # Fitted curve
        ax.plot(
            I_smooth,
            P_smooth,
            linestyle="--",
            color="black",
            linewidth=1.2,
            label="Quadratic fit",
            zorder=2,
        )

        # Fit equation inside figure
        fit_text = (
            rf"$P_{{\rm abs}} = "
            rf"{a:.3e}I_c^2"
            rf"{b:+.3e}I_c"
            rf"{c:+.3f}$"
            "\n"
            rf"$R^2 = {r_squared:.4f}$"
        )

        # ax.text(
        #     0.05,
        #     0.94,
        #     fit_text,
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=7.2,
        # )

        print("\nCurrent-power quadratic fit")
        print("--------------------------------")
        print(
            f"P_abs = {a:.6e} I^2 "
            f"{b:+.6e} I "
            f"{c:+.6e}"
        )
        print(f"R^2 = {r_squared:.6f}")

    ax.set_xlabel(
        r"Coil current amplitude, $I_c$ (A)"
    )

    ax.set_ylabel(
        r"Absorbed power, $P_{\mathrm{abs}}$ (kW)"
    )

    ax.legend(
        loc="lower right",
        handlelength=2.2,
    )

    style_axes(ax)
    panel_label(ax, "(a)")


    # ============================================================
    # (b) CURRENT VS MAXIMUM TEMPERATURE
    # ============================================================

    ax = axes[1]
    colour, ls, marker = styles[1]

    ax.plot(
        current,
        Tmax,
        color=colour,
        linestyle=ls,
        marker=marker,
    )

    ax.set_xlabel(
        r"Coil current amplitude, $I_c$ (A)"
    )

    ax.set_ylabel(
        r"Maximum temperature, $T_{\max}$ (K)"
    )

    style_axes(ax)
    panel_label(ax, "(b)")


    # ============================================================
    # (c) CURRENT VS MAXIMUM VELOCITY
    # ============================================================

    ax = axes[2]
    colour, ls, marker = styles[2]

    ax.plot(
        current,
        Umax,
        color=colour,
        linestyle=ls,
        marker=marker,
    )

    ax.set_xlabel(
        r"Coil current amplitude, $I_c$ (A)"
    )

    ax.set_ylabel(
        r"Maximum velocity, $U_{\max}$ (m s$^{-1}$)"
    )

    style_axes(ax)
    panel_label(ax, "(c)")

    fig.tight_layout(w_pad=1.65)

    save_figure(
        fig,
        "04_current_power_temperature_velocity",
    )


def _profile_case_states() -> list[tuple[float, int, dict[str, np.ndarray]]]:
    out = []
    for current, target_power in POWER_CASES.items():
        if target_power not in PROFILE_POWERS:
            continue
        try:
            out.append((current, target_power, load_state(POWER_FILES[current])))
        except Exception as exc:
            warnings.warn(f"Skipping profile case {target_power} kW: {exc}")
    return out


def plot_power_temperature_profiles() -> None:
    cases = _profile_case_states()
    if not cases:
        return

    styles = line_styles(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    for (current, power, s), (colour, ls, _) in zip(cases, styles):
        z, Tz, _ = axial_profile(s, s["T_c"])
        r, Tr, _ = radial_profile(s, s["T_c"])
        label = f"{power:g} kW ({current:g} A)"
        axes[0].plot(z * 1000, Tz, color=colour, linestyle=ls, label=label)
        axes[1].plot(r * 1000, Tr, color=colour, linestyle=ls, label=label)

    axes[0].set_xlabel(r"Axial position, $z$ (mm)")
    axes[0].set_ylabel(r"Temperature, $T$ (K)")
    style_axes(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel(r"Radial position, $r$ (mm)")
    axes[1].set_ylabel(r"Temperature, $T$ (K)")
    style_axes(axes[1])
    panel_label(axes[1], "(b)")
    axes[1].legend(loc="best", handlelength=2.6)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "05_power_temperature_profiles")


def plot_power_velocity_profiles() -> None:
    cases = _profile_case_states()
    if not cases:
        return

    styles = line_styles(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    for (current, power, s), (colour, ls, _) in zip(cases, styles):
        z, uz_z, _ = axial_profile(s, s["uz_c"])
        r, uz_r, _ = radial_profile(s, s["uz_c"])
        label = f"{power:g} kW ({current:g} A)"
        axes[0].plot(z * 1000, uz_z, color=colour, linestyle=ls, label=label)
        axes[1].plot(r * 1000, uz_r, color=colour, linestyle=ls, label=label)

    axes[0].axhline(0.0, linewidth=0.6, color="0.45")
    axes[1].axhline(0.0, linewidth=0.6, color="0.45")
    axes[0].set_xlabel(r"Axial position, $z$ (mm)")
    axes[0].set_ylabel(r"Axial velocity, $u_z$ (m s$^{-1}$)")
    style_axes(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel(r"Radial position, $r$ (mm)")
    axes[1].set_ylabel(r"Axial velocity, $u_z$ (m s$^{-1}$)")
    style_axes(axes[1])
    panel_label(axes[1], "(b)")
    axes[1].legend(loc="best", handlelength=2.6)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "06_power_velocity_profiles")


def plot_power_response(power_table: pd.DataFrame) -> None:
    """Temperature and velocity directly against absorbed power."""
    if power_table.empty:
        return

    power = power_table["Power used for plots (kW)"].to_numpy(float)
    styles = line_styles(2)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))

    c, ls, mk = styles[0]
    axes[0].plot(power, power_table["Tmax (K)"], color=c, linestyle=ls, marker=mk)
    axes[0].set_xlabel(r"Absorbed plasma power, $P_{\mathrm{abs}}$ (kW)")
    axes[0].set_ylabel(r"Maximum temperature, $T_{\max}$ (K)")
    style_axes(axes[0])
    panel_label(axes[0], "(a)")

    c, ls, mk = styles[1]
    axes[1].plot(power, power_table["Umax (m/s)"], color=c, linestyle=ls, marker=mk)
    axes[1].set_xlabel(r"Absorbed plasma power, $P_{\mathrm{abs}}$ (kW)")
    axes[1].set_ylabel(r"Maximum velocity, $U_{\max}$ (m s$^{-1}$)")
    style_axes(axes[1])
    panel_label(axes[1], "(b)")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "07_power_temperature_velocity_response")


# =============================================================================
# OPTIONAL COMPUTATIONAL-MESH FIGURE
# =============================================================================

def centres_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("At least two centres are required to construct edges.")
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def plot_production_mesh() -> None:
    if PRODUCTION_MESH not in MESH_FILES:
        return

    try:
        s = load_state(MESH_FILES[PRODUCTION_MESH])
    except Exception as exc:
        warnings.warn(f"Cannot plot production mesh: {exc}")
        return

    z, r = coordinate_axes(s)
    ze = centres_to_edges(z) * 1000.0
    re = centres_to_edges(r) * 1000.0

    fig, ax = plt.subplots(figsize=(3.45, 5.5))
    for radius in re:
        ax.plot(np.full_like(ze, radius), ze, color="0.25", linewidth=0.24)
    for axial in ze:
        ax.plot(re, np.full_like(re, axial), color="0.25", linewidth=0.24)

    ax.set_xlabel(r"Radial position, $r$ (mm)")
    ax.set_ylabel(r"Axial position, $z$ (mm)")
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax, grid=False)
    fig.tight_layout()
    save_figure(fig, "08_production_mesh")


# =============================================================================
# EXPORT
# =============================================================================

def save_tables(power_table: pd.DataFrame, mesh_table: pd.DataFrame) -> None:
    power_table.to_csv(OUTPUT_DIR / "power_results.csv", index=False)
    mesh_table.to_csv(OUTPUT_DIR / "mesh_independence_detailed.csv", index=False)

    if not mesh_table.empty:
        paper_cols = [
            "Mesh", "Nz", "Nr", "Cells", "Tmax (K)",
            "Tmax error vs finest (%)", "Umax (m/s)",
            "Umax error vs finest (%)",
        ]
        mesh_table[paper_cols].to_csv(
            OUTPUT_DIR / "mesh_independence_paper_table.csv", index=False
        )


def check_power_mesh_consistency() -> None:
    shapes = {}
    for current, target_power in POWER_CASES.items():
        try:
            s = load_state(POWER_FILES[current])
            shapes[target_power] = s["T_c"].shape
        except Exception:
            continue

    if not shapes:
        return

    unique = set(shapes.values())
    if len(unique) != 1:
        warnings.warn(
            "Power cases do not all use the same physical mesh. For a clean journal "
            "comparison, rerun every operating condition on the same production mesh. "
            f"Found: {shapes}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_journal_style()
    check_power_mesh_consistency()

    power_table = build_power_table()
    mesh_table = build_mesh_table()

    print_diagnostics(power_table, mesh_table)
    save_tables(power_table, mesh_table)

    # Mesh-independence study.
    plot_mesh_convergence(mesh_table)
    plot_mesh_temperature_profiles()
    plot_mesh_velocity_profiles()

    # Operating-condition study.
    plot_operating_relationships(power_table)
    plot_power_temperature_profiles()
    plot_power_velocity_profiles()
    plot_power_response(power_table)

    # Methodology/supporting figure.
    plot_production_mesh()

    print(f"\nFinished. Journal outputs saved to: {OUTPUT_DIR.resolve()}")
    print("Primary manuscript figures: 01, 02/03, 04, and optionally 05/06 or 07.")


