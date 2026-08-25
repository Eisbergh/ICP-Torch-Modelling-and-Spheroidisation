# ==================================================================================================
#                                   MAGNETIC FIELD VALIDATION
# ==================================================================================================
#
# Reproduces the two electromagnetic validation cases used by:
#
# Busse et al. (2021)
# "Numerical modeling of an inductively coupled plasma torch using OpenFOAM"
# Computers & Fluids 216, 104807.
#
# TEST 1:
#   Coil in air
#   sigma = 0 S/m
#   I = 161 A
#   Compare centreline H(z) against Biot-Savart analytical solution.
#
# TEST 2:
#   Conducting plasma region
#   sigma = 2500 S/m
#   I = 161 A
#   Compare radial H(xc) against skin-depth exponential approximation.
#
# Uses the CURRENT magnetic solver directly.
#
# ==================================================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from input import (
    grid,
    torch,
    Coils,
    OMEGA,
    MU0,
    IC,
)

from Fundamental_Methods.magnetic_field import ElectroMagnetic


# ==================================================================================================
#                                         SETTINGS
# ==================================================================================================

OUTPUT_DIR = Path("journal_plots/magnetic_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Number of fixed-point iterations of your EM solver.
AIR_ITERATIONS = 5
CONDUCTING_ITERATIONS = 50

# Busse validation conductivity
SIGMA_TEST = 2500.0      # S/m

# Busse Fig. 3 published values
BUSSE_AIR_NUMERICAL_MAX = 4516.0    # A/m
BUSSE_AIR_ANALYTICAL_MAX = 4506.0   # A/m

# Busse reports < 3% difference inside the skin depth.
BUSSE_SKIN_ERROR_LIMIT = 3.0         # %

# Optional:
#
# If you digitise Busse Fig. 3(b) and Fig. 4(a) using WebPlotDigitizer,
# save them as:
#
#     busse_fig3.csv
#     busse_fig4.csv
#
# with two columns:
#
# Fig. 3:
#     x = zc [mm]
#     y = H [A/m]
#
# Fig. 4:
#     x = xc [mm]
#     y = H [A/m]
#
# The script will add them automatically if present.

BUSSE_FIG3_CSV = Path("busse_fig3.csv")
BUSSE_FIG4_CSV = Path("busse_fig4.csv")


# ==================================================================================================
#                                    CONDUCTIVITY FUNCTIONS
# ==================================================================================================

def sigma_zero(T):
    """
    Coil-in-air validation:
        sigma = 0 S/m
    """
    return np.zeros_like(T, dtype=float)


def sigma_constant_2500(T):
    """
    Skin-depth validation:
        sigma = 2500 S/m
    """
    return np.full_like(T, SIGMA_TEST, dtype=float)


# ==================================================================================================
#                                       SOLVE EM CASE
# ==================================================================================================

def solve_magnetic_case(sigma_function, iterations):
    """
    Solve the current magnetic-field formulation for a prescribed
    electrical conductivity.

    Returns
    -------
    dict containing:
        R
        Z
        A
        Bz
        Br
        Hz
        Hr
        Hmag
        error
    """

    mag = ElectroMagnetic(
        grid=grid,
        Coils=Coils,
        omega=OMEGA,
        mu0=MU0,
        Ic=IC,
        sigmaf=sigma_function,
    )

    # Temperature is irrelevant here because conductivity is prescribed.
    T_dummy = np.full_like(
        mag.R,
        300.0,
        dtype=float,
    )

    A_guess = np.zeros_like(
        mag.R,
        dtype=np.complex128,
    )

    A, convergence_error = mag.magnetic_vector_solver(
        temp=T_dummy,
        iterations=iterations,
        A_guess=A_guess,
    )

    Bz, Br, Hz, Hr = mag.BzBr(A)

    Hmag = np.sqrt(
        np.abs(Hz)**2
        +
        np.abs(Hr)**2
    )

    result = {
        "R": mag.R.copy(),
        "Z": mag.Z.copy(),
        "A": A.copy(),
        "Bz": Bz.copy(),
        "Br": Br.copy(),
        "Hz": Hz.copy(),
        "Hr": Hr.copy(),
        "Hmag": Hmag.copy(),
        "error": convergence_error,
    }

    return result


# ==================================================================================================
#                             TEST 1 — ANALYTICAL COIL-IN-AIR FIELD
# ==================================================================================================

def analytical_axis_field(z):
    """
    Analytical magnetic-field strength on the axis of three
    circular current loops.

    Busse et al. Eq. (18):

                    I Rc^2
        H(z) = sum ---------
                   2[Rc^2 + (z-zi)^2]^(3/2)

    Parameters
    ----------
    z : ndarray
        Axial position [m].

    Returns
    -------
    H : ndarray
        Magnetic-field strength [A/m].
    """

    z = np.asarray(z)

    # input.py stores coils as:
    #
    # Coils =
    # [
    #     [z1, Rc],
    #     [z2, Rc],
    #     [z3, Rc],
    # ]
    #
    coil_z = Coils[:, 0]
    coil_r = Coils[:, 1]

    H = np.zeros_like(z, dtype=float)

    for zi, Rc in zip(coil_z, coil_r):

        H += (
            IC
            * Rc**2
            /
            (
                2.0
                *
                (
                    Rc**2
                    +
                    (z - zi)**2
                )**1.5
            )
        )

    return H


# ==================================================================================================
#                                   OPTIONAL BUSSE DATA
# ==================================================================================================

def load_digitised_data(filename):
    """
    Read optional two-column CSV digitised from a published Busse figure.

    Expected:
        first column  = x
        second column = H
    """

    if not filename.exists():
        return None

    try:

        data = np.genfromtxt(
            filename,
            delimiter=",",
            skip_header=1,
        )

        if data.ndim != 2 or data.shape[1] < 2:
            return None

        return data[:, 0], data[:, 1]

    except Exception as error:

        print(
            f"Could not read {filename}: {error}"
        )

        return None


# ==================================================================================================
#                             TEST 1 — COIL IN AIR
# ==================================================================================================

def validate_coil_in_air():
    """
    Reproduce Busse Fig. 3.

    sigma = 0 S/m
    I = 161 A

    Magnetic field is compared along the centre axis beginning at
    the middle coil turn.
    """

    print()
    print("=" * 80)
    print("TEST 1 — COIL IN AIR")
    print("=" * 80)

    result = solve_magnetic_case(
        sigma_function=sigma_zero,
        iterations=AIR_ITERATIONS,
    )

    R = result["R"]
    Z = result["Z"]
    Hz = result["Hz"]

    # --------------------------------------------------------------
    # Centre-axis coordinates
    # --------------------------------------------------------------

    z = Z[:, 0]

    # On the symmetry axis:
    #
    #     Hr = 0
    #
    # so H = |Hz|.
    #
    # Your first radial column represents the axis / near-axis line.

    H_numerical = np.abs(
        Hz[:, 0]
    )

    H_analytical = analytical_axis_field(z)

    # --------------------------------------------------------------
    # Busse starts Fig. 3 at the middle coil turn.
    #
    # Current coil locations:
    #     63 mm
    #     92 mm
    #     121 mm
    #
    # Therefore z_middle = 92 mm.
    # --------------------------------------------------------------

    middle_index = len(Coils) // 2

    z_middle = Coils[
        middle_index,
        0
    ]

    mask = (
        (z >= z_middle)
        &
        (z <= torch.Lz)
    )

    z_plot = z[mask]

    H_num_plot = H_numerical[mask]
    H_ana_plot = H_analytical[mask]

    # Coordinate used by Busse:
    #
    # zc = 0 at the middle coil.

    zc_mm = (
        z_plot
        -
        z_middle
    ) * 1000.0

    # --------------------------------------------------------------
    # Relative difference
    # --------------------------------------------------------------

    RD = (
        np.abs(
            H_num_plot
            -
            H_ana_plot
        )
        /
        np.maximum(
            np.abs(H_ana_plot),
            1e-30
        )
        *
        100.0
    )

    # --------------------------------------------------------------
    # Values around middle coil
    # --------------------------------------------------------------

    i_middle = np.argmin(
        np.abs(
            z
            -
            z_middle
        )
    )

    H_num_middle = H_numerical[i_middle]

    H_ana_middle = H_analytical[i_middle]

    difference_middle = (
        abs(
            H_num_middle
            -
            H_ana_middle
        )
        /
        H_ana_middle
        *
        100.0
    )

    difference_from_busse = (
        abs(
            H_num_middle
            -
            BUSSE_AIR_NUMERICAL_MAX
        )
        /
        BUSSE_AIR_NUMERICAL_MAX
        *
        100.0
    )

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print(
        f"Middle coil position        : "
        f"{z_middle*1000:.3f} mm"
    )

    print(
        f"Numerical H near axis       : "
        f"{H_num_middle:.3f} A/m"
    )

    print(
        f"Analytical H                : "
        f"{H_ana_middle:.3f} A/m"
    )

    print(
        f"Local numerical error       : "
        f"{difference_middle:.4f} %"
    )

    print()

    print(
        f"Busse numerical maximum     : "
        f"{BUSSE_AIR_NUMERICAL_MAX:.1f} A/m"
    )

    print(
        f"Busse analytical maximum    : "
        f"{BUSSE_AIR_ANALYTICAL_MAX:.1f} A/m"
    )

    print(
        f"Difference from Busse max   : "
        f"{difference_from_busse:.4f} %"
    )

    print()

    print(
        f"Maximum error over plotted region : "
        f"{np.nanmax(RD):.3f} %"
    )

    print(
        f"Final solver iteration error      : "
        f"{result['error']:.3e}"
    )

    # --------------------------------------------------------------
    # Optional Busse digitised data
    # --------------------------------------------------------------

    busse_data = load_digitised_data(
        BUSSE_FIG3_CSV
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.2, 4.2)
    )

    ax.plot(
        zc_mm,
        H_num_plot,
        linewidth=2.0,
        label="Present numerical solution",
    )

    ax.plot(
        zc_mm,
        H_ana_plot,
        linestyle="--",
        linewidth=1.8,
        label="Biot–Savart analytical solution",
    )

    # Published Busse maximum
    ax.scatter(
        [0.0],
        [BUSSE_AIR_NUMERICAL_MAX],
        marker="x",
        s=55,
        label="Busse et al. numerical maximum",
        zorder=10,
    )

    if busse_data is not None:

        x_busse, H_busse = busse_data

        ax.plot(
            x_busse,
            H_busse,
            linestyle=":",
            linewidth=1.8,
            label="Busse et al. numerical",
        )

    ax.set_xlabel(
        r"Axial distance from middle coil, $z_c$ (mm)"
    )

    ax.set_ylabel(
        r"Magnetic-field strength, $H$ (A m$^{-1}$)"
    )

    ax.set_xlim(
        0,
        (torch.Lz - z_middle) * 1000
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "01_coil_in_air_validation.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Error plot
    # --------------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(6.2, 3.8)
    )

    ax2.plot(
        zc_mm,
        RD,
        linewidth=2.0,
    )

    ax2.set_xlabel(
        r"Axial distance from middle coil, $z_c$ (mm)"
    )

    ax2.set_ylabel(
        "Relative difference (%)"
    )

    ax2.grid(
        alpha=0.25
    )

    fig2.tight_layout()

    fig2.savefig(
        OUTPUT_DIR /
        "02_coil_in_air_relative_error.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Save data
    # --------------------------------------------------------------

    table = np.column_stack(
        [
            z_plot,
            zc_mm,
            H_num_plot,
            H_ana_plot,
            RD,
        ]
    )

    np.savetxt(
        OUTPUT_DIR /
        "coil_in_air_validation.csv",
        table,
        delimiter=",",
        header=(
            "z_m,"
            "zc_mm,"
            "H_numerical_A_per_m,"
            "H_analytical_A_per_m,"
            "relative_difference_percent"
        ),
        comments="",
    )

    return {
        "result": result,
        "zc_mm": zc_mm,
        "H_numerical": H_num_plot,
        "H_analytical": H_ana_plot,
        "RD": RD,
    }


# ==================================================================================================
#                               TEST 2 — SKIN DEPTH
# ==================================================================================================

def validate_skin_depth():
    """
    Reproduce Busse Fig. 4.

    sigma = 2500 S/m
    I = 161 A

    Evaluate H radially inward from the torch wall at the position
    of the middle coil.
    """

    print()
    print("=" * 80)
    print("TEST 2 — SKIN-DEPTH / INDUCTION VALIDATION")
    print("=" * 80)

    result = solve_magnetic_case(
        sigma_function=sigma_constant_2500,
        iterations=CONDUCTING_ITERATIONS,
    )

    R = result["R"]
    Z = result["Z"]
    Hmag = result["Hmag"]

    # --------------------------------------------------------------
    # Skin depth
    #
    # delta = sqrt(2 / (omega mu sigma))
    # --------------------------------------------------------------

    delta = np.sqrt(
        2.0
        /
        (
            OMEGA
            *
            MU0
            *
            SIGMA_TEST
        )
    )

    delta_mm = delta * 1000.0

    # --------------------------------------------------------------
    # Position of middle coil
    # --------------------------------------------------------------

    middle_index = len(Coils) // 2

    z_middle = Coils[
        middle_index,
        0
    ]

    z_axis = Z[:, 0]

    iz = np.argmin(
        np.abs(
            z_axis
            -
            z_middle
        )
    )

    actual_z = z_axis[iz]

    # --------------------------------------------------------------
    # Radial profile
    # --------------------------------------------------------------

    r = R[iz, :]

    H_num = Hmag[iz, :]

    # Busse coordinate:
    #
    # xc = 0 at inner torch wall
    # xc increases towards centreline.
    #
    # xc = R_torch - r

    xc = torch.Lr - r

    # Sort from wall -> centreline
    order = np.argsort(xc)

    xc = xc[order]
    H_num = H_num[order]
    r = r[order]

    # --------------------------------------------------------------
    # Estimate H0 at the wall xc = 0.
    #
    # If the grid contains xc=0, use it directly.
    # Otherwise linearly extrapolate from the first two points.
    # --------------------------------------------------------------

    if abs(xc[0]) < 1e-12:

        H0 = H_num[0]

    else:

        slope = (
            H_num[1]
            -
            H_num[0]
        ) / (
            xc[1]
            -
            xc[0]
        )

        H0 = (
            H_num[0]
            -
            slope
            *
            xc[0]
        )

    # --------------------------------------------------------------
    # Analytical skin-depth approximation
    #
    # H = H0 exp(-xc / delta)
    # --------------------------------------------------------------

    H_analytical = (
        H0
        *
        np.exp(
            -xc
            /
            delta
        )
    )

    H_num_norm = H_num / H0

    H_ana_norm = np.exp(
        -xc
        /
        delta
    )

    # --------------------------------------------------------------
    # Relative difference
    # --------------------------------------------------------------

    RD = (
        np.abs(
            H_num
            -
            H_analytical
        )
        /
        np.maximum(
            np.abs(H_analytical),
            1e-30
        )
        *
        100.0
    )

    # Busse only treats exponential approximation as a valid
    # validation comparison within the skin depth.

    skin_mask = (
        xc
        <=
        delta
    )

    max_error_skin = np.nanmax(
        RD[skin_mask]
    )

    # --------------------------------------------------------------
    # H(delta) / H0
    # --------------------------------------------------------------

    H_delta_norm = np.interp(
        delta,
        xc,
        H_num_norm,
    )

    analytical_delta = np.exp(-1.0)

    delta_error = (
        abs(
            H_delta_norm
            -
            analytical_delta
        )
        /
        analytical_delta
        *
        100.0
    )

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print(
        f"Conductivity                 : "
        f"{SIGMA_TEST:.1f} S/m"
    )

    print(
        f"Frequency                    : "
        f"{OMEGA/(2*np.pi)/1e6:.3f} MHz"
    )

    print(
        f"Calculated skin depth        : "
        f"{delta_mm:.4f} mm"
    )

    print(
        f"Busse skin depth             : "
        f"~5.8 mm"
    )

    print()

    print(
        f"Requested middle-coil z      : "
        f"{z_middle*1000:.3f} mm"
    )

    print(
        f"Actual numerical profile z   : "
        f"{actual_z*1000:.3f} mm"
    )

    print()

    print(
        f"H(delta)/H0 numerical        : "
        f"{H_delta_norm:.5f}"
    )

    print(
        f"H(delta)/H0 analytical       : "
        f"{analytical_delta:.5f}"
    )

    print(
        f"Difference at delta          : "
        f"{delta_error:.3f} %"
    )

    print()

    print(
        f"Maximum RD for xc <= delta   : "
        f"{max_error_skin:.3f} %"
    )

    print(
        f"Busse reported               : "
        f"< {BUSSE_SKIN_ERROR_LIMIT:.1f} %"
    )

    if max_error_skin <= BUSSE_SKIN_ERROR_LIMIT:

        print(
            "RESULT                       : "
            "Within Busse's reported 3% range."
        )

    else:

        print(
            "RESULT                       : "
            "Larger deviation than Busse's reported result."
        )

    print(
        f"Final solver iteration error : "
        f"{result['error']:.3e}"
    )

    # --------------------------------------------------------------
    # Optional digitised Busse data
    # --------------------------------------------------------------

    busse_data = load_digitised_data(
        BUSSE_FIG4_CSV
    )

    # --------------------------------------------------------------
    # Absolute H plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.2, 4.2)
    )

    ax.plot(
        xc * 1000,
        H_num,
        linewidth=2.0,
        label="Present numerical solution",
    )

    ax.plot(
        xc * 1000,
        H_analytical,
        linestyle="--",
        linewidth=1.8,
        label="Skin-depth approximation",
    )

    if busse_data is not None:

        x_busse, H_busse = busse_data

        ax.plot(
            x_busse,
            H_busse,
            linestyle=":",
            linewidth=1.8,
            label="Busse et al. numerical",
        )

    ax.axvline(
        delta_mm,
        linestyle=":",
        linewidth=1.2,
        label=rf"$\delta={delta_mm:.2f}$ mm",
    )

    ax.set_xlabel(
        r"Distance from torch wall, $x_c$ (mm)"
    )

    ax.set_ylabel(
        r"Magnetic-field strength, $H$ (A m$^{-1}$)"
    )

    ax.set_xlim(
        0,
        torch.Lr * 1000
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "03_skin_depth_validation.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Normalised plot
    #
    # This is particularly useful for the paper because it compares
    # the FIELD DECAY independently of the exact surface magnitude.
    # --------------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(6.2, 4.2)
    )

    ax2.plot(
        xc * 1000,
        H_num_norm,
        linewidth=2.0,
        label="Present numerical solution",
    )

    ax2.plot(
        xc * 1000,
        H_ana_norm,
        linestyle="--",
        linewidth=1.8,
        label=r"$\exp(-x_c/\delta)$",
    )

    ax2.scatter(
        [delta_mm],
        [np.exp(-1.0)],
        marker="o",
        s=35,
        label=r"$H(\delta)/H_0=e^{-1}$",
        zorder=10,
    )

    ax2.axvline(
        delta_mm,
        linestyle=":",
        linewidth=1.2,
    )

    ax2.set_xlabel(
        r"Distance from torch wall, $x_c$ (mm)"
    )

    ax2.set_ylabel(
        r"Normalised magnetic field, $H/H_0$"
    )

    ax2.set_xlim(
        0,
        torch.Lr * 1000
    )

    ax2.set_ylim(
        bottom=0
    )

    ax2.grid(
        alpha=0.25
    )

    ax2.legend(
        frameon=False
    )

    fig2.tight_layout()

    fig2.savefig(
        OUTPUT_DIR /
        "04_skin_depth_normalised.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Relative difference around skin depth
    # --------------------------------------------------------------

    fig3, ax3 = plt.subplots(
        figsize=(6.2, 3.8)
    )

    plot_mask = (
        xc
        <=
        8.0 / 1000.0
    )

    ax3.plot(
        xc[plot_mask] * 1000,
        RD[plot_mask],
        linewidth=2.0,
    )

    ax3.axvline(
        delta_mm,
        linestyle=":",
        linewidth=1.2,
        label=rf"$\delta={delta_mm:.2f}$ mm",
    )

    ax3.axhline(
        BUSSE_SKIN_ERROR_LIMIT,
        linestyle="--",
        linewidth=1.2,
        label="Busse et al.: 3%",
    )

    ax3.set_xlabel(
        r"Distance from torch wall, $x_c$ (mm)"
    )

    ax3.set_ylabel(
        "Relative difference (%)"
    )

    ax3.set_xlim(
        0,
        8
    )

    ax3.grid(
        alpha=0.25
    )

    ax3.legend(
        frameon=False
    )

    fig3.tight_layout()

    fig3.savefig(
        OUTPUT_DIR /
        "05_skin_depth_relative_error.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Save data
    # --------------------------------------------------------------

    table = np.column_stack(
        [
            r,
            xc,
            xc * 1000,
            H_num,
            H_analytical,
            H_num_norm,
            H_ana_norm,
            RD,
        ]
    )

    np.savetxt(
        OUTPUT_DIR /
        "skin_depth_validation.csv",
        table,
        delimiter=",",
        header=(
            "r_m,"
            "xc_m,"
            "xc_mm,"
            "H_numerical_A_per_m,"
            "H_analytical_A_per_m,"
            "H_numerical_normalised,"
            "H_analytical_normalised,"
            "relative_difference_percent"
        ),
        comments="",
    )

    return {
        "result": result,
        "xc_mm": xc * 1000,
        "H_numerical": H_num,
        "H_analytical": H_analytical,
        "RD": RD,
        "delta_mm": delta_mm,
    }


# ==================================================================================================
#                             COMBINED JOURNAL FIGURE
# ==================================================================================================

def plot_combined_validation(air, skin):
    """
    Produce one concise two-panel figure suitable for the paper.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.0),
    )

    # ----------------------------------------------------------------------------------------------
    # (a) Coil in air
    # ----------------------------------------------------------------------------------------------

    ax = axes[0]

    ax.plot(
        air["zc_mm"],
        air["H_numerical"],
        linewidth=2.0,
        label="Present solver",
    )

    ax.plot(
        air["zc_mm"],
        air["H_analytical"],
        linestyle="--",
        linewidth=1.8,
        label="Analytical",
    )

    ax.scatter(
        [0],
        [BUSSE_AIR_NUMERICAL_MAX],
        marker="x",
        s=45,
        label="Busse et al.",
    )

    busse3 = load_digitised_data(
        BUSSE_FIG3_CSV
    )

    if busse3 is not None:

        xb, Hb = busse3

        ax.plot(
            xb,
            Hb,
            linestyle=":",
            linewidth=1.5,
            label="Busse et al. digitised",
        )

    ax.set_xlabel(
        r"$z_c$ (mm)"
    )

    ax.set_ylabel(
        r"$H$ (A m$^{-1}$)"
    )

    ax.set_title(
        "(a) Coil in air"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    # ----------------------------------------------------------------------------------------------
    # (b) Skin effect
    # ----------------------------------------------------------------------------------------------

    ax = axes[1]

    H0 = skin["H_numerical"][0]

    ax.plot(
        skin["xc_mm"],
        skin["H_numerical"] / H0,
        linewidth=2.0,
        label="Present solver",
    )

    ax.plot(
        skin["xc_mm"],
        skin["H_analytical"] / skin["H_analytical"][0],
        linestyle="--",
        linewidth=1.8,
        label="Analytical",
    )

    ax.axvline(
        skin["delta_mm"],
        linestyle=":",
        linewidth=1.2,
        label=rf"$\delta={skin['delta_mm']:.2f}$ mm",
    )

    ax.scatter(
        [skin["delta_mm"]],
        [np.exp(-1)],
        marker="o",
        s=30,
        zorder=10,
    )

    ax.set_xlabel(
        r"$x_c$ (mm)"
    )

    ax.set_ylabel(
        r"$H/H_0$"
    )

    ax.set_title(
        r"(b) $\sigma=2500$ S m$^{-1}$"
    )

    ax.set_xlim(
        0,
        torch.Lr * 1000
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "06_magnetic_validation_combined.png",
        dpi=DPI,
        bbox_inches="tight",
    )


# ==================================================================================================
#                                           RUN
# ==================================================================================================

def magnetic_validation():

    print()
    print("#" * 80)
    print("MAGNETIC FIELD VALIDATION")
    print("#" * 80)

    print()
    print("Current model parameters:")
    print(f"Coil current = {IC:.3f} A")
    print(f"Frequency    = {OMEGA/(2*np.pi)/1e6:.3f} MHz")
    print(f"Torch radius = {torch.Lr*1000:.3f} mm")

    print()

    print("Coil geometry:")

    for i, coil in enumerate(Coils):

        print(
            f"Coil {i+1}: "
            f"z = {coil[0]*1000:.3f} mm, "
            f"R = {coil[1]*1000:.3f} mm"
        )

    # Test 1
    air = validate_coil_in_air()

    # Test 2
    skin = validate_skin_depth()

    # Combined journal plot
    plot_combined_validation(
        air,
        skin,
    )

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"Figures and CSV files saved in:\n"
        f"    {OUTPUT_DIR.resolve()}"
    )

    plt.show()

    return air, skin


if __name__ == "__main__":

    magnetic_validation()

    