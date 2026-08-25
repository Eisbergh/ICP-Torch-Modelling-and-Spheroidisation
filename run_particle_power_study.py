import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from input import torch, parameterfs
from Particle_Solver.particles import ParticleParametricStudy
from Particle_Solver.particle_postprocessing import ParticlePostProcessing


# =============================================================================
# POWER CASES
# =============================================================================

POWER_CASES = {
    2: "120_steady",
    3: "134_steady",
    4: "147_steady",
    5: "161_steady",
    6: "173_steady",
    7: "183_steady",
    8: "194_steady",
    9: "204_steady",
    10: "214_steady",
    11: "223_steady",
    12: "232_steady",
    13: "240_steady",
    14: "250_steady",
    15: "260_steady",
}


# =============================================================================
# PARTICLE STUDY
# =============================================================================

study = ParticleParametricStudy(
    power_cases=POWER_CASES,

    # These are only needed so the study object has its normal settings.
    diameters_um=np.arange(20, 401, 10),
    radial_positions_mm=np.linspace(0.1, 1.5, 8),

    torch=torch,
    parameterfs=parameterfs,

    dt=2e-5,
    z0=0.001,

    # Injection velocity = carrier-gas velocity.
    uzp0=None,
    urp0=0.0,

    Tp0=300.0,
    max_steps=100000,

    # Wall collisions reflect and particle continues.
    wall_mode="reflect",

    # Collision coefficients.
    wall_restitution=0.2,
    wall_friction=0.8,

    hot_temperature=8000.0,
    mass_loss_limit_pct=10.0,

    output_dir="particle_results",
)

# study.run(filename="particle_parametric_results.csv")

# =============================================================================
# POST-PROCESSING OBJECT
# =============================================================================

post = ParticlePostProcessing(
    study,

    output_dir="particle_results/journal_plots",

    cmap="inferno",

    dpi=600,

    # Plasma temperature scale.
    Tmin=300.0,
    Tmax=10500.0,

    contour_levels=48,

    # Save both PNG and PDF.
    save_pdf=True,

    # Journal text size.
    font_size=9,

    # Trajectory line thickness.
    trajectory_lw=1.55,

    # Compact colourbar.
    colorbar_height=0.70,
    colorbar_width=0.012,
)


# =============================================================================
# 1. DIAMETER TRAJECTORIES
# =============================================================================
# Main figure for showing the influence of particle size.

# post.plot_diameter_trajectories(
#     power_kW=5,

#     diameters_um=[
#         50,
#         100,
#         150,
#         200,
#         300,
#     ],

#     r0_mm=1.1,

#     orientation="horizontal",

#     show_streamlines=False,

#     # Star = fully molten.
#     show_full_melt=True,

#     # X = wall reflection.
#     show_wall_hits=True,
# )


# =============================================================================
# 2. INJECTION-RADIUS TRAJECTORIES
# =============================================================================
# Shows how radial injection position affects particle path.

# post.plot_radial_trajectories(
#     power_kW=10,

#     diameter_um=200,

#     radial_positions_mm=[
#         0.1,
#         0.5,
#         0.9,
#         1.3,
#         1.5,
#     ],

#     orientation="horizontal",

#     show_streamlines=False,

#     show_full_melt=True,

#     show_wall_hits=True,

# )


# =============================================================================
# 3. DIAMETER TRAJECTORIES AT DIFFERENT POWERS
# =============================================================================
# Excellent figure for comparing 5, 10 and 15 kW.

post.plot_diameter_power_comparison(
    powers_kW=[5, 10, 15],
    diameters_um=[50, 100, 150, 200, 300],
    r0_mm=1.1,

    orientation="horizontal",

    figsize=(10, 4.5),
    box_aspect=0.125,
    panel_spacing=0.3,

    show_streamlines=False,
    show_full_melt=True,
    show_wall_hits=True,
)

# =============================================================================
# 4. RADIAL-POSITION EFFECT AT DIFFERENT POWERS
# =============================================================================

# post.plot_radial_power_comparison(
#     powers_kW=[5, 10, 15],

#     diameter_um=50,

#     radial_positions_mm=[
#         0.1,
#         0.5,
#         0.9,
#         1.3,
#         1.5,
#     ],

#     orientation="horizontal",

#     show_streamlines=False,
#     show_full_melt=True,
#     show_wall_hits=True,

#     # Whole figure
#     figsize=(10, 4.5),

#     # Height / width of each torch panel
#     box_aspect=0.125,

#     # Vertical spacing between the three torches
#     panel_spacing=0.3,
# )

# =============================================================================
# 5. DETAILED SINGLE-PARTICLE THERMAL HISTORY
# =============================================================================
# This gives:
#
#   trajectory
#   particle temperature
#   local plasma temperature
#   liquid fraction
#   remaining particle mass
#
# This is one of the strongest plots for the paper.

# post.plot_single_particle_diagnostics(
#     power_kW=5,

#     diameter_um=50,

#     r0_mm=1.1,

#     orientation="horizontal",

#     show_streamlines=False,
# )


# =============================================================================
# LOAD FULL PARAMETRIC STUDY RESULTS
# =============================================================================

df = pd.read_csv(
    "particle_results/particle_parametric_results.csv"
)


# =============================================================================
# 6. PARTICLE STATE / SPHEROIDIZATION MAP
# =============================================================================
# Shows:
# unmelted
# partially melted
# fully melted with acceptable loss
# excessive evaporation
# fully evaporated

post.plot_particle_state_map(
    df=df,

    r0_mm=1.5,

    mass_loss_limit_pct=10.0,

    show_upstream=True,

    upstream_scope="selected_radius",

    filename="particle_state_map_r1.5_with_upstream.png",
)


# =============================================================================
# 7. FULL-MELT LOCATION MAP
# =============================================================================
# Shows where along the torch the particles become completely molten.

post.plot_melt_location_map(
    df=df,

    # Median full-melt position over radial injection positions.
    agg="median",

    # Hatch a power/diameter condition when at least
    # 50% of the radial injections are lost upstream.
    upstream_fraction_threshold=0.50,

    show_upstream_region=True,

    # IMPORTANT:
    # Include particles that melt before later escaping upstream.
    melt_population="all",

    filename="full_melt_location_with_upstream_loss.png",
)


# =============================================================================
# 8. PROCESSING ENVELOPE
# =============================================================================
# Defines an acceptable spheroidization operating window.
#
# Here:
# at least 50% of tested injection radii must result in
# complete melting while remaining below 10% mass loss.

post.plot_processing_envelope(
    df=df,

    # At least half of the tested injection positions
    # must satisfy the criterion.
    required_radial_fraction=0.50,

    # Engineering material-retention criterion.
    mass_loss_limit_pct=10.0,

    # Hatch a power/diameter combination when at least
    # half of the radial injections escape upstream.
    upstream_fraction_threshold=0.50,

    show_thermal_window=True,
    show_upstream_region=True,

    # Successful processing requires downstream recovery.
    require_outlet=True,

    filename="processing_envelope_with_upstream_loss.png",
)


# =============================================================================
# 9. WALL-CONTACT FRACTION MAP
# =============================================================================
# Useful for determining how important the reflection model actually is.

post.plot_wall_contact_fraction_map(
    df=df,
)


# =============================================================================
# 10. RADIAL MIGRATION / EXIT POSITION
# =============================================================================
# Shows whether particles are focused towards or away from the axis.

post.plot_exit_radius_response(
    df=df,

    power_kW=5,

    diameters_um=[
        50,
        100,
        150,
        200,
        300,
    ],
)


post.plot_mass_loss_map(
    df=df,

    required_radial_fraction=0.50,

    mass_loss_limit_pct=10.0,

    upstream_fraction_threshold=0.50,

    show_upstream_region=True,
)


post.plot_particle_power_diagnostics(
    df=df,
    diameters_um=[50, 100, 150, 200, 300],
    radial_quantile=0.50,
    show_spread=True,
    spread_quantiles=(0.25, 0.75),
    mass_loss_limit_pct=10.0,
)

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="hot_time_ms",
#     diameters_um=[50, 100, 150, 200, 300],
#     radial_quantile=0.50,
#     show_spread=True,
# )

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="residence_time_ms",
#     diameters_um=[50, 100, 150, 200, 300],
# )

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="thermal_exposure_Ks",
#     diameters_um=[50, 100, 150, 200, 300],
# )

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="Tp_max_K",
#     diameters_um=[50, 100, 150, 200, 300],
# )

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="mass_loss_pct",
#     diameters_um=[50, 100, 150, 200, 300],
#     mass_loss_limit_pct=10.0,
# )

# post.plot_particle_metric_vs_power(
#     df=df,
#     metric="Urel_max_m_s",
#     diameters_um=[50, 100, 150, 200, 300],
# )

# =============================================================================
# SHOW EVERYTHING
# =============================================================================

plt.show()

