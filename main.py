import Fundamental_Methods.helpers
import input
from Fundamental_Methods.live_simulation_modular import animate


# animate(input, Fundamental_Methods.helpers)


# from Post_Process.mesh_study import main; main()


from input import torch, Coils, OMEGA, MU0, IC
from Post_Process.post_processing import ICPPost
import matplotlib.pyplot as plt


# post = ICPPost.from_file(
#     "120_steady",
#     torch=torch,
#     coils=Coils,
#     mu0=MU0,
#     omega=OMEGA,
#     coil_current=IC,
# )
# fontsize = 10
# aspect = 5
# post.print_summary()
# post.plot_2d("T", cmap="viridis", fontsize=fontsize, box_aspect=aspect)
# post.plot_2d_with_streamlines("uz", cmap="viridis", streamline_color="black", 
#                               streamline_lw=0.55, box_aspect=aspect, fontsize=fontsize)
# post.plot_2d("P", cmap="magma", fontsize=fontsize, box_aspect=aspect, ylabel=False)
# post.plot_2d("Hmag", cmap="viridis", fontsize=fontsize, box_aspect=aspect)
# post.plot_axial(r=0.0, key="uz")
# post.plot_radial(z=0.102, key="T")
# # post.plot_grid()
# # post.plot_staggered_grid()
# # post.plot_sheath_mesh_zoom(z_max=0.055,r_min=0.015,r_max=0.026,)
# # "YlOrRd_r"
# # # Coil-in-air validation only:
# # post.plot_Hz_axis_validation(start_at_middle_coil=True)
# # post.plot_Hz_axis_relative_difference(start_at_middle_coil=True)

# plt.show()



post_2kW = ICPPost.from_file(
    "120_steady",
    torch=torch,
    coils=Coils,
)

post_8kW = ICPPost.from_file(
    "194_steady",
    torch=torch,
    coils=Coils,
)

post_15kW = ICPPost.from_file(
    "260_steady",
    torch=torch,
    coils=Coils,
)

fig, axs, cb = ICPPost.plot_horizontal_comparison(
    posts=[post_2kW, post_8kW, post_15kW],
    key="T",
    titles=["2 kW", "8 kW", "15 kW"],
    cmap="inferno",
    levels=60,
    streamlines=True,
    density=0.75,
    streamline_color="black",
    streamline_lw=0.45,

    fill_walls=True,
    wall_color="white",
    show_geometry=True,
    geometry_color="black",

    figsize=(6.0, 8.0),
    box_aspect=5.0,
    fontsize=8,
)
plt.show()

fig, axs, cb = ICPPost.plot_horizontal_stacked_comparison(
    posts=[
        post_2kW,
        post_8kW,
        post_15kW,
    ],

    key="uz",

    titles=[
        "2 kW",
        "8 kW",
        "15 kW",
    ],

    cmap="viridis",
    levels=60,

    streamlines=True,
    density=0.75,
    streamline_color="black",
    streamline_lw=0.45,

    fill_walls=True,
    wall_color="black",
    geometry_color="black",
    geometry_lw=1.0,

    figsize=(10, 5.0),
    panel_aspect=0.12,

    cbar_width=0.03,
    cbar_space=0.08,
    cbar_shrink=0.78,

    fontsize=8,
)

plt.show()



