import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from input import torch


class TorchParticleAnimator:
    """
    Animate a single particle inside the RF-ICP torch geometry.

    This object:
        - draws the plasma field,
        - overlays the torch geometry using parameters from input.py,
        - animates one particle by repeatedly calling solver.step(particle).
    """

    def __init__(
        self,
        solver,
        particle,
        steps_per_frame=1,
        max_frames=2000,
        copy_particle=True,
        background="T",
        show_streamlines=True,
        trail=True,
        trail_length=400,
        particle_size=45,
        interval=30,
        figscale=1.0,
        torch_object=torch,
        field_levels=80,
    ):
        self.solver = solver
        self.fields = solver.fields
        self.torch = torch_object

        if copy_particle:
            self.particle = copy.deepcopy(particle)
        else:
            self.particle = particle

        self.steps_per_frame = steps_per_frame
        self.max_frames = max_frames
        self.background = background
        self.show_streamlines = show_streamlines
        self.trail = trail
        self.trail_length = trail_length
        self.particle_size = particle_size
        self.interval = interval
        self.field_levels = field_levels

        self.z_history = []
        self.r_history = []
        self.t_history = []
        self.Tp_history = []
        self.dp_history = []
        self.x_history = []

        self.fig, self.ax = plt.subplots(
            figsize=(4.2 * figscale, 8.0 * figscale),
            dpi=120,
        )

        self.anim = None

        self._setup_plot()

    # ------------------------------------------------------------
    # Torch parameter helper
    # ------------------------------------------------------------

    def _torch_get(self, names, default=None, required=False):
        """
        Safely get torch parameters from either:
            torch.Lz
            torch["Lz"]

        names can be a string or list/tuple of possible names.
        """

        if isinstance(names, str):
            names = [names]

        for name in names:
            if hasattr(self.torch, name):
                return getattr(self.torch, name)

            if isinstance(self.torch, dict) and name in self.torch:
                return self.torch[name]

        if required:
            raise AttributeError(f"Could not find required torch parameter: {names}")

        return default

    def _torch_dimensions(self):
        """
        Reads the important torch dimensions.

        Expected names, based on your current code:
            Lz
            Lr
            Lz_carrier
            Lr_carrier
            Lz_sheath
            Lr_sheath

        It also accepts a few alternative names.
        """

        Lz = self._torch_get(["Lz", "length", "torch_length"], required=True)
        Lr = self._torch_get(["Lr", "radius", "torch_radius"], required=True)

        Lz_carrier = self._torch_get(["Lz_carrier", "carrier_length"], default=None)
        Lr_carrier = self._torch_get(["Lr_carrier", "carrier_radius"], default=None)

        Lz_sheath = self._torch_get(["Lz_sheath", "sheath_length"], default=None)
        Lr_sheath = self._torch_get(["Lr_sheath", "sheath_radius"], default=None)

        t_sheath = self._torch_get(["t_sheath", "sheath_thickness"], default=0.0)

        return {
            "Lz": Lz,
            "Lr": Lr,
            "Lz_carrier": Lz_carrier,
            "Lr_carrier": Lr_carrier,
            "Lz_sheath": Lz_sheath,
            "Lr_sheath": Lr_sheath,
            "t_sheath": t_sheath,
        }

    # ------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------

    def _get_axes(self):
        """
        Extract 1D z and r axes from 2D meshgrid arrays.

        Assumed array shape:
            Z.shape = R.shape = field.shape = (Nz, Nr)

        Therefore:
            z_axis = Z[:, 0]
            r_axis = R[0, :]
        """

        Z = self.fields.Z
        R = self.fields.R

        z_axis = Z[:, 0]
        r_axis = R[0, :]

        return z_axis, r_axis

    def _get_background_field(self):
        if self.background == "T":
            return self.fields.T, "Temperature [K]"

        if self.background == "uz":
            return self.fields.uz, "Axial velocity [m/s]"

        if self.background == "ur":
            return self.fields.ur, "Radial velocity [m/s]"

        if self.background == "speed":
            speed = np.sqrt(self.fields.uz**2 + self.fields.ur**2)
            return speed, "Velocity magnitude [m/s]"

        raise ValueError("background must be one of: 'T', 'uz', 'ur', 'speed'")

    # ------------------------------------------------------------
    # Torch drawing
    # ------------------------------------------------------------

    def _draw_torch_geometry(self):
        """
        Draws torch walls and inlet lips using torch parameters.
        """

        dims = self._torch_dimensions()

        Lz = dims["Lz"]
        Lr = dims["Lr"]

        Lz_carrier = dims["Lz_carrier"]
        Lr_carrier = dims["Lr_carrier"]

        Lz_sheath = dims["Lz_sheath"]
        Lr_sheath = dims["Lr_sheath"]
        t_sheath = dims["t_sheath"]

        wall_lw = 2.2
        thin_lw = 1.2

        # Centreline
        self.ax.plot(
            [0, Lz],
            [0, 0],
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.6,
            zorder=8,
        )

        # Outer torch wall
        self.ax.plot(
            [0, Lz],
            [Lr, Lr],
            linewidth=wall_lw,
            color="black",
            zorder=9,
        )

        # Inlet plane
        self.ax.plot(
            [0, 0],
            [0, Lr],
            linewidth=thin_lw,
            color="black",
            alpha=0.7,
            zorder=9,
        )

        # Outlet plane
        self.ax.plot(
            [Lz, Lz],
            [0, Lr],
            linewidth=thin_lw,
            color="black",
            alpha=0.7,
            zorder=9,
        )

        # Carrier inlet wall
        if Lz_carrier is not None and Lr_carrier is not None:
            self.ax.plot(
                [0, Lz_carrier],
                [Lr_carrier, Lr_carrier],
                linewidth=wall_lw,
                color="black",
                zorder=10,
            )

            # Carrier inlet lip / step
            self.ax.plot(
                [Lz_carrier, Lz_carrier],
                [0, Lr_carrier],
                linewidth=wall_lw,
                color="black",
                zorder=10,
            )

            self.ax.text(
                Lz_carrier,
                Lr_carrier,
                " carrier",
                fontsize=8,
                va="bottom",
                ha="left",
                color="black",
                zorder=11,
            )

        # Sheath inlet wall
        if Lz_sheath is not None and Lr_sheath is not None:
            self.ax.plot(
                [0, Lz_sheath],
                [Lr_sheath, Lr_sheath],
                linewidth=wall_lw,
                color="black",
                zorder=10,
            )

            # Optional inner sheath wall if t_sheath exists
            if t_sheath is not None and t_sheath > 0:
                Lr_sheath_inner = Lr_sheath - t_sheath

                self.ax.plot(
                    [0, Lz_sheath],
                    [Lr_sheath_inner, Lr_sheath_inner],
                    linewidth=thin_lw,
                    color="black",
                    linestyle=":",
                    alpha=0.8,
                    zorder=10,
                )

            # Sheath lip / step
            self.ax.plot(
                [Lz_sheath, Lz_sheath],
                [Lr_sheath, Lr],
                linewidth=wall_lw,
                color="black",
                zorder=10,
            )

            self.ax.text(
                Lz_sheath,
                Lr_sheath,
                " sheath",
                fontsize=8,
                va="bottom",
                ha="left",
                color="black",
                zorder=11,
            )

        # Light grey regions outside the main visible torch area
        # This makes the solid/inlet structure easier to see.
        if Lz_carrier is not None and Lr_carrier is not None:
            self.ax.add_patch(
                Rectangle(
                    (0, 0),
                    Lz_carrier,
                    Lr_carrier,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=9,
                )
            )

        if Lz_sheath is not None and Lr_sheath is not None:
            self.ax.add_patch(
                Rectangle(
                    (0, Lr_sheath),
                    Lz_sheath,
                    max(Lr - Lr_sheath, 0),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=9,
                )
            )

    # ------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------

    def _setup_plot(self):
        Z = self.fields.Z
        R = self.fields.R

        z_axis, r_axis = self._get_axes()
        field, label = self._get_background_field()

        dims = self._torch_dimensions()
        Lz = dims["Lz"]
        Lr = dims["Lr"]

        # Background field
        self.contour = self.ax.contourf(
            Z,
            R,
            field,
            levels=self.field_levels,
            cmap="inferno" if self.background == "T" else "viridis",
            zorder=1,
        )

        self.cbar = self.fig.colorbar(
            self.contour,
            ax=self.ax,
            fraction=0.045,
            pad=0.03,
        )
        self.cbar.set_label(label)

        # Streamlines
        if self.show_streamlines:
            try:
                self.ax.streamplot(
                    z_axis,
                    r_axis,
                    self.fields.uz.T,
                    self.fields.ur.T,
                    density=1.25,
                    linewidth=0.55,
                    arrowsize=0.7,
                    color="white",
                    zorder=5,
                )
            except Exception as error:
                print("Streamplot skipped:")
                print(error)

        # Torch geometry overlay
        self._draw_torch_geometry()

        # Particle trail
        if self.trail:
            (self.trail_line,) = self.ax.plot(
                [],
                [],
                linewidth=2.0,
                color="cyan",
                alpha=0.95,
                zorder=20,
            )
        else:
            self.trail_line = None

        # Particle marker
        self.particle_marker = self.ax.scatter(
            [self.particle.z],
            [self.particle.r],
            s=self.particle_size,
            marker="o",
            facecolor="white",
            edgecolor="cyan",
            linewidth=1.5,
            zorder=25,
        )

        # Information box
        self.info_text = self.ax.text(
            0.03,
            0.97,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.88,
            ),
            zorder=30,
        )

        self.ax.set_xlabel("Axial position, z [m]")
        self.ax.set_ylabel("Radial position, r [m]")
        self.ax.set_title("Single titanium particle trajectory in RF-ICP torch")

        self.ax.set_xlim(0, Lz)
        self.ax.set_ylim(0, Lr)

        # Important: this makes the torch geometry look physically correct.
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.grid(False)

        self.fig.tight_layout()

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def _store_current_state(self):
        self.z_history.append(self.particle.z)
        self.r_history.append(self.particle.r)
        self.t_history.append(getattr(self.particle, "t", 0.0))
        self.Tp_history.append(self.particle.Tp)
        self.dp_history.append(self.particle.dp)
        self.x_history.append(getattr(self.particle, "x", 0.0))

    def _update_plot(self):
        self.particle_marker.set_offsets([[self.particle.z, self.particle.r]])

        if self.trail and self.trail_line is not None:
            z_data = np.array(self.z_history)
            r_data = np.array(self.r_history)

            if self.trail_length is not None:
                z_data = z_data[-self.trail_length:]
                r_data = r_data[-self.trail_length:]

            self.trail_line.set_data(z_data, r_data)

        text = (
            f"t  = {getattr(self.particle, 't', 0.0):.4e} s\n"
            f"z  = {self.particle.z:.4e} m\n"
            f"r  = {self.particle.r:.4e} m\n"
            f"Tp = {self.particle.Tp:.1f} K\n"
            f"dp = {self.particle.dp * 1e6:.2f} μm\n"
            f"x  = {getattr(self.particle, 'x', 0.0):.3f}"
        )

        self.info_text.set_text(text)

    def _animate_frame(self, frame_index):
        if self.solver.is_finished(self.particle):
            self._update_plot()
            return self._artists()

        for _ in range(self.steps_per_frame):
            if self.solver.is_finished(self.particle):
                break

            self.solver.step(self.particle)

        self._store_current_state()
        self._update_plot()

        return self._artists()

    def _artists(self):
        artists = [self.particle_marker, self.info_text]

        if self.trail_line is not None:
            artists.append(self.trail_line)

        return artists

    # ------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------

    def animate(self):
        self._store_current_state()

        self.anim = FuncAnimation(
            self.fig,
            self._animate_frame,
            frames=self.max_frames,
            interval=self.interval,
            blit=False,
            repeat=False,
        )

        plt.show()

        return self.anim

    def save(self, filename, fps=30, dpi=200):
        if self.anim is None:
            self.anim = FuncAnimation(
                self.fig,
                self._animate_frame,
                frames=self.max_frames,
                interval=self.interval,
                blit=False,
                repeat=False,
            )

        self.anim.save(filename, fps=fps, dpi=dpi)

    def get_history(self):
        return {
            "t": np.array(self.t_history),
            "z": np.array(self.z_history),
            "r": np.array(self.r_history),
            "Tp": np.array(self.Tp_history),
            "dp": np.array(self.dp_history),
            "x": np.array(self.x_history),
        }
    