from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator

from Particle_Solver.particles import PlasmaFields
from Particle_Solver.particle_parameters import Tmp, Tbp


class ParticlePostProcessing:
    """
    Journal-quality post-processing for the RF-ICP Lagrangian particle model.

    The class recreates selected particles through ParticleParametricStudy,
    stores their detailed histories, and produces publication-ready figures.

    Main trajectory figures
    -----------------------
    plot_diameter_trajectories(...)
    plot_radial_trajectories(...)
    plot_diameter_power_comparison(...)
    plot_radial_power_comparison(...)

    Process/diagnostic figures
    --------------------------
    plot_particle_state_map(...)
    plot_single_particle_diagnostics(...)
    plot_melt_location_map(...)
    plot_processing_envelope(...)
    plot_exit_radius_response(...)
    plot_wall_contact_fraction_map(...)

    Notes
    -----
    * Wall impacts are plotted as trajectory events. They are not assumed to
      terminate the trajectory when the particle solver uses wall_mode="reflect".
    * Temperature backgrounds use one common scale for multi-power figures.
    * Figures are saved as both PNG and PDF by default.
    """

    def __init__(
        self,
        study,
        output_dir="particle_results/trajectory_plots",
        cmap="inferno",
        dpi=600,
        Tmin=300.0,
        Tmax=None,
        contour_levels=48,
        wall_facecolor="0.84",
        save_pdf=True,
        font_size=9,
        trajectory_lw=1.55,
        colorbar_height=0.70,
        colorbar_width=0.012,
    ):
        self.study = study
        self.torch = study.torch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cmap = cmap
        self.dpi = int(dpi)
        self.Tmin = float(Tmin)
        self.Tmax = None if Tmax is None else float(Tmax)
        self.contour_levels = int(contour_levels)
        self.wall_facecolor = wall_facecolor
        self.save_pdf = bool(save_pdf)
        self.font_size = int(font_size)
        self.trajectory_lw = float(trajectory_lw)
        self.colorbar_height = float(colorbar_height)
        self.colorbar_width = float(colorbar_width)

        self._field_cache = {}
        self._trajectory_cache = {}

        # Compact journal styling.  Remove/modify this block if your journal
        # requires a different font family.
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size + 1,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": self.dpi,
        })

    # ======================================================================
    # DATA ACCESS
    # ======================================================================

    def _nearest_power(self, power_kW):
        powers = np.asarray(list(self.study.power_cases.keys()), dtype=float)
        idx = int(np.argmin(np.abs(powers - float(power_kW))))
        return float(powers[idx])

    def _state_name(self, power_kW):
        power = self._nearest_power(power_kW)
        for key, value in self.study.power_cases.items():
            if np.isclose(float(key), power):
                return power, value
        raise KeyError(f"Could not resolve a saved state for {power_kW} kW.")

    def _load_fields(self, power_kW):
        power, state_name = self._state_name(power_kW)
        if power not in self._field_cache:
            self._field_cache[power] = PlasmaFields.load(state_name)
        return power, self._field_cache[power]

    def clear_cache(self):
        """Clear cached fields and selected trajectory histories."""
        self._field_cache.clear()
        self._trajectory_cache.clear()

    def run_case(self, power_kW, diameter_um, r0_mm):
        """Run one selected particle with full history and cache the result."""
        power = self._nearest_power(power_kW)
        key = (float(power), float(diameter_um), float(r0_mm))

        if key in self._trajectory_cache:
            return self._trajectory_cache[key]

        _, fields = self._load_fields(power)
        solver = self.study._make_solver(fields, store_history=True)
        particle = self.study._make_particle(diameter_um, r0_mm)
        history = solver.solve(particle, max_steps=self.study.max_steps)

        if history is None or len(history) == 0:
            raise RuntimeError(
                f"No trajectory history returned for {power:g} kW, "
                f"{diameter_um:g} um, r0={r0_mm:g} mm."
            )

        summary = solver.summary(particle)
        outcome = self.study._classify(summary) if hasattr(self.study, "_classify") else None

        result = {
            "power_kW": power,
            "diameter_um": float(diameter_um),
            "r0_mm": float(r0_mm),
            "particle": particle,
            "history": np.asarray(history, dtype=float),
            "summary": summary,
            "outcome": outcome,
        }
        self._trajectory_cache[key] = result
        return result

    # ======================================================================
    # GENERAL HELPERS
    # ======================================================================

    @staticmethod
    def _centres_to_edges(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])

        edges = np.empty(len(values) + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return edges

    @staticmethod
    def _line_colors(n):
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(int(n))]

    def _temperature_limits(self, powers_kW):
        maxima = []
        for power in powers_kW:
            _, fields = self._load_fields(power)
            maxima.append(float(np.nanmax(fields.T)))

        vmin = self.Tmin
        vmax = self.Tmax if self.Tmax is not None else np.ceil(max(maxima) / 500.0) * 500.0
        return float(vmin), float(vmax)

    def _format_colorbar(self, cbar, vmin=None, vmax=None, label="Plasma temperature [K]"):
        cbar.set_label(label, labelpad=7)
        cbar.ax.tick_params(direction="out", length=3.0, width=0.7)

        if vmin is None or vmax is None:
            cbar.locator = MaxNLocator(nbins=6)
            cbar.update_ticks()
            return

        locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
        ticks = locator.tick_values(vmin, vmax)
        ticks = ticks[(ticks >= vmin - 1e-12) & (ticks <= vmax + 1e-12)]
        if len(ticks):
            cbar.set_ticks(ticks)

    def _compact_colorbar(
        self,
        fig,
        mappable,
        axes,
        vmin=None,
        vmax=None,
        label="Plasma temperature [K]",
        height_frac=None,
        width=None,
        pad=0.018,
    ):
        """
        Add a deliberately short, thin journal-style colorbar.

        axes may be one Axes object or a list/array of Axes objects.  The bar
        is centred vertically relative to the actual plotting axes.
        """
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        axes = np.asarray(axes, dtype=object).ravel().tolist()

        # Equal-aspect axes can change position after drawing, so update first.
        fig.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        x1 = max(box.x1 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        y1 = max(box.y1 for box in boxes)

        height_frac = self.colorbar_height if height_frac is None else float(height_frac)
        width = self.colorbar_width if width is None else float(width)

        full_h = y1 - y0
        cbar_h = full_h * height_frac
        cbar_y = 0.5 * (y0 + y1 - cbar_h)
        cbar_x = x1 + pad

        cax = fig.add_axes([cbar_x, cbar_y, width, cbar_h])
        cbar = fig.colorbar(mappable, cax=cax)
        self._format_colorbar(cbar, vmin=vmin, vmax=vmax, label=label)
        return cbar

    @staticmethod
    def _format_scalar_axis(ax, grid=True):
        ax.tick_params(direction="in", top=True, right=True)
        if grid:
            ax.grid(alpha=0.18, linewidth=0.5)

    # ======================================================================
    # TORCH GEOMETRY
    # ======================================================================

    def _geometry_values_mm(self):
        t = self.torch
        return {
            "Lz": t.Lz * 1e3,
            "Lr": t.Lr * 1e3,
            "zc": t.Lz_carrier * 1e3,
            "rc_i": max(0.0, t.Lr_carrier - getattr(t, "t_carrier", 0.0)) * 1e3,
            "rc_o": t.Lr_carrier * 1e3,
            "zs": t.Lz_sheath * 1e3,
            "rs_i": t.Lr_sheath * 1e3,
            "rs_o": min(t.Lr, t.Lr_sheath + getattr(t, "t_sheath", 0.0)) * 1e3,
        }

    def _draw_geometry(self, ax, orientation="horizontal"):
        g = self._geometry_values_mm()

        if orientation == "vertical":
            # x = r, y = z
            ax.add_patch(Rectangle(
                (g["rc_i"], 0), g["rc_o"] - g["rc_i"], g["zc"],
                facecolor=self.wall_facecolor, edgecolor="black",
                linewidth=0.9, zorder=8,
            ))
            ax.add_patch(Rectangle(
                (g["rs_i"], 0), g["rs_o"] - g["rs_i"], g["zs"],
                facecolor=self.wall_facecolor, edgecolor="black",
                linewidth=0.9, zorder=8,
            ))
            ax.plot([g["Lr"], g["Lr"]], [0, g["Lz"]], color="black", linewidth=1.2, zorder=10)
            ax.plot([0, 0], [0, g["Lz"]], linestyle="--", color="0.35", linewidth=0.6, zorder=7)

        elif orientation == "horizontal":
            # x = z, y = r
            ax.add_patch(Rectangle(
                (0, g["rc_i"]), g["zc"], g["rc_o"] - g["rc_i"],
                facecolor=self.wall_facecolor, edgecolor="black",
                linewidth=0.9, zorder=8,
            ))
            ax.add_patch(Rectangle(
                (0, g["rs_i"]), g["zs"], g["rs_o"] - g["rs_i"],
                facecolor=self.wall_facecolor, edgecolor="black",
                linewidth=0.9, zorder=8,
            ))
            ax.plot([0, g["Lz"]], [g["Lr"], g["Lr"]], color="black", linewidth=1.2, zorder=10)
            ax.plot([0, g["Lz"]], [0, 0], linestyle="--", color="0.35", linewidth=0.6, zorder=7)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")

    def _format_axis(
        self,
        ax,
        orientation,
        title=None,
        show_ylabel=True,
        box_aspect=None,
    ):
        """
        Format a torch trajectory axis.

        Parameters
        ----------
        box_aspect : float or None
            Visual axes height/width ratio.

            None:
                Preserve the true physical torch geometry using equal data
                scaling. For the 200 mm x 25 mm horizontal PL-50 domain this
                corresponds to an apparent ratio of about 0.125.

            float:
                Stretch/compress the displayed axes without changing any
                coordinates or trajectory data. For horizontal torch plots,
                values around 0.15-0.20 make the torch visually taller.
                For vertical plots, the value is still height/width.
        """
        g = self._geometry_values_mm()

        if orientation == "vertical":
            ax.set_xlim(0, g["Lr"])
            ax.set_ylim(0, g["Lz"])
            ax.set_xlabel(r"$r$ [mm]")
            if show_ylabel:
                ax.set_ylabel(r"$z$ [mm]")

        elif orientation == "horizontal":
            ax.set_xlim(0, g["Lz"])
            ax.set_ylim(0, g["Lr"])
            ax.set_xlabel(r"$z$ [mm]")
            if show_ylabel:
                ax.set_ylabel(r"$r$ [mm]")

        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")

        if box_aspect is None:
            # Preserve the true physical geometry.
            ax.set_aspect("equal", adjustable="box")
        else:
            if float(box_aspect) <= 0:
                raise ValueError("box_aspect must be positive or None.")
            # Change only the visual shape of the plotting box.  The physical
            # coordinate limits and all trajectory/plasma data remain unchanged.
            ax.set_aspect("auto")
            ax.set_box_aspect(float(box_aspect))

        ax.tick_params(direction="in", top=True, right=True)

        if title:
            ax.set_title(title, pad=7)

    def _temperature_background(self, ax, fields, vmin, vmax, orientation):
        levels = np.linspace(vmin, vmax, self.contour_levels)

        if orientation == "vertical":
            x = fields.R * 1e3
            y = fields.Z * 1e3
        elif orientation == "horizontal":
            x = fields.Z * 1e3
            y = fields.R * 1e3
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")

        return ax.contourf(
            x, y, fields.T,
            levels=levels,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            extend="both",
            zorder=1,
        )

    def _streamlines(self, ax, fields, orientation, density=0.55):
        r = fields.R[0, :] * 1e3
        z = fields.Z[:, 0] * 1e3

        try:
            if orientation == "vertical":
                ax.streamplot(
                    r, z, fields.ur, fields.uz,
                    density=density, linewidth=0.35, arrowsize=0.55,
                    color="white", zorder=3,
                )
            else:
                ax.streamplot(
                    z, r, fields.uz.T, fields.ur.T,
                    density=density, linewidth=0.35, arrowsize=0.55,
                    color="white", zorder=3,
                )
        except Exception as error:
            print(f"Streamlines skipped: {error}")

    # ======================================================================
    # TRAJECTORIES + EVENTS
    # ======================================================================

    @staticmethod
    def _fate_marker(fate):
        return {
            "outlet": "s",
            "wall_hit": "X",
            "upstream_escape": "<",
            "fully_evaporated": "x",
            "max_steps": "D",
            "active": "o",
        }.get(str(fate), "o")

    def _wall_event_points(self, result):
        particle = result["particle"]
        events = getattr(particle, "wall_events", None)

        if events:
            z = np.asarray([event["z_m"] for event in events], dtype=float)
            r = np.asarray([event["r_m"] for event in events], dtype=float)
            return z, r

        # Backward compatibility with a solver that stores only the first hit.
        z = getattr(particle, "wall_hit_z", None)
        r = getattr(particle, "wall_hit_r", None)
        if z is not None and r is not None:
            return np.asarray([z], dtype=float), np.asarray([r], dtype=float)

        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    def _plot_trajectory(
        self,
        ax,
        result,
        label,
        color,
        orientation,
        show_start=True,
        show_full_melt=True,
        show_wall_hits=True,
        show_end=True,
    ):
        h = result["history"]
        z = h[:, 1] * 1e3
        r = h[:, 2] * 1e3
        x_liq = h[:, 7]

        if orientation == "vertical":
            xx, yy = r, z
        else:
            xx, yy = z, r

        line, = ax.plot(
            xx, yy,
            color=color,
            linewidth=self.trajectory_lw,
            label=label,
            zorder=15,
        )
        line.set_path_effects([
            pe.Stroke(linewidth=self.trajectory_lw + 1.2, foreground="white", alpha=0.82),
            pe.Normal(),
        ])

        if show_start:
            ax.scatter(
                xx[0], yy[0], s=23, marker="o",
                facecolor="white", edgecolor=color,
                linewidth=1.0, zorder=20,
            )

        if show_full_melt and np.nanmax(x_liq) >= 1.0 - 1e-8:
            idx = np.where(x_liq >= 1.0 - 1e-8)[0]
            if len(idx):
                i = int(idx[0])
                ax.scatter(
                    xx[i], yy[i], s=44, marker="*",
                    facecolor=color, edgecolor="white",
                    linewidth=0.65, zorder=22,
                )

        if show_wall_hits:
            wz, wr = self._wall_event_points(result)
            if len(wz):
                if orientation == "vertical":
                    wx, wy = wr * 1e3, wz * 1e3
                else:
                    wx, wy = wz * 1e3, wr * 1e3
                ax.scatter(
                    wx, wy, s=36, marker="X",
                    facecolor="white", edgecolor=color,
                    linewidth=1.0, zorder=23,
                )

        if show_end:
            fate = result["summary"].get("fate", "active")
            marker = self._fate_marker(fate)

            # Matplotlib warns for an unfilled x marker if edgecolor is supplied.
            if marker == "x":
                ax.scatter(xx[-1], yy[-1], s=36, marker=marker, color=color, linewidth=1.2, zorder=24)
            else:
                ax.scatter(
                    xx[-1], yy[-1], s=32, marker=marker,
                    facecolor=color, edgecolor="white",
                    linewidth=0.7, zorder=24,
                )

    @staticmethod
    def event_key_handles():
        return [
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
                   markeredgecolor="black", label="Injection"),
            Line2D([], [], marker="*", linestyle="None", markerfacecolor="0.4",
                   markeredgecolor="white", markersize=9, label="Fully molten"),
            Line2D([], [], marker="X", linestyle="None", markerfacecolor="white",
                   markeredgecolor="0.3", label="Wall reflection"),
            Line2D([], [], marker="s", linestyle="None", markerfacecolor="0.4",
                   markeredgecolor="white", label="Outlet"),
        ]

    # ======================================================================
    # SINGLE-POWER TRAJECTORY FIGURES
    # ======================================================================

    def _plot_single_power(
        self,
        power_kW,
        cases,
        title,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        legend_ncol=None,
        save=True,
        filename="particle_trajectories.png",
    ):
        """
        Plot several particle trajectories on one plasma field.

        figsize controls the complete figure size.
        box_aspect controls the visual height/width ratio of the torch axes.
        """
        power, fields = self._load_fields(power_kW)
        vmin, vmax = self._temperature_limits([power])

        if figsize is None:
            figsize = (5.0, 7.7) if orientation == "vertical" else (9.0, 3.25)

        fig = plt.figure(figsize=figsize)

        if orientation == "horizontal":
            gs = fig.add_gridspec(
                1, 1,
                left=0.075, right=0.84,
                bottom=0.14, top=0.75,
            )
            title_y = 0.965
            legend_y = 0.885
        else:
            gs = fig.add_gridspec(
                1, 1,
                left=0.14, right=0.82,
                bottom=0.08, top=0.82,
            )
            title_y = 0.965
            legend_y = 0.915

        ax = fig.add_subplot(gs[0, 0])
        contour = self._temperature_background(ax, fields, vmin, vmax, orientation)

        if show_streamlines:
            self._streamlines(ax, fields, orientation)

        self._draw_geometry(ax, orientation)

        colors = self._line_colors(len(cases))
        for color, case in zip(colors, cases):
            result = self.run_case(power, case["diameter_um"], case["r0_mm"])
            self._plot_trajectory(
                ax,
                result,
                label=case["label"],
                color=color,
                orientation=orientation,
                show_full_melt=show_full_melt,
                show_wall_hits=show_wall_hits,
            )

        self._format_axis(
            ax,
            orientation,
            box_aspect=box_aspect,
        )
        fig.suptitle(title, y=title_y)

        handles, labels = ax.get_legend_handles_labels()
        if legend_ncol is None:
            legend_ncol = (
                min(5, len(labels))
                if orientation == "horizontal"
                else min(2, len(labels))
            )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.46, legend_y),
            ncol=legend_ncol,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.35,
        )

        self._compact_colorbar(
            fig,
            contour,
            ax,
            vmin=vmin,
            vmax=vmax,
            height_frac=0.72 if orientation == "horizontal" else 0.62,
            width=0.011 if orientation == "horizontal" else 0.012,
            pad=0.018,
        )

        if save:
            self._save(fig, filename)

        return fig, ax

    def plot_diameter_trajectories(
        self,
        power_kW,
        diameters_um,
        r0_mm,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        save=True,
        filename=None,
    ):
        cases = [
            {
                "diameter_um": float(dp),
                "r0_mm": float(r0_mm),
                "label": rf"{dp:g} $\mu$m",
            }
            for dp in diameters_um
        ]

        if filename is None:
            filename = f"diameter_trajectories_{power_kW:g}kW_r{r0_mm:g}mm.png"

        title = rf"{power_kW:g} kW, $r_0={r0_mm:g}$ mm"

        return self._plot_single_power(
            power_kW=power_kW,
            cases=cases,
            title=title,
            orientation=orientation,
            show_streamlines=show_streamlines,
            show_full_melt=show_full_melt,
            show_wall_hits=show_wall_hits,
            figsize=figsize,
            box_aspect=box_aspect,
            save=save,
            filename=filename,
        )

    def plot_radial_trajectories(
        self,
        power_kW,
        diameter_um,
        radial_positions_mm,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        save=True,
        filename=None,
    ):
        cases = [
            {
                "diameter_um": float(diameter_um),
                "r0_mm": float(r0),
                "label": rf"$r_0={r0:g}$ mm",
            }
            for r0 in radial_positions_mm
        ]

        if filename is None:
            filename = f"radial_trajectories_{power_kW:g}kW_{diameter_um:g}um.png"

        title = rf"{power_kW:g} kW, $d_{{p,0}}={diameter_um:g}$ $\mu$m"

        return self._plot_single_power(
            power_kW=power_kW,
            cases=cases,
            title=title,
            orientation=orientation,
            show_streamlines=show_streamlines,
            show_full_melt=show_full_melt,
            show_wall_hits=show_wall_hits,
            figsize=figsize,
            box_aspect=box_aspect,
            save=save,
            filename=filename,
        )

    def _plot_power_comparison(
        self,
        powers_kW,
        cases,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        panel_spacing=None,
        save=True,
        filename="particle_power_comparison.png",
    ):
        """
        Plot the same particle cases at several powers.

        Parameters
        ----------
        figsize : tuple or None
            Size of the complete multi-panel figure.

        box_aspect : float or None
            Visual height/width ratio of every torch panel.  For horizontal
            torch plots, try 0.15-0.20 when a taller torch is desired.
            None preserves the true physical aspect ratio.

        panel_spacing : float or None
            Spacing between panels.  For horizontal torch orientation this is
            the GridSpec hspace; for vertical torch orientation it is wspace.
        """
        powers = [self._nearest_power(p) for p in powers_kW]
        vmin, vmax = self._temperature_limits(powers)
        colors = self._line_colors(len(cases))

        if orientation == "horizontal":
            nrows = len(powers)

            if figsize is None:
                figsize = (9.0, 2.0 * nrows + 1.05)

            if panel_spacing is None:
                panel_spacing = 0.30

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                nrows,
                1,
                left=0.075,
                right=0.84,
                bottom=0.075,
                top=0.865,
                hspace=float(panel_spacing),
            )
            axes = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]

        elif orientation == "vertical":
            ncols = len(powers)

            if figsize is None:
                figsize = (3.0 * ncols + 1.0, 7.0)

            if panel_spacing is None:
                panel_spacing = 0.18

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                1,
                ncols,
                left=0.065,
                right=0.87,
                bottom=0.09,
                top=0.84,
                wspace=float(panel_spacing),
            )
            axes = [fig.add_subplot(gs[0, j]) for j in range(ncols)]

        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")

        contour = None

        for panel, (ax, power) in enumerate(zip(axes, powers)):
            _, fields = self._load_fields(power)

            contour = self._temperature_background(
                ax,
                fields,
                vmin,
                vmax,
                orientation,
            )

            if show_streamlines:
                self._streamlines(ax, fields, orientation)

            self._draw_geometry(ax, orientation)

            for color, case in zip(colors, cases):
                result = self.run_case(
                    power,
                    case["diameter_um"],
                    case["r0_mm"],
                )

                self._plot_trajectory(
                    ax,
                    result,
                    label=case["label"],
                    color=color,
                    orientation=orientation,
                    show_full_melt=show_full_melt,
                    show_wall_hits=show_wall_hits,
                )

            self._format_axis(
                ax,
                orientation,
                title=f"({chr(97 + panel)}) {power:g} kW",
                show_ylabel=(panel == 0 or orientation == "horizontal"),
                box_aspect=box_aspect,
            )

            if orientation == "horizontal" and panel < len(axes) - 1:
                ax.set_xlabel("")

            if orientation == "vertical" and panel > 0:
                ax.set_ylabel("")

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.46, 0.98),
            ncol=min(len(labels), 6),
            frameon=False,
            handlelength=2.2,
            columnspacing=1.3,
        )

        self._compact_colorbar(
            fig,
            contour,
            axes,
            vmin=vmin,
            vmax=vmax,
            height_frac=0.70,
            width=0.011,
            pad=0.018,
        )

        if save:
            self._save(fig, filename)

        return fig, np.asarray(axes, dtype=object)

    def plot_diameter_power_comparison(
        self,
        powers_kW,
        diameters_um,
        r0_mm,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        panel_spacing=None,
        save=True,
        filename="diameter_power_comparison.png",
    ):
        cases = [
            {
                "diameter_um": float(dp),
                "r0_mm": float(r0_mm),
                "label": rf"{dp:g} $\mu$m",
            }
            for dp in diameters_um
        ]

        return self._plot_power_comparison(
            powers_kW=powers_kW,
            cases=cases,
            orientation=orientation,
            show_streamlines=show_streamlines,
            show_full_melt=show_full_melt,
            show_wall_hits=show_wall_hits,
            figsize=figsize,
            box_aspect=box_aspect,
            panel_spacing=panel_spacing,
            save=save,
            filename=filename,
        )

    def plot_radial_power_comparison(
        self,
        powers_kW,
        diameter_um,
        radial_positions_mm,
        orientation="horizontal",
        show_streamlines=False,
        show_full_melt=True,
        show_wall_hits=True,
        figsize=None,
        box_aspect=None,
        panel_spacing=None,
        save=True,
        filename="radial_power_comparison.png",
    ):
        cases = [
            {
                "diameter_um": float(diameter_um),
                "r0_mm": float(r0),
                "label": rf"$r_0={r0:g}$ mm",
            }
            for r0 in radial_positions_mm
        ]

        return self._plot_power_comparison(
            powers_kW=powers_kW,
            cases=cases,
            orientation=orientation,
            show_streamlines=show_streamlines,
            show_full_melt=show_full_melt,
            show_wall_hits=show_wall_hits,
            figsize=figsize,
            box_aspect=box_aspect,
            panel_spacing=panel_spacing,
            save=save,
            filename=filename,
        )

    @staticmethod
    def add_particle_state(df, mass_loss_limit_pct=10.0):
        """
        Classify the material/thermal state independently of trajectory fate.

        A reflected particle can therefore still be classified as fully melted.
        """
        df = df.copy()

        def classify(row):
            if str(row.get("fate", "")) == "fully_evaporated":
                return "fully evaporated"

            x_max = float(row["x_max"])
            mass_loss = float(row["mass_loss_pct"])

            if x_max >= 1.0 - 1e-8:
                if mass_loss > float(mass_loss_limit_pct):
                    return "excessive evaporation"
                return "fully melted, acceptable loss"

            if x_max > 1e-8:
                return "partially melted"
            return "unmelted"

        df["particle_state"] = df.apply(classify, axis=1)
        return df

    @staticmethod
    def _particle_state_style():
        order = [
            "unmelted",
            "partially melted",
            "fully melted, acceptable loss",
            "excessive evaporation",
            "fully evaporated",
        ]
        colors = [
            "#d9d9d9",
            "#e69f00",
            "#009e73",
            "#d55e00",
            "#7f0000",
        ]
        return order, colors

    def plot_particle_state_map(
        self,
        df,
        r0_mm,
        mass_loss_limit_pct=10.0,
        show_upstream=True,
        upstream_scope="selected_radius",
        upstream_fraction_threshold=0.50,
        figsize=(7.1, 4.6),
        save=True,
        filename=None,
    ):
        """
        Plot particle thermal/material state over power and particle diameter,
        with an optional hatch overlay identifying upstream particle loss.

        Colour indicates thermal/material state:
            - unmelted
            - partially melted
            - fully melted with acceptable mass loss
            - fully melted with excessive mass loss
            - fully evaporated

        Hatching indicates upstream transport failure.

        Parameters
        ----------
        df : pandas.DataFrame
            Particle parametric-study results.

        r0_mm : float
            Radial injection position used for the particle-state colours.

        mass_loss_limit_pct : float
            Maximum acceptable particle mass loss [%].

        show_upstream : bool
            Overlay upstream-loss information.

        upstream_scope : {"selected_radius", "all_radial_fraction"}
            "selected_radius":
                Hatch a cell when the particle at the selected r0_mm
                escapes through the inlet.

            "all_radial_fraction":
                Hatch a power/diameter cell when at least
                upstream_fraction_threshold of ALL tested radial injection
                positions escape upstream.

        upstream_fraction_threshold : float
            Only used when upstream_scope="all_radial_fraction".

            Example:
                0.50 -> hatch when at least 50% of radial injection
                        positions are lost upstream.

        figsize : tuple
            Figure size.

        save : bool
            Save the figure.

        filename : str or None
            Output filename.

        Returns
        -------
        fig, ax
        """

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        if upstream_scope not in {
            "selected_radius",
            "all_radial_fraction",
        }:
            raise ValueError(
                "upstream_scope must be either "
                "'selected_radius' or 'all_radial_fraction'."
            )

        upstream_fraction_threshold = float(
            upstream_fraction_threshold
        )

        if not 0.0 <= upstream_fraction_threshold <= 1.0:
            raise ValueError(
                "upstream_fraction_threshold must lie between 0 and 1."
            )

        # ------------------------------------------------------------------
        # Particle-state classification
        # ------------------------------------------------------------------

        df_state = self.add_particle_state(
            df,
            mass_loss_limit_pct=mass_loss_limit_pct,
        )

        # State colours correspond specifically to selected r0.
        df_r = df_state[
            np.isclose(
                pd.to_numeric(
                    df_state["r0_mm"],
                    errors="coerce",
                ),
                float(r0_mm),
            )
        ].copy()

        if df_r.empty:
            raise ValueError(
                f"No cases found for r0 = {r0_mm} mm."
            )

        # ------------------------------------------------------------------
        # Robust Boolean helper
        # ------------------------------------------------------------------

        def as_bool(series):

            if pd.api.types.is_bool_dtype(series):
                return (
                    series
                    .fillna(False)
                    .astype(bool)
                )

            if pd.api.types.is_numeric_dtype(series):
                return (
                    pd.to_numeric(
                        series,
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(float)
                    != 0.0
                )

            return (
                series.astype(str)
                .str.strip()
                .str.lower()
                .isin([
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                ])
            )

        # ------------------------------------------------------------------
        # Helper for finding upstream loss
        # ------------------------------------------------------------------

        def add_upstream_column(data):

            data = data.copy()

            if "upstream_lost" in data.columns:

                data["upstream_lost_bool"] = as_bool(
                    data["upstream_lost"]
                )

            elif "exit_inlet" in data.columns:

                data["upstream_lost_bool"] = as_bool(
                    data["exit_inlet"]
                )

            elif "fate" in data.columns:

                data["upstream_lost_bool"] = (
                    data["fate"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("upstream_escape")
                )

            else:

                raise KeyError(
                    "Could not determine upstream loss. "
                    "Expected one of: "
                    "'upstream_lost', 'exit_inlet', or 'fate'."
                )

            return data

        # ------------------------------------------------------------------
        # State map
        # ------------------------------------------------------------------

        pivot = df_r.pivot(
            index="power_kW",
            columns="dp0_um",
            values="particle_state",
        )

        pivot = (
            pivot
            .sort_index()
            .reindex(
                sorted(pivot.columns),
                axis=1,
            )
        )

        order, colors = self._particle_state_style()

        state_to_int = {
            state: i
            for i, state in enumerate(order)
        }

        Z = (
            pivot
            .replace(state_to_int)
            .to_numpy(dtype=float)
        )

        powers = pivot.index.to_numpy(
            dtype=float
        )

        diameters = pivot.columns.to_numpy(
            dtype=float
        )

        power_edges = self._centres_to_edges(
            powers
        )

        diameter_edges = self._centres_to_edges(
            diameters
        )

        cmap = ListedColormap(
            colors
        )

        norm = BoundaryNorm(
            np.arange(
                -0.5,
                len(order) + 0.5,
                1.0,
            ),
            cmap.N,
        )

        # ==================================================================
        # FIGURE
        # ==================================================================

        fig, ax = plt.subplots(
            figsize=figsize
        )

        mesh = ax.pcolormesh(
            diameter_edges,
            power_edges,
            Z,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )

        # ==================================================================
        # UPSTREAM-LOSS OVERLAY
        # ==================================================================

        if show_upstream:

            # --------------------------------------------------------------
            # OPTION 1:
            # Was THIS selected-r0 particle lost upstream?
            # --------------------------------------------------------------

            if upstream_scope == "selected_radius":

                upstream_data = add_upstream_column(
                    df_r
                )

                upstream_table = upstream_data.pivot_table(
                    index="power_kW",
                    columns="dp0_um",
                    values="upstream_lost_bool",
                    aggfunc="max",
                )

                upstream_table = upstream_table.reindex(
                    index=powers,
                    columns=diameters,
                )

                upstream_mask = (
                    upstream_table
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )

                hatch_label = (
                    "Particle lost upstream"
                )

            # --------------------------------------------------------------
            # OPTION 2:
            # Fraction lost upstream across ALL radial injections
            # --------------------------------------------------------------

            else:

                all_data = add_upstream_column(
                    df_state
                )

                upstream_table = all_data.pivot_table(
                    index="power_kW",
                    columns="dp0_um",
                    values="upstream_lost_bool",
                    aggfunc="mean",
                )

                upstream_table = upstream_table.reindex(
                    index=powers,
                    columns=diameters,
                )

                upstream_values = upstream_table.to_numpy(
                    dtype=float
                )

                upstream_mask = (
                    np.isfinite(upstream_values)
                    &
                    (
                        upstream_values
                        >= upstream_fraction_threshold
                    )
                )

                hatch_label = (
                    rf"Upstream loss $\geq$ "
                    rf"{100*upstream_fraction_threshold:.0f}%"
                )

            # --------------------------------------------------------------
            # Hatch individual cells exactly
            # --------------------------------------------------------------

            for i in range(len(powers)):
                for j in range(len(diameters)):

                    if not upstream_mask[i, j]:
                        continue

                    x0 = diameter_edges[j]
                    y0 = power_edges[i]

                    width = (
                        diameter_edges[j + 1]
                        - diameter_edges[j]
                    )

                    height = (
                        power_edges[i + 1]
                        - power_edges[i]
                    )

                    rect = Rectangle(
                        (x0, y0),
                        width,
                        height,
                        facecolor="none",
                        edgecolor="0.15",
                        hatch="////",
                        linewidth=0.0,
                        zorder=5,
                    )

                    ax.add_patch(
                        rect
                    )

        # ==================================================================
        # AXES
        # ==================================================================

        ax.set_xlabel(
            r"Initial particle diameter, "
            r"$d_{p,0}$ [$\mu$m]"
        )

        ax.set_ylabel(
            "Plasma power [kW]"
        )

        ax.set_title(
            rf"$r_0={r0_mm:.2f}$ mm"
        )

        self._format_scalar_axis(
            ax,
            grid=False,
        )

        # ==================================================================
        # STATE COLORBAR
        # ==================================================================

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            ticks=np.arange(
                len(order)
            ),
            pad=0.025,
            shrink=0.78,
            aspect=24,
        )

        cbar.ax.set_yticklabels([
            "unmelted",
            "partially melted",
            rf"fully melted, $\leq${mass_loss_limit_pct:g}% loss",
            rf"fully melted, $>${mass_loss_limit_pct:g}% loss",
            "fully evaporated",
        ])

        cbar.ax.tick_params(
            length=0
        )

        # ==================================================================
        # UPSTREAM LEGEND
        # ==================================================================

        if show_upstream:

            upstream_patch = Patch(
                facecolor="white",
                edgecolor="0.20",
                hatch="////",
                label=hatch_label,
            )

            ax.legend(
                handles=[
                    upstream_patch
                ],
                loc="upper right",
                frameon=True,
                facecolor="white",
                framealpha=0.85,
            )

        # # ==================================================================
        # # EXPLANATORY TEXT
        # # ==================================================================

        # annotation = (
        #     rf"thermal state at $r_0={r0_mm:g}$ mm"
        # )

        # if show_upstream:

        #     if upstream_scope == "selected_radius":

        #         annotation += (
        #             "\n"
        #             "hatched: same trajectory lost upstream"
        #         )

        #     else:

        #         annotation += (
        #             "\n"
        #             + rf"hatched: upstream loss $\geq$ "
        #             + rf"{100*upstream_fraction_threshold:.0f}% "
        #             + "across radial injections"
        #         )

        # ax.text(
        #     0.02,
        #     0.98,
        #     annotation,
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=self.font_size - 1,
        #     bbox=dict(
        #         facecolor="white",
        #         edgecolor="none",
        #         alpha=0.76,
        #         pad=2.5,
        #     ),
        #     zorder=10,
        # )

        fig.tight_layout()

        if filename is None:

            filename = (
                f"particle_state_map_"
                f"r{r0_mm:.2f}mm.png"
            )

        if save:
            self._save(
                fig,
                filename,
            )

        return fig, ax

    # ======================================================================
    # REPRESENTATIVE PARTICLE: TRAJECTORY + THERMAL HISTORY
    # ======================================================================

    def plot_single_particle_diagnostics(
        self,
        power_kW,
        diameter_um,
        r0_mm,
        orientation="horizontal",
        show_streamlines=False,
        figsize=(8.5, 6.0),
        box_aspect=None,
        save=True,
        filename=None,
    ):
        """Link one particle trajectory directly to its thermal history."""
        result = self.run_case(power_kW, diameter_um, r0_mm)
        power, fields = self._load_fields(power_kW)
        h = result["history"]

        if h.shape[1] < 11:
            raise ValueError(
                "Detailed diagnostics require the newer 16-column particle history "
                "containing mass and local gas temperature."
            )

        t_ms = h[:, 0] * 1e3
        Tp = h[:, 5]
        x_liq = h[:, 7]
        mp = h[:, 9]
        Tgas = h[:, 10]
        mass_remaining = 100.0 * mp / mp[0]

        vmin, vmax = self._temperature_limits([power])
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1.35, 1.0],
            left=0.09, right=0.84,
            bottom=0.10, top=0.94,
            hspace=0.42, wspace=0.34,
        )

        ax_traj = fig.add_subplot(gs[0, :])
        contour = self._temperature_background(ax_traj, fields, vmin, vmax, orientation)
        if show_streamlines:
            self._streamlines(ax_traj, fields, orientation)
        self._draw_geometry(ax_traj, orientation)
        self._plot_trajectory(
            ax_traj, result,
            label=rf"{diameter_um:g} $\mu$m",
            color=plt.get_cmap("tab10")(0),
            orientation=orientation,
        )
        self._format_axis(
            ax_traj,
            orientation,
            title=rf"{power:g} kW, $d_{{p,0}}={diameter_um:g}$ $\mu$m, $r_0={r0_mm:g}$ mm",
            box_aspect=box_aspect,
        )
        self._compact_colorbar(
            fig, contour, ax_traj,
            vmin=vmin, vmax=vmax,
            height_frac=0.68,
            width=0.011,
            pad=0.016,
        )

        ax_T = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[1, 1])
        ax_m = fig.add_subplot(gs[1, 2])

        ax_T.plot(t_ms, Tgas, linewidth=1.2, label=r"$T_g$")
        ax_T.plot(t_ms, Tp, linewidth=1.7, label=r"$T_p$")
        ax_T.axhline(Tmp, linewidth=0.8, linestyle="--", color="0.35", label=r"$T_m$")
        ax_T.axhline(Tbp, linewidth=0.8, linestyle=":", color="0.35", label=r"$T_b$")
        ax_T.set_ylabel("Temperature [K]")
        ax_T.legend(frameon=False, ncol=2, handlelength=1.5)

        ax_x.plot(t_ms, x_liq, linewidth=1.7)
        ax_x.axhline(1.0, linewidth=0.8, linestyle="--", color="0.35")
        ax_x.set_ylim(-0.03, 1.05)
        ax_x.set_ylabel(r"Liquid fraction, $x$")

        ax_m.plot(t_ms, mass_remaining, linewidth=1.7)
        ax_m.set_ylabel("Mass remaining [%]")
        ax_m.set_ylim(max(0.0, np.nanmin(mass_remaining) - 5.0), 101.0)

        events = getattr(result["particle"], "wall_events", [])
        wall_times = [event["time_s"] * 1e3 for event in events]

        for ax in (ax_T, ax_x, ax_m):
            for tw in wall_times:
                ax.axvline(tw, color="0.45", linestyle=":", linewidth=0.7, alpha=0.8)
            ax.set_xlabel("Particle time [ms]")
            self._format_scalar_axis(ax, grid=True)

        if filename is None:
            filename = f"particle_diagnostics_{power:g}kW_{diameter_um:g}um_r{r0_mm:g}mm.png"
        if save:
            self._save(fig, filename)
        return fig, (ax_traj, ax_T, ax_x, ax_m)

    # ======================================================================
    # FULL-MELT LOCATION MAP
    # ======================================================================

    def plot_melt_location_map(
        self,
        df,
        agg="median",
        upstream_fraction_threshold=0.50,
        show_upstream_region=True,
        melt_population="all",
        figsize=(6.4, 4.3),
        save=True,
        filename="full_melt_location_map.png",
    ):
        """
        Plot the axial location at which particles first become fully molten,
        with an overlay showing regions of significant upstream particle loss.

        Colour:
            Full-melt location, z_melt [mm], aggregated across radial
            injection positions.

        Hatching:
            Power/diameter combinations for which at least
            upstream_fraction_threshold of the radial injection positions
            are lost through the inlet.

        Parameters
        ----------
        df : pandas.DataFrame
            Particle parametric-study results.

        agg : str
            Aggregation method for z_full_melt_mm.
            Typical choice: "median".

        upstream_fraction_threshold : float
            Fraction of radial injection positions that must escape upstream
            before a power/diameter combination is hatched.
            Example:
                0.50 -> at least 50% of radial injections lost upstream.

        show_upstream_region : bool
            Overlay the upstream-loss region.

        melt_population : {"all", "outlet"}
            "all":
                Calculate melt location using every particle that reaches
                complete melting, irrespective of its final trajectory fate.

            "outlet":
                Calculate melt location only for particles that ultimately
                reach the downstream outlet.

            For diagnosing the mechanism, "all" is recommended because it
            reveals particles that melt successfully before later escaping
            upstream.

        figsize : tuple
            Figure size.

        save : bool
            Save the figure.

        filename : str
            Output filename.

        Returns
        -------
        fig, ax
        """

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        upstream_fraction_threshold = float(
            upstream_fraction_threshold
        )

        if not 0.0 <= upstream_fraction_threshold <= 1.0:
            raise ValueError(
                "upstream_fraction_threshold must lie between 0 and 1."
            )

        if melt_population not in {"all", "outlet"}:
            raise ValueError(
                "melt_population must be 'all' or 'outlet'."
            )

        data = df.copy()

        required = [
            "power_kW",
            "dp0_um",
            "z_full_melt_mm",
        ]

        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise KeyError(
                f"Missing required DataFrame columns: {missing}"
            )

        # ------------------------------------------------------------------
        # Convert numerical columns
        # ------------------------------------------------------------------

        for col in [
            "power_kW",
            "dp0_um",
            "z_full_melt_mm",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        # ------------------------------------------------------------------
        # Robust Boolean conversion
        # ------------------------------------------------------------------

        def as_bool(series):
            if pd.api.types.is_bool_dtype(series):
                return series.fillna(False).astype(bool)

            if pd.api.types.is_numeric_dtype(series):
                return (
                    pd.to_numeric(
                        series,
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(float)
                    != 0.0
                )

            return (
                series.astype(str)
                .str.strip()
                .str.lower()
                .isin([
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                ])
            )

        # ------------------------------------------------------------------
        # Determine upstream loss
        # ------------------------------------------------------------------

        if "upstream_lost" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["upstream_lost"]
            )

        elif "exit_inlet" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["exit_inlet"]
            )

        elif "fate" in data.columns:

            data["upstream_lost_bool"] = (
                data["fate"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("upstream_escape")
            )

        else:

            raise KeyError(
                "Could not determine upstream particle loss. "
                "Expected one of: "
                "'upstream_lost', 'exit_inlet', or 'fate'."
            )

        # ------------------------------------------------------------------
        # Determine downstream outlet recovery
        # ------------------------------------------------------------------

        if "reached_outlet" in data.columns:

            data["reached_outlet_bool"] = as_bool(
                data["reached_outlet"]
            )

        elif "fate" in data.columns:

            data["reached_outlet_bool"] = (
                data["fate"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("outlet")
            )

        else:

            data["reached_outlet_bool"] = (
                ~data["upstream_lost_bool"]
            )

            if "max_steps_reached" in data.columns:
                data["reached_outlet_bool"] &= ~as_bool(
                    data["max_steps_reached"]
                )

        # ==================================================================
        # 1. MELT-LOCATION DATA
        # ==================================================================

        melted = data[
            np.isfinite(
                data["z_full_melt_mm"]
            )
        ].copy()

        if melt_population == "outlet":

            melted = melted[
                melted["reached_outlet_bool"]
            ].copy()

        if melted.empty:
            raise ValueError(
                "No fully melted particles are present "
                "for the selected melt population."
            )

        melt_table = melted.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="z_full_melt_mm",
            aggfunc=agg,
        ).sort_index().sort_index(axis=1)

        # ==================================================================
        # 2. UPSTREAM-LOSS FRACTION
        # ==================================================================

        upstream_table = data.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="upstream_lost_bool",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        # ------------------------------------------------------------------
        # Common grid
        # ------------------------------------------------------------------

        powers = np.sort(
            data["power_kW"]
            .dropna()
            .unique()
            .astype(float)
        )

        diameters = np.sort(
            data["dp0_um"]
            .dropna()
            .unique()
            .astype(float)
        )

        melt_table = melt_table.reindex(
            index=powers,
            columns=diameters,
        )

        upstream_table = upstream_table.reindex(
            index=powers,
            columns=diameters,
        )

        melt_values = melt_table.to_numpy(
            dtype=float
        )

        upstream_values = upstream_table.to_numpy(
            dtype=float
        )

        # ==================================================================
        # FIGURE
        # ==================================================================

        fig, ax = plt.subplots(
            figsize=figsize
        )

        # ------------------------------------------------------------------
        # Melt-location heat map
        # ------------------------------------------------------------------

        mesh = ax.pcolormesh(
            self._centres_to_edges(diameters),
            self._centres_to_edges(powers),
            melt_values,
            shading="flat",
            cmap="viridis",
        )

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            pad=0.025,
            shrink=0.80,
            aspect=24,
        )

        if str(agg).lower() == "median":
            statistic_name = "Median"
        else:
            statistic_name = str(agg).capitalize()

        cbar.set_label(
            statistic_name
            + r" full-melt location, $z_{\mathrm{melt}}$ [mm]"
        )

        # ==================================================================
        # UPSTREAM-LOSS OVERLAY
        # ==================================================================

        if show_upstream_region:

            upstream_mask = (
                np.isfinite(upstream_values)
                &
                (
                    upstream_values
                    >= upstream_fraction_threshold
                )
            )

            if np.any(upstream_mask):

                # Hatched region.
                ax.contourf(
                    diameters,
                    powers,
                    upstream_mask.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["////"],
                    zorder=5,
                )

                # Boundary corresponding exactly to the selected
                # upstream-loss fraction.
                finite_upstream = upstream_values[
                    np.isfinite(upstream_values)
                ]

                if (
                    finite_upstream.size
                    and np.nanmin(finite_upstream)
                    <= upstream_fraction_threshold
                    <= np.nanmax(finite_upstream)
                ):
                    upstream_contour = ax.contour(
                        diameters,
                        powers,
                        upstream_values,
                        levels=[
                            upstream_fraction_threshold
                        ],
                        colors="black",
                        linestyles=":",
                        linewidths=1.15,
                        zorder=6,
                    )

                    try:
                        ax.clabel(
                            upstream_contour,
                            fmt={
                                upstream_fraction_threshold:
                                    rf"{100*upstream_fraction_threshold:.0f}\% upstream"
                            },
                            inline=True,
                            fontsize=self.font_size - 1,
                        )
                    except Exception:
                        pass

        # ==================================================================
        # FORMATTING
        # ==================================================================

        ax.set_xlabel(
            r"Initial particle diameter, "
            r"$d_{p,0}$ [$\mu$m]"
        )

        ax.set_ylabel(
            "Plasma power [kW]"
        )

        self._format_scalar_axis(
            ax,
            grid=False,
        )

        # ------------------------------------------------------------------
        # Annotation
        # ------------------------------------------------------------------

        if melt_population == "all":
            population_text = (
                "Melt location: all fully melted trajectories"
            )
        else:
            population_text = (
                "Melt location: outlet-recovered trajectories"
            )

        if show_upstream_region:

            annotation = (
                population_text
                + "\n"
                + rf"hatched: upstream loss $\geq$ "
                + rf"{100*upstream_fraction_threshold:.0f}%"
            )

        else:

            annotation = population_text

        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=self.font_size - 1,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
                pad=2.5,
            ),
            zorder=10,
        )

        fig.tight_layout()

        if save:
            self._save(
                fig,
                filename,
            )

        return fig, ax


    # ======================================================================
    # PROCESSING ENVELOPE
    # ======================================================================

    def plot_processing_envelope(
        self,
        df,
        required_radial_fraction=0.50,
        mass_loss_limit_pct=10.0,
        upstream_fraction_threshold=None,
        show_thermal_window=True,
        show_upstream_region=True,
        require_outlet=True,
        figsize=(6.6, 4.4),
        save=True,
        filename="processing_envelope.png",
    ):
        """
        Plot the titanium particle processing envelope while explicitly
        accounting for upstream particle loss.

        Three quantities are distinguished:

        1. Thermal viability
        A trajectory is thermally acceptable when:
            x_max = 1
            mass_loss_pct <= mass_loss_limit_pct
            particle is not fully evaporated

        2. Recoverable / successful processing
        A trajectory is successful when it is thermally acceptable and,
        when require_outlet=True, reaches the torch outlet.

        3. Upstream loss
        The fraction of radial injection positions that leave through
        the inlet is calculated independently.

        For each (power, diameter), the criterion must be satisfied for at
        least required_radial_fraction of the tested radial injections.

        Parameters
        ----------
        df : pandas.DataFrame
            Particle parametric-study results.

        required_radial_fraction : float
            Minimum fraction of radial injection positions that must satisfy
            the thermal/success criterion. Example:
                0.50 -> at least 50% of radial injections.

        mass_loss_limit_pct : float
            Maximum acceptable particle mass loss [%].

        upstream_fraction_threshold : float or None
            Fraction of radial injections that must be upstream-lost before
            the power/diameter point is marked as an upstream-loss region.
            If None, required_radial_fraction is used.

        show_thermal_window : bool
            Show the thermal-only processing envelope.

        show_upstream_region : bool
            Hatch regions where upstream loss exceeds the selected fraction.

        require_outlet : bool
            If True, the final successful envelope requires the particle to
            reach the torch outlet. Wall reflection itself is NOT failure.

        Returns
        -------
        fig, ax
        """

        required_radial_fraction = float(required_radial_fraction)

        if not 0.0 <= required_radial_fraction <= 1.0:
            raise ValueError(
                "required_radial_fraction must lie between 0 and 1."
            )

        if upstream_fraction_threshold is None:
            upstream_fraction_threshold = required_radial_fraction

        upstream_fraction_threshold = float(upstream_fraction_threshold)

        if not 0.0 <= upstream_fraction_threshold <= 1.0:
            raise ValueError(
                "upstream_fraction_threshold must lie between 0 and 1."
            )

        # ------------------------------------------------------------------
        # Prepare data
        # ------------------------------------------------------------------

        data = df.copy()

        required_columns = [
            "power_kW",
            "dp0_um",
            "x_max",
            "mass_loss_pct",
        ]

        missing = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing:
            raise KeyError(
                f"Missing required DataFrame columns: {missing}"
            )

        for col in [
            "power_kW",
            "dp0_um",
            "x_max",
            "mass_loss_pct",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        data = data.dropna(
            subset=[
                "power_kW",
                "dp0_um",
                "x_max",
                "mass_loss_pct",
            ]
        )

        if data.empty:
            raise ValueError(
                "No valid particle results are present."
            )

        # ------------------------------------------------------------------
        # Helper for Boolean CSV columns
        # ------------------------------------------------------------------

        def as_bool(series):
            """
            Robust conversion because pandas may read Boolean CSV columns
            as bool, int, float or string.
            """

            if pd.api.types.is_bool_dtype(series):
                return series.fillna(False).astype(bool)

            if pd.api.types.is_numeric_dtype(series):
                return (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(0)
                    .astype(float)
                    != 0.0
                )

            return (
                series.astype(str)
                .str.strip()
                .str.lower()
                .isin([
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                ])
            )

        # ------------------------------------------------------------------
        # Determine upstream loss
        # ------------------------------------------------------------------

        if "upstream_lost" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["upstream_lost"]
            )

        elif "exit_inlet" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["exit_inlet"]
            )

        elif "fate" in data.columns:

            data["upstream_lost_bool"] = (
                data["fate"]
                .astype(str)
                .str.lower()
                .eq("upstream_escape")
            )

        else:

            raise KeyError(
                "Could not determine upstream loss. "
                "Expected one of: "
                "'upstream_lost', 'exit_inlet', or 'fate'."
            )

        # ------------------------------------------------------------------
        # Determine whether particle reaches the outlet
        # ------------------------------------------------------------------

        if "reached_outlet" in data.columns:

            data["reached_outlet_bool"] = as_bool(
                data["reached_outlet"]
            )

        elif "fate" in data.columns:

            data["reached_outlet_bool"] = (
                data["fate"]
                .astype(str)
                .str.lower()
                .eq("outlet")
            )

        else:
            # Fallback if using an older CSV.
            data["reached_outlet_bool"] = (
                ~data["upstream_lost_bool"]
            )

            if "max_steps_reached" in data.columns:
                data["reached_outlet_bool"] &= ~as_bool(
                    data["max_steps_reached"]
                )

        # ------------------------------------------------------------------
        # THERMAL ACCEPTABILITY
        # ------------------------------------------------------------------

        melted = (
            data["x_max"].astype(float)
            >= 1.0 - 1e-8
        )

        acceptable_mass = (
            data["mass_loss_pct"].astype(float)
            <= float(mass_loss_limit_pct)
        )

        if "fate" in data.columns:
            not_evaporated = (
                data["fate"]
                .astype(str)
                .str.lower()
                .ne("fully_evaporated")
            )
        else:
            not_evaporated = (
                data["mass_loss_pct"].astype(float)
                < 100.0 - 1e-8
            )

        data["thermal_acceptable"] = (
            melted
            & acceptable_mass
            & not_evaporated
        )

        # ------------------------------------------------------------------
        # FINAL PROCESSING SUCCESS
        # ------------------------------------------------------------------

        if require_outlet:

            data["processing_success"] = (
                data["thermal_acceptable"]
                & data["reached_outlet_bool"]
            )

        else:

            data["processing_success"] = (
                data["thermal_acceptable"]
                & ~data["upstream_lost_bool"]
            )

            if "max_steps_reached" in data.columns:
                data["processing_success"] &= ~as_bool(
                    data["max_steps_reached"]
                )

        # ------------------------------------------------------------------
        # Fractions across radial injection positions
        # ------------------------------------------------------------------

        thermal_fraction = data.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="thermal_acceptable",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        success_fraction = data.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="processing_success",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        upstream_fraction = data.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="upstream_lost_bool",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        # Use one common power-diameter grid
        powers = np.sort(
            data["power_kW"].unique().astype(float)
        )

        diameters = np.sort(
            data["dp0_um"].unique().astype(float)
        )

        thermal_fraction = thermal_fraction.reindex(
            index=powers,
            columns=diameters,
        )

        success_fraction = success_fraction.reindex(
            index=powers,
            columns=diameters,
        )

        upstream_fraction = upstream_fraction.reindex(
            index=powers,
            columns=diameters,
        )

        # ------------------------------------------------------------------
        # Determine lower/upper envelope bounds
        # ------------------------------------------------------------------

        def envelope_bounds(table, threshold):

            lower = []
            upper = []

            for power in powers:

                row = table.loc[power]

                values = row.to_numpy(dtype=float)

                ok = (
                    np.isfinite(values)
                    & (values >= float(threshold))
                )

                acceptable_diameters = (
                    diameters[ok]
                )

                if acceptable_diameters.size == 0:

                    lower.append(np.nan)
                    upper.append(np.nan)

                else:

                    lower.append(
                        float(
                            np.min(
                                acceptable_diameters
                            )
                        )
                    )

                    upper.append(
                        float(
                            np.max(
                                acceptable_diameters
                            )
                        )
                    )

            return (
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
            )

        thermal_min, thermal_max = envelope_bounds(
            thermal_fraction,
            required_radial_fraction,
        )

        success_min, success_max = envelope_bounds(
            success_fraction,
            required_radial_fraction,
        )

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=figsize
        )

        # ==============================================================
        # Thermal-only window
        # ==============================================================

        if show_thermal_window:

            thermal_valid = (
                np.isfinite(thermal_min)
                & np.isfinite(thermal_max)
            )

            ax.fill_between(
                powers,
                thermal_min,
                thermal_max,
                where=thermal_valid,
                interpolate=False,
                facecolor="0.85",
                alpha=0.90,
                linewidth=0,
                label="Thermally viable",
                zorder=1,
            )

            ax.plot(
                powers,
                thermal_min,
                linestyle="--",
                linewidth=1.1,
                color="0.35",
                zorder=3,
            )

            ax.plot(
                powers,
                thermal_max,
                linestyle="--",
                linewidth=1.1,
                color="0.35",
                zorder=3,
            )

        # ==============================================================
        # Final downstream/recoverable processing window
        # ==============================================================

        success_valid = (
            np.isfinite(success_min)
            & np.isfinite(success_max)
        )

        ax.fill_between(
            powers,
            success_min,
            success_max,
            where=success_valid,
            interpolate=False,
            facecolor="0.45",
            alpha=0.7,
            linewidth=0,
            label="Spheroidization success",
            zorder=2,
        )

        ax.plot(
            powers,
            success_min,
            marker="o",
            markersize=4.0,
            linewidth=1.45,
            color="black",
            label="Successful lower bound",
            zorder=5,
        )

        ax.plot(
            powers,
            success_max,
            marker="s",
            markersize=4.0,
            linewidth=1.45,
            color="black",
            label="Successful upper bound",
            zorder=5,
        )

        # ==============================================================
        # Upstream-loss region
        # ==============================================================

        if show_upstream_region:

            upstream_values = (
                upstream_fraction
                .to_numpy(dtype=float)
                .T
            )

            upstream_mask = (
                np.isfinite(upstream_values)
                & (
                    upstream_values
                    >= upstream_fraction_threshold
                )
            )

            if np.any(upstream_mask):

                P, D = np.meshgrid(
                    powers,
                    diameters,
                )

                # Hatching only -- no solid colour.
                ax.contourf(
                    P,
                    D,
                    upstream_mask.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["////"],
                    zorder=4,
                )

                # Boundary of upstream-loss region
                try:
                    ax.contour(
                        P,
                        D,
                        upstream_values,
                        levels=[
                            upstream_fraction_threshold
                        ],
                        colors="0.20",
                        linestyles=":",
                        linewidths=1.1,
                        zorder=5,
                    )
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Formatting
        # ------------------------------------------------------------------

        ax.set_xlabel(
            "Plasma power [kW]"
        )

        ax.set_ylabel(
            r"Initial particle diameter, "
            r"$d_{p,0}$ [$\mu$m]"
        )

        self._format_scalar_axis(
            ax,
            grid=True,
        )

        # ------------------------------------------------------------------
        # Criteria annotation
        # ------------------------------------------------------------------

        # annotation = (
        #     rf"processing success $\geq$ "
        #     rf"{100 * required_radial_fraction:.0f}%"
        #     + "\n"
        #     + rf"mass loss $\leq$ "
        #     rf"{mass_loss_limit_pct:g}%"
        # )

        # if show_upstream_region:

        #     annotation += (
        #         "\n"
        #         + rf"hatched: upstream loss $\geq$ "
        #         rf"{100 * upstream_fraction_threshold:.0f}%"
        #     )

        # ax.text(
        #     0.02,
        #     0.98,
        #     annotation,
        #     transform=ax.transAxes,
        #     va="top",
        #     ha="left",
        #     fontsize=self.font_size - 1,
        #     bbox=dict(
        #         facecolor="white",
        #         edgecolor="none",
        #         alpha=0.78,
        #         pad=2.5,
        #     ),
        #     zorder=10,
        # )

        # ------------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------------

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if show_upstream_region:

            upstream_patch = Patch(
                facecolor="white",
                edgecolor="0.25",
                hatch="////",
                label=(
                    rf"Upstream loss "
                    rf"$\geq$ "
                    rf"{100 * upstream_fraction_threshold:.0f}%"
                ),
            )

            handles.append(
                upstream_patch
            )

            labels.append(
                upstream_patch.get_label()
            )

        ax.legend(
            handles,
            labels,
            frameon=False,
            loc="best",
        )

        fig.tight_layout()

        if save:
            self._save(
                fig,
                filename,
            )

        return fig, ax

    # ======================================================================
    # MASS-LOSS MAP + UPSTREAM LOSS
    # ======================================================================

    def plot_mass_loss_map(
        self,
        df,
        required_radial_fraction=0.50,
        mass_loss_limit_pct=10.0,
        upstream_fraction_threshold=None,
        show_upstream_region=True,
        levels=21,
        cmap="magma",
        vmin=0.0,
        vmax=100.0,
        show_limit=True,
        figsize=(6.4, 4.3),
        save=True,
        filename="mass_loss_map.png",
    ):
        """
        Contour map of particle mass loss over plasma power and initial
        particle diameter, with an overlay showing significant upstream loss.

        Colour:
            Particle mass loss aggregated across radial injection positions.

            required_radial_fraction = 0.50 -> median mass loss
            required_radial_fraction = 0.75 -> 75th-percentile mass loss

        White contour:
            Selected mass-loss criterion.

        Hatching:
            Power/diameter combinations for which at least
            upstream_fraction_threshold of radial injections escape upstream.

        Parameters
        ----------
        df : pandas.DataFrame
            Particle parametric-study results.

        required_radial_fraction : float
            Quantile used to aggregate mass loss across radial injections.
            0.50 gives the median.

        mass_loss_limit_pct : float
            Material-loss threshold shown as the white contour.

        upstream_fraction_threshold : float or None
            Fraction of radial injection positions that must be lost upstream
            before the cell is hatched.

            If None, required_radial_fraction is used. Thus with the usual
            call using 0.50, hatching indicates >=50% upstream loss.

        show_upstream_region : bool
            Show the upstream-loss hatching.

        levels : int
            Number of filled mass-loss contour levels.

        cmap : str
            Colormap for particle mass loss.

        vmin, vmax : float
            Mass-loss colour limits [%].

        show_limit : bool
            Show the selected mass-loss threshold as a white contour.

        figsize : tuple
            Figure size.

        save : bool
            Save figure.

        filename : str
            Output filename.

        Returns
        -------
        fig, ax
        """

        # ==================================================================
        # SETTINGS
        # ==================================================================

        q = float(required_radial_fraction)

        if not 0.0 <= q <= 1.0:
            raise ValueError(
                "required_radial_fraction must lie between 0 and 1."
            )

        if upstream_fraction_threshold is None:
            upstream_fraction_threshold = q

        upstream_fraction_threshold = float(
            upstream_fraction_threshold
        )

        if not 0.0 <= upstream_fraction_threshold <= 1.0:
            raise ValueError(
                "upstream_fraction_threshold must lie between 0 and 1."
            )

        if float(vmax) <= float(vmin):
            raise ValueError(
                "vmax must be greater than vmin."
            )

        # ==================================================================
        # PREPARE DATA
        # ==================================================================

        data = df.copy()

        required = [
            "power_kW",
            "dp0_um",
            "mass_loss_pct",
        ]

        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise KeyError(
                f"Missing required DataFrame columns: {missing}"
            )

        for col in required:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        data = data.dropna(
            subset=required
        )

        if data.empty:
            raise ValueError(
                "No finite mass-loss data are present "
                "in the supplied results."
            )

        # ==================================================================
        # ROBUST BOOLEAN CONVERSION
        # ==================================================================

        def as_bool(series):

            if pd.api.types.is_bool_dtype(series):

                return (
                    series
                    .fillna(False)
                    .astype(bool)
                )

            if pd.api.types.is_numeric_dtype(series):

                return (
                    pd.to_numeric(
                        series,
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(float)
                    != 0.0
                )

            return (
                series
                .astype(str)
                .str.strip()
                .str.lower()
                .isin([
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                ])
            )

        # ==================================================================
        # DETERMINE UPSTREAM LOSS
        # ==================================================================

        if "upstream_lost" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["upstream_lost"]
            )

        elif "exit_inlet" in data.columns:

            data["upstream_lost_bool"] = as_bool(
                data["exit_inlet"]
            )

        elif "fate" in data.columns:

            data["upstream_lost_bool"] = (
                data["fate"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("upstream_escape")
            )

        else:

            if show_upstream_region:
                raise KeyError(
                    "Could not determine upstream particle loss. "
                    "Expected one of: "
                    "'upstream_lost', 'exit_inlet', or 'fate'."
                )

            data["upstream_lost_bool"] = False

        # ==================================================================
        # MASS-LOSS QUANTILE ACROSS RADIAL INJECTIONS
        # ==================================================================

        mass_table = (
            data
            .groupby(
                ["power_kW", "dp0_um"]
            )["mass_loss_pct"]
            .quantile(q)
            .unstack("dp0_um")
            .sort_index()
            .sort_index(axis=1)
        )

        # ==================================================================
        # UPSTREAM-LOSS FRACTION ACROSS RADIAL INJECTIONS
        # ==================================================================

        upstream_table = data.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="upstream_lost_bool",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        # ==================================================================
        # COMMON GRID
        # ==================================================================

        powers = np.sort(
            data["power_kW"]
            .unique()
            .astype(float)
        )

        diameters = np.sort(
            data["dp0_um"]
            .unique()
            .astype(float)
        )

        mass_table = mass_table.reindex(
            index=powers,
            columns=diameters,
        )

        upstream_table = upstream_table.reindex(
            index=powers,
            columns=diameters,
        )

        mass_loss = mass_table.to_numpy(
            dtype=float
        )

        upstream_values = upstream_table.to_numpy(
            dtype=float
        )

        if (
            powers.size < 2
            or diameters.size < 2
        ):
            raise ValueError(
                "At least two power levels and two particle "
                "diameters are required for a contour map."
            )

        # ==================================================================
        # MASS-LOSS CONTOUR
        # ==================================================================

        mass_loss_plot = np.clip(
            mass_loss,
            float(vmin),
            float(vmax),
        )

        contour_levels = np.linspace(
            float(vmin),
            float(vmax),
            int(levels),
        )

        fig, ax = plt.subplots(
            figsize=figsize
        )

        contour = ax.contourf(
            diameters,
            powers,
            mass_loss_plot,
            levels=contour_levels,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            extend="max",
            zorder=1,
        )

        # ==================================================================
        # COLOURBAR
        # ==================================================================

        cbar = fig.colorbar(
            contour,
            ax=ax,
            pad=0.025,
            shrink=0.82,
            aspect=24,
        )

        cbar.set_label(
            "Particle mass loss [%]"
        )

        cbar.locator = MaxNLocator(
            nbins=6
        )

        cbar.update_ticks()

        # ==================================================================
        # MASS-LOSS LIMIT
        # ==================================================================

        if show_limit:

            finite_mass = mass_loss[
                np.isfinite(mass_loss)
            ]

            limit = float(
                mass_loss_limit_pct
            )

            if (
                finite_mass.size
                and np.nanmin(finite_mass) <= limit
                <= np.nanmax(finite_mass)
            ):

                loss_contour = ax.contour(
                    diameters,
                    powers,
                    mass_loss,
                    levels=[limit],
                    colors="white",
                    linewidths=1.4,
                    zorder=7,
                )

                try:
                    ax.clabel(
                        loss_contour,
                        fmt={
                            limit:
                                rf"{limit:g}%"
                        },
                        inline=True,
                        fontsize=self.font_size - 1,
                    )

                except Exception:
                    pass

        # ==================================================================
        # UPSTREAM-LOSS HATCHING
        # ==================================================================

        if show_upstream_region:

            upstream_mask = (
                np.isfinite(upstream_values)
                &
                (
                    upstream_values
                    >= upstream_fraction_threshold
                )
            )

            if np.any(upstream_mask):

                # ----------------------------------------------------------
                # Hatched area
                # ----------------------------------------------------------

                # ax.contourf(
                #     diameters,
                #     powers,
                #     upstream_mask.astype(float),
                #     levels=[0.5, 1.5],
                #     colors="none",
                #     hatches=["////"],
                #     zorder=5,
                # )
                hatched = ax.contourf(
                    diameters,
                    powers,
                    upstream_mask.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["////"],
                    zorder=1,
                )

                for collection in hatched.collections:
                    collection.set_edgecolor("grey")
                    collection.set_linewidth(0.0)

                # ----------------------------------------------------------
                # Boundary of upstream-loss region
                # ----------------------------------------------------------

                finite_upstream = upstream_values[
                    np.isfinite(upstream_values)
                ]

                if (
                    finite_upstream.size
                    and np.nanmin(finite_upstream)
                    <= upstream_fraction_threshold
                    <= np.nanmax(finite_upstream)
                ):

                    upstream_contour = ax.contour(
                        diameters,
                        powers,
                        upstream_values,
                        levels=[
                            upstream_fraction_threshold
                        ],
                        colors="white",
                        linestyles=":",
                        linewidths=1.15,
                        zorder=7,
                    )

                    try:
                        ax.clabel(
                            upstream_contour,
                            fmt={
                                upstream_fraction_threshold:
                                    rf"{100 * upstream_fraction_threshold:.0f}% upstream"
                            },
                            inline=True,
                            fontsize=self.font_size - 1,
                        )

                    except Exception:
                        pass

        # ==================================================================
        # FORMATTING
        # ==================================================================

        ax.set_xlabel(
            r"Initial particle diameter, "
            r"$d_{p,0}$ [$\mu$m]"
        )

        ax.set_ylabel(
            "Plasma power [kW]"
        )

        self._format_scalar_axis(
            ax,
            grid=False,
        )

        # ==================================================================
        # ANNOTATION
        # ==================================================================

        # if np.isclose(q, 0.50):

        #     stat_text = (
        #         "Median mass loss across radial injections"
        #     )

        # else:

        #     stat_text = (
        #         rf"{100*q:g}th-percentile mass loss "
        #         "across radial injections"
        #     )

        # if show_upstream_region:

        #     stat_text += (
        #         "\n"
        #         + rf"hatched: upstream loss $\geq$ "
        #         + rf"{100 * upstream_fraction_threshold:.0f}%"
        #     )

        # ax.text(
        #     0.02,
        #     0.98,
        #     stat_text,
        #     transform=ax.transAxes,
        #     va="top",
        #     ha="left",
        #     fontsize=self.font_size - 1,
        #     bbox=dict(
        #         facecolor="white",
        #         edgecolor="none",
        #         alpha=0.78,
        #         pad=2.5,
        #     ),
        #     zorder=10,
        # )

        # ==================================================================
        # UPSTREAM LEGEND
        # ==================================================================

        if show_upstream_region:

            upstream_patch = Patch(
                facecolor="white",
                edgecolor="0.20",
                hatch="////",
                label=(
                    rf"Upstream loss $\geq$ "
                    rf"{100 * upstream_fraction_threshold:.0f}%"
                ),
            )

            ax.legend(
                handles=[
                    upstream_patch
                ],
                loc="upper right",
                frameon=True,
                facecolor="white",
                framealpha=0.88,
            )

        fig.tight_layout()

        if save:

            self._save(
                fig,
                filename,
            )

        return fig, ax

    # ======================================================================
    # POWER-DEPENDENT PARTICLE RESPONSE
    # ======================================================================

    def plot_particle_metric_vs_power(
        self,
        df,
        metric,
        diameters_um=(50, 100, 150, 200, 300),
        radial_quantile=0.50,
        show_spread=True,
        spread_quantiles=(0.25, 0.75),
        ylabel=None,
        reference_lines=True,
        mass_loss_limit_pct=10.0,
        figsize=(6.2, 4.1),
        save=True,
        filename=None,
    ):
        """
        Plot a particle-response metric against plasma power for selected
        particle diameters, aggregating over the radial injection positions.

        Parameters
        ----------
        df : pandas.DataFrame
            Parametric-study results returned by ParticleParametricStudy.run().
        metric : str
            Column to plot. Recommended values are:
                'hot_time_ms'
                'residence_time_ms'
                'thermal_exposure_Ks'
                'Tp_max_K'
                'mass_loss_pct'
                'Urel_max_m_s'
                't_full_melt_ms'
        diameters_um : sequence
            Particle diameters to include.
        radial_quantile : float
            Quantile across radial injection positions. 0.50 gives the median.
        show_spread : bool
            If True, shade the interval defined by spread_quantiles.
        spread_quantiles : tuple(float, float)
            Lower and upper radial-injection quantiles used for the shaded band.
            The default (0.25, 0.75) is the interquartile range.
        reference_lines : bool
            Add physically useful reference lines where appropriate:
            Tm/Tb for Tp_max_K and the selected mass-loss limit for mass_loss_pct.
        """
        if metric not in df.columns:
            raise KeyError(f"'{metric}' is not present in the supplied DataFrame.")

        q = float(radial_quantile)
        qlo, qhi = map(float, spread_quantiles)
        if not 0.0 <= q <= 1.0:
            raise ValueError("radial_quantile must lie between 0 and 1.")
        if not 0.0 <= qlo <= qhi <= 1.0:
            raise ValueError("spread_quantiles must satisfy 0 <= qlo <= qhi <= 1.")

        data = df.copy()
        for col in ["power_kW", "dp0_um", metric]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["power_kW", "dp0_um", metric])
        if data.empty:
            raise ValueError(f"No finite data are available for '{metric}'.")

        metric_labels = {
            "hot_time_ms": None,
            "residence_time_ms": "Particle residence time [ms]",
            "thermal_exposure_Ks": r"Gas thermal exposure, $\int\max(T_g-T_m,0)\,dt$ [K s]",
            "Tp_max_K": r"Maximum particle temperature, $T_{p,\max}$ [K]",
            "mass_loss_pct": "Particle mass loss [%]",
            "Urel_max_m_s": r"Maximum slip velocity, $U_{rel,\max}$ [m s$^{-1}$]",
            "t_full_melt_ms": "Time to complete melting [ms]",
        }
        hot_T = float(getattr(self.study, "hot_temperature", 8000.0))
        if metric == "hot_time_ms":
            metric_labels[metric] = rf"Time in $T_g>{hot_T:g}$ K [ms]"
        if ylabel is None:
            ylabel = metric_labels.get(metric, metric.replace("_", " "))

        available_d = np.sort(data["dp0_um"].unique().astype(float))
        selected = []
        for requested in diameters_um:
            if available_d.size == 0:
                continue
            actual = float(available_d[np.argmin(np.abs(available_d - float(requested)))])
            if not any(np.isclose(actual, x) for x in selected):
                selected.append(actual)

        fig, ax = plt.subplots(figsize=figsize)
        colors = self._line_colors(len(selected))

        for color, dp in zip(colors, selected):
            d = data[np.isclose(data["dp0_um"].astype(float), dp)]
            grouped = d.groupby("power_kW")[metric]
            centre = grouped.quantile(q).sort_index()
            powers = centre.index.to_numpy(float)
            values = centre.to_numpy(float)

            ax.plot(
                powers, values,
                marker="o", markersize=4.2,
                linewidth=1.45,
                color=color,
                label=rf"{dp:g} $\mu$m",
            )

            if show_spread:
                lower = grouped.quantile(qlo).reindex(centre.index).to_numpy(float)
                upper = grouped.quantile(qhi).reindex(centre.index).to_numpy(float)
                ax.fill_between(powers, lower, upper, color=color, alpha=0.14, linewidth=0)

        if reference_lines and metric == "mass_loss_pct":
            ax.axhline(
                float(mass_loss_limit_pct),
                linestyle="--", linewidth=1.0, color="0.25",
                label=rf"{mass_loss_limit_pct:g}\% mass-loss limit",
            )
        elif reference_lines and metric == "Tp_max_K":
            ax.axhline(Tmp, linestyle="--", linewidth=0.95, color="0.25", label=r"$T_m$")
            ax.axhline(Tbp, linestyle=":", linewidth=1.05, color="0.25", label=r"$T_b$")

        ax.set_xlabel("Plasma power [kW]")
        ax.set_ylabel(ylabel)
        self._format_scalar_axis(ax, grid=True)
        ax.legend(frameon=False, ncol=2)

        if np.isclose(q, 0.50):
            centre_text = "Median across radial injections"
        else:
            centre_text = rf"{100*q:g}th percentile across radial injections"
        if show_spread:
            centre_text += rf"\nshading: {100*qlo:g}-{100*qhi:g}th percentile"
        ax.text(
            0.02, 0.98, centre_text,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=self.font_size - 1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.5),
        )

        fig.tight_layout()
        if filename is None:
            filename = f"{metric}_vs_power.png"
        if save:
            self._save(fig, filename)
        return fig, ax

    def plot_particle_power_diagnostics(
        self,
        df,
        diameters_um=(50, 100, 150, 200, 300),
        radial_quantile=0.50,
        show_spread=True,
        spread_quantiles=(0.25, 0.75),
        mass_loss_limit_pct=10.0,
        figsize=(10.8, 6.8),
        save=True,
        filename="particle_power_diagnostics.png",
    ):
        """
        Six-panel diagnostic figure for testing the mechanism behind the
        power-dependent particle processing window.

        Panels show, from left to right and top to bottom:
            1. hot-plasma residence time
            2. total particle residence time
            3. gas thermal exposure
            4. maximum particle temperature
            5. particle mass loss
            6. maximum particle-gas slip velocity

        Each line is the selected radial quantile for one particle diameter;
        the optional shading shows variation across radial injection position.
        """
        metrics = [
            ("hot_time_ms", None),
            ("residence_time_ms", "Particle residence time [ms]"),
            ("thermal_exposure_Ks", r"Gas thermal exposure [K s]"),
            ("Tp_max_K", r"Maximum particle temperature, $T_{p,\max}$ [K]"),
            ("mass_loss_pct", "Particle mass loss [%]"),
            ("Urel_max_m_s", r"Maximum slip velocity, $U_{rel,\max}$ [m s$^{-1}$]"),
        ]

        q = float(radial_quantile)
        qlo, qhi = map(float, spread_quantiles)
        data = df.copy()
        needed = ["power_kW", "dp0_um"] + [m for m, _ in metrics]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required DataFrame columns: {missing}")
        for col in needed:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        available_d = np.sort(data["dp0_um"].dropna().unique().astype(float))
        selected = []
        for requested in diameters_um:
            actual = float(available_d[np.argmin(np.abs(available_d - float(requested)))])
            if not any(np.isclose(actual, x) for x in selected):
                selected.append(actual)
        colors = self._line_colors(len(selected))

        hot_T = float(getattr(self.study, "hot_temperature", 8000.0))
        metrics[0] = ("hot_time_ms", rf"Time in $T_g>{hot_T:g}$ K [ms]")

        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
        axes = np.asarray(axes).ravel()

        for ax, (metric, ylabel) in zip(axes, metrics):
            dmetric = data.dropna(subset=["power_kW", "dp0_um", metric])
            for color, dp in zip(colors, selected):
                d = dmetric[np.isclose(dmetric["dp0_um"].astype(float), dp)]
                grouped = d.groupby("power_kW")[metric]
                centre = grouped.quantile(q).sort_index()
                powers = centre.index.to_numpy(float)
                values = centre.to_numpy(float)

                ax.plot(
                    powers, values,
                    marker="o", markersize=3.5,
                    linewidth=1.25,
                    color=color,
                    label=rf"{dp:g} $\mu$m",
                )
                if show_spread:
                    lower = grouped.quantile(qlo).reindex(centre.index).to_numpy(float)
                    upper = grouped.quantile(qhi).reindex(centre.index).to_numpy(float)
                    ax.fill_between(powers, lower, upper, color=color, alpha=0.12, linewidth=0)

            if metric == "mass_loss_pct":
                ax.axhline(float(mass_loss_limit_pct), linestyle="--", linewidth=0.9, color="0.25")
            elif metric == "Tp_max_K":
                ax.axhline(Tmp, linestyle="--", linewidth=0.8, color="0.25")
                ax.axhline(Tbp, linestyle=":", linewidth=0.9, color="0.25")

            ax.set_ylabel(ylabel)
            self._format_scalar_axis(ax, grid=True)

        for ax in axes[3:]:
            ax.set_xlabel("Plasma power [kW]")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(selected), frameon=False, bbox_to_anchor=(0.5, 1.01))

        if np.isclose(q, 0.50):
            stat = "median"
        else:
            stat = f"{100*q:g}th percentile"
        if show_spread:
            fig.text(
                0.01, 0.01,
                f"Lines: {stat} across radial injections; shading: {100*qlo:g}-{100*qhi:g}th percentile.",
                fontsize=self.font_size - 1,
            )

        fig.tight_layout(rect=(0, 0.035, 1, 0.94))
        if save:
            self._save(fig, filename)
        return fig, axes

    # ======================================================================
    # RADIAL FOCUSING / DISPERSION
    # ======================================================================

    def plot_exit_radius_response(
        self,
        df,
        power_kW,
        diameters_um,
        figsize=(6.2, 4.1),
        save=True,
        filename=None,
    ):
        """Outlet radius versus injection radius; diagonal means no radial migration."""
        powers = np.sort(df["power_kW"].unique().astype(float))
        p_use = powers[np.argmin(np.abs(powers - float(power_kW)))]
        subset = df[np.isclose(df["power_kW"].astype(float), p_use)]

        fig, ax = plt.subplots(figsize=figsize)
        for dp in diameters_um:
            ds = subset[np.isclose(subset["dp0_um"].astype(float), float(dp))].sort_values("r0_mm")
            ds = ds[ds["fate"].astype(str) == "outlet"]
            if not ds.empty:
                ax.plot(ds["r0_mm"], ds["r_exit_mm"], marker="o", label=rf"{dp:g} $\mu$m")

        rvals = subset["r0_mm"].to_numpy(float)
        if rvals.size:
            lo, hi = np.nanmin(rvals), np.nanmax(rvals)
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, color="0.35", label=r"$r_{exit}=r_0$")

        ax.set_xlabel(r"Injection radius, $r_0$ [mm]")
        ax.set_ylabel(r"Outlet radius, $r_{exit}$ [mm]")
        ax.set_title(f"{p_use:g} kW")
        self._format_scalar_axis(ax, grid=True)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()

        if filename is None:
            filename = f"exit_radius_response_{p_use:g}kW.png"
        if save:
            self._save(fig, filename)
        return fig, ax

    # ======================================================================
    # WALL-CONTACT FRACTION
    # ======================================================================

    def plot_wall_contact_fraction_map(
        self,
        df,
        figsize=(6.2, 4.1),
        save=True,
        filename="wall_contact_fraction_map.png",
    ):
        """Fraction of injection radii that touch a wall for each power/diameter."""
        table = df.pivot_table(
            index="power_kW",
            columns="dp0_um",
            values="hit_wall",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        powers = table.index.to_numpy(float)
        diameters = table.columns.to_numpy(float)

        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.pcolormesh(
            self._centres_to_edges(diameters),
            self._centres_to_edges(powers),
            table.to_numpy(float),
            shading="flat",
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.025, shrink=0.80, aspect=24)
        cbar.set_label("Wall-contact fraction")

        ax.set_xlabel(r"Initial particle diameter, $d_{p,0}$ [$\mu$m]")
        ax.set_ylabel("Plasma power [kW]")
        self._format_scalar_axis(ax, grid=False)
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ======================================================================
    # SAVE
    # ======================================================================

    def _save(self, fig, filename):
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {path}")

        if self.save_pdf:
            pdf_path = path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

            