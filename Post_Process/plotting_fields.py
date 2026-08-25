import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

load_file_name = '161_steady_to_delete'


def main():
    state_d = load(load_file_name)

    fig, ax, artist = plot_state_grid_field(
        state_d,
        field_name="p",
        unit="mm",
        Lr_carrier=0.0037,
        t_carrier=0.0020,
        Lr_sheath=0.0188,
        t_sheath=0.0022,
        Lz_carrier=0.050,
        Lz_sheath=0.050,
        show_internal_walls=True,
        wall_linewidth=3.0,
        value_fmt=".4g",
    )
    plt.show()

    fig, ax, artist = plot_state_grid_field(
        state_d,
        field_name="T",
        unit="mm",
        Lr_carrier=0.0037,
        t_carrier=0.0020,
        Lr_sheath=0.0188,
        t_sheath=0.0022,
        Lz_carrier=0.050,
        Lz_sheath=0.050,
        show_internal_walls=True,
        point_size=18,
        value_fmt=".3e",
    )
    plt.show()

    fig, ax, artist = plot_state_grid_field(
        state_d,
        field_name="uz",
        unit="mm",
        Lr_carrier=0.0037,
        t_carrier=0.0020,
        Lr_sheath=0.0188,
        t_sheath=0.0022,
        Lz_carrier=0.050,
        Lz_sheath=0.050,
        show_internal_walls=True,
        point_size=18,
        value_fmt=".3e",
    )
    plt.show()

    return


def load(file_name: str) -> Dict[str, Any]:
    path = os.path.join("saved_states", f"{file_name}.npz")
    data = np.load(path, allow_pickle=True)

    return {key: data[key] for key in data.files}


# ============================================================
# Small helpers
# ============================================================

def _get_scale(unit):
    """
    Convert plotting units.
    """
    if unit == "mm":
        return 1000.0, "z [mm]", "r [mm]"

    if unit == "m":
        return 1.0, "z [m]", "r [m]"

    raise ValueError("unit must be 'mm' or 'm'.")


def _format_value(value, fmt):
    """
    Format real or complex values nicely.
    """
    value = np.asarray(value)

    if np.iscomplexobj(value):
        return f"{value.real:{fmt}} + {value.imag:{fmt}}j"

    value = float(value)

    if np.isnan(value):
        return "nan"

    return format(value, fmt)


def _draw_grid_lines(ax, z_faces, r_faces, linewidth=0.35, alpha=0.35):
    """
    Draw structured finite-volume grid lines.
    """

    for z in z_faces:
        ax.plot(
            [z, z],
            [r_faces[0], r_faces[-1]],
            color="k",
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )

    for r in r_faces:
        ax.plot(
            [z_faces[0], z_faces[-1]],
            [r, r],
            color="k",
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )


# ============================================================
# Internal wall plotting
# ============================================================

def plot_internal_walls(
    ax,
    state_d,
    Lr_carrier,
    t_carrier,
    Lr_sheath,
    t_sheath,
    Lz_carrier,
    Lz_sheath,
    unit="mm",
    linewidth=2.0,
    color="black",
    zorder=50,
):
    """
    Plot solid black lines at the actual interior walls.

    Parameters are assumed to be in metres:
        Lr_carrier
        t_carrier
        Lr_sheath
        t_sheath
        Lz_carrier
        Lz_sheath

    The grid arrays Zuz/Rur are still taken from state_d.
    """

    scale, _, _ = _get_scale(unit)

    # Physical start of domain from the saved z-face grid
    z0 = float(np.asarray(state_d["Zuz"][:, 0])[0]) * scale

    # Carrier wall
    r_carrier_outer = Lr_carrier * scale
    r_carrier_inner = (Lr_carrier - t_carrier) * scale
    z_carrier_end = Lz_carrier * scale

    ax.plot(
        [z0, z_carrier_end, z_carrier_end, z0, z0],
        [
            r_carrier_inner,
            r_carrier_inner,
            r_carrier_outer,
            r_carrier_outer,
            r_carrier_inner,
        ],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )

    # Sheath wall
    r_sheath_inner = Lr_sheath * scale
    r_sheath_outer = (Lr_sheath + t_sheath) * scale
    z_sheath_end = Lz_sheath * scale

    ax.plot(
        [z0, z_sheath_end, z_sheath_end, z0, z0],
        [
            r_sheath_inner,
            r_sheath_inner,
            r_sheath_outer,
            r_sheath_outer,
            r_sheath_inner,
        ],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )

    return ax


# ============================================================
# Field preparation
# ============================================================

def _prepare_field_for_plot(
    state_d,
    field_name,
    component="real",
    velocity_include_ghosts=False,
):
    """
    Prepare a field for plotting.

    Scalars:
        Plotted on cell centres / cell volumes.

    uz:
        Plotted on Zuz/Ruz, i.e. axial velocity faces.

    ur:
        Plotted on Zur/Rur, i.e. radial velocity faces.
    """

    # ------------------------------------------------------------
    # Axial velocity: lives on Zuz/Ruz
    # ------------------------------------------------------------
    if field_name == "uz":
        F_raw = np.asarray(state_d["uz"])

        if velocity_include_ghosts:
            F = F_raw
            Zp = np.asarray(state_d["Zuz"])
            Rp = np.asarray(state_d["Ruz"])
        else:
            # Keep all z-faces, remove radial ghost positions
            F = F_raw[:, 1:-1]
            Zp = np.asarray(state_d["Zuz"])[:, 1:-1]
            Rp = np.asarray(state_d["Ruz"])[:, 1:-1]

        return {
            "mode": "points",
            "F": F,
            "Zp": Zp,
            "Rp": Rp,
            "label": "uz on axial faces",
        }

    # ------------------------------------------------------------
    # Radial velocity: lives on Zur/Rur
    # ------------------------------------------------------------
    if field_name == "ur":
        F_raw = np.asarray(state_d["ur"])

        if velocity_include_ghosts:
            F = F_raw
            Zp = np.asarray(state_d["Zur"])
            Rp = np.asarray(state_d["Rur"])
        else:
            # Remove axial ghost positions, keep all radial faces
            F = F_raw[1:-1, :]
            Zp = np.asarray(state_d["Zur"])[1:-1, :]
            Rp = np.asarray(state_d["Rur"])[1:-1, :]

        return {
            "mode": "points",
            "F": F,
            "Zp": Zp,
            "Rp": Rp,
            "label": "ur on radial faces",
        }

    # ------------------------------------------------------------
    # Derived velocity magnitude at cell centres
    # ------------------------------------------------------------
    if field_name == "speed":
        uz = np.asarray(state_d["uz"])
        ur = np.asarray(state_d["ur"])

        uzc = 0.5 * (uz[1:, 1:-1] + uz[:-1, 1:-1])
        urc = 0.5 * (ur[1:-1, 1:] + ur[1:-1, :-1])

        F = np.sqrt(uzc**2 + urc**2)

        return {
            "mode": "cells",
            "F": F,
            "label": "speed at cell centres",
        }

    # ------------------------------------------------------------
    # Cell-centred scalar fields
    # ------------------------------------------------------------
    Z = np.asarray(state_d["Z"])
    cell_shape = Z[1:-1, 1:-1].shape

    F_raw = np.asarray(state_d[field_name])

    if F_raw.shape == Z.shape:
        # Field includes ghost cells
        F = F_raw[1:-1, 1:-1]

    elif F_raw.shape == cell_shape:
        # Field is already interior-only, e.g. A, P, Fr, Fz
        F = F_raw

    else:
        raise ValueError(
            f"Field '{field_name}' has shape {F_raw.shape}, but expected either "
            f"{Z.shape} including ghosts or {cell_shape} on interior cells."
        )

    if np.iscomplexobj(F):
        if component == "real":
            F = F.real
            label = f"Re({field_name})"
        elif component == "imag":
            F = F.imag
            label = f"Im({field_name})"
        elif component == "abs":
            F = np.abs(F)
            label = f"|{field_name}|"
        else:
            raise ValueError("component must be 'real', 'imag', or 'abs'.")
    else:
        label = field_name

    return {
        "mode": "cells",
        "F": F,
        "label": label,
    }


# ============================================================
# Main plotting function
# ============================================================

def plot_state_grid_field(
    state_d,
    field_name="T",
    unit="mm",
    component="real",
    cmap_name="viridis",
    show_grid=True,
    grid_linewidth=0.35,
    show_points=True,
    point_size=12,
    label_on_zoom=True,
    max_labels=120,
    value_fmt=".3g",
    title=None,
    velocity_include_ghosts=False,
    show_internal_walls=True,
    wall_linewidth=3.0,
    Lr_carrier=None,
    t_carrier=None,
    Lr_sheath=None,
    t_sheath=None,
    Lz_carrier=None,
    Lz_sheath=None,
):
    """
    Plot a saved ICP state field on the actual finite-volume grid.

    Required in state_d:
        Z, R, Zuz, Ruz, Zur, Rur

    Scalars:
        T, p, rho, P, Fr, Fz, A, etc.
        These are plotted on cell centres / cell volumes.

    Velocities:
        uz is plotted on Zuz/Ruz.
        ur is plotted on Zur/Rur.

    Internal walls:
        Pass Lr_carrier, t_carrier, Lr_sheath, t_sheath,
        Lz_carrier, Lz_sheath if show_internal_walls=True.

    Interactions:
        - Scroll to zoom around the mouse pointer.
        - Click a cell/point to show its exact value.
        - When zoomed in enough, values appear automatically.
    """

    scale, xlabel, ylabel = _get_scale(unit)

    # ------------------------------------------------------------
    # Grid faces from saved state dictionary
    # ------------------------------------------------------------
    z_faces = np.asarray(state_d["Zuz"][:, 0]) * scale
    r_faces = np.asarray(state_d["Rur"][0, :]) * scale

    # Cell centres
    Zc = np.asarray(state_d["Z"][1:-1, 1:-1]) * scale
    Rc = np.asarray(state_d["R"][1:-1, 1:-1]) * scale

    # ------------------------------------------------------------
    # Prepare selected field
    # ------------------------------------------------------------
    prepared = _prepare_field_for_plot(
        state_d,
        field_name=field_name,
        component=component,
        velocity_include_ghosts=velocity_include_ghosts,
    )

    mode = prepared["mode"]
    F = prepared["F"]
    label = prepared["label"]

    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")

    # ------------------------------------------------------------
    # Cell-centred scalar plotting
    # ------------------------------------------------------------
    if mode == "cells":
        F_plot = F.copy()

        # Optional masking if you added mask_solid to state_d
        if "mask_solid" in state_d:
            mask_solid = np.asarray(state_d["mask_solid"])[1:-1, 1:-1]
            if mask_solid.shape == F_plot.shape:
                F_plot = np.ma.array(F_plot, mask=mask_solid)

        artist = ax.pcolormesh(
            z_faces,
            r_faces,
            F_plot.T,
            shading="flat",
            cmap=cmap,
        )

        if show_points:
            ax.scatter(
                Zc.ravel(),
                Rc.ravel(),
                s=point_size,
                color="k",
                alpha=0.55,
                zorder=10,
            )

        X_value = Zc
        Y_value = Rc

    # ------------------------------------------------------------
    # Staggered velocity plotting
    # ------------------------------------------------------------
    elif mode == "points":
        Zp = prepared["Zp"] * scale
        Rp = prepared["Rp"] * scale

        artist = ax.scatter(
            Zp.ravel(),
            Rp.ravel(),
            c=F.ravel(),
            s=point_size,
            cmap=cmap,
            edgecolors="k",
            linewidths=0.25,
            zorder=15,
        )

        X_value = Zp
        Y_value = Rp

    else:
        raise ValueError("Unknown plotting mode.")

    # ------------------------------------------------------------
    # Colour bar
    # ------------------------------------------------------------
    cbar = fig.colorbar(artist, ax=ax)
    cbar.set_label(label)

    # ------------------------------------------------------------
    # Grid lines
    # ------------------------------------------------------------
    if show_grid:
        _draw_grid_lines(
            ax,
            z_faces,
            r_faces,
            linewidth=grid_linewidth,
            alpha=0.35,
        )

    # ------------------------------------------------------------
    # Actual internal wall outlines
    # ------------------------------------------------------------
    if show_internal_walls:
        needed = [
            Lr_carrier,
            t_carrier,
            Lr_sheath,
            t_sheath,
            Lz_carrier,
            Lz_sheath,
        ]

        if any(x is None for x in needed):
            raise ValueError(
                "To plot internal walls, pass:\n"
                "Lr_carrier, t_carrier, Lr_sheath, t_sheath, "
                "Lz_carrier, Lz_sheath\n"
                "or set show_internal_walls=False."
            )

        plot_internal_walls(
            ax=ax,
            state_d=state_d,
            Lr_carrier=Lr_carrier,
            t_carrier=t_carrier,
            Lr_sheath=Lr_sheath,
            t_sheath=t_sheath,
            Lz_carrier=Lz_carrier,
            Lz_sheath=Lz_sheath,
            unit=unit,
            linewidth=wall_linewidth,
            color="black",
            zorder=50,
        )

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        ax.set_title(f"{label} on ICP finite-volume grid")
    else:
        ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(z_faces[0], z_faces[-1])
    ax.set_ylim(r_faces[0], r_faces[-1])

    # ------------------------------------------------------------
    # Annotation box for clicks
    # ------------------------------------------------------------
    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        zorder=80,
    )
    annotation.set_visible(False)

    value_texts = []

    def clear_value_texts():
        nonlocal value_texts

        for txt in value_texts:
            txt.remove()

        value_texts = []

    def update_visible_value_labels(event=None):
        """
        Show values only when few enough cells/points are visible.
        This makes zoomed-in inspection cleaner.
        """

        if not label_on_zoom:
            return

        clear_value_texts()

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        visible = (
            (X_value >= xmin)
            & (X_value <= xmax)
            & (Y_value >= ymin)
            & (Y_value <= ymax)
        )

        count = int(np.sum(visible))

        if count == 0 or count > max_labels:
            fig.canvas.draw_idle()
            return

        inds = np.argwhere(visible)

        for i, j in inds:
            value = F[i, j]

            txt = ax.text(
                X_value[i, j],
                Y_value[i, j],
                _format_value(value, value_fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=70,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    fc="black",
                    ec="none",
                    alpha=0.60,
                ),
            )

            value_texts.append(txt)

        fig.canvas.draw_idle()

    def on_click(event):
        """
        Click nearest cell centre or velocity face point.
        """

        if event.inaxes != ax:
            return

        z = event.xdata
        r = event.ydata

        dist2 = (X_value - z) ** 2 + (Y_value - r) ** 2
        i, j = np.unravel_index(np.argmin(dist2), dist2.shape)

        value = F[i, j]

        annotation.xy = (X_value[i, j], Y_value[i, j])
        annotation.set_text(
            f"{label}\n"
            f"i={i}, j={j}\n"
            f"z={X_value[i, j]:.4g} {unit}\n"
            f"r={Y_value[i, j]:.4g} {unit}\n"
            f"value={_format_value(value, value_fmt)}"
        )
        annotation.set_visible(True)

        fig.canvas.draw_idle()

    def on_scroll(event):
        """
        Smooth scroll-wheel zoom around the mouse pointer.
        """

        if event.inaxes != ax:
            return

        base_scale = 1.25

        if event.button == "up":
            zoom_factor = 1.0 / base_scale
        elif event.button == "down":
            zoom_factor = base_scale
        else:
            return

        x_mouse = event.xdata
        y_mouse = event.ydata

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        new_width = (x1 - x0) * zoom_factor
        new_height = (y1 - y0) * zoom_factor

        relx = (x_mouse - x0) / (x1 - x0)
        rely = (y_mouse - y0) / (y1 - y0)

        ax.set_xlim(
            x_mouse - relx * new_width,
            x_mouse + (1.0 - relx) * new_width,
        )

        ax.set_ylim(
            y_mouse - rely * new_height,
            y_mouse + (1.0 - rely) * new_height,
        )

        update_visible_value_labels()

    # ------------------------------------------------------------
    # Connect events
    # ------------------------------------------------------------
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    ax.callbacks.connect("xlim_changed", update_visible_value_labels)
    ax.callbacks.connect("ylim_changed", update_visible_value_labels)

    update_visible_value_labels()

    return fig, ax, artist

if __name__ == "__main__":
    main()