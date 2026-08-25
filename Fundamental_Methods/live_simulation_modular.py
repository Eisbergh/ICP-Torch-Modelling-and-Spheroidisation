from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

# NOTE:
# This module deliberately does NOT import ``input`` or
# ``Fundamental_Methods.helpers`` here. They are supplied to animate().
# This keeps the viewer reusable for different ICP input files / meshes.

# Runtime objects. They are populated by _configure_runtime() before the GUI starts.
state = None
solver = None
grid = None
torch = None

load = None
save = None
StretchedGridInterpolator = None

FROM_ZERO = None
NO_TEMPERATURE = None
NEW_GRID_INTERPOLATE = None
FLOW_TO_THERMAL_STEPS = None

save_file_name = None
load_file_name = None

# Cached interpolation object for the currently supplied grid.
interp = None


# =============================================================================
# SECTION — SIMULATION SETUP
# =============================================================================

def initialize_simulation(cfg: "AppConfig") -> Dict[str, Any]:
    if FROM_ZERO:
        state_d = state.as_dict()
    else:
        state_d = load(file_name=load_file_name)
        state.load_from_dict(state_d, new_grid=NEW_GRID_INTERPOLATE)

    print("====== Initialization Completed =======")
    print("=======================================")

    state.step = 1
    solver.iterate_once(True)

    print("===== First Iteration Completed =======")
    print("=======================================")

    return state.as_dict()


def step_simulation(cfg: "AppConfig") -> Dict[str, Any]:
    step = state.step

    update_mag = step % int(cfg.update_mag_every) == 0
    save_file = int(cfg.save_file) != 0 and step % int(cfg.save_file) == 0

    if save_file:
        save(state.as_dict(), file_name=cfg.save_name)

    if update_mag:
        state.power_coil()
        print(f"Coil power is {state.P_coil} kW.")

    if step % FLOW_TO_THERMAL_STEPS == 0:
        # print("updated_flow")
        solver.iterate_once(update_mag)
    else:
        print("thermal")
        solver.iterate_once_thermal(update_mag)

    mask_heating = (
        (22 / 1000 > grid.R)
        & (55 / 1000 < grid.Z)
        & (grid.Z < 150 / 1000)
    )

    mask_test = (
        (20 / 1000 > grid.R)
        & (grid.R > grid.torch.Lr_carrier)
        & (50 / 1000 < grid.Z)
        & (grid.Z < 60 / 1000)
    )

    if np.max(state.T[mask_test]) < 8000 and FROM_ZERO and not NO_TEMPERATURE:
        state.T[mask_heating] += 10

    return state.as_dict()


# =============================================================================
# SECTION — CONFIGURATION
# =============================================================================

@dataclass
class AppConfig:
    # ----- UI update rate -----
    plot_every_n_steps: int = 50

    # Hidden FPS throttle
    max_fps: int = 30

    # ----- Domain limits for axes -----
    Lz: float = 0.2
    Lr: float = 0.025

    # ----- Aspect ratio -----
    # r:z = 1:4 => r/z = 0.25
    aspect_ratio_r_over_z: float = 0.25

    # ----- Region overlay -----
    show_regions_overlay: bool = True

    # ----- DISCRETE COLOUR LEVELS -----
    # 0 = continuous
    image_n_levels: int = 10

    # ----- STREAMLINES over plot1 -----
    show_velocity_streamlines: bool = True
    vel_stream_update_every_frames: int = 6
    vel_stream_max_steps: int = 100
    vel_stream_ds: float = 0.0015
    vel_stream_min_speed: float = 1e-6
    vel_stream_line_width: float = 0.2

    # Region seeding box, in z/r metres
    vel_seed_z0: float = 0.00
    vel_seed_z1: float = 0.20
    vel_seed_r0: float = 0.00
    vel_seed_r1: float = 0.025

    # Region seeding density
    vel_seed_nz: int = 15
    vel_seed_nr: int = 10

    # Trace both directions from each seed
    vel_stream_bidirectional: bool = True

    # ----- PSI CONTOURS over plot2 -----
    show_psi_contours: bool = True
    psi_n_contours: int = 18
    psi_contour_line_width: int = 2

    # =============================================================================
    # SIMULATION TUNABLES
    # =============================================================================
    update_mag_every: int = 200
    save_file: int = 0
    save_name: str = ""
    power: float = 1.0
    relax_u: float = 1.0


CFG = AppConfig()


# =============================================================================
# SECTION — WORKER THREAD
# =============================================================================

class SimWorker(QtCore.QObject):
    frame_ready = QtCore.Signal(dict)
    status = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self._state: Optional[Dict[str, Any]] = None
        self._last_emit = 0.0

    @QtCore.Slot()
    def start(self):
        if self._running:
            return

        self._running = True
        self._state = initialize_simulation(self.cfg)

        self.status.emit("Running")
        self._loop()

    @QtCore.Slot()
    def stop(self):
        self._running = False
        self.status.emit("Stopped")

    def _loop(self):
        if self._state is None:
            self.finished.emit()
            return

        cfg = self.cfg

        while self._running:
            frame = step_simulation(cfg)

            if frame["step"] % max(1, cfg.plot_every_n_steps) == 0:
                now = time.perf_counter()
                min_dt = 1.0 / max(1, cfg.max_fps)

                if now - self._last_emit >= min_dt:
                    self._last_emit = now
                    self.frame_ready.emit(dict(frame))

        self.finished.emit()


# =============================================================================
# SECTION — PHYSICAL PLOT GRID HELPERS
# =============================================================================

def get_physical_plot_limits():
    """
    Return physical plotting limits.
    These are the actual torch-domain limits, not ghost-cell limits.
    """

    z0 = 0.0
    z1 = float(CFG.Lz)

    r0 = 0.0
    r1 = float(CFG.Lr)

    return z0, z1, r0, r1


def make_physical_plot_grid():
    """
    Create a regular plotting grid whose points are pixel centres.

    Important:
    - We do NOT trust grid.R_regular/grid.Z_regular coordinates here.
    - We only use their shape.
    - This avoids visual shifts caused by ghost-cell plotting coordinates.
    """

    z0, z1, r0, r1 = get_physical_plot_limits()

    nz, nr = grid.R_regular.shape

    dz = (z1 - z0) / nz
    dr = (r1 - r0) / nr

    z_centres = np.linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, nz)
    r_centres = np.linspace(r0 + 0.5 * dr, r1 - 0.5 * dr, nr)

    Zp, Rp = np.meshgrid(z_centres, r_centres, indexing="ij")

    return Rp, Zp


def make_regular_solid_mask(Rp: np.ndarray, Zp: np.ndarray) -> np.ndarray:
    """
    Build a solid-wall mask on the regular plotting grid.

    This prevents the carrier/sheath wall regions from being displayed
    as if they were plasma gas.
    """

    t = grid.torch

    solid = np.zeros_like(Rp, dtype=bool)

    # Carrier wall
    if all(hasattr(t, name) for name in ["Lr_carrier", "t_carrier", "Lz_carrier"]):
        r_carrier_outer = t.Lr_carrier
        r_carrier_inner = t.Lr_carrier - t.t_carrier

        carrier_wall = (
            (Zp <= t.Lz_carrier)
            & (Rp >= r_carrier_inner)
            & (Rp <= r_carrier_outer)
        )

        solid |= carrier_wall

    # Sheath wall
    if all(hasattr(t, name) for name in ["Lr_sheath", "t_sheath", "Lz_sheath"]):
        r_sheath_inner = t.Lr_sheath
        r_sheath_outer = t.Lr_sheath + t.t_sheath

        sheath_wall = (
            (Zp <= t.Lz_sheath)
            & (Rp >= r_sheath_inner)
            & (Rp <= r_sheath_outer)
        )

        solid |= sheath_wall

    return solid


def mask_solid_for_display(field: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
    """
    Hide solid regions in displayed fields by setting them to NaN.
    """

    out = np.array(field, copy=True, dtype=float)

    if out.shape == solid_mask.shape:
        out[solid_mask] = np.nan

    return out


# =============================================================================
# SECTION — FIELDS TO DISPLAY
# =============================================================================

def cell_center_velocities(frame: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert staggered velocities to cell centres.

    Arrays are indexed as [z, r].
    """

    uz = frame["uz"]
    ur = frame["ur"]

    uzc = 0.5 * (uz[1:, 1:-1] + uz[:-1, 1:-1])
    urc = 0.5 * (ur[1:-1, 1:] + ur[1:-1, :-1])

    return uzc, urc


def compute_fields(frame: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute fields for plotting.

    Important:
    - The solver arrays are sliced to remove ghost cells.
    - The display grid is built from physical pixel centres.
    - Solid wall regions are masked for display.
    """

    uzc, urc = cell_center_velocities(frame)

    interpolator = interp

    R_plot, Z_plot = make_physical_plot_grid()
    solid_mask_plot = make_regular_solid_mask(R_plot, Z_plot)

    rho_raw = interpolator(frame["rho"][1:-1, 1:-1], R_plot, Z_plot)
    T_raw = interpolator(frame["T"][1:-1, 1:-1], R_plot, Z_plot)
    p_raw = interpolator(frame["p"][1:-1, 1:-1], R_plot, Z_plot)

    div_raw = interpolator(frame["div"], R_plot, Z_plot)
    P_raw = interpolator(frame["P"], R_plot, Z_plot)
    Fr_raw = interpolator(frame["Fr"], R_plot, Z_plot)
    Fz_raw = interpolator(frame["Fz"], R_plot, Z_plot)

    uz_raw = interpolator(uzc, R_plot, Z_plot)
    ur_raw = interpolator(urc, R_plot, Z_plot)

    speed_raw = np.sqrt(uz_raw**2 + ur_raw**2)

    # Use raw fields for psi so that NaNs do not break the streamfunction.
    psi = state.mass_streamfunction(uz_raw, ur_raw, rho_raw, R_plot, Z_plot) * 1e6

    return {
        "uz": mask_solid_for_display(uz_raw, solid_mask_plot),
        "ur": mask_solid_for_display(ur_raw, solid_mask_plot),
        "T": mask_solid_for_display(T_raw, solid_mask_plot),
        "p": mask_solid_for_display(p_raw, solid_mask_plot),
        "rho": mask_solid_for_display(rho_raw, solid_mask_plot),
        "div": mask_solid_for_display(div_raw, solid_mask_plot),
        "psi": psi,
        "speed": mask_solid_for_display(speed_raw, solid_mask_plot),
        "P": mask_solid_for_display(P_raw, solid_mask_plot),
        "Fr": mask_solid_for_display(Fr_raw, solid_mask_plot),
        "Fz": mask_solid_for_display(Fz_raw, solid_mask_plot),
    }


def compute_fields_streamlines(frame: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute raw cell-centred velocity for streamline plotting.
    Streamlines still use the original physical grid, not the regular image grid.
    """

    uzc, urc = cell_center_velocities(frame)

    return {"uz": uzc, "ur": urc}


# =============================================================================
# SECTION — COLORMAPS + DISCRETE BANDING + COLORBAR
# =============================================================================

def get_pg_cmap(cmap_name: str):
    try:
        return pg.colormap.get(cmap_name)
    except Exception:
        return pg.colormap.get("viridis")


def make_discrete_pg_colormap(cmap_name: str, n_levels: int) -> pg.ColorMap:
    base = get_pg_cmap(cmap_name)
    n = max(2, int(n_levels))

    lut = base.getLookupTable(0.0, 1.0, n)
    pos = np.linspace(0.0, 1.0, n)
    colors = [tuple(int(x) for x in rgba) for rgba in lut]

    return pg.ColorMap(pos, colors)


def apply_colormap_to_image_and_colorbar(
    img: pg.ImageItem,
    cbar: pg.ColorBarItem,
    cmap_name: str,
    n_levels: int,
) -> None:
    if int(n_levels) <= 1:
        cmap = get_pg_cmap(cmap_name)
        img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        cbar.setColorMap(cmap)
    else:
        cmap_disc = make_discrete_pg_colormap(cmap_name, int(n_levels))
        img.setLookupTable(cmap_disc.getLookupTable(0.0, 1.0, int(n_levels)))
        cbar.setColorMap(cmap_disc)


def quantize(field: np.ndarray, vmin: float, vmax: float, n: int) -> np.ndarray:
    """
    Quantize field into discrete colour levels while preserving NaNs.
    """

    if n <= 1:
        return field

    out = np.full_like(field, np.nan, dtype=float)
    finite = np.isfinite(field)

    if not np.any(finite):
        return out

    bins = np.linspace(vmin, vmax, n + 1)

    idx = np.digitize(field[finite], bins) - 1
    idx = np.clip(idx, 0, n - 1)

    centers = 0.5 * (bins[:-1] + bins[1:])
    out[finite] = centers[idx]

    return out


def set_image_and_levels(
    img: pg.ImageItem,
    field_zr: np.ndarray,
    z0: float,
    z1: float,
    r0: float,
    r1: float,
    symmetric: bool = False,
    n_levels: int = 0,
) -> Tuple[float, float]:
    """
    Plot image using physical-domain extents.
    """

    finite = np.isfinite(field_zr)

    if not np.any(finite):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(field_zr))
        vmax = float(np.nanmax(field_zr))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    data = field_zr

    if n_levels is not None and int(n_levels) >= 2:
        data = quantize(data, vmin, vmax, int(n_levels))

    img.setImage(data, levels=(vmin, vmax), autoLevels=False)

    # This rectangle is the real physical domain.
    # It does not include ghost cells.
    img.setRect(pg.QtCore.QRectF(z0, r0, z1 - z0, r1 - r0))

    return vmin, vmax


def get_colormap_names() -> list[str]:
    return [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "turbo",
        "grey",
        "ice",
        "fire",
    ]


def set_colorbar_level_ticks(
    cbar: pg.ColorBarItem,
    vmin: float,
    vmax: float,
    n_levels: int,
    mode: str = "edges",
    label_every: int = 2,
    fmt: str = "{:.0f}",
) -> None:
    n = int(n_levels)

    if n <= 1 or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        try:
            cbar.axis.setTicks(None)
        except Exception:
            pass
        return

    if mode.lower() == "edges":
        vals_all = np.linspace(vmin, vmax, n + 1)
    else:
        edges = np.linspace(vmin, vmax, n + 1)
        vals_all = 0.5 * (edges[:-1] + edges[1:])

    minor = [(float(v), "") for v in vals_all]
    major = []

    for k, v in enumerate(vals_all):
        if k % max(1, int(label_every)) == 0:
            major.append((float(v), fmt.format(v)))

    cbar.axis.setTicks([major, minor])
    cbar.axis.setStyle(showValues=True)
    cbar.axis.setTextPen(pg.mkPen("w"))
    cbar.axis.setPen(pg.mkPen("w"))


# =============================================================================
# SECTION — STREAMLINE MATH
# =============================================================================

def bilinear_sample_nonuniform(
    field: np.ndarray,
    z1d: np.ndarray,
    r1d: np.ndarray,
    z: float,
    r: float,
) -> Tuple[float, bool]:
    if z < z1d[0] or z > z1d[-1] or r < r1d[0] or r > r1d[-1]:
        return 0.0, False

    iz = np.searchsorted(z1d, z) - 1
    ir = np.searchsorted(r1d, r) - 1

    iz = int(np.clip(iz, 0, len(z1d) - 2))
    ir = int(np.clip(ir, 0, len(r1d) - 2))

    z0, z1 = z1d[iz], z1d[iz + 1]
    r0, r1 = r1d[ir], r1d[ir + 1]

    wz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    wr = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)

    f00 = field[iz, ir]
    f10 = field[iz + 1, ir]
    f01 = field[iz, ir + 1]
    f11 = field[iz + 1, ir + 1]

    f0 = (1 - wz) * f00 + wz * f10
    f1 = (1 - wz) * f01 + wz * f11
    f = (1 - wr) * f0 + wr * f1

    return float(f), True


def trace_streamline(
    uz: np.ndarray,
    ur: np.ndarray,
    z1d: np.ndarray,
    r1d: np.ndarray,
    z_start: float,
    r_start: float,
    ds: float,
    max_steps: int,
    min_speed: float,
    zmin: float,
    zmax: float,
    rmin: float,
    rmax: float,
) -> Optional[np.ndarray]:
    z = float(z_start)
    r = float(r_start)

    pts = []

    for _ in range(max_steps):
        if z < zmin or z > zmax or r < rmin or r > rmax:
            break

        u1, ok1 = bilinear_sample_nonuniform(uz, z1d, r1d, z, r)
        v1, ok2 = bilinear_sample_nonuniform(ur, z1d, r1d, z, r)

        if not (ok1 and ok2):
            break

        sp = (u1 * u1 + v1 * v1) ** 0.5

        if sp < min_speed:
            break

        u1n = u1 / (sp + 1e-30)
        v1n = v1 / (sp + 1e-30)

        zm = z + 0.5 * ds * u1n
        rm = r + 0.5 * ds * v1n

        u2, ok3 = bilinear_sample_nonuniform(uz, z1d, r1d, zm, rm)
        v2, ok4 = bilinear_sample_nonuniform(ur, z1d, r1d, zm, rm)

        if not (ok3 and ok4):
            break

        sp2 = (u2 * u2 + v2 * v2) ** 0.5

        if sp2 < min_speed:
            break

        u2n = u2 / (sp2 + 1e-30)
        v2n = v2 / (sp2 + 1e-30)

        pts.append((z, r))

        z = z + ds * u2n
        r = r + ds * v2n

    if len(pts) < 5:
        return None

    return np.array(pts, dtype=float)


def trace_streamline_bidir(
    uz,
    ur,
    z1d,
    r1d,
    z0,
    r0,
    cfg,
    zmin,
    zmax,
    rmin,
    rmax,
):
    fwd = trace_streamline(
        uz,
        ur,
        z1d,
        r1d,
        z0,
        r0,
        ds=cfg.vel_stream_ds,
        max_steps=cfg.vel_stream_max_steps,
        min_speed=cfg.vel_stream_min_speed,
        zmin=zmin,
        zmax=zmax,
        rmin=rmin,
        rmax=rmax,
    )

    if not cfg.vel_stream_bidirectional:
        return fwd

    bwd = trace_streamline(
        uz,
        ur,
        z1d,
        r1d,
        z0,
        r0,
        ds=-cfg.vel_stream_ds,
        max_steps=cfg.vel_stream_max_steps,
        min_speed=cfg.vel_stream_min_speed,
        zmin=zmin,
        zmax=zmax,
        rmin=rmin,
        rmax=rmax,
    )

    if fwd is None and bwd is None:
        return None

    if bwd is None:
        return fwd

    if fwd is None:
        return bwd

    bwd_rev = bwd[::-1]

    return np.vstack([bwd_rev[:-1], fwd])


def build_streamlines(
    uz: np.ndarray,
    ur: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    cfg: AppConfig,
) -> List[np.ndarray]:
    z1d = Z[:, 0]
    r1d = R[0, :]

    zmin, zmax = 0.0, cfg.Lz
    rmin, rmax = 0.0, cfg.Lr

    z_seeds = np.linspace(cfg.vel_seed_z0, cfg.vel_seed_z1, cfg.vel_seed_nz)
    r_seeds = np.linspace(cfg.vel_seed_r0, cfg.vel_seed_r1, cfg.vel_seed_nr)

    z_seeds = np.clip(z_seeds, zmin + 1e-9, zmax - 1e-9)
    r_seeds = np.clip(r_seeds, rmin + 1e-6, rmax - 1e-6)

    seeds = [(float(zs), float(rs)) for zs in z_seeds for rs in r_seeds]

    lines: List[np.ndarray] = []

    for zs, rs in seeds:
        line = trace_streamline_bidir(
            uz,
            ur,
            z1d,
            r1d,
            zs,
            rs,
            cfg,
            zmin,
            zmax,
            rmin,
            rmax,
        )

        if line is not None:
            lines.append(line)

    return lines


# =============================================================================
# SECTION — PSI CONTOURS OVERLAY
# =============================================================================

def update_psi_isocurves(
    curves: List[pg.IsocurveItem],
    psi: np.ndarray,
    n_levels: int,
) -> None:
    data = psi.T

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    levels = np.linspace(vmin, vmax, n_levels)

    while len(curves) < n_levels:
        curves.append(pg.IsocurveItem(level=0.0))

    for i in range(n_levels):
        curves[i].setData(data)
        curves[i].setLevel(float(levels[i]))

    for i in range(n_levels, len(curves)):
        curves[i].setData(None)


# =============================================================================
# SECTION — MAIN WINDOW
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.cfg = CFG

        self.setWindowTitle("Live Viewer + Streamplots")
        self.resize(1400, 750)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SimWorker] = None
        self._last_frame: Optional[dict] = None
        self._field_choices_ready = False

        self._render_frame_count = 0
        self._vel_stream_items: List[pg.PlotDataItem] = []
        self._psi_curves: List[pg.IsocurveItem] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)

        # -------------------------------------------------------------------------
        # LEFT CONTROLS
        # -------------------------------------------------------------------------
        controls = QtWidgets.QFrame()
        controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls.setFixedWidth(380)

        root.addWidget(controls)

        cL = QtWidgets.QVBoxLayout(controls)

        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setStyleSheet("font-weight: bold;")
        cL.addWidget(self.status_label)

        self.dt_label = QtWidgets.QLabel("dt: -- s")
        self.dt_label.setStyleSheet("font-family: Consolas;")
        cL.addWidget(self.dt_label)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_save_png = QtWidgets.QPushButton("Save screenshot (PNG)")

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.btn_start)
        row1.addWidget(self.btn_stop)

        cL.addLayout(row1)
        cL.addWidget(self.btn_save_png)

        # -------------------------------------------------------------------------
        # SAVE STATE CONTROLS
        # -------------------------------------------------------------------------
        cL.addSpacing(10)
        cL.addWidget(QtWidgets.QLabel("Save simulation state (.npz)"))

        self.in_save_name = QtWidgets.QLineEdit()
        self.in_save_name.setPlaceholderText("e.g. run_165Ni400Nj70")
        self.in_save_name.setText(str(self.cfg.save_name))

        self.btn_save_state_now = QtWidgets.QPushButton("Save state now")

        form_save = QtWidgets.QFormLayout()
        form_save.addRow("Save name:", self.in_save_name)

        cL.addLayout(form_save)
        cL.addWidget(self.btn_save_state_now)

        self.lbl_save_path = QtWidgets.QLabel(
            self._save_path_preview(self.in_save_name.text())
        )
        self.lbl_save_path.setStyleSheet("font-family: Consolas; color: #bbbbbb;")
        cL.addWidget(self.lbl_save_path)

        # -------------------------------------------------------------------------
        # UI REFRESH CONTROLS
        # -------------------------------------------------------------------------
        cL.addSpacing(10)
        cL.addWidget(QtWidgets.QLabel("UI refresh controls:"))

        self.in_plot_every = QtWidgets.QSpinBox()
        self.in_plot_every.setRange(1, 10_000_000)
        self.in_plot_every.setValue(self.cfg.plot_every_n_steps)

        form0 = QtWidgets.QFormLayout()
        form0.addRow("plot every N steps:", self.in_plot_every)

        cL.addLayout(form0)

        # -------------------------------------------------------------------------
        # FIELDS TO DISPLAY
        # -------------------------------------------------------------------------
        cL.addSpacing(10)
        cL.addWidget(QtWidgets.QLabel("Fields to display:"))

        self.cmb_field1 = QtWidgets.QComboBox()
        self.cmb_field2 = QtWidgets.QComboBox()
        self.cmb_field3 = QtWidgets.QComboBox()

        formF = QtWidgets.QFormLayout()
        formF.addRow("plot1 field:", self.cmb_field1)
        formF.addRow("plot2 field:", self.cmb_field2)
        formF.addRow("plot3 field:", self.cmb_field3)

        cL.addLayout(formF)

        # -------------------------------------------------------------------------
        # COLOUR MAP CONTROLS
        # -------------------------------------------------------------------------
        cL.addSpacing(10)
        cL.addWidget(QtWidgets.QLabel("Colours (colormaps):"))

        cmaps = get_colormap_names()

        self.cmb_cmap1 = QtWidgets.QComboBox()
        self.cmb_cmap1.addItems(cmaps)
        self.cmb_cmap1.setCurrentText("viridis")

        self.cmb_cmap2 = QtWidgets.QComboBox()
        self.cmb_cmap2.addItems(cmaps)
        self.cmb_cmap2.setCurrentText("inferno")

        self.cmb_cmap3 = QtWidgets.QComboBox()
        self.cmb_cmap3.addItems(cmaps)
        self.cmb_cmap3.setCurrentText("cividis")

        formC = QtWidgets.QFormLayout()
        formC.addRow("plot1 cmap:", self.cmb_cmap1)
        formC.addRow("plot2 cmap:", self.cmb_cmap2)
        formC.addRow("plot3 cmap:", self.cmb_cmap3)

        cL.addLayout(formC)

        # -------------------------------------------------------------------------
        # DISCRETE COLOUR LEVELS CONTROL
        # -------------------------------------------------------------------------
        self.spin_levels = QtWidgets.QSpinBox()
        self.spin_levels.setRange(0, 256)
        self.spin_levels.setValue(int(self.cfg.image_n_levels))
        self.spin_levels.setToolTip(
            "0 = continuous/smooth.  >=2 = number of discrete colour bands."
        )

        form_levels = QtWidgets.QFormLayout()
        form_levels.addRow("Colour levels (0=smooth):", self.spin_levels)

        cL.addLayout(form_levels)

        # -------------------------------------------------------------------------
        # STREAMLINE CONTROLS — HIDDEN FROM UI BUT STILL AVAILABLE INTERNALLY
        # -------------------------------------------------------------------------
        self.spin_seed_nz = QtWidgets.QSpinBox()
        self.spin_seed_nz.setRange(1, 400)
        self.spin_seed_nz.setValue(self.cfg.vel_seed_nz)

        self.spin_seed_nr = QtWidgets.QSpinBox()
        self.spin_seed_nr.setRange(1, 400)
        self.spin_seed_nr.setValue(self.cfg.vel_seed_nr)

        self.dspin_line_width = QtWidgets.QDoubleSpinBox()
        self.dspin_line_width.setRange(0.1, 10.0)
        self.dspin_line_width.setSingleStep(0.1)
        self.dspin_line_width.setValue(float(self.cfg.vel_stream_line_width))

        # -------------------------------------------------------------------------
        # SIMULATION TUNABLES
        # -------------------------------------------------------------------------
        cL.addSpacing(10)

        simBox = QtWidgets.QGroupBox("Simulation Tunables (add yours here)")
        simLayout = QtWidgets.QFormLayout(simBox)

        self.spin_update_mag = QtWidgets.QSpinBox()
        self.spin_update_mag.setRange(1, 10_000_000)
        self.spin_update_mag.setValue(int(self.cfg.update_mag_every))

        simLayout.addRow("Update EM every N steps:", self.spin_update_mag)

        self.dspin_save = QtWidgets.QDoubleSpinBox()
        self.dspin_save.setRange(0.0, 10000.0)
        self.dspin_save.setValue(float(self.cfg.save_file))

        simLayout.addRow("Auto-save every N steps (0=off):", self.dspin_save)

        cL.addWidget(simBox)
        cL.addStretch(1)

        # -------------------------------------------------------------------------
        # RIGHT PLOTS — THREE PANELS
        # -------------------------------------------------------------------------
        plots = QtWidgets.QWidget()
        root.addWidget(plots, 1)

        vbox = QtWidgets.QVBoxLayout(plots)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)

        def make_panel(title: str):
            glw = pg.GraphicsLayoutWidget()

            plot = glw.addPlot(row=0, col=0, title=title)
            plot.setLabel("bottom", "z [m]")
            plot.setLabel("left", "r [m]")

            img = pg.ImageItem()
            plot.addItem(img)

            cbar = pg.ColorBarItem(values=(0.0, 1.0), width=24)
            glw.addItem(cbar, row=0, col=1)
            cbar.setImageItem(img)

            glw.ci.layout.setColumnMinimumWidth(1, 110)
            glw.ci.layout.setColumnStretchFactor(1, 0)

            cbar.axis.setPen(pg.mkPen("w"))
            cbar.axis.setTextPen(pg.mkPen("w"))

            txt = pg.TextItem(color="w", anchor=(0, 0))
            plot.addItem(txt)

            return glw, plot, img, cbar, txt

        self.glw1, self.plot1, self.img1, self.cbar1, self.txt1 = make_panel("plot1")
        self.glw2, self.plot2, self.img2, self.cbar2, self.txt2 = make_panel("plot2")
        self.glw3, self.plot3, self.img3, self.cbar3, self.txt3 = make_panel("plot3")

        vbox.addWidget(self.glw1, 1)
        vbox.addWidget(self.glw2, 1)
        vbox.addWidget(self.glw3, 1)

        for pw in (self.plot1, self.plot2, self.plot3):
            vb = pw.getViewBox()
            vb.setAspectLocked(True, ratio=self.cfg.aspect_ratio_r_over_z)
            vb.setXRange(0.0, self.cfg.Lz, padding=0.0)
            vb.setYRange(0.0, self.cfg.Lr, padding=0.0)
            vb.setLimits(
                xMin=0.0,
                xMax=self.cfg.Lz,
                yMin=0.0,
                yMax=self.cfg.Lr,
            )
            pw.enableAutoRange(x=False, y=False)

        self._overlay_items = []

        if self.cfg.show_regions_overlay:
            for pw in (self.plot1, self.plot2, self.plot3):
                self._add_regions_overlay(pw)

        self._apply_all_colormaps()

        # -------------------------------------------------------------------------
        # SIGNALS
        # -------------------------------------------------------------------------
        self.cmb_cmap1.currentTextChanged.connect(lambda _: self._apply_all_colormaps())
        self.cmb_cmap2.currentTextChanged.connect(lambda _: self._apply_all_colormaps())
        self.cmb_cmap3.currentTextChanged.connect(lambda _: self._apply_all_colormaps())

        self.spin_levels.valueChanged.connect(lambda _: self._apply_all_colormaps())

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_save_png.clicked.connect(self.on_save_png)

        self.btn_save_state_now.clicked.connect(self.on_save_state_now)
        self.in_save_name.textChanged.connect(self._on_save_name_changed)

    # -------------------------------------------------------------------------
    # SAVE NAME HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _save_path_preview(name: str) -> str:
        name = (name or "").strip()

        if not name:
            return "saved_states/<empty>.npz"

        return f"saved_states/{name}.npz"

    def _on_save_name_changed(self, _txt: str):
        self.lbl_save_path.setText(
            self._save_path_preview(self.in_save_name.text())
        )

        name = self.in_save_name.text().strip()

        if name:
            self.cfg.save_name = name

    def on_save_state_now(self):
        if self._last_frame is None:
            return

        name = self.in_save_name.text().strip()

        if not name:
            return

        save(self._last_frame, file_name=name)

    # -------------------------------------------------------------------------
    # UI APPLY
    # -------------------------------------------------------------------------
    def _apply_all_colormaps(self):
        self.cfg.image_n_levels = int(self.spin_levels.value())

        apply_colormap_to_image_and_colorbar(
            self.img1,
            self.cbar1,
            self.cmb_cmap1.currentText(),
            self.cfg.image_n_levels,
        )

        apply_colormap_to_image_and_colorbar(
            self.img2,
            self.cbar2,
            self.cmb_cmap2.currentText(),
            self.cfg.image_n_levels,
        )

        apply_colormap_to_image_and_colorbar(
            self.img3,
            self.cbar3,
            self.cmb_cmap3.currentText(),
            self.cfg.image_n_levels,
        )

    def _add_regions_overlay(self, plot: pg.PlotItem):
        """
        Draw carrier/sheath geometry from the torch supplied through input_file.

        Nothing here is hard-coded to 50 mm, 18.8 mm, etc. The viewer follows
        the current torch object's geometry.
        """
        t = grid.torch

        pen = pg.mkPen(
            color=(220, 220, 220),
            width=2,
            style=QtCore.Qt.DashLine,
        )

        items = []

        # Carrier outer wall and downstream lip.
        if all(hasattr(t, name) for name in ("Lz_carrier", "Lr_carrier")):
            items.extend([
                pg.PlotDataItem(
                    [0.0, float(t.Lz_carrier)],
                    [float(t.Lr_carrier), float(t.Lr_carrier)],
                    pen=pen,
                ),
                pg.PlotDataItem(
                    [float(t.Lz_carrier), float(t.Lz_carrier)],
                    [0.0, float(t.Lr_carrier)],
                    pen=pen,
                ),
            ])

        # Sheath inner wall and downstream lip.
        if all(hasattr(t, name) for name in ("Lz_sheath", "Lr_sheath")):
            items.extend([
                pg.PlotDataItem(
                    [0.0, float(t.Lz_sheath)],
                    [float(t.Lr_sheath), float(t.Lr_sheath)],
                    pen=pen,
                ),
                pg.PlotDataItem(
                    [float(t.Lz_sheath), float(t.Lz_sheath)],
                    [float(t.Lr_sheath), float(self.cfg.Lr)],
                    pen=pen,
                ),
            ])

        for it in items:
            plot.addItem(it)
            self._overlay_items.append(it)

    def _apply_inputs(self):
        self.cfg.plot_every_n_steps = int(self.in_plot_every.value())

        self.cfg.vel_seed_nz = int(self.spin_seed_nz.value())
        self.cfg.vel_seed_nr = int(self.spin_seed_nr.value())
        self.cfg.vel_stream_line_width = float(self.dspin_line_width.value())

        self.cfg.image_n_levels = int(self.spin_levels.value())

        self.cfg.update_mag_every = int(self.spin_update_mag.value())
        self.cfg.save_file = int(self.dspin_save.value())

        name = self.in_save_name.text().strip()

        if name:
            self.cfg.save_name = name

    # -------------------------------------------------------------------------
    # BUTTONS
    # -------------------------------------------------------------------------
    def on_start(self):
        if self._thread is not None:
            return

        self._apply_inputs()
        self._apply_all_colormaps()

        self._thread = QtCore.QThread()
        self._worker = SimWorker(self.cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)

        self._worker.frame_ready.connect(self.on_frame)
        self._worker.status.connect(self.on_status)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)

        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_refs)

        self._thread.start()

    def on_stop(self):
        if self._worker:
            self._worker.stop()

    def _clear_thread_refs(self):
        self._thread = None
        self._worker = None

    def on_status(self, txt: str):
        self.status_label.setText(txt)

    # -------------------------------------------------------------------------
    # FRAME HANDLER
    # -------------------------------------------------------------------------
    @QtCore.Slot(dict)
    def on_frame(self, frame: dict):
        self._last_frame = frame
        self._render_frame_count += 1

        step = int(frame.get("step", -1))
        t = float(frame.get("time", np.nan))
        dt = float(frame.get("dt", np.nan))

        self.dt_label.setText(
            f"dt: {dt:.3e} s   |   step: {step}   |   t: {t:.3e} s"
        )

        fields = compute_fields(frame)

        if not self._field_choices_ready:
            keys = list(fields.keys())

            for cmb in (self.cmb_field1, self.cmb_field2, self.cmb_field3):
                cmb.clear()
                cmb.addItems(keys)

            self.cmb_field1.setCurrentText("uz" if "uz" in keys else keys[0])

            self.cmb_field2.setCurrentText(
                "T" if "T" in keys else keys[min(1, len(keys) - 1)]
            )

            self.cmb_field3.setCurrentText(
                "div" if "div" in keys else keys[min(2, len(keys) - 1)]
            )

            self._field_choices_ready = True

        f1 = fields[self.cmb_field1.currentText()]
        f2 = fields[self.cmb_field2.currentText()]
        f3 = fields[self.cmb_field3.currentText()]

        z0, z1 = 0.0, self.cfg.Lz
        r0, r1 = 0.0, self.cfg.Lr

        nlev = int(self.cfg.image_n_levels)

        vmin1, vmax1 = set_image_and_levels(
            self.img1,
            f1,
            z0,
            z1,
            r0,
            r1,
            symmetric=False,
            n_levels=nlev,
        )

        vmin2, vmax2 = set_image_and_levels(
            self.img2,
            f2,
            z0,
            z1,
            r0,
            r1,
            symmetric=False,
            n_levels=nlev,
        )

        vmin3, vmax3 = set_image_and_levels(
            self.img3,
            f3,
            z0,
            z1,
            r0,
            r1,
            symmetric=False,
            n_levels=nlev,
        )

        self.cbar1.setLevels((vmin1, vmax1))
        self.cbar2.setLevels((vmin2, vmax2))
        self.cbar3.setLevels((vmin3, vmax3))

        set_colorbar_level_ticks(
            self.cbar1,
            vmin1,
            vmax1,
            nlev,
            mode="edges",
            label_every=1,
            fmt="{:.4g}",
        )

        set_colorbar_level_ticks(
            self.cbar2,
            vmin2,
            vmax2,
            nlev,
            mode="edges",
            label_every=1,
            fmt="{:.4g}",
        )

        set_colorbar_level_ticks(
            self.cbar3,
            vmin3,
            vmax3,
            nlev,
            mode="edges",
            label_every=2,
            fmt="{:.4g}",
        )

        def set_txt(txt_item: pg.TextItem, mn: float, mx: float):
            txt_item.setText(f"min={mn:.3e}\nmax={mx:.3e}")
            txt_item.setPos(0.002, self.cfg.Lr * 0.80)

        set_txt(self.txt1, vmin1, vmax1)
        set_txt(self.txt2, vmin2, vmax2)
        set_txt(self.txt3, vmin3, vmax3)

        fields_streamlines = compute_fields_streamlines(frame)

        self._update_velocity_streamlines(frame, fields_streamlines)
        self._update_psi_contours(fields)

        self._apply_inputs()

    # -------------------------------------------------------------------------
    # STREAMLINES on plot1
    # -------------------------------------------------------------------------
    def _clear_velocity_stream_items(self):
        for it in self._vel_stream_items:
            self.plot1.removeItem(it)

        self._vel_stream_items.clear()

    def _update_velocity_streamlines(
        self,
        frame: dict,
        fields: Dict[str, np.ndarray],
    ):
        if not self.cfg.show_velocity_streamlines:
            self._clear_velocity_stream_items()
            return

        if (
            self._render_frame_count
            % max(1, self.cfg.vel_stream_update_every_frames)
        ) != 0:
            return

        uz = fields["uz"]
        ur = fields["ur"]

        Zc = frame["Z"][1:-1, 1:-1]
        Rc = frame["R"][1:-1, 1:-1]

        if uz.shape != Zc.shape or ur.shape != Zc.shape:
            self._clear_velocity_stream_items()
            return

        lines = build_streamlines(uz, ur, Zc, Rc, self.cfg)

        self._clear_velocity_stream_items()

        pen = pg.mkPen((255, 255, 255), width=self.cfg.vel_stream_line_width)

        for line in lines:
            it = pg.PlotDataItem(line[:, 0], line[:, 1], pen=pen)
            self.plot1.addItem(it)
            self._vel_stream_items.append(it)

    # -------------------------------------------------------------------------
    # PSI CONTOURS on plot2
    # -------------------------------------------------------------------------
    def _ensure_psi_curves_added(self):
        if len(self._psi_curves) == 0:
            for _ in range(self.cfg.psi_n_contours):
                c = pg.IsocurveItem(level=0.0)
                c.setPen(
                    pg.mkPen(
                        (255, 255, 255),
                        width=self.cfg.psi_contour_line_width,
                    )
                )
                self.plot2.addItem(c)
                self._psi_curves.append(c)

    def _update_psi_contours(self, fields: Dict[str, np.ndarray]):
        if not self.cfg.show_psi_contours or "psi" not in fields:
            for c in self._psi_curves:
                c.setData(None)
            return

        self._ensure_psi_curves_added()

        psi = fields["psi"]

        update_psi_isocurves(
            self._psi_curves,
            psi,
            self.cfg.psi_n_contours,
        )

        pen = pg.mkPen(
            (255, 255, 255),
            width=self.cfg.psi_contour_line_width,
        )

        for c in self._psi_curves:
            c.setPen(pen)

    # -------------------------------------------------------------------------
    # SAVE SCREENSHOT
    # -------------------------------------------------------------------------
    def on_save_png(self):
        if self._last_frame is None:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "viewer.png",
            "PNG Files (*.png)",
        )

        if not filename:
            return

        self.grab().save(filename, "PNG")


# =============================================================================
# SECTION — RUNTIME MODULE CONFIGURATION
# =============================================================================

def _load_module_from_path(path, module_name):
    """
    Load a Python module from a .py file path.

    Normally it is preferable to pass an already-imported module to animate(),
    e.g. ``animate(input, helpers)``. This path loader is supplied for cases
    where you explicitly want to pass filenames instead.
    """
    import importlib.util
    from pathlib import Path

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_module(module_or_file, module_name):
    """
    Accept either:
        1. an already imported Python module/object, or
        2. a path to a .py file.
    """
    if isinstance(module_or_file, (str, bytes)):
        return _load_module_from_path(module_or_file, module_name)

    # pathlib.Path without importing pathlib globally.
    if hasattr(module_or_file, "__fspath__"):
        return _load_module_from_path(module_or_file, module_name)

    return module_or_file


def _require_attributes(obj, names, label):
    missing = [name for name in names if not hasattr(obj, name)]

    if missing:
        missing_text = ", ".join(missing)
        raise AttributeError(
            f"{label} is missing required attribute(s): {missing_text}"
        )


def _configure_runtime(input_file, helpers_file):
    """
    Populate the viewer's runtime objects from the supplied ICP input module
    and helpers module.

    Parameters
    ----------
    input_file
        Imported input module OR path to input.py.

    helpers_file
        Imported Fundamental_Methods.helpers module OR path to helpers.py.
    """
    global state, solver, grid, torch
    global load, save, StretchedGridInterpolator
    global FROM_ZERO, NO_TEMPERATURE, NEW_GRID_INTERPOLATE
    global FLOW_TO_THERMAL_STEPS
    global save_file_name, load_file_name
    global interp, CFG

    input_module = _resolve_module(input_file, "_icp_runtime_input")
    helpers_module = _resolve_module(helpers_file, "_icp_runtime_helpers")

    _require_attributes(
        input_module,
        [
            "state",
            "solver",
            "grid",
            "torch",
            "FROM_ZERO",
            "NO_TEMPERATURE",
            "NEW_GRID_INTERPOLATE",
            "FLOW_TO_THERMAL_STEPS",
            "save_file_name",
            "load_file_name",
        ],
        "input_file",
    )

    _require_attributes(
        helpers_module,
        ["load", "save", "StretchedGridInterpolator"],
        "helpers_file",
    )

    # -------------------------------------------------------------------------
    # Solver objects created by input.py
    # -------------------------------------------------------------------------
    state = input_module.state
    solver = input_module.solver
    grid = input_module.grid
    torch = input_module.torch

    # -------------------------------------------------------------------------
    # Run settings defined by input.py
    # -------------------------------------------------------------------------
    FROM_ZERO = bool(input_module.FROM_ZERO)
    NO_TEMPERATURE = bool(input_module.NO_TEMPERATURE)
    NEW_GRID_INTERPOLATE = bool(input_module.NEW_GRID_INTERPOLATE)
    FLOW_TO_THERMAL_STEPS = int(input_module.FLOW_TO_THERMAL_STEPS)

    save_file_name = str(input_module.save_file_name)
    load_file_name = str(input_module.load_file_name)

    # -------------------------------------------------------------------------
    # Helper functions/classes
    # -------------------------------------------------------------------------
    load = helpers_module.load
    save = helpers_module.save
    StretchedGridInterpolator = helpers_module.StretchedGridInterpolator

    # -------------------------------------------------------------------------
    # Viewer geometry now follows the supplied torch automatically.
    # -------------------------------------------------------------------------
    CFG.Lz = float(torch.Lz)
    CFG.Lr = float(torch.Lr)

    CFG.vel_seed_z0 = 0.0
    CFG.vel_seed_z1 = CFG.Lz
    CFG.vel_seed_r0 = 0.0
    CFG.vel_seed_r1 = CFG.Lr

    CFG.save_name = save_file_name

    # -------------------------------------------------------------------------
    # Cached interpolator must be rebuilt for the supplied mesh.
    # -------------------------------------------------------------------------
    interp = StretchedGridInterpolator(
        grid.R[1:-1, 1:-1],
        grid.Z[1:-1, 1:-1],
        kx=1,
        ky=1,
    )

    return input_module, helpers_module


# =============================================================================
# SECTION — MAIN PUBLIC FUNCTION
# =============================================================================

def animate(input_file, helpers_file):
    """
    Launch the live ICP simulation viewer.

    Recommended usage
    -----------------
    import input
    from Fundamental_Methods import helpers
    from live_simulation_modular import animate

    animate(input, helpers)

    File-path usage is also supported
    ---------------------------------
    animate(
        "input.py",
        "Fundamental_Methods/helpers.py",
    )

    The supplied input module is expected to create the current torch, grid,
    electromagnetic object, state and solver. This animation function then
    uses those exact objects rather than creating a second solver internally.
    """
    _configure_runtime(input_file, helpers_file)

    pg.setConfigOptions(antialias=True)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    # Keep a reference on the QApplication as an additional guard against the
    # window being garbage-collected if animate() is called from an existing
    # Qt application / interactive environment.
    app._icp_live_window = win

    if owns_app:
        return app.exec()

    return win

