from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator

from Particle_Solver.particle_parameters import Hb, Hm, Tbp, Tmp, sigma_s


# =============================================================================
# Plasma fields
# =============================================================================

@dataclass
class PlasmaFields:
    R: np.ndarray
    Z: np.ndarray
    T: np.ndarray
    uz: np.ndarray
    ur: np.ndarray

    @classmethod
    def load(cls, file_name: str):
        """Load a converged torch state and return cell-centred physical fields."""
        path = Path(file_name)
        if path.suffix != ".npz":
            path = Path("saved_states") / f"{file_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Plasma state not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            R = np.asarray(data["R"])[1:-1, 1:-1]
            Z = np.asarray(data["Z"])[1:-1, 1:-1]
            T = np.asarray(data["T"])[1:-1, 1:-1]
            uz = 0.5 * (np.asarray(data["uz"])[1:, 1:-1] + np.asarray(data["uz"])[:-1, 1:-1])
            ur = 0.5 * (np.asarray(data["ur"])[1:-1, 1:] + np.asarray(data["ur"])[1:-1, :-1])

        return cls(R=R, Z=Z, T=T, uz=uz, ur=ur)

    @property
    def z_axis(self):
        return self.Z[:, 0]

    @property
    def r_axis(self):
        return self.R[0, :]

    def interpolate(self, z: float, r: float):
        """Return local plasma temperature and velocity at (z, r)."""
        return (
            self.bilinear_interpolate(self.Z, self.R, self.T, z, r),
            self.bilinear_interpolate(self.Z, self.R, self.uz, z, r),
            self.bilinear_interpolate(self.Z, self.R, self.ur, z, r),
        )

    @staticmethod
    def bilinear_interpolate(Z, R, F, z, r):
        """
        Bilinear interpolation on the structured non-uniform R-Z grid.

        Query points are clipped to the cell-centre field extent. This avoids
        accidental extrapolation when a particle sits very close to a physical
        boundary while the stored field is cell-centred.
        """
        z_axis = np.asarray(Z[:, 0], dtype=float)
        r_axis = np.asarray(R[0, :], dtype=float)
        zq = float(np.clip(z, z_axis[0], z_axis[-1]))
        rq = float(np.clip(r, r_axis[0], r_axis[-1]))

        i = int(np.clip(np.searchsorted(z_axis, zq) - 1, 0, len(z_axis) - 2))
        j = int(np.clip(np.searchsorted(r_axis, rq) - 1, 0, len(r_axis) - 2))

        z1, z2 = z_axis[i], z_axis[i + 1]
        r1, r2 = r_axis[j], r_axis[j + 1]
        wz = 0.0 if z2 == z1 else (zq - z1) / (z2 - z1)
        wr = 0.0 if r2 == r1 else (rq - r1) / (r2 - r1)

        Q11 = F[i, j]
        Q21 = F[i + 1, j]
        Q12 = F[i, j + 1]
        Q22 = F[i + 1, j + 1]

        return (
            Q11 * (1 - wz) * (1 - wr)
            + Q21 * wz * (1 - wr)
            + Q12 * (1 - wz) * wr
            + Q22 * wz * wr
        )


# =============================================================================
# Particle state
# =============================================================================

@dataclass
class Particle:
    dp0: float
    dp: float
    z: float
    r: float
    uzp: float
    urp: float
    Tp: float
    x: float = 0.0

    # Outcome flags
    spheroid: bool = False              # has ever become fully molten
    fully_melted: bool = False
    evaporated: bool = False
    escaped: bool = False
    hit_wall: bool = False
    exit_inlet: bool = False
    end: bool = False
    max_steps_reached: bool = False
    wall_terminated: bool = False
    t: float = 0.0

    # Mass state (initialised by ParticleSolver.prepare_particle)
    mp0: Optional[float] = None
    mp: Optional[float] = None

    # Event information
    t_melt_start: Optional[float] = None
    z_melt_start: Optional[float] = None
    t_fully_melted: Optional[float] = None
    z_fully_melted: Optional[float] = None
    t_boil_start: Optional[float] = None
    z_boil_start: Optional[float] = None
    wall_hit_z: Optional[float] = None
    wall_hit_r: Optional[float] = None
    wall_name: Optional[str] = None
    wall_hits: int = 0
    wall_events: list = field(default_factory=list)

    # Running diagnostics
    x_max: float = 0.0
    Tp_max: float = 0.0
    Tgas_max: float = 0.0
    r_min: Optional[float] = None
    r_max: Optional[float] = None
    hot_time: float = 0.0
    thermal_exposure: float = 0.0
    E_conv: float = 0.0
    E_rad: float = 0.0
    E_net: float = 0.0
    Re_max: float = 0.0
    Nu_max: float = 0.0
    Urel_max: float = 0.0

    # Latest local values (useful in detailed histories)
    Tgas: float = np.nan
    q_conv: float = np.nan
    q_rad: float = np.nan
    q_net: float = np.nan
    Re: float = np.nan
    Nu: float = np.nan


# =============================================================================
# Particle solver
# =============================================================================

class ParticleSolver:
    """
    One-way Lagrangian particle solver for the steady ICP plasma fields.
    """

    def __init__(
        self,
        fields,
        torch,
        dt,
        parameterfs,
        store_history=True,
        g=9.81,
        Text=350.0,
        dp_cutoff=1e-9,
        rhop_solid=4500.0,
        rhop_liquid=4200.0,
        wall_mode="terminate",
        wall_restitution=0.2,
        wall_friction=0.8,
        hot_temperature=8000.0,
    ):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if wall_mode not in {"terminate", "reflect"}:
            raise ValueError("wall_mode must be 'terminate' or 'reflect'.")

        self.fields = fields
        self.torch = torch
        self.dt = float(dt)
        self.store_history = bool(store_history)
        self.g = float(g)
        self.Text = float(Text)
        self.dp_cutoff = float(dp_cutoff)
        self.rhop_solid = float(rhop_solid)
        self.rhop_liquid = float(rhop_liquid)
        self.wall_mode = wall_mode
        self.wall_restitution = float(wall_restitution)
        self.wall_friction = float(wall_friction)
        self.hot_temperature = float(hot_temperature)

        self.muvf = parameterfs["muvf"]
        self.kf = parameterfs["kf"]
        self.Cpf = parameterfs["Cpf"]
        self.hf = parameterfs["hf"]
        self.Qrf = parameterfs["Qrf"]
        self.Tf = parameterfs["Tf"]
        self.rhof = parameterfs["rhof"]

    # -------------------------------------------------------------------------
    # Main integration
    # -------------------------------------------------------------------------

    def step(self, particle):
        if self.is_finished(particle):
            return

        self.prepare_particle(particle)

        # Explicit evaluation at the current particle position.
        T, uz, ur = self.fields.interpolate(particle.z, particle.r)
        rho = float(self.rhof(T))
        muv = float(self.muvf(T))
        k = float(self.kf(T))

        self.update_velocity(particle, uz, ur, rho, muv)
        self.update_temperature(particle, T, uz, ur, rho, muv, k)

        # Diagnostics correspond to the interval [t, t+dt].
        self.update_diagnostics(particle, T, uz, ur, rho, muv)

        particle.t += self.dt
        self.update_position(particle)

    def solve(self, particle, max_steps=10000):
        self.prepare_particle(particle)
        history = []

        if self.store_history:
            history.append(self.get_state(particle))

        for _ in range(int(max_steps)):
            if self.is_finished(particle):
                break
            self.step(particle)
            if self.store_history:
                history.append(self.get_state(particle))
        else:
            if not self.is_finished(particle):
                particle.max_steps_reached = True

        if self.store_history:
            return np.asarray(history, dtype=float)
        return None

    def prepare_particle(self, particle):
        # Backward compatibility with simpler Particle objects.
        defaults = {
            "t": 0.0, "x": 0.0, "spheroid": False, "fully_melted": False,
            "escaped": False, "evaporated": False, "hit_wall": False,
            "exit_inlet": False, "end": False, "max_steps_reached": False,
            "wall_terminated": False, "wall_hits": 0, "wall_events": [],
            "t_melt_start": None, "z_melt_start": None,
            "t_fully_melted": None, "z_fully_melted": None,
            "t_boil_start": None, "z_boil_start": None,
            "wall_hit_z": None, "wall_hit_r": None, "wall_name": None,
            "x_max": float(getattr(particle, "x", 0.0)),
            "Tp_max": float(getattr(particle, "Tp", 0.0)),
            "Tgas_max": 0.0, "r_min": float(getattr(particle, "r", 0.0)),
            "r_max": float(getattr(particle, "r", 0.0)),
            "hot_time": 0.0, "thermal_exposure": 0.0,
            "E_conv": 0.0, "E_rad": 0.0, "E_net": 0.0,
            "Re_max": 0.0, "Nu_max": 0.0, "Urel_max": 0.0,
            "Tgas": np.nan, "q_conv": np.nan, "q_rad": np.nan,
            "q_net": np.nan, "Re": np.nan, "Nu": np.nan,
        }
        for name, value in defaults.items():
            if not hasattr(particle, name):
                setattr(particle, name, value)

        if getattr(particle, "mp0", None) is None:
            particle.mp0 = np.pi * particle.dp0**3 / 6.0 * self.rhop_solid
        if getattr(particle, "mp", None) is None:
            particle.mp = float(particle.mp0)
        if getattr(particle, "r_min", None) is None:
            particle.r_min = float(particle.r)
        if getattr(particle, "r_max", None) is None:
            particle.r_max = float(particle.r)
        particle.Tp_max = max(float(getattr(particle, "Tp_max", 0.0)), float(particle.Tp))
        particle.x_max = max(float(getattr(particle, "x_max", 0.0)), float(particle.x))

    def is_finished(self, particle):
        return bool(
            getattr(particle, "escaped", False)
            or getattr(particle, "evaporated", False)
            or getattr(particle, "end", False)
            or getattr(particle, "wall_terminated", False)
            or getattr(particle, "max_steps_reached", False)
        )

    def get_state(self, particle):
        """
        History columns. The original first nine columns are deliberately kept
        in the same order so the existing ParticleTrajectoryStudy still works.
        """
        return [
            particle.t, particle.z, particle.r, particle.uzp, particle.urp,
            particle.Tp, particle.dp, particle.x, float(particle.spheroid),
            particle.mp, particle.Tgas, particle.q_conv, particle.q_rad,
            particle.q_net, particle.Re, particle.Nu, float(particle.wall_hits),
        ]

    # -------------------------------------------------------------------------
    # Properties and heat transfer
    # -------------------------------------------------------------------------

    def Up(self, particle, uz, ur):
        return float(np.hypot(particle.uzp - uz, particle.urp - ur))

    def emissivity(self, Tp):
        return 0.45

    def rhop(self, particle):
        """Effective particle density from the current solid/liquid fraction."""
        x = float(np.clip(particle.x, 0.0, 1.0))
        specific_volume = (1.0 - x) / self.rhop_solid + x / self.rhop_liquid
        return 1.0 / specific_volume

    def Rep(self, particle, rho, muv, uz, ur):
        if muv <= 0 or particle.dp <= 0:
            return 0.0
        return particle.dp * self.Up(particle, uz, ur) * rho / muv

    def Cd(self, particle, rho, muv, uz, ur):
        Re = self.Rep(particle, rho, muv, uz, ur)
        if Re <= 1e-12:
            return 0.0
        if Re <= 0.2:
            return 24.0 / Re
        if Re <= 2.0:
            return 24.0 / Re * (1.0 + 3.0 / 16.0 * Re)
        if Re <= 21.0:
            return 24.0 / Re * (1.0 + 0.11 * Re**0.81)
        return 24.0 / Re * (1.0 + 0.189 * Re**0.62)

    def Nu(self, particle, rho, muv, uz, ur):
        Re = self.Rep(particle, rho, muv, uz, ur)
        return 2.0 + 0.515 * np.sqrt(max(Re, 0.0))

    def hc(self, particle, rho, muv, uz, ur, k):
        if particle.dp <= 0:
            return 0.0
        return self.Nu(particle, rho, muv, uz, ur) * k / particle.dp

    def Cpp(self, particle):
        """Titanium particle heat capacity in J kg-1 K-1."""
        Tp = max(float(particle.Tp) / 1000.0, 1e-9)
        index = np.round(np.arctan(particle.Tp - 700.0) / np.pi + 0.5)
        index2 = np.round(np.arctan(particle.Tp - Tmp + 0.0001) / np.pi + 0.5)

        solid = (
            (1 - index) * 22.61942
            + (1 - index) * 18.98795 * Tp
            + (1 - index) * (-18.18735) * Tp**2
            + (1 - index) * 7.080792 * Tp**3
            + (1 - index) * (-0.143457) / Tp**2
            + index * 44.37174
            + index * (-44.09225) * Tp
            + index * 31.70602 * Tp**2
            + index * 0.0052209 * Tp**3
            + index * 0.036168 / Tp**2
        ) * 1000.0 / 47.867
        liquid = 33.51 * 1000.0 / 47.867
        return float(solid * (1 - index2) + liquid * index2)

    def heat_rates(self, particle, T, uz, ur, rho, muv, k):
        hc = self.hc(particle, rho, muv, uz, ur, k)
        area = np.pi * particle.dp**2
        q_conv = area * hc * (T - particle.Tp)
        q_rad = area * sigma_s * self.emissivity(particle.Tp) * (particle.Tp**4 - self.Text**4)
        return float(q_conv), float(q_rad), float(q_conv - q_rad)

    def _update_diameter_from_mass_and_fraction(self, particle):
        if particle.mp <= 0:
            particle.dp = self.dp_cutoff
            return
        x = float(np.clip(particle.x, 0.0, 1.0))
        Vp = particle.mp * ((1.0 - x) / self.rhop_solid + x / self.rhop_liquid)
        particle.dp = float((6.0 * Vp / np.pi) ** (1.0 / 3.0))

    def _record_melt_start(self, particle):
        if particle.t_melt_start is None:
            particle.t_melt_start = particle.t + self.dt
            particle.z_melt_start = particle.z

    def _record_full_melt(self, particle):
        if particle.t_fully_melted is None:
            particle.t_fully_melted = particle.t + self.dt
            particle.z_fully_melted = particle.z
        particle.fully_melted = True
        particle.spheroid = True

    def _record_boil_start(self, particle):
        if particle.t_boil_start is None:
            particle.t_boil_start = particle.t + self.dt
            particle.z_boil_start = particle.z

    def update_temperature(self, particle, T, uz, ur, rho, muv, k):
        """
        Energy-conserving explicit thermal update over one particle timestep.

        q_net is frozen over dt (first-order explicit), but energy that crosses
        Tm or Tb is carried into the next phase instead of being discarded.
        """
        if particle.dp <= self.dp_cutoff or particle.mp <= 0:
            particle.evaporated = True
            return

        q_conv, q_rad, q_net = self.heat_rates(particle, T, uz, ur, rho, muv, k)
        particle.q_conv, particle.q_rad, particle.q_net = q_conv, q_rad, q_net
        particle.E_conv += q_conv * self.dt
        particle.E_rad += q_rad * self.dt
        particle.E_net += q_net * self.dt

        if abs(q_net) <= 1e-30:
            return

        energy = q_net * self.dt

        # ------------------------------------------------------------------
        # Heating
        # ------------------------------------------------------------------
        if energy > 0:
            # 1) Solid sensible heating to melting point.
            if particle.Tp < Tmp:
                Cp = max(self.Cpp(particle), 1e-12)
                needed = particle.mp * Cp * (Tmp - particle.Tp)
                if energy < needed:
                    particle.Tp += energy / (particle.mp * Cp)
                    return
                particle.Tp = Tmp
                energy -= needed
                self._record_melt_start(particle)

            # 2) Latent melting.
            if particle.x < 1.0:
                self._record_melt_start(particle)
                needed = particle.mp * Hm * (1.0 - particle.x)
                if energy < needed:
                    particle.x += energy / (particle.mp * Hm)
                    particle.Tp = Tmp
                    self._update_diameter_from_mass_and_fraction(particle)
                    return
                particle.x = 1.0
                particle.Tp = Tmp
                energy -= needed
                self._update_diameter_from_mass_and_fraction(particle)
                self._record_full_melt(particle)

            # 3) Liquid sensible heating to boiling point.
            if particle.Tp < Tbp:
                Cp = max(self.Cpp(particle), 1e-12)
                needed = particle.mp * Cp * (Tbp - particle.Tp)
                if energy < needed:
                    particle.Tp += energy / (particle.mp * Cp)
                    return
                particle.Tp = Tbp
                energy -= needed
                self._record_boil_start(particle)

            # 4) Evaporation at boiling point.
            if energy > 0 and particle.Tp >= Tbp:
                self._record_boil_start(particle)
                particle.x = 1.0
                dm = energy / Hb
                particle.mp = max(0.0, particle.mp - dm)
                cutoff_mass = self.rhop_liquid * np.pi / 6.0 * self.dp_cutoff**3
                if particle.mp <= cutoff_mass:
                    particle.mp = cutoff_mass
                    particle.dp = self.dp_cutoff
                    particle.evaporated = True
                else:
                    self._update_diameter_from_mass_and_fraction(particle)
                return

        # ------------------------------------------------------------------
        # Cooling / re-solidification
        # ------------------------------------------------------------------
        loss = -energy

        # 1) Cool liquid sensibly to Tm.
        if particle.Tp > Tmp:
            Cp = max(self.Cpp(particle), 1e-12)
            available = particle.mp * Cp * (particle.Tp - Tmp)
            if loss < available:
                particle.Tp -= loss / (particle.mp * Cp)
                return
            particle.Tp = Tmp
            loss -= available

        # 2) Re-solidify latent fraction if necessary.
        if particle.x > 0.0 and loss > 0.0:
            available = particle.mp * Hm * particle.x
            if loss < available:
                particle.x -= loss / (particle.mp * Hm)
                particle.Tp = Tmp
                self._update_diameter_from_mass_and_fraction(particle)
                return
            particle.x = 0.0
            particle.Tp = Tmp
            loss -= available
            self._update_diameter_from_mass_and_fraction(particle)

        # 3) Cool the solid below Tm.
        if loss > 0.0:
            Cp = max(self.Cpp(particle), 1e-12)
            particle.Tp -= loss / (particle.mp * Cp)

    # -------------------------------------------------------------------------
    # Motion and collision treatment
    # -------------------------------------------------------------------------

    def update_velocity(self, particle, uz, ur, rho, muv):
        if particle.dp <= self.dp_cutoff:
            particle.evaporated = True
            return

        rhop = self.rhop(particle)
        Cd = self.Cd(particle, rho, muv, uz, ur)
        Urel = self.Up(particle, uz, ur)
        g_eff = self.g * (1.0 - rho / rhop)

        particle.uzp += self.dt * (
            -0.75 * Cd * (particle.uzp - uz) * Urel * rho / (rhop * particle.dp)
            + g_eff
        )
        particle.urp += self.dt * (
            -0.75 * Cd * (particle.urp - ur) * Urel * rho / (rhop * particle.dp)
        )

    def _wall_geometry(self):
        t = self.torch
        carrier_inner = max(0.0, t.Lr_carrier - getattr(t, "t_carrier", 0.0))
        carrier_outer = t.Lr_carrier
        sheath_inner = t.Lr_sheath
        sheath_outer = min(t.Lr, t.Lr_sheath + getattr(t, "t_sheath", 0.0))
        return carrier_inner, carrier_outer, sheath_inner, sheath_outer

    @staticmethod
    def _segment_face_fraction(a0, a1, face, b0, b1, low, high, tol=1e-12):
        """Return segment fraction f for crossing a constant-a face.

        The segment is (a,b) = (a0,b0) + f[(a1,b1)-(a0,b0)].
        A valid hit has 0 < f <= 1 and the second coordinate lies on the
        finite wall face [low, high].
        """
        da = a1 - a0
        if abs(da) <= tol:
            return None
        f = (face - a0) / da
        if f <= 1e-10 or f > 1.0 + 1e-10:
            return None
        b = b0 + f * (b1 - b0)
        if low - tol <= b <= high + tol:
            return float(np.clip(f, 0.0, 1.0))
        return None

    def _first_wall_collision(self, z0, r0, z1, r1):
        """Return the first solid-wall collision along a trial segment.

        Returns
        -------
        dict or None
            {fraction, name, normal, z, r}.  ``normal`` is ``radial`` when
            the wall normal is in r and ``axial`` when it is in z.
        """
        ci, co, si, so = self._wall_geometry()
        t = self.torch
        candidates = []

        def radial_face(r_face, z_hi, name):
            f = self._segment_face_fraction(r0, r1, r_face, z0, z1, 0.0, z_hi)
            if f is not None:
                candidates.append((f, name, "radial", z0 + f * (z1 - z0), r_face))

        def axial_face(z_face, r_lo, r_hi, name):
            f = self._segment_face_fraction(z0, z1, z_face, r0, r1, r_lo, r_hi)
            if f is not None:
                candidates.append((f, name, "axial", z_face, r0 + f * (r1 - r0)))

        # Outer quartz wall.
        radial_face(t.Lr, t.Lz, "outer_wall")

        # Carrier tube: both radial faces and the downstream lip.
        radial_face(ci, t.Lz_carrier, "carrier_inner_wall")
        radial_face(co, t.Lz_carrier, "carrier_outer_wall")
        axial_face(t.Lz_carrier, ci, co, "carrier_lip")

        # Sheath tube: both radial faces and the downstream lip.
        radial_face(si, t.Lz_sheath, "sheath_inner_wall")
        radial_face(so, t.Lz_sheath, "sheath_outer_wall")
        axial_face(t.Lz_sheath, si, so, "sheath_lip")

        if not candidates:
            return None

        f, name, normal, zh, rh = min(candidates, key=lambda item: item[0])
        return {
            "fraction": f,
            "name": name,
            "normal": normal,
            "z": float(zh),
            "r": float(rh),
        }

    def _record_wall_hit(self, particle, wall_name, z_hit, r_hit, t_hit):
        particle.hit_wall = True
        particle.wall_hits += 1
        particle.wall_name = wall_name
        if particle.wall_hit_z is None:
            particle.wall_hit_z = float(z_hit)
            particle.wall_hit_r = float(r_hit)
        particle.wall_events.append({
            "time_s": float(t_hit),
            "z_m": float(z_hit),
            "r_m": float(r_hit),
            "wall": str(wall_name),
        })

    def _reflect_velocity(self, particle, normal):
        """Apply restitution to the normal velocity and friction tangentially."""
        if normal == "radial":
            particle.urp = -self.wall_restitution * particle.urp
            particle.uzp *= self.wall_friction
        elif normal == "axial":
            particle.uzp = -self.wall_restitution * particle.uzp
            particle.urp *= self.wall_friction
        else:
            raise ValueError("normal must be 'radial' or 'axial'.")

    def _handle_wall_hit(self, particle, hit, t_hit):
        self._record_wall_hit(
            particle, hit["name"], hit["z"], hit["r"], t_hit
        )
        particle.z = hit["z"]
        particle.r = hit["r"]

        if self.wall_mode == "terminate":
            particle.wall_terminated = True
            return

        self._reflect_velocity(particle, hit["normal"])

    def update_position(self, particle):
        """Advance position and resolve wall reflections within the same timestep.

        In ``reflect`` mode, a collision is an event, not a terminal state.  The
        particle is moved to the exact impact point, its velocity is reflected,
        and the unused fraction of the timestep is integrated after the bounce.
        A small loop permits corner/double impacts without allowing an infinite
        collision loop.
        """
        if particle.dp <= self.dp_cutoff:
            particle.evaporated = True
            return

        remaining = self.dt
        elapsed = 0.0
        max_collisions = 6
        t_step_start = particle.t - self.dt  # step() increments time before position update

        for _ in range(max_collisions):
            z0, r0 = float(particle.z), float(particle.r)
            z1 = z0 + remaining * particle.uzp
            r1 = r0 + remaining * particle.urp

            # Candidate wall collision.
            wall_hit = self._first_wall_collision(z0, r0, z1, r1)
            wall_fraction = np.inf if wall_hit is None else wall_hit["fraction"]

            # Axis crossing is symmetry, not a wall collision.
            axis_fraction = np.inf
            if r1 < 0.0 and r1 != r0:
                f = (0.0 - r0) / (r1 - r0)
                if 1e-10 < f <= 1.0:
                    axis_fraction = f

            # Upstream and outlet crossings must compete with wall impacts.
            upstream_fraction = np.inf
            if z1 <= 0.0 and z1 != z0:
                f = (0.0 - z0) / (z1 - z0)
                if 1e-10 < f <= 1.0:
                    upstream_fraction = f

            outlet_fraction = np.inf
            if z1 >= self.torch.Lz and z1 != z0:
                f = (self.torch.Lz - z0) / (z1 - z0)
                if 1e-10 < f <= 1.0:
                    outlet_fraction = f

            first = min(wall_fraction, axis_fraction, upstream_fraction, outlet_fraction)

            # No boundary crossing: accept the full remaining displacement.
            if not np.isfinite(first):
                particle.z = z1
                particle.r = r1
                return

            # Advance exactly to the first boundary event.
            z_hit = z0 + first * (z1 - z0)
            r_hit = r0 + first * (r1 - r0)
            used = first * remaining
            elapsed += used
            remaining *= max(0.0, 1.0 - first)

            if first == upstream_fraction:
                particle.z = 0.0
                particle.r = float(np.clip(r_hit, 0.0, self.torch.Lr))
                particle.exit_inlet = True
                particle.escaped = True
                return

            if first == outlet_fraction:
                particle.z = self.torch.Lz
                particle.r = float(np.clip(r_hit, 0.0, self.torch.Lr))
                particle.end = True
                particle.escaped = True
                return

            if first == axis_fraction:
                particle.z = float(z_hit)
                particle.r = 0.0
                particle.urp = abs(particle.urp)
                if remaining <= 1e-15:
                    return
                continue

            # Solid-wall impact.
            t_hit = t_step_start + elapsed
            self._handle_wall_hit(particle, wall_hit, t_hit)
            if self.wall_mode == "terminate" or remaining <= 1e-15:
                return

        # Corner protection.  If too many collisions occur inside one dt, keep
        # the last valid boundary position and continue on the next timestep.
        return

    # -------------------------------------------------------------------------
    # Diagnostics and summary
    # -------------------------------------------------------------------------

    def update_diagnostics(self, particle, T, uz, ur, rho, muv):
        Re = self.Rep(particle, rho, muv, uz, ur)
        Nu = self.Nu(particle, rho, muv, uz, ur)
        Urel = self.Up(particle, uz, ur)

        particle.Tgas = float(T)
        particle.Re = float(Re)
        particle.Nu = float(Nu)
        particle.Tp_max = max(particle.Tp_max, float(particle.Tp))
        particle.Tgas_max = max(particle.Tgas_max, float(T))
        particle.x_max = max(particle.x_max, float(particle.x))
        particle.r_min = min(particle.r_min, float(particle.r))
        particle.r_max = max(particle.r_max, float(particle.r))
        particle.Re_max = max(particle.Re_max, float(Re))
        particle.Nu_max = max(particle.Nu_max, float(Nu))
        particle.Urel_max = max(particle.Urel_max, float(Urel))

        if T >= self.hot_temperature:
            particle.hot_time += self.dt
        particle.thermal_exposure += max(float(T) - Tmp, 0.0) * self.dt

    def fate(self, particle):
        if particle.evaporated:
            return "fully_evaporated"
        if particle.wall_terminated:
            return "wall_hit"
        if particle.exit_inlet:
            return "upstream_escape"
        if particle.end:
            return "outlet"
        if particle.max_steps_reached:
            return "max_steps"
        return "active"

    def summary(self, particle):
        self.prepare_particle(particle)
        mass_remaining = 100.0 * particle.mp / particle.mp0 if particle.mp0 > 0 else np.nan
        mass_loss = 100.0 - mass_remaining if np.isfinite(mass_remaining) else np.nan
        return {
            "residence_time_s": particle.t,
            "Tp_max_K": particle.Tp_max,
            "Tgas_max_K": particle.Tgas_max,
            "x_final": particle.x,
            "x_max": particle.x_max,
            "fully_melted": bool(particle.fully_melted),
            "spheroidized": bool(particle.spheroid),
            "t_melt_start_s": particle.t_melt_start,
            "z_melt_start_m": particle.z_melt_start,
            "t_full_melt_s": particle.t_fully_melted,
            "z_full_melt_m": particle.z_fully_melted,
            "t_boil_start_s": particle.t_boil_start,
            "z_boil_start_m": particle.z_boil_start,
            "dp_final_m": particle.dp,
            "mass_remaining_pct": mass_remaining,
            "mass_loss_pct": mass_loss,
            "hot_time_s": particle.hot_time,
            "thermal_exposure_Ks": particle.thermal_exposure,
            "E_conv_J": particle.E_conv,
            "E_rad_J": particle.E_rad,
            "E_net_J": particle.E_net,
            "Re_max": particle.Re_max,
            "Nu_max": particle.Nu_max,
            "Urel_max_m_s": particle.Urel_max,
            "r_final_m": particle.r,
            "r_min_m": particle.r_min,
            "r_max_m": particle.r_max,
            "hit_wall": bool(particle.hit_wall),
            "wall_hits": particle.wall_hits,
            "wall_name": particle.wall_name,
            "wall_hit_z_m": particle.wall_hit_z,
            "wall_hit_r_m": particle.wall_hit_r,
            "evaporated": bool(particle.evaporated),
            "exit_inlet": bool(particle.exit_inlet),
            "upstream_lost": bool(particle.exit_inlet),
            "reached_outlet": bool(particle.end),
            "max_steps_reached": bool(particle.max_steps_reached),
            "fate": self.fate(particle),
        }


# =============================================================================
# Compact detailed trajectory study (keeps your current workflow)
# =============================================================================

FONT_SIZE = 9
LINE_WIDTH = 1.5
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "axes.linewidth": 0.8,
    "lines.linewidth": LINE_WIDTH,
    "savefig.dpi": 600,
})


class ParticleTrajectoryStudy:
    """Detailed histories for a small number of representative particles."""

    def __init__(
        self, solver, particle_sizes_um, particle_class, particle_z0, particle_r0,
        uzp0, urp0, Tp0=300.0, max_steps=50000,
    ):
        self.solver = solver
        self.fields = solver.fields
        self.torch = solver.torch
        self.particle_class = particle_class
        self.particle_sizes_um = np.asarray(particle_sizes_um, dtype=float)
        self.particle_sizes_m = self.particle_sizes_um * 1e-6
        self.particle_z0 = np.asarray(particle_z0, dtype=float)
        self.particle_r0 = np.asarray(particle_r0, dtype=float)
        self.uzp0 = uzp0
        self.urp0 = urp0
        self.Tp0 = Tp0
        self.max_steps = max_steps
        self.results = {}
        if not (len(self.particle_sizes_um) == len(self.particle_z0) == len(self.particle_r0)):
            raise ValueError("particle_sizes_um, particle_z0 and particle_r0 must have equal length.")

    def run(self):
        old = self.solver.store_history
        self.solver.store_history = True
        self.results = {}
        try:
            for i, (dp_um, z0, r0) in enumerate(zip(self.particle_sizes_um, self.particle_z0, self.particle_r0)):
                p = self.particle_class(
                    dp0=dp_um * 1e-6, dp=dp_um * 1e-6, z=z0, r=r0,
                    uzp=self.uzp0, urp=self.urp0, Tp=self.Tp0,
                )
                h = self.solver.solve(p, max_steps=self.max_steps)
                key = f"P{i + 1:03d}"
                self.results[key] = {
                    "particle": copy.deepcopy(p),
                    "history": h,
                    "t": h[:, 0], "z": h[:, 1], "r": h[:, 2],
                    "uzp": h[:, 3], "urp": h[:, 4], "Tp": h[:, 5],
                    "dp": h[:, 6], "x": h[:, 7], "spheroid": h[:, 8],
                    "mp": h[:, 9], "Tgas": h[:, 10], "q_conv": h[:, 11],
                    "q_rad": h[:, 12], "q_net": h[:, 13], "Re": h[:, 14], "Nu": h[:, 15],
                    "label": f"{dp_um:g} μm, r₀={r0*1e3:.2f} mm",
                    "summary": self.solver.summary(p),
                }
        finally:
            self.solver.store_history = old
        return self.results

    def print_summary(self):
        if not self.results:
            self.run()
        for key, result in self.results.items():
            s = result["summary"]
            print(f"\n{key}: {result['label']}")
            print(f"  fate             : {s['fate']}")
            print(f"  fully melted     : {s['fully_melted']}")
            print(f"  Tmax particle    : {s['Tp_max_K']:.1f} K")
            print(f"  mass loss        : {s['mass_loss_pct']:.3f} %")
            print(f"  residence time   : {1e3*s['residence_time_s']:.3f} ms")
            print(f"  wall interaction : {s['hit_wall']}")


# =============================================================================
# Power x diameter x radial-position parametric study
# =============================================================================

class ParticleParametricStudy:
    """
    Sweep plasma power, initial particle diameter and injection radius.

    Parameters
    ----------
    power_cases : dict
        Mapping {power_kW: saved_state_name}. Names may be given with or
        without saved_states/ and .npz.
    mass_loss_limit_pct : float
        Analysis criterion used only to label a fully molten particle as
        'successful'. Keep this configurable; it is not a material constant.
    """

    OUTCOME_ORDER = [
        "unmelted", "partially_melted", "successful",
        "excessive_evaporation", "fully_evaporated",
        "wall_hit", "upstream_escape", "max_steps",
    ]

    def __init__(
        self,
        power_cases,
        diameters_um,
        radial_positions_mm,
        torch,
        parameterfs,
        particle_class=Particle,
        dt=2e-4,
        z0=0.001,
        uzp0=None,
        urp0=0.0,
        Tp0=300.0,
        max_steps=50000,
        wall_mode="terminate",
        wall_restitution=0.2,
        wall_friction=0.8,
        hot_temperature=8000.0,
        mass_loss_limit_pct=10.0,
        output_dir="particle_results",
    ):
        self.power_cases = dict(sorted(power_cases.items(), key=lambda kv: float(kv[0])))
        self.diameters_um = np.asarray(diameters_um, dtype=float)
        self.radial_positions_mm = np.asarray(radial_positions_mm, dtype=float)
        self.torch = torch
        self.parameterfs = parameterfs
        self.particle_class = particle_class
        self.dt = float(dt)
        self.z0 = float(z0)
        self.uzp0 = float(torch.U_carrier if uzp0 is None else uzp0)
        self.urp0 = float(urp0)
        self.Tp0 = float(Tp0)
        self.max_steps = int(max_steps)
        self.wall_mode = wall_mode
        self.wall_restitution = float(wall_restitution)
        self.wall_friction = float(wall_friction)
        self.hot_temperature = float(hot_temperature)
        self.mass_loss_limit_pct = float(mass_loss_limit_pct)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame()

        if np.any(self.diameters_um <= 0):
            raise ValueError("All particle diameters must be positive.")
        if np.any(self.radial_positions_mm < 0):
            raise ValueError("Injection radii cannot be negative.")

        carrier_inner_mm = (torch.Lr_carrier - getattr(torch, "t_carrier", 0.0)) * 1e3
        if np.any(self.radial_positions_mm >= carrier_inner_mm):
            raise ValueError(
                f"All r0 values must lie inside the carrier gas bore: r0 < {carrier_inner_mm:.3f} mm."
            )

    def _make_solver(self, fields, dt=None, store_history=False):
        return ParticleSolver(
            fields=fields,
            torch=self.torch,
            dt=self.dt if dt is None else dt,
            parameterfs=self.parameterfs,
            store_history=store_history,
            wall_mode=self.wall_mode,
            wall_restitution=self.wall_restitution,
            wall_friction=self.wall_friction,
            hot_temperature=self.hot_temperature,
        )

    def _make_particle(self, dp_um, r0_mm):
        dp = float(dp_um) * 1e-6
        return self.particle_class(
            dp0=dp, dp=dp, z=self.z0, r=float(r0_mm) * 1e-3,
            uzp=self.uzp0, urp=self.urp0, Tp=self.Tp0,
        )

    def _classify(self, summary):
        if summary["evaporated"]:
            return "fully_evaporated"
        if self.wall_mode == "terminate" and summary["hit_wall"]:
            return "wall_hit"
        if summary["exit_inlet"]:
            return "upstream_escape"
        if summary["max_steps_reached"]:
            return "max_steps"
        if summary["x_max"] <= 1e-6:
            return "unmelted"
        if not summary["fully_melted"]:
            return "partially_melted"
        if summary["mass_loss_pct"] > self.mass_loss_limit_pct:
            return "excessive_evaporation"
        return "successful"

    def run(self, save_csv=True, verbose=True, filename="particle_parametric_results.csv"):
        rows = []
        n_total = len(self.power_cases) * len(self.diameters_um) * len(self.radial_positions_mm)
        case_number = 0

        for power_kw, state_name in self.power_cases.items():
            if verbose:
                print(f"\nLoading {power_kw:g} kW plasma field: {state_name}")
            fields = PlasmaFields.load(state_name)
            solver = self._make_solver(fields, store_history=False)

            for dp_um in self.diameters_um:
                for r0_mm in self.radial_positions_mm:
                    case_number += 1
                    p = self._make_particle(dp_um, r0_mm)
                    solver.solve(p, max_steps=self.max_steps)
                    s = solver.summary(p)
                    outcome = self._classify(s)
                    successful = outcome == "successful"

                    rows.append({
                        "power_kW": float(power_kw),
                        "state_file": str(state_name),
                        "dp0_um": float(dp_um),
                        "r0_mm": float(r0_mm),
                        "z0_mm": self.z0 * 1e3,
                        "uzp0_m_s": self.uzp0,
                        "dt_s": self.dt,
                        "Tp_max_K": s["Tp_max_K"],
                        "Tgas_max_K": s["Tgas_max_K"],
                        "x_final": s["x_final"],
                        "x_max": s["x_max"],
                        "fully_melted": s["fully_melted"],
                        "spheroidized": s["spheroidized"],
                        "t_melt_start_ms": self._ms(s["t_melt_start_s"]),
                        "z_melt_start_mm": self._mm(s["z_melt_start_m"]),
                        "t_full_melt_ms": self._ms(s["t_full_melt_s"]),
                        "z_full_melt_mm": self._mm(s["z_full_melt_m"]),
                        "t_boil_start_ms": self._ms(s["t_boil_start_s"]),
                        "z_boil_start_mm": self._mm(s["z_boil_start_m"]),
                        "residence_time_ms": 1e3 * s["residence_time_s"],
                        "hot_time_ms": 1e3 * s["hot_time_s"],
                        "thermal_exposure_Ks": s["thermal_exposure_Ks"],
                        "dp_final_um": 1e6 * s["dp_final_m"],
                        "mass_remaining_pct": s["mass_remaining_pct"],
                        "mass_loss_pct": s["mass_loss_pct"],
                        "E_conv_J": s["E_conv_J"],
                        "E_rad_J": s["E_rad_J"],
                        "E_net_J": s["E_net_J"],
                        "Re_max": s["Re_max"],
                        "Nu_max": s["Nu_max"],
                        "Urel_max_m_s": s["Urel_max_m_s"],
                        "r_exit_mm": 1e3 * s["r_final_m"],
                        "r_min_mm": 1e3 * s["r_min_m"],
                        "r_max_mm": 1e3 * s["r_max_m"],
                        "radial_excursion_mm": 1e3 * max(
                            abs(s["r_min_m"] - r0_mm * 1e-3),
                            abs(s["r_max_m"] - r0_mm * 1e-3),
                        ),
                        "hit_wall": s["hit_wall"],
                        "wall_hits": s["wall_hits"],
                        "wall_name": s["wall_name"],
                        "upstream_lost": s["upstream_lost"],
                        "reached_outlet": s["reached_outlet"],
                        "max_steps_reached": s["max_steps_reached"],
                        "evaporated": s["evaporated"],
                        "fate": s["fate"],
                        "successful": successful,
                        "outcome": outcome,
                    })

                    if verbose and (case_number % max(1, n_total // 20) == 0 or case_number == n_total):
                        print(f"  {case_number:4d}/{n_total} cases complete")

        self.df = pd.DataFrame(rows)
        if save_csv:
            self.save_csv(filename=filename)
        return self.df

    @staticmethod
    def _ms(value):
        return np.nan if value is None else 1e3 * float(value)

    @staticmethod
    def _mm(value):
        return np.nan if value is None else 1e3 * float(value)

    def save_csv(self, filename="particle_parametric_results.csv"):
        self._require_results()
        path = self.output_dir / filename
        self.df.to_csv(path, index=False)
        print(f"Saved: {path}")
        return path

    def _require_results(self):
        if self.df.empty:
            raise RuntimeError("No parametric results yet. Call study.run() first.")

    # ---------------------------------------------------------------------
    # Tables / aggregated processing-window metrics
    # ---------------------------------------------------------------------

    def success_fraction_table(self):
        """Fraction of injection radii that are successful for each P-dp pair."""
        self._require_results()
        return self.df.pivot_table(
            index="power_kW", columns="dp0_um", values="successful", aggfunc="mean"
        ).sort_index().sort_index(axis=1)

    def aggregate_table(self, value, agg="mean"):
        self._require_results()
        if value not in self.df.columns:
            raise KeyError(f"Unknown result column: {value}")
        return self.df.pivot_table(
            index="power_kW", columns="dp0_um", values=value, aggfunc=agg
        ).sort_index().sort_index(axis=1)

    def outcome_counts(self):
        self._require_results()
        return self.df.groupby("outcome", observed=False).size().sort_values(ascending=False)

    # ---------------------------------------------------------------------
    # Journal-oriented plots
    # ---------------------------------------------------------------------

    @staticmethod
    def _centres_to_edges(x):
        x = np.asarray(x, dtype=float)
        if x.size == 1:
            dx = 1.0
            return np.array([x[0] - dx / 2, x[0] + dx / 2])
        edges = np.empty(x.size + 1)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
        return edges

    def plot_success_fraction_map(self, savepath=None):
        table = self.success_fraction_table()
        powers = table.index.to_numpy(float)
        diameters = table.columns.to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        mesh = ax.pcolormesh(
            self._centres_to_edges(diameters), self._centres_to_edges(powers),
            table.to_numpy(float), shading="flat", vmin=0.0, vmax=1.0, cmap="viridis",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Successful injection fraction, $S$")
        ax.set_xlabel("Initial particle diameter, $d_{p,0}$ [μm]")
        ax.set_ylabel("Plasma power [kW]")
        fig.tight_layout()
        return self._save_figure(fig, savepath, "success_fraction_map.png"), ax

    def plot_mass_loss_map(self, agg="mean", only_fully_melted=True, savepath=None):
        self._require_results()
        data = self.df.copy()
        if only_fully_melted:
            data = data[data["fully_melted"]]
        table = data.pivot_table(
            index="power_kW", columns="dp0_um", values="mass_loss_pct", aggfunc=agg
        ).sort_index().sort_index(axis=1)
        powers = table.index.to_numpy(float)
        diameters = table.columns.to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        mesh = ax.pcolormesh(
            self._centres_to_edges(diameters), self._centres_to_edges(powers),
            table.to_numpy(float), shading="flat", cmap="magma",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(f"{agg.capitalize()} particle mass loss [%]")
        ax.set_xlabel("Initial particle diameter, $d_{p,0}$ [μm]")
        ax.set_ylabel("Plasma power [kW]")
        fig.tight_layout()
        return self._save_figure(fig, savepath, f"mass_loss_map_{agg}.png"), ax

    def plot_outcome_map(self, r0_mm, savepath=None):
        self._require_results()
        r_available = np.sort(self.df["r0_mm"].unique())
        r_use = r_available[np.argmin(np.abs(r_available - r0_mm))]
        data = self.df[np.isclose(self.df["r0_mm"], r_use)].copy()
        code = {name: i for i, name in enumerate(self.OUTCOME_ORDER)}
        data["outcome_code"] = data["outcome"].map(code)
        table = data.pivot(index="power_kW", columns="dp0_um", values="outcome_code").sort_index().sort_index(axis=1)

        colors = ["#d9d9d9", "#fdae61", "#1a9850", "#d73027", "#7f0000", "#542788", "#3288bd", "#000000"]
        cmap = ListedColormap(colors[:len(self.OUTCOME_ORDER)])
        norm = BoundaryNorm(np.arange(-0.5, len(self.OUTCOME_ORDER) + 0.5, 1), cmap.N)
        powers = table.index.to_numpy(float)
        diameters = table.columns.to_numpy(float)

        fig, ax = plt.subplots(figsize=(5.7, 3.7))
        mesh = ax.pcolormesh(
            self._centres_to_edges(diameters), self._centres_to_edges(powers),
            table.to_numpy(float), cmap=cmap, norm=norm, shading="flat",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, ticks=np.arange(len(self.OUTCOME_ORDER)))
        cbar.ax.set_yticklabels([x.replace("_", " ") for x in self.OUTCOME_ORDER])
        ax.set_xlabel("Initial particle diameter, $d_{p,0}$ [μm]")
        ax.set_ylabel("Plasma power [kW]")
        ax.text(0.01, 1.02, f"$r_0$ = {r_use:.2f} mm", transform=ax.transAxes, ha="left", va="bottom")
        fig.tight_layout()
        return self._save_figure(fig, savepath, f"outcome_map_r{r_use:.2f}mm.png"), ax

    def plot_diameter_response(self, power_kW, r0_mm, savepath=None):
        self._require_results()
        p_use = min(self.df["power_kW"].unique(), key=lambda x: abs(x - power_kW))
        r_use = min(self.df["r0_mm"].unique(), key=lambda x: abs(x - r0_mm))
        d = self.df[np.isclose(self.df["power_kW"], p_use) & np.isclose(self.df["r0_mm"], r_use)].sort_values("dp0_um")

        fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
        axes[0, 0].plot(d["dp0_um"], d["Tp_max_K"], marker="o")
        axes[0, 0].set_ylabel("Maximum $T_p$ [K]")
        axes[0, 1].plot(d["dp0_um"], d["x_max"], marker="o")
        axes[0, 1].set_ylabel("Maximum liquid fraction")
        axes[1, 0].plot(d["dp0_um"], d["mass_loss_pct"], marker="o")
        axes[1, 0].set_ylabel("Mass loss [%]")
        axes[1, 1].plot(d["dp0_um"], d["residence_time_ms"], marker="o")
        axes[1, 1].set_ylabel("Residence time [ms]")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        axes[1, 0].set_xlabel("Initial particle diameter [μm]")
        axes[1, 1].set_xlabel("Initial particle diameter [μm]")
        fig.suptitle(f"{p_use:g} kW, $r_0$={r_use:.2f} mm", y=0.995, fontsize=FONT_SIZE)
        fig.tight_layout()
        return self._save_figure(fig, savepath, f"diameter_response_{p_use:g}kW_r{r_use:.2f}mm.png"), axes

    def plot_radial_response(self, power_kW, diameter_um, savepath=None):
        self._require_results()
        p_use = min(self.df["power_kW"].unique(), key=lambda x: abs(x - power_kW))
        dp_use = min(self.df["dp0_um"].unique(), key=lambda x: abs(x - diameter_um))
        d = self.df[np.isclose(self.df["power_kW"], p_use) & np.isclose(self.df["dp0_um"], dp_use)].sort_values("r0_mm")

        fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
        axes[0, 0].plot(d["r0_mm"], d["Tp_max_K"], marker="o")
        axes[0, 0].set_ylabel("Maximum $T_p$ [K]")
        axes[0, 1].plot(d["r0_mm"], d["x_max"], marker="o")
        axes[0, 1].set_ylabel("Maximum liquid fraction")
        axes[1, 0].plot(d["r0_mm"], d["mass_loss_pct"], marker="o")
        axes[1, 0].set_ylabel("Mass loss [%]")
        axes[1, 1].plot(d["r0_mm"], d["r_exit_mm"], marker="o")
        axes[1, 1].set_ylabel("Outlet radius [mm]")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        axes[1, 0].set_xlabel("Injection radius, $r_0$ [mm]")
        axes[1, 1].set_xlabel("Injection radius, $r_0$ [mm]")
        fig.suptitle(f"{p_use:g} kW, $d_{{p,0}}$={dp_use:g} μm", y=0.995, fontsize=FONT_SIZE)
        fig.tight_layout()
        return self._save_figure(fig, savepath, f"radial_response_{p_use:g}kW_{dp_use:g}um.png"), axes

    def _save_figure(self, fig, savepath, default_name):
        if savepath is False:
            return fig
        path = self.output_dir / default_name if savepath is None else Path(savepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
        return fig

    # ---------------------------------------------------------------------
    # Detailed single case and timestep-independence study
    # ---------------------------------------------------------------------

    def run_case(self, power_kW, diameter_um, r0_mm, dt=None, store_history=True):
        p_use = min(self.power_cases, key=lambda x: abs(float(x) - float(power_kW)))
        fields = PlasmaFields.load(self.power_cases[p_use])
        solver = self._make_solver(fields, dt=self.dt if dt is None else dt, store_history=store_history)
        particle = self._make_particle(diameter_um, r0_mm)
        history = solver.solve(particle, max_steps=self.max_steps)
        return particle, history, solver.summary(particle), solver

    def timestep_study(self, dt_values, cases, save_csv=True):
        """
        cases: iterable of (power_kW, diameter_um, r0_mm)
        """
        rows = []
        for power_kw, dp_um, r0_mm in cases:
            p_use = min(self.power_cases, key=lambda x: abs(float(x) - float(power_kw)))
            fields = PlasmaFields.load(self.power_cases[p_use])
            for dt in dt_values:
                solver = self._make_solver(fields, dt=float(dt), store_history=False)
                p = self._make_particle(dp_um, r0_mm)
                solver.solve(p, max_steps=max(self.max_steps, int(self.max_steps * self.dt / float(dt))))
                s = solver.summary(p)
                rows.append({
                    "power_kW": float(p_use), "dp0_um": float(dp_um), "r0_mm": float(r0_mm),
                    "dt_s": float(dt), "Tp_max_K": s["Tp_max_K"], "x_max": s["x_max"],
                    "mass_loss_pct": s["mass_loss_pct"],
                    "t_full_melt_ms": self._ms(s["t_full_melt_s"]),
                    "residence_time_ms": 1e3 * s["residence_time_s"],
                    "r_exit_mm": 1e3 * s["r_final_m"], "fate": s["fate"],
                })
        out = pd.DataFrame(rows)
        if save_csv:
            path = self.output_dir / "particle_timestep_study.csv"
            out.to_csv(path, index=False)
            print(f"Saved: {path}")
        return out

    