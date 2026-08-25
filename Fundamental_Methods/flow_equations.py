import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy as np
from Fundamental_Methods.helpers import StretchedGridInterpolator


class ICPState:
    def __init__(
        self,
        torch,
        grid,
        magclass,
        CFL, 
        parametersf
    ):
        """
        Stores the changing solution variables for the ICP torch solver.

        Cell-centred variables:
            p, pr, T, rho

        Axial-face variable:
            uz

        Radial-face variables:
            ur, G

        """

        self.grid = grid
        self.torch = torch
        self.mag = magclass

        self.temp_boundary = torch.temp_boundary
        self.p_atm = torch.p_atm
        # ------------------------------------------------------------
        # Inlet velocities
        # ------------------------------------------------------------
        self.U_inlet_main = torch.U_inlet_main
        self.U_sheath = torch.U_sheath
        self.U_carrier = torch.U_carrier

        # ------------------------------------------------------------
        # Main solution variables
        # ------------------------------------------------------------
        self.uz = np.ones_like(grid.Zuz) * self.U_inlet_main
        self.ur = np.zeros_like(grid.Zur)
        self.divs = np.zeros_like(grid.R[1:-1, 1:-1])

        self.pr = np.zeros_like(grid.Z)
        self.p = np.ones_like(grid.Z) * self.p_atm

        self.T = np.ones_like(grid.Z) * self.temp_boundary
        self.rho = parametersf["rhof"](self.T)
        self.A = np.zeros_like(self.T[1:-1, 1:-1])
        self.Fz = np.zeros_like(self.A)
        self.Fr = np.zeros_like(self.A)
        self.P = np.zeros_like(self.A)
        self.Hz = np.zeros_like(self.A)
        self.Hr = np.zeros_like(self.A)
        self.E = np.zeros_like(self.A)

        # ------------------------------------------------------------
        # Previous-time-step storage
        # ------------------------------------------------------------
        self.uz_prev = self.uz.copy()
        self.ur_prev = self.ur.copy()
        self.p_prev = self.p.copy()
        self.pr_prev = self.pr.copy()
        self.T_prev = self.T.copy()
        self.rho_prev = self.rho.copy()
        self.A_prev = self.A.copy()

        # ------------------------------------------------------------
        # Parameter functions
        # ------------------------------------------------------------
        self.muvf = parametersf["muvf"]
        self.kf = parametersf["kf"]
        self.Cpf = parametersf["Cpf"]
        self.hf = parametersf["hf"]
        self.Qrf = parametersf["Qrf"]
        self.Tf= parametersf["Tf"]
        self.rhof = parametersf["rhof"]

        # ------------------------------------------------------------
        # Apply initial boundary conditions
        # ------------------------------------------------------------
        self.CFL = CFL
        self.current_time = 0
        self.step = 0
        self.dt = 0
        self.apply_all_boundaries()
        self.store_previous()

        # ------------------------------------------------------------
        # Build ONCE
        # ------------------------------------------------------------
        self.POISSON, self.POISSON_LU = self.build_poisson_solver()



    def store_previous(self):
        """
        Store the current state as the previous time-step state.
        """

        self.uz_prev = self.uz.copy()
        self.ur_prev = self.ur.copy()
        self.p_prev = self.p.copy()
        self.pr_prev = self.pr.copy()
        self.T_prev = self.T.copy()
        self.rho_prev = self.rho.copy()
        self.A_prev = self.A.copy()

        return

    def update_density(self):
        """
        Update density from the current temperature field.
        """

        self.rho = self.rhof(self.T)

        return

    def apply_all_boundaries(self):
        """
        Apply velocity, temperature, and angular momentum boundary conditions.
        """

        self.uz, self.ur = self.apply_velocity_boundaries(self.uz, self.ur)
        self.T = self.apply_temperature_boundaries(self.T)
        self.update_density()

        return
    
    def _between(self, x, a, b, tol=1e-12):
        return (x >= a - tol) & (x <= b + tol)

    def _apply_uz_horizontal_wall_antisym(self, uz, r_wall, z_end, solid_side, tol=1e-12):
        """
        Apply antisymmetric uz across a horizontal wall r = constant.

        This gives no-slip for tangential velocity uz:
            uz_solid = -uz_fluid

        solid_side:
            "above" means solid is at r > r_wall
            "below" means solid is at r < r_wall
        """

        g = self.grid

        r = g.Ruz[0, :]
        z = g.Zuz[:, 0]

        # Avoid the end face z = z_end, because there uz is normal to the short wall
        zmask = z < z_end - tol

        if solid_side == "above":
            fluid_candidates = np.where(r < r_wall - tol)[0]
            solid_candidates = np.where(r > r_wall + tol)[0]

            if len(fluid_candidates) == 0 or len(solid_candidates) == 0:
                return uz

            j_fluid = fluid_candidates[-1]
            j_solid = solid_candidates[0]

        elif solid_side == "below":
            solid_candidates = np.where(r < r_wall - tol)[0]
            fluid_candidates = np.where(r > r_wall + tol)[0]

            if len(fluid_candidates) == 0 or len(solid_candidates) == 0:
                return uz

            j_solid = solid_candidates[-1]
            j_fluid = fluid_candidates[0]

        else:
            raise ValueError("solid_side must be 'above' or 'below'.")

        uz[zmask, j_solid] = -uz[zmask, j_fluid]

        return uz

    def _apply_ur_vertical_wall_antisym(self, ur, z_wall, r_inner, r_outer, tol=1e-12):
        """
        Apply antisymmetric ur across a vertical wall z = constant.

        This gives no-slip for tangential velocity ur:
            ur_solid = -ur_fluid

        For the carrier/sheath end wall:
            solid side is z < z_wall
            fluid side is z > z_wall
        """

        g = self.grid

        z = g.Zur[:, 0]

        solid_candidates = np.where(z < z_wall - tol)[0]
        fluid_candidates = np.where(z > z_wall + tol)[0]

        if len(solid_candidates) == 0 or len(fluid_candidates) == 0:
            return ur

        i_solid = solid_candidates[-1]
        i_fluid = fluid_candidates[0]

        rmask = self._between(g.Rur[i_solid, :], r_inner, r_outer, tol)

        ur[i_solid, rmask] = -ur[i_fluid, rmask]

        return ur

    def _zero_ur_on_horizontal_wall(self, ur, r_wall, z_end, tol=1e-12):
        """
        Set normal radial velocity to zero on a horizontal wall.
        """

        g = self.grid

        mask = (
            np.isclose(g.Rur, r_wall, atol=tol)
            &
            (g.Zur <= z_end + tol)
        )

        ur[mask] = 0.0

        return ur

    def _zero_uz_on_vertical_wall(self, uz, z_wall, r_inner, r_outer, tol=1e-12):
        """
        Set normal axial velocity to zero on a vertical end wall.
        """

        g = self.grid

        mask = (
            np.isclose(g.Zuz, z_wall, atol=tol)
            &
            self._between(g.Ruz, r_inner, r_outer, tol)
        )

        uz[mask] = 0.0

        return uz

    def apply_velocity_boundaries(self, uz, ur):
        """
        Apply velocity boundary conditions.

        Normal velocity on solid walls:
            set directly to zero.

        Tangential velocity next to solid walls:
            use antisymmetric ghost-type condition:
                u_solid = -u_fluid

        For horizontal walls:
            uz is tangential.
            ur is normal.

        For vertical end walls:
            uz is normal.
            ur is tangential.
        """

        g = self.grid
        t = self.torch

        tol = 1e-12

        # ------------------------------------------------------------
        # Geometry radii
        # ------------------------------------------------------------
        r_carrier_outer = t.Lr_carrier
        r_carrier_inner = t.Lr_carrier - t.t_carrier

        r_sheath_inner = t.Lr_sheath
        r_sheath_outer = t.Lr_sheath + t.t_sheath

        # ------------------------------------------------------------
        # Global/basic boundary conditions
        # ------------------------------------------------------------

        # Inlet
        uz[0, 1:-1] = self.U_inlet_main

        # Outlet: Neumann
        uz[-1, 1:-1] = uz[-2, 1:-1]
        ur[-1, 1:-1] = ur[-2, 1:-1]

        # Axis
        uz[:, 0] = uz[:, 1]
        ur[:, 0] = 0.0

        # Outer wall
        uz[:, -1] = 0.0
        ur[:, -1] = 0.0

        # ------------------------------------------------------------
        # Inlet regions
        # ------------------------------------------------------------

        mask_carrier_core_r = g.Rur[0, :] < r_carrier_inner
        mask_carrier_core_z = g.Ruz[0, :] < r_carrier_inner

        mask_sheath_inlet_r = g.Rur[0, :] > r_sheath_outer
        mask_sheath_inlet_z = g.Ruz[0, :] > r_sheath_outer

        uz[0, mask_carrier_core_z] = self.U_carrier
        ur[0, mask_carrier_core_r] = 0.0

        uz[0, mask_sheath_inlet_z] = self.U_sheath
        ur[0, mask_sheath_inlet_r] = 0.0

        # ------------------------------------------------------------
        # First: zero all velocities inside solid wall zones
        # ------------------------------------------------------------

        mask_carrier_wall_z = (
            self._between(g.Ruz, r_carrier_inner, r_carrier_outer, tol)
            &
            (g.Zuz <= t.Lz_carrier + tol)
        )

        mask_carrier_wall_r = (
            self._between(g.Rur, r_carrier_inner, r_carrier_outer, tol)
            &
            (g.Zur <= t.Lz_carrier + tol)
        )

        mask_sheath_wall_z = (
            self._between(g.Ruz, r_sheath_inner, r_sheath_outer, tol)
            &
            (g.Zuz <= t.Lz_sheath + tol)
        )

        mask_sheath_wall_r = (
            self._between(g.Rur, r_sheath_inner, r_sheath_outer, tol)
            &
            (g.Zur <= t.Lz_sheath + tol)
        )

        uz[mask_carrier_wall_z] = 0.0
        ur[mask_carrier_wall_r] = 0.0

        uz[mask_sheath_wall_z] = 0.0
        ur[mask_sheath_wall_r] = 0.0

        # ------------------------------------------------------------
        # Horizontal carrier walls: no-slip for tangential uz
        # ------------------------------------------------------------

        # Inner carrier wall: solid is above the wall
        uz = self._apply_uz_horizontal_wall_antisym(
            uz,
            r_wall=r_carrier_inner,
            z_end=t.Lz_carrier,
            solid_side="above",
            tol=tol
        )

        # Outer carrier wall: solid is below the wall
        uz = self._apply_uz_horizontal_wall_antisym(
            uz,
            r_wall=r_carrier_outer,
            z_end=t.Lz_carrier,
            solid_side="below",
            tol=tol
        )

        # ------------------------------------------------------------
        # Horizontal sheath walls: no-slip for tangential uz
        # ------------------------------------------------------------

        # Inner sheath wall: solid is above the wall
        uz = self._apply_uz_horizontal_wall_antisym(
            uz,
            r_wall=r_sheath_inner,
            z_end=t.Lz_sheath,
            solid_side="above",
            tol=tol
        )

        # Outer sheath wall: solid is below the wall
        uz = self._apply_uz_horizontal_wall_antisym(
            uz,
            r_wall=r_sheath_outer,
            z_end=t.Lz_sheath,
            solid_side="below",
            tol=tol
        )

        # ------------------------------------------------------------
        # Horizontal walls: zero normal ur at actual radial wall faces
        # ------------------------------------------------------------

        ur = self._zero_ur_on_horizontal_wall(
            ur,
            r_wall=r_carrier_inner,
            z_end=t.Lz_carrier,
            tol=tol
        )

        ur = self._zero_ur_on_horizontal_wall(
            ur,
            r_wall=r_carrier_outer,
            z_end=t.Lz_carrier,
            tol=tol
        )

        ur = self._zero_ur_on_horizontal_wall(
            ur,
            r_wall=r_sheath_inner,
            z_end=t.Lz_sheath,
            tol=tol
        )

        ur = self._zero_ur_on_horizontal_wall(
            ur,
            r_wall=r_sheath_outer,
            z_end=t.Lz_sheath,
            tol=tol
        )

        # ------------------------------------------------------------
        # Vertical end walls: no-slip for tangential ur
        # ------------------------------------------------------------

        ur = self._apply_ur_vertical_wall_antisym(
            ur,
            z_wall=t.Lz_carrier,
            r_inner=r_carrier_inner,
            r_outer=r_carrier_outer,
            tol=tol
        )

        ur = self._apply_ur_vertical_wall_antisym(
            ur,
            z_wall=t.Lz_sheath,
            r_inner=r_sheath_inner,
            r_outer=r_sheath_outer,
            tol=tol
        )

        # ------------------------------------------------------------
        # Vertical end walls: zero normal uz at actual end faces
        # This must be AFTER the horizontal antisymmetry.
        # ------------------------------------------------------------

        uz = self._zero_uz_on_vertical_wall(
            uz,
            z_wall=t.Lz_carrier,
            r_inner=r_carrier_inner,
            r_outer=r_carrier_outer,
            tol=tol
        )

        uz = self._zero_uz_on_vertical_wall(
            uz,
            z_wall=t.Lz_sheath,
            r_inner=r_sheath_inner,
            r_outer=r_sheath_outer,
            tol=tol
        )

        uz = np.clip(uz, a_min=-100, a_max=100)
        ur = np.clip(ur, a_min=-30, a_max=30)
        return uz, ur

    def _copy_T_from_fluid_at_horizontal_wall(
        self,
        T,
        r_wall,
        z_end,
        fluid_side,
        tol=1e-12,
    ):
        """
        Copy temperature from the adjacent fluid cell onto a horizontal wall.

        Horizontal wall:
            r = r_wall
            0 <= z <= z_end

        This gives approximately:
            dT/dr = 0

        fluid_side:
            "below" means the fluid is at smaller r than the wall
            "above" means the fluid is at larger r than the wall
        """

        g = self.grid

        r = g.R[0, :]
        z = g.Z[:, 0]

        z_rows = np.where(z <= z_end + tol)[0]

        if len(z_rows) == 0:
            return T

        if fluid_side == "below":
            wall_candidates = np.where(r >= r_wall - tol)[0]
            fluid_candidates = np.where(r < r_wall - tol)[0]

            if len(wall_candidates) == 0 or len(fluid_candidates) == 0:
                return T

            j_wall = wall_candidates[0]
            j_fluid = fluid_candidates[-1]

        elif fluid_side == "above":
            wall_candidates = np.where(r <= r_wall + tol)[0]
            fluid_candidates = np.where(r > r_wall + tol)[0]

            if len(wall_candidates) == 0 or len(fluid_candidates) == 0:
                return T

            j_wall = wall_candidates[-1]
            j_fluid = fluid_candidates[0]

        else:
            raise ValueError("fluid_side must be 'below' or 'above'.")

        T[z_rows, j_wall] = T[z_rows, j_fluid]/2

        return T

    def _copy_T_from_fluid_at_vertical_wall(
        self,
        T,
        z_wall,
        r_min,
        r_max,
        fluid_side,
        tol=1e-12,
    ):
        """
        Copy temperature from the adjacent fluid cell onto a vertical end wall.

        Vertical wall:
            z = z_wall
            r_min <= r <= r_max

        This gives approximately:
            dT/dz = 0

        fluid_side:
            "right" means the fluid is downstream, z > z_wall
            "left" means the fluid is upstream, z < z_wall
        """

        g = self.grid

        z = g.Z[:, 0]
        r = g.R[0, :]

        r_cols = np.where((r >= r_min - tol) & (r <= r_max + tol))[0]

        if len(r_cols) == 0:
            return T

        if fluid_side == "right":
            wall_candidates = np.where(z <= z_wall + tol)[0]
            fluid_candidates = np.where(z > z_wall + tol)[0]

            if len(wall_candidates) == 0 or len(fluid_candidates) == 0:
                return T

            i_wall = wall_candidates[-1]
            i_fluid = fluid_candidates[0]

        elif fluid_side == "left":
            wall_candidates = np.where(z >= z_wall - tol)[0]
            fluid_candidates = np.where(z < z_wall - tol)[0]

            if len(wall_candidates) == 0 or len(fluid_candidates) == 0:
                return T

            i_wall = wall_candidates[0]
            i_fluid = fluid_candidates[-1]

        else:
            raise ValueError("fluid_side must be 'right' or 'left'.")

        T[i_wall, r_cols] = T[i_fluid, r_cols]/2


        return T

    def apply_temperature_boundaries_old(self, T):
        """
        Apply temperature boundary conditions.
        """
        g = self.grid
        t = self.torch

        # Inlet
        T[0, :] = self.temp_boundary
        T[1, :] = self.temp_boundary

        # Outlet: Neumann
        T[-1, :] = T[-2, :]

        # Outer wall
        T[:, -1] = self.temp_boundary
        T[:, -2] = self.temp_boundary

        # Axis: Neumann
        T[:, 0] = T[:, 1]

        # Carrier and sheath inlet/core regions
        # mask_carrier = (
        #     (g.R < (t.Lr_carrier - t.t_carrier))
        #     & (g.Z < t.Lz_carrier)
        # )

        # mask_sheath = (
        #     (g.R > (t.Lr_sheath + t.t_sheath))
        #     & (g.Z < t.Lz_sheath)
        # )

        mask_carrier = (
            (g.R > (t.Lr_carrier - t.t_carrier)) & (g.R < t.Lr_carrier)
            & (g.Z < t.Lz_carrier)
        )

        mask_sheath = (
            (g.R < (t.Lr_sheath + t.t_sheath)) & (g.R > (t.Lr_sheath))
            & (g.Z < t.Lz_sheath)
        )

        T[mask_carrier] = self.temp_boundary
        T[mask_sheath] = self.temp_boundary

        # Temperature floor for stability
        T[T < self.temp_boundary] = self.temp_boundary

        return T

    def apply_temperature_boundaries(self, T):
        """
        Apply temperature boundary conditions.

        Inlet:
            fixed temperature

        Outlet:
            zero-gradient

        Axis:
            symmetry

        Outer wall:
            fixed wall temperature or zero-gradient, depending on what you choose

        Interior carrier/sheath walls:
            copy temperature from adjacent fluid cell
        """

        g = self.grid
        t = self.torch

        # ------------------------------------------------------------
        # Basic domain boundaries
        # ------------------------------------------------------------

        # Inlet
        T[0, :] = self.temp_boundary

        # Outlet: zero-gradient
        T[-1, :] = T[-2, :]

        # Axis: symmetry
        T[:, 0] = T[:, 1]

        # Outer torch wall
        T[:, -1] = self.temp_boundary

        mask_carrier = (
            (g.R > (t.Lr_carrier - t.t_carrier)) &        # Added this.                     
            (g.R < t.Lr_carrier)
            & (g.Z < t.Lz_carrier)
        )

        mask_sheath = (
            (g.R < (t.Lr_sheath + t.t_sheath)) & (g.R > (t.Lr_sheath))
            & (g.Z < t.Lz_sheath)
        )

        T[mask_carrier] = self.temp_boundary      
        T[mask_sheath] = self.temp_boundary

        # ------------------------------------------------------------
        # Carrier inlet wall conditions
        # ------------------------------------------------------------

        # 1. Horizontal carrier wall at r = Lr_carrier
        #    Fluid is above this wall, so copy from smaller r.
        T = self._copy_T_from_fluid_at_horizontal_wall(
            T,
            r_wall=t.Lr_carrier,
            z_end=t.Lz_carrier,
            fluid_side="above",
        )


        # 2. Vertical carrier end wall at z = Lz_carrier
        #    Fluid is downstream of the wall, so copy from larger z.
        T = self._copy_T_from_fluid_at_vertical_wall(
            T,
            z_wall=t.Lz_carrier,
            r_min=t.Lr_carrier-t.t_carrier,
            r_max=t.Lr_carrier,
            fluid_side="right",
        )

        # ------------------------------------------------------------
        # Sheath inlet wall conditions
        # ------------------------------------------------------------

        # 3. Horizontal sheath wall at r = Lr_sheath - t_sheath
        #    Usually the sheath-side fluid is above this wall,
        #    so copy from larger r.
        T = self._copy_T_from_fluid_at_horizontal_wall(
            T,
            r_wall=t.Lr_sheath,
            z_end=t.Lz_sheath,
            fluid_side="below",
        )

        # 4. Vertical sheath end wall at z = Lz_sheath
        #    Fluid is downstream of the wall, so copy from larger z.
        T = self._copy_T_from_fluid_at_vertical_wall(
            T,
            z_wall=t.Lz_sheath,
            r_min=t.Lr_sheath,
            r_max=t.Lr_sheath + t.t_sheath,
            fluid_side="right",
        )

        # ------------------------------------------------------------
        # Temperature floor for stability
        # ------------------------------------------------------------
        T[T < self.temp_boundary] = self.temp_boundary

        return T

    def as_dict(self):
        """
        Return the state as a dictionary for saving.
        """
        divergence_s = self.divergence_s()
        divergence_s[-1, :] = 0
        return {
            "uz": self.uz,
            "ur": self.ur,
            "p": self.p,
            "T": self.T,
            "step": self.step,
            "time": self.current_time,
            "dt":self.dt,
            "rho":self.rho,
            "div": divergence_s,
            "Fz": self.Fz,
            "Fr": self.Fr,
            "P": self.P,
            "A": self.A,
            "Hz": self.Hz,
            "Hr": self.Hr,
            "E": self.E,
            "Z": self.grid.Z, "R": self.grid.R,
            "Zuz": self.grid.Zuz, "Ruz": self.grid.Ruz,
            "Zur": self.grid.Zur, "Rur": self.grid.Rur,
            "volume": self.grid.volume, "volume_uz": self.grid.volume_uz, 
            "volume_ur": self.grid.volume_ur, 
            "r_area": self.grid.r_area, "z_area": self.grid.z_area,
            "r_area_ur": self.grid.r_area_ur, "z_area_ur": self.grid.z_area_ur,
            "r_area_uz": self.grid.r_area_uz, "z_area_uz": self.grid.z_area_uz
        }

    def load_from_dict(self, data, new_grid=False, kx=1, ky=1):
        """
        Load state variables from a dictionary.

        """
        if not new_grid:
            self.uz = data["uz"].copy()
            self.ur = data["ur"].copy()
            self.p = data["p"].copy()
            self.T = data["T"].copy()
            self.rho = data["rho"].copy()

            self.step = int(np.asarray(data["step"]))
            self.current_time = float(np.asarray(data["time"]))
            self.dt = float(np.asarray(data["dt"]))

            self.Fz = data["Fz"].copy()
            self.Fr = data["Fr"].copy()
            self.P = data["P"].copy()
            self.A = data["A"].copy()
            self.Hz = data["Hz"].copy()
            self.Hr = data["Hr"].copy()
            self.E = data["E"].copy()

            self.uz_prev = self.uz.copy()
            self.ur_prev = self.ur.copy()
            self.p_prev = self.p.copy()
            self.T_prev = self.T.copy()
            self.rho_prev = self.rho.copy()

            self.grid.Z = data['Z'].copy(); self.grid.R = data['R'].copy()
            self.grid.Zuz = data['Zuz'].copy(); self.grid.Ruz = data["Ruz"].copy()
            self.grid.Zur = data["Zur"].copy(); self.grid.Rur = data["Rur"].copy()
            self.grid.volume = data["volume"]; self.grid.volume_uz = data["volume_uz"]
            self.grid.volume_ur = data["volume_ur"]; 
            self.grid.r_area = data["r_area"]; self.grid.z_area = data["z_area"]
            self.grid.z_area_uz = data["z_area_uz"]; self.grid.r_area_uz = data["r_area_uz"]
            self.grid.z_area_ur = data["z_area_ur"]; self.grid.r_area_ur = data["r_area_ur"]

        else:
            Z_old = data['Z'].copy(); R_old = data['R'].copy()
            Zuz_old = data['Zuz'].copy(); Ruz_old = data["Ruz"].copy()
            Zur_old = data["Zur"].copy(); Rur_old = data["Rur"].copy()

            g = self.grid

            # T and p grids
            interp_cells = StretchedGridInterpolator(
                R_old, Z_old, kx=kx, ky=ky, bounds_error=False
            )

            self.T = interp_cells(data["T"], g.R, g.Z)
            self.p = interp_cells(data["p"], g.R, g.Z)

            # ur
            interp_ur = StretchedGridInterpolator(
                Rur_old, Zur_old, kx=kx, ky=ky, bounds_error=False
            )

            self.ur = interp_ur(data["ur"], g.Rur, g.Zur)

            # uz
            interp_uz = StretchedGridInterpolator(
                Ruz_old, Zuz_old, kx=kx, ky=ky, bounds_error=False
            )

            self.uz = interp_uz(data["uz"], g.Ruz, g.Zuz)

            # electromagnetic field
            R_old_i = R_old[1:-1, 1:-1]
            Z_old_i = Z_old[1:-1, 1:-1]

            R_new_i = g.R[1:-1, 1:-1]
            Z_new_i = g.Z[1:-1, 1:-1]

            interp_int = StretchedGridInterpolator(
                R_old_i, Z_old_i, kx=kx, ky=ky, bounds_error=False
            )

            self.A = interp_int(data["A"], R_new_i, Z_new_i)
            self.P = interp_int(data["P"], R_new_i, Z_new_i)
            self.Fr = interp_int(data["Fr"], R_new_i, Z_new_i)
            self.Fz = interp_int(data["Fz"], R_new_i, Z_new_i)
            self.Hz = interp_int(data["Hz"], R_new_i, Z_new_i)
            self.Hr = interp_int(data["Hr"], R_new_i, Z_new_i)
            self.E = interp_int(data["E"], R_new_i, Z_new_i)

            self.step = int(np.asarray(data["step"]))
            self.current_time = float(np.asarray(data["time"]))
            self.dt = float(np.asarray(data["dt"]))

            # Very important after interpolation
            self.apply_all_boundaries()
            self.update_density()
            self.POISSON, self.POISSON_LU = self.build_poisson_solver()

        return
    
    def compute_dt(self):
        """
        Compute time step based on CFL condition
        Only on the basis of the interior of the torch.
        CFL is defined at all_variables
        """
        g = self.grid

        uz = 1 / 2 * (self.uz[1:, 1:-1] + self.uz[:-1, 1:-1])
        ur = 1 / 2 * (self.ur[1:-1, 1:] + self.ur[1:-1, :-1])

        # Convective time step limit
        dt_conv_x = (g.Zuz[1:, 1:-1]-g.Zuz[:-1, 1:-1]) / (np.abs(uz) + 1e-10)
        dt_conv_y = (g.Rur[1:-1, 1:]-g.Rur[1:-1, :-1]) / (np.abs(ur) + 1e-10)

        # Viscous time step limit (diffusion)
        mu = self.muvf(self.T[1:-1, 1:-1])
        dt_visc_z = 0.5 * self.rho[1:-1, 1:-1] * (g.Rur[1:-1, 1:]-g.Rur[1:-1, :-1])**2 / (mu + 1e-10)
        dt_visc_r = 0.5 * self.rho[1:-1, 1:-1] * (g.Zuz[1:, 1:-1]-g.Zuz[:-1, 1:-1])**2 / (mu + 1e-10)

        # Thermal diffusion time step limit
        k = self.kf(self.T[1:-1, 1:-1])
        cp = self.Cpf(self.T[1:-1, 1:-1])
        dt_thermal_z = 0.5 * self.rho[1:-1, 1:-1] * (g.Rur[1:-1, 1:]-g.Rur[1:-1, :-1])**2 * cp / (k + 1e-10)
        dt_thermal_r = 0.5 * self.rho[1:-1, 1:-1] * (g.Zuz[1:, 1:-1]-g.Zuz[:-1, 1:-1])**2 * cp / (k + 1e-10)

        # Combined time step
        dt_min = min(
            np.min(dt_conv_x),
            np.min(dt_conv_y),
            np.min(dt_visc_z),
            np.min(dt_visc_r),
            np.min(dt_thermal_z),
            np.min(dt_thermal_r)
        )

        times = [
            np.min(dt_conv_x),
            np.min(dt_conv_y),
            np.min(dt_visc_z),
            np.min(dt_visc_r),
            np.min(dt_thermal_z),
            np.min(dt_thermal_r)
        ]

        self.dt = self.CFL * dt_min

        # Put all timestep limits in a labelled list
        dt_limits = [
            ("convective_z / uz", dt_conv_x),
            ("convective_r / ur", dt_conv_y),
            ("viscous_z", dt_visc_z),
            ("viscous_r", dt_visc_r),
            ("thermal_z", dt_thermal_z),
            ("thermal_r", dt_thermal_r),
        ]

        # Find the minimum value for each mechanism
        min_values = []
        for name, arr in dt_limits:
            local_min = np.nanmin(arr)
            min_values.append((name, local_min))

        # Find which mechanism gives the global minimum
        limiting_name, dt_min = min(min_values, key=lambda x: x[1])

        # Final CFL-scaled timestep
        self.dt = self.CFL * dt_min

        # Store useful debugging information
        self.dt_limiter = limiting_name
        self.dt_limits = dict(min_values)

        return self.dt
    
    def compute_dt_thermal(self, dT_allowed=100.0):
        """
        Compute a time step for temperature-only solving.

        This assumes:
            - uz is fixed
            - ur is fixed
            - p is fixed
            - only T is advanced

        Limits included:
            1. axial advection
            2. radial advection
            3. thermal diffusion
            4. Joule/source heating temperature jump
        """

        g = self.grid
        eps = 1e-30

        # ------------------------------------------------------------
        # Cell-centred velocities
        # ------------------------------------------------------------
        uzc = 0.5 * (self.uz[1:, 1:-1] + self.uz[:-1, 1:-1])
        urc = 0.5 * (self.ur[1:-1, 1:] + self.ur[1:-1, :-1])

        # ------------------------------------------------------------
        # Local grid spacings on cell centres
        # ------------------------------------------------------------
        dz = g.Zuz[1:, 1:-1] - g.Zuz[:-1, 1:-1]
        dr = g.Rur[1:-1, 1:] - g.Rur[1:-1, :-1]

        # ------------------------------------------------------------
        # 1. Advection limits
        # ------------------------------------------------------------
        dt_adv_z = dz / (np.abs(uzc) + eps)
        dt_adv_r = dr / (np.abs(urc) + eps)

        # ------------------------------------------------------------
        # 2. Thermal diffusion limit
        # alpha = k / rho Cp
        # ------------------------------------------------------------
        T_int = self.T[1:-1, 1:-1]
        rho_int = self.rho[1:-1, 1:-1]

        k = self.kf(T_int)
        cp = self.Cpf(T_int)

        alpha = k / (rho_int * cp + eps)

        dt_diff = 0.5 / (
            alpha * (1.0 / (dz**2 + eps) + 1.0 / (dr**2 + eps)) + eps
        )

        # ------------------------------------------------------------
        # 3. Source limit
        # P is Joule heating per volume, shape is interior only.
        # Qrf is radiative/thermal loss per volume.
        # ------------------------------------------------------------
        q_net = self.P - self.Qrf(T_int)

        dTdt_source = q_net / (rho_int * cp + eps)

        dt_source = dT_allowed / (np.max(np.abs(dTdt_source)) + eps)

        # ------------------------------------------------------------
        # Label all limits
        # ------------------------------------------------------------
        dt_limits = [
            ("thermal_adv_z", np.nanmin(dt_adv_z)),
            ("thermal_adv_r", np.nanmin(dt_adv_r)),
            ("thermal_diffusion", np.nanmin(dt_diff)),
            ("thermal_source", dt_source),
        ]

        limiting_name, dt_min = min(dt_limits, key=lambda x: x[1])

        self.dt = self.CFL * dt_min

        self.dt_limiter = limiting_name
        self.dt_limits = dict(dt_limits)

        return self.dt

    def ur__center(self):
        """ur at the center cells"""
        return 1 / 2 * (self.ur[1:-1, 1:] + self.ur[1:-1, :-1])

    def uz__center(uz):
        """uz at the center cells"""
        return 1 / 2 * (uz[1:, 1:-1] + uz[:-1, 1:-1])
    
    def phi_faces(self, phi):
        """
        Calculates the value of property phi on the face between two cell centers.
        """
        z_faces = (phi[1:, 1:-1] + phi[:-1, 1:-1]) / 2
        r_faces = (phi[1:-1, 1:] + phi[1:-1, :-1]) / 2

        return z_faces, r_faces
    
    def phi_faces_upwind(self, phi):
        """
        Calculates the value of property phi on the face between two cells using upwind-ing.
        """
        z_faces = np.where(self.uz[:, 1:-1] > 0, phi[:-1, 1:-1], phi[1:, 1:-1])
        r_faces = np.where(self.ur[1:-1, :] > 0, phi[1:-1, :-1], phi[1:-1, 1:])
        return z_faces, r_faces

    def phi_faces_uz_cells(self, phi):
        """Only for the interior ux cells you are solving for. Simple averaging."""
        z_faces = phi[1:-1, 1:-1]
        r_faces = 0.25 * (phi[2:-1, 1:] + phi[1:-2, 1:] + phi[2:-1, 0:-1] + phi[1:-2, 0:-1])
        return z_faces, r_faces
    
    def u_faces_uz_cells(self):
        """
        Returns average velocity values at faces as follows: WEux, NSux, WEuy, NSuy
        Only for the interior ux cells."""
        WEuz, NSuz = 0.5 * (self.uz[1:, 1:-1] + self.uz[:-1, 1:-1]), 0.5 * (self.uz[1:-1, 1:] + self.uz[1:-1, :-1])
        NSur, WEur = 0.5 * (self.ur[1:, 1:-1] + self.ur[:-1, 1:-1]), 0.5 * (self.ur[:, 2:-1] + self.ur[:, 1:-2])
        return WEuz, NSuz, WEur, NSur

    def ur_center_uz_cells(self):
        """
        Returns the average uy velocity at all the ux locations.
        On the edges uses neumann boundary conditions.
        """
        average = np.zeros_like(self.uz)
        average[:, 1:-1] = 0.25 * (
                self.ur[1:, 1:] + self.ur[0:-1, 1:] + self.ur[0:-1, 0:-1] + self.ur[1:, 0:-1]
        )
        average[:, 0] = average[:, 1]
        average[:, -1] = average[:, -2]

        return average
    
    def phi_faces_ur_cells(self, phi):
        """
        Only for the interior cells you are solving for.  Simple averaging.
        """
        r_faces = phi[1:-1, 1:-1]
        z_faces = 0.25 * (phi[1:, 2:-1] + phi[1:, 1:-2] + phi[:-1, 2:-1] + phi[:-1, 1:-2])
        return z_faces, r_faces
    
    def u_faces_ur_cells(self):
        """
        Returns average velocity values at faces as follows: WEux, NSux, WEuy, NSuy
        Only for the interior uy cells.
        """
        NSuz, WEuz = 0.5 * (self.uz[1:-1, 1:] + self.uz[1:-1, :-1]), 0.5 * (self.uz[2:-1, :] + self.uz[1:-2, :])
        WEur, NSur = 0.5 * (self.ur[1:, 1:-1] + self.ur[:-1, 1:-1]), 0.5 * (self.ur[1:-1, 1:] + self.ur[1:-1, :-1])
        return WEuz, NSuz, WEur, NSur
    
    def uz_at_ur(self):
        """
        Returns the average ux velocity at all the uy locations.
        Uses neumann at the edges where 4 point average not possible.
        """
        average = np.zeros_like(self.ur)
        average[1:-1, :] = 0.25 * (
                self.uz[1:, 1:] + self.uz[0:-1, 1:] + self.uz[0:-1, 0:-1] + self.uz[1:, 0:-1]
        )
        average[0, :] = average[1, :]
        average[-1, :] = average[-2, :]

        return average

    def convective_flux(self, phi):
        """
        Has shape of ux and of uy.
        Returns the flux of property phi on cell faces.
        """
        phi_face = self.phi_faces(phi)
        z_flux = phi_face[0] * self.uz[:, 1:-1]
        r_flux = phi_face[1] * self.ur[1:-1, :]

        return z_flux, r_flux

    def convective_flux_upwind(self, phi):
        """
        Upwind scheme for convective fluxes for interior cells.
        Returns the flux of property phi on cell faces.
        """
        phi_face = self.phi_faces_upwind(phi)
        z_flux = phi_face[0] * self.uz[:, 1:-1]
        r_flux = phi_face[1] * self.ur[1:-1, :]
        return z_flux, r_flux

    def flux_uz_cells(self):
        """
        Returns the flux for the uz velocity cells.
        Only interior.
        """
        flux_phi_cells_we, flux_phi_cells_ns = self.convective_flux(self.rho)
        flux_we = 0.5 * (flux_phi_cells_we[1:, :] + flux_phi_cells_we[:-1, :])

        flux_ns = 0.5 * (flux_phi_cells_ns[1:, :] + flux_phi_cells_ns[:-1, :])

        return flux_we, flux_ns

    def flux_uz_cells_upwind(self):
        """
        Returns the flux for the ux velocity cells.
        Only interior.
        """
        flux_phi_cells_we, flux_phi_cells_ns = self.convective_flux_upwind(self.rho)
        flux_we = 0.5 * (flux_phi_cells_we[1:, :] + flux_phi_cells_we[:-1, :])
        flux_ns = 0.5 * (flux_phi_cells_ns[1:, :] + flux_phi_cells_ns[:-1, :])
        return flux_we, flux_ns

    def flux_ur_cells(self):
        """
        Returns the flux for the uy velocity cells.
        Only interior.
        """
        flux_phi_cells_we, flux_phi_cells_ns = self.convective_flux(self.rho)
        flux_ns = 0.5 * (flux_phi_cells_ns[:, 1:] + flux_phi_cells_ns[:, :-1])
        flux_we = 0.5 * (flux_phi_cells_we[:, 1:] + flux_phi_cells_we[:, :-1])

        return flux_we, flux_ns

    def flux_ur_cells_upwind(self):
        """
        Returns the flux for the uy velocity cells.
        Only interior.
        """
        flux_phi_cells_we, flux_phi_cells_ns = self.convective_flux_upwind(self.rho)
        flux_ns = 0.5 * (flux_phi_cells_ns[:, 1:] + flux_phi_cells_ns[:, :-1])
        flux_we = 0.5 * (flux_phi_cells_we[:, 1:] + flux_phi_cells_we[:, :-1])

        return flux_we, flux_ns

    def divergence_s(self):
        """
        Determines divergence of field.
        """
        rho_z_faces, rho_r_faces = self.phi_faces(self.rho)
        g = self.grid

        self.div = (
                (self.uz[1:, 1:-1]*rho_z_faces[1:, :]*g.z_area[1:, :] - self.uz[:-1, 1:-1]*rho_z_faces[:-1, :]*g.z_area[:-1, :]) +
                (self.ur[1:-1, 1:]*rho_r_faces[:, 1:]*g.r_area[:, 1:] - self.ur[1:-1, :-1]*rho_r_faces[:, :-1]*g.r_area[:, :-1])
        )

        self.div[g.mask_solid_inner] = 0

        return self.div / g.volume
    
    def divergence_outside(self, uz, ur):
        """
        Determines divergence of field.
        """
        rho_z_faces, rho_r_faces = self.phi_faces(self.rho)
        g = self.grid

        div = (
                (uz[1:, 1:-1]*rho_z_faces[1:, :]*g.z_area[1:, :] - uz[:-1, 1:-1]*rho_z_faces[:-1, :]*g.z_area[:-1, :]) +
                (ur[1:-1, 1:]*rho_r_faces[:, 1:]*g.r_area[:, 1:] - ur[1:-1, :-1]*rho_r_faces[:, :-1]*g.r_area[:, :-1])
        )

        div[g.mask_solid_inner] = 0

        return div / g.volume
    
    @staticmethod
    def mass_streamfunction(uz, ur, rho, R, Z):
        """
        Computes mass streamfunction psi for axisymmetric flow.
        Contours of psi are mass flow lines.

        Arrays are indexed as [z, r].
        """

        Nz, Nr = uz.shape
        psi = np.zeros((Nz, Nr))

        # Grid spacing (assumed structured)
        dz = Z[1, 0] - Z[0, 0]
        dr = R[0, 1] - R[0, 0]

        # ---- Radial integration (primary) ----
        for i in range(Nz):
            psi[i, 0] = 0.0
            for j in range(1, Nr):
                r_face = 0.5 * (R[i, j] + R[i, j-1])
                rho_uz_face = 0.5 * (rho[i, j] * uz[i, j] +
                                    rho[i, j-1] * uz[i, j-1])
                psi[i, j] = psi[i, j-1] + r_face * rho_uz_face * dr

        return psi
    
    def power_coil(self):

        # self.P is the dissipated plasma power density.  
  
        self.P_coil = np.sum(self.grid.volume * self.P)   # Busse et al defintion.

        return self.P_coil
    
    def pressure_poisson_matrix_sparse(self): 
        """
        Build the pressure Poisson LHS matrix using a finite-volume stencil.

        Important:
        - Fluid cells next to solid cells use zero-normal-gradient pressure:
            dp/dn = 0
        This means no pressure-flux through the carrier/sheath walls.
        - Solid cells are decoupled using identity rows:
            p_solid = 0
        The actual value inside the solid is not physical.
        """

        g = self.grid

        Nj = g.Nj + 2
        Ni = g.Ni + 2
        n = Ni * Nj

        solid = g.mask_solid.astype(bool)

        def idx(i, j):
            return Nj * i + j

        rows = []
        cols = []
        data = []

        def add(row, col, value):
            rows.append(row)
            cols.append(col)
            data.append(value)

        # ------------------------------------------------------------
        # Loop over full pressure grid, including ghost cells
        # ------------------------------------------------------------
        for i in range(Ni):
            for j in range(Nj):

                ind = idx(i, j)

                # ------------------------------------------------------------
                # Domain boundary conditions
                # ------------------------------------------------------------

                # Inlet: dp/dz = 0
                if i == 0:
                    add(ind, ind, 1.0)
                    add(ind, idx(i + 1, j), -1.0)
                    continue

                # Outlet/reference: p = 0
                if i == Ni - 1:
                    add(ind, ind, 1.0)
                    continue

                # Axis: dp/dr = 0
                if j == 0:
                    add(ind, ind, 1.0)
                    add(ind, idx(i, j + 1), -1.0)
                    continue

                # Outer wall: dp/dr = 0
                if j == Nj - 1:
                    add(ind, ind, 1.0)
                    add(ind, idx(i, j - 1), -1.0)
                    continue

                # ------------------------------------------------------------
                # Solid cells: decouple them from the pressure solve
                # ------------------------------------------------------------
                if solid[i, j]:
                    add(ind, ind, 1.0)
                    continue

                # ------------------------------------------------------------
                # Finite-volume Poisson coefficients for fluid cells
                # ------------------------------------------------------------

                # Cell volume.  g.volume is interior only, so use i-1, j-1.
                vol = g.volume[i - 1, j - 1]

                # Face areas
                A_z_minus = g.z_area[i - 1, j - 1]
                A_z_plus  = g.z_area[i,     j - 1]

                A_r_minus = g.r_area[i - 1, j - 1]
                A_r_plus  = g.r_area[i - 1, j]

                # Distances between pressure cell centres
                dz_minus = g.Z[i, j] - g.Z[i - 1, j]
                dz_plus  = g.Z[i + 1, j] - g.Z[i, j]

                dr_minus = g.R[i, j] - g.R[i, j - 1]
                dr_plus  = g.R[i, j + 1] - g.R[i, j]

                # Diffusive/Poisson coefficients
                cS = A_z_minus / dz_minus / vol
                cN = A_z_plus  / dz_plus  / vol

                cW = A_r_minus / dr_minus / vol
                cE = A_r_plus  / dr_plus  / vol

                # This creates:
                #     L(p) = cE(pE-pP) + cW(pW-pP)
                #          + cN(pN-pP) + cS(pS-pP)
                #
                # Therefore diagonal is negative.
                aP = 0.0

                def add_neighbour(coeff, ii, jj):
                    """
                    Add neighbour contribution.

                    If the neighbour is solid:
                        zero pressure flux through the wall:
                            p_neighbour = p_P
                        therefore:
                            coeff*(p_neighbour - p_P) = 0

                        So we add NOTHING.

                    If the neighbour is fluid:
                        add +coeff*p_neighbour
                        and -coeff*p_P to the diagonal.
                    """
                    nonlocal aP

                    if solid[ii, jj]:
                        # Zero pressure flux through solid wall.
                        # Do not connect through the wall.
                        return

                    add(ind, idx(ii, jj), coeff)
                    aP -= coeff

                # Axial neighbours
                add_neighbour(cS, i - 1, j)
                add_neighbour(cN, i + 1, j)

                # Radial neighbours
                add_neighbour(cW, i, j - 1)
                add_neighbour(cE, i, j + 1)

                # Diagonal
                add(ind, ind, aP)

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

        return A

    def build_poisson_solver(self):
        """
        Returns a callable solve(rhs_flat) that solves A x = rhs_flat fast.
        """
        A = self.pressure_poisson_matrix_sparse().tocsc()
        lu = spla.splu(A, permc_spec="COLAMD", diag_pivot_thresh=1)  # factorization once
        return A, lu
    

class ICPSolver:

    def __init__(self, torch, grid, state, magclass, solver_params):
        # input classes
        self.torch = torch
        self.grid = grid
        self.state = state
        self.mag = magclass

        # Solver solution parameters.
        self.alpha = solver_params["alpha"]
        self.alpha_T = solver_params["alpha_T"] 
        self.INITIAL_STABLE_POINTS = solver_params["INITIAL_STABLE_POINTS"]
        self.N_DIVERGENCE_ITERATIONS = solver_params["N_DIVERGENCE_ITERATIONS"]
        self.N_MAGNETIC_FIELD_ITERATIONS = solver_params["N_MAGNETIC_FIELD_ITERATIONS"]
        self.poisson_alpha = solver_params["poisson_alpha"]

        # Residuals and convergence
        self.residual_history = []
        self.previous_qoi = None

        # Choose scheme
        self.scheme = self.power_law

    def matrix_pressure(self, pressure_poisson_rhs):
        """
        pressure_poisson_rhs is (Ni+2, Nj+2) in your current usage
        returns p_next (Ni+2, Nj+2)

        If your rhs is (Ni, Nj) without ghost cells, then pass it already embedded
        into a (Ni+2, Nj+2) dummy before calling (like you do).
        """
        g = self.grid
        rhs_flat = pressure_poisson_rhs.reshape(-1)
        p_flat = self.state.POISSON_LU.solve(rhs_flat)
        return p_flat.reshape(g.Ni + 2, g.Nj + 2)
    

    def velocity_pressure_correction(self, uz_tent, ur_tent, pressure_correction_next):
        """
        Correcting the pressure and velocity via iterations and the pressure poisson equation at each time step.
        """
        g = self.grid; s = self.state

        uz_next = np.zeros_like(uz_tent)
        ur_next = np.zeros_like(ur_tent)
        # Update the pressure
        p_next = pressure_correction_next
        # Correct the velocities to be incompressible

        rho_z_faces, rho_r_faces = self.state.phi_faces(s.rho)

        p_correction_grad_z = (
                (pressure_correction_next[2:-1, 1:-1] - pressure_correction_next[1:-2, 1:-1])
                /
                (g.Z[2:-1, 1:-1]-g.Z[1:-2, 1:-1])
        )

        uz_next[1:-1, 1:-1] = (
                uz_tent[1:-1, 1:-1] * rho_z_faces[1:-1, :] -
                s.dt

                * p_correction_grad_z
        ) / rho_z_faces[1:-1, :]

        p_correction_grad_r = (
                (pressure_correction_next[1:-1, 2:-1] -
                pressure_correction_next[1:-1, 1:-2]
                )
                /
                (g.R[1:-1, 2:-1]-g.R[1:-1, 1:-2])
        )

        ur_next[1:-1, 1:-1] = (
                ur_tent[1:-1, 1:-1] * rho_r_faces[:, 1:-1] -
                s.dt

                * p_correction_grad_r
        ) / rho_r_faces[:, 1:-1]

        return (
            uz_next,
            ur_next,
            p_next
        )
    
    @staticmethod
    def hybrid(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
        """
        Hybrid scheme as done in Versreeg pg. 124
        """
        aW = np.maximum.reduce([Fw, Dw + Fw / 2, np.zeros_like(Fw)])
        aE = np.maximum.reduce([-Fe, De - Fe / 2, np.zeros_like(Fe)])
        aS = np.maximum.reduce([Fs, Ds + Fs / 2, np.zeros_like(Fs)])
        aN = np.maximum.reduce([-Fn, Dn - Fn / 2, np.zeros_like(Fn)])
        dF = Fe-Fw+Fn-Fs
        return aW, aE, aS, aN, dF

    @staticmethod
    def exponential(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
        """
        Exponential scheme (exact for 1D steady CD), robust and accurate.
        A(Pe) = Pe / (exp(Pe) - 1) with A(0)=1 by limit.
        """
        eps = 1e-30
        Pe_w = Fw / np.maximum(Dw, eps)
        Pe_e = Fe / np.maximum(De, eps)
        Pe_s = Fs / np.maximum(Ds, eps)
        Pe_n = Fn / np.maximum(Dn, eps)

        def A(Pe):
            # Stable evaluation near zero
            out = np.empty_like(Pe)
            small = np.abs(Pe) < 1e-6
            out[small] = 1.0 - Pe[small]/2.0 + Pe[small]**2/12.0  # series
            out[~small] = Pe[~small] / (np.exp(Pe[~small]) - 1.0)
            # Ensure non-negative
            return np.maximum(out, 0.0)

        aW = Dw * A(Pe_w) + np.maximum(Fw, 0.0)
        aE = De * A(Pe_e) + np.maximum(-Fe, 0.0)
        aS = Ds * A(Pe_s) + np.maximum(Fs, 0.0)
        aN = Dn * A(Pe_n) + np.maximum(-Fn, 0.0)
        dF = Fe - Fw + Fn - Fs
        return aW, aE, aS, aN, dF

    @staticmethod
    def upwind(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
        """
        Pure upwind differencing (bounded, diffusive).
        """
        aW = Dw + np.maximum(Fw, 0.0)
        aE = De + np.maximum(-Fe, 0.0)
        aS = Ds + np.maximum(Fs, 0.0)
        aN = Dn + np.maximum(-Fn, 0.0)
        dF = Fe - Fw + Fn - Fs
        return aW, aE, aS, aN, dF

    @staticmethod
    def central(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
        """
        Central differencing (accurate for |Pe|≲2, may be unbounded for high Pe).
        """
        aW = Dw - 0.5 * Fw
        aE = De + 0.5 * Fe
        aS = Ds - 0.5 * Fs
        aN = Dn + 0.5 * Fn
        # Clip negatives to preserve boundedness if you want a safer CDS
        aW = np.maximum(aW, 0.0)
        aE = np.maximum(aE, 0.0)
        aS = np.maximum(aS, 0.0)
        aN = np.maximum(aN, 0.0)
        dF = Fe - Fw + Fn - Fs
        return aW, aE, aS, aN, dF

    @staticmethod
    def power_law(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
        """
        Patankar power-law scheme (good accuracy, bounded).
        A(Pe) = max(0, (1 - 0.1*|Pe|)^5)
        """
        # Avoid divide-by-zero: where D==0, set Pe=0 so A=1
        eps = 1e-30
        Pe_w = Fw / np.maximum(Dw, eps)
        Pe_e = Fe / np.maximum(De, eps)
        Pe_s = Fs / np.maximum(Ds, eps)
        Pe_n = Fn / np.maximum(Dn, eps)

        def A(Pe):
            return np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe))**5)

        aW = Dw * A(Pe_w) + np.maximum(Fw, 0.0)
        aE = De * A(Pe_e) + np.maximum(-Fe, 0.0)
        aS = Ds * A(Pe_s) + np.maximum(Fs, 0.0)
        aN = Dn * A(Pe_n) + np.maximum(-Fn, 0.0)
        dF = Fe - Fw + Fn - Fs
        return aW, aE, aS, aN, dF

    def energy_s(self):
        """
        The energy transport equation for finite volume cells in the CFD grid.
        Returns the new temperature (enthalpy) after time step dt.
        """
        g = self.grid
        s = self.state
        P = self.mag.P

        T_new = np.zeros_like(s.T)
        h_new = np.zeros_like(s.T)

        h = s.hf(s.T)
        k = s.kf(s.T)
        Cp = s.Cpf(s.T)
        rho = s.rhof(s.T)

        Fwe, Fns = s.convective_flux(rho)
        _, k_r_faces = s.phi_faces_upwind(k)
        k_z_faces, k_r_faces = s.phi_faces(k)
        _, Cp_r_faces = s.phi_faces_upwind(Cp)
        Cp_z_faces, Cp_r_faces = s.phi_faces(Cp)

        Fe = Fwe[1:, :] * g.z_area[1:, :]
        Fw = Fwe[:-1, :] * g.z_area[:-1, :]
        Fn = Fns[:, 1:] * g.r_area[:, 1:]
        Fs = Fns[:, :-1] * g.r_area[:, :-1]
        De = (k_z_faces/Cp_z_faces)[1:, :] * g.z_area[1:, :] / (g.Z[2:, 1:-1]-g.Z[1:-1, 1:-1])
        Dw = (k_z_faces/Cp_z_faces)[:-1, :] * g.z_area[:-1, :] / abs(g.Z[:-2, 1:-1]-g.Z[1:-1, 1:-1])
        Dn = (k_r_faces/Cp_r_faces)[:, 1:] * g.r_area[:, 1:] / (g.R[1:-1, 2:]-g.R[1:-1, 1:-1])
        Ds = (k_r_faces/Cp_r_faces)[:, :-1] * g.r_area[:, :-1] / abs(g.R[1:-1, :-2]-g.R[1:-1, 1:-1])

        solid = g.mask_solid.astype(bool)

        # Interior cell mask
        solid_P = solid[1:-1, 1:-1]

        # Neighbour masks around each interior cell
        solid_E = solid[2:, 1:-1]      # axial +z neighbour
        solid_W = solid[:-2, 1:-1]     # axial -z neighbour
        solid_N = solid[1:-1, 2:]      # radial +r neighbour
        solid_S = solid[1:-1, :-2]     # radial -r neighbour

        # A face is blocked if either side is solid
        block_E = solid_P | solid_E
        block_W = solid_P | solid_W
        block_N = solid_P | solid_N
        block_S = solid_P | solid_S

        # No convective flux through solid walls
        Fe[block_E] = 0.0
        Fw[block_W] = 0.0
        Fn[block_N] = 0.0
        Fs[block_S] = 0.0

        # No conductive flux through insulated solid walls
        # De[block_E] = 0.0
        # Dw[block_W] = 0.0
        # Dn[block_N] = 0.0
        # Ds[block_S] = 0.0
        De[solid_P] = 0.0
        Dw[solid_P] = 0.0
        Dn[solid_P] = 0.0
        Ds[solid_P] = 0.0

        aW, aE, aS, aN, dF = self.scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

        B = (P - s.Qrf(s.T[1:-1, 1:-1])) * g.volume  # Finite volume source term.
        B[solid_P] = 0.0

        for _ in range(5):

            ap0 = g.volume * rho[1:-1, 1:-1] / s.dt

            ap = aN+aS+aW+aE-ap0+dF

            h_new[1:-1, 1:-1] = (
                1 / ap0 * (
                    - ap * h[1:-1, 1:-1]
                    + aN * h[1:-1, 2:]
                    + aS * h[1:-1, :-2]
                    + aE * h[2:, 1:-1]
                    + aW * h[:-2, 1:-1]
                    + B
                )
            )

            T_new[1:-1, 1:-1] = s.Tf(h_new[1:-1, 1:-1])

            s.apply_temperature_boundaries(T_new)

            rho = s.rhof(T_new)

        return T_new

    def z_momentum_s(self):
        """
        z-momentum coefficients for discretization of finite volume elements.
        Returns the new axial velocity after time step dt.
        """
        s = self.state
        g = self.grid
        rho_prev = s.rho_prev
        Fz = self.mag.Fz

        mu = s.muvf(s.T)
        Fwe, Fns = s.flux_uz_cells()

        mu_z_faces, mu_r_faces = s.phi_faces_uz_cells(mu)

        Fe = Fwe[1:, :]*g.z_area_uz[1:, :]
        Fw = Fwe[:-1, :]*g.z_area_uz[:-1, :]
        Fn = Fns[:, 1:]*g.r_area_uz[:, 1:]
        Fs = Fns[:, :-1]*g.r_area_uz[:, :-1]
        De = 2*mu_z_faces[1:, :]*g.z_area_uz[1:, :]/(g.Zuz[2:, 1:-1]-g.Zuz[1:-1, 1:-1])
        Dw = 2*mu_z_faces[:-1, :]*g.z_area_uz[:-1, :]/abs(g.Zuz[:-2, 1:-1]-g.Zuz[1:-1, 1:-1])
        Dn = mu_r_faces[:, 1:]*g.r_area_uz[:, 1:]/(g.Ruz[1:-1, 2:]-g.Ruz[1:-1, 1:-1])
        Ds = mu_r_faces[:, :-1] * g.r_area_uz[:, :-1]/abs(g.Ruz[1:-1, :-2]-g.Ruz[1:-1, 1:-1])

        aW, aE, aS, aN, dF = self.scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

        dur_dz = (s.ur[2:-1, :] - s.ur[1:-2, :]) / (g.Z[2:-1, 1:]-g.Z[1:-2, 1:])
        duz_dr = (s.uz[1:-1, 1:]-s.uz[1:-1, :-1]) / (g.R[2:-1, 1:]-g.R[2:-1, :-1])
        diff_source = (
                mu_r_faces[:, 1:]*dur_dz[:, 1:]*g.r_area_uz[:, 1:] -
                mu_r_faces[:, :-1] * dur_dz[:, :-1] * g.r_area_uz[:, :-1] +
                mu_r_faces[:, 1:]*duz_dr[:, 1:]*g.r_area_uz[:, 1:] -
                mu_r_faces[:, :-1] * duz_dr[:, :-1] * g.r_area_uz[:, :-1]
        ) / g.volume_uz

        B = diff_source * g.volume_uz + 0.5*(Fz[1:, :]+Fz[:-1, :]) * g.volume_uz

        dp = (
            (s.p[2:-1, 1:-1] - s.p[1:-2, 1:-1]) / (g.Z[2:-1, 1:-1]-g.Z[1:-2, 1:-1])
        ) * g.volume_uz
        dp = 0;

        rho_z_faces, rho_r_faces = s.phi_faces(rho_prev)

        ap0 = g.volume_uz * rho_z_faces[1:-1, :] / s.dt

        ap = aW+aE+aS+aN-ap0+dF

        uz_next = np.zeros_like(s.uz)

        # rho_uz_next = np.zeros_like(uz)
        # uz_next = np.zeros_like(uz)
        #
        # rho_uz_next[1:-1, 1:-1] = (
        #     1 / (volume_uz / dt) * (
        #         -ap*uz[1:-1, 1:-1]
        #         + aN * uz[1:-1, 2:]
        #         + aS * uz[1:-1, 0:-2]
        #         + aE * uz[2:, 1:-1]
        #         + aW * uz[:-2, 1:-1]
        #         + B
        #         + dp
        #     )
        # )
        # rho_z_faces, rho_r_faces = phi_faces(rho)
        # uz_next[1:-1, 1:-1] = rho_uz_next[1:-1, 1:-1] / rho_z_faces[1:-1, :]


        uz_next[1:-1, 1:-1] = (
            1 / ap0 * (
                -ap * s.uz[1:-1, 1:-1]
                + aN * s.uz[1:-1, 2:]
                + aS * s.uz[1:-1, 0:-2]
                + aE * s.uz[2:, 1:-1]
                + aW * s.uz[:-2, 1:-1]
                + B
                + dp
            )
        )

        s.uz_tent = uz_next

        return uz_next

    def r_momentum_s(self):
        """
        Radial momentum coefficients for the discretization of the finite volume elements.
        Returns the new radial velocity after time step dt.
        """
        s = self.state
        g = self.grid
        rho_prev = s.rho_prev
        Fr = self.mag.Fr

        mu = s.muvf(s.T)
        rho_z_faces, rho_r_faces = s.phi_faces_upwind(rho_prev)

        Fwe, Fns = s.flux_ur_cells()

        mu_z_faces, mu_r_faces = s.phi_faces_ur_cells(mu)

        Fe = Fwe[1:, :]*g.z_area_ur[1:, :]
        Fw = Fwe[:-1, :]*g.z_area_ur[:-1, :]
        Fn = Fns[:, 1:]*g.r_area_ur[:, 1:]
        Fs = Fns[:, :-1] * g.r_area_ur[:, :-1]
        Dn = 2 * mu_r_faces[:, 1:] * g.r_area_ur[:, 1:] / (g.Rur[1:-1, 2:]-g.Rur[1:-1, 1:-1])
        Ds = 2 * mu_r_faces[:, :-1] * g.r_area_ur[:, :-1] / abs(g.Rur[1:-1, :-2]-g.Rur[1:-1, 1:-1])
        De = mu_z_faces[1:, :] * g.z_area_ur[1:, :] / (g.Zur[2:, 1:-1]-g.Zur[1:-1, 1:-1])
        Dw = mu_z_faces[:-1, :] * g.z_area_ur[:-1, :] / abs(g.Zur[:-2, 1:-1]-g.Zur[1:-1, 1:-1])

        aW, aE, aS, aN, dF = self.scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

        duz_dr = (s.uz[:, 2:-1] - s.uz[:, 1:-2]) / (g.R[1:, 2:-1]-g.R[1:, 1:-2])
        dur_dz = (s.ur[1:, 1:-1]-s.ur[:-1, 1:-1]) / (g.Z[1:, 2:-1]-g.Z[:-1 , 2:-1])

        diff_source = (
                mu_z_faces[1:, :] * duz_dr[1:, :] * g.z_area_ur[1:, :] -
                mu_z_faces[:-1, :] * duz_dr[:-1, :] * g.z_area_ur[:-1, :] +
                mu_z_faces[1:, :] * dur_dz[1:, :] * g.z_area_ur[1:, :] -
                mu_z_faces[:-1, :] * dur_dz[:-1, :] * g.z_area_ur[:-1, :]
        ) / g.volume_ur


        B = (
            diff_source*g.volume_ur
            +
            0.5*(Fr[:, 1:]+Fr[:, :-1]) * g.volume_ur
        )

        dp = (
                (s.p[1:-1, 2:-1] - s.p[1:-1, 1:-2]) / (g.R[1:-1, 2:-1]-g.R[1:-1, 1:-2])
        ) * g.volume_ur
        dp = 0;

        Sp = (
                -4*np.pi*s.phi_faces(mu)[1][:, 1:-1] * (g.Zur[2:, 1:-1]-g.Zur[1:-1, 1:-1]) *
                np.log((g.R[1:-1, 2:-1])/(g.R[1:-1, 1:-2]))
        )

        ap0 = g.volume_ur * rho_r_faces[:, 1:-1] / s.dt

        ap = aN+aS+aW+aE-ap0+dF-Sp

        # ur_next = np.zeros_like(ur)
        # rho_ur_next = np.zeros_like(ur)
        #
        # rho_ur_next[1:-1, 1:-1] = (
        #     1 / (volume_ur/dt) * (
        #         -ap*ur[1:-1, 1:-1]
        #         + aN * ur[1:-1, 2:]
        #         + aS * ur[1:-1, :-2]
        #         + aE * ur[2:, 1:-1]
        #         + aW * ur[:-2, 1:-1]
        #         + B
        #         + dp
        #     )
        # )
        #
        # rho_z_faces, rho_r_faces = phi_faces_upwind(rho, uz, ur)
        #
        # ur_next[1:-1, 1:-1] = rho_ur_next[1:-1, 1:-1] / rho_r_faces[:, 1:-1]

        ur_next = np.zeros_like(s.ur)

        ur_next[1:-1, 1:-1] = (
            1 / ap0 * (
                -ap * s.ur[1:-1, 1:-1]
                + aN * s.ur[1:-1, 2:]
                + aS * s.ur[1:-1, :-2]
                + aE * s.ur[2:, 1:-1]
                + aW * s.ur[:-2, 1:-1]
                + B
                + dp
            )
        )

        s.ur_tent = ur_next

        return ur_next
    
    def pressure_correction_iterate(self, N_DIVERGENCE_ITERATIONS):
        """
        Iterates the pressure correction.
        Solves ∇²p = div / dt iteratively to reduce the error of the matrix solution.
        """
        g = self.grid; s = self.state
        dummy = np.zeros_like(s.p_prev)
        s.apply_velocity_boundaries(s.uz_tent, s.ur_tent)
        div = s.divergence_outside(s.uz_tent, s.ur_tent)
        uz_next = s.uz_tent.copy()
        ur_next = s.ur_tent.copy()
        drho_dt = (s.rho-s.rho_prev)[1:-1, 1:-1] / s.dt

        for n in range(N_DIVERGENCE_ITERATIONS):

            divs_orig = div + drho_dt*self.poisson_alpha
            pressure_poisson_rhs = divs_orig/s.dt

            dummy[1:-1, 1:-1] = pressure_poisson_rhs
            dummy[g.mask_solid] = 0.0   # cosmetic; decoupled rows ignore this anyway
            dummy[-1, :] = 0

            pressure_correction_next = self.matrix_pressure(dummy)

            uz_next, ur_next, p_next = self.velocity_pressure_correction(uz_next, ur_next, pressure_correction_next)

            uz_next, ur_next = s.apply_velocity_boundaries(uz_next, ur_next)
            div = s.divergence_outside(uz_next, ur_next)

        s.apply_velocity_boundaries(uz_next, ur_next)

        divs = drho_dt*self.poisson_alpha+div         # mass balance basically.

        return uz_next, ur_next, p_next, divs
    
    def calculate_residuals(
        self,
        tol_update=1e-6,
        tol_mass=1e-5,
        tol_qoi=1e-6,
        store_history=True
    ):
        """
        Calculate residuals for pseudo-time convergence to steady state.

        Use this AFTER solver.iterate_once(...).

        This version has no error handling:
        - assumes s.divs exists
        - assumes g.mask_solid exists
        - assumes s.A, s.A_prev, s.P, s.Fr, s.Fz exist
        - assumes array shapes are correct
        """

        s = self.state
        g = self.grid
        eps = 1e-30

        dt = s.dt

        # ------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------
        def l2_norm(a):
            return np.sqrt(np.sum(a**2))

        def rel_l2(new, old):
            return l2_norm(new - old) / (l2_norm(new) + eps)

        def abs_rate(new, old):
            return l2_norm((new - old) / dt)

        # ------------------------------------------------------------
        # Interior slices
        # ------------------------------------------------------------
        cell = np.s_[1:-1, 1:-1]
        vel = np.s_[1:-1, 1:-1]

        # ------------------------------------------------------------
        # Relative update residuals
        # ------------------------------------------------------------
        update = {}

        update["uz"] = rel_l2(s.uz[vel], s.uz_prev[vel])
        update["ur"] = rel_l2(s.ur[vel], s.ur_prev[vel])
        update["T"] = rel_l2(s.T[cell], s.T_prev[cell])
        update["p"] = rel_l2(s.p[cell], s.p_prev[cell])
        update["rho"] = rel_l2(s.rho[cell], s.rho_prev[cell])
        update["A"] = rel_l2(s.A, s.A_prev)

        # ------------------------------------------------------------
        # Pseudo-time derivative residuals
        # ------------------------------------------------------------
        rate = {}

        rate["uz"] = abs_rate(s.uz[vel], s.uz_prev[vel])
        rate["ur"] = abs_rate(s.ur[vel], s.ur_prev[vel])
        rate["T"] = abs_rate(s.T[cell], s.T_prev[cell])
        rate["p"] = abs_rate(s.p[cell], s.p_prev[cell])
        rate["rho"] = abs_rate(s.rho[cell], s.rho_prev[cell])
        rate["A"] = abs_rate(s.A, s.A_prev)

        # ------------------------------------------------------------
        # Mass / continuity residual
        # ------------------------------------------------------------
        div_mass = s.divs
        div_mass[-1, :] = 0.0  
        vol = g.volume

        fluid_mask = ~g.mask_solid[1:-1, 1:-1]

        local_mdot_error = div_mass * vol

        mass_abs_error = np.sum(np.abs(local_mdot_error[fluid_mask]))
        mass_signed_error = np.sum(local_mdot_error[fluid_mask])
        mass_l2_error = l2_norm(local_mdot_error[fluid_mask])

        rho_z_faces, rho_r_faces = s.phi_faces(s.rho)

        mdot_inlet = np.sum(
            np.abs(s.uz[0, 1:-1] * rho_z_faces[0, :] * g.z_area[0, :])
        )

        mdot_outlet = np.sum(
            np.abs(s.uz[-1, 1:-1] * rho_z_faces[-1, :] * g.z_area[-1, :])
        )

        mdot_axis = np.sum(
            np.abs(s.ur[1:-1, 0] * rho_r_faces[:, 0] * g.r_area[:, 0])
        )

        mdot_wall = np.sum(
            np.abs(s.ur[1:-1, -1] * rho_r_faces[:, -1] * g.r_area[:, -1])
        )

        mdot_scale = mdot_inlet + mdot_outlet + mdot_axis + mdot_wall + eps

        mass = {}

        mass["absolute"] = mass_abs_error / mdot_scale
        mass["signed"] = mass_signed_error / mdot_scale
        mass["l2"] = mass_l2_error / mdot_scale
        mass["dimensional_abs_kg_per_s"] = mass_abs_error
        mass["dimensional_signed_kg_per_s"] = mass_signed_error
        mass["mdot_scale_kg_per_s"] = mdot_scale

        # ------------------------------------------------------------
        # Quantities of interest
        # ------------------------------------------------------------
        qoi = {}

        qoi["T_max"] = np.max(s.T[cell])
        qoi["T_mean"] = np.mean(s.T[cell])
        qoi["uz_max"] = np.max(s.uz[vel])
        qoi["ur_abs_max"] = np.max(np.abs(s.ur[vel]))
        qoi["p_min"] = np.min(s.p[cell])
        qoi["p_max"] = np.max(s.p[cell])
        qoi["joule_power"] = np.sum(s.P * g.volume)
        qoi["Fr_abs_max"] = np.max(np.abs(s.Fr))
        qoi["Fz_abs_max"] = np.max(np.abs(s.Fz))

        # ------------------------------------------------------------
        # QoI relative changes
        # ------------------------------------------------------------
        qoi_change = {}

        if self.previous_qoi is None:
            for key in qoi:
                qoi_change[key] = np.inf
        else:
            for key in qoi:
                qoi_change[key] = abs(qoi[key] - self.previous_qoi[key]) / (
                    abs(qoi[key]) + eps
                )

        self.previous_qoi = qoi.copy()

        max_qoi_change = max(qoi_change.values())

        # ------------------------------------------------------------
        # Overall convergence decision
        # ------------------------------------------------------------
        max_update = max(
            update["uz"],
            update["ur"],
            update["T"],
            update["rho"]
        )

        converged = (
            max_update < tol_update
            and mass["absolute"] < tol_mass
            and max_qoi_change < tol_qoi
            and s.step > self.INITIAL_STABLE_POINTS
        )

        residuals = {
            "step": s.step,
            "time": s.current_time,
            "dt": s.dt,

            "update": update,
            "rate": rate,
            "mass": mass,
            "qoi": qoi,
            "qoi_change": qoi_change,

            "max_update": max_update,
            "max_qoi_change": max_qoi_change,
            "converged": converged,
        }

        if store_history:
            self.residual_history.append(residuals)

        return residuals

    def iterate_once_thermal(self, update_magnetic_field=False):
        """
        Temperature-only iteration.

        This advances only the energy equation.

        Fixed:
            uz
            ur
            p

        Updated:
            T
            rho, optional, from new T
            electromagnetic field, optional

        Notes:
            - No momentum equation is solved.
            - No pressure correction is solved.
            - No divergence correction is applied.
        """

        s = self.state

        # ------------------------------------------------------------
        # Store old fields
        # ------------------------------------------------------------
        s.store_previous()

        uz_fixed = s.uz.copy()
        ur_fixed = s.ur.copy()
        p_fixed = s.p.copy()

        # ------------------------------------------------------------
        # Compute thermal-only time step
        # ------------------------------------------------------------
        s.compute_dt_thermal()

        # ------------------------------------------------------------
        # Optional EM update
        # ------------------------------------------------------------
        # If your Joule heating P depends strongly on T through sigma(T),
        # keep this True every N steps.
        # If you want completely frozen heating, keep this False.
        if update_magnetic_field:
            self.mag.overall_maxwell(
                s.T[1:-1, 1:-1],
                self.N_MAGNETIC_FIELD_ITERATIONS,
                s.A_prev,
            )

            s.Fr = self.mag.Fr
            s.Fz = self.mag.Fz
            s.P = self.mag.P
            s.A = self.mag.A
            s.Hz = self.mag.Hz
            s.Hr = self.mag.Hr
            s.E = self.mag.E

        # ------------------------------------------------------------
        # Solve energy only
        # ------------------------------------------------------------
        T_next = self.energy_s()

        s.T = self.alpha_T * T_next + (1.0 - self.alpha_T) * s.T_prev

        # ------------------------------------------------------------
        # Reapply thermal boundaries
        # ------------------------------------------------------------
        s.T = s.apply_temperature_boundaries(s.T)

        s.update_density()

        # ------------------------------------------------------------
        # Advance pseudo-time
        # ------------------------------------------------------------
        s.step += 1
        s.current_time += s.dt

        return

    def iterate_once(self, update_magnetic_field=False):

        s = self.state
        s.compute_dt()
        s.store_previous()

        if update_magnetic_field:
            self.mag.overall_maxwell(s.T[1:-1, 1:-1], self.N_MAGNETIC_FIELD_ITERATIONS, s.A_prev)
            s.Fr = self.mag.Fr; s.Fz = self.mag.Fz; s.P = self.mag.P; s.A = self.mag.A; s.Hz = self.mag.Hz; s.Hr = self.mag.Hr; s.E = self.mag.E

        T_next = self.energy_s()
        s.T = T_next * self.alpha_T + (1 - self.alpha_T) * s.T_prev

        if s.step > self.INITIAL_STABLE_POINTS:
            self.z_momentum_s(); self.r_momentum_s()
            uz_next, ur_next, p_next, div = self.pressure_correction_iterate(self.N_DIVERGENCE_ITERATIONS)
        
            s.uz, s.ur, s.p, s.divs = (uz_next * self.alpha + (1 - self.alpha) * s.uz_prev, ur_next * self.alpha + (1 - self.alpha) * s.ur_prev, p_next, div)

            res = self.calculate_residuals()
            if s.step % 100 == 0:
                print(
                    f"step={res['step']}, "
                    f"R_update={res['max_update']:.3e}, "
                    f"R_mass={res['mass']['absolute']:.3e}, "
                    f"R_qoi={res['max_qoi_change']:.3e}, "
                    f"Tmax={res['qoi']['T_max']:.1f}"
                )

        s.apply_all_boundaries()
        s.update_density()
        s.step += 1
        s.current_time += s.dt*self.alpha

        return 

        
