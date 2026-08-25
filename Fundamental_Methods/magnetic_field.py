import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import time


def quadratic_extrapolate_last_column(B, x):
    """
    Extrapolate B[:, -1] from B[:, -4], B[:, -3], B[:, -2]
    using quadratic Lagrange extrapolation on a nonuniform grid.
    """
    x1 = x[-4]
    x2 = x[-3]
    x3 = x[-2]
    x4 = x[-1]

    L1 = ((x4 - x2) * (x4 - x3)) / ((x1 - x2) * (x1 - x3))
    L2 = ((x4 - x1) * (x4 - x3)) / ((x2 - x1) * (x2 - x3))
    L3 = ((x4 - x1) * (x4 - x2)) / ((x3 - x1) * (x3 - x2))

    B[:, -1] = L1 * B[:, -4] + L2 * B[:, -3] + L3 * B[:, -2]
    return B


class ElectroMagnetic:

    def __init__(self, grid, Coils, omega, mu0, Ic, sigmaf):
        self.R = grid.R[1:-1, 1:-1]
        self.Z = grid.Z[1:-1, 1:-1]

        self.Ni = int(grid.Ni)  # z
        self.Nj = int(grid.Nj)  # r

        self.Coils = Coils[:, ::-1]   # keep your original reversal
        self.omega = omega
        self.mu0 = mu0
        self.Ic = Ic
        self.sigmaf = sigmaf

        if self.R.shape != (self.Ni, self.Nj) or self.Z.shape != (self.Ni, self.Nj):
            raise ValueError(
                f"R,Z must be shape (Ni,Nj)=(Nz,Nr)=({self.Ni},{self.Nj}). "
                f"Got R{self.R.shape}, Z{self.Z.shape}."
            )

        # Precompute geometry-only stencil coefficients (kE, kW, kN, kS) and
        # the sigma-independent part of kP, plus the flattened index map for
        # the interior nodes. These never change after construction, so we
        # build them once here instead of recomputing per matrix_setup call.
        self._build_stencil_geometry()
        self._build_b_vector_geometry()

    def _build_stencil_geometry(self):
        """
        Vectorized, one-time computation of the geometric stencil
        coefficients over the full interior of the grid (equivalent to
        calling the old scalar kE/kW/kN/kS/kP methods for every (iz, ir)
        pair, but done with numpy slicing instead of a Python loop).
        """
        Ni, Nr = self.Ni, self.Nj
        R, Z = self.R, self.Z
        self.Fz = np.zeros_like(R)
        self.Fz = np.zeros_like(R)
        self.Fr = np.zeros_like(R)
        self.P = np.zeros_like(R)
        self.Hz = np.zeros_like(R)
        self.Hr = np.zeros_like(R)
        self.E = np.zeros_like(R)


        # Interior slice: iz in [1, Ni-2], ir in [1, Nr-2]
        Rc = R[1:-1, 1:-1]

        hplus_r = R[1:-1, 2:] - R[1:-1, 1:-1]
        hminus_r = R[1:-1, 1:-1] - R[1:-1, :-2]

        hplus_z = Z[2:, 1:-1] - Z[1:-1, 1:-1]
        hminus_z = Z[1:-1, 1:-1] - Z[:-2, 1:-1]

        # kE
        second_e = 2.0 / hplus_r / (hplus_r + hminus_r)
        first_e = hminus_r / hplus_r / (hplus_r + hminus_r)
        kE = second_e + first_e / Rc

        # kW
        second_w = 2.0 / hminus_r / (hplus_r + hminus_r)
        first_w = -hplus_r / hminus_r / (hplus_r + hminus_r)
        kW = second_w + first_w / Rc

        # kN
        kN = 2.0 / hplus_z / (hplus_z + hminus_z) * np.ones_like(Rc)

        # kS
        kS = 2.0 / hminus_z / (hplus_z + hminus_z) * np.ones_like(Rc)

        # sigma-independent part of kP (everything except the -1j*mu0*omega*sigma term)
        base_kP = (
            -2.0 / hplus_r / hminus_r
            - 2.0 / hplus_z / hminus_z
            + (hplus_r - hminus_r) / (hplus_r * hminus_r) / Rc
            - 1.0 / (Rc ** 2)
        )

        self._kE = kE
        self._kW = kW
        self._kN = kN
        self._kS = kS
        self._base_kP = base_kP

        # Flattened interior index grid: k = iz * Nr + ir
        iz_idx, ir_idx = np.meshgrid(
            np.arange(1, Ni - 1), np.arange(1, Nr - 1), indexing="ij"
        )
        self._k_interior = (iz_idx * Nr + ir_idx).astype(np.int64)

        n = Ni * Nr
        boundary_mask = np.ones(n, dtype=bool)
        boundary_mask[self._k_interior.ravel()] = False
        self._k_boundary = np.nonzero(boundary_mask)[0]

        # Pre-flatten neighbor index offsets (constant, geometry-only)
        k = self._k_interior
        self._idx_self = k.ravel()
        self._idx_west = (k - 1).ravel()
        self._idx_east = (k + 1).ravel()
        self._idx_south = (k - Nr).ravel()
        self._idx_north = (k + Nr).ravel()

    @staticmethod
    def Gf(m):
        return ((2 - m ** 2) * sc.special.ellipk(m ** 2) - 2 * sc.special.ellipe(m ** 2)) / m

    @staticmethod
    def kjf(rj, Rb, zj, Zb):
        return np.sqrt(4 * Rb * rj / ((rj + Rb) ** 2 + (Zb - zj) ** 2))

    @staticmethod
    def kif(Ri, Rb, Zi, Zb):
        return np.sqrt(4 * Rb * Ri / ((Ri + Rb) ** 2 + (Zb - Zi) ** 2))

    def matrix_setup(self, sigma):
        """
        Vectorized assembly of the sparse system matrix. Replaces the old
        per-node Python loop + lil_matrix scalar writes with a single
        COO construction built from precomputed geometric coefficients.

        Only the diagonal (kP) depends on sigma; kE/kW/kN/kS are reused
        from _build_stencil_geometry().
        """
        Nz, Nr = self.Ni, self.Nj
        n = Nz * Nr

        # sigma-dependent diagonal term, evaluated only on interior nodes
        kP = self._base_kP - 1j * self.mu0 * self.omega * sigma[1:-1, 1:-1]

        rows = np.concatenate([
            self._idx_self, self._idx_self, self._idx_self,
            self._idx_self, self._idx_self,
            self._k_boundary,
        ])
        cols = np.concatenate([
            self._idx_self, self._idx_west, self._idx_east,
            self._idx_south, self._idx_north,
            self._k_boundary,
        ])
        data = np.concatenate([
            kP.ravel(),
            self._kW.ravel(),
            self._kE.ravel(),
            self._kS.ravel(),
            self._kN.ravel(),
            np.ones(self._k_boundary.size, dtype=np.complex128),
        ])

        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)
        return A.tocsr()
    
    def _build_b_vector_geometry(self):
        """
        Precompute all geometry-only quantities used in b_vector.

        This makes b_vector much faster because the expensive elliptic
        integral kernels depend only on grid and coil geometry, not on A or sigma.
        """

        Nz, Nr = self.Ni, self.Nj

        # ------------------------------------------------------------
        # Interior points that contribute to the plasma integral
        # ------------------------------------------------------------
        R_int = self.R[1:-1, 1:-1]
        Z_int = self.Z[1:-1, 1:-1]

        dz = 0.5 * (
            np.abs(self.Z[1:-1, 1:-1] - self.Z[2:, 1:-1])
            + np.abs(self.Z[1:-1, 1:-1] - self.Z[:-2, 1:-1])
        )

        dr = 0.5 * (
            np.abs(self.R[1:-1, 1:-1] - self.R[1:-1, :-2])
            + np.abs(self.R[1:-1, 1:-1] - self.R[1:-1, 2:])
        )

        self._b_prod1_factor = (
            -1j * self.omega * self.mu0 / (2.0 * np.pi)
            * dr
            * dz
            * np.sqrt(R_int)
        )

        R_int_flat = R_int.ravel()
        Z_int_flat = Z_int.ravel()

        # ------------------------------------------------------------
        # Boundary points where b is nonzero
        # ------------------------------------------------------------
        # z = 0 and z = Lz boundaries, excluding the outer-r corner
        ir_mid = np.arange(1, Nr - 1)

        iz_bottom = np.zeros_like(ir_mid)
        ir_bottom = ir_mid

        iz_top = np.full_like(ir_mid, Nz - 1)
        ir_top = ir_mid

        # outer radial boundary r = Rmax
        iz_outer = np.arange(Nz)
        ir_outer = np.full_like(iz_outer, Nr - 1)

        self._b_iz = np.concatenate([iz_top, iz_bottom, iz_outer])
        self._b_ir = np.concatenate([ir_top, ir_bottom, ir_outer])

        Rb = self.R[self._b_iz, self._b_ir]
        Zb = self.Z[self._b_iz, self._b_ir]

        self._b_Rb = Rb
        self._b_Zb = Zb

        # ------------------------------------------------------------
        # Plasma boundary kernel
        # Shape:
        #     K_plasma: (number of boundary points, number of interior points)
        # ------------------------------------------------------------
        kj = np.sqrt(
            4.0 * Rb[:, None] * R_int_flat[None, :]
            / (
                (Rb[:, None] + R_int_flat[None, :])**2
                + (Zb[:, None] - Z_int_flat[None, :])**2
            )
        )

        self._K_plasma = (
            np.sqrt(1.0 / Rb)[:, None]
            * self.Gf(kj)
        )

        # ------------------------------------------------------------
        # Coil contribution is completely constant
        # ------------------------------------------------------------
        ki = np.sqrt(
            4.0 * Rb[:, None] * self.Coils[None, :, 0]
            / (
                (Rb[:, None] + self.Coils[None, :, 0])**2
                + (Zb[:, None] - self.Coils[None, :, 1])**2
            )
        )

        prod2 = (
            self.mu0 * self.Ic / (2.0 * np.pi)
            * np.sqrt(self.Coils[:, 0])
        )

        K_coil = (
            np.sqrt(1.0 / Rb)[:, None]
            * self.Gf(ki)
        )

        self._b_coil = K_coil @ prod2

        return

    def b_vector(self, Afield, sigma):
        """
        Fast RHS construction using precomputed boundary kernels.
        """

        Nz, Nr = self.Ni, self.Nj

        mock = np.zeros((Nz, Nr), dtype=np.complex128)

        # Interior plasma source term
        prod1 = (
            self._b_prod1_factor
            * Afield[1:-1, 1:-1]
            * sigma[1:-1, 1:-1]
        )

        prod1_flat = prod1.ravel()

        # Boundary values = plasma contribution + coil contribution
        boundary_values = self._K_plasma @ prod1_flat + self._b_coil

        mock[self._b_iz, self._b_ir] = boundary_values

        return mock.reshape(-1)

    def magnetic_vector_solver_old_works(self, temp, iterations, A_guess):
        sigma = self.sigmaf(temp)

        B0 = self.b_vector(A_guess, sigma)
        D = self.matrix_setup(sigma)

        x0 = A_guess.reshape(-1)
        x = splinalg.bicgstab(D, B0, x0=x0)[0]
        A = x.reshape(self.Ni, self.Nj)

        A_prev = A.copy()

        alpha = 0.8

        for _ in range(iterations):
            B = self.b_vector(A, sigma)
            x, info = splinalg.bicgstab(D, B, x0=A_prev.reshape(-1))
            A_new = x.reshape(self.Ni, self.Nj)
            A_prev, A = A, A_new*alpha + (1-alpha)*A
            max_error = np.max(np.abs(A - A_prev) / (np.abs(A) + 1e-12))

        return A, max_error
    
    def magnetic_vector_solver(self, temp, iterations, A_guess):
        sigma = self.sigmaf(temp)
        D = self.matrix_setup(sigma).tocsc()

        self.LU = splinalg.splu(D)

        A = A_guess.copy()
        alpha = 0.8
        max_error = np.inf

        for i in range(iterations):
            A_prev = A.copy()

            B = self.b_vector(A_prev, sigma)

            A_new = self.LU.solve(B).reshape(self.Ni, self.Nj)

            A = alpha*A_new + (1.0 - alpha)*A_prev

            max_error = np.max(
                np.abs(A - A_prev) / (np.abs(A) + 1e-12)
            )

        return A, max_error
    
    def Electric_Field(self, A):
        return -1j * self.omega * A

    def BzBr(self, A):
        Nz, Nr = self.Ni, self.Nj
        Bz = np.zeros((Nz, Nr), dtype=np.complex128)
        Br = np.zeros((Nz, Nr), dtype=np.complex128)

        AR = A * self.R
        r_vec = self.R[0, :]

        dz = 1 / 2 * (abs(self.Z[1:-1, :] - self.Z[2:, :])
                      + abs(self.Z[1:-1, :] - self.Z[0:-2, :]))
        dr = 1 / 2 * (abs(self.R[:, 1:-1] - self.R[:, 0:-2])
                      + abs(self.R[:, 1:-1] - self.R[:, 2:]))

        # Try again  Best way
        Bz[:, 1:-1] = (
                (1 / self.R[:, 1:-1]) *
                (AR[:, 2:] - AR[:, :-2]) / (2.0 * dr)
        )
        # Bz[:, -1] = Bz[:, -2]
        slope = (Bz[:, -2] - Bz[:, -3]) / (self.R[:, -2] - self.R[:, -3])
        # Bz[:, -1] = slope * (self.R[:, -1] - self.R[:, -2]) + Bz[:, -2]
        Bz[:, -1] = quadratic_extrapolate_last_column(Bz, r_vec)[:, -1]

        Bz[:, 0] = Bz[:, 1]  # axis copy
        Bz[:, 0] = (Bz[:, 2]-Bz[:, 1]) / (self.R[:, 2] - self.R[:, 1])*(self.R[:, 0]-self.R[:, 1]) + Bz[:, 1]

        Br[1:-1, :] = -(A[2:, :] - A[:-2, :]) / (2.0 * dz)
        Br[0, :] = Br[1, :]
        Br[-1, :] = Br[-2, :]

        Hz = Bz / self.mu0
        Hr = Br / self.mu0
        return Bz, Br, Hz, Hr

    def FzFrP(self, T, E, Br, Bz):
        sigma = self.sigmaf(T)
        one = E * np.conjugate(Br)
        two = E * np.conjugate(Bz)
        three = E * np.conjugate(E)

        Fz = -0.5 * sigma * one.real
        Fr = 0.5 * sigma * two.real
        P = 0.5 * sigma * three.real
        return Fz, Fr, P

    def overall_maxwell(self, T, iterations, A_guess):
        A, err = self.magnetic_vector_solver(T, iterations, A_guess)    # Try number 2.
        Bz, Br, Hz, Hr = self.BzBr(A)
        E = self.Electric_Field(A)
        Fz, Fr, P = self.FzFrP(T, E, Br, Bz)
        self.Fz = Fz; self.Fr = Fr; self.P = P; self.A = A; self.Hz = Hz; self.Hr = Hr; self.E = E
        return Fz, Fr, P, A, Hz, Hr, E

    def FzFrP_overall(self, T, A):
        Bz, Br, Hz, Hr = self.BzBr(A)
        E = self.Electric_Field(A)
        Fz, Fr, P = self.FzFrP(T, E, Br, Bz)
        self.Fz = Fz; self.Fr = Fr; self.P = P; self.Hz = Hz; self.Hr = Hr;
        return Fz, Fr, P, Hz, Hr
    