import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ONE-SIDED GEOMETRIC SPACING
# ============================================================

def stretch_factor_one_sided(delta, L, N, sf_guess=1.1, tol=1e-7, max_iter=50, name="ug"):
    """
    Calculates geometric stretching factor.

    delta : first cell spacing
    L     : total length
    N     : number of intervals
    """

    sf = sf_guess

    for k in range(max_iter):
        f = L*(sf - 1.0) - delta*(sf**N - 1.0)
        fprime = L - N*delta*sf**(N - 1)

        if abs(fprime) < 1e-14:
            raise ValueError("Derivative became too small while solving stretch factor.")

        sf_new = sf - f/fprime

        if abs(sf_new - sf) < tol:
            return sf_new

        sf = sf_new

    print(f"stretch failed after max_iter {name}.")
    return sf


def one_sided_geometric_spacing(L, N, delta, sf_guess=1.1, name="ug"):
    """
    Returns stretched coordinates from 0 to L.

    N = number of intervals
    Number of points = N + 1
    """

    sf = stretch_factor_one_sided(delta, L, N, sf_guess, name=name)

    s = np.zeros(N + 1)

    for i in range(1, N + 1):
        s[i] = s[i-1] + delta*sf**(i-1)

    s[-1] = L

    return s, sf


# ============================================================
# 2. VINOKUR SPACING
# ============================================================

def calculate_x_parameter(b):
    if b < 1.0:
        if b < 0.26938972:
            pi = np.pi
            x = pi*(1.0 - b + b**2 - (1.0 + pi**2/6.0)*b**3
                    + 6.794732*b**4 - 13.205501*b**5 + 11.726095*b**6)
        else:
            c = 1.0 - b
            x = np.sqrt(6.0*c)*(1.0
                                 + 0.15*c
                                 + 0.057321429*c**2
                                 + 0.048774238*c**3
                                 - 0.053337753*c**4
                                 + 0.075845134*c**5)

    elif b == 1.0:
        x = 0.0

    else:
        if b < 2.7829681:
            c = b - 1.0
            x = np.sqrt(6.0*c)*(1.0
                                 - 0.15*c
                                 + 0.057321429*c**2
                                 - 0.024907295*c**3
                                 + 0.0077424461*c**4
                                 - 0.0010794123*c**5)
        else:
            v = np.log(b)
            w = 1.0/b - 0.028527431
            x = (v + (1.0 + 1.0/v)*np.log(2.0*v)
                 - 0.02041793
                 + 0.24902722*w
                 + 1.9496443*w**2
                 - 2.6294547*w**3
                 + 8.56795911*w**4)

    return x


def quadratic_fit_spacing(d):
    denom = -(d[0, 0] - d[1, 0])*(d[1, 0] - d[2, 0])*(d[2, 0] - d[0, 0])

    a11 = d[1, 0] - d[2, 0]
    a21 = d[2, 0]**2 - d[1, 0]**2
    a31 = d[1, 0]*d[2, 0]*(d[1, 0] - d[2, 0])

    a12 = d[2, 0] - d[0, 0]
    a22 = d[0, 0]**2 - d[2, 0]**2
    a32 = d[2, 0]*d[0, 0]*(d[2, 0] - d[0, 0])

    a13 = d[0, 0] - d[1, 0]
    a23 = d[1, 0]**2 - d[0, 0]**2
    a33 = d[0, 0]*d[1, 0]*(d[0, 0] - d[1, 0])

    b1 = (a11*d[0, 1] + a12*d[1, 1] + a13*d[2, 1]) / denom
    b2 = (a21*d[0, 1] + a22*d[1, 1] + a23*d[2, 1]) / denom
    b3 = (a31*d[0, 1] + a32*d[1, 1] + a33*d[2, 1]) / denom

    disc = b2*b2 - 4.0*b1*b3

    if disc < 0.0:
        raise ValueError("Negative discriminant in quadratic correction.")

    dd1 = (-b2 + np.sqrt(disc))/(2.0*b1)
    dd2 = (-b2 - np.sqrt(disc))/(2.0*b1)
    dd3 = d[2, 0]

    return dd1 if abs(dd1 - dd3) < abs(dd2 - dd3) else dd2


def vinokur(lmax, smax, ds1e=0.0, ds2e=0.0):
    """
    Vinokur stretching.

    lmax : number of points
    smax : total length
    ds1e : desired first spacing
    ds2e : desired last spacing
    """

    s = np.zeros(lmax)
    dsavg = smax / (lmax - 1)

    if ds1e >= smax or ds1e < 0.0:
        ds1e = dsavg

    if ds2e >= (smax - ds1e) or ds2e < 0.0:
        ds2e = dsavg

    if ds1e == 0.0 and ds2e == 0.0:
        kase = 0
        ds1e = dsavg
        ds2e = dsavg
        nlast = 4
    elif ds1e == 0.0:
        kase = 1
        nlast = 1
    elif ds2e == 0.0:
        kase = 2
        nlast = 1
    else:
        kase = 0
        nlast = 4

    d1 = np.zeros((4, 2))
    d2 = np.zeros((4, 2))

    dss1 = 0.0
    dss2 = 0.0

    for n in range(1, nlast + 1):

        if n <= 2:
            ds1 = ds1e - 0.5*dss1
            ds2 = ds2e + 0.5*dss2

            d1[n-1, 0] = ds1
            d2[n-1, 0] = ds2

        elif n == 3:
            ds1 = -d1[0, 1]*(d1[1, 0] - d1[0, 0])/(d1[1, 1] - d1[0, 1]) + d1[0, 0]
            ds2 = -d2[0, 1]*(d2[1, 0] - d2[0, 0])/(d2[1, 1] - d2[0, 1]) + d2[0, 0]

            if ds1 < 0.0:
                ds1 = 0.5*min(d1[0, 0], d1[1, 0])
            if ds2 < 0.0:
                ds2 = 0.5*min(d2[0, 0], d2[1, 0])

            d1[n-1, 0] = ds1
            d2[n-1, 0] = ds2

        elif n == 4:
            try:
                ds1 = quadratic_fit_spacing(d1)
                ds2 = quadratic_fit_spacing(d2)

                if ds1 < 0.0 or ds2 < 0.0:
                    break

            except Exception:
                break

        s0 = smax / (lmax - 1) / ds1
        s1 = smax / (lmax - 1) / ds2

        b = np.sqrt(s0*s1)
        a = np.sqrt(s0/s1)

        if kase == 1:
            b = s1
        elif kase == 2:
            b = s0

        xpar = calculate_x_parameter(b)

        if kase in [1, 2]:
            s[0] = 0.0
            s[-1] = smax

            for i in range(1, lmax - 1):
                j = lmax - 1 - i
                xi = i / (lmax - 1)

                if b > 1.0001:
                    u1 = 1.0 + np.tanh(xpar/2.0*(xi - 1.0))/np.tanh(xpar/2.0)
                elif b < 0.9999:
                    u1 = 1.0 + np.tan(xpar/2.0*(xi - 1.0))/np.tan(xpar/2.0)
                else:
                    u1 = xi*(1.0 - 0.5*(b - 1.0)*(1.0 - xi)*(2.0 - xi))

                u2 = np.sinh(xi*xpar)/np.sinh(xpar) if abs(xpar) > 1e-12 else xi

                if kase == 1:
                    fact = abs(ds1e)
                    s[j] = ((1.0 - fact)*(1.0 - u1) + fact*(1.0 - u2))*smax
                elif kase == 2:
                    fact = abs(ds2e)
                    s[i] = ((1.0 - fact)*u1 + fact*u2)*smax

        else:
            for i in range(lmax):
                xi = i / (lmax - 1)

                cnum = xpar*(xi - 0.5)
                cden = xpar/2.0

                if b < 0.9999:
                    cc = np.tan(cnum)/np.tan(cden)
                    u = 0.5*(1.0 + cc)
                elif 0.9999 <= b <= 1.0001:
                    u = xi*(1.0 + 2.0*(b - 1.0)*(xi - 0.5)*(1.0 - xi))
                else:
                    cc = np.tanh(cnum)/np.tanh(cden)
                    u = 0.5*(1.0 + cc)

                s[i] = u*smax/(a + (1.0 - a)*u)

        if lmax >= 4:
            dss1 = -s[3] + 4.0*s[2] - 5.0*s[1] + 2.0*s[0]
            dss2 = (2.0*s[-1] - 5.0*s[-2] + 4.0*s[-3] - s[-4]) / 2.0

        es1 = s[1] - s[0]
        es2 = s[-1] - s[-2]

        if n != 4:
            d1[n-1, 1] = es1 - ds1e
            d2[n-1, 1] = es2 - ds2e

    s[0] = 0.0
    s[-1] = smax

    return s


class Line:

    def __init__(self, P1, P2, N, delta=None, stretch_type="geometric",
                 function_x=False, function_y=False, stretched=False, stretch_side="start",
                 f=None, args=None, sf_guess=1.1):

        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.N = N
        self.delta = delta
        self.stretch_type = stretch_type
        self.function_x = function_x
        self.stretched = stretched
        self.stretch_side = stretch_side
        self.f = f
        self.function_y = function_y
        self.args = args if args is not None else ()
        self.sf_guess = sf_guess
        self.x, self.y = self.generate()

    def spacing(self, L):
        if not self.stretched:
            return np.linspace(0, L, self.N)

        if self.delta is None:
            raise ValueError("For stretched=True, you must give delta.")

        if self.stretch_type == "geometric":
            s, sf = one_sided_geometric_spacing(
                L,
                self.N - 1,
                self.delta,
                self.sf_guess, name=f"{self.P1} and {self.P2}",
            )

        elif self.stretch_type == "vinokur":
            if isinstance(self.delta, (list, tuple, np.ndarray)):
                del1 = self.delta[0]
                del2 = self.delta[1]
            else:
                del1 = self.delta
                del2 = 0.0

            s = vinokur(self.N, L, ds1e=del1, ds2e=del2)

        else:
            raise ValueError("stretch_type must be 'geometric' or 'vinokur'.")

        if self.stretch_side == "end":
            s = L - s[::-1]

        elif self.stretch_side != "start":
            raise ValueError("stretch_side must be 'start' or 'end'.")

        return s

    def generate(self):
        if not self.function_x and not self.function_y:
            return self.generate_straight()
        elif self.function_x:
            return self.generate_fx(self.f, *self.args)
        elif self.function_y:
            return self.generate_fy(self.f, *self.args)

    def generate_straight(self):
        P1 = self.P1
        P2 = self.P2

        L = np.linalg.norm(P2 - P1)

        s = self.spacing(L)
        t = s / L

        x = P1[0] + t*(P2[0] - P1[0])
        y = P1[1] + t*(P2[1] - P1[1])
        return x, y

    def generate_fx_old(self, f, *args):
        P1 = self.P1
        P2 = self.P2
        L = np.linalg.norm(P2[0] - P1[0])
        s = self.spacing(L)
        t = s / L
        x = P1[0] + t*(P2[0] - P1[0])
        y = f(x, *args)
        return x, y
    
    def generate_fx(self, f, *args):
        P1 = self.P1
        P2 = self.P2

        # Use many temporary points to estimate the curve length
        n_sample = 5000

        x_sample = np.linspace(P1[0], P2[0], n_sample)
        y_sample = f(x_sample, *args)

        # Distance between neighbouring sampled points
        dx = np.diff(x_sample)
        dy = np.diff(y_sample)

        ds = np.sqrt(dx**2 + dy**2)

        # Cumulative arc length
        arc = np.zeros(n_sample)
        arc[1:] = np.cumsum(ds)

        L = arc[-1]

        if L == 0:
            raise ValueError("Curve has zero arc length.")

        # Now apply your normal spacing function to the ARC LENGTH
        s = self.spacing(L)

        # Find the x-values that correspond to those arc-length positions
        x = np.interp(s, arc, x_sample)

        # Then calculate y from the function
        y = f(x, *args)

        return x, y
    
    def generate_fy_old(self, f, *args):
        P1 = self.P1
        P2 = self.P2
        L = np.linalg.norm(P2[1] - P1[1])
        s = self.spacing(L)
        t = s / L
        y = P1[1] + t*(P2[1] - P1[1])
        x = f(y, *args)
        return x, y
    
    def generate_fy(self, f, *args):
        P1 = self.P1
        P2 = self.P2

        n_sample = 5000

        y_sample = np.linspace(P1[1], P2[1], n_sample)
        x_sample = f(y_sample, *args)

        dx = np.diff(x_sample)
        dy = np.diff(y_sample)

        ds = np.sqrt(dx**2 + dy**2)

        arc = np.zeros(n_sample)
        arc[1:] = np.cumsum(ds)

        L = arc[-1]

        if L == 0:
            raise ValueError("Curve has zero arc length.")

        s = self.spacing(L)

        y = np.interp(s, arc, y_sample)
        x = f(y, *args)

        return x, y

    def points(self):
        return self.x, self.y

    def reverse(self):
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        return self

    def plot_line(self):
        plt.plot(self.x, self.y, "k-", linewidth=2)
        plt.axis("equal")


class ICPGrid:


    def __init__(self, torch):

        """
        Sheath
        Sheath wall

        Main

        Carrier wall
        Carrier
        
        
        """

        self.torch = torch
        self.Njcarrier = torch.Njcarrier
        self.Njtcarrier = torch.Njtcarrier
        self.Njmain = torch.Njmain
        self.Njtsheath = torch.Njtsheath
        self.Njsheath = torch.Njsheath
        self.Ni_inlet = torch.Ni_inlet
        self.Ni_outlet = torch.Ni_outlet

        self.delta = torch.delta

        self.build_faces()
        self.build_centres()
        self.build_meshes()
        self.build_geometry()
        self.build_regular()
        self.build_masks()
        print("=========== Geometry Built ============")
        print("=======================================")

        self.Ni, self.Nj = np.shape(self.Z[1:-1, 1:-1])

        pass

    def build_regular(self):
        dz = self.torch.Lz / self.torch.Ni_regular
        dr = self.torch.Lr / self.torch.Nj_regular

        z_coordinates_regular = np.linspace(-dz/2, self.torch.Lz+dz/2, self.torch.Ni_regular+2)
        r_coordinates_regular = np.linspace(-dr/2, self.torch.Lr+dr/2, self.torch.Nj_regular+2)
        self.Z_regular, self.R_regular = np.meshgrid(z_coordinates_regular, r_coordinates_regular, indexing="ij")
        return 

    def build_faces(self):
        yline1 = Line([0, 0], [0, self.torch.Lr_carrier-self.torch.t_carrier], 
                      N=self.Njcarrier, delta=[self.delta[0], self.delta[1]], stretch_type="vinokur", stretched=True)
        yline2 = Line([0, self.torch.Lr_carrier-self.torch.t_carrier], [0, self.torch.Lr_carrier], 
                      N=self.Njtcarrier, delta=[self.delta[1], self.delta[2]], stretch_type="vinokur", stretched=True)
        yline3 = Line([0, self.torch.Lr_carrier], [0, self.torch.Lr_sheath], 
                      N=self.Njmain, delta=[self.delta[2], self.delta[3]], stretch_type="vinokur", stretched=True)
        yline4 = Line([0, self.torch.Lr_sheath], [0, self.torch.Lr_sheath+self.torch.t_sheath], N=self.Njtsheath, 
                      delta=[self.delta[3], self.delta[4]], stretch_type="vinokur", stretched=True)
        yline5 = Line([0, self.torch.Lr_sheath+self.torch.t_sheath], [0, self.torch.Lr], 
                      N=self.Njsheath, delta=[self.delta[4], self.delta[5]], stretch_type="vinokur", stretched=True)

        xline1 = Line([0, 0], [self.torch.Lz_sheath, 0], N=self.Ni_inlet, 
                      delta=[self.delta[6], self.delta[7]], stretch_type="vinokur", stretched=False)
        xline2 = Line([self.torch.Lz_sheath, 0], [self.torch.Lz, 0], N=self.Ni_outlet, 
                      delta=self.delta[7], stretched=False)

        self.r_faces = np.concatenate([yline1.y[:-1], yline2.y[:-1], yline3.y[:-1], yline4.y[:-1], yline5.y[:]])
        self.z_faces = np.concatenate([xline1.x[:-1], xline2.x[:]])

        return
    
    def build_centres(self):
        """
        Build the cell-centre coordinates from the face coordinates.

        Pressure, temperature, density, etc. live here.
        """

        # Physical cell centres
        self.r_centres = 0.5 * (self.r_faces[1:] + self.r_faces[:-1])
        self.z_centres = 0.5 * (self.z_faces[1:] + self.z_faces[:-1])

        # Ghost cell centres
        self.r_ghost_axis = -self.r_centres[0]
        self.r_ghost_wall = 2*self.r_faces[-1] - self.r_centres[-1]

        self.z_ghost_inlet = 2*self.z_faces[0] - self.z_centres[0]
        self.z_ghost_outlet = 2*self.z_faces[-1] - self.z_centres[-1]

        # Main pressure/temperature/density coordinates including ghost cells
        self.r_coordinates = np.concatenate(
            ([self.r_ghost_axis], self.r_centres, [self.r_ghost_wall])
        )

        self.z_coordinates = np.concatenate(
            ([self.z_ghost_inlet], self.z_centres, [self.z_ghost_outlet])
        )

        # Staggered velocity coordinates
        self.r_coordinates_ur = self.r_faces
        self.z_coordinates_uz = self.z_faces

        self.r_coordinates_uz = self.r_coordinates
        self.z_coordinates_ur = self.z_coordinates

        return

    def build_meshes(self):
        """
        Build meshgrid arrays.

        Main mesh:
            self.Z, self.R

        Axial velocity mesh:
            self.Zuz, self.Ruz

        Radial velocity mesh:
            self.Zur, self.Rur
        """

        # Main cell-centred mesh
        # Used for p, T, rho, etc.
        self.Z, self.R = np.meshgrid(
            self.z_coordinates,
            self.r_coordinates,
            indexing="ij"
        )

        # Axial velocity mesh
        # uz is located on z-faces and radial centres
        self.Zuz, self.Ruz = np.meshgrid(
            self.z_coordinates_uz,
            self.r_coordinates_uz,
            indexing="ij"
        )

        # Radial velocity mesh
        # ur is located on axial centres and radial faces
        self.Zur, self.Rur = np.meshgrid(
            self.z_coordinates_ur,
            self.r_coordinates_ur,
            indexing="ij"
        )

        return
    
    def build_masks(self):
        """
        Build masks for solid wall zones.

        Masks are built for:
            - cell-centred fields: p, T, rho
            - axial velocity field: uz
            - radial velocity field: ur
        """

        t = self.torch

        # ============================================================
        # Geometry definitions
        # ============================================================

        r_carrier_inner = t.Lr_carrier - t.t_carrier
        r_carrier_outer = t.Lr_carrier

        r_sheath_inner = t.Lr_sheath
        r_sheath_outer = t.Lr_sheath + t.t_sheath

        z_carrier_end = t.Lz_carrier
        z_sheath_end = t.Lz_sheath

        # ============================================================
        # Cell-centred masks: p, T, rho
        # Shape: same as self.Z and self.R
        # ============================================================

        self.mask_carrier_wall = (
            (self.R >= r_carrier_inner)
            & (self.R <= r_carrier_outer)
            & (self.Z <= z_carrier_end)
        )

        self.mask_sheath_wall = (
            (self.R >= r_sheath_inner)
            & (self.R <= r_sheath_outer)
            & (self.Z <= z_sheath_end)
        )

        self.mask_solid = (
            self.mask_carrier_wall
            | self.mask_sheath_wall
        )

        # ============================================================
        # Cell-centred masks without boundary
        # Shape: same as self.Z[1:-1, 1:-1] and self.R[1:-1, 1:-1]
        # ============================================================

        self.mask_carrier_wall_inner = (
            (self.R[1:-1, 1:-1] >= r_carrier_inner)
            & (self.R[1:-1, 1:-1] <= r_carrier_outer)
            & (self.Z[1:-1, 1:-1] <= z_carrier_end)
        )

        self.mask_sheath_wall_inner = (
            (self.R[1:-1, 1:-1] >= r_sheath_inner)
            & (self.R[1:-1, 1:-1] <= r_sheath_outer)
            & (self.Z[1:-1, 1:-1] <= z_sheath_end)
        )

        self.mask_solid_inner = (
            self.mask_carrier_wall_inner
            | self.mask_sheath_wall_inner
        )

        # ============================================================
        # uz masks
        # Shape: same as self.Zuz and self.Ruz
        # ============================================================

        self.mask_carrier_wall_uz = (
            (self.Ruz >= r_carrier_inner)
            & (self.Ruz <= r_carrier_outer)
            & (self.Zuz <= z_carrier_end)
        )

        self.mask_sheath_wall_uz = (
            (self.Ruz >= r_sheath_inner)
            & (self.Ruz <= r_sheath_outer)
            & (self.Zuz <= z_sheath_end)
        )

        self.mask_solid_uz = (
            self.mask_carrier_wall_uz
            | self.mask_sheath_wall_uz
        )

        # ============================================================
        # ur masks
        # Shape: same as self.Zur and self.Rur
        # ============================================================

        self.mask_carrier_wall_ur = (
            (self.Rur >= r_carrier_inner)
            & (self.Rur <= r_carrier_outer)
            & (self.Zur <= z_carrier_end)
        )

        self.mask_sheath_wall_ur = (
            (self.Rur >= r_sheath_inner)
            & (self.Rur <= r_sheath_outer)
            & (self.Zur <= z_sheath_end)
        )

        self.mask_solid_ur = (
            self.mask_carrier_wall_ur
            | self.mask_sheath_wall_ur
        )

        return
    
    def plot_cell_faces(self, ax=None, unit="mm", linewidth=0.5, highlight_regions=True):
        """
        Plot the physical cell faces as grid lines.

        The carrier and sheath regions are highlighted with thick black lines.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if unit == "mm":
            scale = 1000
            xlabel = "z [mm]"
            ylabel = "r [mm]"
        else:
            scale = 1
            xlabel = "z [m]"
            ylabel = "r [m]"

        z = self.z_faces * scale
        r = self.r_faces * scale

        # ------------------------------------------------------------
        # Thin normal grid lines
        # ------------------------------------------------------------
        for zi in z:
            ax.plot([zi, zi], [r[0], r[-1]], "k-", linewidth=linewidth, alpha=0.35)

        for rj in r:
            ax.plot([z[0], z[-1]], [rj, rj], "k-", linewidth=linewidth, alpha=0.35)

        # ------------------------------------------------------------
        # Highlight carrier and sheath regions
        # ------------------------------------------------------------
        if highlight_regions:
            z0 = self.z_faces[0] * scale
            z1 = self.torch.Lz_sheath * scale

            # Carrier gas region
            r_carrier_inner = (self.torch.Lr_carrier - self.torch.t_carrier) * scale
            r_carrier_outer = self.torch.Lr_carrier * scale

            # Sheath gas region
            r_sheath_inner = self.torch.Lr_sheath * scale
            r_sheath_outer = (self.torch.Lr_sheath + self.torch.t_sheath) * scale

            # Carrier box
            ax.plot(
                [z0, z1, z1, z0, z0],
                [r_carrier_inner, r_carrier_inner, r_carrier_outer, r_carrier_outer, r_carrier_inner],
                "k-",
                linewidth=2.5
            )

            # Sheath box
            ax.plot(
                [z0, z1, z1, z0, z0],
                [r_sheath_inner, r_sheath_inner, r_sheath_outer, r_sheath_outer, r_sheath_inner],
                "k-",
                linewidth=2.5
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("ICP torch cell faces")
        ax.set_aspect("equal", adjustable="box")

        return ax
    
    def plot_cell_centres(self, ax=None, unit="mm", show_faces=True, markersize=8):
        """
        Plot the cell centres on top of the cell faces.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if show_faces:
            ax = self.plot_cell_faces(
                ax=ax,
                unit=unit,
                linewidth=0.4,
                highlight_regions=True
            )

        if unit == "mm":
            scale = 1000
        else:
            scale = 1

        Zc, Rc = np.meshgrid(
            self.z_centres * scale,
            self.r_centres * scale,
            indexing="ij"
        )

        ax.scatter(
            Zc,
            Rc,
            s=markersize,
            marker="o"
        )

        ax.set_title("ICP torch cell faces with cell centres")
        ax.set_aspect("equal", adjustable="box")

        return ax

    def build_geometry(self):
        """
        Build finite-volume areas and volumes.

        This follows the same indexing logic as the original standalone code:

            z_area
            r_area
            volume

            z_area_uz
            r_area_uz
            volume_uz

            z_area_ur
            r_area_ur
            volume_ur
        """

        Z = self.Z
        R = self.R

        Zuz = self.Zuz
        Ruz = self.Ruz

        Zur = self.Zur
        Rur = self.Rur

        # =====================================================================
        # Main pressure / temperature / density control volumes
        # =====================================================================

        self.z_area = (
            np.ones_like(Z[1:, 1:-1]) * np.pi *
            (
                Rur[1:, 1:]**2
                - np.maximum.reduce(
                    [
                        Rur[1:, :-1],
                        np.zeros_like(Rur[1:, :-1])
                    ]
                )**2
            )
        )

        self.r_area = (
            np.ones_like(R[1:-1, 1:])
            * (Zuz[1:, 1:] - Zuz[:-1, 1:])
            * 2 * np.pi * Rur[1:-1, :]
        )

        self.volume = (
            np.pi
            * (Rur[1:-1, 1:]**2 - Rur[1:-1, :-1]**2)
            * (Zuz[1:, 1:-1] - Zuz[:-1, 1:-1])
        )

        # =====================================================================
        # Axial velocity control volumes: uz
        # =====================================================================

        self.z_area_uz = (
            np.ones_like(Zuz[1:, 1:-1]) * np.pi *
            (
                Rur[1:-1, 1:]**2
                - np.maximum.reduce(
                    [
                        Rur[1:-1, :-1],
                        np.zeros_like(Rur[1:-1, :-1])
                    ]
                )**2
            )
        )

        self.r_area_uz = (
            np.ones_like(Ruz[1:-1, 1:])
            * (Z[2:-1, 1:] - Z[1:-2, 1:])
            * 2 * np.pi * Rur[2:-1, :]
        )

        # This is the non-uniform-grid equivalent of your old "* dz"
        dz_uz = Z[2:-1, 1:-1] - Z[1:-2, 1:-1]

        self.volume_uz = (
            np.pi
            * (Rur[2:-1, 1:]**2 - Rur[1:-2, :-1]**2)
            * dz_uz
        )

        # =====================================================================
        # Radial velocity control volumes: ur
        # =====================================================================

        self.z_area_ur = (
            np.ones_like(Zur[1:, 1:-1]) * np.pi *
            (
                R[1:, 2:-1]**2
                - np.maximum.reduce(
                    [
                        R[1:, 1:-2],
                        np.zeros_like(R[1:, 1:-2])
                    ]
                )**2
            )
        )

        self.r_area_ur = (
            np.ones_like(Rur[1:-1, 1:])
            * (Zuz[1:, 1:-1] - Zuz[:-1, 1:-1])
            * 2 * np.pi * R[1:-1, 1:-1]
        )

        self.volume_ur = (
            np.pi
            * (R[1:-1, 2:-1]**2 - R[1:-1, 1:-2]**2)
            * (Zuz[1:, 2:-1] - Zuz[:-1, 2:-1])
        )

        return


