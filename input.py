import numpy as np
import matplotlib.pyplot as plt
from Parameters.mesh import ICPGrid

class ICPTorchOriginal:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 9    # cell faces below carrier wall
        self.Njtcarrier = 10    # cell faces in carrier wall
        self.Njmain = 40       # cell faces in main flow area
        self.Njtsheath = 10     # cell faces in sheath wall
        self.Njsheath = 12     # cell faces above sheath wall

        self.Ni_inlet = 60     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 120   # cell faces after sheath and carrier

        self.delta = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 1, 1]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


class ICPTorchDouble:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 13    # cell faces below carrier wall
        self.Njtcarrier = 13    # cell faces in carrier wall
        self.Njmain = 56       # cell faces in main flow area
        self.Njtsheath = 14     # cell faces in sheath wall
        self.Njsheath = 17     # cell faces above sheath wall

        self.Ni_inlet = 85     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 170   # cell faces after sheath and carrier

        self.delta = np.array([0.07, 0.14, 0.14, 0.14, 0.14, 0.07, 0.5, 0.5]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


class ICPTorchTriple:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 18    # cell faces below carrier wall
        self.Njtcarrier = 18    # cell faces in carrier wall
        self.Njmain = 79       # cell faces in main flow area
        self.Njtsheath = 20     # cell faces in sheath wall
        self.Njsheath = 24     # cell faces above sheath wall

        self.Ni_inlet = 120     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 240   # cell faces after sheath and carrier

        self.delta = np.array([0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.35, 0.35]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


class ICPTorchHalf:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 6    # cell faces below carrier wall
        self.Njtcarrier = 7    # cell faces in carrier wall
        self.Njmain = 28       # cell faces in main flow area
        self.Njtsheath = 7     # cell faces in sheath wall
        self.Njsheath = 9     # cell faces above sheath wall

        self.Ni_inlet = 43     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 85   # cell faces after sheath and carrier

        self.delta = np.array([0.14, 0.28, 0.28, 0.28, 0.28, 0.14, 1.4, 1.4]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


class ICPTorchQuatre:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 5    # cell faces below carrier wall
        self.Njtcarrier = 6    # cell faces in carrier wall
        self.Njmain = 20       # cell faces in main flow area
        self.Njtsheath = 6     # cell faces in sheath wall
        self.Njsheath = 7     # cell faces above sheath wall

        self.Ni_inlet = 30     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 60   # cell faces after sheath and carrier

        self.delta = np.array([0.20, 0.4, 0.4, 0.4, 0.4, 0.2, 2, 2]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


class ICPTorchProduc:
    def __init__(self):
        self.Lz = 200/1000
        self.Lr = 25/1000

        self.Lr_sheath = 18.8/1000
        self.t_sheath = 2.2/1000
        self.Lz_sheath = 50/1000

        self.Lr_carrier = 3.7/1000
        self.t_carrier = 2/1000
        self.Lz_carrier = 50/1000

        self.t_wall = 3.5/1000

        self.Q_main = 3
        self.Q_sheath = 31
        self.Q_carrier = 1    # Check hierso Jacobus.

        self.U_inlet_main = self.Q_main / 1000 / 60 / (np.pi*(self.Lr_sheath**2-self.Lr_carrier**2))      # m/s
        self.U_sheath = self.Q_sheath / 1000 / 60 / (np.pi*(self.Lr**2-(self.Lr_sheath+self.t_sheath)**2))     # m/s
        self.U_carrier = self.Q_carrier / 1000 / 60 / (np.pi*(self.Lr_carrier-self.t_carrier)**2)         # m/s

        self.temp_boundary = 300   # K
        self.p_atm = 0        # Pa

        self.Njcarrier = 6    # cell faces below carrier wall
        self.Njtcarrier = 7    # cell faces in carrier wall
        self.Njmain = 28       # cell faces in main flow area
        self.Njtsheath = 7     # cell faces in sheath wall
        self.Njsheath = 9     # cell faces above sheath wall

        self.Ni_inlet = 43     # cell faces in region where sheath and carrier lie
        self.Ni_outlet = 85   # cell faces after sheath and carrier

        self.delta = np.array([0.14, 0.28, 0.28, 0.28, 0.28, 0.14, 1.4, 1.4]) / 1000 

        self.Ni_regular = 80
        self.Nj_regular = 40


torch = ICPTorchProduc()      # Half is production.

grid = ICPGrid(torch=torch)


# ==================================================================================================================== #
#                                              Magnetic Field
# ==================================================================================================================== #

KW = 5
IC = 161

OMEGA = 2*np.pi*3*10**6  # Angular frequency
MU0 = 4* np.pi*1e-7  # Permeability of free space

Coils = np.array(
    [[63 / 1000, 33 / 1000],
     [92 / 1000, 33 / 1000],
     [121 / 1000, 33 / 1000]
     ]
)


# ==================================================================================================================== #
#                                              Fluid Properties
# ==================================================================================================================== #

from Parameters.parameters import hf, Cpf, kf, rhof, EquationOfState, tempf, muvf, sigmaf, Qrf, d_dT_rhof, Prf


def sigmaf_2500(T):
    return np.ones_like(T)*2500


muvf = muvf
hf = hf
Cpf = Cpf
kf = kf
rhof = rhof
eos = EquationOfState()
Tf = tempf
sigmaf = sigmaf
Qrf = Qrf
d_dT_rhof = d_dT_rhof
pf = eos.pressure_eos
Prf = Prf

parameterfs = {"muvf":muvf, "hf":hf, "Cpf":Cpf, "kf":kf, 
               "rhof":rhof, "Tf":Tf, "Qrf":Qrf, "sigmaf":sigmaf}


# ==================================================================================================================== #
#                                      dt calcs and solver iterations
# ==================================================================================================================== #


CFL = 1.5                         # CFL conditions  was 1.2 or 0.5

current_time = 0
N_DIVERGENCE_ITERATIONS = 10      # 10 also works very good 
N_MAGNETIC_FIELD_ITERATIONS = 7   # 6 is stable.
INITIAL_STABLE_POINTS = 100
FLOW_TO_THERMAL_STEPS = 1
alpha = 0.6                       # iteration relaxation
alpha_T = 0.6                     # iteration relaxation
poisson_alpha = 0.8                 # Pressure poisson stabilization


solver_params = {"alpha":alpha, "alpha_T":alpha_T, "poisson_alpha":poisson_alpha, 
                 "N_DIVERGENCE_ITERATIONS": N_DIVERGENCE_ITERATIONS, "INITIAL_STABLE_POINTS": INITIAL_STABLE_POINTS, 
                 "N_MAGNETIC_FIELD_ITERATIONS": N_MAGNETIC_FIELD_ITERATIONS}


# ==================================================================================================================== #
#                                              Animation / Saving
# ==================================================================================================================== #

FROM_ZERO = False
NO_TEMPERATURE = True
NEW_GRID_INTERPOLATE = True

# 260 is 15 kW
# 250 is 14 kW
# 240 is 13 kW
# 231.5 is 12 kW
# 223 is 11 kW
# 214 is 10 kW
# 204 is 9 kW
# 194 is 8 kW
# 183 is 7 kW
# 173 is 6 kW
# 161 is 5 kW
# 147 is 4 kW
# 134 is 3 kW
# 120 is 2 kW
# Below this the torch cannot operate properly. The plasma is not stable and the flow is not laminar. The torch will not ignite.

save_file_name = "161_steady_produc"
load_file_name = "161_steady_produc"

# ==================================================================================================================== #
#                                              Initialization
# ==================================================================================================================== #

from Fundamental_Methods.flow_equations import ICPState, ICPSolver
from Fundamental_Methods.magnetic_field import ElectroMagnetic

magclass = ElectroMagnetic(grid=grid, Coils=Coils, omega=OMEGA, mu0=MU0, Ic=IC, sigmaf=sigmaf)
state = ICPState(torch, grid, magclass, CFL, parameterfs)
solver = ICPSolver(torch, grid, state, magclass, solver_params)


if __name__=="__main__":
    # grid.plot_cell_faces()
    # plt.show()

    # grid.plot_cell_centres()
    # plt.show()
    # pass
    print(torch.U_inlet_main)


    # Gaan kyk na die energy equation!
