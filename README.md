# ICP-Torch-Modelling-and-Spheroidisation

A Python-based multiphysics model for a radio-frequency inductively coupled plasma (RF-ICP) torch and the thermal processing of titanium particles.

The repository contains a two-dimensional axisymmetric model of a Tekna PL-50-type ICP torch, coupling:

a frequency-domain electromagnetic solver,

a variable-density thermo-fluid solver on a staggered finite-volume grid,

temperature-dependent argon properties,

Joule-heating and Lorentz-force source terms,

one-way coupled Lagrangian titanium-particle tracking,

particle heating, melting and evaporation,

particle-wall interaction modelling,

mesh, operating-condition and electromagnetic validation tools, and

publication-oriented post-processing and particle-processing maps.

The model was developed for investigating how absorbed plasma power, particle diameter, injection position and particle injection velocity affect titanium-particle melting, evaporation, residence time, trajectory and downstream recovery.

Scientific scope

The plasma model represents a 2D axisymmetric RF-ICP torch under local thermodynamic equilibrium (LTE). The electromagnetic field is solved in the frequency domain using the azimuthal magnetic vector potential, with time-averaged Joule heating and Lorentz forces coupled to the thermo-fluid solution.

The thermo-fluid solver uses a structured, non-uniform staggered grid. Pressure, temperature and density are stored at cell centres, while the axial and radial velocity components are stored on their corresponding cell faces. A pressure-correction/projection procedure is used to enforce variable-density mass conservation.

After convergence of a plasma operating condition, the steady plasma fields can be supplied to a one-way coupled Lagrangian particle model. Titanium particles are tracked through the torch while resolving aerodynamic drag, gravity with buoyancy correction, convective heat transfer, radiative heat loss, sensible heating, melting and evaporation.

Complete melting is used as the practical numerical criterion for a particle becoming thermally capable of spheroidisation. Particle thermal state is nevertheless considered separately from particle transport fate, because a particle may melt successfully but subsequently collide with a wall or escape through the upstream boundary.

Reference operating condition

The default geometry and operating parameters are based on the Tekna PL-50 torch configuration used in the associated study.

Parameter

Default value

RF frequency

3 MHz

Carrier argon flow

1 slpm

Main argon flow

3 slpm

Sheath argon flow

31 slpm

Torch axial length

200 mm

Torch radius

25 mm

Coil axial positions

63, 92 and 121 mm

Coil radius

33 mm

Reference coil current

161 A

Reference absorbed power

approximately 5 kW

Wall/inlet temperature

300 K

The operating parameters, torch geometry, mesh selection, solver relaxation parameters and file names used for loading/saving states are defined primarily in input.py.

Repository structure

ICP-Torch-Modelling-and-Spheroidisation/
│
├── Fundamental_Methods/
│   ├── flow_equations.py
│   ├── helpers.py
│   ├── live_simulation_modular.py
│   └── magnetic_field.py
│
├── Parameters/
│   ├── mesh.py
│   ├── parameters.py
│   └── temperature_field.py
│
├── Particle_Solver/
│   ├── particle_animation.py
│   ├── particle_parameters.py
│   ├── particle_postprocessing.py
│   └── particles.py
│
├── Post_Process/
│   ├── mesh_study.py
│   ├── plotting_fields.py
│   └── post_processing.py
│
├── saved_states/
│   └── *.npz
│
├── input.py
├── main.py
├── run_particle_power_study.py
├── validate_magnetic_field.py
└── README.md

Main modules

input.py
Defines the torch geometry, mesh, coil geometry, RF frequency, coil current, argon-property functions, CFL number, relaxation factors and solver objects. It is the main configuration file for the plasma model.

Fundamental_Methods/magnetic_field.py
Contains the frequency-domain electromagnetic solver for the complex azimuthal magnetic vector potential. The calculated field is used to obtain the electric field, magnetic field, Joule power density and Lorentz-force components.

Fundamental_Methods/flow_equations.py
Contains the main ICP state and thermo-fluid solver classes, including the staggered-grid flow equations, thermal solution, pressure correction, boundary conditions and pseudo-time advancement.

Fundamental_Methods/live_simulation_modular.py
Provides the optional PySide6/pyqtgraph live simulation interface for monitoring the developing plasma solution.

Particle_Solver/particles.py
Contains the plasma-field loader, particle state representation, single-particle trajectory/thermal solution and parametric particle-study tools.

Particle_Solver/particle_postprocessing.py
Produces publication-oriented trajectory figures, thermal-history plots, particle-state maps, melt-location maps, wall-contact diagnostics and processing envelopes.

Post_Process/post_processing.py
Loads converged .npz plasma states, converts staggered fields to common cell-centred fields and provides contour, profile, streamline and comparison plotting functions.

Post_Process/mesh_study.py
Performs mesh-sensitivity and absorbed-power analysis from converged plasma states and produces publication-quality figures and tables.

validate_magnetic_field.py
Runs electromagnetic validation cases based on the ICP literature, including a coil-in-air Biot-Savart comparison and a conducting-medium skin-depth comparison.

run_particle_power_study.py
Defines the power/particle parametric study and generates the particle-processing figures used to investigate the spheroidisation operating window.

Requirements

The current repository should be run with Python 3.10 or newer.

Core dependencies are:

numpy
scipy
pandas
matplotlib
PySide6
pyqtgraph

Install them with:

pip install numpy scipy pandas matplotlib PySide6 pyqtgraph

PySide6 and pyqtgraph are required for the live simulation GUI. The core numerical and Matplotlib post-processing routines do not otherwise depend on the GUI.

Installation

Clone the repository:

git clone https://github.com/Eisbergh/ICP-Torch-Modelling-and-Spheroidisation.git
cd ICP-Torch-Modelling-and-Spheroidisation

It is recommended to use a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

or on Linux/macOS:

source .venv/bin/activate

Then install the dependencies:

pip install numpy scipy pandas matplotlib PySide6 pyqtgraph

Running the code

Run commands from the repository root, because several scripts use relative paths such as saved_states/, particle_results/ and journal_plots/.

1. Post-process converged plasma states

The committed main.py currently contains examples that load converged states and compare fields at approximately 2, 8 and 15 kW.

python main.py

The plotting examples use ICPPost and can be modified to display quantities such as:

plasma temperature,

axial/radial velocity,

velocity magnitude,

pressure,

magnetic-field magnitude,

Joule power density,

Lorentz-force components,

axial and radial profiles, and

streamlines.

Converged plasma solutions are stored in compressed NumPy .npz files inside saved_states/.

2. Run the live coupled plasma solver

main.py imports the live simulation interface, but the call is currently commented out.

To launch the GUI, enable:

animate(input, Fundamental_Methods.helpers)

near the top of main.py, then run:

python main.py

The simulation can either begin from the configured initial state or continue from a saved solution, depending on the flags in input.py.

Important configuration variables include:

FROM_ZERO
NO_TEMPERATURE
NEW_GRID_INTERPOLATE
save_file_name
load_file_name
IC
CFL
N_DIVERGENCE_ITERATIONS
N_MAGNETIC_FIELD_ITERATIONS
alpha
alpha_T
poisson_alpha

3. Electromagnetic validation

Run:

python validate_magnetic_field.py

The script contains two principal tests:

Coil in air (sigma = 0 S/m) with comparison against the analytical Biot-Savart centreline magnetic field.

Conducting-medium test (sigma = 2500 S/m) with comparison against the analytical skin-depth approximation.

Validation plots are written to:

journal_plots/magnetic_validation/

Optional digitised literature data can also be supplied through busse_fig3.csv and busse_fig4.csv as described in the validation script.

4. Mesh and absorbed-power study

Run:

python Post_Process/mesh_study.py

This script reads the converged states defined in POWER_FILES and MESH_FILES and generates journal-quality figures and CSV summaries. The current power calibration spans approximately 2-15 kW.

5. Particle spheroidisation study

The default particle study in run_particle_power_study.py uses:

absorbed powers from approximately 2 to 15 kW,

initial particle diameters from 20 to 400 micrometres,

eight radial injection positions from 0.1 to 1.5 mm,

an initial particle temperature of 300 K,

an explicit particle time step of 2e-5 s, and

reflective wall collisions with restitution/friction coefficients defined in the script.

To generate the full parametric data set, uncomment:

study.run(filename="particle_parametric_results.csv")

in run_particle_power_study.py, then run:

python run_particle_power_study.py

The generated CSV is placed under:

particle_results/particle_parametric_results.csv

The remainder of the script can then produce figures including:

particle trajectories at different powers,

diameter and radial-injection comparisons,

detailed particle thermal histories,

spheroidisation/thermal-state maps,

median full-melt location,

particle mass-loss maps,

upstream-loss regions,

wall-contact fractions,

exit-radius response, and

combined thermal and downstream processing envelopes.

Processing-envelope definition

The particle study distinguishes between thermal capability and successful downstream processing.

A particle is considered thermally capable of spheroidisation after complete melting. For the processing maps used in the associated study, a power/diameter condition is generally considered thermally viable when at least 50% of the sampled radial injection positions:

reach complete melting, and

retain at least 90% of their initial mass.

The downstream-processing envelope imposes the additional requirement that the particle reaches the downstream outlet. This distinction is important at high absorbed power, where small particles may melt successfully but become strongly entrained by the plasma recirculation and leave through the upstream boundary.

The 10% mass-loss criterion is an engineering screening threshold used in the present analysis, not a universal physical spheroidisation limit.

Numerical model summary

Electromagnetics

frequency-domain formulation,

complex azimuthal magnetic vector potential,

non-uniform finite-difference stencil,

temperature-dependent electrical conductivity,

non-local electromagnetic boundary treatment,

time-averaged Joule heating, and

time-averaged Lorentz force.

Thermo-fluid model

2D axisymmetric formulation,

structured non-uniform staggered mesh,

finite-volume conservation equations,

variable-density argon,

temperature-dependent transport and thermodynamic properties,

Patankar-type power-law convection-diffusion treatment,

sensible-enthalpy energy formulation,

radiative heat-loss source term,

iterative pressure correction, and

pseudo-time convergence to a stationary plasma field.

Particle model

one-way plasma-to-particle coupling,

Lagrangian particle tracking,

aerodynamic drag,

gravity with buoyancy correction,

convective plasma-to-particle heat transfer,

particle radiation,

sensible heating,

latent melting,

liquid heating,

evaporation and diameter/mass reduction,

bilinear interpolation of the non-uniform plasma field,

upstream/downstream trajectory classification, and

optional reflective particle-wall interactions.

Validation and verification

The numerical framework has been assessed using a combination of analytical and published reference cases. The repository includes the electromagnetic validation script described above, while the associated study additionally reports comparisons with established Tekna PL-50 numerical solutions and a mesh-sensitivity analysis.

The model was used to investigate the coupled thermal and hydrodynamic consequences of increasing absorbed plasma power. A central result of the study is that increasing power changes plasma velocity much more strongly than peak plasma temperature, making particle trajectory and residence time essential when defining a useful spheroidisation window.

Model assumptions and limitations

The present framework should be interpreted within the assumptions of the model:

two-dimensional axisymmetry,

local thermodynamic equilibrium,

optically thin plasma treatment,

laminar plasma flow,

thermophysical properties treated primarily as functions of temperature,

one-way particle coupling,

dilute-particle limit,

spherical/equivalent particle representation for drag and heat transfer,

lumped particle temperature,

no explicit evolving irregular-particle geometry,

no particle charging effects,

simplified reflective wall interaction, and

no conductive particle-wall heat transfer during impacts.

For dense industrial powder loading, two-way plasma-particle coupling may become important and is not represented by the current particle model.

Associated research

This repository supports the work:

Numerical analysis of inductively coupled plasma torch dynamics for titanium particle spheroidisation using Python
Hendrik J. Greeff and Samuel A. Iwarere
Department of Chemical Engineering, University of Pretoria, South Africa.

The study investigates how RF-ICP operating power influences both plasma thermal behaviour and particle transport, and uses combined thermal/transport processing envelopes to identify conditions suitable for titanium-particle spheroidisation.

Publication details and a formal citation can be added here once available.

Reproducibility notes

The saved_states/ directory contains converged plasma-field files used by the post-processing and particle routines. File names generally identify the coil-current condition used to obtain each state.

For publication-quality reproduction, use the exact file mappings defined in the analysis scripts rather than assuming that the numerical part of a file name is the absorbed power. Coil current and absorbed plasma power are related non-linearly in the model.

Because the project is research code, several scripts are intentionally configured by editing variables directly rather than through a command-line interface. Users should inspect input.py, Post_Process/mesh_study.py and run_particle_power_study.py before starting a new calculation.

Contributing

This repository was developed primarily as a research solver rather than as a general-purpose CFD package. Bug reports, numerical-validation suggestions and improvements to documentation or reproducibility are welcome through GitHub issues or pull requests.

License

No software license is currently included in the repository. A license should be added before redistribution or reuse terms are assumed.
