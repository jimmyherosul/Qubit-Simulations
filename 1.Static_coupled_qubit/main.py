# ----------------------------- Import all packages and libraries -----------------------------
from Tuning_parameters.static_coupled_qubit import tuning_parameters
from Physics.energy_eigenvalues import energy_eigenvalues
from Physics.unitary_time_evolution_operators import static_unitary_operator
from Physics.vector_coordinates import vector_coordinates
from Physics.state_probabilities import state_probabilities
from Plots.bloch_sphere_plot import create_bloch_sphere, update_bloch_sphere
from Plots.bloch_coordinate_plot import create_coordinate_plot, update_coordinate_plot
from Plots.state_probability_plot import create_probability_plot, update_probability_plot
from Plots.energy_eigenvalues_plot import create_energy_plot
from Plots.animation_control import animation_control

from qutip import Qobj, basis
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

# To check the version of qutip
#import qutip
#print(qutip.__version__)

# ----------------------------- Fixed parameters -----------------------------
hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # For simplicity, the mean unperturbed energy of the two basis states |0> and |1> is set to zero
data_points = 50              # Number of data points across time
interval = 1                  # Time interval (in milliseconds) between time frames in animation   

# ----------------------------- Input parameters -----------------------------
#initial_state = basis(2, 0)                                      # initial state |0>
#initial_state = basis(2, 1)                                      # initial state |1>
#initial_state = (basis(2, 0) + (1+0j)*basis(2, 1)).unit()        # initial state |+>
#initial_state = (basis(2, 0) + (-1+0j)*basis(2, 1)).unit()       # initial state |->
#initial_state = (basis(2, 0) + (0+1j)*basis(2, 1)).unit()        # initial state |+i>
#initial_state = (basis(2, 0) + (0-1j)*basis(2, 1)).unit()        # initial state |-i>

# Select the test simulations to run in parallel and set their respective input parameters (i.e. initial state, pulse durations, detunings and I/Q coupling strengths)
input_parameters_in_parallel = [
    {"initial_state": basis(2, 0), "t_duration": 10e-11, "Delta": -1e-6, "V_I_mag": 10e-6, "V_Q_mag": 1e-6, "color": "blue"},
    {"initial_state": basis(2, 0), "t_duration": 10e-11, "Delta": -1e-6, "V_I_mag": 20e-6, "V_Q_mag": 1e-6, "color": "orange"},
    {"initial_state": basis(2, 0), "t_duration": 10e-11, "Delta": -1e-6, "V_I_mag": 30e-6, "V_Q_mag": 1e-6, "color": "green"},
    {"initial_state": basis(2, 0), "t_duration": 10e-11, "Delta": -1e-6, "V_I_mag": 40e-6, "V_Q_mag": 1e-6, "color": "purple"},
]

# Compute the time-evolved qubit states for all independent test simulations in parallel
def test_operation(input_parameters_in_parallel):

    # Initialize list for storing the results from all test simulations in parallel
    results_in_parallel = []

    # For each independent test simulation
    for inputs in input_parameters_in_parallel:

        # Extract the input parameters of the current test simulation
        input_state = inputs["initial_state"]
        t_duration = inputs["t_duration"]
        Delta = inputs["Delta"]
        V_I_mag = inputs["V_I_mag"]
        V_Q_mag = inputs["V_Q_mag"]
        curve_colour = inputs["color"]

        # Generate the tuning parameters of the current test simulation from the input parameters applied over the entire time duration
        tuning_param = tuning_parameters(t_duration, Delta, V_I_mag, V_Q_mag, hbar, data_points)

        # Extract the tuning parameters of the current test simulation
        t_points = tuning_param["t_points"]
        V_I_pulse = tuning_param["V_I_pulse"]
        V_Q_pulse = tuning_param["V_Q_pulse"]
        V_mag = tuning_param["V_mag"]
        Omega_r = tuning_param["Omega_r"]
        Omega_0 = tuning_param["Omega_0"]

        # Initialize list for accumulating the time-evolved states of the current test simulation
        qubit_states_test = []

        # Compute the time-evolved states over all time points of the current test simulation
        U_static = static_unitary_operator(t_points, hbar, E_mean, V_I_pulse, V_Q_pulse, Omega_r, Omega_0)
        for i in range(len(t_points)):
            qubit_state_t = Qobj(U_static[:, :, i]) * input_state
            qubit_states_test.append(qubit_state_t)

        # Append the complete results of the current test simulation to the collection of parallel simulation results
        results_in_parallel.append({
            "t_points": np.array(t_points),
            "Delta": Delta,
            "V_I_mag": V_I_mag,
            "V_Q_mag": V_Q_mag,
            "V_mag": V_mag,
            "curve_colour": curve_colour,
            "Omega_r": np.full_like(t_points, Omega_r, dtype=float),
            "Omega_0": np.full_like(t_points, Omega_0, dtype=float),
            "qubit_states": qubit_states_test,
            "final_state": qubit_states_test[-1]
        })

    return results_in_parallel

results_in_parallel = test_operation(input_parameters_in_parallel)

# For each test simulation in results_in_parallel, compute the Bloch-vector coordinates of the qubit state and the probabilities of measuring it in |0> and |1> at each time step
for test_sim in results_in_parallel:
    qubit_states = test_sim["qubit_states"]
    
    test_sim["coordinates"] = vector_coordinates(qubit_states)
    test_sim["probabilities"] = state_probabilities(qubit_states)

# ----------------------------- Create Bloch spheres -----------------------------
plt.close('all')

# Initialize list for storing the Bloch spheres and the results of all test simulations
test_sim_properties = []

fig_bloch = plt.figure(figsize=(4 + 3*len(results_in_parallel), 4))
fig_bloch.suptitle("Time-Evolution of Qubit States", fontsize=14, y=0.98)
fig_bloch.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.18, wspace=0.00, hspace=0.25)
fig_bloch_grid = fig_bloch.add_gridspec(1, len(results_in_parallel)+1, width_ratios=[1.00] + [0.85]*len(results_in_parallel), wspace=0.00)

# Create Bloch sphere for each test simulation running in parallel
for sim_index, test_sim in enumerate(results_in_parallel):
    bloch_sphere_test = create_bloch_sphere(figure=fig_bloch, subplot_position=fig_bloch_grid[0, sim_index+1])

    # Append the Bloch spheres and the results of current test simulation
    test_sim_properties.append({
        "test_sim": test_sim, 
        "bloch_sphere_test": bloch_sphere_test,
        "coordinates_test": test_sim["coordinates"],
        "probabilities_test": test_sim["probabilities"],
        "colour_test": test_sim["curve_colour"] 
        })

# ----------------------------- Plot Energy-Delta diagram -----------------------------
ax_energy = fig_bloch.add_subplot(fig_bloch_grid[0, 0])

# Retrieve the values of Delta and V_mag across the different test simulations in results_in_parallel
Delta_values = np.array([test_sim["Delta"] for test_sim in results_in_parallel])
V_mag_values = np.array([test_sim["V_mag"] for test_sim in results_in_parallel])

# Compute symmetric detuning range that includes all specified detunings and coupling strengths, with some margin
Delta_limit = 1.1*max(np.max(np.abs(Delta_values)), np.max(np.abs(V_mag_values)), 1e-12)
Delta_range = np.linspace(-Delta_limit, Delta_limit, 500)

# Create the energy diagram
energy_diagram = create_energy_plot(ax_energy, results_in_parallel)

# Plot the unperturbed energy curve over detuning 
energy_diagram["unperturbed_energy"]["E_plus"].set_data(Delta_range*1e6, Delta_range*1e6)
energy_diagram["unperturbed_energy"]["E_minus"].set_data(Delta_range*1e6, -Delta_range*1e6)

# Plot the coupled energy curves over detuning at different coupling strengths, including the currently tested values
for sim_index, test_sim in enumerate(results_in_parallel):
    Delta = test_sim["Delta"]
    V_mag = test_sim["V_mag"]

    energy_values = energy_eigenvalues(V_mag=V_mag, Delta_range=Delta_range, Delta=Delta, E_mean=E_mean)

    energy_function = energy_diagram["perturbed_energy"][sim_index]
    energy_function["E_plus"].set_data(Delta_range*1e6, (energy_values["E_plus"] - E_mean)*1e6)
    energy_function["E_minus"].set_data(Delta_range*1e6, (energy_values["E_minus"] - E_mean)*1e6)
    energy_function["E_plus_current"].set_data([Delta*1e6], [(energy_values["E_plus_current"] - E_mean)*1e6])
    energy_function["E_minus_current"].set_data([Delta*1e6], [(energy_values["E_minus_current"] - E_mean)*1e6])

ax_energy.relim()
ax_energy.autoscale_view()

# ----------------------------- Create all Time-plots -----------------------------
fig_time = plt.figure(figsize=(4*len(results_in_parallel), 6))
fig_time.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.09, wspace=0.30, hspace=0.35)
fig_time_grid = fig_time.add_gridspec(2, len(results_in_parallel), wspace=0.30, hspace=0.35)

# For each test simulation
for sim_index, properties in enumerate(test_sim_properties):

    # Retrieve the results for current test simulation and its corresponding colour code  
    test_sim = properties["test_sim"]
    V_mag_ueV = int(round(test_sim["V_mag"]*1e6))
    Delta_ueV = test_sim["Delta"]*1e6
    colour_test = properties["colour_test"]

    # Plot the Bloch-vector coordinates with the corresponding colour for current test simulation
    ax_coordinates = fig_time.add_subplot(fig_time_grid[0, sim_index])
    coordinate_plot_test = create_coordinate_plot(ax_coordinates, test_sim, fr"$|V|={V_mag_ueV}$ ueV, $\Delta={Delta_ueV:g}$ ueV", False)
    ax_coordinates.title.set_color(colour_test)
    properties["coordinate_plot_test"] = coordinate_plot_test

    # Plot the state probabilities with the corresponding colour for current test simulation
    ax_probabilities = fig_time.add_subplot(fig_time_grid[1, sim_index], sharex=ax_coordinates)
    probability_plot_test = create_probability_plot(ax_probabilities, test_sim, "", False)
    properties["probability_plot_test"] = probability_plot_test

# ----------------------------- Animate the combined Bloch spheres and all Time plots -----------------------------
# Number of time frames set by the number of data points in the test simulations
# (All test simulations have same number of data points, so the maximum is the same for every simulation)
n_frames = max(len(test_sim["t_points"]) for test_sim in results_in_parallel)  

# Update the animation of Bloch sphere and all Time plots
def update_animation(frame):

    # For each test simulation
    for properties in test_sim_properties:
        test_sim = properties["test_sim"]

        qubit_states_test = test_sim["qubit_states"]
        coordinates_test = properties["coordinates_test"]
        probabilities_test = properties["probabilities_test"]
        colour_test = properties["colour_test"]

        # Update Bloch sphere and all Time plots frame by frame for current test simulation
        update_bloch_sphere(frame, properties["bloch_sphere_test"], qubit_states_test[frame], coordinates_test, colour_test, colour_test)
        update_coordinate_plot(frame, test_sim, properties["coordinate_plot_test"], coordinates_test)
        update_probability_plot(frame, test_sim, properties["probability_plot_test"], probabilities_test, test_sim["Omega_r"], "Omega_r")

    fig_time.canvas.draw_idle()

    # Stop running the animation after updating the final frame 
    if frame >= n_frames - 1:
        anim.event_source.stop()

# Run the animation of Bloch spheres and all Time plots, with the Pause/Play control
anim = FuncAnimation(fig_bloch, update_animation, frames=n_frames, interval=interval, blit=False, repeat=False)
controller = animation_control(anim=anim, animation_figure=fig_bloch, button_figure=fig_bloch)

plt.show()
