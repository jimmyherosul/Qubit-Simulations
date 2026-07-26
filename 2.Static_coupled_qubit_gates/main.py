# ----------------------------- Import all packages and libraries -----------------------------
from Tuning_parameters.static_coupled_qubit_gate import tuning_parameters
from Physics.unitary_time_evolution_operators import static_unitary_operator
from Physics.vector_coordinates import vector_coordinates
from Physics.state_probabilities import state_probabilities
from Plots.bloch_sphere_plot import create_bloch_sphere, update_bloch_sphere
from Plots.tuning_signal_plot import create_signal_plot, update_signal_plot
from Plots.vector_coordinate_plot import create_coordinate_plot, update_coordinate_plot
from Plots.state_probability_plot import create_probability_plot, update_probability_plot
from Plots.animation_control import animation_control

from qutip import Qobj, basis
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- Fixed parameters -----------------------------
hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # For simplicity, the mean unperturbed energy of the two basis states |0> and |1> is set to zero
data_points = 50              # Number of data points across time
interval = 1                  # Time interval (in milliseconds) between time frames 

# ----------------------------- Input parameters -----------------------------
# REMINDER:
# For X gate: Delta = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For Y gate: Delta = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For Z gate: Delta > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For S gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/4
# For T gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/8
# For H gate: Delta < 0, V_I_mag = Delta or 0, V_Q_mag = 0 or Delta, and t_duration such that (Omega_r/2)*t_duration = pi/8
# For I gate: Delta, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0

# Hadamard similarity transformations: HXH=Z, HYH=-Y, HZH=X

# Select an initial state
initial_state = basis(2, 0)                                      # initial state |0>
#initial_state = basis(2, 1)                                      # initial state |1>
#initial_state = (basis(2, 0) + (1+0j)*basis(2, 1)).unit()        # initial state |+>
#initial_state = (basis(2, 0) + (-1+0j)*basis(2, 1)).unit()       # initial state |->
#initial_state = (basis(2, 0) + (0+1j)*basis(2, 1)).unit()        # initial state |+i>
#initial_state = (basis(2, 0) + (0-1j)*basis(2, 1)).unit()        # initial state |-i>

# Select a sequence of single-qubit gates and set the appropriate input parameters (i.e. gate, detunings and I/Q coupling strengths) for each gate
input_parameters_in_series = [
    #{"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    #{"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    {"gate": "X", "Delta": 10e-9, "V_I_mag": 10e-6, "V_Q_mag": 10e-9},
    #{"gate": "X", "Delta": 10e-9, "V_I_mag": 10e-6, "V_Q_mag": 10e-9},
    {"gate": "Y", "Delta": 10e-9, "V_I_mag": 10e-9, "V_Q_mag": 10e-6},
    #{"gate": "Z", "Delta": 10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "S", "Delta": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "T", "Delta": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    {"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
]

# ----------------------------- Compute the time-evolved qubit states for the full gate sequence -----------------------------
def gate_operation(initial_state, input_parameters_in_series):

    # Initialize the time duration accumulated from completed gates in the sequence
    t_accumulated = 0

    # Initialize lists for accumulating the results of each completed gate in the sequence
    t_points_sequence = []
    Delta_sequence = []
    V_I_mag_sequence = []
    V_Q_mag_sequence = []
    Omega_0_sequence = []
    Omega_r_sequence = []
    qubit_states_sequence = []

    # Assign the initial state as the input state to the gate in the sequence
    input_state = initial_state    

    # For each gate in the sequence
    for inputs in input_parameters_in_series:

        # Extract the input parameters of the current gate
        gate = inputs["gate"]
        Delta = inputs["Delta"]
        V_I_mag = inputs["V_I_mag"]
        V_Q_mag = inputs["V_Q_mag"]

        # Generate the tuning parameters of the current gate from the input parameters applied over the entire time duration
        tuning_param = tuning_parameters(gate, Delta, V_I_mag, V_Q_mag, hbar, data_points)

        # Extract the tuning parameters of the current gate
        t_points = tuning_param["t_points"]
        Delta_pulse = tuning_param["Delta_pulse"]
        V_I_pulse = tuning_param["V_I_pulse"]
        V_Q_pulse = tuning_param["V_Q_pulse"]
        Omega_r = tuning_param["Omega_r"]
        Omega_0 = tuning_param["Omega_0"]

        # Initialize list for accumulating the time-evolved states of the current gate
        qubit_states_gate = []

        # Compute the time-evolved states over all time points of the current gate
        U_static = static_unitary_operator(t_points, hbar, E_mean, V_I_pulse, V_Q_pulse, Omega_r, Omega_0)
        for i in range(len(t_points)):
            qubit_state_t = Qobj(U_static[:, :, i]) * input_state
            qubit_states_gate.append(qubit_state_t)

        # Shift the elapsed time points of the current gate by the time duration accumulated from previous gates
        t_points_shifted = t_points + t_accumulated

        # Set the start index for concatenating the shifted time points of the current gate onto the complete time sequence:
        # include all points for the first gate (start_index = 0), but skip the first point of each subsequent gate (start_index = 1)
        # to avoid repeating the boundary point between consecutive gates
        if len(t_points_sequence) == 0:
            start_index = 0          
        else:
            start_index = 1

        # Concatenate the results of the current gate onto the complete gate sequence results 
        t_points_sequence.extend(t_points_shifted[start_index:])
        Delta_sequence.extend(Delta_pulse[start_index:])
        V_I_mag_sequence.extend(V_I_pulse[start_index:])
        V_Q_mag_sequence.extend(V_Q_pulse[start_index:])
        Omega_r_sequence.extend(np.full_like(t_points[start_index:], Omega_r, dtype=float))
        Omega_0_sequence.extend(np.full_like(t_points[start_index:], Omega_0, dtype=float))
        qubit_states_sequence.extend(qubit_states_gate[start_index:])

        # Update the input state of the next gate in the sequence with the final state of the current gate
        input_state = qubit_states_gate[-1]

        # Update the accumulated time duration by adding the duration of the current gate
        t_accumulated += t_points[-1]

    return {
        "t_points": np.array(t_points_sequence),
        "Delta_pulse": np.array(Delta_sequence),
        "V_I_pulse": np.array(V_I_mag_sequence),
        "V_Q_pulse": np.array(V_Q_mag_sequence),
        "Omega_r": np.array(Omega_r_sequence),
        "Omega_0": np.array(Omega_0_sequence),
        "qubit_states": qubit_states_sequence,
        "final_state": input_state
    }

sequence_results = gate_operation(initial_state, input_parameters_in_series)

# Compute the Bloch-vector coordinates of the qubit state and the probabilities of measuring it in |0> and |1> at each time step
qubit_states = sequence_results["qubit_states"]

coordinates = vector_coordinates(qubit_states)
probabilities = state_probabilities(qubit_states)

# ----------------------------- Create Bloch sphere -----------------------------
plt.close('all')

fig_bloch = plt.figure(figsize=(7, 6))
fig_bloch.suptitle("Time-Evolution of Qubit State", fontsize=14, x=0.51, y=0.95)
fig_bloch_grid = fig_bloch.add_gridspec(1, 1, width_ratios=[1.00], wspace=0.00)
bloch_view = create_bloch_sphere(figure=fig_bloch, subplot_position=fig_bloch_grid[0, 0])

# ----------------------------- Create all Time-plots -----------------------------
fig_time, (ax_tuning, ax_coordinates, ax_probabilities) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
fig_time.subplots_adjust(hspace=0.45, bottom=0.15, right=0.72)

# Specifiy all relevant signals and frequencies to be read out by the signal plot
static_signal_spec = [
    {"key": "V_I_pulse", "label": r"$V_{I_{mag}}$", "readout_label": "V_I"},
    {"key": "V_Q_pulse", "label": r"$V_{Q_{mag}}$", "readout_label": "V_Q"},
    {"key": "Delta_pulse", "label": r"$\Delta$", "readout_label": "Delta"}
]
static_frequency_spec = [{"key": "Omega_0", "readout_label": "Omega_0"}]

# Time-plot of Coupling V(t) and Detuning Delta signals
tuning_plot = create_signal_plot(ax_tuning, sequence_results, static_signal_spec, "Time-Evolution of Coupling and Detuning signals")

# Time-plot of Bloch-vector coordinates
coordinate_plot = create_coordinate_plot(ax_coordinates, sequence_results, "Time-Evolution of Bloch-vector coordinates", True)

# Time-plot of qubit-state probabilities
probability_plot = create_probability_plot(ax_probabilities, sequence_results, "Time-Evolution of Qubit-state probabilities", True)

# ----------------------------- Animate the Bloch sphere and all Time plots -----------------------------
# Number of time frames set by the number of data points in the test simulations
n_frames = len(sequence_results["t_points"])        

# Update the animation of Bloch sphere and all Time plots
def update_animation(frame):

    # Update Bloch sphere and all Time plots over time frames
    update_bloch_sphere(frame, bloch_view, qubit_states[frame], coordinates, 'red', 'blue')
    update_signal_plot(frame, sequence_results, tuning_plot, static_signal_spec, static_frequency_spec)
    update_coordinate_plot(frame, sequence_results, coordinate_plot, coordinates)
    update_probability_plot(frame, sequence_results, probability_plot, probabilities, sequence_results["Omega_r"], "Omega_r")

    fig_time.canvas.draw_idle()

    # Stop running the animation after updating the final frame
    if frame >= n_frames - 1:
        controller.finish_animation()

# Run the animation of Bloch spheres and all Time plots, with the Pause/Play control
anim = FuncAnimation(fig_bloch, update_animation, frames=n_frames, interval=interval, blit=False, repeat=False)
controller = animation_control(anim=anim, animation_figure=fig_bloch, button_figure=fig_bloch)

plt.show()
