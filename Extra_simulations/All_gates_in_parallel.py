# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from qutip import Bloch, Qobj, basis, expect, sigmax, sigmay, sigmaz, ket2dm

# To check the version of qutip
#import qutip
#print(qutip.__version__)


# ----------------------------- Parameters -----------------------------
# Delta - Detuning (i.e. half-energy difference between the two basis states) in eV
# V_mag - Coupling strength between the two basis states |0> and |1> in eV
# Omega_r - Rabi frequency in rad/s
# Omega_0 - Unperturbed Detuning frequency in rad/s
# t_duration - Pulse duration for the gate in s
# phi - Phase of the coupling term V between the two basis states |0> and |1> in rad

# where Delta, V_mag, t_duration, and phi are tuning parameters for implementing logic gates.

# Throughout this simulation, energy is in eV, time is in s and phase is in rad. Using the reduced Planck constant, the (angular) frequency is therefore in rad/s, unless otherwise specified.
# Note that for a frequency range of up to 10GHz (6.3x10^10 rad/s), the corresponding energy range is up to approx 41ueV, which is typical for superconducting qubits.

hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # For simplicity, the mean unperturbed energy of the two basis states |0> and |1> is set to zero

# Generating tuning parameters for gate operation
def tuning_parameters(gate_name, Delta, V_I_mag, V_Q_mag):

    # All I/Q Coupling and Detuning signals are generated as a square-wave pulse over t_duration
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_0 = 2*Delta/hbar 
    Omega_r = 2*np.sqrt(Delta**2 + V_mag**2)/hbar 

    # For X gate: Delta = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
    if gate_name == 'X':
        t_duration = (np.pi/2)*(2/Omega_r)
    
    # For Y gate: Delta = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
    elif gate_name == 'Y':      
        t_duration = (np.pi/2)*(2/Omega_r)

    # For Z gate: Delta > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
    elif gate_name == 'Z':      
        t_duration = (np.pi/2)*(2/Omega_r)

    # For S gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/4
    elif gate_name == 'S':      
        t_duration = (np.pi/4)*(2/Omega_r)
        
    # For T gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/8
    elif gate_name == 'T':      
        t_duration = (np.pi/8)*(2/Omega_r)

    # For H gate: Delta < 0, V_I_mag = Delta or 0, V_Q_mag = 0 or Delta, and t_duration such that (Omega_r/2)*t_duration = pi/8
    elif gate_name == 'H':      
        t_duration = (np.pi/2)*(2/Omega_r)

    # For I gate: Delta, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0
    elif gate_name == 'I':   
        t_duration = 0*(2/Omega_r)

    t_points = np.linspace(0, t_duration, 50) 
    Delta_signal = np.full_like(t_points, Delta)
    V_I_amplitude = np.full_like(t_points, V_I_mag)
    V_Q_amplitude = np.full_like(t_points, V_Q_mag)

    return {
        "Omega_r": Omega_r, 
        "Omega_0": Omega_0, 
        "t_points": t_points, 
        "Delta_signal": Delta_signal, 
        "V_I_amplitude": V_I_amplitude, 
        "V_Q_amplitude": V_Q_amplitude
        }


# ----------------------------- Unitary time-evolution operator for statically coupled two-level system ----------------------------- 
def unitary_operator(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, Omega_r, Omega_0):        
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[np.cos(Omega_r*t/2) + 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2), 
                                            -1j*(V_I_amplitude - 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2)],
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2), 
                                            np.cos(Omega_r*t/2) - 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2)]])
    
    return U


# ----------------------------- Time evolution of Qubit state -----------------------------
# REMINDER:
# For X gate: Delta = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For Y gate: Delta = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For Z gate: Delta > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/2
# For S gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/4
# For T gate: Delta < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_r/2)*t_duration = pi/8
# For H gate: Delta < 0, V_I_mag = Delta or 0, V_Q_mag = 0 or Delta, and t_duration such that (Omega_r/2)*t_duration = pi/8
# For I gate: Delta, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0

# Select the single-qubit gates to run in parallel and set the appropriate detuning, I/Q coupling strengths, and initial states
# Each dictionary represents one independent gate simulation.
gate_sequence = [
    {
        "gate": "X",
        "Delta": 0,
        "V_I_mag": 10e-6,
        "V_Q_mag": 0,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
    {
        "gate": "Y",
        "Delta": 0,
        "V_I_mag": 0,
        "V_Q_mag": 10e-6,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
    {
        "gate": "Z",
        "Delta": 10e-6,
        "V_I_mag": 0,
        "V_Q_mag": 0,
        "initial_state": (basis(2, 0) + (1+0j)*basis(2, 1)).unit(),        # initial state |+>
        "initial_state_name": "|+>",
        "initial_state_label": r"$|+\rangle$"
    },
    {
        "gate": "S",
        "Delta": -10e-6,
        "V_I_mag": 0,
        "V_Q_mag": 0,
        "initial_state": (basis(2, 0) + (1+0j)*basis(2, 1)).unit(),        # initial state |+>
        "initial_state_name": "|+>",
        "initial_state_label": r"$|+\rangle$"
    },
    {
        "gate": "T",
        "Delta": -10e-6,
        "V_I_mag": 0,
        "V_Q_mag": 0,
        "initial_state": (basis(2, 0) + (1+0j)*basis(2, 1)).unit(),        # initial state |+>
        "initial_state_name": "|+>",
        "initial_state_label": r"$|+\rangle$"
    },
    {
        "gate": "H",
        "Delta": -10e-6,
        "V_I_mag": -10e-6,
        "V_Q_mag": 0,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
]


# Compute the time-evolved qubit states for each gate independently
# Each gate starts from its own specified initial state, so the gates are simulated in parallel rather than as a serial sequence.
def gate_operation(gate_settings):
    gate = gate_settings["gate"]
    Delta = gate_settings["Delta"]
    V_I_mag = gate_settings["V_I_mag"]
    V_Q_mag = gate_settings["V_Q_mag"]
    initial_state = gate_settings["initial_state"]
    initial_state_name = gate_settings["initial_state_name"]
    initial_state_label = gate_settings["initial_state_label"]

    # Generate tuning parameters (i.e. time duration, I/Q coupling signals and detuning signal) for the current gate.
    # The time points for each gate in tuning_parameters function always start from t = 0.
    gate_param = tuning_parameters(gate, Delta, V_I_mag, V_Q_mag)

    t_current_gate = gate_param["t_points"]

    # Apply the corresponding unitary time-evolution operator to the specified initial state for this gate
    U_gate = unitary_operator(
        t_current_gate,
        hbar,
        E_mean,
        gate_param["V_I_amplitude"],
        gate_param["V_Q_amplitude"],
        gate_param["Omega_r"],
        gate_param["Omega_0"]
        )

    qubit_states = []

    for i in range(len(t_current_gate)):
        state_t = Qobj(U_gate[:, :, i]) * initial_state

        # Store the resulting qubit state from each time point
        qubit_states.append(state_t)

    gate_info = {
        "gate": gate,
        "Delta": Delta,
        "V_I_mag": V_I_mag,
        "V_Q_mag": V_Q_mag,
        "Omega_r": gate_param["Omega_r"],
        "Omega_0": gate_param["Omega_0"],
        "t_start": 0,
        "t_end": t_current_gate[-1],
        "t_duration": t_current_gate[-1],
        "initial_state_name": initial_state_name,
        "initial_state_label": initial_state_label
    }

    return {
        "t_points": np.array(t_current_gate),
        "Delta_signal": np.array(gate_param["Delta_signal"]),
        "V_I_amplitude": np.array(gate_param["V_I_amplitude"]),
        "V_Q_amplitude": np.array(gate_param["V_Q_amplitude"]),
        "qubit_states": qubit_states,
        "gate_info": gate_info
    }


def compute_bloch_data(qubit_states):
    # Obtaining the Bloch-sphere coordinates of the qubit states at each time step
    x = np.array([expect(sigmax(), state).real for state in qubit_states])
    y = np.array([expect(sigmay(), state).real for state in qubit_states])
    z = np.array([expect(sigmaz(), state).real for state in qubit_states])
    r_mag = np.sqrt(x**2 + y**2 + z**2)

    # Computing the probabilities of measuring the qubit in states |0> and |1> at each time step
    p0 = np.array([ket2dm(state)[0, 0].real for state in qubit_states])
    p1 = np.array([ket2dm(state)[1, 1].real for state in qubit_states])
    p_total = p0 + p1

    return {
        "x": x,
        "y": y,
        "z": z,
        "r_mag": r_mag,
        "p0": p0,
        "p1": p1,
        "p_total": p_total
    }


# Store the independent simulation result for each gate in the gate_sequence list
parallel_gate_results = []

for gate_settings in gate_sequence:
    gate_param = gate_operation(gate_settings)
    gate_param["bloch_data"] = compute_bloch_data(gate_param["qubit_states"])
    parallel_gate_results.append(gate_param)


# ----------------------------- Creating Bloch sphere and all Time-plots in two 2x3 windows -----------------------------
plt.close('all')

interval = 10       # Time between frames in milliseconds

animation_controls = []        # Keeps all animations and buttons alive while the figures are open
plot_controls = []             # Stores the artists that must be updated for each gate

if len(parallel_gate_results) == 0:
    print("No gates selected. Add at least one gate dictionary to gate_sequence.")

else:
    # ----------------------------- Creating combined Bloch-sphere window -----------------------------
    fig_bloch = plt.figure(figsize=(15, 9), num="Bloch Spheres - Parallel Single-Qubit Gates")
    fig_bloch.suptitle("Time-Evolution of Quantum States for Single-Qubit Logic Gates", fontsize=16, y=0.98)
    fig_bloch.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.12,
        wspace=0.10,
        hspace=0.25
    )

    bloch_axes = []
    bloch_spheres = []

    for subplot_index in range(6):
        ax_bloch = fig_bloch.add_subplot(2, 3, subplot_index + 1, projection='3d')
        bloch_axes.append(ax_bloch)

        if subplot_index < len(parallel_gate_results):
            b_lab = Bloch(axes=ax_bloch)
            b_lab.font_size = 10
            bloch_spheres.append(b_lab)
        else:
            ax_bloch.set_axis_off()

    # ----------------------------- Creating combined Time-plot window -----------------------------
    fig_time = plt.figure(figsize=(18, 10), num="Time Plots - Parallel Single-Qubit Gates")
    fig_time.suptitle("Time Plots for Logic Single-Qubit Gates", fontsize=16, y=0.98)

    outer_grid = fig_time.add_gridspec(
        2, 3,
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.06,
        wspace=0.28,
        hspace=0.35
    )

    for subplot_index in range(6):
        row = subplot_index // 3
        col = subplot_index % 3

        if subplot_index < len(parallel_gate_results):
            gate_param = parallel_gate_results[subplot_index]
            gate_name = gate_param["gate_info"]["gate"]
            initial_state_name = gate_param["gate_info"]["initial_state_name"]
            initial_state_label = gate_param["gate_info"]["initial_state_label"]
            gate_label = f"{gate_name} Gate"

            qubit_states = gate_param["qubit_states"]
            bloch_data = gate_param["bloch_data"]
            x = bloch_data["x"]
            y = bloch_data["y"]
            z = bloch_data["z"]
            r_mag = bloch_data["r_mag"]
            p0 = bloch_data["p0"]
            p1 = bloch_data["p1"]
            p_total = bloch_data["p_total"]

            t_ns = gate_param["t_points"] * 1e9
            t_min_ns = t_ns[0]
            t_max_ns = t_ns[-1]

            # This prevents a singular x-axis if the identity gate has zero duration
            if np.isclose(t_min_ns, t_max_ns):
                t_max_ns = t_min_ns + 1e-3

            max_energy_ueV = max(
                np.max(np.abs(gate_param["Delta_signal"]))*1e6,
                np.max(np.abs(gate_param["V_I_amplitude"]))*1e6,
                np.max(np.abs(gate_param["V_Q_amplitude"]))*1e6,
                10
            )
            energy_limit = 1.1 * max_energy_ueV

            # Each cell in the 2x3 time-plot window contains three stacked plots for one gate
            inner_grid = outer_grid[row, col].subgridspec(3, 1, hspace=0.15)
            ax_tuning = fig_time.add_subplot(inner_grid[0, 0])
            ax_coordinates = fig_time.add_subplot(inner_grid[1, 0], sharex=ax_tuning)
            ax_probabilities = fig_time.add_subplot(inner_grid[2, 0], sharex=ax_tuning)

            # ----------------------------- Plotting Time-evolution of Coupling V and Detuning Delta signals -----------------------------
            ax_tuning.set_title(f"{gate_label}, Initial State: {initial_state_label}", fontsize=10, pad=4)
            ax_tuning.set_xlim(t_min_ns, t_max_ns)
            ax_tuning.set_ylim(-energy_limit, energy_limit)
            ax_tuning.set_ylabel("Energy (ueV)", fontsize=8)
            ax_tuning.tick_params(axis='both', labelsize=7)
            ax_tuning.minorticks_on()
            ax_tuning.grid(True, which="major", linestyle="-", linewidth=0.6)
            ax_tuning.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.7)

            coupling_I, = ax_tuning.plot([], [], label=r"$V_I$")
            coupling_Q, = ax_tuning.plot([], [], label=r"$V_Q$")
            detuning_line, = ax_tuning.plot([], [], label=r"$\Delta$")

            time_tuning_cursor = ax_tuning.axvline(
                gate_param["t_points"][0]*1e9,
                color='r',
                linestyle='--',
                linewidth=1.0
            )

            tuning_cursor, = ax_tuning.plot([], [], 'ro', markersize=3)
            ax_tuning.legend(loc="upper right", fontsize=6)

            # ----------------------------- Plotting Time-evolution of Bloch-vector coordinates -----------------------------
            ax_coordinates.set_xlim(t_min_ns, t_max_ns)
            ax_coordinates.set_ylim(-1.1, 1.1)
            ax_coordinates.set_ylabel("Pauli Expectation\n values", fontsize=8)
            ax_coordinates.tick_params(axis='both', labelsize=7)
            ax_coordinates.minorticks_on()
            ax_coordinates.grid(True, which="major", linestyle="-", linewidth=0.6)
            ax_coordinates.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.7)

            coordinate_x, = ax_coordinates.plot([], [], label=r"$\langle\sigma_x\rangle$")
            coordinate_y, = ax_coordinates.plot([], [], label=r"$\langle\sigma_y\rangle$")
            coordinate_z, = ax_coordinates.plot([], [], label=r"$\langle\sigma_z\rangle$")
            r_magnitude, = ax_coordinates.plot([], [], color="black", label=r"$|\mathbf{r}|$")

            time_coordinate_cursor = ax_coordinates.axvline(
                gate_param["t_points"][0]*1e9,
                color='r',
                linestyle='--',
                linewidth=1.0
            )

            coordinate_cursor, = ax_coordinates.plot([], [], 'ro', markersize=3)
            ax_coordinates.legend(loc="upper right", fontsize=6)

            # ----------------------------- Plotting Time-evolution of transition probabilities -----------------------------
            ax_probabilities.set_xlim(t_min_ns, t_max_ns)
            ax_probabilities.set_ylim(0, 1.1)
            ax_probabilities.set_xlabel("Time (ns)", fontsize=8)
            ax_probabilities.set_ylabel("Probability", fontsize=8)
            ax_probabilities.tick_params(axis='both', labelsize=7)
            ax_probabilities.minorticks_on()
            ax_probabilities.grid(True, which="major", linestyle="-", linewidth=0.6)
            ax_probabilities.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.7)

            prob_0, = ax_probabilities.plot([], [], label=r"$P_0$")
            prob_1, = ax_probabilities.plot([], [], label=r"$P_1$")
            prob_total, = ax_probabilities.plot([], [], color="black", label=r"$P_{\mathrm{total}}$")

            time_probability_cursor = ax_probabilities.axvline(
                gate_param["t_points"][0]*1e9,
                color='r',
                linestyle='--',
                linewidth=1.0
            )

            probability_cursor, = ax_probabilities.plot([], [], 'ro', markersize=3)
            ax_probabilities.legend(loc="upper right", fontsize=6)

            plt.setp(ax_tuning.get_xticklabels(), visible=False)
            plt.setp(ax_coordinates.get_xticklabels(), visible=False)

            plot_controls.append({
                "gate_param": gate_param,
                "bloch_sphere": bloch_spheres[subplot_index],
                "bloch_axis": bloch_axes[subplot_index],
                "gate_label": gate_label,
                "initial_state_name": initial_state_name,
                "initial_state_label": initial_state_label,
                "qubit_states": qubit_states,
                "x": x,
                "y": y,
                "z": z,
                "r_mag": r_mag,
                "p0": p0,
                "p1": p1,
                "p_total": p_total,
                "coupling_I": coupling_I,
                "coupling_Q": coupling_Q,
                "detuning_line": detuning_line,
                "time_tuning_cursor": time_tuning_cursor,
                "tuning_cursor": tuning_cursor,
                "coordinate_x": coordinate_x,
                "coordinate_y": coordinate_y,
                "coordinate_z": coordinate_z,
                "r_magnitude": r_magnitude,
                "time_coordinate_cursor": time_coordinate_cursor,
                "coordinate_cursor": coordinate_cursor,
                "prob_0": prob_0,
                "prob_1": prob_1,
                "prob_total": prob_total,
                "time_probability_cursor": time_probability_cursor,
                "probability_cursor": probability_cursor
            })

        else:
            blank_axis = fig_time.add_subplot(outer_grid[row, col])
            blank_axis.set_axis_off()

    # ----------------------------- Animating the combined Bloch sphere and combined Time plots -----------------------------
    n_frames = max(len(gate_param["t_points"]) for gate_param in parallel_gate_results)

    def update_animation(frame):
        global paused, animation_finished

        for controls in plot_controls:
            gate_param = controls["gate_param"]
            frame_i = min(frame, len(gate_param["t_points"]) - 1)

            t_current = gate_param["t_points"][frame_i] * 1e9
            qubit_states = controls["qubit_states"]
            x = controls["x"]
            y = controls["y"]
            z = controls["z"]
            p0 = controls["p0"]
            p1 = controls["p1"]
            r_mag = controls["r_mag"]
            p_total = controls["p_total"]

            b_lab = controls["bloch_sphere"]
            ax_bloch = controls["bloch_axis"]

            # Updating Bloch sphere in the laboratory reference frame over time
            b_lab.clear()
            b_lab.point_color = ['r']
            b_lab.vector_color = ['b']

            b_lab.add_states(qubit_states[frame_i])
            b_lab.add_points([x[frame_i], y[frame_i], z[frame_i]], 's')
            b_lab.add_points([x[:frame_i+1], y[:frame_i+1], z[:frame_i+1]], 'l')
            b_lab.make_sphere()
            ax_bloch.set_title(f"{controls['gate_label']} from {controls['initial_state_label']}", fontsize=10, pad=4)

            # Updating I/Q Coupling and Detuning signals over time frames 
            controls["coupling_I"].set_data(gate_param["t_points"][:frame_i+1]*1e9, gate_param["V_I_amplitude"][:frame_i+1]*1e6)
            controls["coupling_Q"].set_data(gate_param["t_points"][:frame_i+1]*1e9, gate_param["V_Q_amplitude"][:frame_i+1]*1e6)
            controls["detuning_line"].set_data(gate_param["t_points"][:frame_i+1]*1e9, gate_param["Delta_signal"][:frame_i+1]*1e6)

            controls["time_tuning_cursor"].set_xdata([t_current, t_current])

            controls["tuning_cursor"].set_data(
                [t_current, t_current, t_current],
                [gate_param["V_I_amplitude"][frame_i]*1e6, gate_param["V_Q_amplitude"][frame_i]*1e6, gate_param["Delta_signal"][frame_i]*1e6]
            )

            # Updating Bloch-vector coordinates over time frames
            controls["coordinate_x"].set_data(gate_param["t_points"][:frame_i+1]*1e9, x[:frame_i+1])
            controls["coordinate_y"].set_data(gate_param["t_points"][:frame_i+1]*1e9, y[:frame_i+1])
            controls["coordinate_z"].set_data(gate_param["t_points"][:frame_i+1]*1e9, z[:frame_i+1])
            controls["r_magnitude"].set_data(gate_param["t_points"][:frame_i+1]*1e9, r_mag[:frame_i+1])

            controls["time_coordinate_cursor"].set_xdata([t_current, t_current])

            controls["coordinate_cursor"].set_data(
                [t_current, t_current, t_current, t_current],
                [x[frame_i], y[frame_i], z[frame_i], r_mag[frame_i]]
            )

            # Updating qubit state probabilities over time frames
            controls["prob_0"].set_data(gate_param["t_points"][:frame_i+1]*1e9, p0[:frame_i+1])
            controls["prob_1"].set_data(gate_param["t_points"][:frame_i+1]*1e9, p1[:frame_i+1])
            controls["prob_total"].set_data(gate_param["t_points"][:frame_i+1]*1e9, p_total[:frame_i+1])

            controls["time_probability_cursor"].set_xdata([t_current, t_current])

            controls["probability_cursor"].set_data(
                [t_current, t_current, t_current],
                [p0[frame_i], p1[frame_i], p_total[frame_i]]
            )

        fig_time.canvas.draw_idle()

        if frame >= n_frames - 1:
            anim.event_source.stop()
            button.label.set_text("Play")
            paused = True
            animation_finished = True

    anim = FuncAnimation(
        fig_bloch,
        update_animation,
        frames=n_frames,
        interval=interval,
        blit=False,
        repeat=False
    )

# ----------------------------- Pause/Play button -----------------------------
paused = True
animation_finished = False
anim.event_source.stop()

def toggle_animation(event):
    global paused, animation_finished

    if paused:
        if animation_finished:
            anim.frame_seq = anim.new_frame_seq()
            animation_finished = False

        anim.event_source.start()
        button.label.set_text("Pause")
        paused = False
    else:
        anim.event_source.stop()
        button.label.set_text("Play")
        paused = True

button_ax = fig_bloch.add_axes([0.45, 0.03, 0.1, 0.05])
button = Button(button_ax, "Play")

button.on_clicked(toggle_animation)


# Stop the animation immediately after the first draw event
def pause_animation_on_first_draw(event):
    global paused, animation_finished

    if event.canvas == fig_bloch.canvas:
        anim.event_source.stop()
        button.label.set_text("Play")
        paused = True
        animation_finished = False
        fig_bloch.canvas.mpl_disconnect(first_draw_cid)

first_draw_cid = fig_bloch.canvas.mpl_connect(
    "draw_event",
    pause_animation_on_first_draw
)

plt.show()