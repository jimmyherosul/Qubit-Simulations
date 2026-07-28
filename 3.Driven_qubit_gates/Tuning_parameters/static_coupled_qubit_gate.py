import numpy as np


# ----------------------------- Tuning parameters -----------------------------
# Delta - Detuning (i.e. Unperturbed half-energy difference between the two basis states) in eV
# V_mag - Coupling strength between the two basis states |0> and |1> in eV
# Omega_r - Rabi frequency in rad/s
# Omega_0 - Unperturbed Detuning frequency (i.e. Natural transition/resonant frequency of the two-level system) in rad/s
# t_duration - Pulse duration for the gate in s
# phi - Phase of the coupling term V between the two basis states |0> and |1> in rad

# where Delta, V_mag, t_duration, and phi are tuning parameters for implementing logic gates.

# Throughout this simulation, energy is in eV, time is in s and phase is in rad. Using the reduced Planck constant, the (angular) frequency is therefore in rad/s, unless otherwise specified.
# Note that for a frequency range of up to 10GHz (6.3x10^10 rad/s), the corresponding energy range is up to approx 41ueV, which is typical for superconducting qubits.

def tuning_parameters(gate_name, Delta, V_I_mag, V_Q_mag, hbar, data_points):
    
    # All I/Q Coupling and Detuning pulses are generated as a square-wave pulse over t_duration
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_0 = (2/hbar)*Delta 
    Omega_r = np.sqrt(Omega_0**2 + (2*V_mag/hbar)**2)
    
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

    t_points = np.linspace(0, t_duration, data_points) 
    Delta_pulse = np.full_like(t_points, Delta)
    V_I_pulse = np.full_like(t_points, V_I_mag)
    V_Q_pulse = np.full_like(t_points, V_Q_mag)
    
    return {
        "Omega_r": Omega_r, 
        "Omega_0": Omega_0, 
        "t_points": t_points, 
        "Delta_pulse": Delta_pulse, 
        "V_I_pulse": V_I_pulse, 
        "V_Q_pulse": V_Q_pulse,
        "V_mag": V_mag
    }
