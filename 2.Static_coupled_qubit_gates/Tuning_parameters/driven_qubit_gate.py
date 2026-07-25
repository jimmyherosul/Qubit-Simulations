import numpy as np

# ----------------------------- Tuning parameters -----------------------------
# Delta - Detuning (i.e. Unperturbed half-energy difference between the two basis states) in eV
# Delta_eff - Effective Detuning in rotating frame in eV
# V_mag - Coupling strength between the two basis states |0> and |1> in eV
# Omega_d - External Drive frequency in rad/s 
# Omega_R - Driven Rabi frequency in rad/s
# Omega_0 - Unperturbed Effective Detuning frequency (i.e. Natural transition/resonant frequency of the two-level system) in rad/s
# t_duration - Pulse duration for the gate in s
# phi - Phase of the coupling term V between the two basis states |0> and |1> in rad

# where Delta_eff (and by extension Omega_d), V_mag, t_duration, and phi are tuning parameters for implementing logic gates.

# Throughout this simulation, energy is in eV, time is in s and phase is in rad. Using the reduced Planck constant, the (angular) frequency is therefore in rad/s, unless otherwise specified.
# Note that for a frequency range of up to 10GHz (6.3x10^10 rad/s), the corresponding energy range is up to approx 41ueV, which is typical for superconducting qubits.

def tuning_parameters(gate_name, Delta_eff, V_I_mag, V_Q_mag, hbar, data_points, Delta):

    # All I/Q Coupling and Detuning signals are generated as a square-wave pulse over t_duration
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_0 = (2/hbar)*Delta  
    Omega_d = (2/hbar)*(Delta - Delta_eff)
    Omega_eff = Omega_0 - Omega_d
    Omega_R = np.sqrt(Omega_eff**2 + (2*V_mag/hbar)**2) 

    # For X gate: Delta_eff = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    if gate_name == 'X':
        t_duration = (np.pi/2)*(2/Omega_R)

    # For Y gate: Delta_eff = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    elif gate_name == 'Y':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For Z gate: Delta_eff > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    elif gate_name == 'Z':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For S gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/4
    elif gate_name == 'S':      
        t_duration = (np.pi/4)*(2/Omega_R)

    # For T gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/8
    elif gate_name == 'T':      
        t_duration = (np.pi/8)*(2/Omega_R)

    # For H gate: Delta_eff < 0, V_I_mag = Delta_eff or 0, V_Q_mag = 0 or Delta_eff, and t_duration such that (Omega_R/2)*t_duration = pi/8
    elif gate_name == 'H':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For I gate: Delta_eff, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0
    elif gate_name == 'I':   
        t_duration = 0*(2/Omega_R)

    t_points = np.linspace(0, t_duration, data_points) 
    Delta_eff_pulse = np.full_like(t_points, Delta_eff)
    V_I_pulse = np.full_like(t_points, V_I_mag)
    V_Q_pulse = np.full_like(t_points, V_Q_mag)
        
    # The laboratory-frame Coupling signal rotates at the drive frequency and are obtained from |V|e^{i*(Omega_d*t-phi)}.
    V_signal_lab = V_I_pulse*np.cos(Omega_d*t_points) + V_Q_pulse*np.sin(Omega_d*t_points)          # real component of (Complex) Coupling signal
    V_signal_lab_90 = V_Q_pulse*np.cos(Omega_d*t_points) - V_I_pulse*np.sin(Omega_d*t_points)       # imaginary component of (Complex) Coupling signal

    return {
        "Omega_R": Omega_R, 
        "Omega_0": Omega_0,
        "Omega_eff": Omega_eff,
        "Omega_d": Omega_d, 
        "t_points": t_points, 
        "Delta_eff_pulse": Delta_eff_pulse, 
        "V_I_pulse": V_I_pulse, 
        "V_Q_pulse": V_Q_pulse, 
        "V_mag": V_mag,
        "V_signal_lab": V_signal_lab, 
        "V_signal_lab_90": V_signal_lab_90
    }
