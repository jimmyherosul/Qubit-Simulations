import numpy as np

# ----------------------------- Unitary time-evolution operator for statically coupled two-level system -----------------------------
def static_unitary_operator(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, Omega_r, Omega_0):        
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[np.cos(Omega_r*t/2) + 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2), 
                                            -1j*(V_I_amplitude - 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2)],
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2), 
                                            np.cos(Omega_r*t/2) - 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2)]])
    return U


# ----------------------------- Unitary time-evolution operator for sinusoidally-driven two-level system in stationary/laboratory frame of reference ----------------------------- 
# Note that its corresponding time-evolution operator in the rotating frame of reference is equivalent to the statically coupled two-level system with Omega_r and Omega_0 replaced by Omega_R and Omega_eff, respectively
def driven_unitary_operator(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, V_signal_lab, V_signal_lab_90, Omega_R, Omega_eff, Omega_d):
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[(np.cos(Omega_d*t) + 1j*np.sin(Omega_d*t))*(np.cos(Omega_R*t/2) + 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2)),
                                            -1j*(V_signal_lab - 1j*V_signal_lab_90)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2)], 
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2),
                                            np.cos(Omega_R*t/2) - 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2)]])
    return U