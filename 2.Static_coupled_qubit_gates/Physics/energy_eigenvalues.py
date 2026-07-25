import numpy as np

def energy_eigenvalues(V_mag, Delta_range, Delta, E_mean):
    E_plus = E_mean + np.sqrt(Delta_range**2 + V_mag**2)
    E_minus = E_mean - np.sqrt(Delta_range**2 + V_mag**2)
    E_plus_current = E_mean + np.sqrt(Delta**2 + V_mag**2)
    E_minus_current = E_mean - np.sqrt(Delta**2 + V_mag**2)

    return {
        "E_plus": E_plus,
        "E_minus": E_minus,
        "E_plus_current": E_plus_current,
        "E_minus_current": E_minus_current
    }
