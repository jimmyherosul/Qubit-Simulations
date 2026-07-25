import numpy as np
from qutip import expect, sigmax, sigmay, sigmaz

def vector_coordinates(qubit_states):

    # Obtaining the Bloch-vector coordinates of the qubit state at each time step
    x = np.array([expect(sigmax(), state).real for state in qubit_states])
    y = np.array([expect(sigmay(), state).real for state in qubit_states])
    z = np.array([expect(sigmaz(), state).real for state in qubit_states])
    r_mag = np.sqrt(x**2 + y**2 + z**2)

    return {
        "x": x,
        "y": y,
        "z": z,
        "r_mag": r_mag,
    }