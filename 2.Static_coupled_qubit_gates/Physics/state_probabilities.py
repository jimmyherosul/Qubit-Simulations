import numpy as np
from qutip import ket2dm

def state_probabilities(qubit_states):
    
    # Computing the probabilities of measuring the qubit state in states |0> and |1> at each time step
    p0 = np.array([ket2dm(state)[0, 0].real for state in qubit_states])
    p1 = np.array([ket2dm(state)[1, 1].real for state in qubit_states])
    p_total = p0 + p1

    return {
        "p0": p0,
        "p1": p1,
        "p_total": p_total
    }