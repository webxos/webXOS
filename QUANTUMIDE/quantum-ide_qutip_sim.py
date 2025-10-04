import qutip as qt
import numpy as np

def run_qutip_sim(circuit_type: str, num_qubits: int = 2) -> dict:
    """Simulate quantum circuit with QuTiP. Returns probabilities and diagnostics."""
    try:
        if circuit_type == 'bell':
            H = qt.hadamard_transform(1)
            CNOT = qt.gate_expand_2toN(qt.cnot(), 2, [0, 1])
            state = H * CNOT * qt.tensor([qt.basis(2, 0), qt.basis(2, 0)])
            probs = np.abs(state.full().flatten())**2
            ideal_bell = np.array([0.5, 0, 0, 0.5])
            fidelity = float(qt.fidelity(state, qt.Qobj(ideal_bell.reshape(2, 2))))
            return {'probabilities': probs.tolist(), 'fidelity': fidelity, 'qubits': num_qubits}
        return {'error': f'Unsupported circuit: {circuit_type}'}
    except Exception as e:
        return {'error': f'Simulation failed: {str(e)}'}