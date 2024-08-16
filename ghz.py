from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

def create_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit

simulator = Aer.get_backend('aer_simulator_statevector')
circuit = create_ghz_circuit(n_qubits=20)
circuit.measure_all()
circuit = transpile(circuit, simulator)
job = simulator.run(circuit)
result = job.result()

print(result.get_counts())
print(f'backend: {result.backend_name}')
