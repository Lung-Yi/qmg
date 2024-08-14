from qiskit import execute
from qiskit_aer import Aer
import numpy as np
from typing import List, Union

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .utils import MoleculeQuantumStateGenerator, CircuitBuilder, DynamicCircuitBuilder

class MoleculeGenerator():
    def __init__(self, num_heavy_atom:int, all_weight_vector:Union[List[float], np.ndarray]=None,
                 temperature:float=0.2, dynamic_circuit:bool=True, remove_bond_disconnection:bool=True, chemistry_constraint:bool=True):
        self.num_heavy_atom = num_heavy_atom
        self.all_weight_vector = all_weight_vector
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.temperature = temperature
        self.dynamic_circuit = dynamic_circuit
        self.num_qubits = num_heavy_atom*(num_heavy_atom+1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.data_generator = MoleculeQuantumStateGenerator(heavy_atom_size=num_heavy_atom, ncpus=1, sanitize_method="strict")
        
    def generate_quantum_circuit(self, random_seed):
        if self.dynamic_circuit:
            self.qc = DynamicCircuitBuilder(self.num_heavy_atom, self.temperature, 
                                 self.remove_bond_disconnection, self.chemistry_constraint).generate_quantum_circuit(self.all_weight_vector, random_seed)
        else:
            self.qc = CircuitBuilder(self.num_heavy_atom, self.temperature, 
                                 self.remove_bond_disconnection, self.chemistry_constraint).generate_quantum_circuit(self.all_weight_vector, random_seed)
        
    def update_weight_vector(self, all_weight_vector):
        self.all_weight_vector = all_weight_vector

    def sample_molecule(self, num_sample, random_seed:int=0):
        self.generate_quantum_circuit(random_seed)
        backend = Aer.get_backend('aer_simulator')
        results = execute(self.qc, backend=backend, shots=num_sample).result()
        counts = results.get_counts(self.qc)

        smiles_dict = {}
        num_valid_molecule = 0
        for key, value in counts.items():
            if self.dynamic_circuit:
                key = "".join(key.split())
            smiles = self.data_generator.QuantumStateToSmiles(self.data_generator.post_process_quantum_state(key))
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + value
            if smiles:
                num_valid_molecule += value
        validity = num_valid_molecule / num_sample
        diversity = (len(smiles_dict.keys()) - 1) / num_sample
        return smiles_dict, validity, diversity
    
if __name__ == "__main__":
    mg = MoleculeGenerator(7)
    smiles_dict, validity, diversity = mg.sample_molecule(4096, random_seed=6)
    print(smiles_dict)
    print("Validity: {:.2f}%".format(validity*100))
    print("Diversity: {:.2f}%".format(diversity*100))