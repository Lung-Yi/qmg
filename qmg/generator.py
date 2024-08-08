from qiskit import execute
from qiskit_aer import Aer
import numpy as np
from typing import List, Union

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .utils import MoleculeQuantumStateGenerator, CircuitBuilder

class MoleculeGenerator():
    def __init__(self, num_heavy_atom:int, all_weight_vector:Union[List[float], np.ndarray]=None,
                 temperature:float=0.2, remove_bond_disconnection:bool=True, chemistry_constraint:bool=True):
        self.num_heavy_atom = num_heavy_atom
        self.random_weights = not all_weight_vector
        self.all_weight_vector = all_weight_vector
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint = chemistry_constraint
        self.temperature = temperature
        self.num_qubits = num_heavy_atom*(num_heavy_atom+1)
        self.num_ancilla_qubits = num_heavy_atom - 1
        self.data_generator = MoleculeQuantumStateGenerator(heavy_atom_size=num_heavy_atom, ncpus=1, sanitize_method="strict")
        
    def generate_quantum_circuit(self, random_seed):
        self.qc = CircuitBuilder(self.num_heavy_atom, self.temperature, 
                                 self.remove_bond_disconnection, self.chemistry_constraint).generate_quantum_circuit(self.all_weight_vector, random_seed)
        
    def update_weight_vector(self, all_weight_vector):
        self.all_weight_vector = all_weight_vector

    def sample_molecule(self, num_sample, random_seed:int=0):
        self.generate_quantum_circuit(random_seed)
        simulator = Aer.get_backend('qasm_simulator')
        results = execute(self.qc, backend=simulator, shots=num_sample).result()
        counts = results.get_counts(self.qc)

        smiles_dict = {}
        num_valid_molecule = 0
        for key, value in counts.items():
            smiles = self.data_generator.QuantumStateToSmiles(self.data_generator.post_process_quantum_state(key))
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + value
            if smiles:
                num_valid_molecule += value
        validity = num_valid_molecule / num_sample
        diversity = (len(smiles_dict.keys()) - 1) / num_sample
        return smiles_dict, validity, diversity
    
if __name__ == "__main__":
    mg = MoleculeGenerator(4)
    smiles_dict, validity, diversity = mg.sample_molecule(1000, random_seed=0)
    print(smiles_dict)
    print("Validity: {:.2f}%".format(validity*100))
    print("Diversity: {:.2f}%".format(diversity*100))