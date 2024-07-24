from rdkit import Chem
import numpy as np
import itertools
import multiprocessing
import pandas as pd

def subfunction_generate_state(cls, node_vector, adjacency_matrix, new_index):
    """Wrapper function for the use of multiprocessing. """
    new_node_vector, new_adjacency_matrix = cls.permutate_connectivity(node_vector, adjacency_matrix, new_index)
    new_quantum_state = cls.ConnectivityToQuantumState(new_node_vector, new_adjacency_matrix)
    return new_quantum_state

class MoleculeQuantumStateGenerator():
    def __init__(self, heavy_atom_size=5, ncpus=4):
        self.size = heavy_atom_size
        self.effective_numbers = list(range(heavy_atom_size))
        self.ncpus = ncpus
        self.atom_type_to_idx = {"C": 1, "O":2, "N": 3} # only supports C, O, N atoms now.
        self.bond_type_to_idx = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3}
        self.idx_to_atom_type = {1: "C", 2: "O", 3: "N"}
        self.idx_to_bond_type = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
        self.qubits_per_type_atom = int(np.ceil(np.log2(len(self.atom_type_to_idx) + 1))) # How many qubits required for describing the quantum state of atom type
        self.qubits_per_type_bond = int(np.ceil(np.log2(len(self.bond_type_to_idx) + 1))) # How many qubits required for describing the quantum state of bond type
        self.n_qubits = int(self.size * self.qubits_per_type_atom + self.size * (self.size-1) / 2 * self.qubits_per_type_bond)

    def decimal_to_binary(self, x, padding_length=2):
        """
        Parameters:
        x (int): The decimal value.

        Returns:
        str: A binary bit string.
        """
        bit = "0"*(padding_length-1) + bin(x)[2:]
        return bit[-padding_length:] # -2 means we only take 4 possible states

    def SmilesToConnectivity(self, smiles):
        """
        Generate a molecular graph from a SMILES string.

        Parameters:
        smiles (str): The SMILES string representing the molecule.

        Returns:
        tuple: A tuple containing the node vector (np.ndarray) and the adjacency matrix (np.ndarray).
        """
        node_vector = np.zeros(self.size)
        adjacency_matrix = np.zeros((self.size, self.size))
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_type = atom.GetSymbol()
            node_vector[idx] = self.atom_type_to_idx[atom_type]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjacency_matrix[i][j] = self.bond_type_to_idx[bond.GetBondType()]
            adjacency_matrix[j][i] = self.bond_type_to_idx[bond.GetBondType()]
        return node_vector, adjacency_matrix

    def ConnectivityToSmiles(self, node_vector, adjacency_matrix):
        """
        Generate a SMILES string from the molecular graph.

        Returns:
        str: The SMILES string representing the molecule.
        """
        mol = Chem.RWMol()
        mapping_num_2_molIdx = {}
        for i, atom_type_idx in enumerate(node_vector):
            if atom_type_idx == 0:
                continue
            a = Chem.Atom(self.idx_to_atom_type[atom_type_idx])
            a.SetAtomMapNum(i)
            molIdx = mol.AddAtom(a)
            mapping_num_2_molIdx.update({i: molIdx})
        # add bonds between adjacent atoms, only traverse half the matrix
        for ix, row in enumerate(adjacency_matrix):
            for iy_, bond_type_idx in enumerate(row[ix+1:]):
                iy = ix + iy_ + 1
                if bond_type_idx == 0:
                    continue
                else:
                    bond_type = self.idx_to_bond_type[bond_type_idx]
                    try:
                        mol.AddBond(mapping_num_2_molIdx[ix], mapping_num_2_molIdx[iy], bond_type)
                    except:
                        return None
        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    def ConnectivityToQuantumState(self, node_vector, adjacency_matrix):
        """
        Generate the quantum state (bit vector) based on the molecular connectivity.
        The preceding bits represent the atom type, and the subsequent bits represent the connectivity.
        
        Returns:
        np.ndarray: computational quantum state.
        """
        quantum_state = ""
        for atom_idx in node_vector:
            quantum_state += self.decimal_to_binary(int(atom_idx), padding_length=self.qubits_per_type_atom)
        for ix, row in enumerate(adjacency_matrix):
            for bond_type_idx in row[ix+1:]:
                quantum_state += self.decimal_to_binary(int(bond_type_idx), padding_length=self.qubits_per_type_bond)
        return quantum_state

    def QuantumStateToConnectivity(self, quantum_state):
        node_state = quantum_state[:2*self.size]
        bond_state = quantum_state[2*self.size:]
        node_vector = np.zeros(self.size)
        adjacency_matrix = np.zeros((self.size, self.size))
        for i in range(0, len(node_state), 2):
            node_vector[i//2] = int(node_state[i:i+2], 2)
        row = 0
        for i in range(0, len(bond_state), 2):
            idx = i // 2
            if idx == (2*(self.size-1) - (row+1) + 1) * (row+1) / 2:
                row += 1
            column = int(idx - (2*(self.size-1) - row + 1) * row / 2) + row + 1
            bond_type_idx = int(bond_state[i:i+2], 2)
            adjacency_matrix[row][column] = bond_type_idx
            adjacency_matrix[column][row] = bond_type_idx
        return node_vector, adjacency_matrix
    
    def QuantumStateToSmiles(self, quantum_state):
        return self.ConnectivityToSmiles(*self.QuantumStateToConnectivity(quantum_state))
    
    def QuantumStateToStateVector(self, quantum_state):
        stat_vector = np.zeros(2**self.n_qubits)
        decimal = int(quantum_state, 2)
        stat_vector[-1-decimal] = 1
        return stat_vector
    
    def QuantumStateToDecimal(self, quantum_state):
        decimal = int(quantum_state, 2)
        return decimal
    
    def generate_permutations(self, k):
        """
        Generate all possible permutations of k elements from the given list of elements.

        :param k: Number of elements to choose for permutations
        :return: List of permutations
        """
        return list(itertools.permutations(self.effective_numbers, k))
    
    def enumerate_all_quantum_states(self, smiles):
        """
        Generate all possible quantum states representing the given molecule SMILES.

        :return: List of quantum states (str)
        """
        node_vector, adjacency_matrix = self.SmilesToConnectivity(smiles)
        all_permutation_index = self.generate_permutations(np.count_nonzero(node_vector))
        args = [(self, node_vector, adjacency_matrix, new_index) for new_index in all_permutation_index]
        with multiprocessing.Pool(processes=self.ncpus) as pool:
            all_quantum_states = pool.starmap(subfunction_generate_state, args)

        return list(set(all_quantum_states))

    def permutate_connectivity(self, node_vector, adjacency_matrix, new_index):
        mapping_dict = {old: new for old, new in enumerate(new_index)}
        new_node_vector = np.zeros(self.size)
        new_adjacency_matrix = np.zeros((self.size, self.size))
        # atom
        for old, new in mapping_dict.items():
            new_node_vector[new] = node_vector[old]
        # bond
        for ix, row in enumerate(adjacency_matrix):
            for iy_, bond_type_idx in enumerate(row[ix+1:]):
                if not bond_type_idx:
                    continue
                iy = ix + iy_ + 1
                ix_new = mapping_dict[ix]
                iy_new = mapping_dict[iy]
                new_adjacency_matrix[ix_new][iy_new] = bond_type_idx
                new_adjacency_matrix[iy_new][ix_new] = bond_type_idx
        return new_node_vector, new_adjacency_matrix

    def generate_valid_mask(self, data: pd.DataFrame):
        """
        :return: binary valid quantum states mask (np.ndarray)
        """
        valid_state_vector_mask = np.zeros(2**self.n_qubits)
        for decimal_index in set(data["decimal_index"]):
            valid_state_vector_mask[int(decimal_index)] = 1
        return valid_state_vector_mask

if __name__ == "__main__":
    smiles = "CCOC=NC(O)C"
    QG = MoleculeQuantumStateGenerator(9)
    node_vector, adjacency_matrix = QG.SmilesToConnectivity(smiles)
    print("node_vector", node_vector)
    print("adjacency_matrix")
    print(adjacency_matrix)
    print(QG.ConnectivityToSmiles(node_vector, adjacency_matrix))
    quantum_state = QG.ConnectivityToQuantumState(node_vector, adjacency_matrix)
    print(quantum_state)
    node_vector, adjacency_matrix = QG.QuantumStateToConnectivity(quantum_state)
    print("node_vector", node_vector)
    print("adjacency_matrix")
    print(adjacency_matrix)
    print(QG.ConnectivityToSmiles(node_vector, adjacency_matrix))
    all_quantum_states = QG.enumerate_all_quantum_states(smiles)
    print(all_quantum_states[:100])