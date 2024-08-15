# QMG: Quantum computing-based molecule generation
This repository demonstrates the useage of quantum circuits for the generation of small molecules.

## OS Requirements
This repository requires to operate on **Linux** operating system.

## Python Dependencies
* Python (version >= 3.7)
* qiskit (version <= 0.46.2)
* qiskit-aer (version <= 0.14.2)
* rdkit (version >= 2024.3.3)
* matplotlib (version >=3.3.4)
* numpy (version >= 1.16.4)
* pandas (version >= 2.0.3)

## Example script for unconditional generation of small molecules (with number of heavy atoms <=5).

```python
from qmg.generator import MoleculeGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# parameter settings
num_heavy_atom = 5
num_sample = 20000
random_seed = 0

mg = MoleculeGenerator(num_heavy_atom) 
smiles_dict, validity, diversity = mg.sample_molecule(num_sample, ransom_seed)
print(smiles_dict)
print("Validity: {:.2f}%".format(validity*100))
print("Diversity: {:.2f}%".format(diversity*100))

# Example outputs:
# {'NOn1[nH]o1': 1, None: 4401, 'OC1NNO1': 18, 'C1CO1': 1, 'NCCNO': 1, 'ON1ONO1': 2, 'C#[N+][O-]': 1, 'ONN1CO1': 4, ...
# Validity: 78.00%
# Diversity: 3.17%
```


## Example script for conditional generation of small molecules with epoxide substructure (with number of heavy atoms <=7).

```python
from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# parameter settings
num_heavy_atom = 7
random_seed = 3
smarts = "[O:1]1[C:2][C:3]1"
disable_connectivity_position = [1]

cwg = ConditionalWeightsGenerator(num_heavy_atom, smarts=smarts, disable_connectivity_position=disable_connectivity_position)
random_weight_vector = cwg.generate_conditional_random_weights(random_seed)
mg = MoleculeGenerator(num_heavy_atom, all_weight_vector=random_weight_vector) 
smiles_dict, validity, diversity = mg.sample_molecule(10000)
print(smiles_dict)
print("Validity: {:.2f}%".format(validity*100))
print("Diversity: {:.2f}%".format(diversity*100))

# Example outputs:
# {'CN(N)C1OC1N': 1, 'NC12OC1n1on12': 1, 'NNNC12OC1O2': 1, 'CNOCC1CO1': 1, 'NN(O)C1(N)CO1': 16, 'NNCCC1CO1': 1, 'NN1OC2OC21': 1, 'OC1=NC2(CO2)N1': 1,...}
# Validity: 62.00%
# Diversity: 4.41%
```