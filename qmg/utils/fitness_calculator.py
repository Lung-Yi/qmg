from rdkit import Chem
from rdkit.Chem import Descriptors

class FitnessCalculator():
    def __init__(self, task, distribution_learning=True):
        self.task = task
        self.distribution_learning = distribution_learning

    def calc_property(self, mol):
        if self.task == "qed":
            return Descriptors.qed(mol)

    def calc_score(self, smiles_dict: dict):
        total_count = 0
        property_sum = 0
        for smiles, count in smiles_dict.items():
            total_count += count
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                property_sum += self.calc_property(mol) * count
        return property_sum / total_count
    
    def generate_distribution(self, smiles_dict: dict):
        data_list = []
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                data_list += [0] * count
            else:
                property = self.calc_property(mol)
                data_list += [property] * count
        return data_list
    
    def generate_property_distribution(self, smiles_dict: dict):
        data_list = []
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                property = self.calc_property(mol)
                data_list += [property]
        return data_list

    def generate_property_dict(self, smiles_dict: dict):
        prop_dict = {}
        for smiles, count in smiles_dict.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                property = self.calc_property(mol)
                prop_dict.update({smiles: property})
        return prop_dict