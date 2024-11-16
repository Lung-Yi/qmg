from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
import numpy as np

class FitnessCalculator():
    def __init__(self, task, distribution_learning=True):
        self.task = task
        self.distribution_learning = distribution_learning

    def calc_property(self, mol):
        if self.task == "qed":
            return Descriptors.qed(mol)
        elif self.task == "logP":
            return Descriptors.MolLogP(mol)
        elif self.task == "tpsa":
            return Descriptors.TPSA(mol)
        elif self.task in ["sascore", "SAscore"]:
            return sascorer.calculateScore(mol)

    def calc_score(self, smiles_dict: dict, condition_score=None):
        if self.task == "validity":
            total_samples = sum(smiles_dict.values())
            return (total_samples - smiles_dict.get(None, 0) - smiles_dict.get("None", 0)) / total_samples
        elif self.task == "uniqueness":
            smiles_dict_copy = smiles_dict.copy()
            smiles_dict_copy.pop("None", None)
            smiles_dict_copy.pop(None, None)
            total_valid_samples = sum(smiles_dict_copy.values())
            total_unique_smiles = len(smiles_dict_copy.keys())
            return total_unique_smiles / total_valid_samples
        
        total_count = 0
        property_sum = 0
        for smiles, count in smiles_dict.items():
            total_count += count
            mol = Chem.MolFromSmiles(str(smiles))
            if mol == None:
                continue
            else:
                if condition_score:
                    property_sum += np.abs(self.calc_property(mol) - condition_score) * count
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

class FitnessCalculatorWrapper():
    def __init__(self, task, condition:None):
        self.task_list = task
        self.condition_list = [float(x) if x not in [None, "None"] else None for x in condition]
        self.function_dict = {task: FitnessCalculator(task) for task in self.task_list}
        self.task_condition = {task: condition for task, condition in zip(self.task_list, self.condition_list)}
    
    def evaluate(self, smiles_dict):
        score_dict = {task: (self.function_dict[task].calc_score(smiles_dict, self.task_condition[task]), None) 
                      for task in self.task_list}
        return score_dict