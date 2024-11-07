from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator
from rdkit import RDLogger
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax import SearchSpace, ParameterType, RangeParameter
from ax.core.observation import ObservationFeatures
from ax.core.arm import Arm
import torch
import argparse
import logging

torch.set_default_dtype(torch.float64)
RDLogger.DisableLog('rdApp.*')

def setup_logger(file_name):
    logger = logging.getLogger('MoleculeGeneratorLogger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heavy_atom', type=int, default=5)
    parser.add_argument('--num_sample', type=int, default=10000)
    parser.add_argument('--smarts', type=str)
    parser.add_argument('--disable_connectivity_position', nargs='+', type=int, default=None)
    parser.add_argument('--no_chemistry_constraint', action='store_true')
    parser.add_argument('--num_iterations', type=int)
    args = parser.parse_args()

    if args.no_chemistry_constraint:
        file_name = f"results_constrained_bo/num_{args.num_heavy_atom}_atoms_sample_{args.num_sample}.log"
    else:
        file_name = f"results_constrained_bo/num_{args.num_heavy_atom}_atoms_sample_{args.num_sample}.log"
    logger = setup_logger(file_name)
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: {args.smarts}")
    logger.info(f"disable_connectivity_position: {args.disable_connectivity_position}")
    logger.info("Using cuda:", torch.cuda.is_available())

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=args.smarts, disable_connectivity_position=args.disable_connectivity_position)
    random_weight_vector = cwg.generate_conditional_random_weights(random_seed=0)

    number_flexible_parameters = len(random_weight_vector[cwg.parameters_indicator == 0.])
    logger.info("Number of flexible parameters:", number_flexible_parameters)
    random_weight_vector[cwg.parameters_indicator == 0.] = np.random.rand(len(random_weight_vector[cwg.parameters_indicator == 0.]))


    fc = FitnessCalculator(task="qed")

    ######################## Generation Strategy ###################################
    model_dict = {'MOO': Models.MOO, 'GPEI': Models.GPEI, 'SAASBO': Models.SAASBO,}
    gs = GenerationStrategy(
        steps=[
    #         only use this when there is no initial data
            GenerationStep(
            model=Models.SOBOL, 
            num_trials=5,
            max_parallelism=1,
            model_kwargs={"seed": 42}, 
            ),
            GenerationStep(
                model=model_dict['GPEI'],
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                max_parallelism=1,  # Parallelism limit for this step, often lower than for Sobol
                model_kwargs = {"torch_dtype": torch.float64, "torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                },
            ),
        ]
    )
    ax_client = AxClient(random_seed = 42, generation_strategy = gs) # set the random seed for BO for reproducibility
    ax_client.create_experiment(
        name="moo_experiment",
        parameters=[
            {
                "name": f"x{i+1}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float"
            }
            for i in range(number_flexible_parameters)
        ],
        objectives={
            "qed": ObjectiveProperties(minimize=False,),
            "uniqueness": ObjectiveProperties(minimize=False,),
        },
        overwrite_existing_experiment=True,
        is_test=True,
    )

    def evaluate(parameters):
        partial_inputs = np.array([parameters.get(f"x{i+1}") for i in range(number_flexible_parameters)])
        inputs = random_weight_vector
        inputs[cwg.parameters_indicator == 0.] = partial_inputs
        if not args.no_chemistry_constraint:
            inputs = cwg.apply_chemistry_constraint(inputs)
        mg = MoleculeGenerator(args.num_heavy_atom, all_weight_vector=inputs)
        smiles_dict, validity, diversity = mg.sample_molecule(args.num_sample)
        logger.info("Validity: {:.2f}%".format(validity * 100))
        logger.info("Diversity: {:.2f}%".format(diversity * 100))
        # In our case, standard error is 0, since we are computing a synthetic function.
        # Set standard error to None if the noise level is unknown.
        return {"qed": (fc.calc_score(smiles_dict), None), "uniqueness": (diversity, None)}

    for i in range(args.num_iterations + 5):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

    trial_df = ax_client.get_trials_data_frame()
    trial_df.to_csv(f"results_constrained_bo/num_{args.num_heavy_atom}.csv", index=False)

    
    

