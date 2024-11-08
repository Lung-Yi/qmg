from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator, FitnessCalculator
from rdkit import RDLogger
import numpy as np

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str,)
    parser.add_argument('--num_heavy_atom', type=int, default=5)
    parser.add_argument('--num_sample', type=int, default=10000)
    parser.add_argument('--smarts', type=str)
    parser.add_argument('--disable_connectivity_position', nargs='+', type=int, default=None)
    parser.add_argument('--no_chemistry_constraint', action='store_true')
    parser.add_argument('--num_iterations', type=int)
    args = parser.parse_args()

    if args.no_chemistry_constraint:
        data_dir = "results_unconstrained_bo"
    else:
        data_dir = "results_chemistry_constraint_bo"
    file_name = f"{data_dir}/{args.task_name}.log"

    logger = setup_logger(file_name)
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: {args.smarts}")
    logger.info(f"disable_connectivity_position: {args.disable_connectivity_position}")
    logger.info(f"Using cuda: {torch.cuda.is_available()}")

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=args.smarts, disable_connectivity_position=args.disable_connectivity_position)
    random_weight_vector = cwg.generate_conditional_random_weights(random_seed=0)

    number_flexible_parameters = len(random_weight_vector[cwg.parameters_indicator == 0.])
    logger.info(f"Number of flexible parameters: {number_flexible_parameters}")
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
        qed_score = fc.calc_score(smiles_dict)
        logger.info("qed: {:.3f}".format(qed_score))
        logger.info("Validity: {:.2f}%".format(validity * 100))
        logger.info("Diversity: {:.2f}%".format(diversity * 100))
        # Set standard error to None if the noise level is unknown.
        return {"qed": (qed_score, None), "uniqueness": (diversity, None)}

    for i in range(args.num_iterations + 5):
        logger.info(f"Iteration number: {i}")
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        trial_df = ax_client.get_trials_data_frame()
        trial_df.to_csv(f"{data_dir}/{args.task_name}.csv", index=False)

    
    

