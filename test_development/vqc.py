import matplotlib.pyplot as plt
import pennylane as qml

from pennylane import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd
import sys
sys.path.append("../")
from qmg.utils import MoleculeQuantumStateGenerator

num_heavy_atom = 3
data_path = f"../dataset/chemical_space/effective_{num_heavy_atom}.csv"
data_generator = MoleculeQuantumStateGenerator(heavy_atom_size=num_heavy_atom, ncpus=16)
data = pd.read_csv(data_path)

n_qubits = data_generator.n_qubits # + n_a_qubits  # Total number of qubits / N
q_depth = 3  # Depth of the parameterised quantum circuit / D
n_generators = 1  # Number of subgenerators for the patch method / N_G
sample_num = 2000 # Number of sampled molecules for validating the training
# data_reuploading = False

dev = qml.device("lightning.qubit", wires=n_qubits) # Quantum simulator
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Enable CUDA device if available
device = torch.device("cpu")
print(device)

valid_state_mask = data_generator.generate_valid_mask(data)
valid_state_mask = torch.Tensor(valid_state_mask).to(device)

@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(noise, weights, data_reuploading):
    weights = weights.reshape(q_depth, n_qubits * 3)
    for i in range(q_depth):
        if data_reuploading:
            for j in range(n_qubits):
                qml.RY(noise[j], wires=j)
        else:
            if i == 0:
                for j in range(n_qubits):
                    qml.RY(noise[j], wires=j)
        # Parameterised layer
        for y in range(n_qubits):
            qml.RZ(weights[i][3*y], wires=y)
            qml.RX(weights[i][3*y+1], wires=y)
            qml.RY(weights[i][3*y+2], wires=y)
        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CNOT(wires=[y, y + 1])
        else:
            qml.CNOT(wires=[y+1, 0])
    return qml.probs(wires=list(range(n_qubits)))

def binary_tensor_to_string(tensor):
    flat_tensor = tensor.view(-1).tolist()
    binary_string = ''.join(map(str, flat_tensor))
    return binary_string

def calc_validity_and_uniqueness(smiles_list):
    valid_smiles_list = [i for i in smiles_list if i is not None]
    return len(valid_smiles_list) / len(smiles_list), len(set(valid_smiles_list)) / len(smiles_list)

# def partial_measure(noise, weights):
#     # Non-linear Transform
#     probs = quantum_circuit(noise, weights)
#     probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
#     probsgiven0 /= torch.sum(probsgiven0)

#     # # Post-Processing
#     # probsgiven = probsgiven0 / torch.max(probsgiven0)
#     return probsgiven0

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""
    def __init__(self, n_generators, valid_state_mask, q_delta=1, temperature=5, data_reuploading=False):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
            temperature (float, optional): parameter adjusting the output probability distribution.
            data_reuploading (bool): whether to use the data reuploading technique.
        """
        super().__init__()
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits * 3), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
        self.valid_state_mask = valid_state_mask
        self.noise_to_probability_linear_layer = nn.Linear(n_qubits, 2**n_qubits)
        self.softmax_layer = nn.Softmax(dim=1)
        self.temperature = temperature
        self.data_reuploading = data_reuploading
        for param in self.noise_to_probability_linear_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** n_qubits # 2 ** (n_qubits - n_a_qubits)
        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        outputs = torch.Tensor(x.size(0), 0).to(device)
        # Iterate over all sub-generators
        for params in self.q_params:
            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = quantum_circuit(elem, params, self.data_reuploading).float().unsqueeze(0) # partial_measure
                patches = torch.cat((patches, q_out))
            # Each batch of patches is concatenated with each other to create a batch of images
            outputs = torch.cat((outputs, patches), 1)

        # converted input noise
        converted_x = self.noise_to_probability_linear_layer(x) * self.temperature# - 10 * (1 - self.valid_state_mask) 
        # converted_x = self.softmax_layer(converted_x)
        return outputs, converted_x
    
    def random_sample(self, sample_num, fixed_noise=False):
        weights = torch.tensor([])
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in ["noise_to_probability_linear_layer.weight", "noise_to_probability_linear_layer.bias"]:
                continue
            weights = torch.cat([weights, param.data])
        if fixed_noise:
            dev_sample = qml.device("default.qubit", wires=n_qubits, shots=sample_num)
        else:
            dev_sample = qml.device("default.qubit", wires=n_qubits, shots=1)
        @qml.qnode(dev_sample)
        def quantum_circuit_sample(noise, weights, data_reuploading):
            weights = weights.reshape(q_depth, n_qubits * 3)
            for i in range(q_depth):
                if data_reuploading:
                    for j in range(n_qubits):
                        qml.RY(noise[j], wires=j)
                else:
                    if i == 0:
                        for j in range(n_qubits):
                            qml.RY(noise[j], wires=j)
                # Parameterised layer
                for y in range(n_qubits):
                    qml.RZ(weights[i][3*y], wires=y)
                    qml.RX(weights[i][3*y+1], wires=y)
                    qml.RY(weights[i][3*y+2], wires=y)
                # Control Z gates
                for y in range(n_qubits - 1):
                    qml.CNOT(wires=[y, y + 1])
                else:
                    qml.CNOT(wires=[y+1, 0])
            return qml.sample()
        
        
        if fixed_noise:
            noise = torch.rand(n_qubits, device=device) * np.pi / 2
            sampled_quantum_states = quantum_circuit_sample(noise, weights, self.data_reuploading) # 2-dimensional torch.tensor
        else:
            sampled_quantum_states = []
            for i in range(sample_num):
                noise = torch.rand(n_qubits, device=device) * np.pi / 2
                sampled_quantum_states.append(quantum_circuit_sample(noise, weights, self.data_reuploading))

        sampled_quantum_states = [binary_tensor_to_string(qs) for qs in sampled_quantum_states]
        smiles_list = []
        for q in sampled_quantum_states:
            smiles_list.append(data_generator.QuantumStateToSmiles(q))
        return smiles_list
    
class valid_state_loss(nn.Module):
    def __init__(self, valid_state_mask: torch.tensor, reduction="mean"):
        """
        Parameters
        ----------
        valid_state_mask :  torch.tensor
            binart tensor, 1 indicates valid quantum state, and 0 indicates invalid.
        """
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.reduction = reduction

    def forward(self, predictions):
        loss = (predictions * self.valid_state_mask).sum(dim=1)
        if self.reduction == "mean":
            return torch.mean(-torch.log(loss))
        elif self.reduction == "sum":
            return torch.sum(-torch.log(loss))
        else:
            return -torch.log(loss)

class jenson_shannon_divergence(nn.Module):
    def __init__(self, valid_state_mask, reduction="batchmean"):
        """
        Parameters
        ----------
        valid_state_mask :  torch.tensor
            binart tensor, 1 indicates valid quantum state, and 0 indicates invalid.
        """
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, outputs, coverted_noise):
        outputs = outputs[:, self.valid_state_mask == 1.]
        coverted_noise = coverted_noise[:, self.valid_state_mask == 1.]
        converted_noise_probability = self.softmax_layer(coverted_noise)
        total_m = 0.5 * (outputs + converted_noise_probability)
        loss = 0.0
        loss += self.kl_div(outputs.log(), total_m) 
        loss += self.kl_div(converted_noise_probability.log(), total_m) 
        return (0.5 * loss)
    
class diversity_loss(nn.Module):
    def __init__(self, valid_state_mask, reduction="batchmean"):
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)

    def jensen_shannon_divergence(self, ps, qs):
        m = 0.5 * (ps + qs)
        return 0.5 * (self.kl_div(ps.log(), m) + self.kl_div(qs.log(), m))
    
    def forward(self, distributions):
        distributions = distributions[:, self.valid_state_mask == 1.]
        reversed_distributions = torch.flip(distributions, dims=[0])
        return - self.jensen_shannon_divergence(distributions, reversed_distributions)
    
generator = PatchQuantumGenerator(n_generators, valid_state_mask, data_reuploading=False).to(device)
print(generator)

import os

save_dir = "models_3_data_reuploading"

os.makedirs(f"{save_dir}/heavy_atoms_{num_heavy_atom}", exist_ok=True)
criterion_js = jenson_shannon_divergence(valid_state_mask=valid_state_mask, reduction="batchmean")
criterion_valid = valid_state_loss(valid_state_mask)
criterion_diversity = diversity_loss(valid_state_mask=valid_state_mask, reduction="batchmean")
best_loss = 1e10

batch_size = 32
opt = optim.Adam(generator.parameters(), lr=0.1)
loss_js_history = []
loss_valid_state_history = []
loss_valid_history = []
valid_per_steps = 10

steps = 100
for i in range(steps):
    # Noise follwing a uniform distribution in range [0,pi/2)
    noise = torch.rand(batch_size, n_qubits, device=device) * np.pi / 2
    outputs, converted_noise = generator(noise)
    loss_diversity = criterion_diversity(outputs)

    loss_valid_state = criterion_valid(outputs)
    loss_valid_state_history.append(loss_valid_state.detach().cpu())
    loss = loss_valid_state + loss_diversity
        
    opt.zero_grad()
    loss.backward()
    opt.step()
    if float(loss.detach().cpu()) < best_loss:
        best_loss = float(loss.detach().cpu())
        torch.save(generator.state_dict(), f'{save_dir}/heavy_atoms_{num_heavy_atom}/best_generator.pt')

    print(f"Step {i+1}", loss_diversity, loss_valid_state, loss)
    if (i+1) % valid_per_steps == 0:
        sample_smiles_list = generator.random_sample(sample_num=sample_num)
        validity, uniqueness = calc_validity_and_uniqueness(sample_smiles_list)
        print(f"Step {i+1}, sampling {sample_num} molecules, validity: {validity*100}%, uniqueness: {uniqueness*100}%.")
        torch.save(generator.state_dict(), f'{save_dir}/heavy_atoms_{num_heavy_atom}/generator_{i+1}_steps.pt')
