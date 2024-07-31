import torch
import torch.nn as nn
import pennylane as qml

from pennylane import numpy as np

class PatchQuantumGenerator(nn.Module):
    def __init__(self, n_generators, valid_state_mask, q_depth, n_qubits, q_delta=1, temperature=5, data_reuploading=False):
        super().__init__()
        self.q_params = nn.ParameterList(
            [nn.Parameter(q_delta * torch.rand(q_depth * n_qubits * 3), requires_grad=True) for _ in range(n_generators)]
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
        patch_size = 2 ** n_qubits
        outputs = torch.Tensor(x.size(0), 0).to(device)
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = quantum_circuit(elem, params, self.data_reuploading).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            outputs = torch.cat((outputs, patches), 1)
        converted_x = self.noise_to_probability_linear_layer(x) * self.temperature
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
                for y in range(n_qubits):
                    qml.RZ(weights[i][3*y], wires=y)
                    qml.RX(weights[i][3*y+1], wires=y)
                    qml.RY(weights[i][3*y+2], wires=y)
                for y in range(n_qubits - 1):
                    qml.CNOT(wires=[y, y + 1])
                else:
                    qml.CNOT(wires=[y+1, 0])
            return qml.sample()

        if fixed_noise:
            noise = torch.rand(n_qubits, device=device) * np.pi / 2
            sampled_quantum_states = quantum_circuit_sample(noise, weights, self.data_reuploading)
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
        for y in range(n_qubits):
            qml.RZ(weights[i][3*y], wires=y)
            qml.RX(weights[i][3*y+1], wires=y)
            qml.RY(weights[i][3*y+2], wires=y)
        for y in range(n_qubits - 1):
            qml.CNOT(wires=[y, y + 1])
        else:
            qml.CNOT(wires=[y+1, 0])
    return qml.probs(wires=list(range(n_qubits)))
