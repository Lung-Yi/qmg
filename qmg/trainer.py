import torch
import torch.optim as optim

from .data_utils import MoleculeDataLoader
from .models import PatchQuantumGenerator, valid_state_loss, jenson_shannon_divergence, diversity_loss
from .utils import calc_validity_and_uniqueness

def train(generator, data_loader, criterion_valid, criterion_js, criterion_diversity, optimizer, save_dir, steps, valid_per_steps, sample_num, device):
    best_loss = 1e10
    loss_valid_state_history = []

    for i in range(steps):
        noise = torch.rand(batch_size, n_qubits, device=device) * np.pi / 2
        outputs, converted_noise = generator(noise)
        loss_diversity = criterion_diversity(outputs)
        loss_valid_state = criterion_valid(outputs)
        loss_valid_state_history.append(loss_valid_state.detach().cpu())
        loss = loss_valid_state + loss_diversity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if float(loss.detach().cpu()) < best_loss:
            best_loss = float(loss.detach().cpu())
            torch.save(generator.state_dict(), f'{save_dir}/heavy_atoms_{num_heavy_atom}/best_generator.pt')

        print(f"Step {i+1}", loss_diversity, loss_valid_state, loss)
        if (i+1) % valid_per_steps == 0:
            sample_smiles_list = generator.random_sample(sample_num=sample_num)
            validity, uniqueness = calc_validity_and_uniqueness(sample_smiles_list)
            print(f"Step {i+1}, sampling {sample_num} molecules, validity: {validity*100}%, uniqueness: {uniqueness*100}%.")
            torch.save(generator.state_dict(), f'{save_dir}/heavy_atoms_{num_heavy_atom}/generator_{i+1}_steps.pt')

if __name__ == "__main__":
    from qmg.utils import MoleculeQuantumStateGenerator
    import os

    num_heavy_atom = 3
    data_path = f"../dataset/chemical_space/effective_{num_heavy_atom}.csv"
    data_generator = MoleculeQuantumStateGenerator(heavy_atom_size=num_heavy_atom, ncpus=16)
    device = torch.device("cpu")
    data_loader = MoleculeDataLoader(data_path, data_generator, device)
    valid_state_mask = data_loader.load_data()

    n_qubits = data_generator.n_qubits
    q_depth = 3
    n_generators = 1
    sample_num = 2000

    generator = PatchQuantumGenerator(n_generators, valid_state_mask, q_depth, n_qubits, data_reuploading=False).to(device)
    print(generator)

    save_dir = "models_3_data_reuploading"
    os.makedirs(f"{save_dir}/heavy_atoms_{num_heavy_atom}", exist_ok=True)

    criterion_js = jenson_shannon_divergence(valid_state_mask=valid_state_mask, reduction="batchmean")
    criterion_valid = valid_state_loss(valid_state_mask)
    criterion_diversity = diversity_loss(valid_state_mask=valid_state_mask, reduction="batchmean")

    batch_size = 32
    optimizer = optim.Adam(generator.parameters(), lr=0.1)
    valid_per_steps = 10
    steps = 100

    train(generator, data_loader, criterion_valid, criterion_js, criterion_diversity, optimizer, save_dir, steps, valid_per_steps, sample_num, device)
