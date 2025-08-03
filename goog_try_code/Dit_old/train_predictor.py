import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from qiskit import QuantumCircuit
from quantum_rl.predictor import CircuitPredictor
from quantum_rl.models import QuantumRepresentationTransformer as AdvancedTransformer
from quantum_rl.constants import LATENT_DIM, NUM_LAYERS, NUM_HEADS, NUM_GATE_TYPES, MAX_QUBITS, MAX_GATES
from tqdm import tqdm
import numpy as np
import wandb
import glob
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictorDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        # only for extracting features (no model inference)
        self.feature_extractor = CircuitPredictor(None)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        circuit_info = entry.get('circuit_info', {})
        exec_res = entry.get('execution_result', {})
        # reconstruct Qiskit circuit
        n_qubits = circuit_info.get('n_qubits', 0)
        qc = QuantumCircuit(n_qubits)
        gates = circuit_info.get('gates', [])
        wires_list = circuit_info.get('wires_list', [])
        params = circuit_info.get('params', [])
        params_idx = circuit_info.get('params_idx', [])
        param_dict = {i: params[j] for j, i in enumerate(params_idx)}
        for i, gate in enumerate(gates):
            gl = gate.lower()
            ws = wires_list[i] if i < len(wires_list) else []
            pv = param_dict.get(i, None)
            try:
                if gl == 'h' and len(ws) >= 1:
                    qc.h(ws[0])
                elif gl == 'x' and len(ws) >= 1:
                    qc.x(ws[0])
                elif gl == 'y' and len(ws) >= 1:
                    qc.y(ws[0])
                elif gl == 'z' and len(ws) >= 1:
                    qc.z(ws[0])
                elif gl == 's' and len(ws) >= 1:
                    qc.s(ws[0])
                elif gl == 't' and len(ws) >= 1:
                    qc.t(ws[0])
                elif gl == 'cnot' and len(ws) >= 2:
                    qc.cx(ws[0],ws[1])
                elif gl in ['rx','ry','rz'] and pv is not None and len(ws) >= 1:
                    getattr(qc, gl)(pv, ws[0])
                elif gl in ['cz','swap'] and len(ws) >= 2:
                    getattr(qc, gl)(ws[0], ws[1])
                elif gl == 'ccx' and len(ws) >= 3:
                    qc.ccx(ws[0], ws[1], ws[2])
                elif gl == 'barrier':
                    qc.barrier()
                elif gl == 'measure':
                    qc.measure_all()
            except Exception as e:
                print(f"Error reconstructing circuit: {e}")
        # extract feature sequences using CircuitPredictor interface
        feats = self.feature_extractor.circuit_to_features(qc)
        gate_type_seq = feats['gate_types'].squeeze(0).long().to(DEVICE)      # [MAX_GATES]
        qubit_seq = feats['qubit_features'].squeeze(0).float().to(DEVICE)    # [MAX_GATES, MAX_QUBITS]
        param_seq = feats['params'].squeeze(0).float().to(DEVICE)            # [MAX_GATES, param_dim]
        gate_seq = gate_type_seq
        features = torch.cat([
            gate_type_seq.float().unsqueeze(-1),
            qubit_seq,
            param_seq
        ], dim=1).to(DEVICE)
        # labels
        fidelity = exec_res.get('robust_fidelity', 0.0)
        exp = exec_res.get('expressibility', {})
        if 'entropy_based' in exp:
            expressibility = exp['entropy_based'].get('expressibility_value', 0.0)
        elif 'classical_shadow' in exp:
            expressibility = exp['classical_shadow'].get('normalized_distance', 0.0)
        else:
            print("Error extracting expressibility")
            expressibility = 0.0
        entanglement = exec_res.get('entanglement', 0.0)
        labels = torch.FloatTensor([fidelity, expressibility, entanglement])
        return gate_seq, qubit_seq, param_seq, gate_type_seq, features, labels


def train_predictor(json_path, epochs=50, batch_size=16, lr=1e-4, resume=None):
    dataset = PredictorDataset(json_path)
    # 70/30 train-val split
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # W&B setup
    wandb.login(key="384cc7281342d4bfde8762663c05de6458c9d0c2")
    wandb.init(project="quantum_predictor", config={"epochs": epochs, "batch_size": batch_size, "lr": lr})
    # model init
    model = AdvancedTransformer(
        dim=LATENT_DIM,
        depth=NUM_LAYERS,
        heads=NUM_HEADS,
        num_gate_types=NUM_GATE_TYPES,
        max_qubits=MAX_QUBITS,
        max_gates=MAX_GATES,
        mlp_dim=LATENT_DIM * 4,
        dim_head=(LATENT_DIM // NUM_HEADS if NUM_HEADS > 0 else LATENT_DIM),
        dropout=0.1
    ).to(DEVICE)
    # resume from existing weights if provided
    if resume == 'yes':
        resume_path = "grid_ansatz/models/best_quantum_transformer.pth"
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        print(f"Resumed training from {resume_path}")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    # ensure checkpoint directory exists
    ckpt_dir = os.path.join('grid_ansatz', 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    # threshold for saving checkpoints (train loss <= threshold)
    threshold = 0.015
    train_losses_list = []
    grad_norms_list = []
    val_loss_list = []
    for epoch in range(epochs):
        total_loss = 0.0
        grad_norms = []
        for gate_seq, qubit_seq, param_seq, gate_type_seq, features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            gate_seq = gate_seq.to(DEVICE)
            qubit_seq = qubit_seq.to(DEVICE)
            param_seq = param_seq.to(DEVICE)
            gate_type_seq = gate_type_seq.to(DEVICE)
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(gate_types=gate_seq, qubit_features=qubit_seq, params=param_seq)
            loss = criterion(preds, labels)
            loss.backward()
            # gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            optimizer.step()
            total_loss += loss.item() * gate_seq.size(0)
        avg_loss = total_loss / train_size
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        train_losses_list.append(avg_loss)
        grad_norms_list.append(avg_grad)
        # save checkpoint at threshold loss
        if avg_loss <= threshold:
            ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at epoch {epoch+1} with train loss {avg_loss:.4f}: {ckpt_path}")
            wandb.save(ckpt_path)
        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for gate_seq, qubit_seq, param_seq, _, _, labels in val_loader:
                # Move inputs and labels to device
                gate_seq, qubit_seq, param_seq, labels = [x.to(DEVICE) for x in (gate_seq, qubit_seq, param_seq, labels)]
                # Forward pass
                preds = model(gate_types=gate_seq, qubit_features=qubit_seq, params=param_seq)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        # Align shapes in case of mismatch
        if val_preds.shape[0] != val_labels.shape[0]:
            min_len = min(val_preds.shape[0], val_labels.shape[0])
            val_preds = val_preds[:min_len]
            val_labels = val_labels[:min_len]
        val_mse = np.mean((val_preds - val_labels) ** 2, axis=0)
        val_loss_list.append(val_mse.mean())
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "train_grad_norm": avg_grad,
            "val_mse_fidelity": val_mse[0],
            "val_mse_expressibility": val_mse[1],
            "val_mse_entanglement": val_mse[2]
        })
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}, grad_norm: {avg_grad:.4f}, val_mse: {val_mse}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_quantum_transformer.pth')
            print("Saved best_quantum_transformer.pth")
    # post-training summary and checkpoint cleanup
    print("\nEpoch summary:")
    for i, (tr, vl, gr) in enumerate(zip(train_losses_list, val_loss_list, grad_norms_list), 1):
        print(f"Epoch {i}: Train Loss={tr:.4f}, Val Loss={vl:.4f}, Grad Norm={gr:.4f}")
    keep = int(input("Enter epoch number to keep checkpoint: "))
    for filepath in glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pth')):
        if f'ckpt_epoch_{keep}_' not in filepath:
            os.remove(filepath)
            print(f"Removed {filepath}")
    print(f"Kept checkpoint for epoch {keep}.")
    # visualize metrics after training
    epochs = list(range(1, len(train_losses_list) + 1))
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(epochs, train_losses_list, marker='o', color='blue', label='Train Loss')
    axs[0].set_title('Training Loss')
    axs[0].set_ylabel('Loss')
    axs[1].plot(epochs, grad_norms_list, marker='s', color='red', label='Grad Norm')
    axs[1].set_title('Gradient Norm')
    axs[1].set_ylabel('Norm')
    axs[2].plot(epochs, val_loss_list, marker='^', color='green', label='Validation Loss')
    axs[2].set_title('Validation Loss')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Epoch')
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Quantum Circuit Predictor')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None, help='Path to pretrained model state dict for resuming training')
    args = parser.parse_args()
    # hardcoded JSON data path (use forward slashes to avoid escape issues)
    data_path = "/mnt/c/Users/jungh/Documents/GitHub/Quantum_Stock_Prediction/grid_ansatz/grid_circuits/mega_results/batch_1_results_20250529_101750.json"
    train_predictor(data_path, args.epochs, args.batch_size, args.lr, args.resume)
