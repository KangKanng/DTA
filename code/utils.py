import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.nn import global_add_pool, global_max_pool
from torch.nn.functional import softmax
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import networkx as nx
import matplotlib.pyplot as plt
from layers import *

# Data Preprocessing
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    
    node_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        hybridization = atom.GetHybridization()
        chirality = atom.GetChiralTag()
        degree = atom.GetDegree()
        node_feature = [atomic_num, int(hybridization), int(chirality), degree]
        node_features.append(node_feature)
        G.add_node(atom.GetIdx(), features=node_feature)
    
    edge_features = []
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_dir = bond.GetBondDir()
        aromatic = bond.GetIsAromatic()
        edge_feature = [bond_type, int(bond_dir), int(aromatic)]
        edge_features.append(edge_feature)
        G.add_edge(start_idx, end_idx)
    
    node_features = torch.tensor(node_features, dtype=torch.float) # [num_nodes, num_node_features], num_node_features=4
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous() # [2, num_nodes]
    edge_attr = torch.tensor(edge_features, dtype=torch.float) # [num_edges, num_edge_features], num_edge_features=3

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(G.nodes), num_edges=len(G.edges))

# Checkpoints

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_loss_{loss:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {checkpoint_path}, Epoch: {epoch+1}, Loss: {loss:.4f}")
    return model, optimizer, epoch, loss


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, checkpoint_dir="E:/AIDD_project/checkpoints/best_model"):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            verbose (bool): Print a message when training stops early.
            checkpoint_path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, self.checkpoint_path)
            if self.verbose:
                print(f"Validation loss improved. Model saved to {self.checkpoint_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "CI": ci
    }

def log_file(epoch, train_loss, test_loss, metrics, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
                f"MSE = {metrics['MSE']:.4f}, RMSE = {metrics['RMSE']:.4f}, R² = {metrics['R²']:.4f}\n")

def save_predictions(smiles, sequences, predictions, ground_truths, output_path):
    data = {
        "SMILES": smiles,
        "Sequence": sequences,
        "Predicted Affinity": predictions,
        "Ground Truth Affinity": ground_truths
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def plot_affinity_scatter(predictions, ground_truths, output_file="affinity_scatter_plot.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(predictions, ground_truths, alpha=0.7, edgecolor='k')
    plt.plot([min(ground_truths), max(ground_truths)],
             [min(ground_truths), max(ground_truths)], 'r--', lw=2)  # Line y=x for reference
    plt.title("Affinity Predictions vs Ground Truths", fontsize=14)
    plt.xlabel("Predicted Affinity", fontsize=12)
    plt.ylabel("Ground Truth Affinity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weight=10):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, preds, targets):
        weights = torch.where(targets > 5, self.weight, 1.0)
        return torch.mean(weights * (preds - targets) ** 2)
    
class AmplifiedLoss(nn.Module):
    def __init__(self, alpha=1.5):
        super(AmplifiedLoss, self).__init__()
        self.alpha = alpha 

    def forward(self, preds, targets):
        error = (preds - targets) ** 2
        amplification = torch.exp(self.alpha * (targets / targets.max()))
        return torch.mean(amplification * error)