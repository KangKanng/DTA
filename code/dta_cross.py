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
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import networkx as nx

from data import custom_collate, CustomDataset
from model import AffinityPredictionModel

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




# Load Dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    smiles_strs, graphs, sequences, affinities = [], [], [], []
    for _, row in data.iterrows():
        smiles_strs.append(row['iso_smiles'])
        graph = smiles_to_graph(row['iso_smiles'])
        if graph is None:
            continue
        graphs.append(graph)
        sequences.append(row['target_sequence'])
        affinities.append(row['affinity'])
    return smiles_strs, graphs, sequences, torch.tensor(affinities, dtype=torch.float)

# Train and Evaluate
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    with tqdm(data_loader, desc="Training", unit="batch") as pbar:
        for batch in pbar:
            batch = batch.to(device)
            graphs = batch
            # graphs = Batch.from_data_list(graphs).to(device)
            sequences = batch.sequences
            affinities = batch.affinities

            optimizer.zero_grad()
            predictions = model(sequences, graphs)
            loss = criterion(predictions, affinities)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with tqdm(data_loader, desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                sequences = batch.sequences
                affinities = batch.affinities.to(device)

                predictions = model(sequences, batch)
                loss = criterion(predictions, affinities)
                total_loss += loss.item()

                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(affinities.cpu().numpy())

                pbar.set_postfix(loss=loss.item())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    avg_loss = total_loss / len(data_loader)

    if return_predictions:
        return avg_loss, metrics, np.array(all_preds), np.array(all_targets)

    return avg_loss, metrics

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
    def __init__(self, patience=5, verbose=False, checkpoint_path="./checkpoints_cross"):
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
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': val_loss,
            # }, self.checkpoint_path)
            # if self.verbose:
            #     print(f"Validation loss improved. Model saved to {self.checkpoint_path}")
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
    def get_cindex(Y, P):
        sum = 0
        pair = 0
        
        for i in range(1, len(Y)):
            for j in range(0, i):
                if i is not j:
                    if(Y[i] > Y[j]):
                        pair +=1
                        sum +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
            
                
        if pair is not 0:
            return sum/pair
        else:
            return 0
    ci = get_cindex(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2,
        "CI": ci
    }

def log_file(epoch, train_loss, test_loss, metrics, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
                f"MSE = {metrics['MSE']:.4f}, RMSE = {metrics['RMSE']:.4f}, R² = {metrics['R²']:.4f}, CI = {metrics['CI']:.4f}\n")

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


# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "davis"
    batch_size = 128
    learning_rate = 1e-3

    if dataset == "davis":
        train_file = "../data/Davis/Davis_train.csv"
        test_file = "../data/Davis/Davis_test.csv"

    smiles_train, graphs_train, sequences_train, affinities_train = load_dataset(train_file)
    smiles_test, graphs_test, sequences_test, affinities_test = load_dataset(test_file)
    print(len(graphs_train))
    print(len(graphs_test))
    train_idx = range(len(graphs_train))
    test_idx = range(len(graphs_test))


    train_data = CustomDataset([graphs_train[i] for i in train_idx],
                            [sequences_train[i] for i in train_idx],
                            [affinities_train[i] for i in train_idx])

    test_data = CustomDataset([graphs_test[i] for i in test_idx],
                            [sequences_test[i] for i in test_idx],
                            [affinities_test[i] for i in test_idx])


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


    # Initialize model
    model = AffinityPredictionModel(protein_dim=1024, drug_dim=128, hidden_dim=64, attention_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    checkpoint_dir = "./checkpoints_cross"
    best_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # early_stopping = EarlyStopping(patience=10, verbose=True, checkpoint_path=os.path.join(checkpoint_dir, "best_model.pt"))

    start_epoch = -1

    log_file_path = "training_log.json"

    # Training loop
    epochs = 100 # 50
    for epoch in range(start_epoch + 1, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_metrics, predictions, ground_truths = evaluate_model(model, test_loader, criterion, device, return_predictions=True)

        # Save predictions
        if epoch == epochs - 1:
            save_predictions(
                [smiles_test[i] for i in range(len(smiles_test))],
                [sequences_test[test_idx[i]] for i in range(len(test_idx))],
                predictions,
                ground_truths,
                "test_results.csv"
            )


        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Metrics: MSE: {test_metrics['MSE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, R^2: {test_metrics['R^2']:.4f}")

        mse, rmse, r2, ci = test_metrics['MSE'], test_metrics['RMSE'], test_metrics['R^2'], test_metrics['CI']
        metrics = {"MSE": mse, "RMSE": rmse, "R²": r2, "CI": ci}
        log_file(epoch + 1, train_loss, test_loss, metrics, log_file_path)

        # save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_dir)

        # best chkpt
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     save_checkpoint(model, optimizer, epoch, test_loss, os.path.join(checkpoint_dir, "best_model.pt"))

        # early_stopping(test_loss, model, optimizer, epoch)
        # if early_stopping.early_stop:
        #     print("Early stopping. Exiting training loop.")
        #     break
