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
from utils import *

class AffinityPredictionModel(nn.Module):
    def __init__(self, protein_dim, drug_dim, hidden_dim, attention_dim, capsule_dim):
        super(AffinityPredictionModel, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.drug_encoder = GNNEncoder(input_dim=4, hidden_dim=hidden_dim, output_dim=drug_dim, edge_dim=3)
        
        # Cross Layer
        cross_dim = max(protein_dim, drug_dim) * 3
        self.cross_layer = CrossLayer(input_dim=cross_dim, hidden_dim=hidden_dim)
        
        # Attention Layer
        self.attention_layer = CrossAttentionLayer(protein_dim, drug_dim, attention_dim, attention_dim) # cross-attn
        self.self_attention = SelfAttentionLayer(input_dim=drug_dim, attention_dim=attention_dim) # self-attn
        
        # Capsule Layer
        self.protein_capsule = CapsuleLayer(input_dim=protein_dim, output_dim=capsule_dim, num_capsules=8)
        self.drug_capsule = CapsuleLayer(input_dim=drug_dim, output_dim=capsule_dim, num_capsules=8)
        
        output_dim = protein_dim + drug_dim + 1 + attention_dim + attention_dim * 2
        self.fc1 = nn.Linear(output_dim, output_dim // 2) 
        self.fc2 = nn.Linear(output_dim // 2, output_dim // 4)
        self.fc3 = nn.Linear(output_dim // 4, 1)
        self.dropout = nn.Dropout(p=0.2)
        # self.fc = nn.Linear(output_dim, 1)

    def forward(self, protein_seq, drug_graph):
        protein_embedding = self.protein_encoder(protein_seq)
        
        batch = drug_graph.batch.to(protein_embedding.device)
        x = drug_graph.x
        edge_index = drug_graph.edge_index
        edge_attr = drug_graph.edge_attr
               
        drug_embedding = self.drug_encoder(x, edge_index, edge_attr, drug_graph.batch)
        
        # print(f"Protein embedding size: {protein_embedding.size()}")
        # print(f"Drug embedding size: {drug_embedding.size()}")
               
        assert protein_embedding.size(0) == drug_embedding.size(0), \
            f"Batch size mismatch: {protein_embedding.size(0)} (protein) vs {drug_embedding.size(0)} (drug)"
            
        cross = self.cross_layer(protein_embedding, drug_embedding)
        attention_output = self.attention_layer(protein_embedding, drug_embedding) # Cross
        self_attn = self.self_attention(drug_embedding) # Self
        # print("Attention output shape:", attention_output.shape)
        # print("Attention weight shape:", attention_weights.shape)

        protein_caps = self.protein_capsule(protein_embedding)
        drug_caps = self.drug_capsule(drug_embedding)
        
        combined = torch.cat([protein_embedding, drug_embedding, cross, attention_output, protein_caps, drug_caps], dim=-1)
        
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        # return self.fc(combined).squeeze()
        return x.squeeze()

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
    loss_fn = AmplifiedLoss(alpha=1.5)
    
    affinity_values = torch.cat([batch.affinities for batch in data_loader])
    affinity_min = affinity_values.min().item()
    affinity_max = affinity_values.max().item()
    # print("Main:", affinity_min)
    # print("Max: ", affinity_max)

    def scale_affinities(affinities):
        return (affinities - affinity_min) / (affinity_max - affinity_min)

    def inverse_scale_affinities(scaled_affinities):
        return scaled_affinities * (affinity_max - affinity_min) + affinity_min
    
    with tqdm(data_loader, desc="Training", unit="batch") as pbar:
        for batch in pbar:
            batch = batch.to(device)
            graphs = batch
            # graphs = Batch.from_data_list(graphs).to(device)
            sequences = batch.sequences
            affinities = batch.affinities
            
            scaled_affinities = scale_affinities(affinities)

            optimizer.zero_grad()
            predictions = model(sequences, graphs)
            # loss = criterion(predictions, affinities)
            loss = loss_fn(predictions, scaled_affinities)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(data_loader), (affinity_min, affinity_max)

def evaluate_model(model, data_loader, criterion, device, scaling_params, return_predictions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    affinity_min, affinity_max = scaling_params
    def inverse_scale_affinities(scaled_affinities):
        return scaled_affinities * (affinity_max - affinity_min) + affinity_min
    
    with tqdm(data_loader, desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                sequences = batch.sequences
                affinities = batch.affinities.to(device)
                
                predictions = model(sequences, batch)
                predictions_rescaled = inverse_scale_affinities(predictions)
                loss = criterion(predictions_rescaled, affinities)
                total_loss += loss.item()
                
                all_preds.extend(predictions_rescaled.cpu().numpy())
                all_targets.extend(affinities.cpu().numpy())
                
                pbar.set_postfix(loss=loss.item())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    avg_loss = total_loss / len(data_loader)
    
    if return_predictions:
        return avg_loss, metrics, np.array(all_preds), np.array(all_targets)
    
    return avg_loss, metrics


# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_file = "../data/Davis/Davis_cold.csv"
    raw_data = pd.read_csv(dataset_file)
    train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    train_file = "../data/Davis/train.csv"
    test_file = "../data/Davis/test.csv"
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
        
    smiles_train, graphs_train, sequences_train, affinities_train = load_dataset(train_file)
    smiles_test, graphs_test, sequences_test, affinities_test = load_dataset(test_file)
    train_idx = range(len(graphs_train))
    test_idx = range(len(graphs_test))

    # graphs, sequences, affinities = load_dataset(dataset_file)
    # train_idx, test_idx = train_test_split(range(len(graphs)), test_size=0.2, random_state=42)


    # train_data = [(graphs[i], sequences[i], affinities[i]) for i in train_idx]
    # test_data = [(graphs[i], sequences[i], affinities[i]) for i in test_idx]
    
    class CustomDataset:
        def __init__(self, graphs, sequences, affinities):
            self.graphs = graphs
            self.sequences = sequences
            self.affinities = affinities

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            data = self.graphs[idx]
            data.sequences = self.sequences[idx]
            data.affinities = self.affinities[idx]
            return data

    train_data = CustomDataset([graphs_train[i] for i in train_idx],
                            [sequences_train[i] for i in train_idx],
                            [affinities_train[i] for i in train_idx])

    test_data = CustomDataset([graphs_test[i] for i in test_idx],
                            [sequences_test[i] for i in test_idx],
                            [affinities_test[i] for i in test_idx])
    
    # print(train_data[0])
    
    def custom_collate(batch):
        graphs = Batch.from_data_list([item[0] for item in batch])  # Combine graphs
        sequences = [item[1] for item in batch]  # Keep sequences as a list
        affinities = torch.tensor([item[2] for item in batch], dtype=torch.float)
        return graphs, sequences, affinities

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # for batch in train_loader:
    #     print(batch)
    #     break


    # Initialize model
    model = AffinityPredictionModel(protein_dim=1024, drug_dim=256, hidden_dim=64, attention_dim=512, capsule_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    checkpoint_dir = "E:/AIDD_project/checkpoints"
    best_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    start_epoch = -1
    
    # # load
    # model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, "E:/AIDD_project/checkpoints/epoch_1_loss_1.1149.pt")
    
    log_file_path = "training_log.txt"

    # Training loop
    epochs = 10 # 50
    for epoch in range(start_epoch + 1, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, scaling_params = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_metrics, predictions, ground_truths = evaluate_model(model, test_loader, criterion, device, scaling_params, return_predictions=True)
        
        plot_affinity_scatter(predictions, ground_truths, f"affinity_scatter_epoch_{epoch+1}.png")
        
        # # Save predictions
        # if epoch == epochs - 1:
        #     save_predictions(
        #         [smiles_test[i] for i in range(len(smiles_test))],
        #         [sequences[test_idx[i]] for i in range(len(test_idx))],
        #         predictions,
        #         ground_truths,
        #         "test_results.csv"
        #     )
        
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Metrics: MSE: {test_metrics['MSE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, R^2: {test_metrics['R^2']:.4f}, CI: {test_metrics['CI']:.4f}")
        
        mse, rmse, r2, ci = test_metrics['MSE'], test_metrics['RMSE'], test_metrics['R^2'], test_metrics['CI']
        metrics = {"MSE": mse, "RMSE": rmse, "R²": r2, "CI": ci}
        log_file(epoch + 1, train_loss, test_loss, metrics, log_file_path)
        
        save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_dir)

        # # best chkpt
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     save_checkpoint(model, optimizer, epoch, test_loss, os.path.join(checkpoint_dir, "best_model.pt"))
            
        early_stopping(test_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping. Exiting training loop.")
            break
