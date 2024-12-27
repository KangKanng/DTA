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

class CNNblcok(nn.Module):
    def __init__(self, char_set_len):
        super().__init__()
        self.embed = nn.Embedding(char_set_len, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=8, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=12, stride=1)
    
    def forward(self, fasta):
        v = self.embed(fasta).transpose(1,2)
        v = F.relu(self.conv1(v))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        v, _  = torch.max(v, -1)
        return v

class ProteinEncoder(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert", device="cuda"):
        super(ProteinEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def forward(self, batch_sequences):
        tokens = self.tokenizer(batch_sequences, padding=True, truncation=True, max_length=1000, return_tensors="pt")
        # tokens = {key: val.to(self.device) for key, val in tokens.items()}
        tokens = {key: val.to(next(self.model.parameters()).device) for key, val in tokens.items()}
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim):
        super(GNNEncoder, self).__init__()
        # self.conv1 = GINEConv(nn.Linear(input_dim + edge_dim, hidden_dim)) # node + edge features
        # self.conv2 = GINEConv(nn.Linear(hidden_dim + edge_dim, output_dim))

        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)

        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ), edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()

        device = x.device
        # batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        # print(f"Batch tensor: {batch}")
        # print(f"Batch tensor shape: {batch.shape}")
        return global_mean_pool(x, batch)
        # return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))

class CrossLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, protein_embedding, drug_embedding):
        if protein_embedding.size(1) != drug_embedding.size(1):
            max_dim = max(protein_embedding.size(1), drug_embedding.size(1))
            protein_embedding = torch.cat([protein_embedding, torch.zeros(protein_embedding.size(0), max_dim - protein_embedding.size(1)).to(protein_embedding.device)], dim=-1)
            drug_embedding = torch.cat([drug_embedding, torch.zeros(drug_embedding.size(0), max_dim - drug_embedding.size(1)).to(drug_embedding.device)], dim=-1)

        cross_product = protein_embedding * drug_embedding  # element-wise

        concatenated = torch.cat([protein_embedding, drug_embedding, cross_product], dim=-1)
        input_dim = concatenated.size(1)

        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        interaction = self.fc3(x)

        return interaction


class AttentionLayer(nn.Module):
    def __init__(self, protein_dim, drug_dim, hidden_dim):
        super(AttentionLayer, self).__init__()

        self.query_linear = nn.Linear(protein_dim, hidden_dim)
        self.key_linear = nn.Linear(drug_dim, hidden_dim)
        self.value_linear = nn.Linear(drug_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, protein_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, protein_embedding, drug_embedding):
        query = self.query_linear(protein_embedding)
        key = self.key_linear(drug_embedding)  # (batch_size, drug_dim) -> (batch_size, hidden_dim)
        value = self.value_linear(drug_embedding)  # (batch_size, drug_dim) -> (batch_size, hidden_dim)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, hidden_dim) x (batch_size, hidden_dim).T = (batch_size, 1, seq_len)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))  # Scale
        attention_weights = self.softmax(attention_scores)  # (batch_size, 1, seq_len)
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_dim) -> (batch_size, 1, hidden_dim)
        output = self.output_linear(attention_output.squeeze(1))  # (batch_size, hidden_dim) -> (batch_size, protein_dim)

        return output, attention_weights


class CrossAttentionLayer(nn.Module):
    def __init__(self, protein_dim, drug_dim, attention_dim):
        super(CrossAttentionLayer, self).__init__()
        self.protein_proj = nn.Linear(protein_dim, attention_dim)
        self.drug_proj = nn.Linear(drug_dim, attention_dim)

        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=4, dropout=0.1)

    def forward(self, protein_embedding, drug_embedding):
        protein_proj = self.protein_proj(protein_embedding)  # (batch_size, attention_dim)
        drug_proj = self.drug_proj(drug_embedding)  # (batch_size, attention_dim)

        protein_proj = protein_proj.unsqueeze(0)  # (1, batch_size, attention_dim)
        drug_proj = drug_proj.unsqueeze(0)  # (1, batch_size, attention_dim)

        # Q: protein; K/V: drug
        attention_output, attention_weights = self.attention(protein_proj, drug_proj, drug_proj)
        return attention_output.squeeze(0), attention_weights.squeeze(0) # (1, batch_size, attention_dim


class AffinityPredictionModel(nn.Module):
    def __init__(self, protein_dim, drug_dim, hidden_dim, attention_dim):
        super(AffinityPredictionModel, self).__init__()
        self.protein_encoder = CNNblcok(25+1)
        self.drug_encoder = GNNEncoder(input_dim=4, hidden_dim=hidden_dim, output_dim=drug_dim, edge_dim=3)
        cross_dim = max(protein_dim, drug_dim) * 3
        self.cross_layer = CrossLayer(input_dim=cross_dim, hidden_dim=hidden_dim)
        self.attention_layer = CrossAttentionLayer(protein_dim, drug_dim, attention_dim)

        output_dim = protein_dim + drug_dim + 1 + attention_dim
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
        attention_output, attention_weights = self.attention_layer(protein_embedding, drug_embedding)
        combined = torch.cat([protein_embedding, drug_embedding, cross, attention_output], dim=-1)

        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        # return self.fc(combined).squeeze()
        return x.squeeze()