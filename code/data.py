import torch
from torch_geometric.data import Data, Batch

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 }

CHARPROTLEN = 25

def label_chars(chars, max_len, char_set):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X

def custom_collate(batch):
    graphs = Batch.from_data_list([item[0] for item in batch])  # Combine graphs
    sequences = [label_chars(item[1], 1000, CHARPROTSET) for item in batch]  # Keep sequences as a list
    affinities = torch.tensor([item[2] for item in batch], dtype=torch.float)
    return graphs, sequences, affinities


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
