import torch
from torch.utils.data import Dataset
import argparse
import sys
import os
from CONSTANTS import DATA_DIR



class EmbeddingDataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser):
        self.args = args
        file_path = os.path.join(DATA_DIR[args.dataset], 'embeddings', f'{os.path.basename(args.bert_model)}_{args.bert_pool}.pt')
        self.embedding = torch.load(file_path, map_location='cpu')
        args.embedding_dim = self.embedding.shape[1]
        args.in_channels = self.embedding.shape[1]            
    
    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(self, idx):
        return self.embedding[idx]



