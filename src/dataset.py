from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import pandas as pd

from src.paths import CHUNK_DIR, DATA_DIR
import json
from os.path import realpath, join
import os

class IMDB(Dataset):
    r"""Initializes document and corresponding label"""
    def __init__(self, split = "train"):
        super().__init__()
        imdb = load_dataset('imdb')
        return imdb[split]
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        output = {
            'text' : row[1],
            'label': row.label,
        }
        return output

    def __len__(self):
        return len(self.data)