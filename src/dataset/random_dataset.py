import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, length, data_shape):
        self.length = length
        self.data_shape = data_shape

        self.data = torch.randn(length, *data_shape)
        self.label = torch.randint(0, 10, (length,))
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.length
