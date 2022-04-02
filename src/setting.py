import torch
import torch.nn as nn

class config(object):
    epochs = 10
    batch_size = 4
    lr = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    