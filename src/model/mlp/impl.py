import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = None
        self.gelu = None
        self.c_proj = None
        self.dropout = None

    def forward(self, x):
        raise NotImplementedError
