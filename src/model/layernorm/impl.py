import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = None
        self.bias = None

    def forward(self, input):
        raise NotImplementedError
