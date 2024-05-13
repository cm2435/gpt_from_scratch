import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention.impl import CausalSelfAttention
from src.model.layernorm.impl import LayerNorm
from src.model.mlp.impl import MLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = None
        self.attn = None
        self.ln_2 = None
        self.mlp = None

    def forward(self, x):
        raise NotImplementedError
