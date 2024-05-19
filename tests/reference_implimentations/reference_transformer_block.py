import torch.nn as nn
import torch.nn.functional as F
import torch 

from tests.reference_implimentations.reference_causal_attention import (
    CausalSelfAttention,
)
from tests.reference_implimentations.reference_layernorm import LayerNorm
from tests.reference_implimentations.reference_mlp import MLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
