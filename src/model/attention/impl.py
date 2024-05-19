import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Separate linear layers for key, query, and value
        self.key = None
        self.query = None
        self.value = None

        # Output projection
        self.c_proj = None

        # Regularization
        self.attn_dropout = None
        self.resid_dropout = None

        # Number of heads for multi head attention, embedding size, and dropout rate.
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.tensor, use_dropout: bool = False) -> torch.tensor:
        """
        Perform multi-head attention using three projection matrices.

        B = batch size
        S = source sequence length
        C = Embedding dimension

        :param x: Either encoder or decoder hidden states. Shape: (B, S, C)

        :return: Contextualized token embeddings.
        """

        B, T, C = x.size()  # Batch size, sequence length, embedding size

        # 1. Generate queries, keys, and values for all heads

        # 2.
        # Reshape and transpose to separate heads and prepare for multi-head attention
        # Reshape from (B, T, n_embd) -> (B, T, n_head, head_dim)
        # head_dim is calculated as n_embd // n_head
        # Then transpose to (B, n_head, T, head_dim) to make heads a separate dimension

        # 3.
        # Reshape again to merge the batch size and number of heads into a single dimension
        # This prepares the tensors for batch matrix multiplication
        # Reshape from (B, n_head, T, head_dim) to (B * n_head, T, head_dim) for q
        # and to (B * n_head, head_dim, T) for k to allow batched matrix multiplication

        # 4.
        # Perform batch matrix multiplication to compute attention scores
        # The result is a tensor of shape (B * n_head, T, T) representing attention scores

        # 5.
        # Apply causal mask by setting future positions to negative infinity
        # This mask ensures that each position in the sequence can only attend to the current and previous positions.
        # The resulting mask looks like this for a sequence length T=3:
        # [[  0, -inf, -inf],
        #  [  0,    0, -inf],
        #  [  0,    0,    0]]
        # We add this mask to the scores tensor, broadcasting it to match the dimensions [B, n_head, T, T].
        # This ensures that the softmax operation later will assign zero probability to attending to future positions,
        # effectively masking out any future information and preserving the causality of the model.

        # 6.
        # Apply softmax to normalize the scores

        # 7.
        # Fidn the Attention output
        # (B, n_head, T, T) * (B, n_head, T, hs) -> (B, n_head, T, hs)

        # 8. Final!
        # Concatenate the outputs of different heads and project
        raise NotImplementedError
