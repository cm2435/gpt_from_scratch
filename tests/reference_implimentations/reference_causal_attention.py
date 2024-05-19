import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Separate linear layers for key, query, and value
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Number of heads for multi head attention, embedding size, and dropout rate.
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, use_dropout: bool = False):
        """
        Perform multi-head attention using three projection matrices.

        B = batch size
        S = source sequence length
        C = Embedding dimension

        :param x: Either encoder or decoder hidden states. Shape: (B, S, C)

        :return: Contextualized token embeddings.
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding size

        # Generate queries, keys, and values for all heads
        q = self.query(x)  # Shape: (B, T, n_embd)
        k = self.key(x)  # Shape: (B, T, n_embd)
        v = self.value(x)  # Shape: (B, T, n_embd)

        # Reshape and transpose to separate heads and prepare for multi-head attention
        # Reshape from (B, T, n_embd) -> (B, T, n_head, head_dim)
        # head_dim is calculated as n_embd // n_head
        # Then transpose to (B, n_head, T, head_dim) to make heads a separate dimension
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Reshape again to merge the batch size and number of heads into a single dimension
        # This prepares the tensors for batch matrix multiplication
        # Reshape from (B, n_head, T, head_dim) to (B * n_head, T, head_dim) for q
        # and to (B * n_head, head_dim, T) for k to allow batched matrix multiplication
        q = q.reshape(-1, T, C // self.n_head)  # Shape: (B * n_head, T, head_dim)
        k = k.reshape(-1, T, C // self.n_head).transpose(1, 2)  # Shape: (B * n_head, head_dim, T)
        v = v.reshape(-1, T, C // self.n_head)  # Shape: (B* n_head, T, head_dimention)

        # Perform batch matrix multiplication to compute attention scores
        # The result is a tensor of shape (B * n_head, T, T) representing attention scores
        scores = torch.bmm(q, k)  # Shape: (B * n_head, T, T)
        scores = scores / math.sqrt(C // self.n_head)
        scores = scores.view(B, self.n_head, T, T)

        # Apply causal mask by setting future positions to negative infinity
        # This mask ensures that each position in the sequence can only attend to the current and previous positions.
        # The resulting mask looks like this for a sequence length T=3:
        # [[  0, -inf, -inf],
        #  [  0,    0, -inf],
        #  [  0,    0,    0]]
        # We add this mask to the scores tensor, broadcasting it to match the dimensions [B, n_head, T, T].
        # This ensures that the softmax operation later will assign zero probability to attending to future positions,
        # effectively masking out any future information and preserving the causality of the model.
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        scores = scores + mask[None, None, :, :].to(scores.device)

        # Apply softmax to normalize the scores
        att = F.softmax(scores, dim=-1)
        if use_dropout:
            att = self.attn_dropout(att)

        # Attention output
        # (B, n_head, T, T) * (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = torch.bmm(att.view(-1, T, T), v)
        y = y.view(B, self.n_head, T, C // self.n_head)

        # Concatenate the outputs of different heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(y)
        y = self.c_proj(y)

        return y
