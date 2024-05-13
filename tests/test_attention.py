import unittest

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from src.model.attention.impl import CausalSelfAttention

# Comment out the above and uncomment the below if you want to test the reference implimentation
# from tests.reference_implimentations.reference_causal_attention import CausalSelfAttention


class TestCausalSelfAttention(unittest.TestCase):
    class Config:
        def __init__(self, n_embd, n_head, dropout, bias):
            self.n_embd = n_embd
            self.n_head = n_head
            self.dropout = dropout
            self.bias = bias

    def setUp(self):
        config = self.Config(n_embd=512, n_head=8, dropout=0.0, bias=True)
        self.mha = CausalSelfAttention(config)
        self.pytorch_mha = MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.0,
            bias=True,
        )

        self.pytorch_mha.in_proj_weight.data = torch.cat(
            [
                self.mha.query.weight.data,
                self.mha.key.weight.data,
                self.mha.value.weight.data,
            ],
            dim=0,
        )
        self.pytorch_mha.in_proj_bias.data = torch.cat(
            [
                self.mha.query.bias.data,
                self.mha.key.bias.data,
                self.mha.value.bias.data,
            ],
            dim=0,
        )
        self.pytorch_mha.out_proj.weight.data = self.mha.c_proj.weight.data
        self.pytorch_mha.out_proj.bias.data = self.mha.c_proj.bias.data

    def test_forward(self):
        x = torch.randn(4, 10, 512, dtype=torch.float)
        output = self.mha.forward(x)
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertFalse(torch.any(torch.isnan(output)))

    def test_future_masking(self):
        B, T, C = 2, 3, 512
        x = torch.randn(B, T, C, dtype=torch.float)

        output = self.mha.forward(x)
        self.assertEqual(output.shape, (B, T, C))
        self.assertFalse(torch.any(torch.isnan(output)))

        q = x.transpose(0, 1)
        k = x.transpose(0, 1)
        v = x.transpose(0, 1)

        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1).to(x.device)

        attn_output, _ = self.pytorch_mha(q, k, v, attn_mask=mask, is_causal=True)
        attn_output = attn_output.transpose(0, 1)
        torch.testing.assert_close(output, attn_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
