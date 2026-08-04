#Cross-Attention Layer

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def cross_attention(q, k, v):
    d_k = q.size(-1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    attn_weights = F.softmax(attn_scores, dim=-1)
    outputs = torch.matmul(attn_weights, v)
    return outputs, attn_weights


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        # Init layers #
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, enc_output, dec_output):
        Q = self.W_q(dec_output)
        K = self.W_k(enc_output)
        V = self.W_v(enc_output)

        output, attn_weights = cross_attention(Q, K, V)
        return output, attn_weights

Q = torch.tensor([[10, 0, 0, 0], [0, 10, 0, 0]]).float()
K = torch.tensor([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]]).float()
V = torch.tensor([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 30, 0]]).float()
assert cross_attention(Q, K, V)[0].sum() == 30

