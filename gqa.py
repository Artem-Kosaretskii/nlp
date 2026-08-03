import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.0, bias=True):

        super().__init__()
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_groups * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_groups * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.heads_per_group = num_heads // num_groups

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, L, _ = x.size()
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # (B, G, L, d)
        v = self.v_proj(x).view(B, L, self.num_groups, self.head_dim).transpose(1, 2)  # (B, G, L, d)

        k = k.reshape(B * self.num_groups, L, self.head_dim)
        v = v.reshape(B * self.num_groups, L, self.head_dim)

        k_expanded = k.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1)
        v_expanded = v.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1)
        k_expanded = k_expanded.reshape(B, self.num_groups * self.heads_per_group, L, self.head_dim)
        v_expanded = v_expanded.reshape(B, self.num_groups * self.heads_per_group, L, self.head_dim)
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3))  # (B, H, L, L)
        attn_weights = attn_weights * self.scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)  # (B, H, L, L)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v_expanded)  # (B, H, L, d)
        attn_output = attn_output.view(B, self.num_groups, self.heads_per_group, L, self.head_dim)
        attn_output = attn_output.reshape(B, self.num_heads, L, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)  # (B, L, D)

        out = self.out_proj(attn_output)  # (B, L, D)
        return out

batch_size = 2
seq_len = 10
embed_dim = 64
num_heads = 16
num_groups = 4

model = GQA(embed_dim=embed_dim, num_heads=num_heads, num_groups=num_groups)
x = torch.randn(batch_size, seq_len, embed_dim)

causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))  # (L, L)
causal_mask = causal_mask.unsqueeze(0)

print("Input shape:", x.shape)
out = model(x, attn_mask=causal_mask)  # (B, L, D)
print("Output shape:", out.shape)
