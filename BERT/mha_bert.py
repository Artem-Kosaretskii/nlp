import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = self.linear1(x)   # (seq_len, batch, dim_ff)
        x = self.relu(x)
        x = self.linear2(x)   # (seq_len, batch, d_model)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        attn_out, _ = self.self_attn(x, x, x)     # Multi-Head Attention
        x = self.norm1(x + attn_out)              # Add & Norm
        ff_out = self.ff(x)                       # FeedForward
        x = self.norm2(x + ff_out)               # Add & Norm
        return x


class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.WO = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        seq_len, batch, d_model = query.shape
        Q = self.WQ(query)   # (seq_len, batch, d_model)
        K = self.WK(key)     # (seq_len, batch, d_model)
        V = self.WV(value)   # (seq_len, batch, d_model)

        #    (seq_len, batch, d_model) → (batch, seq_len, d_model)
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        #    (batch, seq_len, d_model) → (batch, seq_len, H, d_head) → (batch, H, seq_len, d_head)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (batch, H, seq, d_head) × (batch, H, d_head, seq) → (batch, H, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # (seq_len, seq_len) → (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        # (batch, H, seq, seq) × (batch, H, seq, d_head)
        out = torch.matmul(attention, V)
        # (batch, H, seq, d_head) → (batch, seq, H, d_head) → (batch, seq, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, d_model)
        out = self.WO(out)  # (batch, seq_len, d_model)
        # (seq_len, batch, d_model)
        out = out.permute(1, 0, 2)
        return out



def main(vis=False):
    d_model = 64
    seq_len = 128
    batch_size = 16
    vocab_size = 16384
    tokens = torch.randint(0, vocab_size, (seq_len, batch_size))
    print(tokens.shape)
    emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    x = emb_layer(tokens)
    print(x.shape)
    pos = torch.arange(seq_len).unsqueeze(1)  # [[0],[1],...,[seq_len - 1]]
    k = torch.arange(0, d_model, 2)
    angle_rates = 1 / torch.pow(10000, k / d_model)
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(pos * angle_rates)
    pos_enc[:, 1::2] = torch.cos(pos * angle_rates)
    pos_enc = pos_enc.unsqueeze(1).repeat(1, batch_size, 1)  # (seq_len, batch_size, d_model)
    print(pos_enc.shape)
    h = x + pos_enc
    print(h.shape)
    if vis:
        make_plot(pos)

    d_model, dim_ff = 64, 256
    ff_block = FeedForward(d_model, dim_ff)
    seq_len, batch_size = 8, 2
    x = torch.randn(seq_len, batch_size, d_model)
    y = ff_block(x)
    print(x.shape, '->', y.shape)

    d_model, num_heads, dim_ff = 64, 8, 256
    encoder_block = TransformerEncoderBlock(d_model, num_heads, dim_ff)
    x = torch.randn(16, 2, d_model)          # (seq_len=10, batch_size=2, d_model=64)
    y = encoder_block(x)
    print(y.shape)  # (10, 2, 64)

    d_model, num_heads = 64, 8
    mha = MyMultiHeadAttention(d_model, num_heads)
    x = torch.randn(16, 2, d_model)  # (seq_len=10, batch=2, d_model=64)
    y = mha(x, x, x, mask=None)
    print(f'Tensor shape after MyMHA: {y.shape}')  # (10, 2, 64)


def make_plot(pos_enc):
    """
    Visualization
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(pos_enc, aspect='auto', cmap='viridis')
    plt.xlabel('d_model')
    plt.ylabel('position')
    plt.title('sin-cos pos coding')
    plt.colorbar(label='PE value')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
