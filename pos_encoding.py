import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, pos_encoding=None, dropout=0.1):

        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.pos_encoding = pos_encoding
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == 'alibi':
            self._init_alibi_biases()
        elif pos_encoding == 'rope':
            self._init_rope_frequencies()

    def _init_alibi_biases(self):
        slopes = torch.tensor(self._get_alibi_slopes(self.num_heads))
        self.register_buffer('alibi_slopes', slopes.view(1, self.num_heads, 1, 1))

    def _get_alibi_slopes(self, n_heads):
        slopes = torch.ones(n_heads)
        slopes *= 2 ** (-8 / n_heads)
        slopes = torch.cumprod(slopes, dim=0)

        return slopes

    def _init_rope_frequencies(self):
        freqs = self._get_rope_frequencies()
        self.register_buffer('rope_freqs', freqs)

    def _get_rope_frequencies(self):
        base = 10000.0
        dim = self.head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def _apply_rope(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len).float()
        freqs = self.rope_freqs.view(1, 1, 1, head_dim // 2)
        positions = positions.view(1, 1, seq_len, 1)
        theta = positions * freqs
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_reshaped = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        x1_rot = x1 * cos_theta - x2 * sin_theta
        x2_rot = x1 * sin_theta + x2 * cos_theta
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
        x_rot = x_rot.view(batch_size, num_heads, seq_len, head_dim)
        return x_rot

    def _apply_alibi(self, scores):
        batch_size, head_num, seq_len, seq_len = scores.shape
        positions = torch.arange(seq_len)
        relative_positions = positions.view(1, 1, -1) - positions.view(1, -1, 1)
        alibi_bias = self.alibi_slopes.reshape(head_num, 1, 1) * relative_positions.abs().neg()
        scores = scores + alibi_bias
        return scores

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.pos_encoding == 'rope':
            Q = self._apply_rope(Q)
            K = self._apply_rope(K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if self.pos_encoding == 'alibi':
            scores = self._apply_alibi(scores)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.wo(context)

        return output, attention_weights


def test_position_invariance():
    d_model = 64
    num_heads = 4
    token = torch.randn(1, 1, d_model)
    fixed_seq = torch.randn(1, 4, d_model)
    seq1 = torch.cat([token, fixed_seq], dim = 1)
    seq2 = torch.cat([fixed_seq, token], dim = 1)
    for pos_encoding in  [None, 'alibi', 'rope']:
        print(f"\\n=== Positional sensitivity test ({pos_encoding}) ===")
        attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            pos_encoding,
            dropout=0.0
        )
        attention.eval()

        with torch.no_grad():
            output1, weights1 = attention(seq1)
            output2, weights2 = attention(seq2)

        token_output1 = output1[:, 0,:]
        token_output2 = output2[:, 4,:]

        diff = torch.abs(token_output1 - token_output2).mean().item()
        print(f"Difference: {diff:.8f}")

        if pos_encoding is None:
            if diff < 1e-6:
                print("A small difference without the positional encoding")
            else:
                print("A big difference without the positional encoding")
        else:
            if diff > 1e-3:
                print("A big difference with the positional encoding")
            else:
                print("A small difference with the positional encoding")

print("="*60)
print("Positional encodings testing")
print("="*60)

test_position_invariance()

print("\\nDone")
