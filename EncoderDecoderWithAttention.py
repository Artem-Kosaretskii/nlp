# Cross-Attention Layer
# and
# EncoderDecoderWithAttention

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


class EncoderDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, start_token_id, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.cross_attn = CrossAttentionLayer(d_model=hidden_dim, d_k=hidden_dim, d_v=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim * 2, vocab_size)  # concat(dec_output, context)
        self.start_token_id = start_token_id
        self.max_len = max_len

    def forward(self, src, tgt=None):
        batch_size = src.size(0)

        embedded_src = self.embedding(src)                           # [B, T_src, E]
        encoder_outputs, (h, c) = self.encoder(embedded_src)          # enc_outputs: [B, T_src, H]
        input_token = torch.full((batch_size,), self.start_token_id, dtype=torch.long, device=src.device)

        logits_history = []
        attn_history = []
        dec_hidden, dec_cell = h, c

        for _ in range(self.max_len):
            embedded_t = self.embedding(input_token).unsqueeze(1)     # [B, 1, E]
            dec_output, (dec_hidden, dec_cell) = self.decoder(embedded_t, (dec_hidden, dec_cell))  # [B, 1, H]

            # CrossAttention
            context, attn = self.cross_attn(encoder_outputs, dec_output)   # [B, 1, H], [B, 1, T_src]
            attn_history.append(attn)

            concat_vec = torch.cat([dec_output, context], dim=-1)     # [B, 1, 2H]
            step_logits = self.lm_head(concat_vec)                    # [B, 1, vocab]
            logits_history.append(step_logits)
            input_token = step_logits.argmax(dim=-1).squeeze(1)       # [B]

        logits = torch.cat(logits_history, dim=1)                     # [B, max_len, vocab]
        attn_history = torch.cat(attn_history, dim=1)                 # [B, max_len, T_src]
        return logits, attn_history

def test_shapes():
    vocab_size = 50
    embed_dim = 16
    hidden_dim = 32
    start_token_id = 1
    max_len = 5

    model = EncoderDecoderWithAttention(vocab_size, embed_dim, hidden_dim,
                                        start_token_id=start_token_id,
                                        max_len=max_len)

    src = torch.randint(0, vocab_size, (2, 7))   # batch=2, src_len=7
    logits, attn = model(src)

    assert logits.shape == (2, max_len, vocab_size), f"Wrong logits shape: {logits.shape}"
    assert attn.shape == (2, max_len, src.size(1)), f"Wrong attn weights shape: {attn.shape}"
    print("Shapes test passed")


def test_greedy_generation():
    vocab_size = 10
    embed_dim = 8
    hidden_dim = 16
    start_token_id = 0
    max_len = 3

    model = EncoderDecoderWithAttention(vocab_size, embed_dim, hidden_dim,
                                        start_token_id=start_token_id,
                                        max_len=max_len)

    src = torch.randint(0, vocab_size, (1, 4))   # batch=1
    logits, attn = model(src)

    preds = logits.argmax(dim=-1)  # \[1, max_len\]
    print("Predicted sequence:", preds.tolist())
    print("Attn weights:\\n", attn)


# Run tests
test_shapes()
test_greedy_generation()
