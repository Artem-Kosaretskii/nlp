# Cross-Attention Layer
# and
# EncoderDecoderWithAttention
# and
#Teacher Forcing

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, start_token_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.cross_attn = CrossAttentionLayer(d_model=hidden_dim, d_k=hidden_dim, d_v=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim * 2, vocab_size)
        self.start_token_id = start_token_id

    def forward(self, src, tgt):
        batch_size, tgt_len = tgt.shape

        embedded_src = self.embedding(src)
        encoder_outputs, (h, c) = self.encoder(embedded_src)

        start_tokens = torch.full((batch_size, 1), self.start_token_id,
                                  dtype=torch.long, device=src.device)  # [B, 1]
        decoder_inputs = torch.cat([start_tokens, tgt[:, :-1]], dim=1)  # shift right
        embedded_trg = self.embedding(decoder_inputs)

        dec_output, (dec_hidden, dec_cell) = self.decoder(embedded_trg, (h, c))

        context, attn = self.cross_attn(encoder_outputs, dec_output)

        concat_vec = torch.cat([dec_output, context], dim=-1)
        logits = self.lm_head(concat_vec)
        return logits, attn

def test_teacher_forcing():
    vocab_size = 20
    embed_dim = 8
    hidden_dim = 16
    start_token_id = 0

    model = EncoderDecoderWithAttention(vocab_size, embed_dim, hidden_dim, start_token_id)

    src = torch.randint(0, vocab_size, (2, 5))   # batch=2, src_len=5
    tgt = torch.randint(0, vocab_size, (2, 6))   # batch=2, tgt_len=6

    logits, attn = model(src, tgt)

    # Check shapes
    assert logits.shape == (2, 6, vocab_size)
    assert attn.shape == (2, 6, src.size(1))
    print("Logits and weights shape match")


    # Check loss computation works
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, vocab_size), tgt.reshape(-1))
    print("Loss item:", loss.item())


# Run test
test_teacher_forcing()
