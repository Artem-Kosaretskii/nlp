import torch
import torch.nn.functional as F
import numpy as np


class AttentionWithKVCache:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)

        torch.manual_seed(42)
        for layer in [self.q_proj, self.k_proj, self.v_proj]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def generation_step(self, new_token_embeddings, cache_k, cache_v):

        new_k = self.k_proj(new_token_embeddings)  # [batch_size, 1, hidden_size]
        new_v = self.v_proj(new_token_embeddings)  # [batch_size, 1, hidden_size]

        cache_k = torch.cat([cache_k, new_k], dim=1)
        cache_v = torch.cat([cache_v, new_v], dim=1)

        q = self.q_proj(new_token_embeddings)  # [batch_size, 1, hidden_size]

        attn_weights = torch.matmul(q, cache_k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, cache_v)

        return attn_output, cache_k, cache_v


def test_attention_with_kv_cache():

    batch_size = 2
    hidden_size = 8
    seq_len = 3

    attention = AttentionWithKVCache(hidden_size)
    full_sequence = torch.randn(batch_size, seq_len, hidden_size)

    K_full = attention.k_proj(full_sequence)
    V_full = attention.v_proj(full_sequence)
    Q_full = attention.q_proj(full_sequence[:, -1:, :])

    attn_weights_full = torch.matmul(Q_full, K_full.transpose(1, 2)) / (hidden_size ** 0.5)
    attn_weights_full = F.softmax(attn_weights_full, dim=-1)
    expected_output = torch.matmul(attn_weights_full, V_full)

    cache_k = torch.zeros((batch_size, 0, hidden_size))
    cache_v = torch.zeros((batch_size, 0, hidden_size))

    for i in range(seq_len):
        token_emb = full_sequence[:, i:i + 1, :]
        output, cache_k, cache_v = attention.generation_step(token_emb, cache_k, cache_v)

    assert torch.allclose(output, expected_output, atol=1e-6), "Results don't match with full attention"
    assert torch.allclose(cache_k, K_full, atol=1e-6), "K cache don't match"
    assert torch.allclose(cache_v, V_full, atol=1e-6), "V cache don't match"
    print("Matching is proved")
    zero_input = torch.zeros(1, 1, hidden_size)
    cache_k_zero = torch.zeros(1, 0, hidden_size)
    cache_v_zero = torch.zeros(1, 0, hidden_size)
    output_zero, _, _ = attention.generation_step(zero_input, cache_k_zero, cache_v_zero)
    assert not torch.isnan(output_zero).any(), "NaN detected"
    print("All correct")
    print("Done")


def example_usage():
    print("An example of KV-cache usage:")
    print("-" * 50)

    # Инициализация
    hidden_size = 64
    batch_size = 1
    attention = AttentionWithKVCache(hidden_size)

    cache_k = torch.zeros((batch_size, 0, hidden_size))
    cache_v = torch.zeros((batch_size, 0, hidden_size))
    num_tokens = 5
    print(f"Generation of {num_tokens} tokens:")

    for step in range(num_tokens):
        new_token_emb = torch.randn(batch_size, 1, hidden_size)

        attn_output, cache_k, cache_v = attention.generation_step(new_token_emb, cache_k, cache_v)

        print(f"Step {step + 1}:")
        print(f"  K cache size: {cache_k.shape}")
        print(f"  V cache size: {cache_v.shape}")
        print(f"  Attention output: {attn_output.shape}")

    print("\nDone")


if __name__ == "__main__":
    test_attention_with_kv_cache()
    print("\n")
    example_usage()

