import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def sliding_window_attention(Q, K, V, window_size):
    # Q,K,V (seq_len, batch, d_model)
    seq_len = Q.size(0)

    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)
        mask[i, left:right] = 0

    Q, K, V = Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1)  # shape: (batch, seq_len, d_model)
    mask = mask.unsqueeze(0).expand(Q.size(0), -1, -1)
    attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    attn_out = attn_out.transpose(0, 1)

    return attn_out, mask

def test_attention():
    batch, heads, seq_len, head_dim = 8, 4, 512, 32
    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    query = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    import torch.utils.benchmark as benchmark
    def bench(f, *args):
        t0 = benchmark.Timer(stmt="f(*args)", globals={"f": f, "args": args})
        return t0.blocked_autorange().mean * 1e6

    print("Test...")

    with sdpa_kernel(SDPBackend.MATH):
        t_math = bench(F.scaled_dot_product_attention, query, key, value)
    print(f"Math implementation: {t_math:.1f} мкс")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        try:
            t_flash = bench(F.scaled_dot_product_attention, query, key, value)
            print(f"Flash Attention impl.: {t_flash:.1f} мкс")
        except RuntimeError as e:
            print("Flash Attention is not supported:", e)

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        t_eff = bench(F.scaled_dot_product_attention, query, key, value)
    print(f"Memory-Efficient impl.: {t_eff:.1f} мкс")


def main():

    seq_len = 16
    batch = 2
    d_model = 16
    window_size = 2
    tokens = torch.randn(seq_len, batch, d_model)
    attn_out, mask = sliding_window_attention(tokens, tokens, tokens, window_size)

    print(attn_out.shape)
    print(mask[0, 5].detach().numpy().round(1))

    test_attention()


if __name__ == "__main__":
    main()
