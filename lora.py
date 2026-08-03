import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class SoftPrompt(nn.Module):
    def __init__(self, k=20, d=768):
        super().__init__()
        self.k = k
        self.emb = nn.Parameter(torch.randn(k, d))

    def forward(self, input_ids, model_embed):
        # input_ids: (B, L)
        B, L = input_ids.shape
        tok_emb = model_embed(input_ids)          # (B, L, d)
        soft = self.emb.unsqueeze(0).expand(B, -1, -1)  # (B, k, d)
        return torch.cat([soft, tok_emb], dim=1)  # (B, k+L, d)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

    def forward(self, x):
        base_out = self.base(x)
        update = self.B(self.A(x)) * self.scaling
        return base_out + update

#LoRA
lora = LoRALinear(128, 128, 4, 16)
x = torch.randn(2, 5, 128)
print(lora(x).size())

# P-tuning
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print(model)
p_tuning = SoftPrompt(k=20, d=model.config.hidden_size)
input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
output = p_tuning(input_ids, model.get_input_embeddings())
print(output.size())
