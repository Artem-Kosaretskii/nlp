import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cpu' # cuda is not available with q_int8
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
model_cpu = model.to(device)

text = 'This is a sample sentence for quantization benchmark.'
inputs = tokenizer(text, return_tensors='pt').to(device)

n_runs = 50

with torch.inference_mode():
    start = time.time()
    for _ in range(n_runs):
        _ = model_cpu(**inputs)
    t_orig = (time.time() - start) / n_runs
print(f'FP32 avg inference time (per run): {t_orig:.6f} s')

print('Applying dynamic quantization...')
quantized_model = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.eval()

with torch.inference_mode():
    start = time.time()
    for _ in range(n_runs):
        _ = quantized_model(**inputs)
    t_orig = (time.time() - start) / n_runs
print(f'Quantized avg inference time (per run): {t_orig:.6f} s')

with torch.inference_mode():
    logits_fp32 = model_cpu(**inputs).logits.detach()
    logits_q = quantized_model(**inputs).logits.detach()

print('Logit examples (FP32 vs Quantized):')
print(logits_fp32[0][:6].tolist())
print(logits_q[0][:6].tolist())

l2 = torch.norm(logits_fp32 - logits_q).item()
max_abs = torch.max(torch.abs(logits_fp32 - logits_q)).item()

tmp_fp = 'model_fp32.pth'
tmp_q = 'model_q.pth'
torch.save(model_cpu.state_dict(), tmp_fp)
torch.save(quantized_model.state_dict(), tmp_q)
print('FP32 size (MB):', os.path.getsize(tmp_fp)/1024/1024)
print('Quant size (MB):', os.path.getsize(tmp_q)/1024/1024)
