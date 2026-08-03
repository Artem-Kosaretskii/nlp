import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()


@torch.no_grad()
def sample_next_token(logits, generated_ids, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):

    if repetition_penalty is not None and repetition_penalty > 1.0 and generated_ids is not None and generated_ids.numel() > 0:
        vals, counts = torch.unique(generated_ids, return_counts=True)
        logits = logits.clone()
        for v, c in zip(vals, counts):
            logits[..., v] /= (repetition_penalty ** c.item())

    temp = max(1e-6, float(temperature))
    logits = logits / temp

    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, k=top_k)[0][..., -1, None]
        mask = logits < kth
        logits = logits.masked_fill(mask, float('-inf'))

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumprobs <= top_p
        keep_mask[..., 0] = True
        filtered = torch.full_like(sorted_probs, float('-inf'))
        filtered[keep_mask] = torch.log(sorted_probs[keep_mask])
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, sorted_idx, filtered)

    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    print("Top-5 candidates:")
    for pid, pval in zip(top_ids.tolist(), top_probs.tolist()):
        print(f"  {tokenizer.decode([pid]):<15} : {pval:.4f}")
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id


@torch.no_grad()
def generate_custom(prompt, max_new_tokens=128, temperature=0.9, top_k=20, top_p=0.9, repetition_penalty=1.1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        next_logits = outputs.logits[:, -1, :].squeeze(0)
        next_id = sample_next_token(
            next_logits, generated_ids=generated[0],
            temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
        )
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0])


print(generate_custom("Come up with a funny but polite toast about developers:", max_new_tokens=128))

messages = [{"role": "user", "content": "Give a short and precise answer to the question: What is Bayesian probability?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

out_beam = model.generate(
    inputs,
    max_new_tokens=128,
    num_beams=4,
    early_stopping=True,
    length_penalty=0.8,
    do_sample=False
)
print(tokenizer.batch_decode(out_beam))

messages = [{"role": "user", "content": "Come up with three unique coffee shop slogans using wordplays with the word 'bayes'."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

out_sample = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
)
print(tokenizer.batch_decode(out_sample))
