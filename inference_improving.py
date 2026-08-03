import re
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset

SEED = 42
ANSWER_TAG = '[ANSWER]'
NUM_RE = re.compile(r"[-+]?(?:\d+\s*/\s*\d+|(?:\d+|\d+)(?:[.,]\d+)?)")
MODEL_ID = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
ds = load_dataset("t-tech/T-math", split="train")
random.seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def extract_answer(text: str):
    if text is None:
        return False, None
    assistant_text = text.split('assistant')[-1]
    if ANSWER_TAG not in assistant_text:
        return False, None
    answer = assistant_text.split(ANSWER_TAG)[-1]
    return True, answer

def normalize(s: str) -> str:
    if s is None:
        return ""
    raw = str(s).strip().lower()
    m = NUM_RE.search(raw)
    if m:
        num = m.group(0).replace(" ", "").replace(",", ".")
        try:
            if "." in num:
                v = str(float(num)).rstrip("0").rstrip(".")
            else:
                v = str(int(num))
            return v
        except ValueError:
            return num
    raw = raw
    raw = re.sub(r"\\s+", " ", raw).rstrip(" .,")
    return raw

def compute_metrics(preds, refs):
    parsed = 0
    correct = 0
    for p, r in zip(preds, refs):
        ok, val = extract_answer(p)
        if ok:
            parsed += 1
            if normalize(val) == r:
                correct += 1
    n = len(refs)
    return {
        "format_rate": parsed / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "count": n,
    }

def generate_baseline(question: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Solve a problem and give an answer ONLY number after '[ANSWER]'.\n"
            f"Задача: {question}\n"
        ),
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt'
    )
    with torch.no_grad():
        out = model.generate(input_ids=inputs.to(model.device), max_new_tokens=64, do_sample=False)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def generate_few_shot(question: str, idx: int, n_examples: int = 3, multiple_msg: bool = False) -> str:
    candidates = list(range(len(ds)))
    candidates.remove(idx)
    sample_idxs = random.sample(candidates, n_examples)

    if not multiple_msg:
        few_shot_prompt = "Solve a problem and give an answer ONLY number after '[ANSWER]'.\n\n"
        for ex_idx in sample_idxs:
            ex = ds[ex_idx]
            few_shot_prompt += (
                f"Problem: {ex['question']}\n"
                f"[ANSWER] {ex['verifiable_answer']}\n\n"
            )
        few_shot_prompt += f"Задача: {question}\n"
        messages = [{
            "role": "user",
            "content": few_shot_prompt
        }]

    else:

        messages = [
            {
                "role": "system",
                "content": "Solve a problem and give an answer ONLY number after '[ANSWER]'."
            }
        ]

        for ex_idx in sample_idxs:
            ex = ds[ex_idx]
            messages.append({
                "role": "user",
                "content": f"Problem: {ex['question']}"
            })
            messages.append({
                "role": "assistant",
                "content": f"[ANSWER] {ex['verifiable_answer']}"
            })

        messages.append({
            "role": "user",
            "content": f"Problem: {question}"
        })

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt'
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.to(model.device),
            max_new_tokens=64,
            do_sample=False
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def build_messages(question: str, idx: int, n_examples: int = 3, sol=False):
    if sol:
        candidates = [i for i in range(len(ds)) if len(ds[i]['solutions']) > 0 and i != idx]
    else:
        candidates = list(range(len(ds)))
        candidates.remove(idx)
    sample_idxs = random.sample(candidates, n_examples)

    if sol:
        messages = [{
                "role": "system",
                "content": "Write '[SOLUTION]' and a solution after it continuous step by step. "
                           "Write '[ANSWER]' and give an answer ONLY number.\n"
            }]
    else:
        messages = [{"role": "system", "content": "Solve a problem and give an answer ONLY number after '[ANSWER]'."}]

    for ex_idx in sample_idxs:
        ex = ds[ex_idx]
        messages.append({
            "role": "user",
            "content": f"Problem: {ex['question']}"
        })
        if sol:
            messages.append({
                "role": "assistant",
                "content": f"[SOLUTION] {ex['solutions'][0]} [ANSWER] {ex['verifiable_answer']}"
            })
        else:
            messages.append({
                "role": "assistant",
                "content": f"[ANSWER] {ex['verifiable_answer']}"
            })
    messages.append({
        "role": "user",
        "content": f"Problem: {question}"
    })
    return messages

def generate_batch(batch_indices, n_examples: int = 3, do_sample=False, sol=False):
    batch_messages = [build_messages(ds[i]["question"], i, n_examples, sol) for i in batch_indices]
    chat_inputs = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(chat_inputs, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = model.generate(**inputs.to(model.device), max_new_tokens=64, do_sample=do_sample)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def build_prompt(question: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Solve a problem and give an answer ONLY number after '[ANSWER]'.\n"
            f"Problem: {question}\n"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

def generate_reasoning(prompts, batch_size=1):
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        chat_inputs = prompts[i:i+batch_size]
        inputs = tokenizer(
            chat_inputs,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                max_new_tokens=32768,
                do_sample=True
            )
        batch_decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(batch_decoded)
    return preds

def main():

    # Baseline: ZERO-SHOT
    preds_raw = [generate_baseline(r["question"]) for r in tqdm(ds)]
    refs = [r["verifiable_answer"] for r in ds]
    print("Prediction:\n", preds_raw[0])
    print(compute_metrics(preds_raw, refs)) # {'format_rate': 0.6193353474320241, 'accuracy': 0.02416918429003021, 'count': 331}  - very low

    # FEW-SHOT
    preds_raw = [generate_few_shot(r["question"], i, n_examples=3) for i, r in enumerate(tqdm(ds))]
    refs = [r["verifiable_answer"] for r in ds]
    print("Prediction:\n", preds_raw[0])
    print(compute_metrics(preds_raw, refs)) # {'format_rate': 1.0, 'accuracy': 0.021148036253776436, 'count': 331} - still very low

    # FEW-SHOT (multiple messages)
    preds_raw = [generate_few_shot(r["question"], i, n_examples=3, multiple_msg=True) for i, r in enumerate(tqdm(ds))]
    refs = [r["verifiable_answer"] for r in ds]
    print("Prediction:\n", preds_raw[0])
    print(compute_metrics(preds_raw, refs)) # {'format_rate': 1.0, 'accuracy': 0.021148036253776436, 'count': 331}

    # Chain-of-Thoughts - BATCHING
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    BATCH_SIZE = 4
    preds_raw = []

    for start in tqdm(range(0, len(ds), BATCH_SIZE)):
        batch_idxs = list(range(start, min(start + BATCH_SIZE, len(ds))))
        preds_raw.extend(generate_batch(batch_idxs, n_examples=3))
    refs = [r["verifiable_answer"] for r in ds]
    print(compute_metrics(preds_raw, refs))

    # Chain-of-Thoughts - BATCHING WITH SOLUTIONS
    BATCH_SIZE = 4
    preds_raw = []
    for start in tqdm(range(0, len(ds), BATCH_SIZE)):
        batch_idxs = list(range(start, min(start + BATCH_SIZE, len(ds))))
        preds_raw.extend(generate_batch(
            batch_idxs,
            n_examples=3,
            do_sample=True,
            sol=True))
    refs = [r["verifiable_answer"] for r in ds]
    print("Prediction:\n", preds_raw[0])
    print(compute_metrics(preds_raw, refs))

    # Reasoning
    BATCH_SIZE = 4
    prompts = []
    number_of_prompts = ds.shape[0]
    for i in range(number_of_prompts):
        r = ds[i]
        prompts.append(build_prompt(r["question"]))
    preds_raw = generate_reasoning(prompts, batch_size=BATCH_SIZE)
    refs = [r["verifiable_answer"] for r in ds]
    print("Prediction:\n", preds_raw[0])
