import torch
import re
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from corus import load_factru
from sklearn.metrics import precision_score, recall_score, f1_score

def whitespace_tokenize_with_offset(text: str):
    tokens = []
    spans = []
    for m in re.finditer(r'\S+', text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans

def map_object_type(obj_type: str) -> str:

    t = (obj_type or "").lower()
    if "person" in t or t in {"person", "name", "surname", "firstname", "patronymic"}:
        return "PER"
    if "org" in t or "organization" in t or "company" in t or "org_name" in t or "org_descr" in t:
        return "ORG"
    if "loc" in t or "location" in t or "geo" in t or "place" in t or "loc_name" in t:
        return "LOC"
    return "MISC"


def tokenize_and_align_labels(examples_batch):
    tokenized = tokenizer(
        examples_batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = []
    for i, word_labels in enumerate(examples_batch["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(word_labels[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

def get_flat_labels_and_preds_from_model(tokenized_split, model, device, max_samples=None):
    """
    tokenized_split: dataset split (list-like of examples with keys 'input_ids','attention_mask','labels')
    """
    y_true = []
    y_pred = []
    for i, ex in enumerate(tokenized_split):
        if max_samples is not None and i >= max_samples:
            break

        input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(device)
        attention_mask = torch.tensor([ex["attention_mask"]], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape (1, seq_len, num_labels)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()  # list длины seq_len

        true_labels = ex["labels"]  # список длиной seq_len; элементы -100 или id
        filtered_true = []
        filtered_pred = []
        for p, t in zip(preds, true_labels):
            if t == -100:
                continue
            filtered_true.append(int(t))
            filtered_pred.append(int(p))

        minlen = min(len(filtered_true), len(filtered_pred))
        if minlen == 0:
            continue
        y_true.extend(filtered_true[:minlen])
        y_pred.extend(filtered_pred[:minlen])

    return y_true, y_pred

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dir_path = "./factRuEval-2016/"
records = list(load_factru(dir_path))
print("Records number:", len(records))
for i in range(50):
    print(records[i].text[:128])

unique_labels = set()
for record in records:
    for obj in record.objects:
        unique_labels.add(obj.type)
print("Unique labels:\n", unique_labels)

examples = []

for rec in records:
    text = rec.text
    tokens, token_spans = whitespace_tokenize_with_offset(text)
    token_labels = ["O"] * len(tokens)
    for obj in rec.objects:
        base_type = map_object_type(obj.type)
        for span in obj.spans:
            span_start = span.start
            span_end = span.stop
            overlapping_idxs = []
            for i, (t_start, t_end) in enumerate(token_spans):
                if not (t_end <= span_start or t_start >= span_end):
                    overlapping_idxs.append(i)
            if not overlapping_idxs:
                continue
            for j, tok_idx in enumerate(overlapping_idxs):
                if token_labels[tok_idx] != "O":
                    continue
                prefix = "B" if j == 0 else "I"
                token_labels[tok_idx] = f"{prefix}-{base_type}"

    examples.append({"id": rec.id, "text": rec.text, "tokens": tokens, "tags": token_labels})

print(f"Examples length: {len(examples)}")
print("Example tokens/tags:", examples[2]["tokens"][:20], examples[2]["tags"][:20])

unique_labels = set()
for ex in examples:
    unique_labels.update(ex["tags"])
unique_labels.add("O")
label_list = sorted(unique_labels)
label2id = {lab: i for i, lab in enumerate(label_list)}
id2label = {i: lab for lab, i in label2id.items()}

for ex in examples:
    ex["tags"] = [label2id[t] for t in ex["tags"]]

full_ds = Dataset.from_list(examples)
split = full_ds.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": split["train"], "test": split["test"]})
print(dataset)

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["text", "tokens", "tags", "id"]
)

print(tokenized_dataset)
print(tokenized_dataset["train"][0])

data_collator = DataCollatorForTokenClassification(tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)
print("Train samples:", len(tokenized_dataset["train"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
model.to(device)
model.eval()

y_true, y_pred = get_flat_labels_and_preds_from_model(tokenized_dataset["test"], model, device, max_samples=200)

print("Samples used (token-level):", len(y_true))
print("Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
print("F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))

num_epochs = 20
learning_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_dataloader):.4f}")

model.eval()
with torch.no_grad():
    y_true_ft, y_pred_ft = get_flat_labels_and_preds_from_model(tokenized_dataset["test"], model, device, max_samples=200)

print("After fine-tuning:")
print("Precision:", precision_score(y_true_ft, y_pred_ft, average="macro", zero_division=0))
print("Recall:   ", recall_score (y_true_ft, y_pred_ft, average="macro", zero_division=0))
print("F1:       ", f1_score   (y_true_ft, y_pred_ft, average="macro", zero_division=0))

print(f"Delta F1: {f1_score(y_true_ft, y_pred_ft, average='macro') - f1_score(y_true, y_pred, average='macro'):.4f}")
