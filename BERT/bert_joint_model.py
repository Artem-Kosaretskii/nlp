from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, load_dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import nltk
import time
import re
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

class NerDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super(NerDataLoader, self).__init__(
            dataset=dataset,
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=shuffle
        )


class JointModel(nn.Module):
    def __init__(self, model_name, max_len, id2label, label2id, len_cls, use_uncertainty_weight=False):
        super(JointModel, self).__init__()
        self.model_name = model_name
        self.base = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        self.len_cls = len_cls
        self.dropout = nn.Dropout(0.1)
        self.linear_ner = nn.Linear(len(label2id), len(label2id))
        self.linear_cls = nn.Linear(len(label2id), self.len_cls)
        self.sigma_ner = nn.Parameter(torch.exp(torch.tensor(1)))
        self.sigma_cls = nn.Parameter(torch.exp(torch.tensor(1)))
        self.use_uncertainty_weight = use_uncertainty_weight

    def forward(self, input_ids, attention_mask, labels=None, cls_vec=None, token_type_ids=None):
        loss_out = None
        output = self.base(input_ids, attention_mask=attention_mask)
        x = self.dropout(output['logits'])
        b, t, c = x.shape
        cls_pooling = x[:, 0, :]
        ner_output = self.linear_ner(x).view(b, t, c)
        cls_output = self.linear_cls(cls_pooling)
        if labels is not None and cls_vec is not None:
            ner_loss = F.cross_entropy(ner_output.view(b * t, c), labels.view(b * t), ignore_index=-100)
            cls_loss = F.binary_cross_entropy(torch.sigmoid(cls_output), cls_vec.to(torch.float))
            if self.use_uncertainty_weight:
                loss_ner_term = torch.exp(-2.0 * torch.log(self.sigma_ner)) * ner_loss + torch.log(self.sigma_ner)
                loss_cls_term = torch.exp(-2.0 * torch.log(self.sigma_cls)) * cls_loss + torch.log(self.sigma_cls)
                loss_out = loss_ner_term + loss_cls_term
            else:
                loss_out = cls_loss + ner_loss
        return ner_output, cls_output, loss_out


def tokenize_and_align_labels(batch):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    labels = []
    for i, word_labels in enumerate(batch["tags"]):
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


def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_metrics(dataset, model, device):

    y_true_ner, y_pred_ner, y_true_cls, y_pred_cls = [], [], [], []
    for i, ex in enumerate(dataset):
        input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(device)
        attention_mask = torch.tensor([ex["attention_mask"]], dtype=torch.long).to(device)

        with torch.no_grad():
            out_ner, out_cls, loss = model(input_ids=input_ids, attention_mask=attention_mask)
            preds_ner = torch.argmax(out_ner, dim=-1).squeeze(0).cpu().tolist()
            preds_cls = torch.round(torch.sigmoid(out_cls)).squeeze(0).cpu().tolist()

        true_labels_ner, true_labels_cls = ex["labels"], ex["cls_vec"]
        filtered_true_ner, filtered_pred_ner = [], []

        for p, t in zip(preds_ner, true_labels_ner):
            if t == -100:
                continue
            filtered_true_ner.append(int(t))
            filtered_pred_ner.append(int(p))

        min_len = min(len(filtered_true_ner), len(filtered_pred_ner))
        if not (min_len == 0):
            y_true_ner.extend(filtered_true_ner[:min_len])
            y_pred_ner.extend(filtered_pred_ner[:min_len])

        min_len = min(len(true_labels_cls), len(preds_cls))
        if not (min_len == 0):
            y_true_cls.extend(true_labels_cls[:min_len])
            y_pred_cls.extend(preds_cls[:min_len])

        y_true_ner.extend(filtered_true_ner)
        y_pred_ner.extend(filtered_pred_ner)
        y_true_cls.extend(true_labels_cls)
        y_pred_cls.extend(preds_cls)

    ps_ner = precision_score(y_true_ner, y_pred_ner, average="macro", zero_division=0)
    rs_ner = recall_score(y_true_ner, y_pred_ner, average="macro", zero_division=0)
    f1_ner = f1_score(y_true_ner, y_pred_ner, average="macro", zero_division=0)
    ps_cls = precision_score(y_true_cls, y_pred_cls, average="micro", zero_division=0)
    rs_cls = recall_score(y_true_cls, y_pred_cls, average="micro", zero_division=0)
    f1_cls = f1_score(y_true_cls, y_pred_cls, average="micro", zero_division=0)

    return ps_ner, rs_ner, f1_ner, ps_cls, rs_cls, f1_cls


def inference(texts, model, device):
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        with torch.no_grad():
            input_ids = torch.tensor([tokenized[0].ids], dtype=torch.long).to(device)
            attention_mask = torch.tensor([tokenized[0].attention_mask], dtype=torch.long).to(device)
            out_ner, out_cls, loss = model(input_ids=input_ids, attention_mask=attention_mask)
            preds_ner = torch.argmax(out_ner, dim=-1).squeeze(0).cpu().tolist()[1:len(tokens)+1]
            preds_cls = torch.sigmoid(out_cls).squeeze(0).cpu().tolist()

            print(f'\n{text}')

            print('CLS probabilities:')
            for i in range(len(preds_cls)):
                if preds_cls[i] >= treshold:
                    print(f'{CLS_VEC[i]}: {preds_cls[i]}')

            print('Token predictions:')
            for i in range(len(preds_ner)):
                print(f'{tokens[i]}: {id2label[preds_ner[i]]}')


num_epochs = 50
warmup_steps = 2
max_lr = 7e-5
min_lr = max_lr * 0.7
batch_size = 2
max_len = 128
treshold = 0.9
device = torch.device('cuda')
base_model_name = 'cointegrated/rubert-tiny2'
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
data_collator = DataCollatorForTokenClassification(tokenizer)
ds = load_dataset('danasone/nerel')['train']
CLS_VEC = ['WORKPLACE', 'ALTERNATIVE_NAME', 'WORKS_AS', 'PARTICIPANT_IN', 'POINT_IN_TIME',
            'TAKES_PLACE_IN', 'HEADQUARTERED_IN', 'ORIGINS_FROM', 'LOCATED_IN', 'AGENT',
            'AGE_IS', 'HAS_CAUSE', 'PRODUCES', 'AWARDED_WITH', 'PART_OF', 'IDEOLOGY_OF',
            'MEMBER_OF', 'CONVICTED_OF', 'INANIMATE_INVOLVED', 'SUBEVENT_OF', 'SUBORDINATE_OF',
            'KNOWS', 'MEDICAL_CONDITION', 'PARENT_OF', 'PLACE_RESIDES_IN', 'OWNER_OF',
            'ABBREVIATION', 'FOUNDED_BY', 'ORGANIZES', 'PENALIZED_AS']
unique_labels = set()
for rec in ds:
    unique_labels.update(rec["tags"])
unique_labels.add("O")
label_list = sorted(unique_labels)
label2id = {lab: i for i, lab in enumerate(label_list)}
id2label = {i: lab for lab, i in label2id.items()}


# DATA
transformed_ds = []
i = 0
for rec in ds:
    i += 1
    tags = [label2id[t] for t in rec["tags"]]
    transformed_ds.append({
            "id": i,
            "tokens": rec['tokens'],
            "token_spans": rec['token_spans'],
            "tags": tags,
            "cls_vec": rec['cls_vec'],
            "text": rec['text'],
    })
full_ds = Dataset.from_list(transformed_ds)
split = full_ds.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": split["train"], "test": split["test"]})
tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['text', 'tokens', 'token_spans', 'tags', 'id'])
train_dataloader = NerDataLoader(tokenized_ds["train"], batch_size=batch_size, shuffle=True)
test_dataloader = NerDataLoader(tokenized_ds["test"], batch_size=batch_size, shuffle=True)


# TRAIN
model = JointModel(
    model_name=base_model_name,
    max_len=max_len,
    id2label=id2label,
    label2id=label2id,
    len_cls=len(CLS_VEC),
    use_uncertainty_weight=True
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

logs = {'train_loss': [], 'test_loss': [], 'test_f1_ner':[], 'test_f1_cls': []}
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        ner_out, cls_out, loss = model(**batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(epoch, warmup_steps, num_epochs, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch + 1} avg train loss: {avg_loss:.4f}, lr {lr:.7f}")
    logs['train_loss'].append(avg_loss)

    # Evaluation
    total_loss = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            ner_out, cls_out, loss = model(**batch)
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch + 1} avg test loss: {avg_loss:.4f}, lr {lr:.7f}")
    ps_ner, rs_ner, f1_ner, ps_cls, rs_cls, f1_cls = get_metrics(tokenized_ds['test'], model, device)
    logs['test_loss'].append(avg_loss)
    logs['test_f1_ner'].append(f1_ner)
    logs['test_f1_cls'].append(f1_cls)
    print(f'NER - Precision score: {ps_ner:.4f}, Recall score: {rs_ner:.4f}, F1 score: {f1_ner:.4f}')
    print(f'CLS - F1 score: {f1_cls:.4f}')

checkpoint_path = f'./model_ner_{base_model_name[:4]}_{int(time.time())}.pt'
checkpoint_dir = './'
checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'logs': logs}
torch.save(checkpoint, checkpoint_path)
pickle.dump(logs, open('logs.pkl', 'wb'))

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes[0, 0].plot(np.arange(1, len(logs['train_loss'])+1, 1), logs['train_loss'])
axes[0, 0].set(xlabel='epoch', ylabel='loss', title='train loss')
axes[0, 0].grid()
axes[0, 1].plot(np.arange(1, len(logs['train_loss'])+1, 1), logs['test_loss'])
axes[0, 1].set(xlabel='epoch', ylabel='loss', title='test loss')
axes[0, 1].grid()
axes[1, 0].plot(np.arange(1, len(logs['train_loss'])+1, 1), logs['test_f1_ner'])
axes[1, 0].set(xlabel='epoch', ylabel='f1 ner', title='f1 ner')
axes[1, 0].grid()
axes[1, 1].plot(np.arange(1, len(logs['train_loss'])+1, 1), logs['test_f1_cls'])
axes[1, 1].set(xlabel='epoch', ylabel='f1 cls', title='f1 cls')
axes[1, 1].grid()
plt.show()

#  INFERENCE
model = JointModel(
    model_name=base_model_name,
    max_len=max_len,
    id2label=id2label,
    label2id=label2id,
    len_cls=len(CLS_VEC),
    use_uncertainty_weight=True
)
checkpoint = torch.load(f'{checkpoint_dir}model_ner_{base_model_name[:4]}.pt', weights_only=False, map_location='cpu')
texts = [
    'Иван Иванов, сотрудник Газпрома, полетел в Париж 15 мая 2023 года',
    'Россия и Турция подписали в минувшую среду 12 мая межправительственное соглашение о взаимной отмене виз.',
    'Валерий Зорькин, председатель Конституционного суда, высказал сомнения в правильности решения о переезде суда в Петербург.',
    'В среду началась новая сессия Конгресса США. Вновь избранные конгрессмены были приведены к присяге.',
    '30 июля в Москве состоялось заседание российского правительства.',
    'Минсельхоз РФ запретил ввозить овощи и фрукты из Молдавии.',
    'Реформа системы корпоративного налогообложения в Новой Зеландии будет проведена не ранее, чем через два года.',
    'В начале июне в Турине состоятся выборы президента международной шахматной федерации ФИДЕ.',
    'Страны Европейского союза запретили въезд на свою территорию президенту Белоруссии Александру Лукашенко.'
]
model.load_state_dict(checkpoint['model'])
model.to(device)
with torch.inference_mode():
    model.eval()
    start = time.time()
    inference(texts, model, device)
    finish = time.time()

# QUANTIZATION
with torch.inference_mode():
    model.to(torch.device('cpu'))
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    start_q = time.time()
    inference(texts, model, torch.device('cpu'))
    finish_q = time.time()

print(f'Inference time non-quant: {finish-start}, with quantization {finish_q-start_q}')
