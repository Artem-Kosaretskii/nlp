from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

ds = load_dataset('zloelias/lenta-ru', split="train[:1%]")
model_name = 'sergeyzh/rubert-tiny-turbo'

tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = ds.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'topic'])
data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, device_map='auto')

num_epochs = 8
batch_size = 16
learning_rate = 5e-5

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()
model.train()
total_loss = 0.0
n_batches = 0
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
