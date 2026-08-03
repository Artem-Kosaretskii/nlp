from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

num_epochs = 8
batch_size = 16
learning_rate = 5e-5
accumulation_steps = 4
ds = load_dataset('zloelias/lenta-ru', split="train[:1%]")
model_name = 'sergeyzh/rubert-tiny-turbo'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = ds.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'topic'])
data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, device_map='auto')

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):

        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        n_batches += 1

    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
    print(f"Epoch {epoch + 1} effective optimizer steps: {n_batches // accumulation_steps}")
