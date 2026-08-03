import torch
import torch.distributed as dist
from datasets import load_dataset, tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from torch.amp import GradScaler, autocast
import os

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def demo_ddp(rank, world_size):
    ds = load_dataset('zloelias/lenta-ru', split="train[:1%]")
    tokenized_dataset = ds.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'topic'])
    data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(rank)

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
    setup(rank, world_size)

    model = DDP(model, device_ids=[rank])
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
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2  # GPU number
    torch.multiprocessing.spawn(demo_ddp, args=(world_size,), nprocs=world_size)
