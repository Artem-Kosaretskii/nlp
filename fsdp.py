import os
import torch
import torch.distributed as dist
from datasets import load_dataset, tqdm
from torch import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


# инициализация процессов как в DDP
def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    tmp = torch.zeros(1, device=f"cuda:{local_rank}")
    del tmp
    torch.cuda.synchronize()

    if world_size > 1:
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                rank=global_rank,
                                world_size=world_size)

    return global_rank, world_size, local_rank


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# загрузка FSDP
def demo_fsdp(rank):
    num_epochs = 8
    batch_size = 16
    learning_rate = 5e-5
    ds = load_dataset('zloelias/lenta-ru', split="train[:1%]")
    tokenized_dataset = ds.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'topic'])
    data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    global_rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{rank}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    model = FSDP(model)
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