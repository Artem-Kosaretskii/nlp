import torch
import lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer

model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SST2LightningModule(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)



def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Trainer + DDP + mixed precision

model = SST2LightningModule(model_name)
ds = load_dataset('zloelias/lenta-ru', split="train[:1%]")
tokenized_dataset = ds.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'topic'])
data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding=True)

num_epochs = 8
batch_size = 16
learning_rate = 5e-5

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)
trainer = pl.Trainer(accelerator='gpu', devices=2, strategy='ddp', precision=16, max_epochs=3)

trainer.fit(model, train_dataloader)
