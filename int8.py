import torch
from transformers import AutoModelForSequenceClassification
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.train()


qconfig = get_default_qat_qconfig("fbgemm")
model.qconfig = qconfig

# fake-quant/observers
prepare_qat(model, inplace=True)
train_loader = []

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for epoch in range(1, 4):
    for batch in train_loader:

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
model_int8 = convert(model.cpu())
