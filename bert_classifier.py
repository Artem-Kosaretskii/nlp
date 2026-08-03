from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

def main():
    lr = 2e-5
    epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_range = 10000
    test_range = 5000
    imdb = load_dataset("imdb")

    train_dataset = imdb['train'].shuffle(seed=42).select(range(train_range))
    test_dataset = imdb['test'].shuffle(seed=42).select(range(test_range))

    train_loader = create_dataloader(train_dataset)
    test_loader = create_dataloader(test_dataset)
    config = BertConfig.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', config=config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} [Eval]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_samples

        print(f"\nEpoch {epoch} results:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Eval : loss={val_loss:.4f}, acc={val_acc:.4f}\n")

    print(f"Final Eval Accuracy: {val_acc:.4f}")

def tokenize_batch(batch):
    texts = [x['text'] for x in batch]
    labels = [x['label'] for x in batch]
    encoding = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': torch.tensor(labels)
    }

def create_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=tokenize_batch)


if __name__ == '__main__':
    main()
