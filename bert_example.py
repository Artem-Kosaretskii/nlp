from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

sentence = "Paris is the capital of [MASK]."
inputs = tokenizer(sentence, return_tensors="pt")

mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

with torch.no_grad():
    y_hat = model(inputs.input_ids)
logits = y_hat.logits[0, mask_index]
predicted_index = torch.argmax(torch.softmax(y_hat.logits[0, 6], dim=-1))
print(f'{" ".join(sentence.split(" ")[0: -1])} {tokenizer.decode(predicted_index).title()}')
