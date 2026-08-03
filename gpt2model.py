import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from torch.nn import CrossEntropyLoss

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=512,
    n_layer=4,
    n_head=4,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

gpt2 = GPT2Model(config)
embedding_weight = gpt2.get_input_embeddings().weight

lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
lm_head.weight = embedding_weight

texts = [
    "Hello, how are you?",
    "This is a test sentence"
]
print(f'Old pad token {tokenizer.pad_token}')
tokenizer.pad_token = tokenizer.eos_token
print(f'New pad token {tokenizer.pad_token}')
texts = [text + '<|endoftext|>' for text in texts]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
print(input_ids)
print(attention_mask)
print(tokenizer.batch_decode(input_ids))
outputs = gpt2(input_ids=input_ids, attention_mask=attention_mask)
hidden_states = outputs.last_hidden_state
logits = lm_head(hidden_states)
targets = input_ids[:, 1:].contiguous()
targets_attention_mask = attention_mask[:, 1:]
logits = logits[:, :-1, :].contiguous()
print(tokenizer.decode(input_ids[0, :-1]))
print(tokenizer.decode(targets[0]))
targets[targets_attention_mask == 0] = -100
print(targets)
logits = logits.view(-1, config.vocab_size)
targets = targets.view(-1)
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, targets)
print(loss.item())
