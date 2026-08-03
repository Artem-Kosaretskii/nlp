import os
import re
import pickle
import math
import torch
import inspect
import time
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import nltk
from nltk import tokenize
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, TrainerCallback
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from datasets import Dataset, DatasetDict
from dataclasses import dataclass

nltk.download('punkt')

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.residual_scale = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        bs = config.block_size
        self.register_buffer('bias', torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))
        self.flash_attn = config.flash_attn

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.residual_scale = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash_attn: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.test_prompts = ['Transformers are amazing']
        self.random_seed = 42
        self.max_gen_length = 128

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'residual_scale'):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1)) if labels is not None else None
        # return logits, loss
        return {'logits': logits, 'loss': loss}

    def generation(self, tokenizer, device):
        special_tokens = [tokenizer.init_kwargs['eos_token']]
        tokens = tokenizer(self.test_prompts, padding=True, padding_side='left')['input_ids']
        tokens = torch.tensor(tokens, dtype=torch.long)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(self.seed)
        while xgen.size(1) < self.max_gen_length:
            with torch.no_grad():
                outputs = self(xgen)
                logits, loss = outputs['logits'], outputs['loss']
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(tokens.shape[0]):
            tokens = xgen[i, :self.max_gen_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(decoded.replace(special_tokens[0], ' '))

    def configure_optimizers(self, weight_decay, learning_rate, device, master_process):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(decay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f'Using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class SaveCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        tokenizer = kwargs['processing_class']
        model = kwargs['model']
        m = kwargs['metrics']
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': kwargs['optimizer'].state_dict(),
            'lr_scheduler': kwargs['lr_scheduler'].state_dict(),
            'metrics': m,
            'log_history': state.log_history[len(state.log_history)-1]
        }
        torch.save(checkpoint, f"./results_pretrain/checkpoint_e{m['epoch']}_loss_{m['eval_loss']:.4f}")
        model.eval()
        model.generation(tokenizer, args._setup_devices)
        model.train()


def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["train"]


def raw_to_csv(raw_dataset_path, max_lat_chars, max_lat_words, double_p_1, double_p_2, min_sentence):

    ru_sentences = []
    lat_pattern = '[a-zA-Z]{' + str(max_lat_chars) + ',}'
    filenames = next(os.walk(raw_dataset_path), (None, None, []))[2]
    for filename in filenames:
        print(filename)
        with open(f'{raw_dataset_path}/{filename}', 'r') as f:
            text = f.read()
        sentences = tokenize.sent_tokenize(text)
        for sentence in sentences:
            if len(re.findall(lat_pattern, sentence)) < max_lat_words:
                sentence = re.sub(double_p_1, ' ', sentence)
                sentence = re.sub(double_p_2, '', sentence)
                if len(sentence) > min_sentence:
                    ru_sentences.append(sentence)

    df = pd.DataFrame(ru_sentences, columns=['train'])
    df.to_csv('dataset.csv', encoding='utf-8', sep=';')
    return 0

def get_chunked(df, tokenizer, chunk_size, limit=None):

    dataset = DatasetDict()
    chunked = {'input_ids': [], 'labels': []}
    residual = [0]
    ds_size = df.shape[0] if limit is None else limit
    for i in range(ds_size):
        print(i)
        tokenized = residual + tokenizer.encode(df.loc[i]['train']) + [0]
        chunk_number = len(tokenized) // chunk_size
        if len(tokenized) % chunk_size != 0:
            for j in range(chunk_number):
                chunked['input_ids'].append(tokenized[j * chunk_size: (j + 1) * chunk_size])
                chunked['labels'].append(tokenized[j * chunk_size + 1 : (j + 1) * chunk_size + 1])
        if len(tokenized) % chunk_size != 0:
            residual = tokenized[chunk_number * chunk_size:]
        else:
            residual = []
    chunked_dataset = Dataset.from_dict(chunked)
    dataset['train'] = chunked_dataset
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def collate_fn(batch):

    input_ids = []
    labels = []
    for i in range(len(batch)):
        input_ids.append(batch[i]['input_ids'])
        labels.append(batch[i]['labels'])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return {'input_ids': input_ids,'labels': labels}



def main(raw_convert=False, train_tokenizer=False, make_chunks=False):

    chunk_size = 512
    raw_dataset_path = 'RNVLM'
    max_lat_chars = 3
    max_lat_words = 1
    double_p_1 = '([\s\n]{1,}[\s]{1,})|(\n)|(;)'
    double_p_2 = '(,{2,})|(\.{2,})|(-{2,})'
    min_sentence = 16
    vocab_size = 4096
    n_layer = 4 #16
    n_head = 4 # 16
    n_embd = 128 # 1024
    random_seed = 42
    epochs = 128
    warmup_steps = epochs // 25
    weight_decay = 0.1
    learning_rate = 6e-4
    max_gen_length = 128
    batch_size = 16
    test_size = 0.05
    special_tokens = ["<|endoftext|>"]
    test_prompts = [
        "Все мысли, которые имеют огромные последствия",
        "Сила войска зависит от его духа",
        "Мысль о том, что он принес страдания",
        "Человек сознает себя свободным",
        "Что бы ни случилось, я всегда буду",
        "Любовь мешает смерти",
        "Нет, жизнь не кончена",
        "Всякая мысль, даже самая простая",
        "Война не любезность, а самое гадкое дело",
        "Чтобы жить честно"
    ]

    if raw_convert:
        raw_to_csv(raw_dataset_path, max_lat_chars, max_lat_words, double_p_1, double_p_2, min_sentence)
    df = pd.read_csv('dataset.csv', sep=';', encoding='utf-8')

    if train_tokenizer:
        dataset = Dataset.from_pandas(df)
        tokenizer_model = Tokenizer(models.BPE())
        tokenizer_model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer_model.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
        tokenizer_model.save("bpe_tokenizer.json")
    else:
        tokenizer_model = Tokenizer.from_file("bpe_tokenizer.json")
    tokenizer_model.decoder = decoders.ByteLevel()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        bos_token=special_tokens[0],
        eos_token=special_tokens[0],
        pad_token=special_tokens[0],
    )

    if make_chunks:
        dataset = get_chunked(df, tokenizer, chunk_size)
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset = dataset['train'].train_test_split(test_size=test_size, seed=random_seed)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=chunk_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        flash_attn=True
    )
    model = GPT(config)
    model.test_prompts = test_prompts
    model.seed = random_seed
    model.max_gen_length = max_gen_length

    training_args = TrainingArguments(
        torch_compile=True,
        output_dir='./results_pretrain',
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        optim='adamw_torch_fused',
        lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=4,
        # bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        eval_steps=10,
        remove_unused_columns=False,
        save_strategy='no',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
        processing_class=tokenizer,
        callbacks=[SaveCallback]
    )
    trainer.train()

    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, f'./results/final_checkpoint_{int(time.time())}.pt')
    print('Done')


if __name__ == '__main__':
    main()
