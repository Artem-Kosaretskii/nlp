import torch
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


class SaveCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        tokenizer = kwargs['processing_class']
        model = kwargs['model']
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': kwargs['optimizer'].state_dict(),
            'lr_scheduler': kwargs['lr_scheduler'].state_dict(),
            'metrics': kwargs['metrics'],
        }
        torch.save(checkpoint, f"./sft_results/checkpoint_e{kwargs['metrics']['epoch']}")
        questions_rus = ["сколько планет в нашей солнечной системе?","расскажи стих","когда собирать крыжовник?","Как быстро выучить новый язык?"]
        model.eval()
        for i in range(len(questions_rus)):
            messages = [{"role": "user", "content": f"{questions_rus[i]}"}]
            start = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{questions_rus[i]}<|im_end|>\n<|im_start|>'
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            print(tokenizer.batch_decode(outputs)[0].replace(start, ''))
        model.train()

batch_size = 16
logging_steps = 8
max_length = 64
epochs = 16
weight_decay = 0.01
learning_rate = 6e-5
test_size = 0.05
random_seed = 42
max_new_tokens = 128
device = 'cuda'

model_id = 'Qwen/Qwen2.5-0.5B'
dataset_id = 'd0rj/alpaca-cleaned-ru'
dataset = load_dataset(dataset_id)['train']

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)
questions_rus = [
    "сколько планет в нашей солнечной системе?",
    "расскажи стих",
    "когда собирать крыжовник?",
    "Как быстро выучить новый язык?"
]


def preprocess_function(example):
    return {
        "prompt": [
            {"role": "system", "content": example['input']},
            {"role": "user", "content": example['instruction']}
        ],
        "completion": [
            {"role": "assistant", "content": example['output']}
        ],
    }



def generate(model, tokenizer, prompt, max_new_tokens=16):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0])


dataset = dataset.map(preprocess_function, remove_columns=['input', 'instruction', 'output'])
dataset = dataset.train_test_split(test_size=test_size, seed=random_seed)

args = SFTConfig(
    torch_compile=True,
    optim='adamw_torch_fused',
    output_dir="./sft_results",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    logging_steps=logging_steps,
    max_length=max_length,
    num_train_epochs=epochs,
    lr_scheduler_type="cosine",
    learning_rate=learning_rate,
    # warmup_steps=warmup_steps,
    # weight_decay=weight_decay,
    report_to="none",#'clearml',
    run_name='SFT',
    bf16=True,
    eval_strategy="epoch",
    eval_steps=10,
    save_strategy='no'
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=peft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    callbacks=[SaveCallback]
)
trainer.train()
