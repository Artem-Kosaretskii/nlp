from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import torch

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    load_in_8bit=True, # будем использовать int8 при обучении
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(model,
                                        r=8, # ранг
                                        lora_alpha=16, # вес добавления адаптера
                                        # слои, к которым применяем
                                        target_modules=["q_proj", "k_proj", "v_proj"]
                                    )


examples = [
    {
        "prompt": "Explain why the sky is blue.",
        "chosen": "Because air molecules scatter short wavelengths of light more strongly than long ones, we see predominantly the blue part of the spectrum.",
    },
    {
        "prompt": "Give me some safe advice on storing passwords.",
        "chosen": "Use a password manager and enable two-factor authentication; don't repeat the same password on different websites.",
    },
]

def examples_to_messages(examples):
    data = {'messages': []}

    for example in examples:
        data['messages'].append([
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': example['chosen']}
        ])
    return Dataset.from_dict(data)

ds = examples_to_messages(examples)
ds = ds.map(lambda x: {'text': tokenizer.apply_chat_template(x['messages'], tokenize=False)})

config = SFTConfig(
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    max_length=512,
    num_train_epochs=3,
    report_to='none',
    logging_steps=1,
    save_strategy='no',
    dataset_text_field = "text",
    gradient_accumulation_steps=1
)

trainer = SFTTrainer(
    model,
    args=config,
    train_dataset=ds,
    processing_class=tokenizer,
)

trainer.train()