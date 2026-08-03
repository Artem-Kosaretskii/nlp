from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def examples_to_messages(examples):
    data = {'chosen': [], 'rejected': []}
    for example in examples:
        data['chosen'].append([
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': example['chosen']}
        ])
        data['rejected'].append([
            {'role': 'user', 'content': example['prompt']},
            {'role': 'assistant', 'content': example['rejected']}
        ])
    return Dataset.from_dict(data)


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

config = DPOConfig(
    beta=0.1,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    max_length=512,
    # max_prompt_length=256,
    num_train_epochs=3,
    report_to='none',
    logging_steps=1,
    save_strategy='no'
)

examples = [
    {
        "prompt": "Explain why the sky is blue.",
        "chosen": "Because air molecules scatter short wavelengths of light more strongly than long ones, we see predominantly the blue part of the spectrum.",
        "rejected": "Because that's how nature wanted it, and it's simply beautiful.",
    },
    {
        "prompt": "Give me some safe advice on storing passwords.",
        "chosen": "Use a password manager and enable two-factor authentication; don't repeat the same password on different websites.",
        "rejected": "Write down all your passwords in a note on your phone so you don't forget them.",
    },
]


ds = examples_to_messages(examples)
trainer = DPOTrainer(
    model,
    ref_model,
    args=config,
    train_dataset=ds,
    processing_class=tokenizer,
)

trainer.train()
