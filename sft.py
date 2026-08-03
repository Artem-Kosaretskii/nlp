from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from clearml import Task
from transformers import TrainerCallback, TrainerState, TrainerControl
import os

def load_first_k_examples(k=1000):
    ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train")
    ds_small = ds.select(range(k))
    ds_small = ds_small.rename_column("conversation", "messages")
    return ds_small


def run_classic_sft(ds):
    model_id = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    cfg = SFTConfig(
        output_dir="full_sft",
        per_device_train_batch_size=1,
        logging_steps=1,
        max_length=1024,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        report_to='clearml',
        run_name='SFT'
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

ds = load_first_k_examples()
run_classic_sft(ds)
