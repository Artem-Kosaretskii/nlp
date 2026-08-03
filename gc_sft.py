from transformers import BitsAndBytesConfig
from peft import LoraConfig
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


def run_gc_sft(ds):
    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id)
    qconf = BitsAndBytesConfig(load_in_8bit=True)
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj"]
    )
    cfg = SFTConfig(
        output_dir="sft-int8",
        per_device_train_batch_size=1,
        logging_steps=1,
        max_length=1024,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        report_to='none',
        run_name='8bit',
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        quantization_config=qconf
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()


ds = load_first_k_examples(1000)
run_gc_sft(ds)
