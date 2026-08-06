import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import Levenshtein
from peft import LoraConfig, get_peft_model
# pip install protobuf
# pip install sentencepiece
# pip install tiktoken

def compute_cer(preds, refs):
    errors = 0
    lens = 0
    for p, r in zip(preds, refs):
        errors += Levenshtein.distance(p, r)
        lens += len(r)
    return round(errors / lens * 100, 2)


def compute_metrics_with_evaluate(preds, refs, squad_metric):
    ids = list(range(len(refs)))
    predictions = [{"id": str(i), "prediction_text": p} for i, p in zip(ids, preds)]
    references = [{"id": str(i), "answers": {"text": [r], "answer_start": [0]}} for i, r in zip(ids, refs)]
    squad_res = squad_metric.compute(predictions=predictions, references=references)
    cer = compute_cer(preds, refs)
    return {
        "EM": squad_res["exact_match"],
        "F1": squad_res["f1"],
        "CER": cer,
        "count": len(refs)
    }


def format_example(ex):
    q = ex["question"].strip()
    c = ex["context"].strip()
    y = ex["answers"]["text"][0].strip() if ex["answers"]["text"] else ""
    src = f"context: {c} \nquestion: {q}"
    tgt = y
    return {"input_text": src, "labels_text": tgt}


def tokenize(batch, tokenizer, max_src_len, max_tgt_len):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_src_len,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        batch["labels_text"],
        max_length=max_tgt_len,
        truncation=True,
        padding="max_length",
    )
    labels_ids = labels["input_ids"]
    labels_ids = [[(tid if tid != tokenizer.pad_token_id else -100) for tid in seq] for seq in labels_ids]
    model_inputs["labels"] = labels_ids
    return model_inputs


def main(lora=False):
    MODEL_ID = "ai-forever/ruT5-base"
    MAX_SRC_LEN = 512
    MAX_TGT_LEN = 64
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    squad_metric = evaluate.load("squad")
    raw = load_dataset("kuznetsoffandrey/sberquad")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    formatted_train = raw["train"].shard(num_shards=10, index=0).map(format_example)
    formatted_val = raw["validation"].shard(num_shards=10, index=0).map(format_example)

    cols_to_remove = [c for c in raw["train"].column_names if c not in ["input_text", "labels_text"]]
    formatted_train = formatted_train.remove_columns(cols_to_remove)
    formatted_val = formatted_val.remove_columns(cols_to_remove)

    tokenized_train = formatted_train.map(tokenize, batched=True, remove_columns=["input_text", "labels_text"], fn_kwargs={'tokenizer': tokenizer, 'max_src_len': MAX_SRC_LEN, 'max_tgt_len': MAX_TGT_LEN})
    tokenized_val = formatted_val.map(tokenize, batched=True, remove_columns=["input_text", "labels_text"], fn_kwargs={'tokenizer': tokenizer, 'max_src_len': MAX_SRC_LEN, 'max_tgt_len': MAX_TGT_LEN})

    print(f'Tokenized keys: {tokenized_train[0].keys()}')

    if not lora:

        args = Seq2SeqTrainingArguments(
            output_dir="rut5_base_full",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            num_train_epochs=3,
            logging_steps=1,
            eval_strategy="epoch",
            predict_with_generate=True,
            gradient_accumulation_steps=1,
            optim='adafactor',
            report_to='none',
            save_strategy='no'
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=collator,
        )

        trainer.train()

        res = trainer.predict(tokenized_val)
        preds = res.predictions
        preds[preds < 0] = 0
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.array(tokenized_val['labels'])
        labels[labels < 0] = 0
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(compute_metrics_with_evaluate(pred_texts, label_texts, squad_metric))

    else:

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q", "k", "v", ],
            task_type="SEQ_2_SEQ_LM",
        )

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            output_dir="rut5_base_full",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            num_train_epochs=3,
            logging_steps=1,
            eval_strategy="epoch",
            predict_with_generate=True,
            gradient_accumulation_steps=1,
            optim='adafactor',
            report_to = 'none',
            save_strategy = 'no'
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=collator,
        )

        trainer.train()

        res = trainer.predict(tokenized_val)
        preds = res.predictions
        preds[preds < 0] = 0
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.array(tokenized_val['labels'])
        labels[labels < 0] = 0
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(compute_metrics_with_evaluate(pred_texts, label_texts, squad_metric))


if __name__ == '__main__':

    main(lora=False)
