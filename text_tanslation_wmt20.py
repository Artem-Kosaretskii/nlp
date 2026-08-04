import torch
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import json
import evaluate
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
mdl = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to("cuda")

def _ok(s: str, max_len: int) -> bool:
    return bool(s) and len(s) <= max_len


def _extract_translation(ex):
    tr = ex.get("translation (translation)")
    if isinstance(tr, str):
        try:
            tr = json.loads(tr)
        except Exception:
            tr = {"en": ex.get("en", ""), "ru": ex.get("ru", "")}
    en = tr.get("en", "")
    ru = tr.get("ru", "")
    return {"src": en, "tgt": ru}


def compute_bleu_chrf(hyp: list[str], ref: list[str]):
    bleu = evaluate.load('bleu')
    chrf = evaluate.load('chrf')
    bleu = bleu.compute(predictions=hyp, references=ref)
    chrf = chrf.compute(predictions=hyp, references=ref)
    return {"bleu": bleu['bleu'], "chrf": chrf['score']}


def load_wmt(max_len: int = 1000, test_size: float = 0.20, seed: int = 42):
    raw = load_dataset("yezhengli9/wmt20-en-ru")
    base = raw["train"].map(_extract_translation, remove_columns=raw["train"].column_names)
    base = base.filter(lambda ex: _ok(ex["src"], max_len) and _ok(ex["tgt"], max_len))

    tmp = base.shuffle(seed=seed).train_test_split(test_size=test_size, seed=seed)
    ds = DatasetDict({
        "train": tmp["train"],
        "test": tmp["test"],
    })
    return ds



def eval_batch(ds, N=100):
    @torch.no_grad()
    def translate_batch(texts: list[str], max_new_tokens: int = 128) -> list[str]:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(mdl.device) for k, v in enc.items()}
        out = mdl.generate(**enc, max_new_tokens=max_new_tokens)
        return tok.batch_decode(out, skip_special_tokens=True)

    val_src = [r["src"] for r in ds["test"]][:N]
    val_ref = [r["tgt"] for r in ds["test"]][:N]
    hyp = translate_batch(val_src, max_new_tokens=128)
    metrics = compute_bleu_chrf(hyp, val_ref)
    print({k: round(v, 2) for k, v in metrics.items()})


def main(test_helsnlp=False, test_mt5=True):

    ds = load_wmt()
    if test_helsnlp:
        eval_batch(ds, 100)
    if test_mt5:
        "loading MT5..."
        tok = AutoTokenizer.from_pretrained("google/mt5-small")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

        def make_tokenize_fn(tok, max_src=256, max_tgt=256):
            def _fn(batch):
                src = list(batch["src"])
                tgt = batch["tgt"]
                model_inputs = tok(src, max_length=max_src, truncation=True)
                # with tok.as_target_tokenizer():
                labels = tok(tgt, max_length=max_tgt, truncation=True)
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            return _fn

        print("Tokenizing...")
        tok_fn = make_tokenize_fn(tok)
        train_ds = ds["train"].map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
        val_ds = ds["test"].map(tok_fn, batched=True, remove_columns=ds["test"].column_names)
        collate = DataCollatorForSeq2Seq(tokenizer=tok, model=mdl)
        args = Seq2SeqTrainingArguments(
            'results',
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            logging_strategy='steps',
            logging_steps=1,
            predict_with_generate=True,
            generation_max_length=128,
            report_to="none",
            save_strategy='no',
            # bf16=True
        )
        print("Fine-tuning...")
        trainer = Seq2SeqTrainer(model=mdl,
                                 args=args,
                                 train_dataset=train_ds,
                                 eval_dataset=val_ds,
                                 # tokenizer=tok,
                                 data_collator=collate)
        trainer.train()
        print("Prediction and metrics...")
        res = trainer.predict(val_ds)
        preds = res.predictions
        preds[preds < 0] = 0
        hyp = [text.replace('<extra_id_0>', '').strip()
               for text in tok.batch_decode(preds, skip_special_tokens=True)]
        val_ref = [r["tgt"] for r in ds["test"]]
        print(compute_bleu_chrf(hyp, val_ref))


if __name__ == '__main__':
    main()
