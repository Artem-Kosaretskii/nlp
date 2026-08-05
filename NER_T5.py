from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import DatasetDict, Dataset
from IPython.display import HTML, display
import torch
from typing import Any, List, Tuple, Dict
import evaluate
import json
import numpy as np
import html

seqeval = evaluate.load('seqeval')
model_name = 'google/mt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def visualize_tokens(tokens, tags):

    html_output = ""
    i = 0
    while i < len(tokens):
        tag = tags[i]

        if tag.startswith("B-"):
            ent_type = tag[2:]
            ent_tokens = [tokens[i]]

            j = i + 1
            while j < len(tokens) and tags[j] == f"I-{ent_type}":
                ent_tokens.append(tokens[j])
                j += 1

            ent_text = " ".join(ent_tokens)
            html_output += f"<span style='background-color: #ffd54f; padding:2px; margin:1px; border-radius:4px;'>{html.escape(ent_text)} <sub>{ent_type}</sub></span> "
            i = j
        else:
            html_output += html.escape(tokens[i]) + " "
            i += 1

    display(HTML(html_output))


def evaluate_t5_with_seqeval(trainer, dataset, raw_dataset, labels_list, tokenizer):
    predictions = trainer.predict(dataset, max_length=256)

    preds = np.where(predictions.predictions != -100, predictions.predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, )
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    true_labels = []
    pred_labels = []

    for i in range(len(dataset)):
        original_tokens = raw_dataset[i]["tokens"]
        true_tags = raw_dataset[i]["ner_tags"]
        true_labels.append(true_tags)

        pred_bio_tags = t5_output_to_bio(decoded_preds[i], original_tokens, labels_list)
        pred_labels.append(pred_bio_tags)

    t5_seqeval_results = seqeval.compute(predictions=pred_labels, references=true_labels)
    return t5_seqeval_results


@torch.no_grad
def t5_inference(sentence: str, model, tokenizer) -> List[str]:
    input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=256)
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return decoded_output


def extract_entities_intervals(tags: List[int]) -> List[Tuple[str, int, int]]:
    """
    returns: (label, start_idx, end_idx).
    """
    entities = []
    start, end, ent_type = None, None, None

    for i, label in enumerate(tags):

        if label == "O":
            if ent_type is not None:
                entities.append((ent_type, start, end))
                ent_type, start, end = None, None, None
        elif label.startswith("B-"):
            if ent_type is not None:
                entities.append((ent_type, start, end))
            ent_type = label[2:]
            start, end = i, i
        elif label.startswith("I-") and ent_type == label[2:]:
            end = i
        else:
            if ent_type is not None:
                entities.append((ent_type, start, end))
            ent_type, start, end = None, None, None

    if ent_type is not None:
        entities.append((ent_type, start, end))

    return entities


def make_target_from_entities(tokens: List[str], tags: List[int]) -> str:
    ents = extract_entities_intervals(tags)
    parts = []
    for ty, s, e in ents:
        text = ' '.join(tokens[s:e+1])
        parts.append(f'"{text}": "{ty}"')
    return "{" + (', '.join(parts) if parts else '') + "}"


def prepare_seq2seq_dataset(dataset):
    def _convert_split(split):
        records = []
        for ex in dataset[split]:
            input_text = " ".join(ex["tokens"])
            target_text = make_target_from_entities(ex["tokens"], ex["ner_tags"])
            records.append({"input_text": input_text, "target_text": target_text})
        return Dataset.from_list(records)

    out = {}
    for split in dataset.keys():
        out[split] = _convert_split(split)
    return DatasetDict(out)


def tokenize_seq2seq(batch):
    model_inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=256)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def t5_output_to_bio(decoded_text, tokens, label_list):
    ner_tags = ["O"] * len(tokens)
    entities = {}

    try:
        entities = eval(decoded_text)
    except:
        decoded_text = decoded_text.replace("{", "").replace("}", "")
        parts = [p.strip() for p in decoded_text.split(',') if p.strip()]
        for p in parts:
            if ':' in p:
                text, typ = p.split(':', 1)
                entities[text.strip()] = typ.strip()

    for entity_text, entity_type in entities.items():
        ent_toks = entity_text.split()
        for i in range(len(tokens) - len(ent_toks) + 1):
            window = tokens[i:i+len(ent_toks)]

            if [w.lower().strip('.,') for w in window] == [w.lower().strip('.,') for w in ent_toks]:
                b_label = f"B-{entity_type}"
                i_label = f"I-{entity_type}"
                if b_label in label_list:
                    ner_tags[i] = b_label
                    for j in range(1, len(ent_toks)):
                        if i + j < len(tokens):
                            ner_tags[i+j] = i_label if i_label in label_list else ner_tags[i+j]
                break
    return ner_tags


def main():

    with open("./medicine_dataset/train_v1.jsonl.txt", "r") as fp:
        train_ds = [json.loads(x) for x in fp.readlines()]
        train_ds = Dataset.from_list(train_ds)
    with open("./medicine_dataset/dev_v1.jsonl.txt", "r") as fp:
        dev_ds = [json.loads(x) for x in fp.readlines()]
        dev_ds = Dataset.from_list(dev_ds)
    with open("./medicine_dataset/test_v1.jsonl.txt", "r") as fp:
        test_ds = [json.loads(x) for x in fp.readlines()]
        test_ds = Dataset.from_list(test_ds)

    ner_dataset = DatasetDict()
    ner_dataset["train"] = train_ds
    ner_dataset["dev"] = dev_ds
    ner_dataset["test"] = test_ds
    labels_list = set()
    for x in ner_dataset["train"]:
        labels_list = labels_list.union(x["ner_tags"])
    labels_list = list(labels_list)
    test_sentence_tokens = ner_dataset['test'][0]["tokens"]
    test_sent = " ".join(test_sentence_tokens)

    seq2seq_dataset = prepare_seq2seq_dataset(ner_dataset)
    tokenized_dataset = seq2seq_dataset.map(tokenize_seq2seq, batched=True)
    print(seq2seq_dataset['train'][0])

    args = Seq2SeqTrainingArguments(
        output_dir="./t5_ner",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        logging_dir="./logs",
        report_to="none",
        save_strategy='no'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    raw_t5_output = '{"мазь": "Drugform"}'
    tags = t5_output_to_bio(raw_t5_output, test_sentence_tokens, labels_list)
    print("Converted tags: ", tags)

    raw_t5_output = t5_inference(test_sent, model, tokenizer)
    tags = t5_output_to_bio(raw_t5_output, test_sentence_tokens, labels_list)
    print("T5 output: ", raw_t5_output)
    print("Converted tags: ", tags)

    visualize_tokens(test_sentence_tokens, tags)

    num_samples = 100
    eval_t5_results = evaluate_t5_with_seqeval(trainer, tokenized_dataset['test'].select(range(num_samples)), ner_dataset['test'].select(range(num_samples)), labels_list, tokenizer)
    print(eval_t5_results)


if __name__ == '__main__':
    main()
