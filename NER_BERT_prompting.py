from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
from datasets import DatasetDict, Dataset
from IPython.display import HTML, display
from typing import Any, List, Tuple, Dict
import evaluate
import json
import numpy as np
import html

seqeval = evaluate.load('seqeval')


def convert_pipeline_output_to_bio(pipeline_output: List[Dict], tokens: List[str], label_list: List[str]) -> List[str]:
    tags = ["O"] * len(tokens)
    token_char_indices = []
    current_char_idx = 0
    for token in tokens:
        token_char_indices.append((current_char_idx, current_char_idx + len(token)))
        current_char_idx += len(token) + 1

    for entity in pipeline_output:
        start_char = entity['start']
        end_char = entity['end']
        entity_type = entity['entity_group']

        start_token_idx = -1
        end_token_idx = -1

        for i, (token_start_char, token_end_char) in enumerate(token_char_indices):
            if start_char >= token_start_char and start_char < token_end_char:
                 start_token_idx = i
            if end_char > token_start_char and end_char <= token_end_char:
                 end_token_idx = i

        if start_token_idx != -1 and end_token_idx != -1:
            tags[start_token_idx] = f"B-{entity_type}"
            for i in range(start_token_idx + 1, end_token_idx + 1):
                tags[i] = f"I-{entity_type}"
    return tags


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


def convert_preds_to_labels(predictions: np.ndarray, label_ids: np.ndarray, label_list: List[str]) -> Tuple[List[List[str]], List[List[str]]]:

    if predictions.ndim == 3:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions
    true_labels = []
    pred_labels = []
    for pred_row, label_row in zip(preds, label_ids):
        tl_row = []
        pl_row = []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            tl_row.append(label_list[l])
            pl_row.append(label_list[p])
        true_labels.append(tl_row)
        pred_labels.append(pl_row)
    return pred_labels, true_labels


def tokenize_and_align_labels(batch, tokenizer, labels_list):

    tokenized_inputs = tokenizer(batch['tokens'],
                                 is_split_into_words=True,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=128,
                                 return_tensors=None)
    all_labels = []
    for i, labels in enumerate(batch['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_list.index(labels[word_idx]))
            else:
                if labels[word_idx] == "O":
                    label_ids.append(labels_list.index(labels[word_idx]))
                else:
                    label_ids.append(labels_list.index("I-" + labels[word_idx].split("-")[1]))
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs['labels'] = all_labels
    return tokenized_inputs


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

    for i in range(3):
        ex = ner_dataset['train'][i]
        print('\nExample', i)
        visualize_tokens(ex['tokens'], ex['ner_tags'])


    def compute_metrics_trainer(eval_pred: Any) -> Dict[str, Any]:

        predictions, label_ids = eval_pred
        pred_labels, true_labels = convert_preds_to_labels(predictions, label_ids, labels_list)
        result = seqeval.compute(predictions=pred_labels, references=true_labels)

        return {
            'precision': result.get('precision', None) or result.get('overall_precision'),
            'recall': result.get('recall', None) or result.get('overall_recall'),
            'f1': result.get('f1', None) or result.get('overall_f1'),
        }

    model_name = 'bert-base-cased'
    num_labels = len(labels_list)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenized_ds = ner_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "labels_list": labels_list})
    args = TrainingArguments(
        output_dir='bert-ner',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",
        save_strategy='no'
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['dev'],
        processing_class=tokenizer,
        compute_metrics=compute_metrics_trainer,
    )
    trainer.train()

    model.config.label2id = {l: i for i, l in enumerate(labels_list)}
    model.config.id2label = {i: l for i, l in enumerate(labels_list)}
    p = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    sent = " ".join(ner_dataset['test'][0]["tokens"])
    print(p(sent))

    test_sentence_tokens = ner_dataset['test'][0]["tokens"]
    test_sent = " ".join(test_sentence_tokens)
    pipeline_result = p(test_sent)
    bio_tags = convert_pipeline_output_to_bio(pipeline_result, test_sentence_tokens, labels_list)
    print("Tokens:", test_sentence_tokens)
    print("BIO Tags:", bio_tags)

    visualize_tokens(test_sentence_tokens, bio_tags)
    eval_bert_finetune_metrics = trainer.evaluate(tokenized_ds['test'])
    print(eval_bert_finetune_metrics)


if __name__ == '__main__':
    main()

