from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
from datasets import DatasetDict, Dataset, tqdm
from IPython.display import HTML, display
import torch
import re
from typing import Any, List, Tuple, Dict
import evaluate
import json
import numpy as np
import html

seqeval = evaluate.load('seqeval')

@torch.no_grad
def llm_inference(model, tokenizer, sentence):
    prompt = 'Extract entities (ADR, DI, Drugclass, Drugform, Drugname, Finding) from the sentence. Return as JSON list of {"text":..., "type":...}.\n\n'+sentence+'\n\n ```json\n'
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content


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


def schema_to_bio(tokens, parsed, label_list):
    ner_tags = [label_list.index('O')] * len(tokens)
    for ent in parsed:
        text = ent.text
        typ = str(ent.type)
        if not text or not typ:
            continue
        ent_toks = text.split()
        #
        for i in range(len(tokens) - len(ent_toks) + 1):
            window = tokens[i:i+len(ent_toks)]
            if [w.lower().strip('.,') for w in window] == [w.lower().strip('.,') for w in ent_toks]:
                b_label = f'B-{typ}'
                i_label = f'I-{typ}'
                if b_label in label_list:
                    ner_tags[i] = label_list.index(b_label)
                    for j in range(1, len(ent_toks)):
                        ner_tags[i+j] = label_list.index(i_label) if i_label in label_list else ner_tags[i+j]
                break
    return ner_tags


def prompt_output_to_bio(tokens: List[str], prompt_output: str, label_list: List[str]) -> List[int]:
    ner_tags = [label_list.index('O')] * len(tokens)
    parsed = None
    pattern = r"```json\n([\s\S]*?)\n```"
    matches = re.findall(pattern, prompt_output)
    try:
        parsed = json.loads(matches[0])
    except Exception:
        parts = [p.strip() for p in prompt_output.replace(';', '\n').split('\n') if p.strip()]
        parsed = []
        for p in parts:
            if ':' in p:
                left, right = p.split(':', 1)
                parsed.append({'text': left.strip().strip('"'), 'type': right.strip()})

    if isinstance(parsed, list):
        for ent in parsed:
            text = ent.get('text') if isinstance(ent, dict) else None
            typ = ent.get('type') if isinstance(ent, dict) else None
            if not text or not typ:
                continue
            ent_toks = text.split()
            for i in range(len(tokens) - len(ent_toks) + 1):
                window = tokens[i:i+len(ent_toks)]
                if [w.lower().strip('.,') for w in window] == [w.lower().strip('.,') for w in ent_toks]:
                    b_label = f'B-{typ}'
                    i_label = f'I-{typ}'
                    if b_label in label_list:
                        ner_tags[i] = label_list.index(b_label)
                        for j in range(1, len(ent_toks)):
                            ner_tags[i+j] = label_list.index(i_label) if i_label in label_list else ner_tags[i+j]
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

    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model.config.label2id = {l: i for i, l in enumerate(labels_list)}
    model.config.id2label = {i: l for i, l in enumerate(labels_list)}
    sent = " ".join(ner_dataset['test'][0]["tokens"])
    print(f'Sentence: {sent}')

    test_sentence_tokens = ner_dataset['test'][0]["tokens"]
    test_sent = " ".join(test_sentence_tokens)
    print("Tokens:", test_sentence_tokens)

    raw_output = llm_inference(model, tokenizer, test_sent)
    print('LLM output:', raw_output)

    bio_tags = prompt_output_to_bio(test_sentence_tokens, raw_output, labels_list)
    print('Converted tags:', [labels_list[tag] for tag in bio_tags])

    visualize_tokens(test_sentence_tokens, [labels_list[tag] for tag in bio_tags])

if __name__ == '__main__':
    main()