import nltk
from typing import List, Tuple
from transformers import pipeline

nltk.download('punkt')
nltk.download('punkt_tab')


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    tokens = nltk.word_tokenize(text)
    offsets = []
    pos = 0
    for tok in tokens:
        start = text.find(tok, pos)
        end = start + len(tok)
        offsets.append((tok, start, end))
        pos = end

    return offsets


def spans_to_bio(tokens_off: List[Tuple[str, int, int]], spans: List[Tuple[int, int, str]]) -> List[str]:

    bio = ['O'] * len(tokens_off)

    for span_start, span_end, label in spans:
        first_token_in_span = True
        for i, (token, t_start, t_end) in enumerate(tokens_off):

            if t_end <= span_start or t_start >= span_end:
                continue

            if first_token_in_span:
                bio[i] = f"B-{label}"
                first_token_in_span = False
            else:
                bio[i] = f"I-{label}"
    return bio


def bio_to_bioes(bio: List[str]) -> List[str]:
    bioes = []
    n = len(bio)

    for i, tag in enumerate(bio):
        if tag == 'O':
            bioes.append('O')
            continue
        prefix, label = tag.split('-', 1)

        if prefix == 'B':
            if i + 1 < n and bio[i + 1] == f'I-{label}':
                bioes.append(f'B-{label}')
            else:
                bioes.append(f'S-{label}')

        elif prefix == 'I':
            if i + 1 < n and bio[i + 1] == f'I-{label}':
                bioes.append(f'I-{label}')
            else:
                bioes.append(f'E-{label}')
        else:
            bioes.append(tag)

    return bioes


def merge_entities(ner_results):
    entities = []
    current = None
    for res in ner_results:
        word = res['word']
        label = res['entity']  # B-PER, I-PER, O ..
        ent_type = label.split('-')[-1] if label != 'O' else None

        if word.startswith('##'):
            word = word[2:]
            if entities:
                entities[-1]['word'] += word
                entities[-1]['end'] = res['end']
            continue

        if label.startswith('B-') or label.startswith('S-'):
            entities.append({'word': word, 'type': ent_type, 'start': res['start'], 'end': res['end']})

        elif label.startswith('I-') or label.startswith('E-'):
            if entities and entities[-1]['type'] == ent_type:
                entities[-1]['word'] += ' ' + word
                entities[-1]['end'] = res['end']
            else:
                entities.append({'word': word, 'type': ent_type, 'start': res['start'], 'end': res['end']})
    return entities

def compute_f1(predicted, actual):
    pred_set = set(predicted)
    actual_set = set(actual)
    true_positives = len(pred_set & actual_set)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(pred_set)
    recall = true_positives / len(actual_set)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def main():

    result = None

    # Test
    test_text = "Иван Петров работает"
    result = tokenize_with_offsets(test_text)
    print("Result:")
    for token, start, end in result:
        print(f"'{token}' -> [{start}-{end})")

    text = "Иван Иванов работает в Яндексе"
    spans = [(0, 11, "PER"), (23, 29, "ORG")]  # "Иван Иванов" и "Яндекс"
    tokens_off = tokenize_with_offsets(text)
    bio_tags = spans_to_bio(tokens_off, spans)
    print("Result BIO:")
    for (token, start, end), bio_tag in zip(tokens_off, bio_tags):
        print(f"{token:12} -> {bio_tag}")

    # Full test
    text = "Иван Иванович Петров работает в Google"
    spans = [(0, 20, "PER"), (31, 37, "ORG")]  # длинная персона и короткая организация

    tokens_off = tokenize_with_offsets(text)
    bio_tags = spans_to_bio(tokens_off, spans)
    bioes_tags = bio_to_bioes(bio_tags)

    print("BIO vs BIOES:")
    for (token, _, _), bio_tag, bioes_tag in zip(tokens_off, bio_tags, bioes_tags):
        print(f"{token:12} {bio_tag:8} -> {bioes_tag:8}")


    # compute metrics
    pred = [(0, 4, "PER"), (17, 21, "LOC")]
    actual = [(0, 4, "PER"), (17, 26, "LOC")]
    print(compute_f1(pred, actual))  # 0.5

    raw_ner = [
        {'word': 'Газ', 'entity': 'B-ORG', 'score': 0.95, 'start': 10, 'end': 13},
        {'word': '##проме', 'entity': 'I-ORG', 'score': 0.93, 'start': 13, 'end': 18},
        {'word': '-', 'entity': 'O', 'score': 0.00, 'start': 18, 'end': 19},
        {'word': 'он', 'entity': 'O', 'score': 0.00, 'start': 20, 'end': 22}
    ]
    merged = []
    for res in raw_ner:
        word = res['word']
        if word.startswith('##'):
            merged[-1]['word'] += word[2:]
            merged[-1]['end'] = res['end']
        else:
            merged.append({'word': word, 'entity': res['entity'], 'start': res['start'], 'end': res['end']})
    print(merged)


    ner_pipeline = pipeline("ner", model="nesemenpolkov/msu-wiki-ner", tokenizer="nesemenpolkov/msu-wiki-ner")
    sentence = "Иван Иванович Иванов работает в Газпроме."
    raw_results = ner_pipeline(sentence)
    print(raw_results)

    merged = merge_entities(raw_results)
    print("Merged entities:", merged)


    return result

if __name__ == '__main__':
    main()

