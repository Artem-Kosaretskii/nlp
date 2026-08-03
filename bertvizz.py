from transformers import AutoTokenizer, AutoModel
from bertviz import head_view
import torch


def analyze_attention(sentence, layer=0, head=0):

    model_name = 'cointegrated/rubert-tiny2'
    attention_model = AutoModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print(f"Sentence analyzing: {sentence}")
    print(f"Tokens: {tokens}")

    with torch.no_grad():
        outputs = attention_model(**inputs)
        attentions = outputs.attentions

    print(f"Layer number: {len(attentions)}")
    html = head_view(attentions, tokens, layer=layer, heads=[head], html_action='return')
    return html


def main():

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)

    sentence = "Иван Иванович Иванов работает в Газпроме."
    inputs = tokenizer.encode(sentence, return_tensors='pt')
    outputs = model(inputs)
    attention = outputs.attentions  # a list of matrices

    # head_view
    html0 = head_view(attention, tokenizer.convert_ids_to_tokens(inputs[0]), layer=0, heads=0, html_action='return')

    example_sentence = "Иван Петров работает в Google."
    html1 = analyze_attention(example_sentence)

    example_sentence = "Иван Иванович Иванов работает в Газпроме."
    html2 = analyze_attention(example_sentence, layer=2, head=3)


if __name__ == '__main__':
    main()
