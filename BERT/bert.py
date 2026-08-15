import torch
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForSequenceClassification

assert torch.cuda.is_available()


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    text_a = 'The cat sat on the mat'
    text_b = 'It was very sleepy'

    inputs = tokenizer(text_a, text_b, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    is_next_prob = torch.softmax(logits, dim=1)[0][1].item()
    not_next_prob = torch.softmax(logits, dim=1)[0][0].item()

    print(f"IsNext prob: {not_next_prob:.4f}")
    print(f"NotNext prob: {is_next_prob:.4f}")
    print("\n" + "="*80 + "\n")

    text_a = 'The cat sat on the mat'
    text_b = 'The Eiffel Tower is located in Paris'

    inputs = tokenizer(text_a, text_b, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    is_next_prob = torch.softmax(logits, dim=1)[0][1].item()
    not_next_prob = torch.softmax(logits, dim=1)[0][0].item()

    print(f"IsNext prob: {not_next_prob:.4f}")
    print(f"NotNext prob: {is_next_prob:.4f}")
    print("\n" + "="*80 + "\n")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()

    # GPU check
    lengths = [128, 256, 512]

    for seq_len in lengths:
        x = torch.randint(0, model.config.vocab_size, (1, seq_len)).cuda()
        torch.cuda.reset_peak_memory_stats()
        _ = model(x)
        print(f"seq len {seq_len}: peak consumption ≈ {torch.cuda.max_memory_allocated()/1024**2:.0f} Mb")


if __name__ == 'main':
    main()
