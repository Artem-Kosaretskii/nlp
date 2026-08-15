from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from dotenv import load_dotenv


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output


def process_inputs(pairs, model, tokenizer, max_length, prefix_tokens, suffix_tokens):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, model, token_true_id, token_false_id):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def rerank_example(model_name, query, candidates):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")
    reranked = rerank(query, candidates, tokenizer, model)
    print(reranked)


def rerank(query, candidates, tokenizer, model):
    pairs = [(query, cand) for cand in candidates]
    inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits).flatten().tolist()
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)


def main(do_example=False):
    load_dotenv()
    if do_example:
        rerank_example(
            model_name="cross-encoder/stsb-roberta-base",
            query="How to choose a laptop for work?",
            candidates=[
            "2023 rating of best laptops for office work",
            "Comparison of Intel Core i5 vs i7 processors",
            "How to improve performance of an old laptop",
            "Optimal laptop specifications for programmers",
            "Difference between SSD and HDD drives",
            "10 common mistakes when buying a laptop",
            "How to connect a laptop to a TV",
            "Best budget laptops under 50,000 rubles",
            "What graphics card is needed for graphic design work",
            "How to extend laptop battery life",
            ]
        )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = ['How long is the flight to Mars?', 'Does corn grow in the shade?']

    documents = [
        "The flight time to Mars depends on many factors, including the trajectory, the ship's speed, and the planetary alignment. On average, the flight could take between 6 and 9 months. The fastest way to reach Mars is estimated to take about 70-80 days, but would require a significant amount of fuel.",
        "Corn doesn't grow well in the shade. It requires sufficient sunlight for normal development and fruiting.",
        "Corn care includes watering, hoeing, weeding, fertilizing, and removing side shoots. It's important to provide the corn with sufficient moisture, especially during flowering and ear formation, and to keep the soil loose and weed-free.",
    ]

    pairs = []
    for q in queries:
        for d in documents:
            pairs.append(format_instruction(task, q, d))

    inputs = process_inputs(pairs, model, tokenizer, max_length, prefix_tokens, suffix_tokens)
    scores = compute_logits(inputs, model, token_true_id, token_false_id)

    print("scores: ", [round(x, 6) for x in scores])




if __name__ == '__main__':
    main()
