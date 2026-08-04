import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
model.eval()

def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings

def bertscore_pair(hyp, ref):

    h = get_embeddings(hyp)   # [len_h, d]
    r = get_embeddings(ref)   # [len_r, d]

    h = torch.nn.functional.normalize(h, p=2, dim=1)
    r = torch.nn.functional.normalize(r, p=2, dim=1)

    sim = torch.matmul(h, r.T)  # [len_h, len_r]

    # Precision: max r for each token
    P = sim.max(dim=1).values.mean().item()
    # Recall: max r for each token
    R = sim.max(dim=0).values.mean().item()
    # F1
    F1 = 2 * P * R / (P + R + 1e-8)

    return P, R, F1


def main():
    # Пример
    hyp = "Сегодня будет краткий дождь и прохладный ветер."
    ref = "Сегодня ожидается непродолжительный дождь и прохладный ветер."

    P, R, F1 = bertscore_pair(hyp, ref)
    print(f"P={P:.4f}, R={R:.4f}, F1={F1:.4f}")


if __name__ == '__main__':
    main()
