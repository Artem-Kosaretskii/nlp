from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pairs = [
    {
        "premise_en": "The bank by the river was steep and muddy.",
        "hypothesis_en": "The financial institution was near the water.",
        "premise_ru": "Берег у реки был крутым и грязным.",
        "hypothesis_ru": "Финансовое учреждение было рядом с водой."
    },
    {
        "premise_en": "The doctor advised the lawyer because she felt unwell.",
        "hypothesis_en": "The lawyer was feeling unwell.",
        "premise_ru": "Доктор дал совет адвокату, потому что она плохо себя чувствовала.",
        "hypothesis_ru": "Адвокат плохо себя чувствовал."
    },
    {
        "premise_en": "Despite initial promising results announced in the press conference, the drug failed in clinical trials because of unexpected side effects observed in elderly patients.",
        "hypothesis_en": "Side effects caused the drug to fail.",
        "premise_ru": "Несмотря на первоначальные многообещающие результаты, объявленные на пресс-конференции, препарат провалился в клинических испытаниях из-за неожиданных побочных эффектов, наблюдаемых у пожилых пациентов.",
        "hypothesis_ru": "Побочные эффекты стали причиной провала препарата."
    }
]

model.eval()

for pair in pairs:
    inputs = tokenizer(pair['premise_en'], pair['hypothesis_en'], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    probs = probs[0].cpu().numpy()
    labels_ru = ["aftermath", "neutral", "contradiction"]
    pred_label_ru = labels_ru[probs.argmax()]

    print(f"Premise: {pair['premise_ru']}")
    print(f"Hypothesis: {pair['hypothesis_ru']}")
    print(f"Prediction: {pred_label_ru} (sureness: {probs.max():.2%})")
    print(f"Probs: [aftermath: {probs[0]:.2%}, neutral: {probs[1]:.2%}, contradiction: {probs[2]:.2%}]")
    print("-" * 80)
