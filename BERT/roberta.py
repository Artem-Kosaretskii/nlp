from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Loading a tokenizer and a model XLM-R for the sentiment analysis
model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Creating a pipeline for classification
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "This movie is absolutely fantastic! I loved every moment of it.",
    "Этот фильм просто ужасен, потратил время зря.",
    "¡Me encanta este producto! Es increíble y muy útil.",
    "I'm not sure about this book, it's okay I guess.",
    "Сервис отличный, всем рекомендую!",
    "No me gusta nada, muy decepcionante."
]

# Classification for each text
for i, text in enumerate(texts):
    result = classifier(text)
    print(f"Текст {i+1}: {text}")
    print(f"Тональность: {result[0]['label']} (уверенность: {result[0]['score']:.3f})")
    print("-" * 50)
