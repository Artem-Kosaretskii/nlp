import time
from transformers import AutoModel, pipeline

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    return total_size_mb

def measure_inference_time(classifier, texts, num_runs=10):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        for text in texts:
            _ = classifier(text)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    avg_time_per_text = avg_time / len(texts)
    return avg_time_per_text * 1000


def main(size_test=False, time_test=False):

    models_to_test = [
        "distilbert-base-uncased",
        "cointegrated/rubert-tiny",
        "microsoft/MiniLM-L12-H384-uncased"
    ]

    test_texts = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "The service was okay, nothing special.",
        "Outstanding experience! Highly recommend!",
        "Poor customer support, took forever."
    ]

    if size_test:
        print('Model size test')
        print("=" * 40)

        for model_name in models_to_test:
            print(f"\nLoading {model_name}...")

            model = AutoModel.from_pretrained(model_name)

            size_mb = get_model_size_mb(model)

            num_params = sum(p.numel() for p in model.parameters())

            print(f"Size: {size_mb:.1f} МБ")
            print(f"Params: {num_params:.1f}")
            print("-" * 40)

    if time_test:
        print('Model time test')
        print("=" * 40)

        for model_name in models_to_test:
            print(f"\nSpeed test: {model_name}")
            print("=" * 50)
            classifier = pipeline("text-classification", model=model_name)
            avg_time = measure_inference_time(classifier, test_texts)
            print(f"Average text processing time: {avg_time:.3f} ms")


if __name__ == '__main__':
    main(True, True)

