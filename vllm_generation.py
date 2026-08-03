from string import Template
import gc
import torch
from vllm import LLM, SamplingParams
import time
import copy
from dataclasses import dataclass
from typing import List
import textwrap
import random
import re
from math import log
from collections import Counter
from difflib import SequenceMatcher
import datasets
from datasets import load_dataset
import pandas as pd


def check_h1_tags(description):
    h1_pattern = r'<h1>(.*?)</h1>'
    matches = re.findall(h1_pattern, description, re.DOTALL | re.IGNORECASE)
    if len(matches) != 1:
        return 0
    h1_content = matches[0].strip()
    if not h1_content or len(h1_content) < 3:
        return 0
    return 1

def similarity_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def calculate_lexical_diversity(text):
    clean_text = re.sub(r'<[^>]+>', '', text)
    words = re.findall(r'\b\w+\b', clean_text.lower())
    if len(words) < 5:
        return 0.0

    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    base_diversity = unique_words / total_words
    repeat_penalty = 1.0
    for i in range(len(words) - 3):
        sequence = ' '.join(words[i:i+3])
        if sequence in ' '.join(words[i+3:]):
            repeat_penalty *= 0.7

    diversity = base_diversity * repeat_penalty
    return min(diversity, 1.0)

def check_paragraph_formatting(description):

    p_pattern = r'<p>(.*?)</p>'
    paragraphs = re.findall(p_pattern, description, re.DOTALL | re.IGNORECASE)

    if not paragraphs:
        return 0.0

    valid_count = 0
    for paragraph in paragraphs:
        clean_content = re.sub(r'<[^>]+>', '', paragraph).strip()
        if (len(clean_content) >= 10 and
            not clean_content.isdigit() and
            len(set(clean_content.split())) >= 3):
            valid_count += 1

    return valid_count / len(paragraphs) if paragraphs else 0.0

def is_characteristic_present(text, char_name, char_value, threshold=0.6):

    text_lower = text.lower()
    char_name_lower = char_name.lower()
    char_value_lower = str(char_value).lower()
    if (char_name_lower in text_lower and char_value_lower in text_lower):
        return True
    name_similarity = similarity_ratio(char_name_lower, text_lower)
    value_similarity = similarity_ratio(char_value_lower, text_lower)
    return (name_similarity >= threshold and value_similarity >= threshold)

def check_paragraph_tags(description, characteristics, threshold=0.6):
    if not characteristics:
        return 1.0

    p_pattern = r'<p>(.*?)</p>'
    paragraphs = re.findall(p_pattern, description, re.DOTALL | re.IGNORECASE)

    if not paragraphs:
        return 0.0

    unique_paragraphs = set()
    duplicate_penalty = 1.0

    for paragraph in paragraphs:
        clean_text = re.sub(r'<[^>]+>', '', paragraph).strip()
        if clean_text in unique_paragraphs:
            duplicate_penalty *= 0.5
        unique_paragraphs.add(clean_text)

    char_matched = {char: False for char in characteristics}
    valid_matches = 0

    for paragraph in paragraphs:
        clean_text = re.sub(r'<[^>]+>', '', paragraph).strip()

        matched_chars = []
        for char_name, char_value in characteristics.items():
            if is_characteristic_present(clean_text, char_name, char_value, threshold):
                matched_chars.append(char_name)

        if len(matched_chars) == 1 and not char_matched.get(matched_chars[0], False):
            char_matched[matched_chars[0]] = True
            valid_matches += 1

    coverage_score = sum(1 for matched in char_matched.values() if matched) / len(characteristics)
    return coverage_score * duplicate_penalty

def check_all_characteristics_covered(description, characteristics, threshold=0.5):
    if not characteristics:
        return 1.0

    clean_description = re.sub(r'<[^>]+>', '', description)
    found_chars = set()

    for char_name, char_value in characteristics.items():
        if is_characteristic_present(clean_description, char_name, char_value, threshold):
            found_chars.add(char_name)

    return len(found_chars) / len(characteristics)

def check_text_quality(description):
    clean_text = re.sub(r'<[^>]+>', '', description)
    words = clean_text.split()

    if len(words) < 10:
        return 0.0

    word_counts = Counter(words)
    unique_ratio = len(word_counts) / len(words)

    repeat_penalty = 1.0
    for word, count in word_counts.items():
        if count > len(words) * 0.1:
            repeat_penalty *= 0.7

    return min(unique_ratio * repeat_penalty, 1.0)

def calculate_total_score(description, characteristics):

    h1_score = check_h1_tags(description)
    paragraph_score = check_paragraph_tags(description, characteristics, threshold=0.7)
    coverage_score = check_all_characteristics_covered(description, characteristics, threshold=0.6)
    diversity_score = calculate_lexical_diversity(description)
    formatting_score = check_paragraph_formatting(description)
    text_quality_score = check_text_quality(description)

    weights = {
        'h1': 0.10,
        'paragraphs': 0.25,
        'coverage': 0.15,
        'diversity': 0.20,
        'formatting': 0.15,
        'quality': 0.15
    }

    total_score = (
        h1_score * weights['h1'] +
        paragraph_score * weights['paragraphs'] +
        coverage_score * weights['coverage'] +
        diversity_score * weights['diversity'] +
        formatting_score * weights['formatting'] +
        text_quality_score * weights['quality']
    ) * 100

    return {
        'total_score': round(total_score, 2),
        'detailed_scores': {
            'h1_present': h1_score,
            'paragraphs_validation': round(paragraph_score, 2),
            'characteristics_coverage': round(coverage_score, 2),
            'lexical_diversity': round(diversity_score, 2),
            'paragraph_formatting': round(formatting_score, 2),
            'text_quality': round(text_quality_score, 2)
        }
    }


@dataclass
class SampleGenerationResult():
  description: str
  latency: float
  total_score: float
  detailed_score: dict

@dataclass
class FullGenerationResult():
  throughput: float
  latency_avg: float
  total_score_avg: float
  outputs: List[SampleGenerationResult]

def print_compact_stats(results: FullGenerationResult) -> None:

    print(" СТАТИСТИКА ГЕНЕРАЦИИ")
    print(f"   Throughput: {results.throughput:.2f} req/sec")
    print(f"   Avg Latency: {results.latency_avg:.3f} sec")
    print(f"   Samples: {len(results.outputs)}")
    print(f"   Avg Score: {sum(s.total_score for s in results.outputs)/len(results.outputs):.3f}")

    if len(results.outputs) >= 3:
        print("\n 3 случайных описания:")
        random_indices = random.sample(range(len(results.outputs)), min(3, len(results.outputs)))
        for i, idx in enumerate(random_indices, 1):
            desc = results.outputs[idx].description
            print(f"   {i}. {desc}")


def run_generation(input_data, sampling_params, llm):
    start_time = time.time()
    generated_data = llm.generate(input_data, sampling_params)
    end_time = time.time()

    latency = end_time - start_time
    throughput = len(input_data)/(end_time - start_time)

    latency_list = [latency for _ in range(len(generated_data))]

    return generated_data, throughput, latency_list


def prepare_results(generated_data, throughput, latency_list):
    generation_results = []
    for output, features, latency in zip(generated_data, result_data, latency_list):
        description = output.outputs[0].text
        score_info = calculate_total_score(output.outputs[0].text, features)

        sample_info = SampleGenerationResult(
            latency=latency,
            description=description,
            total_score=score_info['total_score'],
            detailed_score=copy.deepcopy(score_info['detailed_scores'])
        )
        generation_results.append(sample_info)

    full_generation_result = FullGenerationResult(
        throughput=throughput,
        latency_avg=sum([x.latency for x in generation_results])/len(generation_results),
        total_score_avg=sum([x.total_score for x in generation_results])/len(generation_results),
        outputs=generation_results
    )
    return full_generation_result


model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    max_model_len=20_000
)
prompt_template = Template("""
    Сгенерируй описание товара на основе представленных характеристик.

    ХАРАКТЕРИСТИКИ:
    $features

    Строго следуй инструкции по генерации

    ИНСТРУКЦИЯ:
    $instruction

    Также учти, что НЕЛЬЗЯ добавлять никакие дополнительные размышления после параграфов.
    Нужно только ровно одно описание.
    """
)
instruction = """
- Описание должно начинаться с заголовка, выделенного html-тегами <h1> и </h1>
- Все характеристики должны быть учтены в описании
- Под каждую характеристику должен быть выделен отдельный абзац
- каждый абзац должен быть выделен html тегами <p> и </p>
- язык должен быть разнообразным согласно простой метрике лексического разнообразия VocD
- отсутствие повторов
"""

input_data = [prompt_template.substitute(features=str(features), instruction=instruction) for features in result_data]

sampling_params = SamplingParams(temperature=0.1,
                                top_p=0.9,
                                max_tokens=600,
                                presence_penalty=1.2,
                                repetition_penalty=1,
                                )

SEED = 42
random.seed(SEED)
dataset = load_dataset("UniqueData/asos-e-commerce-dataset")
data = dataset['train']
random_indices = random.sample(range(len(data)), 100)
random_samples = data.select(random_indices)
target_features = ["name", "size", "category", "price", "color"]
result_data = []
for sample in random_samples:
  target_features_sample = {feature_name: sample.get(feature_name) for feature_name in target_features}
  result_data.append(target_features_sample)
df = pd.DataFrame(result_data)
generated_data, throughput, latency_list = run_generation(input_data, sampling_params, llm)
print(f"Длительность ответа на единичный запрос {sum(latency_list)/len(latency_list)}")
print(f"Количество запросов, обрабатываемых в секунду {throughput}")

full_generation_result = prepare_results(generated_data, throughput, latency_list)
print_compact_stats(full_generation_result)
