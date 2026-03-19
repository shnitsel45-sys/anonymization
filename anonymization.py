#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение библиотек для анонимизации текста"""

import time
import re
from typing import List, Tuple
from dataclasses import dataclass

# ============================================================
# 1. MICROSOFT PRESIDIO
# ============================================================
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def anonymize_with_presidio(text: str) -> Tuple[str, float]:
    start_time = time.time()
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language='en')
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    elapsed = time.time() - start_time
    return anonymized_text.text, elapsed


# ============================================================
# 2. TANAOS / ARTIFEX
# ============================================================
from artifex import Artifex


def anonymize_with_tanaos(text: str) -> Tuple[str, float]:
    start_time = time.time()
    model = Artifex()
    anonymized_text = model.text_anonymization(text)
    elapsed = time.time() - start_time
    return anonymized_text[0], elapsed


# ============================================================
# 3. SPACY NER
# ============================================================
import spacy
nlp = spacy.load("en_core_web_sm")


def anonymize_with_spacy(text: str) -> Tuple[str, float]:
    start_time = time.time()
    doc = nlp(text)
    anonymized = text
    entities = sorted([(ent.start_char, ent.end_char, ent.label_)
                       for ent in doc.ents], reverse=True)
    for start, end, label in entities:
        replacement = f"[{label}]"
        anonymized = anonymized[:start] + replacement + anonymized[end:]
    elapsed = time.time() - start_time
    return anonymized, elapsed


# ============================================================
# 4. REGEX-BASED
# ============================================================
def anonymize_with_regex(text: str) -> Tuple[str, float]:
    start_time = time.time()
    anonymized = text
    patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),
        (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PERSON]'),
        (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]'),
        (r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '[MONEY]'),
        (r'\b\d+(?:\.\d+)?%', '[PERCENTAGE]'),
    ]
    for pattern, replacement in patterns:
        anonymized = re.sub(pattern, replacement, anonymized)
    elapsed = time.time() - start_time
    return anonymized, elapsed


# ============================================================
# 5. LLAMA-CPP-PYTHON (локальный GGUF)
# ============================================================
from llama_cpp import Llama
import os

_llama_cpp_model = None


def _get_llama_cpp_model():
    global _llama_cpp_model
    if _llama_cpp_model is None:
        model_file = "phi-2.Q4_K_M.gguf"
        print(f"   Загрузка модели: {model_file}...")
        _llama_cpp_model = Llama(
            model_path=model_file,
            n_ctx=2048,
            n_batch=512,
            n_threads=4,
            verbose=False,
            use_mmap=True,
            use_mlock=False
        )
        print(f"   Модель загружена")
    return _llama_cpp_model


def anonymize_with_llama_cpp(text: str) -> Tuple[str, float]:
    start_time = time.time()
    prompt = f"""Instruct: Anonymize the text by replacing sensitive information with tags.
Replace: names–>[PERSON], organizations–>[ORG], locations–>[LOC], emails–>[EMAIL], phones–>[PHONE], date–>[DATETIME]

Input: {text}

Output:"""
    model = _get_llama_cpp_model()
    response = model(
        prompt,
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["Instruct:", "Input:", "\n\n"],
        echo=False
    )
    anonymized_text = response['choices'][0]['text'].strip()
    elapsed = time.time() - start_time
    return anonymized_text, elapsed


# ============================================================
# 6. HUGGING FACE TRANSFORMERS NER
# ============================================================
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

NER_MODEL_NAME = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def anonymize_with_hf_transformers(text: str) -> Tuple[str, float]:
    start_time = time.time()
    entities = ner_pipeline(text)
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    anonymized = text
    label_map = {
        'PER': '[PERSON]',
        'ORG': '[ORGANIZATION]',
        'LOC': '[LOCATION]',
        'MISC': '[MISC]'
    }
    for entity in entities:
        replacement = label_map.get(entity['entity_group'], f"[{entity['entity_group']}]")
        anonymized = anonymized[:entity['start']] + replacement + anonymized[entity['end']:]
    elapsed = time.time() - start_time
    return anonymized, elapsed


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
@dataclass
class MethodResult:
    name: str
    anonymized_text: str
    elapsed_time: float
    available: bool
    method_type: str


def compare_anonymization_methods(text: str) -> List[MethodResult]:
    methods = [
        ("Microsoft Presidio", anonymize_with_presidio, True, "ml-based"),
        ("tanaos/Artifex", anonymize_with_tanaos, True, "ml-based"),
        ("spaCy NER", anonymize_with_spacy, True, "ml-based"),
        ("Regex-based", anonymize_with_regex, True, "rule-based"),
        ("llama-cpp", anonymize_with_llama_cpp, True, "llm-based"),

    ]
    #("HF Transformers NER", anonymize_with_hf_transformers, True, "ml-based"),
    results = []
    for name, func, available, method_type in methods:
        anonymized, elapsed = func(text)
        results.append(MethodResult(name, anonymized, elapsed, available, method_type))
    return results


def print_results(results: List[MethodResult], original_text: str):
    print("=" * 120)
    print("СРАВНЕНИЕ БИБЛИОТЕК ДЛЯ АНОНИМИЗАЦИИ ТЕКСТА")
    print("=" * 120)

    print(f"\nИСХОДНЫЙ ТЕКСТ:\n{original_text}\n")
    print("=" * 120)

    print(f"\n{'Метод':<30} {'Тип':<15} {'Время (сек)':<15} {'Статус':<10}")
    print("-" * 120)

    for result in results:
        status = "OK" if result.available else "N/A"
        time_str = f"{result.elapsed_time:.4f}" if result.available else "N/A"
        print(f"{result.name:<30} {result.method_type:<15} {time_str:<15} {status:<10}")

    print("-" * 120)

    print("\n" + "=" * 120)
    print("ОБРАБОТАННЫЕ ТЕКСТЫ")
    print("=" * 120)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.name}")
        print(f"   Тип: {result.method_type}")
        print(f"   Время: {result.elapsed_time:.4f} сек")
        print(f"   Результат:\n   {result.anonymized_text}")
        print("   " + "-" * 115)

    available_results = [r for r in results if r.available]
    if available_results:
        print("\n" + "=" * 120)
        print("СТАТИСТИКА")
        print("=" * 120)

        fastest = min(available_results, key=lambda x: x.elapsed_time)
        slowest = max(available_results, key=lambda x: x.elapsed_time)
        avg_time = sum(r.elapsed_time for r in available_results) / len(available_results)

        print(f"   Самый быстрый:  {fastest.name} ({fastest.elapsed_time:.4f} сек)")
        print(f"   Самый медленный: {slowest.name} ({slowest.elapsed_time:.4f} сек)")
        print(f"   Среднее время:  {avg_time:.4f} сек")
        print(f"   Доступно методов: {len(available_results)}/{len(results)}")

    print("\n" + "=" * 120)
    print("Готово!")
    print("=" * 120)


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    TEST_TEXT = """
    John Smith works at Microsoft Corporation in Redmond, Washington. 
    His email is john.smith@microsoft.com and phone number is 555-123-4567. 
    He lives at 123 Main Street, Seattle, WA 98101. 
    His salary is $150,000 per year. 
    The meeting is scheduled for 03/15/2024 at 2:30 PM.
    His IP address is 192.168.1.100 and credit card number is 4532-1234-5678-9012.
    """

    print("Запуск сравнения методов анонимизации...\n")
    results = compare_anonymization_methods(TEST_TEXT.strip())
    print_results(results, TEST_TEXT.strip())