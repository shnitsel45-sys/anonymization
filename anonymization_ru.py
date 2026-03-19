#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение библиотек для анонимизации текста (Русский язык)"""

import time
import re
from typing import List, Tuple
from dataclasses import dataclass

# ============================================================
# 1. MICROSOFT PRESIDIO
# ============================================================
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Глобальные переменные для кэширования
_presidio_analyzer = None
_presidio_anonymizer = None


def _get_presidio_analyzer():
    """Создание анализатора с поддержкой русского языка"""
    global _presidio_analyzer

    if _presidio_analyzer is None:
        # Загрузка реестра распознавателей из файла
        registry_provider = RecognizerRegistryProvider(
            conf_file="./presidio-ru-recognizers.yml"
        )
        registry = registry_provider.create_recognizer_registry()

        # Загрузка NLP двигателя из файла
        nlp_provider = NlpEngineProvider(
            conf_file="./presidio-ru-nlp.yml"
        )
        nlp_engine = nlp_provider.create_engine()

        # Создание анализатора
        _presidio_analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["ru"]
        )

    return _presidio_analyzer


def _get_presidio_anonymizer():
    """Получение анонимизатора"""
    global _presidio_anonymizer

    if _presidio_anonymizer is None:
        _presidio_anonymizer = AnonymizerEngine()

    return _presidio_anonymizer


def anonymize_with_presidio(text: str) -> Tuple[str, float]:
    start_time = time.time()

    try:
        analyzer = _get_presidio_analyzer()
        anonymizer = _get_presidio_anonymizer()

        results = analyzer.analyze(text=text, language='ru')
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

        elapsed = time.time() - start_time
        return anonymized_text.text, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        return f"Ошибка Presidio: {str(e)}", elapsed


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
nlp = spacy.load("ru_core_news_sm")


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
        (r'\+7\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}', '[PHONE]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),
        (r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', '[DATE]'),
        (r'[\$₽€]\s?\d+(?:\s?\d{3})*(?:[.,]\d{2})?', '[MONEY]'),
        (r'\b\d+(?:[.,]\d+)?%', '[PERCENTAGE]'),
        (r'\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b', '[CREDIT_CARD]'),
        (r'\b\d{10}\b', '[INN]'),
    ]
    for pattern, replacement in patterns:
        anonymized = re.sub(pattern, replacement, anonymized)
    elapsed = time.time() - start_time
    return anonymized, elapsed


# ============================================================
# 5. LLAMA-CPP-PYTHON
# ============================================================
from llama_cpp import Llama
import os

_llama_cpp_model = None


def _get_llama_cpp_model():
    global _llama_cpp_model
    if _llama_cpp_model is None:
        model_file = "Phi-4-mini-instruct-Q4_0.gguf"
        print(f"   Загрузка модели: {model_file}...")
        _llama_cpp_model = Llama(
            model_path=model_file,
            n_ctx=4096,
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
    prompt = f"""<|im_start|>system
    Ты — инструмент анонимизации данных. Твоя задача — заменить всю чувствительную информацию в тексте на теги.
    Правила:
    - Имена людей -> [PERSON]
    - Компании -> [ORG]
    - Города/Страны -> [LOC]
    - Email -> [EMAIL]
    - Телефоны -> [PHONE]
    - Даты -> [DATE]
    - Деньги -> [MONEY]
    - IP адреса -> [IP]
    - Другая чувствительная информация -> [OTHER]
    Ничего не объясняй. Верни только изменённый текст.<|im_end|>
    <|im_start|>user
    Текст: {text}<|im_end|>
    <|im_start|>assistant
    """
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
        #("spaCy NER", anonymize_with_spacy, True, "ml-based"),
        #("Regex-based", anonymize_with_regex, True, "rule-based"),
        ("llama-cpp", anonymize_with_llama_cpp, True, "llm-based")
    ]
    results = []
    for name, func, available, method_type in methods:
        anonymized, elapsed = func(text)
        results.append(MethodResult(name, anonymized, elapsed, available, method_type))
    return results


def print_results(results: List[MethodResult], original_text: str):
    print("=" * 120)
    print("СРАВНЕНИЕ БИБЛИОТЕК ДЛЯ АНОНИМИЗАЦИИ ТЕКСТА (РУССКИЙ)")
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
    # Русский тестовый текст с PII
    TEST_TEXT = """
    Иван Петров работает в компании Яндекс в Москве на улице Тверская, дом 15.
    Его электронная почта ivan.petrov@yandex.ru и телефон +7 (999) 123-45-67.
    Он живёт в Санкт-Петербурге по адресу Невский проспект, 100, квартира 50.
    Его зарплата составляет 150000 рублей в месяц.
    Встреча запланирована на 15.03.2024 в 14:30.
    Его IP-адрес 192.168.1.100 и номер карты 4532-1234-5678-9012.
    ИНН: 771234567890.
    """

    print("Запуск сравнения методов анонимизации...\n")
    results = compare_anonymization_methods(TEST_TEXT.strip())
    print_results(results, TEST_TEXT.strip())