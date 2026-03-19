#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение библиотек для анонимизации текста (Китайский язык)"""

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
    """Создание анализатора с поддержкой китайского языка"""
    global _presidio_analyzer

    if _presidio_analyzer is None:
        # Загрузка реестра распознавателей из файла
        registry_provider = RecognizerRegistryProvider(
            conf_file="./presidio-zh-recognizers.yml"
        )
        registry = registry_provider.create_recognizer_registry()

        # Загрузка NLP двигателя из файла
        nlp_provider = NlpEngineProvider(
            conf_file="./presidio-zh-nlp.yml"
        )
        nlp_engine = nlp_provider.create_engine()

        # Создание анализатора
        _presidio_analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["zh"]
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

        results = analyzer.analyze(text=text, language='zh')
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

        elapsed = time.time() - start_time
        return anonymized_text.text, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        return f"Ошибка Presidio: {str(e)}", elapsed


# ============================================================
# 2. LLAMA-CPP-PYTHON (локальный GGUF)
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
Replace: names→[PERSON], organizations→[ORG], locations→[LOC], emails→[EMAIL], phones→[PHONE], date→[DATE]

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
        #("llama-cpp", anonymize_with_llama_cpp, True, "llm-based"),
    ]
    results = []
    for name, func, available, method_type in methods:
        anonymized, elapsed = func(text)
        results.append(MethodResult(name, anonymized, elapsed, available, method_type))
    return results


def print_results(results: List[MethodResult], original_text: str):
    print("=" * 120)
    print("СРАВНЕНИЕ БИБЛИОТЕК ДЛЯ АНОНИМИЗАЦИИ ТЕКСТА (КИТАЙСКИЙ)")
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
    # Китайский тестовый текст с PII
    TEST_TEXT = """
    刘强在中国工商银行工作，位于北京市西城区。
他的电子邮件是 liu.qiang@icbc.com.cn，电话号码是 +86 135-9999-1111。
他住在广东省深圳市南山区科技园路 300 号。
他的年薪是 950000 人民币。
会议安排在 2025 年 1 月 19 日下午 4:15。
他的 IP 地址是 10.10.10.100，信用卡号是 4539-8765-4321-0987。
身份证号：440305199007251234。
刘强在中国工商银行工作，位于北京市西城区。
他的电子邮件是 liu.qiang@icbc.com.cn，电话号码是 +86 135-9999-1111。
他住在广东省深圳市南山区科技园路 300 号。
他的年薪是 950000 人民币。
会议安排在 2025 年 1 月 19 日下午 4:15。
他的 IP 地址是 10.10.10.100，信用卡号是 4539-8765-4321-0987。
身份证号：440305199007251234。
刘强在中国工商银行工作，位于北京市西城区。
他的电子邮件是 liu.qiang@icbc.com.cn，电话号码是 +86 135-9999-1111。
他住在广东省深圳市南山区科技园路 300 号。
他的年薪是 950000 人民币。
会议安排在 2025 年 1 月 19 日下午 4:15。
他的 IP 地址是 10.10.10.100，信用卡号是 4539-8765-4321-0987。
身份证号：440305199007251234。
刘强在中国工商银行工作，位于北京市西城区。
他的电子邮件是 liu.qiang@icbc.com.cn，电话号码是 +86 135-9999-1111。
他住在广东省深圳市南山区科技园路 300 号。
他的年薪是 950000 人民币。
会议安排在 2025 年 1 月 19 日下午 4:15。
他的 IP 地址是 10.10.10.100，信用卡号是 4539-8765-4321-0987。
身份证号：440305199007251234。
    """

    print("Запуск сравнения методов анонимизации...\n")
    results = compare_anonymization_methods(TEST_TEXT.strip())
    print_results(results, TEST_TEXT.strip())