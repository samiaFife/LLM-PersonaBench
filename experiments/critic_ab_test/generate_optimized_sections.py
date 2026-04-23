#!/usr/bin/env python3
"""
Генерирует оптимизированные секции critic и task из новых мета-промптов.
Использует HyPE для создания улучшенных версий секций.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.registry import get_model


def load_meta_prompts(json_path: str):
    """Загружает мета-промпты из JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_original_texts():
    """Извлекает оригинальные тексты секций из system.py."""
    from experiments.critic_ab_test.new_meta_prompts import ORIGINAL_TEXTS

    return ORIGINAL_TEXTS


def optimize_section(
    model, meta_prompt: str, original_text: str, section_name: str
) -> str:
    """
    Оптимизирует секцию используя мета-промпт.

    Args:
        model: LLM модель
        meta_prompt: Полный мета-промпт с {original_text} плейсхолдером
        original_text: Оригинальный текст секции
        section_name: Название секции (critic/task)

    Returns:
        Оптимизированный текст секции
    """
    # Подставляем оригинальный текст в мета-промпт
    full_prompt = meta_prompt.replace("{original_text}", original_text)

    print(f"\n{'=' * 60}")
    print(f"🔄 Оптимизация {section_name.upper()}...")
    print(f"{'=' * 60}")
    print(f"Meta-prompt length: {len(full_prompt)} chars")
    print(f"Original text length: {len(original_text)} chars")

    # Вызываем модель
    try:
        response = model.generate(full_prompt)

        # Извлекаем результат из <result_prompt> тегов
        if "<result_prompt>" in response and "</result_prompt>" in response:
            start = response.find("<result_prompt>") + len("<result_prompt>")
            end = response.find("</result_prompt>")
            optimized = response[start:end].strip()
        else:
            optimized = response.strip()

        print(f"✅ Оптимизировано! Новая длина: {len(optimized)} chars")
        print(f"\n📄 Результат:\n{optimized[:200]}...")

        return optimized

    except Exception as e:
        print(f"❌ Ошибка при оптимизации {section_name}: {e}")
        raise


def generate_optimized_sections(
    meta_prompts_path: str = None,
    output_dir: str = None,
    model_name: str = "gpt-4o-mini",
):
    """
    Главная функция для генерации оптимизированных секций.

    Args:
        meta_prompts_path: Путь к JSON с мета-промптами
        output_dir: Директория для сохранения результатов
        model_name: Название модели для оптимизации
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if meta_prompts_path is None:
        # Используем последний сгенерированный файл
        meta_dir = Path("experiments/critic_ab_test/meta_prompts_v2")
        json_files = sorted(meta_dir.glob("prepared_meta_prompts_*.json"))
        if not json_files:
            raise FileNotFoundError("Не найдены файлы мета-промптов!")
        meta_prompts_path = str(json_files[-1])

    if output_dir is None:
        output_dir = f"experiments/critic_ab_test/optimized_sections_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 ГЕНЕРАЦИЯ ОПТИМИЗИРОВАННЫХ СЕКЦИЙ")
    print("=" * 70)
    print(f"📁 Meta-prompts: {meta_prompts_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {model_name}")

    # Загружаем данные
    meta_data = load_meta_prompts(meta_prompts_path)
    original_texts = extract_original_texts()

    # Инициализируем модель
    print("\n⚙️  Инициализация модели...")
    model_config = {
        "model_name": model_name,
        "provider": "openrouter",
        "temperature": 0.7,
        "timeout": 120,
        "max_retries": 3,
        "max_completion_tokens": 2000,
        "rate_limit": {
            "enabled": True,
            "requests_per_second": 1.0,
            "max_bucket_size": 5,
        },
    }
    model = get_model(model_config)

    results = {
        "timestamp": timestamp,
        "meta_prompts_source": meta_prompts_path,
        "model": model_name,
        "sections": {},
    }

    # Оптимизируем CRITIC (новый вариант)
    print("\n" + "=" * 70)
    print("📝 CRITIC СЕКЦИЯ")
    print("=" * 70)

    critic_variants = meta_data["sections"]["critic"]
    new_critic = next(v for v in critic_variants if not v["is_baseline"])

    optimized_critic = optimize_section(
        model, new_critic["full_meta_prompt"], original_texts["critic"], "critic"
    )

    results["sections"]["critic"] = {
        "variant_id": new_critic["variant_id"],
        "variant_name": new_critic["variant_name"],
        "original": original_texts["critic"],
        "optimized": optimized_critic,
    }

    # Сохраняем отдельно
    with open(output_path / "critic_optimized.txt", "w", encoding="utf-8") as f:
        f.write(optimized_critic)

    # Оптимизируем TASK (новый вариант)
    print("\n" + "=" * 70)
    print("📝 TASK СЕКЦИЯ")
    print("=" * 70)

    task_variants = meta_data["sections"]["task"]
    new_task = next(v for v in task_variants if not v["is_baseline"])

    optimized_task = optimize_section(
        model, new_task["full_meta_prompt"], original_texts["task"], "task"
    )

    results["sections"]["task"] = {
        "variant_id": new_task["variant_id"],
        "variant_name": new_task["variant_name"],
        "original": original_texts["task"],
        "optimized": optimized_task,
    }

    # Сохраняем отдельно
    with open(output_path / "task_optimized.txt", "w", encoding="utf-8") as f:
        f.write(optimized_task)

    # Сохраняем полный результат
    results_file = output_path / "optimization_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Создаем файл с полным system для использования
    system_config = {
        "critic_formulations": optimized_critic,
        "task": optimized_task,
        "output_format": "For each statement, respond with only the number (1-5) that best represents you.",
        "role_definition": "You are completing a personality assessment. Answer honestly based on your personality profile.",
    }

    system_file = output_path / "system_optimized.json"
    with open(system_file, "w", encoding="utf-8") as f:
        json.dump(system_config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print("=" * 70)
    print(f"📁 Результаты сохранены в: {output_dir}")
    print(f"   - critic_optimized.txt")
    print(f"   - task_optimized.txt")
    print(f"   - optimization_results.json")
    print(f"   - system_optimized.json (полный system для использования)")

    return output_dir


def main():
    """Точка входа."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Генерация оптимизированных секций из новых мета-промптов"
    )
    parser.add_argument(
        "--meta-prompts",
        type=str,
        default=None,
        help="Путь к JSON с мета-промптами (по умолчанию - последний сгенерированный)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Директория для сохранения (по умолчанию - auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Модель для оптимизации (по умолчанию: gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Запускаем
    output_dir = generate_optimized_sections(
        meta_prompts_path=args.meta_prompts,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print(f"\n💡 Для запуска теста используй:")
    print(
        f"   python experiments/critic_ab_test/run_full_test.py --system-path {output_dir}/system_optimized.json"
    )


if __name__ == "__main__":
    main()
