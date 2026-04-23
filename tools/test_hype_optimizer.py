"""
Пример использования HyPE (Hypothetical Prompt Engineering) Optimizer.

HyPE - это zero-shot мета-оптимизатор, который улучшает промпты
на основе структурированного мета-промпта.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.registry import get_model
from src.meta_optimizer import HyPEOptimizer, HypeMetaPromptConfig


def main():
    # Конфигурация модели для оптимизации
    model_config = {
        "provider": "openrouter",
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "timeout": 120,
        "max_retries": 3,
        "max_completion_tokens": 4000,
        "rate_limit": {
            "enabled": True,
            "requests_per_second": 1.0,
            "max_bucket_size": 10,
        },
    }

    print("🔄 Загрузка модели...")
    model = get_model(model_config)
    print(f"✅ Модель загружена: {model.model_name}")

    # Создаём оптимизатор с дефолтной конфигурацией
    config = HypeMetaPromptConfig(
        target_prompt_form="instructional ",
        require_markdown_prompt=True,
    )

    optimizer = HyPEOptimizer(model=model, config=None)
    print("✅ HyPE Optimizer создан")

    # Пример: оптимизация простого промпта
    original_prompt = """
You are a simulated person completing IPIP-NEO personality questionnaire.
Answer based on your personality traits.
"""

    print("\n" + "=" * 60)
    print("ОРИГИНАЛЬНЫЙ ПРОМПТ:")
    print("=" * 60)
    print(original_prompt)

    print("\n" + "=" * 60)
    print("МЕТА-ПРОМПТ (что отправляется в модель):")
    print("=" * 60)
    print(optimizer.meta_prompt)

    # Оптимизация (требуется OPENROUTER_API_KEY)
    print("\n" + "=" * 60)
    print("ОПТИМИЗАЦИЯ...")
    print("=" * 60)

    # Раскомментируйте для реального запуска:
    try:
        optimized = optimizer.optimize(
            prompt=original_prompt,
            meta_info={
                "task": "personality simulation",
                "domain": "psychology",
                "target": "IPIP-NEO-120 questionnaire"
            }
        )
        print("\n✅ ОПТИМИЗИРОВАННЫЙ ПРОМПТ:")
        print(optimized)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

    # print("\n" + "=" * 60)
    # print("СТРУКТУРА ОПТИМИЗАТОРА")
    # print("=" * 60)
    # print(f"Sections: {optimizer.get_section('prompt_structure')[:200]}...")
    # print(f"\nConstraints: {optimizer.get_section('constraints')}")


if __name__ == "__main__":
    main()
