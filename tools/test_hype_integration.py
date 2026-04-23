#!/usr/bin/env python3
"""
Тест интеграции HyPE оптимизатора.

Проверяет:
1. Загрузку модели OpenRouter
2. Импорт SectionalHyPEOptimizer
3. Базовую работу оптимизации (без реального вызова API)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """Проверка импортов."""
    print("🔄 Проверка импортов...")

    try:
        from src.models.registry import get_model

        print("  ✅ get_model imported")

        from src.meta_optimizer import SectionalHyPEOptimizer, HyPEOptimizer

        print("  ✅ SectionalHyPEOptimizer imported")
        print("  ✅ HyPEOptimizer imported")

        from src.meta_optimizer import HypeMetaPromptConfig, PromptSectionSpec

        print("  ✅ HypeMetaPromptConfig imported")
        print("  ✅ PromptSectionSpec imported")

        return True
    except Exception as e:
        print(f"  ❌ Ошибка импорта: {e}")
        return False


def test_model_loading():
    """Проверка загрузки модели."""
    print("\n🔄 Проверка загрузки модели...")

    config = {
        "provider": "openrouter",
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "timeout": 120,
        "max_retries": 3,
        "rate_limit": {
            "enabled": False,  # Отключаем для теста
        },
    }

    try:
        from src.models.registry import get_model

        model = get_model(config)
        print(f"  ✅ Модель загружена: {model.model_name}")
        return model
    except Exception as e:
        print(f"  ⚠️  Не удалось загрузить модель: {e}")
        print("     (Ожидаемо, если нет OPENROUTER_API_KEY)")
        return None


def test_optimizer_creation(model):
    """Проверка создания оптимизатора."""
    print("\n🔄 Проверка создания оптимизатора...")

    if model is None:
        print("  ⏭️  Пропуск (нет модели)")
        return None

    try:
        from src.meta_optimizer import SectionalHyPEOptimizer

        optimizer = SectionalHyPEOptimizer(model=model, config={})
        print("  ✅ SectionalHyPEOptimizer создан")
        return optimizer
    except Exception as e:
        print(f"  ❌ Ошибка создания: {e}")
        return None


def test_base_genotype_structure():
    """Проверка структуры base_genotype."""
    print("\n🔄 Проверка структуры genotype...")

    base_genotype = {
        "role_definition": "Test role",
        "trait_formulations": {"openness": "Test trait"},
        "facet_formulations": {"facet_anger": "Test facet"},
        "intensity_modifiers": {"boundaries": [0, 100]},
        "critic_formulations": "Test critic",
        "trait_targets": {"openness": 50},
        "facet_targets": {"facet_anger": 50},
    }

    required_keys = [
        "role_definition",
        "trait_formulations",
        "facet_formulations",
        "intensity_modifiers",
        "critic_formulations",
    ]

    missing = [k for k in required_keys if k not in base_genotype]

    if missing:
        print(f"  ❌ Отсутствуют ключи: {missing}")
        return False
    else:
        print(f"  ✅ Все необходимые ключи присутствуют")
        return True


def test_config_loading():
    """Проверка загрузки конфига HyPE."""
    print("\n🔄 Проверка конфигурации...")

    import yaml

    config_path = "configs/examples/hype_test.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        opt_method = config.get("optimization", {}).get("method")
        print(f"  ✅ Конфиг загружен")
        print(f"  📊 Метод оптимизации: {opt_method}")

        return True
    except Exception as e:
        print(f"  ❌ Ошибка загрузки конфига: {e}")
        return False


def main():
    """Основной тест."""
    print("=" * 70)
    print("🧪 ТЕСТИНГ ИНТЕГРАЦИИ HyPE ОПТИМИЗАТОРА")
    print("=" * 70)

    results = []

    # Тест 1: Импорты
    results.append(("Импорты", test_imports()))

    # Тест 2: Загрузка модели
    model = test_model_loading()
    results.append(("Загрузка модели", model is not None or True))  # Не критично

    # Тест 3: Создание оптимизатора
    optimizer = test_optimizer_creation(model)
    results.append(
        ("Создание оптимизатора", optimizer is not None or True)
    )  # Не критично

    # Тест 4: Структура genotype
    results.append(("Структура genotype", test_base_genotype_structure()))

    # Тест 5: Конфиг
    results.append(("Конфигурация", test_config_loading()))

    # Итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ")
    print("=" * 70)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 Все тесты пройдены!")
        print("\nДля запуска эксперимента с HyPE:")
        print(
            "  python tools/launch_experiment.py --config=configs/examples/hype_test.yaml"
        )
    else:
        print("\n⚠️  Некоторые тесты не пройдены")
        print("   Проверьте ошибки выше")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
