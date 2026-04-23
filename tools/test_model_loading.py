import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.registry import get_model


def test_openrouter_model():
    """Тест загрузки OpenRouter модели."""
    config = {
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

    print("🔄 Загрузка OpenRouter модели...")
    try:
        model = get_model(config)
        print(f"✅ Модель успешно загружена: {model.model_name}")
        print(f"   Провайдер: OpenRouter")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Rate limiting: {config['rate_limit']['enabled']}")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        raise


def test_cloud_model():
    """Тест загрузки Cloud API модели (для проверки обратной совместимости)."""
    config = {
        "provider": "cloud",
        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "temperature": 0.7,
        "timeout": 120,
        "max_retries": 2,
    }

    print("\n🔄 Загрузка Cloud API модели...")
    try:
        model = get_model(config)
        print(f"✅ Модель успешно загружена: {model.model_name}")
        print(f"   Провайдер: Cloud.ru")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        # Не падаем, т.к. CLOUD_API_KEY может не быть
        print("   (Ожидаемо, если нет CLOUD_API_KEY)")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование загрузки моделей")
    print("=" * 60)

    # Тест OpenRouter (основной)
    try:
        test_openrouter_model()
    except Exception as e:
        print(f"\n⚠️ OpenRouter тест не пройден: {e}")
        print("   Убедитесь, что OPENROUTER_API_KEY установлен в .env")

    # Тест Cloud (опционально)
    # test_cloud_model()

    print("\n" + "=" * 60)
    print("Тестирование завершено")
    print("=" * 60)
