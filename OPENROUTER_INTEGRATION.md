# OpenRouter Integration

Поддержка [OpenRouter](https://openrouter.ai) — единый API для доступа к сотням LLM.

## Что добавлено

### `OpenRouterAPIModel` ([`src/models/providers/openrouter_api.py`](src/models/providers/openrouter_api.py))

- Обёртка над `langchain_openai.ChatOpenAI` с `base_url = https://openrouter.ai/api/v1`
- Rate limiting через `InMemoryRateLimiter` (настраивается или отключается)
- Опциональный `extra_body` для ограничения провайдеров OpenRouter

### Registry ([`src/models/registry.py`](src/models/registry.py))

Добавлена поддержка `provider: "openrouter"` — все параметры через YAML.

## Настройка

### API ключ

```bash
export OPENROUTER_API_KEY="sk-or-..."
# или добавить в .env файл
```

### Конфигурация модели (YAML)

```yaml
model:
  provider: "openrouter"
  model_name: "openai/gpt-4o-mini"   # любая модель из openrouter.ai/models
  temperature: 0.7
  timeout: 120
  max_retries: 3
  max_completion_tokens: 4000
  # Опционально: ограничение провайдеров
  extra_body:
    allowed_providers: ["google-vertex"]
  # Rate limiting (включён по умолчанию)
  rate_limit:
    enabled: true
    requests_per_second: 1.0
    max_bucket_size: 10
```

### Запуск

```bash
python tools/launch_experiment.py --config=configs/examples/hype_quick_test.yaml
```

## Использование HyPEOptimizer с OpenRouter

```python
from src.models.registry import get_model
from src.meta_optimizer import HyPEOptimizer, HypeMetaPromptConfig

model = get_model({
    "provider": "openrouter",
    "model_name": "openai/gpt-4o-mini",
    "temperature": 0.7,
    "rate_limit": {"enabled": True},
})

config = HypeMetaPromptConfig(target_prompt_form="instructional ")
optimizer = HyPEOptimizer(model=model, config=config)

optimized_prompt = optimizer.optimize(
    prompt="You are a simulated person...",
    meta_info={"task": "personality simulation"},
)
```

## Параметры модели

| Параметр | Описание | Обязательный |
|----------|----------|--------------|
| `provider` | `"openrouter"` | ✅ |
| `model_name` | Имя модели (см. openrouter.ai/models) | ✅ |
| `temperature` | 0.0–1.0 | — |
| `timeout` | Таймаут запроса в секундах | — |
| `max_retries` | Количество повторных попыток | — |
| `max_completion_tokens` | Лимит токенов в ответе | — |
| `extra_body` | Доп. параметры API (allowed_providers и т.д.) | — |
| `rate_limit.enabled` | Включить rate limiting | — |
| `rate_limit.requests_per_second` | Запросов в секунду | — |
| `rate_limit.max_bucket_size` | Размер bucket'а | — |

## Тестирование

```bash
python tools/test_model_loading.py
python tools/test_hype_optimizer.py
```
