# Документация: Пайплайн оптимизации промптов

## Обзор

Пайплайн оптимизации промптов для симуляции личности.
Все методы оптимизации генотипов подключаются через единый интерфейс `OptimizerRegistry` —
добавление нового метода не требует изменений в `person_type_opt.py`.

## Текущий пайплайн

```
launch_experiment.py
    │
    ▼
run_experiment(config)
    │
    ├── Загрузка конфига, модели, данных участников
    │
    ▼
Для каждого кластера:
    │
    ├── Создание base_genotype (роль, черты, фасеты и т.д.)
    ├── Разделение участников: train (60%) / test (40%)
    │
    ▼
    Запуск БЕЗ оптимизации на test (базовая оценка)
    │
    ▼
    OptimizerRegistry.create(config["optimization"]["method"], model, config)
    │   → "hype"      → SectionalHyPEOptimizer
    │   → "evolution" → EvolutionOptimizer
    │   → "none"      → NoOpOptimizer (возвращает base_genotype без изменений)
    │
    ▼
    optimizer.optimize(base_genotype, evaluator, train_participants) → genotype
    │
    ▼
    Запуск С оптимизированным промптом на test
    │
    ▼
    Сохранение результатов
```

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| [`tools/launch_experiment.py`](tools/launch_experiment.py) | Точка входа — загружает конфиг, вызывает `run_experiment()` |
| [`src/simulator/person_type_opt.py`](src/simulator/person_type_opt.py) | Основная логика эксперимента — `run_experiment()` |
| [`src/optimizers/base.py`](src/optimizers/base.py) | `BaseOptimizer`, `OptimizerRegistry`, `NoOpOptimizer` |
| [`src/optimizers/evolution.py`](src/optimizers/evolution.py) | `EvolutionOptimizer` — обёртка над `GAEvoluter` |
| [`src/optimizers/__init__.py`](src/optimizers/__init__.py) | Публичный API пакета, регистрация всех оптимизаторов |
| [`src/evolution/evoluter.py`](src/evolution/evoluter.py) | Реализация GA (`GAEvoluter`) |
| [`src/meta_optimizer/sectional_hype.py`](src/meta_optimizer/sectional_hype.py) | `SectionalHyPEOptimizer` — HyPE оптимизация |
| [`src/meta_optimizer/hype.py`](src/meta_optimizer/hype.py) | Базовые классы HyPE (`HyPEOptimizer`) |
| [`src/meta_optimizer/hyper_templates.py`](src/meta_optimizer/hyper_templates.py) | Шаблоны промптов для HyPE |
| [`src/evolution/my_evaluator.py`](src/evolution/my_evaluator.py) | Скоринг (`MyEvaluator`) — общий для всех методов |

## Структура конфига

```yaml
name: "my_experiment"

model:
  model_name: "gpt-4o-mini"
  provider: "openrouter"
  temperature: 0.7

# Метод оптимизации: "hype", "evolution", "none", или "" (пропустить)
optimization:
  method: "hype"   # ← единственное поле для выбора метода

# Настройки эволюции (нужны только при method: "evolution")
evolution:
  algorithm: "ga"
  population_size: 10
  num_generations: 5
  mutation_rate: 0.15
  crossover_rate: 0.8
  selection_method: "tournament"
  participant_batch_size: 10
  llm_for_evolution: "gpt-4o-mini"
  genotype_params:
    role_definition: true
    trait_formulations: true
    facet_formulations: true
    intensity_modifiers: false
    critic_formulations: true
    template_structure: false

data:
  file_path: "data/raw/df_ipipneo_120_clusters"
  clusters: [2]
  num_participants: 20

prompt:
  traits_path: "src/prompt/traits.json"
  facets_path: "src/prompt/facets.json"
  system_path: "src/prompt/system.json"

experiment:
  seed: 2026
  save_every_generation: false
```

## Как работает скоринг

### MyEvaluator

Класс `MyEvaluator` используется всеми оптимизаторами для оценки промптов:

```python
from src.evolution.my_evaluator import MyEvaluator

evaluator = MyEvaluator(
    evo_args,          # Аргументы эволюции (parse_args_from_yaml)
    task,              # Определение задачи (вопросы, формат)
    model,             # LLM модель
    fixed_modifiers,   # Модификаторы интенсивности
    template_genotype, # Базовый генотип
    config,            # Полный конфиг
)
evaluator.dev_participants = train_participants

score = evaluator.forward(prompt_str, config)  # → float
```

### Метрики скоринга

- **Similarity** — близость симулированных ответов к целевым профилям личности
- **Average difference** — средняя абсолютная разница
- **Pearson correlation** — корреляция между симулированными и целевыми оценками черт
- **MAE** — средняя абсолютная ошибка

## Как добавить новый метод оптимизации

### Шаг 1: Создай класс оптимизатора

```python
# Файл: src/optimizers/my_method.py

from src.optimizers.base import BaseOptimizer

class MyMethodOptimizer(BaseOptimizer):
    def optimize(
        self,
        base_genotype: dict,
        evaluator,           # MyEvaluator
        dev_participants,    # pandas DataFrame
    ) -> dict:
        # ... твоя логика ...
        return optimized_genotype
```

### Шаг 2: Зарегистрируй в src/optimizers/__init__.py

```python
# src/optimizers/__init__.py
from src.optimizers.my_method import MyMethodOptimizer

OptimizerRegistry.register("my_method", MyMethodOptimizer)
```

### Шаг 3: Создай конфиг

```yaml
# configs/examples/my_method_test.yaml
optimization:
  method: "my_method"
  # специфичные настройки твоего метода

data:
  clusters: [2]
  num_participants: 20
```

### Шаг 4: Запускай

```bash
python tools/launch_experiment.py --config=examples/my_method_test.yaml
```

**Изменений в `person_type_opt.py` не требуется.**

## Ключевые интерфейсы

### BaseOptimizer

```python
class BaseOptimizer(ABC):
    def __init__(self, model, config: dict): ...

    @abstractmethod
    def optimize(
        self,
        base_genotype: dict,
        evaluator,        # MyEvaluator
        dev_participants, # pandas DataFrame
    ) -> dict: ...        # возвращает оптимизированный генотип
```

### OptimizerRegistry

```python
from src.optimizers import OptimizerRegistry

# Регистрация
OptimizerRegistry.register("my_method", MyMethodOptimizer)

# Создание по имени из конфига
optimizer = OptimizerRegistry.create("hype", model=model, config=config)

# Список зарегистрированных методов
OptimizerRegistry.list_optimizers()  # ["none", "hype", "evolution"]
```

### Структура генотипа

```python
genotype = {
    "role_definition": "...",           # Описание системной роли
    "trait_formulations": {             # Описания черт
        "openness": "...",
        "conscientiousness": "...",
        # ...
    },
    "facet_formulations": {             # Описания фасетов
        "facet_imagination": "...",
        # ...
    },
    "intensity_modifiers": {...},       # Никогда не модифицируется
    "critic_formulations": "...",       # Внутренний критик
    "template_structure": "...",        # Формат шаблона
    "trait_targets": {...},             # Целевые значения черт
    "facet_targets": {...},             # Целевые значения фасетов
}
```

## Зарегистрированные методы

| method | Класс | Описание |
|--------|-------|----------|
| `"hype"` | `SectionalHyPEOptimizer` | Посекционная оптимизация через HyPE |
| `"evolution"` | `EvolutionOptimizer` | Генетический алгоритм (GA) |
| `"none"` | `NoOpOptimizer` | Без оптимизации, возвращает base_genotype; after-test всё равно выполняется |
| `""` / не задан | — | Полный пропуск оптимизации и after-test |

## Заметки

- Все оптимизаторы используют один и тот же `MyEvaluator` для честного сравнения
- Эвалватор использует train_participants (60%) для оптимизации
- Финальная оценка всегда делается на test_participants (40%)
- Результаты сохраняются в `results_experiments/<experiment_id>/`
