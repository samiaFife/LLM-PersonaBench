# LLM-PersonaBench

Бенчмарк для оценки того, насколько хорошо LLM может симулировать личность человека при прохождении психометрического опросника **IPIP-NEO-120** (Big Five / OCEAN).

Основная идея: задать модели «генотип» — набор текстовых описаний черт личности — и измерить, насколько её ответы на 120 вопросов совпадают с реальными ответами людей из соответствующего психометрического кластера. Затем попробовать **оптимизировать** этот генотип разными методами и сравнить результаты.

---

## Что здесь есть

- **Скоринг**: LLM проходит IPIP-NEO-120, её ответы сравниваются с реальными данными участников по метрикам similarity, Pearson correlation, MAE
- **Два встроенных метода оптимизации генотипа**: генетический алгоритм (`evolution`) и HyPE (`hype`)
- **Единый интерфейс** для подключения своего метода оптимизации — 3 шага, без правки основного пайплайна

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Данные

Датасет с реальными ответами участников IPIP-NEO-120, кластеризованными по психотипам:

- Скачать `df_ipipneo_120_clusters` (CSV) — ссылка в [`data/README.md`](data/README.md)
- Положить в `data/raw/df_ipipneo_120_clusters`

### 3. API ключ

Бенчмарк работает через **OpenRouter** (любая модель) или **Yandex Cloud API**.

```bash
# Для OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Для Yandex Cloud
export YC_API_KEY="..."
```

### 4. Запуск

```bash
# Быстрый тест — HyPE оптимизация, кластер 2, 10 участников
python tools/launch_experiment.py --config=examples/hype_quick_test.yaml

# Генетический алгоритм
python tools/launch_experiment.py --config=examples/cluster2_ga.yaml
```

Результаты сохраняются в `results_experiments/<experiment_id>/`.

---

## Как подключить свой метод оптимизации

Весь пайплайн работает через `OptimizerRegistry`. Добавление нового метода — **3 шага, без правки `person_type_opt.py`**.

### Шаг 1 — Создай класс оптимизатора

```python
# src/optimizers/my_method.py

from src.optimizers.base import BaseOptimizer

class MyMethodOptimizer(BaseOptimizer):
    """
    Твой метод оптимизации генотипа.
    
    self.model  — LLM (тот же что для симуляции, или отдельный из evolution.llm_for_evolution)
    self.config — полный конфиг эксперимента (dict)
    """

    def optimize(
        self,
        base_genotype: dict,
        evaluator,        # MyEvaluator — вызови evaluator.forward(prompt_str, config) → float
        dev_participants, # pandas DataFrame, train-часть (60% участников кластера)
    ) -> dict:
        """
        Оптимизировать генотип.
        
        Returns:
            Оптимизированный генотип (dict той же структуры что base_genotype)
        """
        # Пример: просто вернуть базовый генотип без изменений
        return base_genotype.copy()
```

**Структура генотипа** (`base_genotype`):

```python
{
    "role_definition": str,          # системная роль LLM
    "trait_formulations": {          # описания 5 черт OCEAN
        "openness": str,
        "conscientiousness": str,
        "extraversion": str,
        "agreeableness": str,
        "neuroticism": str,
    },
    "facet_formulations": {          # описания 30 фасетов
        "facet_imagination": str,
        # ...
    },
    "intensity_modifiers": {...},    # НЕ ТРОГАТЬ — фиксированные модификаторы
    "critic_formulations": str,      # инструкции для внутренней проверки
    "template_structure": str,       # шаблон промпта
    "trait_targets": {...},          # целевые значения черт для кластера
    "facet_targets": {...},          # целевые значения фасетов для кластера
}
```

**Как использовать evaluator** для оценки кандидата:

```python
from src.evolution.utils import genotype_to_evoprompt_str

# Конвертируй генотип в строку
prompt_str = genotype_to_evoprompt_str(candidate_genotype, self.config)

# Получи скор (чем больше — тем лучше, диапазон ~0..1)
score = evaluator.forward(prompt_str, self.config)
```

### Шаг 2 — Зарегистрируй метод

```python
# src/optimizers/__init__.py — добавь две строки:

from src.optimizers.my_method import MyMethodOptimizer
OptimizerRegistry.register("my_method", MyMethodOptimizer)
```

### Шаг 3 — Создай конфиг

```yaml
# configs/examples/my_method_test.yaml

name: "my_method_test"

model:
  model_name: "mistralai/mistral-7b-instruct:free"
  provider: "openrouter"
  temperature: 0.7
  timeout: 120
  max_retries: 3

optimization:
  method: "my_method"   # ← имя из OptimizerRegistry.register()

# evolution нужен только для инициализации MyEvaluator (оставь пустым если не используешь GA)
evolution:
  algorithm: null

data:
  file_path: "data/raw/df_ipipneo_120_clusters"
  clusters: [2]          # кластер(ы) для теста
  num_participants: 20   # train=12, test=8

prompt:
  traits_path: "src/prompt/traits.json"
  facets_path: "src/prompt/facets.json"
  system_path: "src/prompt/system.json"

experiment:
  seed: 2026
  save_every_generation: false
```

### Запуск

```bash
python tools/launch_experiment.py --config=examples/my_method_test.yaml
```

---

## Метрики скоринга

Для каждого участника кластера LLM проходит все 120 вопросов, ответы сравниваются с реальными:

| Метрика | Описание |
|---------|----------|
| `mean_similarity` | Основная метрика — доля совпадающих ответов (1-5 шкала) |
| `mean_pearson_corr` | Корреляция Пирсона между симулированными и реальными ответами |
| `mean_avg_diff` | Средняя абсолютная разница ответов |
| `mean_mae_35` | MAE по 35 измерениям (5 черт + 30 фасетов) |
| `mean_similarity_traits` | Similarity только по 5 чертам OCEAN |
| `mean_similarity_facets` | Similarity только по 30 фасетам |

Оптимизация ведётся на **train** (60% участников кластера), финальная оценка — на **test** (40%).

---

## Структура проекта

```
src/
├── optimizers/          # Единый интерфейс оптимизаторов
│   ├── base.py          # BaseOptimizer, OptimizerRegistry, NoOpOptimizer
│   ├── evolution.py     # EvolutionOptimizer (обёртка над GAEvoluter)
│   └── __init__.py      # Регистрация всех методов
├── meta_optimizer/      # HyPE — метапромптная оптимизация
│   ├── sectional_hype.py
│   ├── hype.py
│   └── hyper_templates.py
├── evolution/           # Генетический алгоритм (GA)
│   ├── evoluter.py
│   ├── my_evaluator.py  # MyEvaluator — скоринг промптов
│   └── ...
├── simulator/
│   └── person_type_opt.py  # Главный цикл эксперимента
├── models/              # Провайдеры LLM (cloud, openrouter)
├── prompt/              # Базовые генотипы (traits, facets, system)
└── utils/               # Метрики, парсинг, утилиты

tools/
└── launch_experiment.py # Точка входа

configs/
├── examples/            # Готовые конфиги для быстрого старта
└── experiments/         # Конфиги реальных экспериментов

data/
├── IPIP-NEO/120/        # Вопросы опросника
└── raw/                 # Датасет с ответами участников (скачать отдельно)
```

---

## Встроенные методы оптимизации

| `optimization.method` | Класс | Описание |
|-----------------------|-------|----------|
| `"evolution"` | `EvolutionOptimizer` | Генетический алгоритм: кроссовер + мутация генотипов через LLM |
| `"hype"` | `SectionalHyPEOptimizer` | HyPE: посекционная оптимизация через мета-промпт |
| `"none"` | `NoOpOptimizer` | Без оптимизации, but after-test всё равно выполняется |
| `""` / не задан | — | Полный пропуск оптимизации и after-test |

---

## Провайдеры моделей

| `provider` | Описание |
|------------|----------|
| `"openrouter"` | [OpenRouter](https://openrouter.ai) — доступ к сотням моделей через один API |
| `"cloud"` | Yandex Cloud API (внутренний) |

Пример конфига для OpenRouter с бесплатной моделью — [`configs/examples/hype_quick_test.yaml`](configs/examples/hype_quick_test.yaml).

---

## Подробная документация

- [`OPTIMIZATION_PIPELINE.md`](OPTIMIZATION_PIPELINE.md) — детальное описание пайплайна, интерфейсов и примеры
