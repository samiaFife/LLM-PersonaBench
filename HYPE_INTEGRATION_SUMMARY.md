# HyPE Интеграция — Сводка

> **Актуально после рефакторинга апрель 2026.**
> Старый `if hype_enabled / else evolution` заменён на `OptimizerRegistry`.

## Что реализовано

### SectionalHyPEOptimizer

**Файл:** [`src/meta_optimizer/sectional_hype.py`](src/meta_optimizer/sectional_hype.py)

Наследует `BaseOptimizer`. Оптимизирует genotype по секциям через мета-промпты HyPE:

| Секция | Действие |
|--------|----------|
| `role_definition` | Оптимизируется |
| `critic_formulations` | Оптимизируется |
| `output_format` | Оптимизируется |
| `trait_formulations` | Копируется из base (заглушка) |
| `facet_formulations` | Копируется из base (заглушка) |
| `intensity_modifiers` | Никогда не трогается |

**Ключевые методы:**
- `optimize(base_genotype, evaluator, dev_participants)` — основной пайплайн (реализует `BaseOptimizer`)
- `optimize_genotype()` — deprecated alias для обратной совместимости
- `get_optimization_log()` — список стадий с оценками

### Интеграция в пайплайн

**Файл:** [`src/simulator/person_type_opt.py`](src/simulator/person_type_opt.py)

Выбор метода через конфиг:
```yaml
optimization:
  method: "hype"   # или "evolution", "none"
```

Диспетчеризация через [`OptimizerRegistry`](src/optimizers/__init__.py) — никакого `if/else` в `person_type_opt.py`:
```python
optimizer = OptimizerRegistry.create(optimization_method, model, config)
genotype = optimizer.optimize(base_genotype, evaluator, train_participants)
```

### Конфигурация

Пример: [`configs/examples/hype_quick_test.yaml`](configs/examples/hype_quick_test.yaml)

```yaml
optimization:
  method: "hype"

evolution:
  algorithm: null  # не нужен при method=hype
```

## Как использовать

```bash
# Запуск с HyPE
python tools/launch_experiment.py --config=examples/hype_quick_test.yaml

# Запуск с эволюцией
python tools/launch_experiment.py --config=examples/cluster2_ga.yaml
```

## Сравнение методов

| Аспект | Эволюция (GA) | HyPE |
|--------|---------------|------|
| Вызовов LLM | 16+ (поколения × особи) | 3 (по одному на секцию) |
| Скорость | Медленно | Быстро |
| Итеративность | Да | Нет (один проход) |
| Контроль | Меньше | Больше (кастомные мета-промпты) |

## Оценка результата

- `BASELINE` — базовый genotype до оптимизации
- `AFTER_<SECTION>` — после каждой секции
- `FINAL` — финальный результат
- Если `FINAL < BASELINE` → выводится предупреждение

## Файлы

| Файл | Назначение |
|------|------------|
| [`src/meta_optimizer/sectional_hype.py`](src/meta_optimizer/sectional_hype.py) | Основной оптимизатор |
| [`src/meta_optimizer/hype.py`](src/meta_optimizer/hype.py) | Базовый HyPEOptimizer |
| [`src/meta_optimizer/hyper_templates.py`](src/meta_optimizer/hyper_templates.py) | Шаблоны мета-промптов |
| [`src/optimizers/base.py`](src/optimizers/base.py) | BaseOptimizer, OptimizerRegistry |
| [`src/optimizers/__init__.py`](src/optimizers/__init__.py) | Регистрация всех методов |
| [`configs/examples/hype_quick_test.yaml`](configs/examples/hype_quick_test.yaml) | Быстрый тест |
| [`configs/examples/hype_test.yaml`](configs/examples/hype_test.yaml) | Полный тест |
