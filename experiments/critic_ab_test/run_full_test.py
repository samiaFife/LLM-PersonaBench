#!/usr/bin/env python3
"""
Запускает тестирование с оптимизированными секциями (critic + task) на 12 участниках.
Сравнивает: baseline system vs optimized system
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from tqdm import tqdm
from src.models.registry import get_model
from src.simulator.person_type_opt import (
    _load_traits,
    _load_facets,
    _load_system,
    _load_trait_target_values,
    _load_facet_target_values,
)
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.parse_args import parse_args_from_yaml


def load_optimized_system(system_path: str) -> dict:
    """Загружает оптимизированный system."""
    with open(system_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_system_config(base_system: dict, optimized_sections: dict) -> dict:
    """Создает конфиг system с оптимизированными секциями."""
    config = base_system.copy()
    config.update(optimized_sections)
    return config


async def evaluate_participant(
    model,
    evaluator: MyEvaluator,
    participant_id: int,
    system_config: dict,
    traits: dict,
    facets: dict,
    trait_targets: dict,
    facet_targets: dict,
    logger=None,
) -> dict:
    """
    Оценивает одного участника с заданным system конфигом.

    Returns:
        dict с результатами оценки
    """
    from src.simulator.person_type_opt import simulate_person_type_opt

    # Получаем целевые значения для участника
    participant_trait_targets = {
        trait: trait_targets[trait][participant_id] for trait in trait_targets
    }
    participant_facet_targets = {
        facet: facet_targets[facet][participant_id] for facet in facet_targets
    }

    # Симулируем ответы
    responses = await simulate_person_type_opt(
        model=model,
        traits=traits,
        facets=facets,
        system=system_config,
        trait_target_values=participant_trait_targets,
        facet_target_values=participant_facet_targets,
        num_questions=120,
        logger=logger,
    )

    # Оцениваем
    result = evaluator.evaluate_single_participant(
        participant_id=participant_id,
        responses=responses,
        trait_targets=participant_trait_targets,
        facet_targets=participant_facet_targets,
    )

    return result


async def run_comparison_test(
    optimized_system_path: str,
    config_path: str = "configs/examples/hype_ab_test_quick.yaml",
    output_dir: str = None,
    model_name: str = "gpt-4o-mini",
):
    """
    Запускает сравнительный тест: baseline vs optimized.

    Args:
        optimized_system_path: Путь к system_optimized.json
        config_path: Путь к конфигу
        output_dir: Директория для результатов
        model_name: Модель для тестирования
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = f"experiments/critic_ab_test/results/full_test_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 ПОЛНЫЙ ТЕСТ: Baseline vs Optimized")
    print("=" * 70)
    print(f"📁 Optimized system: {optimized_system_path}")
    print(f"📁 Config: {config_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {model_name}")

    # Загружаем конфиг
    config = parse_args_from_yaml(config_path)

    # Загружаем данные
    print("\n📊 Загрузка данных...")
    traits = _load_traits(config)
    facets = _load_facets(config)
    base_system = _load_system(config)
    trait_targets = _load_trait_target_values(config)
    facet_targets = _load_facet_target_values(config)

    # Загружаем оптимизированный system
    optimized_sections = load_optimized_system(optimized_system_path)
    optimized_system = create_system_config(base_system, optimized_sections)

    # Инициализируем модель и evaluator
    print("⚙️  Инициализация...")
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
    evaluator = MyEvaluator(
        trait_targets=trait_targets,
        facet_targets=facet_targets,
        aggregate_method="mean",
    )

    # Получаем список участников
    num_participants = config.data.num_participants
    print(f"\n👥 Участников: {num_participants}")

    results = {
        "timestamp": timestamp,
        "model": model_name,
        "num_participants": num_participants,
        "baseline": {"participants": []},
        "optimized": {"participants": []},
    }

    # Тестируем baseline
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ BASELINE")
    print("=" * 70)

    for pid in tqdm(range(num_participants), desc="Baseline"):
        result = await evaluate_participant(
            model=model,
            evaluator=evaluator,
            participant_id=pid,
            system_config=base_system,
            traits=traits,
            facets=facets,
            trait_targets=trait_targets,
            facet_targets=facet_targets,
        )
        results["baseline"]["participants"].append(result)

    # Тестируем optimized
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ OPTIMIZED")
    print("=" * 70)

    for pid in tqdm(range(num_participants), desc="Optimized"):
        result = await evaluate_participant(
            model=model,
            evaluator=evaluator,
            participant_id=pid,
            system_config=optimized_system,
            traits=traits,
            facets=facets,
            trait_targets=trait_targets,
            facet_targets=facet_targets,
        )
        results["optimized"]["participants"].append(result)

    # Агрегируем результаты
    print("\n📊 Агрегация результатов...")

    # Считаем средние метрики
    baseline_scores = [p["overall_score"] for p in results["baseline"]["participants"]]
    optimized_scores = [
        p["overall_score"] for p in results["optimized"]["participants"]
    ]

    results["summary"] = {
        "baseline_mean": sum(baseline_scores) / len(baseline_scores),
        "baseline_std": pd.Series(baseline_scores).std(),
        "optimized_mean": sum(optimized_scores) / len(optimized_scores),
        "optimized_std": pd.Series(optimized_scores).std(),
        "improvement": (sum(optimized_scores) - sum(baseline_scores))
        / len(baseline_scores)
        * 100,
    }

    # Сохраняем результаты
    results_file = output_path / "test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Создаем DataFrame для удобного анализа
    df_data = []
    for variant, data in [
        ("baseline", results["baseline"]),
        ("optimized", results["optimized"]),
    ]:
        for pid, presult in enumerate(data["participants"]):
            df_data.append(
                {
                    "variant": variant,
                    "participant_id": pid,
                    "overall_score": presult["overall_score"],
                    "trait_score": presult.get("trait_score", 0),
                    "facet_score": presult.get("facet_score", 0),
                }
            )

    df = pd.DataFrame(df_data)
    csv_file = output_path / "results.csv"
    df.to_csv(csv_file, index=False)

    # Печатаем сводку
    print("\n" + "=" * 70)
    print("📋 РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(
        f"Baseline mean:   {results['summary']['baseline_mean']:.3f} ± {results['summary']['baseline_std']:.3f}"
    )
    print(
        f"Optimized mean:  {results['summary']['optimized_mean']:.3f} ± {results['summary']['optimized_std']:.3f}"
    )
    print(f"Improvement:     {results['summary']['improvement']:+.1f}%")

    print(f"\n💾 Результаты сохранены в: {output_dir}")
    print(f"   - test_results.json (полные данные)")
    print(f"   - results.csv (CSV для анализа)")

    return results


def main():
    """Точка входа."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Тестирование с оптимизированными секциями на 12 участниках"
    )
    parser.add_argument(
        "--system-path",
        type=str,
        required=True,
        help="Путь к system_optimized.json с оптимизированными секциями",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/hype_ab_test_quick.yaml",
        help="Путь к конфигу (по умолчанию: hype_ab_test_quick.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Директория для результатов"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Модель для тестирования"
    )

    args = parser.parse_args()

    # Проверяем что файл существует
    if not Path(args.system_path).exists():
        print(f"❌ Файл не найден: {args.system_path}")
        print("💡 Сначала сгенерируй оптимизированные секции:")
        print("   python experiments/critic_ab_test/generate_optimized_sections.py")
        return

    # Запускаем тест
    asyncio.run(
        run_comparison_test(
            optimized_system_path=args.system_path,
            config_path=args.config,
            output_dir=args.output_dir,
            model_name=args.model,
        )
    )


if __name__ == "__main__":
    main()
