#!/usr/bin/env python3
"""
Прогон оценки: Baseline vs Optimized (critic + task) на 12 участниках с параллелизацией.
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from src.models.registry import get_model
from src.simulator.person_type_opt import (
    _load_traits,
    _load_facets,
    _load_system,
    _load_trait_target_values,
    _load_facet_target_values,
)
from src.utils.personality_match import evaluate_participants_batch


def load_config(config_path: str):
    """Загружает YAML конфиг."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_optimized_system(system_path: str) -> dict:
    """Загружает оптимизированный system."""
    with open(system_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_genotype_from_system(system_config: dict, traits: dict, facets: dict) -> dict:
    """Создает genotype из system конфига."""
    return {
        "system": system_config,
        "traits": traits,
        "facets": facets,
    }


def run_evaluation(
    optimized_system_path: str,
    config_path: str = "configs/examples/hype_ab_test_quick.yaml",
    output_dir: str = None,
    model_name: str = "gpt-4o-mini",
    batch_size: int = 4,
):
    """
    Запускает оценку baseline vs optimized.

    Args:
        optimized_system_path: Путь к system_optimized.json
        config_path: Путь к конфигу
        output_dir: Директория для результатов
        model_name: Модель для оценки
        batch_size: Размер батча для параллельной оценки
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = f"experiments/critic_ab_test/results/evaluation_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 ОЦЕНКА: Baseline vs Optimized (Critic + Task)")
    print("=" * 80)
    print(f"📁 Optimized system: {optimized_system_path}")
    print(f"📁 Config: {config_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {model_name}")
    print(f"⚡ Batch size: {batch_size}")

    # Загружаем конфиг
    config = load_config(config_path)

    # Загружаем данные
    print("\n📊 Загрузка данных...")
    traits = _load_traits(config)
    facets = _load_facets(config)
    baseline_system = _load_system(config)
    trait_targets = _load_trait_target_values(config)
    facet_targets = _load_facet_target_values(config)

    # Загружаем оптимизированный system
    optimized_sections = load_optimized_system(optimized_system_path)
    optimized_system = baseline_system.copy()
    optimized_system.update(optimized_sections)

    # Инициализируем модель
    print("⚙️  Инициализация модели...")
    model_config = {
        "model_name": model_name,
        "provider": "openrouter",
        "temperature": 0.7,
        "timeout": 120,
        "max_retries": 3,
        "max_completion_tokens": 2000,
        "rate_limit": {
            "enabled": True,
            "requests_per_second": 2.0,
            "max_bucket_size": 10,
        },
    }
    model = get_model(model_config)

    # Загружаем данные участников
    print("\n📊 Загрузка участников...")
    data_path = config.data.file_path
    clusters = config.data.clusters

    # Загружаем первый кластер
    import pickle

    cluster_file = Path(data_path) / f"cluster_{clusters[0]}.pkl"
    with open(cluster_file, "rb") as f:
        participants_df = pickle.load(f)

    # Берем первых num_participants
    num_participants = config.data.num_participants
    participants_df = participants_df.head(num_participants)

    print(f"👥 Участников: {len(participants_df)}")
    print(f"📝 Вопросов на участника: 120")

    # Формируем task
    task = {
        "description": "Complete the IPIP-NEO-120 personality questionnaire",
        "questions": list(range(1, 121)),
    }

    results = {
        "timestamp": timestamp,
        "model": model_name,
        "num_participants": num_participants,
        "batch_size": batch_size,
    }

    # === BASELINE ===
    print("\n" + "=" * 80)
    print("🧪 ОЦЕНКА BASELINE")
    print("=" * 80)

    baseline_genotype = build_genotype_from_system(baseline_system, traits, facets)
    baseline_scores = evaluate_participants_batch(
        participants_df, baseline_genotype, task, model, batch_size=batch_size
    )

    baseline_results = []
    for idx, score in enumerate(baseline_scores):
        baseline_results.append(
            {
                "participant_id": idx,
                "overall_score": score.get("similarity", 0),
                "trait_score": score.get("trait_similarity", 0),
                "facet_score": score.get("facet_similarity", 0),
            }
        )

    results["baseline"] = baseline_results

    # === OPTIMIZED ===
    print("\n" + "=" * 80)
    print("🧪 ОЦЕНКА OPTIMIZED (Critic + Task)")
    print("=" * 80)

    optimized_genotype = build_genotype_from_system(optimized_system, traits, facets)
    optimized_scores = evaluate_participants_batch(
        participants_df, optimized_genotype, task, model, batch_size=batch_size
    )

    optimized_results = []
    for idx, score in enumerate(optimized_scores):
        optimized_results.append(
            {
                "participant_id": idx,
                "overall_score": score.get("similarity", 0),
                "trait_score": score.get("trait_similarity", 0),
                "facet_score": score.get("facet_similarity", 0),
            }
        )

    results["optimized"] = optimized_results

    # === АГРЕГАЦИЯ ===
    print("\n📊 Агрегация результатов...")

    df_baseline = pd.DataFrame(baseline_results)
    df_optimized = pd.DataFrame(optimized_results)

    # Считаем средние
    baseline_mean = df_baseline["overall_score"].mean()
    baseline_std = df_baseline["overall_score"].std()
    optimized_mean = df_optimized["overall_score"].mean()
    optimized_std = df_optimized["overall_score"].std()
    improvement = (
        ((optimized_mean - baseline_mean) / baseline_mean * 100)
        if baseline_mean > 0
        else 0
    )

    results["summary"] = {
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "optimized_mean": float(optimized_mean),
        "optimized_std": float(optimized_std),
        "improvement_percent": float(improvement),
    }

    # === СОХРАНЕНИЕ ===
    # JSON с полными результатами
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV для анализа
    df_comparison = pd.DataFrame(
        {
            "participant_id": range(num_participants),
            "baseline_score": df_baseline["overall_score"].values,
            "optimized_score": df_optimized["overall_score"].values,
            "improvement": df_optimized["overall_score"].values
            - df_baseline["overall_score"].values,
        }
    )
    csv_file = output_path / "comparison.csv"
    df_comparison.to_csv(csv_file, index=False)

    # Детальные результаты
    df_baseline["variant"] = "baseline"
    df_optimized["variant"] = "optimized"
    df_all = pd.concat([df_baseline, df_optimized], ignore_index=True)
    df_all.to_csv(output_path / "detailed_results.csv", index=False)

    # === ВЫВОД РЕЗУЛЬТАТОВ ===
    print("\n" + "=" * 80)
    print("📋 РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"\n📊 Baseline:")
    print(f"   Mean: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"   Trait: {df_baseline['trait_score'].mean():.4f}")
    print(f"   Facet: {df_baseline['facet_score'].mean():.4f}")

    print(f"\n📊 Optimized (Critic + Task):")
    print(f"   Mean: {optimized_mean:.4f} ± {optimized_std:.4f}")
    print(f"   Trait: {df_optimized['trait_score'].mean():.4f}")
    print(f"   Facet: {df_optimized['facet_score'].mean():.4f}")

    print(f"\n🎯 Улучшение: {improvement:+.2f}%")

    if improvement > 0:
        print("✅ Optimized лучше baseline!")
    elif improvement < 0:
        print("⚠️  Optimized хуже baseline")
    else:
        print("➡️  Результаты одинаковые")

    # Сравнение по участникам
    print("\n📋 По участникам:")
    print("-" * 60)
    print(f"{'ID':>4} {'Baseline':>10} {'Optimized':>10} {'Diff':>10}")
    print("-" * 60)
    for _, row in df_comparison.iterrows():
        diff = row["improvement"]
        sign = "+" if diff > 0 else ""
        print(
            f"{int(row['participant_id']):>4} {row['baseline_score']:>10.4f} {row['optimized_score']:>10.4f} {sign}{diff:>9.4f}"
        )
    print("-" * 60)

    print(f"\n💾 Результаты сохранены в: {output_dir}")
    print(f"   - evaluation_results.json")
    print(f"   - comparison.csv")
    print(f"   - detailed_results.csv")

    return results


def main():
    """Точка входа."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Оценка Baseline vs Optimized на 12 участниках"
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
        help="Путь к конфигу",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Директория для результатов"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Модель для оценки"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Размер батча для параллельной оценки (по умолчанию: 4)",
    )

    args = parser.parse_args()

    # Проверяем что файл существует
    if not Path(args.system_path).exists():
        print(f"❌ Файл не найден: {args.system_path}")
        print("\n💡 Сначала сгенерируй оптимизированные секции:")
        print("   python experiments/critic_ab_test/generate_optimized_sections.py")
        return

    # Запускаем оценку
    run_evaluation(
        optimized_system_path=args.system_path,
        config_path=args.config,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
