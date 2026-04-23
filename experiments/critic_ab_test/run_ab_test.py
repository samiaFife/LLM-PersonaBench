#!/usr/bin/env python3
"""
A/B тестирование различных вариантов critic_formulations.

Запускает оптимизацию с разными мета-промптами и сравнивает результаты.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from tqdm import tqdm
from experiments.critic_ab_test.critic_variants import (
    CRITIC_VARIANTS,
    TEST_ORDER,
    BASE_CONFIG,
)
from src.models.registry import get_model
from src.meta_optimizer import (
    SectionalHyPEOptimizer,
    HypeMetaPromptConfig,
    PromptSectionSpec,
)
from src.simulator.person_type_opt import (
    _load_traits,
    _load_facets,
    _load_system,
    _load_trait_target_values,
    _load_facet_target_values,
)
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.parse_args import parse_args_from_yaml
from src.evolution.utils import genotype_to_evoprompt_str, parse_str_to_genotype
from src.utils.personality_match import (
    evaluate_participants_batch,
    normalize_participant_score,
    aggregate_stage_metrics,
)


# Setup logging
def setup_logging(timestamp: str):
    """Setup logging to file and console."""
    logs_dir = Path("experiments/critic_ab_test/logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f"ab_test_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger("critic_ab_test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


class CriticABTester:
    """Тестирует различные варианты critic через A/B тестирование."""

    def __init__(self, config_path: str = "configs/examples/hype_test.yaml"):
        self.config = self._load_config(config_path)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger, self.log_file = setup_logging(self.timestamp)
        self.stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "total_time": 0,
            "start_time": None,
            "end_time": None,
        }

    def _load_config(self, config_path: str) -> dict:
        """Загружает YAML конфиг."""
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup(self):
        """Настройка окружения для тестирования."""
        self.stats["start_time"] = time.time()
        self.logger.info("=" * 70)
        self.logger.info("🚀 НАСТРОЙКА A/B ТЕСТИРОВАНИЯ CRITIC")
        self.logger.info("=" * 70)

        print("=" * 70)
        print("🚀 НАСТРОЙКА A/B ТЕСТИРОВАНИЯ CRITIC")
        print("=" * 70)

        # Загрузка данных
        print("\n📦 Загрузка данных...")
        self.logger.info("Loading data...")
        self.traits = _load_traits(self.config)
        self.facets = _load_facets(self.config)
        self.system = _load_system(self.config)
        self.trait_target_values = _load_trait_target_values(self.config)
        self.facet_target_values = _load_facet_target_values(self.config)

        # Загрузка участников
        import pandas as pd

        self.data_participants = pd.read_csv(self.config["data"]["file_path"])

        cluster = BASE_CONFIG["cluster"]
        n_participants = BASE_CONFIG["num_participants"]

        cluster_participants = self.data_participants[
            self.data_participants["clusters"] == cluster
        ].iloc[:n_participants]
        train_size = int(n_participants * 0.6)

        self.train_participants = cluster_participants.iloc[:train_size]
        self.test_participants = cluster_participants.iloc[train_size:]

        info_msg = f"Cluster: {cluster}, Train: {len(self.train_participants)}, Test: {len(self.test_participants)}"
        self.logger.info(info_msg)
        print(f"   Кластер: {cluster}")
        print(f"   Train: {len(self.train_participants)} участников")
        print(f"   Test: {len(self.test_participants)} участников")

        # Загрузка модели
        print("\n🤖 Загрузка модели...")
        self.logger.info("Loading model...")
        self.model = get_model(self.config["model"])
        model_msg = f"Model loaded: {self.model.model_name}"
        self.logger.info(model_msg)
        print(f"   Модель: {self.model.model_name}")

        # Загрузка вопросов
        print("\n📋 Загрузка IPIP-NEO...")
        self.logger.info("Loading IPIP-NEO questions...")
        with open("data/IPIP-NEO/120/questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ipip_neo_questions = data.get("questions")
        self.logger.info(f"Loaded {len(self.ipip_neo_questions)} questions")
        print(f"   Вопросов: {len(self.ipip_neo_questions)}")

        # Task
        self.task = {
            "task": self.system["task"],
            "ipip_neo": self.ipip_neo_questions,
            "response_format": self.system["response_format"],
        }

        # Создание base_genotype
        cluster_trait_targets = self.trait_target_values.get(cluster, {})
        cluster_facet_targets = self.facet_target_values.get(cluster, {})
        trait_formulations = self.traits[cluster]
        facet_formulations = self.facets[cluster]

        self.base_genotype = {
            "role_definition": self.system["role"],
            "trait_formulations": trait_formulations,
            "facet_formulations": facet_formulations,
            "intensity_modifiers": self.system["intensity_modifiers"],
            "critic_formulations": self.system["critic_internal"],
            "template_structure": self.system["template_structure"],
            "trait_targets": {
                k: cluster_trait_targets[k]
                for k in trait_formulations
                if k in cluster_trait_targets
            },
            "facet_targets": {
                k: cluster_facet_targets[k]
                for k in facet_formulations
                if k in cluster_facet_targets
            },
        }

        # Add evolution config for MyEvaluator
        if "evolution" not in self.config:
            self.config["evolution"] = {}
        self.config["evolution"]["genotype_params"] = {
            "role_definition": True,
            "trait_formulations": True,
            "facet_formulations": True,
            "critic_formulations": True,
            "intensity_modifiers": False,
            "template_structure": False,
        }
        self.config["evolution"]["participant_batch_size"] = 4  # Parallel evaluation

        self.logger.info("Setup completed successfully")
        print("\n✅ Настройка завершена")

    def _create_evaluator(self):
        """Создаёт evaluator для оценки."""
        # Create minimal evolution config for evaluator (we're testing HyPE, not evolution)
        evo_config = {
            "algorithm": "ga",  # Dummy value, not used for HyPE
            "population_size": 1,
            "num_generations": 1,
            "mutation_rate": 0.0,
            "crossover_rate": 0.0,
            "selection_method": "tournament",
            "llm_for_evolution": self.config["model"]["model_name"],
            "participant_batch_size": 4,
            "genotype_params": {
                "role_definition": True,
                "trait_formulations": True,
                "facet_formulations": True,
                "critic_formulations": True,
                "intensity_modifiers": False,
                "template_structure": False,
            },
        }
        evo_args = parse_args_from_yaml(evo_config)

        evaluator = MyEvaluator(
            evo_args,
            self.task,
            self.model,
            self.system["intensity_modifiers"],
            template_genotype=self.base_genotype,
            config=self.config,
        )
        evaluator.dev_participants = self.train_participants
        return evaluator

    def test_variant(self, variant_id: str):
        """Тестирует один вариант critic."""
        variant = CRITIC_VARIANTS[variant_id]
        self.stats["total_tests"] += 1

        print(f"\n{'=' * 70}")
        print(f"🧪 ТЕСТИРОВАНИЕ: {variant['name']}")
        print(f"   {variant['description']}")
        print(f"{'=' * 70}")

        self.logger.info(f"Testing variant: {variant['name']} ({variant_id})")

        start_time = time.time()

        try:
            # Копируем base_genotype
            test_genotype = self.base_genotype.copy()

            if variant["meta_optimizer"]:
                # Генерируем через HyPE
                print("   🔄 Генерация через HyPE...")
                self.logger.info("Generating via HyPE...")

                # Создаём кастомный оптимизатор с нужным meta_config
                meta_config = HypeMetaPromptConfig(
                    target_prompt_form="instructional ",
                    include_role=True,
                    section_specs=[
                        PromptSectionSpec(
                            name=variant["meta_config"]["section_specs"][0]["name"],
                            description=variant["meta_config"]["section_specs"][0][
                                "description"
                            ],
                        )
                    ],
                    constraints=variant["meta_config"].get("constraints", []),
                    recommendations=variant["meta_config"].get("recommendations", []),
                )

                optimizer = SectionalHyPEOptimizer(model=self.model, config=self.config)

                # Оптимизируем только critic
                optimized_critic = optimizer._optimize_critic_with_config(
                    self.base_genotype["critic_formulations"], meta_config
                )

                test_genotype["critic_formulations"] = optimized_critic
                self.logger.info("HyPE optimization completed")

            else:
                # Используем готовый промпт
                print("   📋 Используем baseline...")
                self.logger.info("Using baseline prompt")
                test_genotype["critic_formulations"] = variant["prompt"]

            # Оценка
            print("   📊 Оценка genotype...")
            self.logger.info("Evaluating genotype...")
            evaluator = self._create_evaluator()

            # Baseline
            base_score = self._evaluate_genotype(
                self.base_genotype, evaluator, "baseline"
            )
            print(f"   📊 Baseline: {base_score:.4f}")
            self.logger.info(f"Baseline score: {base_score:.4f}")

            # Тестовый вариант
            test_score = self._evaluate_genotype(test_genotype, evaluator, variant_id)
            print(f"   📊 {variant_id}: {test_score:.4f}")
            self.logger.info(f"Test score: {test_score:.4f}")

            # Разница
            improvement = test_score - base_score
            print(f"   📈 Изменение: {improvement:+.4f}")
            self.logger.info(f"Improvement: {improvement:+.4f}")

            elapsed = time.time() - start_time
            self.logger.info(f"Test completed in {elapsed:.2f} seconds")

            # Сохраняем результат
            result = {
                "variant_id": variant_id,
                "variant_name": variant["name"],
                "baseline_score": base_score,
                "test_score": test_score,
                "improvement": improvement,
                "critic_text": test_genotype["critic_formulations"],
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }

            self.results.append(result)
            self.stats["successful_tests"] += 1

            return result

        except Exception as e:
            self.stats["failed_tests"] += 1
            import traceback

            tb_str = traceback.format_exc()
            error_msg = f"Error testing {variant_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback:\n{tb_str}")
            print(f"   ❌ ОШИБКА: {e}")
            print(f"   📋 Подробности в логе: {self.log_file}")

            # Save error result
            result = {
                "variant_id": variant_id,
                "variant_name": variant["name"],
                "baseline_score": None,
                "test_score": None,
                "improvement": None,
                "critic_text": None,
                "elapsed_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            }
            self.results.append(result)
            return result

    def _evaluate_genotype(self, genotype: dict, evaluator, label: str) -> float:
        """Оценивает genotype."""
        prompt_str = genotype_to_evoprompt_str(genotype, self.config)
        score = evaluator.forward(prompt_str, self.config)
        return float(score)

    def run_all_tests(self):
        """Запускает все тесты."""
        print(f"\n🎯 БУДЕТ ПРОТЕСТИРОВАНО {len(TEST_ORDER)} ВАРИАНТОВ")
        self.logger.info(f"Starting A/B test with {len(TEST_ORDER)} variants")

        # Use tqdm for progress bar
        for variant_id in tqdm(TEST_ORDER, desc="Testing variants", unit="variant"):
            if variant_id not in CRITIC_VARIANTS:
                msg = f"Skipping {variant_id} - not found in config"
                self.logger.warning(msg)
                print(f"⚠️  Пропуск {variant_id} - не найден в конфиге")
                continue

            self.test_variant(variant_id)

        self.stats["end_time"] = time.time()
        self.stats["total_time"] = self.stats["end_time"] - self.stats["start_time"]

        self.logger.info(
            f"All tests completed in {self.stats['total_time']:.2f} seconds"
        )
        self.logger.info(
            f"Successful: {self.stats['successful_tests']}, Failed: {self.stats['failed_tests']}"
        )

        print(f"\n{'=' * 70}")
        print("🏁 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print(f"{'=' * 70}")

    def save_results(self):
        """Сохраняет результаты в отдельную папку."""
        # Создаём папку с timestamp
        output_dir = (
            Path("experiments/critic_ab_test/results") / f"results_{self.timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON с полными результатами
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "config": BASE_CONFIG,
                    "stats": self.stats,
                    "results": self.results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # CSV для анализа
        csv_path = output_dir / "summary.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)

        # JSON с краткой статистикой
        stats_path = output_dir / "stats.json"

        # Calculate additional statistics
        successful_results = [r for r in self.results if r.get("status") == "success"]
        if successful_results:
            improvements = [r["improvement"] for r in successful_results]
            stats_summary = {
                "timestamp": self.timestamp,
                "total_variants": len(TEST_ORDER),
                "successful_tests": self.stats["successful_tests"],
                "failed_tests": self.stats["failed_tests"],
                "total_time_seconds": self.stats["total_time"],
                "average_time_per_variant": self.stats["total_time"] / len(TEST_ORDER)
                if TEST_ORDER
                else 0,
                "best_variant": max(successful_results, key=lambda x: x["improvement"])[
                    "variant_name"
                ]
                if successful_results
                else None,
                "best_improvement": max(improvements) if improvements else None,
                "average_improvement": sum(improvements) / len(improvements)
                if improvements
                else None,
                "worst_improvement": min(improvements) if improvements else None,
                "variants_better_than_baseline": len(
                    [i for i in improvements if i > 0]
                ),
                "variants_worse_than_baseline": len([i for i in improvements if i < 0]),
            }

            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Stats summary saved: {stats_path}")

        self.logger.info(f"Results saved to: {output_dir}")
        print(f"\n💾 Результаты сохранены в папку:")
        print(f"   {output_dir}/")
        print(f"   ├── results.json")
        print(f"   ├── summary.csv")
        print(f"   └── stats.json")
        print(f"\n📝 Лог сохранён: {self.log_file}")

    def print_summary(self):
        """Выводит сводку результатов."""
        print(f"\n{'=' * 70}")
        print("📊 СВОДКА РЕЗУЛЬТАТОВ")
        print(f"{'=' * 70}")

        self.logger.info("=" * 70)
        self.logger.info("RESULTS SUMMARY")
        self.logger.info("=" * 70)

        if not self.results:
            print("❌ Нет результатов")
            self.logger.error("No results to display")
            return

        # Сортируем по improvement (только успешные)
        successful_results = [r for r in self.results if r.get("status") == "success"]
        sorted_results = sorted(
            successful_results, key=lambda x: x["improvement"], reverse=True
        )

        print(f"\n{'Rank':<6} {'Variant':<30} {'Baseline':<10} {'Test':<10} {'Δ':<10}")
        print("-" * 70)

        self.logger.info(
            f"{'Rank':<6} {'Variant':<30} {'Baseline':<10} {'Test':<10} {'Delta':<10}"
        )
        self.logger.info("-" * 70)

        for i, result in enumerate(sorted_results, 1):
            line = (
                f"{i:<6} {result['variant_name']:<30} "
                f"{result['baseline_score']:<10.4f} "
                f"{result['test_score']:<10.4f} "
                f"{result['improvement']:+.4f}"
            )
            print(line)
            self.logger.info(line)

        if sorted_results:
            # Лучший результат
            best = sorted_results[0]
            print(f"\n🏆 ЛУЧШИЙ ВАРИАНТ: {best['variant_name']}")
            print(f"   Улучшение: {best['improvement']:+.4f}")
            print(f"   Итоговый score: {best['test_score']:.4f}")

            self.logger.info(f"BEST VARIANT: {best['variant_name']}")
            self.logger.info(f"Improvement: {best['improvement']:+.4f}")
            self.logger.info(f"Final score: {best['test_score']:.4f}")

            # Статистика
            improvements = [r["improvement"] for r in successful_results]
            avg_improvement = sum(improvements) / len(improvements)

            print(f"\n📈 СТАТИСТИКА:")
            print(f"   Среднее улучшение: {avg_improvement:+.4f}")
            print(f"   Лучшее: {max(improvements):+.4f}")
            print(f"   Худшее: {min(improvements):+.4f}")
            print(
                f"   Успешных тестов: {self.stats['successful_tests']}/{self.stats['total_tests']}"
            )
            print(f"   Общее время: {self.stats['total_time']:.2f} сек")

            self.logger.info(f"Statistics:")
            self.logger.info(f"  Average improvement: {avg_improvement:+.4f}")
            self.logger.info(f"  Best: {max(improvements):+.4f}")
            self.logger.info(f"  Worst: {min(improvements):+.4f}")
            self.logger.info(
                f"  Successful: {self.stats['successful_tests']}/{self.stats['total_tests']}"
            )
            self.logger.info(f"  Total time: {self.stats['total_time']:.2f} seconds")

        # Failed tests
        failed_results = [r for r in self.results if r.get("status") == "failed"]
        if failed_results:
            print(f"\n❌ НЕУДАЧНЫЕ ТЕСТЫ:")
            self.logger.warning("Failed tests:")
            for r in failed_results:
                msg = f"   - {r['variant_name']}: {r.get('error', 'Unknown error')}"
                print(msg)
                self.logger.warning(msg)


def main():
    """Основной запуск."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B тестирование critic variants")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/hype_test.yaml",
        help="Путь к конфигурационному файлу",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Быстрый тест с минимальными параметрами"
    )

    args = parser.parse_args()

    # Если --quick, используем быстрый конфиг
    config_path = (
        "configs/examples/hype_ab_test_quick.yaml" if args.quick else args.config
    )

    print(f"📋 Используем конфиг: {config_path}")

    tester = CriticABTester(config_path)

    # Настройка
    tester.setup()

    # Запуск тестов
    tester.run_all_tests()

    # Сводка
    tester.print_summary()

    # Сохранение
    tester.save_results()

    print(f"\n{'=' * 70}")
    print("✅ A/B ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
