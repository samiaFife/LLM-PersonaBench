"""
Sequential section optimization pipeline.
Optimizes sections one by one: critic -> task -> role -> output_format
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
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
from src.evolution.utils import genotype_to_evoprompt_str

# Import variant configs
from experiments.critic_ab_test.critic_variants import (
    CRITIC_VARIANTS,
    TEST_ORDER as CRITIC_ORDER,
)
from experiments.critic_ab_test.variants.task_variants import (
    TASK_VARIANTS,
    TEST_ORDER as TASK_ORDER,
)
from experiments.critic_ab_test.variants.role_variants import (
    ROLE_VARIANTS,
    TEST_ORDER as ROLE_ORDER,
)
from experiments.critic_ab_test.variants.output_variants import (
    OUTPUT_VARIANTS,
    TEST_ORDER as OUTPUT_ORDER,
)


SECTION_CONFIGS = {
    "critic_formulations": {
        "variants": CRITIC_VARIANTS,
        "order": CRITIC_ORDER,
        "display_name": "Critic",
    },
    "task": {
        "variants": TASK_VARIANTS,
        "order": TASK_ORDER,
        "display_name": "Task",
    },
    "role_definition": {
        "variants": ROLE_VARIANTS,
        "order": ROLE_ORDER,
        "display_name": "Role",
    },
    "output_format": {
        "variants": OUTPUT_VARIANTS,
        "order": OUTPUT_ORDER,
        "display_name": "Output Format",
    },
}


def setup_logging(timestamp: str):
    """Setup logging."""
    logs_dir = Path("experiments/critic_ab_test/logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"sequential_opt_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("sequential_opt")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


class SectionOptimizer:
    """Optimizes a single section while keeping others frozen."""

    def __init__(self, config_path: str = "configs/examples/hype_ab_test_quick.yaml"):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger, self.log_file = setup_logging(self.timestamp)
        self.setup()

    def _load_config(self, config_path: str) -> dict:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup(self):
        """Setup environment."""
        self.logger.info("=" * 70)
        self.logger.info("🚀 SETUP: Sequential Section Optimization")
        self.logger.info("=" * 70)

        # Load data
        self.logger.info("Loading data...")
        self.traits = _load_traits(self.config)
        self.facets = _load_facets(self.config)
        self.system = _load_system(self.config)
        self.trait_target_values = _load_trait_target_values(self.config)
        self.facet_target_values = _load_facet_target_values(self.config)

        # Load participants
        import pandas as pd

        self.data_participants = pd.read_csv(self.config["data"]["file_path"])

        cluster = self.config["data"]["clusters"][0]
        n_participants = self.config["data"]["num_participants"]
        cluster_participants = self.data_participants[
            self.data_participants["clusters"] == cluster
        ].iloc[:n_participants]
        train_size = int(n_participants * 0.6)

        self.train_participants = cluster_participants.iloc[:train_size]
        self.test_participants = cluster_participants.iloc[train_size:]

        self.logger.info(
            f"Cluster: {cluster}, Train: {len(self.train_participants)}, Test: {len(self.test_participants)}"
        )

        # Load model
        self.logger.info("Loading model...")
        self.model = get_model(self.config["model"])
        self.logger.info(f"Model: {self.model.model_name}")

        # Load questions
        with open("data/IPIP-NEO/120/questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ipip_neo_questions = data.get("questions")

        self.task = {
            "task": self.system["task"],
            "ipip_neo": self.ipip_neo_questions,
            "response_format": self.system["response_format"],
        }

        # Create base genotype
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

        # Setup evolution config
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
        self.config["evolution"]["participant_batch_size"] = 4

        self.logger.info("Setup completed")

    def _create_evaluator(self):
        """Create evaluator."""
        evo_config = {
            "algorithm": "ga",
            "population_size": 1,
            "num_generations": 1,
            "mutation_rate": 0.0,
            "crossover_rate": 0.0,
            "selection_method": "tournament",
            "llm_for_evolution": self.config["model"]["model_name"],
            "participant_batch_size": 4,
            "genotype_params": self.config["evolution"]["genotype_params"],
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

    def _evaluate_genotype(self, genotype: dict) -> float:
        """Evaluate genotype and return score."""
        evaluator = self._create_evaluator()
        prompt_str = genotype_to_evoprompt_str(genotype, self.config)
        score = evaluator.forward(prompt_str, self.config)
        return float(score)

    def optimize_section(
        self, section_name: str, current_genotype: dict, output_dir: Path
    ) -> tuple:
        """
        Optimize one section.

        Returns:
            (best_genotype, best_result)
        """
        section_config = SECTION_CONFIGS[section_name]
        variants = section_config["variants"]
        order = section_config["order"]
        display_name = section_config["display_name"]

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"🔧 OPTIMIZING SECTION: {display_name}")
        self.logger.info(f"{'=' * 70}")

        print(f"\n{'=' * 70}")
        print(f"🔧 ОПТИМИЗАЦИЯ СЕКЦИИ: {display_name}")
        print(f"{'=' * 70}")

        # Evaluate baseline first
        baseline_score = self._evaluate_genotype(current_genotype)
        self.logger.info(f"Baseline score: {baseline_score:.4f}")
        print(f"📊 Baseline score: {baseline_score:.4f}")

        results = []
        best_score = baseline_score
        best_genotype = current_genotype.copy()
        best_result = None

        # Test each variant
        for variant_id in tqdm(
            order, desc=f"Testing {display_name} variants", unit="variant"
        ):
            if variant_id not in variants:
                continue

            variant = variants[variant_id]
            self.logger.info(f"\nTesting variant: {variant['name']}")
            print(f"\n🧪 Тестирование: {variant['name']}")

            # Create test genotype
            test_genotype = current_genotype.copy()

            if variant["meta_optimizer"]:
                # Generate via HyPE
                print("   🔄 Генерация через HyPE...")
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

                # Optimize the specific section
                if section_name == "critic_formulations":
                    produced_text = optimizer._optimize_critic_with_config(
                        current_genotype[section_name], meta_config
                    )
                elif section_name == "task":
                    produced_text = optimizer._optimize_output_format(
                        current_genotype["task"]
                    )
                elif section_name == "role_definition":
                    produced_text = optimizer._optimize_role(
                        current_genotype["role_definition"]
                    )
                elif section_name == "output_format":
                    produced_text = optimizer._optimize_output_format(
                        current_genotype["output_format"]
                    )
                else:
                    raise ValueError(f"Unknown section: {section_name}")

                test_genotype[section_name] = produced_text

                # Build full prompt with all sections
                full_prompt_sections = {
                    "role": test_genotype["role_definition"],
                    "critic": test_genotype["critic_formulations"],
                    "task": test_genotype["task"],
                    "output_format": test_genotype["output_format"],
                    "traits": "[traits and facets from genotype]",
                }

            else:
                # Use baseline
                print("   📋 Используем baseline...")
                produced_text = variant["prompt"]
                test_genotype[section_name] = produced_text
                full_prompt_sections = None

            # Evaluate
            print("   📊 Оценка...")
            test_score = self._evaluate_genotype(test_genotype)
            improvement = test_score - baseline_score

            print(f"   📊 Score: {test_score:.4f} (Δ {improvement:+.4f})")
            self.logger.info(
                f"Score: {test_score:.4f} (improvement: {improvement:+.4f})"
            )

            # Build result
            result = {
                "section": section_name,
                "variant_id": variant_id,
                "variant_name": variant["name"],
                "baseline_score": baseline_score,
                "test_score": test_score,
                "improvement": improvement,
                "meta_prompt": variant.get("meta_config", {}),
                "produced_prompt": produced_text,
                "full_prompt_sections": full_prompt_sections,
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)

            # Update best if improved
            if test_score > best_score:
                best_score = test_score
                best_genotype = test_genotype.copy()
                best_result = result
                print(f"   ✅ НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ!")
                self.logger.info(f"New best score: {best_score:.4f}")

        # Save section results
        section_dir = output_dir / f"step_{section_name}"
        section_dir.mkdir(exist_ok=True)

        with open(section_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "section": section_name,
                    "baseline_score": baseline_score,
                    "best_score": best_score,
                    "best_variant": best_result["variant_name"]
                    if best_result
                    else None,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save best genotype
        with open(section_dir / "best_genotype.json", "w", encoding="utf-8") as f:
            json.dump(best_genotype, f, indent=2, ensure_ascii=False)

        return best_genotype, best_result

    def run_sequential_optimization(self):
        """Run full sequential optimization pipeline."""
        # Create output directory
        output_dir = (
            Path("experiments/critic_ab_test/results")
            / f"sequential_opt_{self.timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"\nOutput directory: {output_dir}")
        print(f"\n📁 Результаты будут сохранены в: {output_dir}")

        # Define optimization order
        optimization_order = [
            "critic_formulations",
            "task",
            "role_definition",
            "output_format",
        ]

        # Start with base genotype
        current_genotype = self.base_genotype.copy()
        all_results = []

        # Optimize each section
        for i, section_name in enumerate(optimization_order, 1):
            print(f"\n{'=' * 70}")
            print(f"📍 ШАГ {i}/{len(optimization_order)}")
            print(f"{'=' * 70}")

            best_genotype, best_result = self.optimize_section(
                section_name, current_genotype, output_dir
            )

            # Update current genotype with best result
            current_genotype = best_genotype
            all_results.append(best_result)

            # Print summary for this step
            if best_result:
                print(f"\n✅ Шаг {i} завершён. Лучший результат:")
                print(f"   Вариант: {best_result['variant_name']}")
                print(
                    f"   Score: {best_result['test_score']:.4f} (Δ {best_result['improvement']:+.4f})"
                )

        # Save final results
        with open(output_dir / "final_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "optimization_order": optimization_order,
                    "final_genotype": current_genotype,
                    "step_results": all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Print final summary
        print(f"\n{'=' * 70}")
        print("🏁 ПОСЛЕДОВАТЕЛЬНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"{'=' * 70}")
        print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")

        for i, result in enumerate(all_results, 1):
            if result:
                print(f"\nШаг {i} ({result['section']}):")
                print(f"   Вариант: {result['variant_name']}")
                print(f"   Score: {result['test_score']:.4f}")
                print(f"   Улучшение: {result['improvement']:+.4f}")

        print(f"\n💾 Все результаты сохранены в: {output_dir}")
        self.logger.info(f"Optimization completed. Results saved to: {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sequential section optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/hype_ab_test_quick.yaml",
        help="Config file path",
    )

    args = parser.parse_args()

    optimizer = SectionOptimizer(args.config)
    optimizer.run_sequential_optimization()


if __name__ == "__main__":
    main()
