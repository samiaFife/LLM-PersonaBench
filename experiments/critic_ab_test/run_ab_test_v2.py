#!/usr/bin/env python3
"""
A/B test with new meta-prompts for critic and task sections.
12 participants, parallel evaluation (batch_size=4).
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

# Import new meta-prompts
from experiments.critic_ab_test.new_meta_prompts import (
    CRITIC_META_PROMPT,
    CRITIC_BASELINE,
    TASK_META_PROMPT,
    TASK_BASELINE,
    ORIGINAL_TEXTS,
)


def setup_logging(timestamp: str):
    """Setup logging."""
    logs_dir = Path("experiments/critic_ab_test/logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"ab_test_v2_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("ab_test_v2")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


class ABTestV2:
    """A/B test with new meta-prompts."""

    def __init__(self, config_path: str = "configs/examples/hype_ab_test_quick.yaml"):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger, self.log_file = setup_logging(self.timestamp)
        self.results = []
        self.setup()

    def _load_config(self, config_path: str) -> dict:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup(self):
        """Setup environment."""
        self.logger.info("=" * 70)
        self.logger.info("🚀 A/B TEST V2: New Meta-Prompts")
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
        print(
            f"📊 Train: {len(self.train_participants)}, Test: {len(self.test_participants)}"
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
        self.config["evolution"]["participant_batch_size"] = 4  # Parallel evaluation

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
        """Evaluate genotype."""
        evaluator = self._create_evaluator()
        prompt_str = genotype_to_evoprompt_str(genotype, self.config)
        score = evaluator.forward(prompt_str, self.config)
        return float(score)

    def _optimize_with_meta_prompt(
        self, section_name: str, meta_prompt_config: dict
    ) -> str:
        """Optimize section using new meta-prompt."""
        original_text = ORIGINAL_TEXTS[section_name]

        if meta_prompt_config.get("meta_config") is None:
            # Baseline - return as-is
            return original_text

        # Build HyPE config
        meta_config = HypeMetaPromptConfig(
            target_prompt_form="instructional ",
            include_role=True,
            section_specs=[
                PromptSectionSpec(
                    name=meta_prompt_config["meta_config"]["section_specs"][0]["name"],
                    description=meta_prompt_config["meta_config"]["section_specs"][0][
                        "description"
                    ],
                )
            ],
            constraints=meta_prompt_config["meta_config"].get("constraints", []),
            recommendations=meta_prompt_config["meta_config"].get(
                "recommendations", []
            ),
        )

        optimizer = SectionalHyPEOptimizer(model=self.model, config=self.config)

        # Optimize specific section
        if section_name == "critic":
            return optimizer._optimize_critic_with_config(original_text, meta_config)
        elif section_name == "task":
            return optimizer._optimize_output_format(original_text)
        else:
            raise ValueError(f"Unknown section: {section_name}")

    def test_section(self, section_name: str, variants: List[dict]) -> dict:
        """Test section with variants."""
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"🔧 TESTING: {section_name.upper()}")
        self.logger.info(f"{'=' * 70}")

        print(f"\n{'=' * 70}")
        print(f"🔧 ТЕСТИРУЕМ: {section_name.upper()}")
        print(f"{'=' * 70}")

        # Baseline score
        baseline_score = self._evaluate_genotype(self.base_genotype)
        self.logger.info(f"Baseline score: {baseline_score:.4f}")
        print(f"📊 Baseline: {baseline_score:.4f}")

        results = []
        best_score = baseline_score
        best_genotype = self.base_genotype.copy()
        best_variant = "baseline"

        for variant in tqdm(variants, desc=f"Testing {section_name}", unit="variant"):
            variant_name = variant["variant_name"]
            print(f"\n🧪 {variant_name}")

            # Create test genotype
            test_genotype = self.base_genotype.copy()

            # Optimize section
            print("   🔄 Оптимизация...")
            optimized_text = self._optimize_with_meta_prompt(section_name, variant)

            if section_name == "critic":
                test_genotype["critic_formulations"] = optimized_text
            elif section_name == "task":
                test_genotype["task"] = optimized_text

            # Evaluate
            print("   📊 Оценка...")
            test_score = self._evaluate_genotype(test_genotype)
            improvement = test_score - baseline_score

            print(f"   📊 Score: {test_score:.4f} (Δ {improvement:+.4f})")
            self.logger.info(f"{variant_name}: {test_score:.4f} (Δ {improvement:+.4f})")

            result = {
                "section": section_name,
                "variant_id": variant["variant_id"],
                "variant_name": variant_name,
                "baseline_score": baseline_score,
                "test_score": test_score,
                "improvement": improvement,
                "optimized_text": optimized_text,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

            if test_score > best_score:
                best_score = test_score
                best_genotype = test_genotype.copy()
                best_variant = variant_name
                print(f"   ✅ НОВЫЙ ЛУЧШИЙ!")

        return {
            "section": section_name,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "best_variant": best_variant,
            "results": results,
            "best_genotype": best_genotype,
        }

    def run_test(self):
        """Run full A/B test."""
        # Create output directory
        output_dir = (
            Path("experiments/critic_ab_test/results") / f"ab_test_v2_{self.timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Результаты: {output_dir}")

        all_results = {}

        # Test CRITIC
        critic_variants = [CRITIC_BASELINE, CRITIC_META_PROMPT]
        critic_results = self.test_section("critic", critic_variants)
        all_results["critic"] = critic_results

        # Update base genotype with best critic
        self.base_genotype = critic_results["best_genotype"]
        print(
            f"\n✅ CRITIC: лучший вариант '{critic_results['best_variant']}' с score {critic_results['best_score']:.4f}"
        )

        # Test TASK (with best critic)
        task_variants = [TASK_BASELINE, TASK_META_PROMPT]
        task_results = self.test_section("task", task_variants)
        all_results["task"] = task_results

        print(
            f"\n✅ TASK: лучший вариант '{task_results['best_variant']}' с score {task_results['best_score']:.4f}"
        )

        # Save results
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "config": {
                        "participants": 12,
                        "batch_size": 4,
                        "sections": ["critic", "task"],
                    },
                    "results": all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Summary
        print(f"\n{'=' * 70}")
        print("📊 ИТОГИ")
        print(f"{'=' * 70}")
        print(f"\nCRITIC:")
        print(f"   Baseline: {critic_results['baseline_score']:.4f}")
        print(
            f"   Best: {critic_results['best_score']:.4f} ({critic_results['best_variant']})"
        )
        print(
            f"   Δ: {critic_results['best_score'] - critic_results['baseline_score']:+.4f}"
        )

        print(f"\nTASK:")
        print(f"   Baseline: {task_results['baseline_score']:.4f}")
        print(
            f"   Best: {task_results['best_score']:.4f} ({task_results['best_variant']})"
        )
        print(
            f"   Δ: {task_results['best_score'] - task_results['baseline_score']:+.4f}"
        )

        print(f"\n💾 Сохранено: {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B Test V2 with new meta-prompts")
    parser.add_argument(
        "--config", type=str, default="configs/examples/hype_ab_test_quick.yaml"
    )

    args = parser.parse_args()

    test = ABTestV2(args.config)
    test.run_test()


if __name__ == "__main__":
    main()
