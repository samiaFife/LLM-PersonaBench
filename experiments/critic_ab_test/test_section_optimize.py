"""
Test script for previewing HyPE output for different sections.
Generates prompts without evaluation - just to see what HyPE produces.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.registry import get_model
from src.meta_optimizer import (
    SectionalHyPEOptimizer,
    HypeMetaPromptConfig,
    PromptSectionSpec,
)
from src.simulator.person_type_opt import _load_system

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
    "critic": {
        "variants": CRITIC_VARIANTS,
        "order": CRITIC_ORDER,
        "display_name": "Critic",
        "system_key": "critic_internal",
        "optimize_func": "_optimize_critic_with_config",
    },
    "task": {
        "variants": TASK_VARIANTS,
        "order": TASK_ORDER,
        "display_name": "Task",
        "system_key": "task",
        "optimize_func": "_optimize_output_format",
    },
    "role": {
        "variants": ROLE_VARIANTS,
        "order": ROLE_ORDER,
        "display_name": "Role",
        "system_key": "role",
        "optimize_func": "_optimize_role",
    },
    "output": {
        "variants": OUTPUT_VARIANTS,
        "order": OUTPUT_ORDER,
        "display_name": "Output Format",
        "system_key": None,  # Not in system.py directly
        "optimize_func": "_optimize_output_format",
    },
}


def preview_section_variants(
    section_name: str, config_path: str = "configs/examples/hype_ab_test_quick.yaml"
):
    """Generate and display HyPE output for all variants of a section."""

    print("=" * 80)
    print(f"🔍 ПРЕДПРОСМОТР: {SECTION_CONFIGS[section_name]['display_name']}")
    print("=" * 80)

    # Load config
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load system
    system = _load_system(config)

    # Load model
    print("\n🤖 Загрузка модели...")
    model = get_model(config["model"])
    print(f"   Модель: {model.model_name}")

    # Get section config
    section_config = SECTION_CONFIGS[section_name]
    variants = section_config["variants"]
    order = section_config["order"]

    # Get baseline text
    if section_config["system_key"]:
        baseline_text = system[section_config["system_key"]]
    else:
        # For output_format, combine task + response_format
        baseline_text = f"{system['task']}\n\n{system['response_format']}"

    results = []

    for variant_id in order:
        if variant_id not in variants:
            continue

        variant = variants[variant_id]

        print(f"\n{'=' * 80}")
        print(f"📝 ВАРИАНТ: {variant['name']}")
        print(f"   {variant['description']}")
        print(f"{'=' * 80}")

        if variant["meta_optimizer"]:
            # Generate via HyPE
            print("\n🔄 Генерация через HyPE...")

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

            optimizer = SectionalHyPEOptimizer(model=model, config=config)

            # Call appropriate optimization function
            if section_name == "critic":
                produced_text = optimizer._optimize_critic_with_config(
                    baseline_text, meta_config
                )
            elif section_name in ["task", "output"]:
                produced_text = optimizer._optimize_output_format(baseline_text)
            elif section_name == "role":
                produced_text = optimizer._optimize_role(baseline_text)
            else:
                raise ValueError(f"Unknown section: {section_name}")

            print("\n✅ СГЕНЕРИРОВАННЫЙ ПРОМПТ:")
            print("-" * 80)
            print(produced_text)
            print("-" * 80)

            result = {
                "variant_id": variant_id,
                "variant_name": variant["name"],
                "meta_prompt": variant.get("meta_config", {}),
                "produced_prompt": produced_text,
                "baseline_text": baseline_text,
            }

        else:
            # Use baseline
            print("\n📋 BASELINE (без изменений):")
            print("-" * 80)
            print(baseline_text)
            print("-" * 80)

            result = {
                "variant_id": variant_id,
                "variant_name": variant["name"],
                "meta_prompt": None,
                "produced_prompt": baseline_text,
                "baseline_text": baseline_text,
            }

        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/critic_ab_test/preview_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{section_name}_preview_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "section": section_name,
                "timestamp": timestamp,
                "model": model.model_name,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 Результаты сохранены в: {output_file}")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview HyPE output for different sections without evaluation"
    )
    parser.add_argument(
        "section",
        type=str,
        choices=["critic", "task", "role", "output"],
        help="Section to preview (critic, task, role, output)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/examples/hype_ab_test_quick.yaml",
        help="Config file path",
    )

    args = parser.parse_args()

    preview_section_variants(args.section, args.config)


if __name__ == "__main__":
    main()
