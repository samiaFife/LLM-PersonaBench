#!/usr/bin/env python3
"""
Extract full meta-prompts for all section variants.
Builds complete meta-prompts including user query section.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.meta_optimizer.hyper_templates import (
    HypeMetaPromptBuilder,
    HypeMetaPromptConfig,
    PromptSectionSpec,
)

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


def build_full_meta_prompt(
    section_name: str, variant_config: dict, original_text: str
) -> str:
    """
    Build complete meta-prompt for a variant.

    Args:
        section_name: Name of the section (critic, task, role, output)
        variant_config: Variant configuration with meta_config
        original_text: Original text from system.py (baseline)

    Returns:
        Complete meta-prompt string
    """
    if not variant_config.get("meta_optimizer", False):
        # For baseline, return simple instruction
        return f"""You are an expert prompt engineer.

Your task is to optimize the following {section_name} section for an IPIP-NEO personality simulation.

Current {section_name}:
{original_text}

Return the optimized version wrapped in <result_prompt> tags.
"""

    # Build meta config from variant
    meta_config = HypeMetaPromptConfig(
        target_prompt_form="instructional ",
        include_role=True,
        section_specs=[
            PromptSectionSpec(
                name=variant_config["meta_config"]["section_specs"][0]["name"],
                description=variant_config["meta_config"]["section_specs"][0][
                    "description"
                ],
            )
        ],
        constraints=variant_config["meta_config"].get("constraints", []),
        recommendations=variant_config["meta_config"].get("recommendations", []),
    )

    # Build the meta-prompt
    builder = HypeMetaPromptBuilder(meta_config)

    # Domain context for all sections
    domain_context = """Domain Context:
- Framework: Big Five (OCEAN) personality model
- Instrument: IPIP-NEO-120 questionnaire (120 questions)
- Task: Simulate human with specific personality profile
- Response: Likert scale 1-5 (Very Inaccurate to Very Accurate)
- Goal: Maintain consistency across all 120 responses
- Structure: 5 traits × 6 facets = 30 facets with intensity modifiers"""

    # Section-specific context
    section_contexts = {
        "critic": """
Section Type: Internal Self-Reflection Guidelines
Purpose: Guide consistency checking before answering each question
Key Challenge: Align 30 facets with 5 traits using intensity modifiers""",
        "task": """
Section Type: Task Instructions  
Purpose: Explain how to complete the questionnaire based on personality profile
Key Challenge: Map personality intensity to Likert scale responses""",
        "role": """
Section Type: Role Definition
Purpose: Define the persona as a simulated person with Big Five traits
Key Challenge: Create authentic self-perception, not role-play""",
        "output": """
Section Type: Response Format Specification
Purpose: Specify exact output format for machine parsing
Key Challenge: Valid JSON array of exactly 120 objects""",
    }

    # Build the full meta-prompt with user query
    meta_prompt = builder.build_meta_prompt()

    # Add user query section
    user_query = f"""
{domain_context}
{section_contexts.get(section_name, "")}

User Query:
Optimize the following {section_name} section:

<user_query>
{original_text}
</user_query>
"""

    full_prompt = meta_prompt + user_query

    return full_prompt


def extract_all_meta_prompts():
    """Extract meta-prompts for all sections and variants."""

    print("=" * 80)
    print("🔍 ИЗВЛЕЧЕНИЕ МЕТА-ПРОМПТОВ")
    print("=" * 80)

    # Original texts from system.py
    original_texts = {
        "critic": """Reflect on these descriptions as if they are your own self-perception, and respond to questions by evaluating how accurately they describe you, as a real person would. 
Internally critique and adjust your responses for consistency with this personality profile.""",
        "task": """You are to complete the IPIP-NEO personality questionnaire based on how you perceive yourself, drawing from your defined personality profile (traits and facets with intensity modifiers).
Reflect internally using your profile before selecting. 
For each statement, evaluate how accurately it describes you and select one option from the scale (1-5), where:
- 1 is "Very Inaccurate"
- 2 is "Moderately Inaccurate"
- 3 is "Neither Accurate Nor Inaccurate"
- 4 is "Moderately Accurate"
- 5 is "Very Accurate"
Answer all questions""",
        "role": """You are a simulated person embodying a specific personality type from the Big Five model (OCEAN), based on traits and facets. 
Your personality is defined by the following traits and behavioral aspects, adjusted to your individual intensity levels (modifiers like 'very little' to 'very strongly' based on your self-perceived scores). 
Description of your personality:""",
        "output": """Answer strictly in the JSON format: a JSON array of 120 objects.
Each object must have {{"question_id": number from 1 to 120, "answer": number from 1 to 5}}.
Example of format: [{{"question_id": 1, "answer": 3}}, {{"question_id": 2, "answer": 5}}, ..., {{"question_id": 120, "answer": 4}}]

Output ONLY the pure JSON array. No additional text, explanations, markdown, or anything else.""",
    }

    # Sections to process
    sections = {
        "critic": (CRITIC_VARIANTS, CRITIC_ORDER),
        "task": (TASK_VARIANTS, TASK_ORDER),
        "role": (ROLE_VARIANTS, ROLE_ORDER),
        "output": (OUTPUT_VARIANTS, OUTPUT_ORDER),
    }

    results = {}

    for section_name, (variants, order) in sections.items():
        print(f"\n📝 Обработка секции: {section_name}")

        section_prompts = []

        for variant_id in order:
            if variant_id not in variants:
                continue

            variant = variants[variant_id]

            # Build full meta-prompt
            meta_prompt = build_full_meta_prompt(
                section_name, variant, original_texts[section_name]
            )

            section_prompts.append(
                {
                    "variant_id": variant_id,
                    "variant_name": variant["name"],
                    "description": variant["description"],
                    "meta_optimizer": variant["meta_optimizer"],
                    "full_meta_prompt": meta_prompt,
                }
            )

            print(f"   ✅ {variant['name']}")

        results[section_name] = section_prompts

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/critic_ab_test/meta_prompts")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"meta_prompts_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "sections": results,
                "total_variants": sum(len(prompts) for prompts in results.values()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n" + "=" * 80)
    print(f"💾 Сохранено в: {output_file}")
    print(f"📊 Всего вариантов: {sum(len(prompts) for prompts in results.values())}")
    print("=" * 80)

    # Also print summary
    print("\n📋 СВОДКА:")
    for section_name, prompts in results.items():
        print(f"   {section_name}: {len(prompts)} вариантов")
        for p in prompts:
            print(f"      - {p['variant_name']}")


def main():
    """Main entry point."""
    extract_all_meta_prompts()


if __name__ == "__main__":
    main()
