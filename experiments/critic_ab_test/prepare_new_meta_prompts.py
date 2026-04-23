#!/usr/bin/env python3
"""
Prepare new meta-prompts for critic and task sections.
Logs full meta-prompts with everything except {original_text} placeholder.
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
from experiments.critic_ab_test.new_meta_prompts import (
    META_INFO_SECTION,
    CRITIC_META_PROMPT,
    TASK_META_PROMPT,
    CRITIC_BASELINE,
    TASK_BASELINE,
    ORIGINAL_TEXTS,
)


def build_full_meta_prompt(meta_prompt_config, section_name):
    """
    Build complete meta-prompt with everything except {original_text}.

    Structure:
    1. ROLE_LINE
    2. TASK_SECTION (custom from config)
    3. STRUCTURE_SECTION
    4. RECOMMENDATIONS_SECTION
    5. CONSTRAINTS_SECTION
    6. OUTPUT_FORMAT_SECTION
    7. META_INFO_SECTION
    8. USER_QUERY
    """
    variant_id = meta_prompt_config["variant_id"]
    variant_name = meta_prompt_config["variant_name"]
    task_section = meta_prompt_config["task_section"]
    meta_config = meta_prompt_config.get("meta_config")

    if meta_config is None:
        # Baseline - simple prompt
        full_prompt = f"""You are an expert prompt engineer.

{task_section}

{META_INFO_SECTION}

User Query:
Optimize the following {section_name.upper()}_SECTION:

<{section_name}_section>
{{original_text}}
</{section_name}_section>

Return the optimized version wrapped in <result_prompt> tags."""

        return {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "is_baseline": True,
            "full_meta_prompt": full_prompt,
        }

    # Build meta config for HyPE
    hype_config = HypeMetaPromptConfig(
        target_prompt_form="instructional ",
        include_role=True,
        section_specs=[
            PromptSectionSpec(
                name=meta_config["section_specs"][0]["name"],
                description=meta_config["section_specs"][0]["description"],
            )
        ],
        constraints=meta_config.get("constraints", []),
        recommendations=meta_config.get("recommendations", []),
    )

    # Build sections using builder
    builder = HypeMetaPromptBuilder(hype_config)

    # Replace task section template with custom one BEFORE building
    # Use the exact task_section from the config
    builder.TASK_SECTION_TEMPLATE = task_section + "\n\n"
    builder.config.target_prompt_form = "instructional "

    # Rebuild cache with new task template
    builder._cache_all_sections()

    # Get individual sections (now with custom task)
    role_section = builder.build_role_section()
    structure_section = builder.build_prompt_structure_section()
    recommendations_section = builder.build_recommendations_section()
    constraints_section = builder.build_constraints_section()
    output_format_section = builder.build_output_format_section()

    # Build full prompt manually
    full_prompt = f"""{role_section}{structure_section}{recommendations_section}{constraints_section}{output_format_section}

{META_INFO_SECTION}

User Query:
Optimize the following {section_name.upper()}_SECTION:

<{section_name}_section>
{{original_text}}
</{section_name}_section>"""

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "is_baseline": False,
        "full_meta_prompt": full_prompt,
        "meta_config": meta_config,
    }


def prepare_all_meta_prompts():
    """Prepare and log all meta-prompts."""

    print("=" * 80)
    print("🔍 ПОДГОТОВКА НОВЫХ МЕТА-ПРОМПТОВ")
    print("=" * 80)

    results = {
        "critic": [],
        "task": [],
    }

    # Process CRITIC variants
    print("\n📝 Обработка CRITIC:")
    for variant_config in [CRITIC_BASELINE, CRITIC_META_PROMPT]:
        result = build_full_meta_prompt(variant_config, "critic")
        results["critic"].append(result)
        print(f"   ✅ {result['variant_name']} (baseline={result['is_baseline']})")

    # Process TASK variants
    print("\n📝 Обработка TASK:")
    for variant_config in [TASK_BASELINE, TASK_META_PROMPT]:
        result = build_full_meta_prompt(variant_config, "task")
        results["task"].append(result)
        print(f"   ✅ {result['variant_name']} (baseline={result['is_baseline']})")

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/critic_ab_test/meta_prompts_v2")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"prepared_meta_prompts_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "meta_info": META_INFO_SECTION,
                "sections": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 Сохранено в: {output_file}")

    # Also save human-readable versions
    text_output = output_dir / f"meta_prompts_readable_{timestamp}.txt"
    with open(text_output, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("META-INFO SECTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(META_INFO_SECTION)
        f.write("\n\n")

        for section_name, variants in results.items():
            f.write("=" * 80 + "\n")
            f.write(f"SECTION: {section_name.upper()}\n")
            f.write("=" * 80 + "\n\n")

            for variant in variants:
                f.write(f"--- {variant['variant_name']} ---\n")
                f.write(f"ID: {variant['variant_id']}\n")
                f.write(f"Baseline: {variant['is_baseline']}\n\n")
                f.write(variant["full_meta_prompt"])
                f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"📝 Читаемая версия: {text_output}")

    # Print summary
    print("\n📋 СВОДКА:")
    for section_name, variants in results.items():
        print(f"   {section_name}: {len(variants)} вариантов")
        for v in variants:
            print(f"      - {v['variant_name']} (baseline={v['is_baseline']})")

    print("\n✅ Готово к использованию в A/B тесте!")
    return results


def main():
    """Main entry point."""
    prepare_all_meta_prompts()


if __name__ == "__main__":
    main()
