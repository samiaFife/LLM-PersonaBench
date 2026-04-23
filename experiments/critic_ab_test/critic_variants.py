"""
Варианты мета-промптов для секции critic_formulations.

Каждый вариант - это разный подход к инструкциям самопроверки.
"""

CRITIC_VARIANTS = {
    "v1_baseline": {
        "name": "Baseline (from system.py)",
        "description": "Original critic from the base config.",
        "prompt": (
            "Reflect on these descriptions as if they are your own self-perception, and respond to questions "
            "by evaluating how accurately they describe you, as a real person would.\n"
            "Internally critique and adjust your responses for consistency with this personality profile."
        ),
        "meta_optimizer": False,
    },
    "v2_detailed_checks": {
        "name": "Detailed Consistency Checks",
        "description": "A structured pre-answer checklist with embodied verification ensuring alignment with traits, facets, and intensity modifiers.",
        "prompt": None,
        "meta_optimizer": True,
        "meta_config": {
            "section_specs": [
                {
                    "name": "Embodied Consistency Protocol",
                    "description": "A step-by-step internal checklist starting with embodied resonance ('does this feel like me?'), followed by verification against traits, facets, and intensity modifiers.",
                }
            ],
            "constraints": [
                "Must contain at least three explicit, sequential checks.",
                "First check must assess embodied resonance/emotional fit.",
                "Each subsequent check must reference traits, facets, or intensity modifiers.",
                "Must include a rule for handling conflicts prioritizing embodied sense.",
                "Maximum length: 150 words.",
            ],
            "recommendations": [
                "Use imperatives like 'Feel', 'Check', 'Verify'.",
                "Write as internal monologue blending intuition and logic.",
                "Start with embodied awareness before technical verification.",
                "Keep steps memorable but include feeling/sensing language.",
            ],
        },
    },
}

# Порядок тестирования (чтобы сравнивать последовательно)
TEST_ORDER = [
    "v1_baseline",
    "v2_detailed_checks",
]

# Базовый конфиг для тестирования
BASE_CONFIG = {
    "num_participants": 3,  # Меньше для скорости
    "cluster": 2,  # Тестируем на одном кластере
    "timeout": 120,
    "max_retries": 2,
}
