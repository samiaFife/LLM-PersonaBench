"""
New meta-prompts for critic and task sections.
Uses v2_detailed_checks for critic and v3_structured_full for task.
"""

# Meta-info section (common for both)
META_INFO_SECTION = """Domain Context:
- Framework: Big Five (OCEAN) personality assessment
- Instrument: IPIP-NEO-120 questionnaire (120 questions)
- Traits: 5 (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)  
- Facets: 30 (6 per trait, e.g., facet_anxiety, facet_friendliness, etc.)
- Intensity Modifiers: 5 levels mapped to 0-100 scale:
  * 0-19: "very little"
  * 20-39: "slightly" 
  * 40-59: "moderately"
  * 60-79: "quite strongly"
  * 80-100: "very strongly"
- Scale: 1-5 Likert (1=Very Inaccurate, 5=Very Accurate)"""

# CRITIC: Based on v2_detailed_checks
CRITIC_META_PROMPT = {
    "variant_id": "v2_detailed_checks_v2",
    "variant_name": "Detailed Consistency Checks",
    "task_section": """Your task is to optimize the CRITIC section - internal self-reflection guidelines that help maintain consistency across 120 personality questionnaire responses.

The critic should guide the persona to:
- Check answers against Big Five traits and 30 facets
- Use intensity modifiers appropriately
- Resolve conflicts between facets and traits
- Ensure coherent personality profile

Make it structured but natural, like an authentic internal monologue.""",
    "meta_config": {
        "section_specs": [
            {
                "name": "Pre-Answer Consistency Protocol",
                "description": "A step-by-step internal checklist to verify that each answer matches the defined traits, facets, and intensity modifiers before responding.",
            }
        ],
        "constraints": [
            "Must contain at least three explicit, sequential checks.",
            "Each check must reference either a trait, a facet, or an intensity modifier.",
            "Must include a rule for handling conflicts (e.g., when a facet contradicts its parent trait).",
            "Maximum length: 150 words.",
        ],
        "recommendations": [
            "Use imperatives like 'Check', 'Verify', 'Confirm'.",
            "Write as an internal monologue.",
            "Keep steps short and easy to remember.",
        ],
    },
}

# TASK: Based on v3_structured_full (need to check what this is)
# For now using v2_structured_reflection as base and enhancing
TASK_META_PROMPT = {
    "variant_id": "v3_structured_full_v2",
    "variant_name": "Structured Full Context",
    "task_section": """Your task is to optimize the TASK section - instructions for completing the IPIP-NEO-120 questionnaire.

The task should:
- Explain the 1-5 Likert scale clearly
- Guide how to translate personality profile into responses
- Include step-by-step reflection process
- Feel natural and encourage consistent answering""",
    "meta_config": {
        "section_specs": [
            {
                "name": "Contextual Task Instructions",
                "description": "Instructions that provide full context about the questionnaire, the scale, and how to use the personality profile when answering each question.",
            }
        ],
        "constraints": [
            "Must explain the 1-5 scale (Very Inaccurate to Very Accurate).",
            "Must include at least 3 reflection steps.",
            "Must reference traits, facets, and intensity modifiers.",
            "Keep under 120 words.",
        ],
        "recommendations": [
            "Use clear action verbs: 'Recall', 'Compare', 'Decide'.",
            "Include scale explanation early.",
            "Make reflection steps flow naturally.",
        ],
    },
}

# Baseline variants (for comparison)
CRITIC_BASELINE = {
    "variant_id": "v1_baseline",
    "variant_name": "Baseline",
    "task_section": "Optimize the following critic formulation for IPIP-NEO personality simulation.",
    "meta_config": None,  # No meta-optimizer, use as-is
}

TASK_BASELINE = {
    "variant_id": "v1_baseline",
    "variant_name": "Baseline",
    "task_section": "Optimize the following task instruction for IPIP-NEO personality simulation.",
    "meta_config": None,
}

# Original texts from system.py
ORIGINAL_TEXTS = {
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
}
