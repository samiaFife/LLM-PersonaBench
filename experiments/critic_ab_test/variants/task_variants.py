"""
Variants for task (output_format) section optimization.
"""

TASK_VARIANTS = {
    "v1_baseline": {
        "name": "Baseline (from system.py)",
        "description": "Original task instruction from the base config.",
        "prompt": (
            "You are to complete the IPIP-NEO personality questionnaire based on how you perceive yourself, "
            "drawing from your defined personality profile (traits and facets with intensity modifiers).\n"
            "Reflect internally using your profile before selecting.\n"
            "For each statement, evaluate how accurately it describes you and select one option from the scale (1-5), where:\n"
            '- 1 is "Very Inaccurate"\n'
            '- 2 is "Moderately Inaccurate"\n'
            '- 3 is "Neither Accurate Nor Inaccurate"\n'
            '- 4 is "Moderately Accurate"\n'
            '- 5 is "Very Accurate"\n'
            "Answer all questions."
        ),
        "meta_optimizer": False,
    },
    "v3_structured_full": {
        "name": "Structured Task with Role, Context, and Guidelines",
        "description": "Defines the assistant's role, summarises the task context, and provides step‑by‑step answering guidelines with reflection.",
        "prompt": None,
        "meta_optimizer": True,
        "meta_config": {
            "section_specs": [
                {
                    "name": "Role",
                    "description": "Briefly define the assistant's role as a simulated person embodying a specific Big Five personality profile (traits and facets with intensity modifiers)."
                },
                {
                    "name": "Task Context",
                    "description": "Summarise the user's input and meta-information."
                },
                {
                    "name": "Answer Guidelines",
                    "description": "Provide step‑by‑step instructions for answering each question, including the rating scale and reflection steps to ensure consistency with the profile."
                }
            ],
            "constraints": [
                "Role must mention Big Five traits and facets with intensity modifiers.",
                "Task context must reference the 120 questions and the personality profile.",
                "Answer guidelines must include the full 1‑5 scale with verbal labels (Very Inaccurate ... Very Accurate).",
                "Answer guidelines must contain at least three explicit reflection steps (e.g., recall trait/facet, compare statement, decide score).",
                "Answer guidelines must encourage linking each response to the profile.",
                "Total prompt length under 170 words."
            ],
            "recommendations": [
                "Keep Role concise (1‑2 sentences).",
                "Task Context should be a single sentence capturing the essence.",
                "Answer Guidelines: use imperatives ('Recall', 'Compare', 'Select'), and suggest mental imagery if uncertain.",
                "Maintain a clear separation between the three sections."
            ]
        }
    },
}

TEST_ORDER = ["v1_baseline", "v3_structured_full"]
