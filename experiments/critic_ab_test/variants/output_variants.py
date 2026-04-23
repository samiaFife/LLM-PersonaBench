"""
Variants for output_format section optimization.
"""

OUTPUT_VARIANTS = {
    "v1_baseline": {
        "name": "Baseline (from system.py)",
        "description": "Original output format instruction from the base config.",
        "prompt": (
            "Answer strictly in the JSON format: a JSON array of 120 objects.\n"
            "Each object must have {\"question_id\": number from 1 to 120, \"answer\": number from 1 to 5}.\n"
            "Example of format: [{\"question_id\": 1, \"answer\": 3}, {\"question_id\": 2, \"answer\": 5}, ..., "
            "{\"question_id\": 120, \"answer\": 4}]\n\n"
            "Output ONLY the pure JSON array. No additional text, explanations, markdown, or anything else."
        ),
        "meta_optimizer": False,
    },
    "v2_with_examples": {
        "name": "JSON with Positive/Negative Examples",
        "description": "Includes correct and incorrect output examples to prevent common mistakes.",
        "prompt": None,
        "meta_optimizer": True,
        "meta_config": {
            "section_specs": [
                {
                    "name": "JSON Output Format with Examples",
                    "description": (
                        "Provide your answers as a JSON array of 120 objects with 'question_id' and 'answer'. "
                        "Example: [{\"question_id\":1,\"answer\":3}, ...]. "
                        "Ensure the JSON is valid and contains no additional text, explanations, or markdown. "
                        "Incorrect outputs like 'Here are my answers: [...]' or ```json [...]``` are not allowed."
                    ),
                }
            ],
            "constraints": [
                "Must include at least one correct example.",
                "Must mention common incorrect patterns (extra text, markdown).",
                "Must forbid any non-JSON content.",
                "Keep under 100 words."
            ],
            "recommendations": [
                "Use ✅ and ❌ to clearly distinguish correct/incorrect.",
                "Keep examples very short (2-3 items)."
            ],
        },
    },
    "v3_strict_no_extra": {
        "name": "Strict JSON Only",
        "description": "Extremely strict instruction against any extra output.",
        "prompt": None,
        "meta_optimizer": True,
        "meta_config": {
            "section_specs": [
                {
                    "name": "Strict JSON Only",
                    "description": (
                        "Output ONLY a valid JSON array as specified. "
                        "No introductory phrases, no closing remarks, no markdown formatting. "
                        "Any deviation will break the parsing."
                    ),
                }
            ],
            "constraints": [
                "Must explicitly forbid any additional text.",
                "Must restate the exact JSON structure.",
                "Must be under 50 words.",
                "Must convey that even a single extra character is unacceptable."
            ],
            "recommendations": [
                "Use all caps for emphasis: 'ONLY'.",
                "Be unambiguous and forceful."
            ],
        },
    },
}


TEST_ORDER = ["v1_baseline", "v2_with_examples", "v3_strict_no_extra"]
