"""
Sectional HyPE Optimizer for IPIP-NEO personality simulation prompts.

Optimizes each section of the genotype separately while preserving structure.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.optimizers.base import BaseOptimizer
from src.meta_optimizer.hype import HyPEOptimizer
from src.meta_optimizer.hyper_templates import (
    HypeMetaPromptConfig,
    PromptSectionSpec,
)


class SectionalHyPEOptimizer(BaseOptimizer):
    """
    Optimizes genotype sections separately using HyPE.

    Sections optimized:
    - role_definition
    - critic_formulations
    - output_format (task description)

    Sections preserved (with stubs):
    - trait_formulations (copied from base)
    - facet_formulations (copied from base)
    - intensity_modifiers (never touched)
    - trait_targets (never touched)
    - facet_targets (never touched)
    """

    def __init__(self, model, config: Optional[Dict] = None):
        super().__init__(model=model, config=config)
        self.optimization_log = []

    def optimize(
        self, base_genotype: Dict, evaluator, dev_participants, verbose: bool = True
    ) -> Dict:
        """
        Main optimization pipeline.

        Args:
            base_genotype: Initial genotype dict
            evaluator: MyEvaluator instance for scoring
            dev_participants: Training data for evaluation
            verbose: Print progress

        Returns:
            Optimized genotype dict
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🚀 STARTING SECTIONAL HyPE OPTIMIZATION")
            print("=" * 70)

        # Start with base
        optimized = base_genotype.copy()

        # Evaluate baseline
        base_score = self._evaluate_genotype(base_genotype, evaluator, "BASELINE")
        if verbose:
            print(f"📊 Baseline score: {base_score:.4f}")

        # Optimize sections sequentially
        sections_to_optimize = [
            ("role_definition", self._optimize_role),
            ("output_format", self._optimize_output_format),
            ("critic_formulations", self._optimize_critic),
        ]

        for section_name, optimize_func in sections_to_optimize:
            if verbose:
                print(f"\n🔧 Optimizing: {section_name}")

            current_text = base_genotype.get(section_name, "")
            optimized_text = optimize_func(current_text)

            # Create test genotype with this section optimized
            test_genotype = optimized.copy()
            test_genotype[section_name] = optimized_text

            # Evaluate
            section_score = self._evaluate_genotype(
                test_genotype, evaluator, f"AFTER_{section_name.upper()}"
            )

            # Update optimized genotype
            optimized[section_name] = optimized_text

            if verbose:
                print(f"   Score after {section_name}: {section_score:.4f}")

        # Copy traits and facets (stubs for now)
        optimized["trait_formulations"] = base_genotype.get("trait_formulations", {})
        optimized["facet_formulations"] = base_genotype.get("facet_formulations", {})

        # Final evaluation
        final_score = self._evaluate_genotype(optimized, evaluator, "FINAL")

        if verbose:
            print(f"\nFINAL SCORE: {final_score:.4f}")
            print(f"BASELINE: {base_score:.4f}")
            print(f"IMPROVEMENT: {final_score - base_score:+.4f}")

        # Log results
        self._log_optimization(
            baseline_score=base_score,
            final_score=final_score,
            base_genotype=base_genotype,
            optimized_genotype=optimized,
        )

        if final_score < base_score:
            print(f"\nWARNING: Optimized genotype is WORSE than baseline!")
            print(f"   Returning optimized version anyway (see logs)")

        return optimized

    def _evaluate_genotype(self, genotype: Dict, evaluator, stage: str) -> float:
        """Evaluate genotype and return fitness score."""
        # Lazy import to avoid ipipneo dependency
        from src.evolution.utils import genotype_to_evoprompt_str

        prompt_str = genotype_to_evoprompt_str(genotype, self.config)
        score = evaluator.forward(prompt_str, self.config)

        self.optimization_log.append(
            {
                "stage": stage,
                "score": float(score),
                "genotype_hash": hash(json.dumps(genotype, sort_keys=True)) % 10000,
            }
        )

        return float(score)

    def _optimize_role(self, current_text: str) -> str:
        """Optimize role_definition section."""
        meta_config = HypeMetaPromptConfig(
            target_prompt_form="instructional ",
            include_role=True,
            section_specs=[
                PromptSectionSpec(
                    name="Role Definition",
                    description=(
                        "Define the persona as a simulated human completing a personality questionnaire. "
                        "Emphasize that this is a role-play with specific Big Five (OCEAN) personality traits. "
                        "Make it natural and psychologically grounded. Keep concise but descriptive."
                    ),
                ),
            ],
            constraints=[
                "Must work for IPIP-NEO-120 personality assessment",
                "Should emphasize consistency across 120 questions",
                "Must acknowledge Big Five (OCEAN) traits",
                "Keep it under 200 words",
            ],
        )

        optimizer = HyPEOptimizer(model=self.model, config=meta_config)

        meta_info = {
            "task": "personality simulation",
            "domain": "psychology",
            "model": "Big Five OCEAN",
            "assessment": "IPIP-NEO-120",
        }

        result = optimizer.optimize(
            prompt=current_text, meta_info=meta_info, n_prompts=1
        )
        if isinstance(result, list):
            return result[0] if result else current_text
        return result

    def _optimize_critic_with_config(
        self, current_text: str, meta_config: HypeMetaPromptConfig
    ) -> str:
        """Optimize critic with custom meta config (for A/B testing)."""
        optimizer = HyPEOptimizer(model=self.model, config=meta_config)

        meta_info = {
            "task": "consistency checking",
            "context": "120 questionnaire responses",
            "goal": "maintain coherent personality profile",
        }

        result = optimizer.optimize(
            prompt=current_text, meta_info=meta_info, n_prompts=1
        )
        if isinstance(result, list):
            return result[0] if result else current_text
        return result

    def _optimize_critic(self, current_text: str) -> str:
        """Optimize critic_formulations section."""
        meta_config = HypeMetaPromptConfig(
            target_prompt_form="instructional ",
            include_role=True,
            section_specs=[
                PromptSectionSpec(
                    name="Self-Reflection Guidelines",
                    description=(
                        "Provide internal guidelines for the persona to check answer consistency. "
                        "Include specific instructions for maintaining alignment with Big Five traits and facets. "
                        "Guide on how to use intensity modifiers and ensure test-retest reliability."
                    ),
                ),
            ],
            constraints=[
                "Must guide consistency checking before answering",
                "Should address trait-facet alignment",
                "Must mention intensity modifiers (very strongly/slightly/etc.)",
                "Should promote test-retest reliability mindset",
                "Keep it under 150 words",
            ],
        )

        optimizer = HyPEOptimizer(model=self.model, config=meta_config)

        meta_info = {
            "task": "consistency checking",
            "context": "120 questionnaire responses",
            "goal": "maintain coherent personality profile",
        }

        result = optimizer.optimize(
            prompt=current_text, meta_info=meta_info, n_prompts=1
        )
        # Ensure we return a string
        if isinstance(result, list):
            return result[0] if result else current_text
        return result

    def _optimize_output_format(self, current_text: str) -> str:
        """Optimize output_format (task description for 120 questions)."""
        meta_config = HypeMetaPromptConfig(
            target_prompt_form="instructional ",
            include_role=True,
            section_specs=[
                PromptSectionSpec(
                    name="Task Instructions",
                    description=(
                        "Clear instructions for completing the IPIP-NEO-120 questionnaire. "
                        "Explain how to answer based on personality profile. "
                        "Specify the response format and scale."
                    ),
                ),
            ],
            constraints=[
                "Must mention 120 questions",
                "Must specify 1-5 scale (Very Inaccurate to Very Accurate)",
                "Should reference personality profile from system prompt",
                "Must require JSON array format",
                "Keep it clear and structured",
            ],
        )

        optimizer = HyPEOptimizer(model=self.model, config=meta_config)

        meta_info = {
            "task": "questionnaire completion",
            "num_questions": 120,
            "scale": "1-5 Likert",
            "output_format": "JSON array",
        }

        result = optimizer.optimize(
            prompt=current_text, meta_info=meta_info, n_prompts=1
        )
        # Ensure we return a string
        if isinstance(result, list):
            return result[0] if result else current_text
        return result

    def _log_optimization(
        self,
        baseline_score: float,
        final_score: float,
        base_genotype: Dict,
        optimized_genotype: Dict,
    ):
        """Log optimization results."""
        log_entry = {
            "baseline_score": baseline_score,
            "final_score": final_score,
            "improvement": final_score - baseline_score,
            "optimization_stages": self.optimization_log.copy(),
            "base_genotype_keys": list(base_genotype.keys()),
            "optimized_genotype_keys": list(optimized_genotype.keys()),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"hype_optimization_{timestamp}.json"

        try:
            with open(log_filename, "w") as f:
                json.dump(log_entry, f, indent=2)
            print(f"   Log saved: {log_filename}")
        except Exception as e:
            print(f"   Could not save log: {e}")

    def get_optimization_log(self) -> List[Dict]:
        """Return optimization log."""
        return self.optimization_log.copy()

    def optimize_genotype(
        self, base_genotype: Dict, evaluator, dev_participants, verbose: bool = True
    ) -> Dict:
        """Alias for optimize() — kept for backward compatibility."""
        import warnings
        warnings.warn(
            "optimize_genotype() is deprecated, use optimize() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.optimize(base_genotype, evaluator, dev_participants, verbose=verbose)
