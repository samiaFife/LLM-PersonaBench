"""
EvolutionOptimizer — wraps GAEvoluter behind the BaseOptimizer interface.

Moves GA evolution logic out of person_type_opt.py so the main pipeline
can call optimizer.optimize() uniformly regardless of the method.
"""

from typing import Dict, List, Optional

from src.optimizers.base import BaseOptimizer


class EvolutionOptimizer(BaseOptimizer):
    """
    Wraps GAEvoluter (genetic algorithm) behind the BaseOptimizer interface.

    After optimize() returns, the internal GAEvoluter instance is accessible
    via get_evoluter() so that person_type_opt.py can still read generation_logs
    for result logging without any if/else branching on optimizer type.
    """

    def __init__(self, model, config: Optional[Dict] = None):
        super().__init__(model=model, config=config)
        self._evoluter = None  # set after optimize() is called

    def optimize(
        self,
        base_genotype: Dict,
        evaluator,
        dev_participants,
    ) -> Dict:
        """
        Run GA evolution and return the best genotype dict.

        Args:
            base_genotype: Initial genotype dict
            evaluator: MyEvaluator instance (dev_participants already set by caller)
            dev_participants: Training participants (DataFrame) — passed for API
                              symmetry; evaluator.dev_participants must be set by
                              the caller before this call.

        Returns:
            Best optimised genotype dict
        """
        # Lazy imports to avoid circular dependencies and heavy loading at module level
        from src.evolution.evoluter import GAEvoluter
        from src.evolution.init_population import init_population
        from src.evolution.parse_args import parse_args_from_yaml
        from src.evolution.utils import (
            clean_evoprompt_response,
            parse_str_to_genotype,
            validate_and_repair_genotype,
        )

        evo_config = self.config.get("evolution", {})
        evo_args = parse_args_from_yaml(evo_config)

        fixed_modifiers = evaluator.fixed_modifiers

        evoluter = GAEvoluter(
            evo_args,
            evaluator,
            evolution_model=self.model,
            config=self.config,
        )
        evoluter.population = init_population(
            base_genotype,
            self.config,
            evo_args.popsize,
            self.model,
        )
        evoluter.evolute()

        # Store for external access (generation logs)
        self._evoluter = evoluter

        # Parse best individual back to genotype dict
        best_str_raw = evoluter.population[0]
        best_str = clean_evoprompt_response(best_str_raw)
        best_str = validate_and_repair_genotype(
            best_str, fixed_modifiers, base_genotype, self.config
        )
        best_genotype = parse_str_to_genotype(
            best_str,
            fixed_modifiers,
            self.config,
            template_genotype=base_genotype,
        )
        return best_genotype

    def get_evoluter(self):
        """
        Return the internal GAEvoluter instance after optimize() has been called.

        Useful for reading generation_logs in person_type_opt.py without
        branching on optimizer type.

        Returns:
            GAEvoluter instance, or None if optimize() has not been called yet.
        """
        return self._evoluter

    def get_generation_logs(self) -> List[Dict]:
        """
        Convenience accessor for evoluter.generation_logs.

        Returns:
            List of per-generation log dicts, or [] if not yet run.
        """
        if self._evoluter is None:
            return []
        return getattr(self._evoluter, "generation_logs", [])
