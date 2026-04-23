"""
src.optimizers — unified optimizer package.

Provides a single OptimizerRegistry that dispatches to the correct
optimizer based on config["optimization"]["method"].

Registered methods:
    "none"      → NoOpOptimizer      (returns base_genotype unchanged)
    "hype"      → SectionalHyPEOptimizer
    "evolution" → EvolutionOptimizer

Usage:
    from src.optimizers import OptimizerRegistry

    optimizer = OptimizerRegistry.create(
        config["optimization"]["method"],
        model=model,
        config=config,
    )
    genotype = optimizer.optimize(base_genotype, evaluator, dev_participants)
"""

from src.optimizers.base import BaseOptimizer, NoOpOptimizer, OptimizerRegistry
from src.optimizers.evolution import EvolutionOptimizer

# HyPE lives in src.meta_optimizer (its own domain)
from src.meta_optimizer.sectional_hype import SectionalHyPEOptimizer

# Register all built-in optimizers
# "none" is already registered inside base.py
OptimizerRegistry.register("hype", SectionalHyPEOptimizer)
OptimizerRegistry.register("evolution", EvolutionOptimizer)

__all__ = [
    "BaseOptimizer",
    "NoOpOptimizer",
    "OptimizerRegistry",
    "EvolutionOptimizer",
    "SectionalHyPEOptimizer",
]
