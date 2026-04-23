"""
Backward-compatibility shim.

BaseOptimizer and OptimizerRegistry have moved to src.optimizers.base.
This module re-exports them so that any existing code importing from
src.meta_optimizer.base continues to work without changes.
"""

from src.optimizers.base import BaseOptimizer, NoOpOptimizer, OptimizerRegistry

__all__ = [
    "BaseOptimizer",
    "NoOpOptimizer",
    "OptimizerRegistry",
]
