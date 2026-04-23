# Meta-optimizer package — HyPE prompt optimization
#
# BaseOptimizer and OptimizerRegistry have moved to src.optimizers.
# They are re-exported here for backward compatibility.
from src.optimizers.base import BaseOptimizer, NoOpOptimizer, OptimizerRegistry

from src.meta_optimizer.hype import HyPEOptimizer, Optimizer
from src.meta_optimizer.hyper_templates import (
    HypeMetaPromptBuilder,
    HypeMetaPromptConfig,
    PromptSectionSpec,
)
from src.meta_optimizer.sectional_hype import SectionalHyPEOptimizer

__all__ = [
    # Unified optimizer interface (re-exported from src.optimizers)
    "BaseOptimizer",
    "NoOpOptimizer",
    "OptimizerRegistry",
    # HyPE
    "HyPEOptimizer",
    "Optimizer",
    "HypeMetaPromptBuilder",
    "HypeMetaPromptConfig",
    "PromptSectionSpec",
    # Sectional HyPE
    "SectionalHyPEOptimizer",
]
