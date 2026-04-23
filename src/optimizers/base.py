"""
Base optimizer interface and registry for genotype optimization.

All optimizers must inherit from BaseOptimizer and implement optimize().
Use OptimizerRegistry to register and create optimizers by name from config.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseOptimizer(ABC):
    """
    Abstract base class for all genotype optimizers.

    Subclasses implement optimize() which takes a base_genotype dict,
    an evaluator, and training participants, and returns an optimised
    genotype dict.
    """

    def __init__(self, model, config: Optional[Dict] = None):
        """
        Args:
            model: Language model used for optimization
            config: Full experiment config dict
        """
        self.model = model
        self.config = config or {}

    @abstractmethod
    def optimize(
        self,
        base_genotype: Dict,
        evaluator,
        dev_participants,
    ) -> Dict:
        """
        Optimize the genotype.

        Args:
            base_genotype: Initial genotype dict
            evaluator: MyEvaluator instance for scoring
            dev_participants: Training participants (pandas DataFrame)

        Returns:
            Optimized genotype dict
        """

    def get_name(self) -> str:
        """Return optimizer class name."""
        return self.__class__.__name__


class NoOpOptimizer(BaseOptimizer):
    """
    Pass-through optimizer — returns base_genotype unchanged.

    Use with optimization.method: "none" to skip optimization while
    keeping the pipeline structure intact (useful for baselines).
    """

    def optimize(
        self,
        base_genotype: Dict,
        evaluator,
        dev_participants,
    ) -> Dict:
        """Return base_genotype as-is without any modification."""
        print("⏭️  NoOpOptimizer: пропуск оптимизации, возвращается базовый генотип.")
        return base_genotype.copy()


class OptimizerRegistry:
    """
    Registry for optimizer classes.

    Allows registering optimizers by string name and creating instances
    from config, enabling config-driven dispatch without if/elif chains.

    Usage:
        OptimizerRegistry.register("hype", SectionalHyPEOptimizer)
        optimizer = OptimizerRegistry.create("hype", model, config)
        genotype = optimizer.optimize(base_genotype, evaluator, dev_participants)
    """

    _optimizers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, optimizer_class: type) -> None:
        """
        Register an optimizer class under a string key.

        Args:
            name: Key used in config (e.g. "hype", "evolution", "none")
            optimizer_class: Class that inherits from BaseOptimizer
        """
        if not issubclass(optimizer_class, BaseOptimizer):
            raise ValueError(
                f"{optimizer_class.__name__} must inherit from BaseOptimizer"
            )
        cls._optimizers[name] = optimizer_class

    @classmethod
    def get(cls, name: str) -> type:
        """
        Return optimizer class by name.

        Raises:
            KeyError: if name is not registered
        """
        if name not in cls._optimizers:
            available = list(cls._optimizers.keys())
            raise KeyError(
                f"Optimizer '{name}' not found. Available: {available}"
            )
        return cls._optimizers[name]

    @classmethod
    def create(cls, name: str, model, config: Optional[Dict] = None) -> BaseOptimizer:
        """
        Instantiate an optimizer by name.

        Args:
            name: Registered optimizer name
            model: Language model
            config: Full experiment config dict

        Returns:
            BaseOptimizer instance
        """
        optimizer_class = cls.get(name)
        return optimizer_class(model=model, config=config)

    @classmethod
    def list_optimizers(cls) -> List[str]:
        """Return list of all registered optimizer names."""
        return list(cls._optimizers.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

# NoOp — always available
OptimizerRegistry.register("none", NoOpOptimizer)

# HyPE and Evolution are registered lazily to avoid circular imports.
# They are registered in src/optimizers/__init__.py after all modules load.
