"""
Unit-demand valuation functions.
"""

import numpy as np
from typing import Set, Optional
from .base import BaseValuation


class UnitDemandValuation(BaseValuation):
    """
    Unit-demand valuation: v(S) = max_{i ∈ S} v_i
    
    Agent values only the single most valuable item in the bundle.
    This is a special case of submodular valuations with K_max = 1.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 values: Optional[np.ndarray] = None):
        """
        Initialize unit-demand valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            values: Optional predefined item values
        """
        self._predefined_values = values
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize item values."""
        if self._predefined_values is not None:
            self.item_values = self._predefined_values.copy()
        else:
            self.item_values = self.rng.uniform(0, 1, self.n_items)
    
    def value(self, bundle: Set[int]) -> float:
        """Compute unit-demand value (maximum item value)."""
        if not bundle:
            return 0.0
        return max(self.item_values[i] for i in bundle)
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """
        Marginal value is positive only if item is the new maximum.
        """
        if item in bundle:
            return 0.0
        current_max = self.value(bundle)
        new_value = self.item_values[item]
        return max(0.0, new_value - current_max)
    
    def get_saturation_threshold(self) -> int:
        """Unit-demand has saturation threshold K_max = 1."""
        return 1


class ConstrainedUnitDemandValuation(UnitDemandValuation):
    """
    Unit-demand with feasibility constraints.
    Agent can only use items from a feasible subset.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 feasible_items: Optional[Set[int]] = None):
        """
        Initialize constrained unit-demand valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            feasible_items: Set of items this agent can use
        """
        self._feasible_items = feasible_items
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize with feasibility constraints."""
        super()._initialize()
        if self._feasible_items is None:
            n_feasible = self.rng.integers(1, self.n_items + 1)
            self._feasible_items = set(self.rng.choice(
                self.n_items, size=n_feasible, replace=False
            ))
    
    def value(self, bundle: Set[int]) -> float:
        """Value only considers feasible items."""
        feasible_bundle = bundle & self._feasible_items
        if not feasible_bundle:
            return 0.0
        return max(self.item_values[i] for i in feasible_bundle)
