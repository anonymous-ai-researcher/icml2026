"""
Additive valuation functions.
"""

import numpy as np
from typing import Set, Optional
from .base import BaseValuation


class AdditiveValuation(BaseValuation):
    """
    Additive valuation: v(S) = Σ_{i ∈ S} v_i
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None, 
                 values: Optional[np.ndarray] = None):
        """
        Initialize additive valuation.
        
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
        """Compute additive value of bundle."""
        if not bundle:
            return 0.0
        return sum(self.item_values[i] for i in bundle)
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """Marginal value equals item value for additive functions."""
        if item in bundle:
            return 0.0
        return self.item_values[item]


class BinaryAdditiveValuation(AdditiveValuation):
    """
    Binary additive valuation: v_i ∈ {0, 1}
    """
    
    def _initialize(self):
        """Initialize binary item values."""
        if self._predefined_values is not None:
            self.item_values = self._predefined_values.copy()
        else:
            p = self.rng.uniform(0.3, 0.7)
            self.item_values = self.rng.choice([0.0, 1.0], size=self.n_items, 
                                                p=[1-p, p])


class IdenticalAdditiveValuation(AdditiveValuation):
    """
    All agents share identical additive valuations.
    Useful for testing worst-case scenarios.
    """
    
    _shared_values = None
    
    def _initialize(self):
        """Use shared values across all instances."""
        if IdenticalAdditiveValuation._shared_values is None or \
           len(IdenticalAdditiveValuation._shared_values) != self.n_items:
            IdenticalAdditiveValuation._shared_values = self.rng.uniform(0, 1, self.n_items)
        self.item_values = IdenticalAdditiveValuation._shared_values.copy()
    
    @classmethod
    def reset_shared_values(cls):
        """Reset shared values for new experiment."""
        cls._shared_values = None
