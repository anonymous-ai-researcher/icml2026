"""
General submodular valuation functions.
"""

import numpy as np
from typing import Set, Optional, Callable
from .base import BaseValuation


class SubmodularValuation(BaseValuation):
    """
    General submodular valuation with diminishing returns.
    
    Implements v(S) = Σ_{i ∈ S} v_i · α^{|S ∩ P_i|}
    where P_i is a random predecessor set and α < 1 controls
    the rate of diminishing returns.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 alpha: float = 0.5):
        """
        Initialize submodular valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            alpha: Diminishing returns parameter (0 < alpha < 1)
        """
        self.alpha = alpha
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize item values and interaction structure."""
        self.item_values = self.rng.uniform(0.1, 1, self.n_items)
        
        self.predecessors = []
        items = list(range(self.n_items))
        self.rng.shuffle(items)
        
        for idx, item in enumerate(items):
            n_pred = self.rng.integers(0, min(idx + 1, 5))
            if idx > 0 and n_pred > 0:
                preds = set(self.rng.choice(items[:idx], size=n_pred, replace=False))
            else:
                preds = set()
            
            while len(self.predecessors) <= item:
                self.predecessors.append(set())
            self.predecessors[item] = preds
    
    def value(self, bundle: Set[int]) -> float:
        """
        Compute submodular value with diminishing returns.
        """
        if not bundle:
            return 0.0
        
        total = 0.0
        bundle_list = sorted(bundle)
        
        for item in bundle_list:
            overlap = len(bundle & self.predecessors[item])
            discount = self.alpha ** overlap
            total += self.item_values[item] * discount
        
        max_possible = sum(self.item_values)
        return min(total / max_possible, 1.0)


class BudgetAdditiveValuation(BaseValuation):
    """
    Budget-additive valuation: v(S) = min(B, Σ_{i ∈ S} v_i)
    
    Additive up to a budget cap B. This is submodular.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 budget: Optional[float] = None):
        """
        Initialize budget-additive valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            budget: Maximum value cap
        """
        self._budget = budget
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize item values and budget."""
        self.item_values = self.rng.uniform(0, 1, self.n_items)
        
        if self._budget is not None:
            self.budget = self._budget
        else:
            total = sum(self.item_values)
            self.budget = self.rng.uniform(0.3 * total, 0.7 * total)
    
    def value(self, bundle: Set[int]) -> float:
        """Compute budget-additive value."""
        if not bundle:
            return 0.0
        additive_value = sum(self.item_values[i] for i in bundle)
        return min(self.budget, additive_value) / self.budget
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """Marginal value respects budget constraint."""
        if item in bundle:
            return 0.0
        current = sum(self.item_values[i] for i in bundle) if bundle else 0
        if current >= self.budget:
            return 0.0
        return min(self.item_values[item], self.budget - current) / self.budget


class MatroidRankValuation(BaseValuation):
    """
    Matroid rank valuation based on uniform matroid.
    v(S) = min(|S|, k) for some rank k.
    
    This is a canonical submodular function.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 rank: Optional[int] = None):
        """
        Initialize matroid rank valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            rank: Matroid rank k
        """
        self._rank = rank
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize rank."""
        if self._rank is not None:
            self.rank = self._rank
        else:
            self.rank = self.rng.integers(1, self.n_items // 2 + 1)
    
    def value(self, bundle: Set[int]) -> float:
        """Compute rank value."""
        return min(len(bundle), self.rank) / self.rank
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """Marginal value is 1 if below rank, 0 otherwise."""
        if item in bundle:
            return 0.0
        if len(bundle) >= self.rank:
            return 0.0
        return 1.0 / self.rank


class ConcaveCompositionValuation(BaseValuation):
    """
    Concave over additive: v(S) = f(Σ_{i ∈ S} v_i)
    where f is concave (e.g., sqrt, log).
    
    This is submodular when f is concave and non-decreasing.
    """
    
    def __init__(self, n_items: int, seed: Optional[int] = None,
                 concave_fn: str = "sqrt"):
        """
        Initialize concave composition valuation.
        
        Args:
            n_items: Number of items
            seed: Random seed
            concave_fn: Concave function type ("sqrt", "log", "power")
        """
        self.concave_fn_type = concave_fn
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize item values and concave function."""
        self.item_values = self.rng.uniform(0, 1, self.n_items)
        self._max_value = sum(self.item_values)
        
        if self.concave_fn_type == "sqrt":
            self._concave = lambda x: np.sqrt(x)
        elif self.concave_fn_type == "log":
            self._concave = lambda x: np.log1p(x)
        elif self.concave_fn_type == "power":
            self._concave = lambda x: x ** 0.5
        else:
            self._concave = lambda x: np.sqrt(x)
        
        self._max_f = self._concave(self._max_value)
    
    def value(self, bundle: Set[int]) -> float:
        """Compute concave composition value."""
        if not bundle:
            return 0.0
        additive_value = sum(self.item_values[i] for i in bundle)
        return self._concave(additive_value) / self._max_f
