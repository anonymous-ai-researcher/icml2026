"""
Coverage valuation functions.
"""

import numpy as np
from typing import Set, List, Optional
from .base import BaseValuation


class CoverageValuation(BaseValuation):
    """
    Coverage valuation: v(S) = |⋃_{i ∈ S} C_i|
    
    Each item i covers a set C_i of elements. The value of a bundle
    is the total number of distinct elements covered.
    
    This is a canonical submodular function arising in facility location,
    sensor placement, and information retrieval.
    """
    
    def __init__(self, n_items: int, n_elements: int = None,
                 seed: Optional[int] = None,
                 coverage_sets: Optional[List[Set[int]]] = None):
        """
        Initialize coverage valuation.
        
        Args:
            n_items: Number of items (facilities/sensors)
            n_elements: Number of ground elements to cover
            seed: Random seed
            coverage_sets: Optional predefined coverage sets
        """
        self._n_elements = n_elements or n_items * 3
        self._predefined_coverage = coverage_sets
        super().__init__(n_items, seed)
    
    def _initialize(self):
        """Initialize coverage sets."""
        if self._predefined_coverage is not None:
            self.coverage_sets = self._predefined_coverage
        else:
            self.coverage_sets = []
            for _ in range(self.n_items):
                coverage_size = self.rng.integers(1, self._n_elements // 2 + 1)
                covered = set(self.rng.choice(
                    self._n_elements, size=coverage_size, replace=False
                ))
                self.coverage_sets.append(covered)
        
        self.n_elements = self._n_elements
    
    def value(self, bundle: Set[int]) -> float:
        """Compute coverage value (number of covered elements)."""
        if not bundle:
            return 0.0
        covered = set()
        for i in bundle:
            covered |= self.coverage_sets[i]
        return float(len(covered)) / self.n_elements
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """Marginal value is the number of newly covered elements."""
        if item in bundle:
            return 0.0
        
        currently_covered = set()
        for i in bundle:
            currently_covered |= self.coverage_sets[i]
        
        new_coverage = self.coverage_sets[item] - currently_covered
        return float(len(new_coverage)) / self.n_elements


class WeightedCoverageValuation(CoverageValuation):
    """
    Weighted coverage: v(S) = Σ_{e ∈ ⋃_{i ∈ S} C_i} w_e
    
    Elements have weights; value is total weight of covered elements.
    """
    
    def __init__(self, n_items: int, n_elements: int = None,
                 seed: Optional[int] = None,
                 element_weights: Optional[np.ndarray] = None):
        """
        Initialize weighted coverage valuation.
        
        Args:
            n_items: Number of items
            n_elements: Number of elements
            seed: Random seed
            element_weights: Optional predefined element weights
        """
        self._element_weights = element_weights
        super().__init__(n_items, n_elements, seed)
    
    def _initialize(self):
        """Initialize coverage sets and element weights."""
        super()._initialize()
        if self._element_weights is not None:
            self.element_weights = self._element_weights.copy()
        else:
            self.element_weights = self.rng.uniform(0, 1, self.n_elements)
        
        self._total_weight = np.sum(self.element_weights)
    
    def value(self, bundle: Set[int]) -> float:
        """Compute weighted coverage value."""
        if not bundle:
            return 0.0
        covered = set()
        for i in bundle:
            covered |= self.coverage_sets[i]
        
        total_weight = sum(self.element_weights[e] for e in covered)
        return total_weight / self._total_weight
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """Marginal value is weight of newly covered elements."""
        if item in bundle:
            return 0.0
        
        currently_covered = set()
        for i in bundle:
            currently_covered |= self.coverage_sets[i]
        
        new_coverage = self.coverage_sets[item] - currently_covered
        new_weight = sum(self.element_weights[e] for e in new_coverage)
        return new_weight / self._total_weight


class SetCoverValuation(CoverageValuation):
    """
    Set cover valuation with target coverage requirement.
    Value is 1 if target elements are covered, 0 otherwise.
    """
    
    def __init__(self, n_items: int, n_elements: int = None,
                 seed: Optional[int] = None,
                 target_elements: Optional[Set[int]] = None):
        """
        Initialize set cover valuation.
        
        Args:
            n_items: Number of items
            n_elements: Number of elements
            seed: Random seed
            target_elements: Elements that must be covered for value 1
        """
        self._target_elements = target_elements
        super().__init__(n_items, n_elements, seed)
    
    def _initialize(self):
        """Initialize coverage sets and target."""
        super()._initialize()
        if self._target_elements is not None:
            self.target = self._target_elements
        else:
            target_size = self.rng.integers(1, self.n_elements // 2 + 1)
            self.target = set(self.rng.choice(
                self.n_elements, size=target_size, replace=False
            ))
    
    def value(self, bundle: Set[int]) -> float:
        """Value is fraction of target elements covered."""
        if not bundle:
            return 0.0
        covered = set()
        for i in bundle:
            covered |= self.coverage_sets[i]
        
        target_covered = covered & self.target
        return float(len(target_covered)) / len(self.target)
