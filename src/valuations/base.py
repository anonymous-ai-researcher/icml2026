"""
Base class for valuation functions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Set, List, Optional


class BaseValuation(ABC):
    """Abstract base class for valuation functions."""
    
    def __init__(self, n_items: int, seed: Optional[int] = None):
        """
        Initialize valuation function.
        
        Args:
            n_items: Number of items in the ground set
            seed: Random seed for reproducibility
        """
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize valuation-specific parameters."""
        pass
    
    @abstractmethod
    def value(self, bundle: Set[int]) -> float:
        """
        Compute the value of a bundle.
        
        Args:
            bundle: Set of item indices
            
        Returns:
            Non-negative real value
        """
        pass
    
    def marginal_value(self, bundle: Set[int], item: int) -> float:
        """
        Compute marginal value of adding an item to a bundle.
        
        Args:
            bundle: Current bundle (set of item indices)
            item: Item to add
            
        Returns:
            Marginal value v(S ∪ {item}) - v(S)
        """
        if item in bundle:
            return 0.0
        return self.value(bundle | {item}) - self.value(bundle)
    
    def is_submodular(self, n_tests: int = 100) -> bool:
        """
        Test submodularity via random sampling.
        
        Args:
            n_tests: Number of random tests
            
        Returns:
            True if all tests pass (necessary but not sufficient)
        """
        items = list(range(self.n_items))
        for _ in range(n_tests):
            n_A = self.rng.integers(0, self.n_items)
            n_B = self.rng.integers(n_A, self.n_items + 1)
            
            A_items = self.rng.choice(items, size=n_A, replace=False)
            remaining = [i for i in items if i not in A_items]
            if len(remaining) == 0:
                continue
            B_extra = self.rng.choice(remaining, size=min(n_B - n_A, len(remaining)), replace=False)
            
            A = set(A_items)
            B = A | set(B_extra)
            
            for item in items:
                if item not in B:
                    if self.marginal_value(A, item) < self.marginal_value(B, item) - 1e-9:
                        return False
        return True
    
    def noisy_sample(self, bundle: Set[int], noise_std: float = 0.1) -> float:
        """
        Get a noisy sample of the bundle value.
        
        Args:
            bundle: Set of item indices
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy value observation
        """
        true_value = self.value(bundle)
        noise = self.rng.normal(0, noise_std)
        return max(0.0, true_value + noise)
    
    def bounded_sample(self, bundle: Set[int]) -> float:
        """
        Get a bounded noisy sample in [0, 1].
        Following Definition 2.12: X ∈ [0,1] with E[X] = v(S).
        
        Args:
            bundle: Set of item indices
            
        Returns:
            Bounded noisy observation
        """
        true_value = self.value(bundle)
        true_value = np.clip(true_value, 0, 1)
        noise = self.rng.uniform(-0.5, 0.5) * 0.2
        return np.clip(true_value + noise, 0, 1)


class ValuationProfile:
    """Collection of valuations for multiple agents."""
    
    def __init__(self, valuations: List[BaseValuation]):
        """
        Initialize valuation profile.
        
        Args:
            valuations: List of valuation functions, one per agent
        """
        self.valuations = valuations
        self.n_agents = len(valuations)
        self.n_items = valuations[0].n_items if valuations else 0
        
        for v in valuations:
            if v.n_items != self.n_items:
                raise ValueError("All valuations must have the same number of items")
    
    def __getitem__(self, agent: int) -> BaseValuation:
        """Get valuation for a specific agent."""
        return self.valuations[agent]
    
    def __len__(self) -> int:
        """Number of agents."""
        return self.n_agents
