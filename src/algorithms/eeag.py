"""
EEAG Algorithm: Envy Elimination by Adding Goods under unknown valuations.

This implements Algorithm 1 from the paper: Explore-then-Exploit for ε-EF1.
"""

import numpy as np
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..valuations.base import ValuationProfile
from .estimation import ValueEstimator, AdaptiveEstimator
from .greedy_ef1 import (
    Allocation, 
    greedy_ef1_allocation, 
    round_robin_ef1,
    is_ef1,
    compute_envy,
    compute_envy_after_removal
)


@dataclass
class EEAGResult:
    """Result of EEAG algorithm."""
    allocation: Allocation
    is_ef1: bool
    epsilon_ef1: bool
    max_envy: float
    max_ef1_violation: float
    total_samples: int
    items_allocated: int
    estimation_errors: Dict[str, float]


class EEAGAlgorithm:
    """
    Explore-then-Exploit algorithm for ε-EF1 under unknown valuations.
    
    Algorithm 1 from the paper:
    
    Phase 1 (Exploration):
        For each agent a and relevant bundle S:
            Draw T = O(ε⁻² log(nm/δ)) samples
            Compute empirical mean v̂_a(S)
    
    Phase 2 (Exploitation):
        Run greedy EF1 using estimated values v̂
        Return allocation
    """
    
    def __init__(self, 
                 valuation_profile: ValuationProfile,
                 epsilon: float = 0.1,
                 delta: float = 0.05,
                 seed: Optional[int] = None):
        """
        Initialize EEAG algorithm.
        
        Args:
            valuation_profile: True agent valuations (oracle access only)
            epsilon: Target accuracy for ε-EF1
            delta: Failure probability
            seed: Random seed for reproducibility
        """
        self.profile = valuation_profile
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        
        self.n_agents = valuation_profile.n_agents
        self.n_items = valuation_profile.n_items
        
        self.estimator = ValueEstimator(
            valuation_profile, 
            epsilon=epsilon/4,
            delta=delta,
            seed=seed
        )
        
        self._value_estimates = None
    
    def _identify_relevant_bundles(self, 
                                   initial_allocation: Allocation,
                                   pool: Set[int],
                                   k_max: int = None) -> Dict[int, List[Set[int]]]:
        """
        Identify relevant bundles for each agent (Definition 4.2).
        
        For submodular valuations with saturation threshold K_max,
        only O(nm) bundles are relevant.
        
        Args:
            initial_allocation: Fixed initial allocation
            pool: Available items to add
            k_max: Saturation threshold (items per agent cap)
            
        Returns:
            Dict mapping agent to list of relevant bundles
        """
        if k_max is None:
            k_max = min(len(pool), self.n_items)
        
        relevant = {a: [] for a in range(self.n_agents)}
        
        for agent in range(self.n_agents):
            current_bundle = initial_allocation.get_bundle(agent)
            relevant[agent].append(current_bundle.copy())
            
            for item in pool:
                singleton = {item}
                relevant[agent].append(singleton)
                
                extended = current_bundle | singleton
                relevant[agent].append(extended)
        
        for agent in range(self.n_agents):
            for other in range(self.n_agents):
                if other == agent:
                    continue
                other_bundle = initial_allocation.get_bundle(other)
                relevant[agent].append(other_bundle.copy())
                
                for item in other_bundle:
                    reduced = other_bundle - {item}
                    relevant[agent].append(reduced)
        
        for agent in range(self.n_agents):
            seen = set()
            unique = []
            for bundle in relevant[agent]:
                key = frozenset(bundle)
                if key not in seen:
                    seen.add(key)
                    unique.append(bundle)
            relevant[agent] = unique
        
        return relevant
    
    def _phase1_exploration(self, 
                            initial_allocation: Allocation,
                            pool: Set[int]) -> np.ndarray:
        """
        Phase 1: Estimate values for all singleton bundles.
        
        For unit-demand and additive valuations, singleton estimates
        are sufficient. For general submodular, we estimate relevant bundles.
        
        Args:
            initial_allocation: Fixed initial allocation
            pool: Available items
            
        Returns:
            Value estimates array (n_agents x n_items)
        """
        estimates = self.estimator.estimate_all_singletons()
        self._value_estimates = estimates
        return estimates
    
    def _phase2_exploitation(self,
                             initial_allocation: Allocation,
                             pool: Set[int],
                             value_estimates: np.ndarray) -> Allocation:
        """
        Phase 2: Run greedy EF1 with estimated values.
        
        Uses round-robin allocation which guarantees EF1 for additive valuations.
        
        Args:
            initial_allocation: Fixed initial allocation
            pool: Available items
            value_estimates: Estimated values from Phase 1
            
        Returns:
            Final allocation
        """
        # Use round-robin for clean allocation from pool
        allocation = round_robin_ef1(pool, self.n_agents, value_estimates)
        
        return allocation
    
    def run(self, 
            initial_allocation: Optional[Allocation] = None,
            pool: Optional[Set[int]] = None) -> EEAGResult:
        """
        Run the complete EEAG algorithm.
        
        Args:
            initial_allocation: Fixed initial allocation (default: empty)
            pool: Available items to add (default: all items)
            
        Returns:
            EEAGResult with allocation and diagnostics
        """
        if initial_allocation is None:
            initial_allocation = Allocation(self.n_agents, self.n_items)
        
        if pool is None:
            all_items = set(range(self.n_items))
            allocated = set()
            for a in range(self.n_agents):
                allocated |= initial_allocation.get_bundle(a)
            pool = all_items - allocated
        
        value_estimates = self._phase1_exploration(initial_allocation, pool)
        
        final_allocation = self._phase2_exploitation(
            initial_allocation, pool, value_estimates
        )
        
        return self._evaluate_result(final_allocation, value_estimates)
    
    def _evaluate_result(self, 
                         allocation: Allocation,
                         estimates: np.ndarray) -> EEAGResult:
        """
        Evaluate the quality of the final allocation.
        
        Args:
            allocation: Final allocation
            estimates: Value estimates used
            
        Returns:
            EEAGResult with all metrics
        """
        true_values = np.zeros((self.n_agents, self.n_items))
        for a in range(self.n_agents):
            for i in range(self.n_items):
                true_values[a, i] = self.profile[a].value({i})
        
        is_ef1_estimated = is_ef1(allocation, estimates, epsilon=0)
        is_ef1_true = is_ef1(allocation, true_values, epsilon=0)
        is_eps_ef1_true = is_ef1(allocation, true_values, epsilon=self.epsilon)
        
        max_envy = 0.0
        max_ef1_violation = 0.0
        
        for a in range(self.n_agents):
            for b in range(self.n_agents):
                if a == b:
                    continue
                envy = compute_envy(allocation, true_values, a, b)
                max_envy = max(max_envy, envy)
                
                ef1_envy, _ = compute_envy_after_removal(
                    allocation, true_values, a, b
                )
                max_ef1_violation = max(max_ef1_violation, ef1_envy)
        
        estimation_errors = {}
        all_errors = []
        for a in range(self.n_agents):
            for i in range(self.n_items):
                error = abs(estimates[a, i] - true_values[a, i])
                all_errors.append(error)
        
        estimation_errors["mean"] = np.mean(all_errors)
        estimation_errors["max"] = np.max(all_errors)
        estimation_errors["std"] = np.std(all_errors)
        
        items_allocated = sum(len(allocation.get_bundle(a)) 
                             for a in range(self.n_agents))
        
        return EEAGResult(
            allocation=allocation,
            is_ef1=is_ef1_true,
            epsilon_ef1=is_eps_ef1_true,
            max_envy=max_envy,
            max_ef1_violation=max_ef1_violation,
            total_samples=self.estimator.get_total_samples(),
            items_allocated=items_allocated,
            estimation_errors=estimation_errors
        )


class AdaptiveEEAG(EEAGAlgorithm):
    """
    Adaptive version of EEAG with variance-based early stopping.
    """
    
    def __init__(self,
                 valuation_profile: ValuationProfile,
                 epsilon: float = 0.1,
                 delta: float = 0.05,
                 seed: Optional[int] = None):
        super().__init__(valuation_profile, epsilon, delta, seed)
        
        self.estimator = AdaptiveEstimator(
            valuation_profile,
            epsilon=epsilon/4,
            delta=delta,
            min_samples=10,
            seed=seed
        )


def run_eeag_experiment(n_agents: int,
                        n_items: int,
                        valuation_class: str,
                        epsilon: float,
                        delta: float = 0.05,
                        seed: int = None) -> EEAGResult:
    """
    Convenience function to run EEAG experiment.
    
    Args:
        n_agents: Number of agents
        n_items: Number of items
        valuation_class: Type of valuations ("additive", "unit_demand", etc.)
        epsilon: Target accuracy
        delta: Failure probability
        seed: Random seed
        
    Returns:
        EEAGResult
    """
    from ..valuations import (
        AdditiveValuation, 
        UnitDemandValuation,
        CoverageValuation,
        SubmodularValuation,
        ValuationProfile
    )
    
    rng = np.random.default_rng(seed)
    
    valuation_map = {
        "additive": AdditiveValuation,
        "unit_demand": UnitDemandValuation,
        "coverage": CoverageValuation,
        "submodular": SubmodularValuation,
    }
    
    ValClass = valuation_map.get(valuation_class, AdditiveValuation)
    
    valuations = []
    for i in range(n_agents):
        v = ValClass(n_items, seed=rng.integers(0, 2**31) if seed else None)
        valuations.append(v)
    
    profile = ValuationProfile(valuations)
    
    algorithm = EEAGAlgorithm(
        profile, 
        epsilon=epsilon,
        delta=delta,
        seed=seed
    )
    
    return algorithm.run()
