"""
Fairness metrics and verification.
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from ..algorithms.greedy_ef1 import Allocation


def compute_bundle_value(bundle: Set[int], 
                         values: np.ndarray,
                         agent: int) -> float:
    """
    Compute value of a bundle for an agent.
    
    Args:
        bundle: Set of item indices
        values: Value array (n_agents x n_items)
        agent: Agent index
        
    Returns:
        Total bundle value
    """
    if not bundle:
        return 0.0
    return sum(values[agent, i] for i in bundle)


def compute_envy_matrix(allocation: Allocation,
                        values: np.ndarray) -> np.ndarray:
    """
    Compute pairwise envy matrix.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Matrix E where E[a,b] = v_a(B_b) - v_a(B_a)
    """
    n_agents = allocation.n_agents
    envy_matrix = np.zeros((n_agents, n_agents))
    
    for a in range(n_agents):
        value_a = compute_bundle_value(allocation.get_bundle(a), values, a)
        for b in range(n_agents):
            if a == b:
                continue
            value_b = compute_bundle_value(allocation.get_bundle(b), values, a)
            envy_matrix[a, b] = value_b - value_a
    
    return envy_matrix


def is_envy_free(allocation: Allocation,
                 values: np.ndarray,
                 tolerance: float = 1e-9) -> bool:
    """
    Check if allocation is envy-free.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        tolerance: Numerical tolerance
        
    Returns:
        True if no agent envies another
    """
    envy_matrix = compute_envy_matrix(allocation, values)
    return np.all(envy_matrix <= tolerance)


def compute_ef1_violations(allocation: Allocation,
                           values: np.ndarray) -> np.ndarray:
    """
    Compute EF1 violations for all agent pairs.
    
    For each pair (a, b), compute:
    min_{g in B_b} [v_a(B_b - {g}) - v_a(B_a)]
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Matrix V where V[a,b] = EF1 violation (positive means violation)
    """
    n_agents = allocation.n_agents
    violations = np.zeros((n_agents, n_agents))
    
    for a in range(n_agents):
        value_a = compute_bundle_value(allocation.get_bundle(a), values, a)
        
        for b in range(n_agents):
            if a == b:
                continue
            
            bundle_b = allocation.get_bundle(b)
            if not bundle_b:
                violations[a, b] = 0.0
                continue
            
            min_envy_after_removal = float('inf')
            for item in bundle_b:
                reduced_bundle = bundle_b - {item}
                value_reduced = compute_bundle_value(reduced_bundle, values, a)
                envy = value_reduced - value_a
                min_envy_after_removal = min(min_envy_after_removal, envy)
            
            violations[a, b] = max(0, min_envy_after_removal)
    
    return violations


def is_ef1(allocation: Allocation,
           values: np.ndarray,
           epsilon: float = 0.0) -> bool:
    """
    Check if allocation is ε-EF1.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        epsilon: Allowed violation
        
    Returns:
        True if allocation is ε-EF1
    """
    violations = compute_ef1_violations(allocation, values)
    return np.all(violations <= epsilon)


def compute_efx_violations(allocation: Allocation,
                           values: np.ndarray) -> np.ndarray:
    """
    Compute EFX violations for all agent pairs.
    
    For each pair (a, b), compute:
    max_{g in B_b} [v_a(B_b - {g}) - v_a(B_a)]
    
    EFX requires this to be <= 0 for all pairs.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Matrix V where V[a,b] = EFX violation
    """
    n_agents = allocation.n_agents
    violations = np.zeros((n_agents, n_agents))
    
    for a in range(n_agents):
        value_a = compute_bundle_value(allocation.get_bundle(a), values, a)
        
        for b in range(n_agents):
            if a == b:
                continue
            
            bundle_b = allocation.get_bundle(b)
            if not bundle_b:
                violations[a, b] = 0.0
                continue
            
            max_envy_after_removal = float('-inf')
            for item in bundle_b:
                reduced_bundle = bundle_b - {item}
                value_reduced = compute_bundle_value(reduced_bundle, values, a)
                envy = value_reduced - value_a
                max_envy_after_removal = max(max_envy_after_removal, envy)
            
            violations[a, b] = max(0, max_envy_after_removal)
    
    return violations


def is_efx(allocation: Allocation,
           values: np.ndarray,
           epsilon: float = 0.0) -> bool:
    """
    Check if allocation is ε-EFX.
    
    Args:
        allocation: Current allocation  
        values: Value estimates
        epsilon: Allowed violation
        
    Returns:
        True if allocation is ε-EFX
    """
    violations = compute_efx_violations(allocation, values)
    return np.all(violations <= epsilon)


def compute_proportionality_violations(allocation: Allocation,
                                       values: np.ndarray) -> np.ndarray:
    """
    Compute proportionality violations.
    
    Agent a's proportional share is (1/n) * v_a(all items).
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Array of violations per agent
    """
    n_agents = allocation.n_agents
    n_items = values.shape[1]
    violations = np.zeros(n_agents)
    
    for a in range(n_agents):
        total_value = sum(values[a, i] for i in range(n_items))
        prop_share = total_value / n_agents
        
        bundle_value = compute_bundle_value(allocation.get_bundle(a), values, a)
        violations[a] = max(0, prop_share - bundle_value)
    
    return violations


def compute_nash_welfare(allocation: Allocation,
                         values: np.ndarray) -> float:
    """
    Compute Nash social welfare (product of utilities).
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Nash welfare (geometric mean of utilities)
    """
    n_agents = allocation.n_agents
    utilities = []
    
    for a in range(n_agents):
        u = compute_bundle_value(allocation.get_bundle(a), values, a)
        utilities.append(max(u, 1e-10))
    
    return np.exp(np.mean(np.log(utilities)))


def compute_utilitarian_welfare(allocation: Allocation,
                                values: np.ndarray) -> float:
    """
    Compute utilitarian welfare (sum of utilities).
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Sum of agent utilities
    """
    n_agents = allocation.n_agents
    total = 0.0
    
    for a in range(n_agents):
        total += compute_bundle_value(allocation.get_bundle(a), values, a)
    
    return total


def compute_egalitarian_welfare(allocation: Allocation,
                                values: np.ndarray) -> float:
    """
    Compute egalitarian welfare (minimum utility).
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        Minimum agent utility
    """
    n_agents = allocation.n_agents
    min_utility = float('inf')
    
    for a in range(n_agents):
        u = compute_bundle_value(allocation.get_bundle(a), values, a)
        min_utility = min(min_utility, u)
    
    return min_utility


class FairnessMetrics:
    """Compute all fairness metrics for an allocation."""
    
    def __init__(self, allocation: Allocation, values: np.ndarray):
        self.allocation = allocation
        self.values = values
        
        self._envy_matrix = None
        self._ef1_violations = None
        self._efx_violations = None
    
    @property
    def envy_matrix(self) -> np.ndarray:
        if self._envy_matrix is None:
            self._envy_matrix = compute_envy_matrix(self.allocation, self.values)
        return self._envy_matrix
    
    @property
    def ef1_violations(self) -> np.ndarray:
        if self._ef1_violations is None:
            self._ef1_violations = compute_ef1_violations(self.allocation, self.values)
        return self._ef1_violations
    
    @property
    def efx_violations(self) -> np.ndarray:
        if self._efx_violations is None:
            self._efx_violations = compute_efx_violations(self.allocation, self.values)
        return self._efx_violations
    
    def is_ef(self, tolerance: float = 1e-9) -> bool:
        return np.all(self.envy_matrix <= tolerance)
    
    def is_ef1(self, epsilon: float = 0.0) -> bool:
        return np.all(self.ef1_violations <= epsilon)
    
    def is_efx(self, epsilon: float = 0.0) -> bool:
        return np.all(self.efx_violations <= epsilon)
    
    def max_envy(self) -> float:
        return np.max(self.envy_matrix)
    
    def max_ef1_violation(self) -> float:
        return np.max(self.ef1_violations)
    
    def max_efx_violation(self) -> float:
        return np.max(self.efx_violations)
    
    def nash_welfare(self) -> float:
        return compute_nash_welfare(self.allocation, self.values)
    
    def utilitarian_welfare(self) -> float:
        return compute_utilitarian_welfare(self.allocation, self.values)
    
    def egalitarian_welfare(self) -> float:
        return compute_egalitarian_welfare(self.allocation, self.values)
    
    def summary(self) -> Dict:
        return {
            "is_ef": self.is_ef(),
            "is_ef1": self.is_ef1(),
            "is_efx": self.is_efx(),
            "max_envy": self.max_envy(),
            "max_ef1_violation": self.max_ef1_violation(),
            "max_efx_violation": self.max_efx_violation(),
            "nash_welfare": self.nash_welfare(),
            "utilitarian_welfare": self.utilitarian_welfare(),
            "egalitarian_welfare": self.egalitarian_welfare(),
        }
