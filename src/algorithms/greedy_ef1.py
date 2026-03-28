"""
Greedy EF1 allocation algorithm.
"""

import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from ..valuations.base import ValuationProfile


class Allocation:
    """Represents an allocation of items to agents."""
    
    def __init__(self, n_agents: int, n_items: int):
        """
        Initialize empty allocation.
        
        Args:
            n_agents: Number of agents
            n_items: Number of items
        """
        self.n_agents = n_agents
        self.n_items = n_items
        self.bundles: List[Set[int]] = [set() for _ in range(n_agents)]
        self._item_owner: Dict[int, int] = {}
    
    def allocate(self, item: int, agent: int):
        """
        Allocate an item to an agent.
        
        Args:
            item: Item index
            agent: Agent index
        """
        if item in self._item_owner:
            old_owner = self._item_owner[item]
            self.bundles[old_owner].discard(item)
        
        self.bundles[agent].add(item)
        self._item_owner[item] = agent
    
    def get_bundle(self, agent: int) -> Set[int]:
        """Get the bundle allocated to an agent."""
        return self.bundles[agent].copy()
    
    def get_owner(self, item: int) -> Optional[int]:
        """Get the agent who owns an item, or None if unallocated."""
        return self._item_owner.get(item)
    
    def copy(self) -> "Allocation":
        """Create a deep copy of the allocation."""
        new_alloc = Allocation(self.n_agents, self.n_items)
        for agent in range(self.n_agents):
            new_alloc.bundles[agent] = self.bundles[agent].copy()
        new_alloc._item_owner = self._item_owner.copy()
        return new_alloc


def compute_envy(allocation: Allocation, 
                 values: np.ndarray,
                 agent_a: int, 
                 agent_b: int) -> float:
    """
    Compute envy of agent_a towards agent_b.
    
    Args:
        allocation: Current allocation
        values: Value estimates (n_agents x n_items)
        agent_a: Envying agent
        agent_b: Envied agent
        
    Returns:
        Envy value (positive means a envies b)
    """
    bundle_a = allocation.get_bundle(agent_a)
    bundle_b = allocation.get_bundle(agent_b)
    
    value_a = sum(values[agent_a, i] for i in bundle_a) if bundle_a else 0
    value_b = sum(values[agent_a, i] for i in bundle_b) if bundle_b else 0
    
    return value_b - value_a


def compute_envy_after_removal(allocation: Allocation,
                               values: np.ndarray,
                               agent_a: int,
                               agent_b: int) -> Tuple[float, Optional[int]]:
    """
    Compute minimum envy after removing one item from agent_b's bundle.
    
    This is used for EF1 checking: a does not EF1-envy b if there exists
    some item g in b's bundle such that a doesn't envy b after removing g.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        agent_a: Envying agent
        agent_b: Envied agent
        
    Returns:
        (minimum envy after removal, item to remove)
    """
    bundle_a = allocation.get_bundle(agent_a)
    bundle_b = allocation.get_bundle(agent_b)
    
    if not bundle_b:
        return compute_envy(allocation, values, agent_a, agent_b), None
    
    value_a = sum(values[agent_a, i] for i in bundle_a) if bundle_a else 0
    
    min_envy = float('inf')
    best_item = None
    
    for item in bundle_b:
        reduced_bundle = bundle_b - {item}
        value_b_reduced = sum(values[agent_a, i] for i in reduced_bundle)
        envy = value_b_reduced - value_a
        
        if envy < min_envy:
            min_envy = envy
            best_item = item
    
    return min_envy, best_item


def is_ef1(allocation: Allocation, values: np.ndarray, 
           epsilon: float = 0.0) -> bool:
    """
    Check if allocation is ε-EF1.
    
    Args:
        allocation: Current allocation
        values: Value estimates (n_agents x n_items)
        epsilon: Allowed violation tolerance
        
    Returns:
        True if allocation is ε-EF1
    """
    n_agents = allocation.n_agents
    
    for a in range(n_agents):
        for b in range(n_agents):
            if a == b:
                continue
            
            envy_after_removal, _ = compute_envy_after_removal(
                allocation, values, a, b
            )
            
            if envy_after_removal > epsilon:
                return False
    
    return True


def find_most_envious_pair(allocation: Allocation,
                           values: np.ndarray) -> Tuple[int, int, float]:
    """
    Find the pair (a, b) with maximum envy.
    
    Args:
        allocation: Current allocation
        values: Value estimates
        
    Returns:
        (envying agent, envied agent, envy amount)
    """
    n_agents = allocation.n_agents
    max_envy = float('-inf')
    max_pair = (0, 1)
    
    for a in range(n_agents):
        for b in range(n_agents):
            if a == b:
                continue
            envy = compute_envy(allocation, values, a, b)
            if envy > max_envy:
                max_envy = envy
                max_pair = (a, b)
    
    return max_pair[0], max_pair[1], max_envy


def greedy_ef1_allocation(pool: Set[int],
                          initial_allocation: Allocation,
                          values: np.ndarray,
                          max_iterations: int = None) -> Allocation:
    """
    Greedy EF1 allocation algorithm.
    
    Repeatedly allocates items from the pool to reduce maximum envy,
    following the approach of Lipton et al. (2004).
    
    Args:
        pool: Set of available items to allocate
        initial_allocation: Starting allocation (may be non-empty)
        values: Value estimates (n_agents x n_items)
        max_iterations: Maximum number of items to allocate
        
    Returns:
        EF1 allocation
    """
    allocation = initial_allocation.copy()
    remaining_pool = pool.copy()
    
    if max_iterations is None:
        max_iterations = len(pool) * 10
    
    iterations = 0
    
    while remaining_pool and iterations < max_iterations:
        envious_a, envied_b, max_envy = find_most_envious_pair(allocation, values)
        
        if max_envy <= 0:
            break
        
        best_item = None
        best_reduction = float('-inf')
        
        for item in remaining_pool:
            current_envy = compute_envy(allocation, values, envious_a, envied_b)
            
            test_alloc = allocation.copy()
            test_alloc.allocate(item, envious_a)
            new_envy = compute_envy(test_alloc, values, envious_a, envied_b)
            
            reduction = current_envy - new_envy
            if reduction > best_reduction:
                best_reduction = reduction
                best_item = item
        
        if best_item is not None and best_reduction > 0:
            allocation.allocate(best_item, envious_a)
            remaining_pool.remove(best_item)
        else:
            for item in remaining_pool:
                allocation.allocate(item, envious_a)
                remaining_pool.remove(item)
                break
        
        iterations += 1
        
        if is_ef1(allocation, values):
            break
    
    return allocation


def round_robin_ef1(pool: Set[int],
                    n_agents: int,
                    values: np.ndarray) -> Allocation:
    """
    Round-robin allocation achieving EF1.
    
    Agents take turns picking their most-valued remaining item.
    This achieves EF1 for additive valuations.
    
    Args:
        pool: Set of items to allocate
        n_agents: Number of agents
        values: Value estimates (n_agents x n_items)
        
    Returns:
        EF1 allocation
    """
    allocation = Allocation(n_agents, max(pool) + 1 if pool else 0)
    remaining = list(pool)
    
    agent_order = list(range(n_agents))
    
    while remaining:
        for agent in agent_order:
            if not remaining:
                break
            
            best_item = max(remaining, key=lambda i: values[agent, i])
            allocation.allocate(best_item, agent)
            remaining.remove(best_item)
    
    return allocation
