"""
Value estimation from noisy samples.
"""

import numpy as np
from typing import Set, Dict, Tuple, Optional
from ..valuations.base import BaseValuation, ValuationProfile


class ValueEstimator:
    """
    Estimate bundle values from noisy samples using Hoeffding-based sampling.
    
    For ε-accuracy with probability 1-δ, we need T = O(ε⁻² log(1/δ)) samples
    per (agent, bundle) pair.
    """
    
    def __init__(self, valuation_profile: ValuationProfile,
                 epsilon: float = 0.1,
                 delta: float = 0.05,
                 seed: Optional[int] = None):
        """
        Initialize value estimator.
        
        Args:
            valuation_profile: True valuations (used as oracle)
            epsilon: Target accuracy
            delta: Failure probability
            seed: Random seed
        """
        self.profile = valuation_profile
        self.epsilon = epsilon
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        
        self.n_agents = valuation_profile.n_agents
        self.n_items = valuation_profile.n_items
        
        self._estimates: Dict[Tuple[int, frozenset], float] = {}
        self._sample_counts: Dict[Tuple[int, frozenset], int] = {}
        self._total_samples = 0
    
    def samples_needed(self, n_bundles: int = 1) -> int:
        """
        Compute number of samples needed per bundle for Hoeffding guarantee.
        
        By Hoeffding's inequality:
        P(|X̄ - μ| > ε) ≤ 2 exp(-2Tε²)
        
        For δ/(n_agents * n_bundles) failure probability per estimate:
        T = (1/(2ε²)) * log(2 * n_agents * n_bundles / δ)
        
        Args:
            n_bundles: Number of bundles to estimate
            
        Returns:
            Number of samples needed per bundle
        """
        adjusted_delta = self.delta / (self.n_agents * max(n_bundles, 1))
        T = int(np.ceil((1 / (2 * self.epsilon**2)) * np.log(2 / adjusted_delta)))
        return max(T, 1)
    
    def estimate_value(self, agent: int, bundle: Set[int], 
                       force_resample: bool = False) -> float:
        """
        Estimate value of bundle for agent using noisy samples.
        
        Args:
            agent: Agent index
            bundle: Set of item indices
            force_resample: If True, discard cached estimate
            
        Returns:
            Estimated bundle value
        """
        key = (agent, frozenset(bundle))
        
        if not force_resample and key in self._estimates:
            return self._estimates[key]
        
        valuation = self.profile[agent]
        T = self.samples_needed()
        
        samples = []
        for _ in range(T):
            sample = valuation.bounded_sample(bundle)
            samples.append(sample)
            self._total_samples += 1
        
        estimate = np.mean(samples)
        self._estimates[key] = estimate
        self._sample_counts[key] = T
        
        return estimate
    
    def estimate_all_singletons(self) -> np.ndarray:
        """
        Estimate values of all singleton bundles.
        
        Returns:
            Array of shape (n_agents, n_items) with estimated values
        """
        estimates = np.zeros((self.n_agents, self.n_items))
        
        for agent in range(self.n_agents):
            for item in range(self.n_items):
                estimates[agent, item] = self.estimate_value(agent, {item})
        
        return estimates
    
    def estimate_relevant_bundles(self, 
                                   bundles_per_agent: Dict[int, list]) -> Dict:
        """
        Estimate values for relevant bundles only.
        
        Args:
            bundles_per_agent: Dict mapping agent index to list of bundles
            
        Returns:
            Dict mapping (agent, bundle) to estimated value
        """
        estimates = {}
        
        for agent, bundles in bundles_per_agent.items():
            for bundle in bundles:
                key = (agent, frozenset(bundle))
                estimates[key] = self.estimate_value(agent, bundle)
        
        return estimates
    
    def get_estimation_error(self, agent: int, bundle: Set[int]) -> float:
        """
        Compute actual estimation error (for evaluation only).
        
        Args:
            agent: Agent index
            bundle: Bundle
            
        Returns:
            |v̂(S) - v(S)|
        """
        key = (agent, frozenset(bundle))
        if key not in self._estimates:
            return float('inf')
        
        true_value = self.profile[agent].value(bundle)
        return abs(self._estimates[key] - true_value)
    
    def get_total_samples(self) -> int:
        """Return total number of samples used."""
        return self._total_samples
    
    def get_sample_complexity(self) -> Dict:
        """
        Return sample complexity statistics.
        
        Returns:
            Dict with complexity metrics
        """
        return {
            "total_samples": self._total_samples,
            "unique_bundles": len(self._estimates),
            "samples_per_bundle": self.samples_needed(),
            "epsilon": self.epsilon,
            "delta": self.delta,
        }
    
    def reset(self):
        """Clear all cached estimates."""
        self._estimates.clear()
        self._sample_counts.clear()
        self._total_samples = 0


class AdaptiveEstimator(ValueEstimator):
    """
    Adaptive estimator that adjusts sampling based on variance.
    """
    
    def __init__(self, valuation_profile: ValuationProfile,
                 epsilon: float = 0.1,
                 delta: float = 0.05,
                 min_samples: int = 10,
                 seed: Optional[int] = None):
        """
        Initialize adaptive estimator.
        
        Args:
            valuation_profile: True valuations
            epsilon: Target accuracy
            delta: Failure probability
            min_samples: Minimum samples before variance estimation
            seed: Random seed
        """
        super().__init__(valuation_profile, epsilon, delta, seed)
        self.min_samples = min_samples
        self._sample_variances: Dict[Tuple[int, frozenset], float] = {}
    
    def estimate_value(self, agent: int, bundle: Set[int],
                       force_resample: bool = False) -> float:
        """
        Adaptive estimation with early stopping if variance is low.
        """
        key = (agent, frozenset(bundle))
        
        if not force_resample and key in self._estimates:
            return self._estimates[key]
        
        valuation = self.profile[agent]
        max_samples = self.samples_needed()
        
        samples = []
        for i in range(max_samples):
            sample = valuation.bounded_sample(bundle)
            samples.append(sample)
            self._total_samples += 1
            
            if i >= self.min_samples - 1:
                current_mean = np.mean(samples)
                current_std = np.std(samples, ddof=1)
                
                confidence_width = 1.96 * current_std / np.sqrt(len(samples))
                if confidence_width < self.epsilon / 2:
                    break
        
        estimate = np.mean(samples)
        self._estimates[key] = estimate
        self._sample_counts[key] = len(samples)
        self._sample_variances[key] = np.var(samples) if len(samples) > 1 else 0
        
        return estimate
