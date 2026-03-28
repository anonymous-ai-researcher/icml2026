"""
Sampling utilities for noisy value observations.
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for noisy sampling."""
    noise_model: str = "bounded"
    noise_scale: float = 0.1
    seed: Optional[int] = None


class NoisyOracle:
    """
    Noisy sampling oracle following Definition 2.12.
    
    A value observation for agent a on bundle S returns an independent
    random variable X with E[X] = v_a(S) and X ∈ [0,1].
    """
    
    def __init__(self, 
                 true_value_fn: Callable,
                 config: SamplingConfig = None):
        """
        Initialize noisy oracle.
        
        Args:
            true_value_fn: Function that returns true value
            config: Sampling configuration
        """
        self.true_value_fn = true_value_fn
        self.config = config or SamplingConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        self._sample_count = 0
    
    def sample(self, *args, **kwargs) -> float:
        """
        Get a noisy sample of the true value.
        
        Returns:
            Noisy observation in [0, 1]
        """
        true_value = self.true_value_fn(*args, **kwargs)
        true_value = np.clip(true_value, 0, 1)
        
        if self.config.noise_model == "bounded":
            noise = self.rng.uniform(-0.5, 0.5) * self.config.noise_scale
            sample = np.clip(true_value + noise, 0, 1)
        
        elif self.config.noise_model == "gaussian":
            noise = self.rng.normal(0, self.config.noise_scale)
            sample = np.clip(true_value + noise, 0, 1)
        
        elif self.config.noise_model == "bernoulli":
            sample = float(self.rng.random() < true_value)
        
        else:
            sample = true_value
        
        self._sample_count += 1
        return sample
    
    def get_sample_count(self) -> int:
        """Return total samples drawn."""
        return self._sample_count
    
    def reset_count(self):
        """Reset sample counter."""
        self._sample_count = 0


def hoeffding_samples(epsilon: float, delta: float, 
                      n_estimates: int = 1) -> int:
    """
    Compute samples needed for Hoeffding guarantee.
    
    For bounded random variables in [0,1]:
    P(|X̄ - μ| > ε) ≤ 2 exp(-2Tε²)
    
    For T samples to achieve ε-accuracy with probability 1-δ:
    T ≥ (1/2ε²) log(2/δ)
    
    Args:
        epsilon: Target accuracy
        delta: Failure probability
        n_estimates: Number of estimates (for union bound)
        
    Returns:
        Number of samples needed per estimate
    """
    adjusted_delta = delta / max(n_estimates, 1)
    T = (1 / (2 * epsilon**2)) * np.log(2 / adjusted_delta)
    return int(np.ceil(T))


def empirical_bernstein_samples(epsilon: float, 
                                delta: float,
                                variance_bound: float = 0.25) -> int:
    """
    Compute samples for empirical Bernstein bound.
    
    Tighter than Hoeffding when variance is small.
    
    Args:
        epsilon: Target accuracy
        delta: Failure probability
        variance_bound: Upper bound on variance
        
    Returns:
        Number of samples needed
    """
    T = (2 * variance_bound / epsilon**2) * np.log(3 / delta)
    T += (3 / epsilon) * np.log(3 / delta)
    return int(np.ceil(T))


class SampleComplexityTracker:
    """Track sample complexity across experiments."""
    
    def __init__(self):
        self.records = []
    
    def record(self, n_agents: int, n_items: int, epsilon: float,
               samples_used: int, success: bool):
        """Record an experiment result."""
        self.records.append({
            "n_agents": n_agents,
            "n_items": n_items,
            "epsilon": epsilon,
            "samples": samples_used,
            "success": success,
            "theoretical": hoeffding_samples(epsilon, 0.05, n_agents * n_items)
        })
    
    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.records:
            return {}
        
        samples = [r["samples"] for r in self.records]
        theoretical = [r["theoretical"] for r in self.records]
        success_rate = np.mean([r["success"] for r in self.records])
        
        return {
            "mean_samples": np.mean(samples),
            "std_samples": np.std(samples),
            "mean_theoretical": np.mean(theoretical),
            "ratio": np.mean(samples) / np.mean(theoretical) if theoretical else 0,
            "success_rate": success_rate,
            "n_experiments": len(self.records),
        }
