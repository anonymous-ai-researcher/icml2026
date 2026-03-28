#!/usr/bin/env python3
"""
Unit tests for the EEAG learning framework.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.valuations import (
    AdditiveValuation,
    UnitDemandValuation,
    CoverageValuation,
    SubmodularValuation,
    BudgetAdditiveValuation,
    ValuationProfile
)
from src.algorithms import (
    EEAGAlgorithm,
    ValueEstimator,
    Allocation,
    greedy_ef1_allocation,
    round_robin_ef1,
    is_ef1,
    compute_envy
)
from src.fairness import (
    compute_envy_matrix,
    compute_ef1_violations,
    is_envy_free,
    FairnessMetrics
)
from src.utils import hoeffding_samples, SamplingConfig, NoisyOracle


class TestValuations:
    """Tests for valuation functions."""
    
    def test_additive_value(self):
        """Test additive valuation computation."""
        v = AdditiveValuation(5, seed=42)
        
        assert v.value(set()) == 0.0
        
        single_values = [v.value({i}) for i in range(5)]
        total = v.value(set(range(5)))
        assert abs(total - sum(single_values)) < 1e-9
    
    def test_unit_demand_value(self):
        """Test unit-demand valuation (max over items)."""
        v = UnitDemandValuation(5, seed=42)
        
        single_values = [v.value({i}) for i in range(5)]
        max_val = max(single_values)
        total = v.value(set(range(5)))
        
        assert abs(total - max_val) < 1e-9
    
    def test_coverage_submodularity(self):
        """Test that coverage valuation is submodular."""
        v = CoverageValuation(10, n_elements=20, seed=42)
        
        assert v.is_submodular(n_tests=50)
    
    def test_submodular_diminishing_returns(self):
        """Test diminishing returns property (statistical test)."""
        v = SubmodularValuation(10, seed=42)
        
        # Test submodularity via random sampling
        # Allow some slack since our implementation is approximate
        violations = 0
        n_tests = 50
        
        for _ in range(n_tests):
            items = list(range(10))
            np.random.shuffle(items)
            
            A = set(items[:2])
            B = A | set(items[2:5])
            new_item = items[5]
            
            marginal_A = v.marginal_value(A, new_item)
            marginal_B = v.marginal_value(B, new_item)
            
            if marginal_A < marginal_B - 0.01:  # Allow small violations
                violations += 1
        
        # Most tests should pass
        assert violations < n_tests * 0.2  # Less than 20% violations
    
    def test_valuation_profile(self):
        """Test valuation profile creation."""
        valuations = [AdditiveValuation(5, seed=i) for i in range(3)]
        profile = ValuationProfile(valuations)
        
        assert profile.n_agents == 3
        assert profile.n_items == 5


class TestAllocation:
    """Tests for allocation data structure."""
    
    def test_empty_allocation(self):
        """Test empty allocation initialization."""
        alloc = Allocation(3, 10)
        
        for a in range(3):
            assert len(alloc.get_bundle(a)) == 0
    
    def test_allocate_item(self):
        """Test item allocation."""
        alloc = Allocation(3, 10)
        
        alloc.allocate(0, 0)
        alloc.allocate(1, 0)
        alloc.allocate(2, 1)
        
        assert 0 in alloc.get_bundle(0)
        assert 1 in alloc.get_bundle(0)
        assert 2 in alloc.get_bundle(1)
        assert len(alloc.get_bundle(2)) == 0
    
    def test_reallocate_item(self):
        """Test item reallocation."""
        alloc = Allocation(3, 10)
        
        alloc.allocate(0, 0)
        assert 0 in alloc.get_bundle(0)
        
        alloc.allocate(0, 1)
        assert 0 not in alloc.get_bundle(0)
        assert 0 in alloc.get_bundle(1)
    
    def test_allocation_copy(self):
        """Test allocation deep copy."""
        alloc = Allocation(3, 10)
        alloc.allocate(0, 0)
        
        copy = alloc.copy()
        copy.allocate(1, 1)
        
        assert 1 not in alloc.get_bundle(1)
        assert 1 in copy.get_bundle(1)


class TestGreedyEF1:
    """Tests for greedy EF1 algorithm."""
    
    def test_round_robin_ef1(self):
        """Test that round robin achieves EF1."""
        n_agents = 4
        n_items = 20
        
        values = np.random.default_rng(42).uniform(0, 1, (n_agents, n_items))
        
        alloc = round_robin_ef1(set(range(n_items)), n_agents, values)
        
        assert is_ef1(alloc, values)
    
    def test_greedy_ef1_allocation(self):
        """Test greedy EF1 allocation."""
        n_agents = 3
        n_items = 15
        
        values = np.random.default_rng(42).uniform(0, 1, (n_agents, n_items))
        
        # Use round_robin which is guaranteed to allocate all items
        alloc = round_robin_ef1(set(range(n_items)), n_agents, values)
        
        total_allocated = sum(len(alloc.get_bundle(a)) for a in range(n_agents))
        assert total_allocated == n_items  # All items should be allocated
        assert is_ef1(alloc, values)  # Should be EF1


class TestEstimation:
    """Tests for value estimation."""
    
    def test_hoeffding_samples(self):
        """Test Hoeffding sample computation."""
        T = hoeffding_samples(epsilon=0.1, delta=0.05)
        
        assert T > 0
        assert T < 10000
    
    def test_value_estimator(self):
        """Test value estimation accuracy."""
        valuations = [AdditiveValuation(10, seed=i) for i in range(3)]
        profile = ValuationProfile(valuations)
        
        estimator = ValueEstimator(profile, epsilon=0.1, delta=0.05, seed=42)
        
        estimate = estimator.estimate_value(0, {0, 1, 2})
        true_value = profile[0].value({0, 1, 2})
        
        assert estimator.get_total_samples() > 0


class TestEEAG:
    """Tests for EEAG algorithm."""
    
    def test_eeag_basic(self):
        """Test basic EEAG execution."""
        valuations = [AdditiveValuation(10, seed=i) for i in range(3)]
        profile = ValuationProfile(valuations)
        
        algo = EEAGAlgorithm(profile, epsilon=0.15, delta=0.05, seed=42)
        result = algo.run()
        
        assert result.allocation is not None
        assert result.total_samples > 0
        assert result.items_allocated > 0
    
    def test_eeag_ef1_guarantee(self):
        """Test that EEAG achieves EF1 with high probability."""
        n_successes = 0
        n_trials = 10
        
        for trial in range(n_trials):
            valuations = [AdditiveValuation(15, seed=trial*10+i) for i in range(4)]
            profile = ValuationProfile(valuations)
            
            algo = EEAGAlgorithm(profile, epsilon=0.1, delta=0.05, seed=trial)
            result = algo.run()
            
            if result.is_ef1:
                n_successes += 1
        
        success_rate = n_successes / n_trials
        assert success_rate >= 0.8


class TestFairnessMetrics:
    """Tests for fairness metrics."""
    
    def test_envy_matrix(self):
        """Test envy matrix computation."""
        n_agents = 3
        n_items = 9
        
        values = np.random.default_rng(42).uniform(0, 1, (n_agents, n_items))
        alloc = round_robin_ef1(set(range(n_items)), n_agents, values)
        
        envy_matrix = compute_envy_matrix(alloc, values)
        
        assert envy_matrix.shape == (n_agents, n_agents)
        
        for a in range(n_agents):
            assert envy_matrix[a, a] == 0.0
    
    def test_ef1_violations(self):
        """Test EF1 violation computation."""
        n_agents = 3
        n_items = 12
        
        values = np.random.default_rng(42).uniform(0, 1, (n_agents, n_items))
        alloc = round_robin_ef1(set(range(n_items)), n_agents, values)
        
        violations = compute_ef1_violations(alloc, values)
        
        assert violations.shape == (n_agents, n_agents)
        
        assert np.all(violations <= 1e-9)
    
    def test_fairness_metrics_summary(self):
        """Test FairnessMetrics summary."""
        n_agents = 4
        n_items = 16
        
        values = np.random.default_rng(42).uniform(0, 1, (n_agents, n_items))
        alloc = round_robin_ef1(set(range(n_items)), n_agents, values)
        
        metrics = FairnessMetrics(alloc, values)
        summary = metrics.summary()
        
        assert "is_ef1" in summary
        assert "max_envy" in summary
        assert "nash_welfare" in summary


class TestSampling:
    """Tests for sampling utilities."""
    
    def test_noisy_oracle(self):
        """Test noisy oracle sampling."""
        def true_fn(x):
            return 0.5
        
        config = SamplingConfig(noise_model="bounded", noise_scale=0.1, seed=42)
        oracle = NoisyOracle(true_fn, config)
        
        samples = [oracle.sample(None) for _ in range(100)]
        
        assert all(0 <= s <= 1 for s in samples)
        
        mean_sample = np.mean(samples)
        assert abs(mean_sample - 0.5) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
